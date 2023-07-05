from collections import OrderedDict
import gtimer as gt
import numpy as np
import torch.optim as optim
import torch.nn as nn

from rlkit.core.rl_algorithms.torch_rl_algorithm import TorchTrainer
from rlkit.trainers.sac.util import *
from rlkit.torch.models.ensemble import FlattenEnsembleMLP
from rlkit.util.eval_util import create_stats_ordered_dict
from rlkit.torch.models.dynamics_models.model import DynamicsModel
from rlkit.envs.model_env import ModelEnv
import rlkit.torch.pytorch_util as ptu
from rlkit.core.logging.logging import logger
from rlkit.core.logging.logging import add_prefix


class MVEPOTrainer(TorchTrainer):

    """
    Model-based value expansion (MVE) with dynamics and Q ensemble models.
    Policy optimization where value targets are computed using synthetic model-based rollouts.
    """

    def __init__(
            self,
            env,
            policy,
            qfs: FlattenEnsembleMLP,
            target_qfs: FlattenEnsembleMLP,
            num_qfs: int,

            dynamics_model: DynamicsModel,
            model_env: ModelEnv,
            horizon: int,
            num_particles: int,
            lamb: float = None,
            weighting: str = 'adaptive',                    # 'fixed' or 'adaptive'
            lcb_coeff: float = 1.0,
            use_t_dist: bool = False,
            t_dist_dof: float = None,

            discount: float = 0.99,
            reward_scale: float = 1.0,
            init_alpha: float = 1,

            lr: float = 3e-4,
            optimizer_class=optim.Adam,

            soft_target_tau: float = 5e-3,
            target_update_period: int = 1,
            
            use_automatic_entropy_tuning: bool = True,
            target_entropy=None,
            eta: float = -1.0,

            model_train_period: int = 250,
            sampling_method: str = 'mean',
            indep_sampling: bool = True,
            on_policy: bool = True,
            **kwargs
    ):
        super().__init__()
        self.env = env

        self.qfs = qfs
        self.target_qfs = target_qfs
        self.policy = policy
        self.num_qfs = num_qfs

        self.discount = discount
        self.reward_scale = reward_scale
        self.eta = eta

        # Sync the target network parameters initially
        ptu.copy_model_params_from_to(qfs, target_qfs)
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        
        self.alpha = init_alpha
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=lr,
            )
        
        self.dynamics_model = dynamics_model
        self.model_env = model_env
        self.replay_buffer = None
        self.num_particles = num_particles
        self.horizon = horizon
        self.lamb = lamb
        if isinstance(lamb, str) and (lamb.lower() == "none" or lamb.lower() == "null"):
            self.lamb = None
        assert self.lamb is None or isinstance(self.lamb, float)
        self.lcb_coeff = lcb_coeff
        self.weighting = weighting
        self.use_t_dist = use_t_dist
        self.t_dist_dof = t_dist_dof

        # Initialize the weight tensor used for lambda target computation
        if self.weighting == 'fixed' and self.lamb is not None:
            if self.lamb == 1.0:
                self.lamb_vec = ptu.zeros(horizon + 1)
                self.lamb_vec[-1] = 1.0
            else:
                common_term = (1 - self.lamb) / (1 - self.lamb ** (horizon + 1))
                self.lamb_vec = ptu.tensor([
                    common_term * (self.lamb ** h) for h in range(horizon + 1)
                ])

        self.qf_criterion = nn.MSELoss(reduction='none')        # `reduction` should be `none`
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=lr,
        )
        self.qfs_optimizer = optimizer_class(
            self.qfs.parameters(),
            lr=lr,
        )
        
        self.sampling_method = sampling_method
        self.indep_sampling = indep_sampling
        self.on_policy = on_policy
        self.model_train_period = model_train_period
        
        self._need_to_update_eval_statistics = True
        self._n_train_steps_this_epoch = 0
        self.eval_statistics = OrderedDict()

        self.name = 'mvepo'
        self._network_dict = dict(
            qfs=self.qfs,
            target_qfs=self.target_qfs,
            policy=self.policy,
            dynamics=self.dynamics_model,
        )

    def set_replay_buffer(self, replay_buffer):
        self.replay_buffer = replay_buffer

    def train_from_torch(self, batch):
        """
        Follows the original rlkit implementation with some modifications / updates.
        The Q ensemble part and the diversified gradients are adapted from EDAC (An et al., 2021). 
        The computation of the target Q values is the unique contribution (an extension to MVE).
        """

        gt.blank_stamp()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        # EDAC ()
        if self.eta > 0:
            actions.requires_grad_(True)

        """
        Policy and Alpha Loss
        """
        dist = self.policy(obs)
        new_obs_actions, log_pi = dist.rsample_and_logprob()
        log_pi = log_pi.unsqueeze(-1)
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = self.alpha
        q_new_actions = self.qfs.sample(obs, new_obs_actions, sampling_method=self.sampling_method)
        policy_loss = (alpha * log_pi - q_new_actions).mean()

        """
        QF Loss
        """
        qs_pred = self.qfs(obs, actions)        # (num_qs, batch_size, output_size)

        next_dist = self.policy(next_obs)
        new_next_actions, new_log_pi = next_dist.sample_and_logprob()
        new_log_pi = new_log_pi.unsqueeze(-1)

        # Compute the target Q values from the next state
        target_q_values, mean_mve, var_mve = self.compute_target_qval(
            next_obs,
            new_next_actions, 
        )
        target_q_values -= alpha * new_log_pi

        future_values = (1. - terminals) * self.discount * target_q_values
        q_target = self.reward_scale * rewards + future_values
        q_target = q_target.detach().unsqueeze(0).expand(self.num_qfs, -1, -1)

        qfs_loss = self.qf_criterion(qs_pred, q_target)
        qfs_loss = qfs_loss.mean(dim=(1, 2)).sum()
        
        qfs_loss_total = qfs_loss

        # Add EDAC for diversified gradients
        if self.eta > 0:
            obs_tile = obs.unsqueeze(0).repeat(self.num_qfs, 1, 1)
            actions_tile = actions.unsqueeze(0).repeat(self.num_qfs, 1, 1).requires_grad_(True)
            
            qs_preds_tile = self.qfs(obs_tile, actions_tile)
            qs_pred_grads, = torch.autograd.grad(
                qs_preds_tile.sum(),
                actions_tile,
                retain_graph=True,
                create_graph=True
            )
            qs_pred_grads = qs_pred_grads / (torch.norm(qs_pred_grads, p=2, dim=2).unsqueeze(-1) + 1e-10)
            qs_pred_grads = qs_pred_grads.transpose(0, 1)

            qs_pred_grads = torch.einsum('bik,bjk->bij', qs_pred_grads, qs_pred_grads)
            masks = ptu.eye(self.num_qfs).unsqueeze(dim=0).repeat(qs_pred_grads.size(0), 1, 1)
            qs_pred_grads = (1 - masks) * qs_pred_grads

            grad_loss = torch.mean(torch.sum(qs_pred_grads, dim=(1, 2))) / (self.num_qfs - 1)
            qfs_loss_total += self.eta * grad_loss
        
        """Update networks"""
        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.qfs_optimizer.zero_grad()
        qfs_loss_total.backward()
        self.qfs_optimizer.step()
        
        """Soft update of target network parameters"""
        try_update_target_networks(
            self._n_train_steps_this_epoch,
            self.target_update_period,
            self.soft_target_tau,
            ((self.qfs, self.target_qfs), ),
        )
        self._n_train_steps_this_epoch += 1

        if self._need_to_update_eval_statistics:
            eval_statistics = self.eval_statistics
            eval_statistics['QFs Loss'] = np.mean(ptu.get_numpy(qfs_loss) / self.num_qfs)
            if self.eta > 0:
                self.eval_statistics['Q Grad Loss'] = np.mean(ptu.get_numpy(
                    grad_loss
                ))
            eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))

            eval_statistics.update(create_stats_ordered_dict(
                'Qs Predictions',
                ptu.get_numpy(qs_pred),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Qs Targets',
                ptu.get_numpy(q_target),
            ))

            eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))

            # # Compare the independent sampling approach to the original one
            # mean_indep, var_indep = self.get_target_from_mve_independent_sampling(
            #     next_obs,
            #     new_next_actions, 
            # )
            # # mean_mve = self.eval_statistics['MVE Returns']
            # # var_mve = self.eval_statistics['MVE Return Var']
            # KL_mve_indep = torch.log(torch.sqrt(var_indep)/torch.sqrt(var_mve)) + (var_mve + (mean_mve - mean_indep) ** 2) / (2 * var_indep) - 0.5
            # eval_statistics.update(create_stats_ordered_dict(
            #     'KL(mve|indep)',
            #     ptu.get_numpy(KL_mve_indep)
            # ))
            # eval_statistics.update(create_stats_ordered_dict(
            #     '|mean(mve)-mean(indep)|',
            #     ptu.get_numpy(torch.abs(mean_mve - mean_indep))
            # ))
            # KL_indep_mve = torch.log(torch.sqrt(var_mve)/torch.sqrt(var_indep)) + (var_indep + (mean_mve - mean_indep) ** 2) / (2 * var_mve) - 0.5
            # eval_statistics.update(create_stats_ordered_dict(
            #     'KL(indep|mve)',
            #     ptu.get_numpy(KL_indep_mve)
            # ))
            # entropy_indep = 0.5 * (1 + torch.log(2 * np.pi * var_indep))
            # eval_statistics.update(create_stats_ordered_dict(
            #     'H(indep)',
            #     ptu.get_numpy(entropy_indep)
            # ))
            # eval_statistics.update(create_stats_ordered_dict(
            #     'KL(indep|mve)/H(indep)',
            #     ptu.get_numpy(KL_indep_mve / entropy_indep)
            # ))

            policy_statistics = add_prefix(dist.get_diagnostics(), "policy/")
            eval_statistics.update(policy_statistics)

            if self.use_automatic_entropy_tuning:
                eval_statistics['Alpha'] = alpha.item()
                eval_statistics['Alpha Loss'] = alpha_loss.item()

            self.eval_statistics = eval_statistics
            # Compute statistics using only one batch per epoch
            self._need_to_update_eval_statistics = False

        gt.stamp('sac training', unique=False)

    def compute_target_qval(
            self,
            obs,
            actions,
    ):
        if self.indep_sampling:
            mve = self.get_target_from_mve_independent_sampling
        else:
            mve = self.get_target_from_mve
        mean, var = mve(
            obs=obs,
            actions=actions,
        )
        
        return mean - self.lcb_coeff * torch.sqrt(var), mean, var

    def get_target_from_mve(
            self,
            obs,
            actions,
    ):
        """Computes the target Q value using MVE and the target Q function at (obs, actions).

        There are three weighting schemes:
            (1) weighting=='fixed' and lamb is None: uses a fixed 1/H as the weight for each n-step return
            (2) weighting=='fixed' and lamb is not None: fixed lambda weighting (geometric series)
            (3) weighting=='adaptive': adaptive weights computed via Bayesian posterior over the true return

        Args:
            obs (torch.Tensor)
            actions (torch.Tensor)
        """
        env = self.model_env
        batch_size, num_particles, horizon = obs.size(0), self.num_particles, self.horizon

        # Set the observation tensor (repeat the observation tensor for `num_particles` number of times
        initial_obs_batch = obs.repeat(num_particles, 1)          # [KN, O]
        env.reset(initial_obs_batch, return_as_np=False)

        with torch.no_grad():
            # This tensor records which particles have terminated.
            terminated = ptu.zeros(num_particles * batch_size, 1, dtype=torch.bool)
            
            # This tensor records the timestep when trajectory got terminated.
            terminated_step = ptu.zeros(batch_size, horizon + 1, dtype=torch.bool)
            
            X_returns_h = ptu.zeros(num_particles, batch_size, horizon + 1)                     # Model-based rollout returns [N, K, H]
            Q_term_vals_h = ptu.zeros(self.num_qfs, num_particles, batch_size, horizon + 1)     # Bootstrapped Q values [M, N, K, H]

            aggr_obs_t = obs.clone()            # [N, O], the average trajectory, for getting actions only

            for t in range(horizon + 1):
                # Get the action to sample next state and reward (Note: make sure actions are within the support!)
                if self.on_policy:
                    # on-policy action sampling
                    if t == 0:
                        act_t = actions.clone()                     # Initial action is given
                        action_batch = act_t.repeat(num_particles, 1)   # [KN, A]
                    else:
                        dist = self.policy(env._obs)              # Assuming TanhNormal policy
                        action_batch = dist.sample()

                else:
                    # averaged trajectory action sampling
                    if t == 0:
                        act_t = actions.clone()                     # Initial action is given
                    else:
                        dist = self.policy(aggr_obs_t)              # Assuming TanhNormal policy
                        act_t = dist.sample()

                    # Repeat the action tensor for ``num_particles`` number of times
                    action_batch = act_t.repeat(num_particles, 1)   # [KN, A]

                # (2) update Q(s)
                term_val_at_t = self.target_qfs(                   # [M, KN, 1]
                    torch.cat([env._obs.clone(), action_batch], dim=-1).unsqueeze(0).expand(self.num_qfs, -1, -1),
                    use_propagation=False,
                    only_elite=True,
                )
                term_val_at_t[terminated.unsqueeze(0).expand(self.num_qfs, -1, -1)] = 0             # Terminated states have 0 terminal value
                term_val_at_t = term_val_at_t.view(self.num_qfs, num_particles, batch_size, 1)      # [M, K, N, 1]
                Q_term_vals_h[..., t: t + 1] = term_val_at_t.clone() * self.discount ** t

                if t < horizon:
                    _, rew_t, dones, _ = env.step(action_batch, sample=True, batch_size=num_particles * batch_size)    # Take a step in the model environment
                    rew_t[terminated] = 0               # Reward of already terminated states set to 0
                    terminated |= dones                 # Once terminated, should stay terminated
                    terminated_step[:, t+1:] = terminated.view(num_particles, batch_size, 1).any(dim=0)
                    
                    rew_t_per_rollout = rew_t.clone().view(num_particles, batch_size, -1)
                    obs_t_per_rollout = env._obs.clone().view(num_particles, batch_size, -1)
                    aggr_obs_t = torch.mean(obs_t_per_rollout, dim=0)           # Update the aggregated obs per rollout

                    #################################################################
                    # Update h-step returns
                    # R0 =                                                      Q(s0)
                    # R1 = r(s0) +                                          g x Q(s1)
                    # R2 = r(s0) + g x r(s1) +                            g^2 x Q(s2)
                    # ...
                    # RH = r(s0) + g x r(s1) + ... + g^(H-1) x r(s_H-1) + g^H x Q(sH)
                    #
                    # At each rollout step h, update r(s) for Rh ~ RH, and Q(s) for Rh
                    #################################################################
                    # (1) update r(s)
                    X_returns_h[..., t+1:] += rew_t_per_rollout * (self.discount ** t)
            
            #################################################################
            # Compute MVE (fixed, lambda, adaptive)
            # Target: get returns for each of the rollouts [N]
            # Methodology: assume each h-return in a rollout (composed by K particles)
            #              follows a Gaussian, then aggregate h-returns with SB/MB
            #################################################################
            # (1) permute R_h axis and reshape
            X_returns_h = X_returns_h.permute(0, 2, 1)                      # [K, N, H] -> [K, H, N]
            Q_term_vals_h = Q_term_vals_h.permute(0, 1, 3, 2)               # [M, K, N, H] -> [M, K, H, N]
            terminated_step = terminated_step.permute(1, 0)                 # [N, H] -> [H, N]
            
            # (2) Compute R_h = X_h + Q_h
            R_returns_h = X_returns_h.unsqueeze(0) + Q_term_vals_h          # [M, K, H, N]
            
            if self.weighting == 'adaptive':                
                # (3) Compute the mean and variance from the whole samples directly
                h_return_var, h_return_mean = torch.var_mean(R_returns_h, dim=(0, 1))
                h_return_var[h_return_var < 1e-5] = 1e-5        # Note: due to limited precision, var can be too small

                # (4) Bayesian posterior computation
                if self.use_t_dist:
                    # Student t posterior
                    # the scale parameter being... dof/((dof + H) * sum(preci)) + 1 / (dof + H) * weighted_sample_variance
                    h_return_preci = self.t_dist_dof / (self.t_dist_dof - 1) / h_return_var
                    h_return_preci[terminated_step] = 0
                    sum_of_preci = torch.sum(h_return_preci, dim=0, keepdim=True)
                    weights = h_return_preci / sum_of_preci

                    # Posterior mean (weighted sum of h-step means is the posterior mean)
                    mve_return_mean = torch.sum(h_return_mean * weights, dim=0)

                    # Posterior variance
                    deviation = h_return_mean - mve_return_mean[None, ...]
                    weighted_sample_var = torch.sum(weights * torch.pow(deviation, 2), dim=0)
                    mve_return_var_additional = 1 / (self.t_dist_dof + self.horizon + 1) * weighted_sample_var
                    mve_return_var_gaussian = self.t_dist_dof / (self.t_dist_dof + self.horizon + 1) / sum_of_preci[0]   # [N]
                    mve_return_var = mve_return_var_gaussian + mve_return_var_additional                                 # [N]
                else:
                    # Gaussian posterior
                    h_return_preci = 1 / h_return_var                                   # rho_n for all n [H, N]
                    h_return_preci[terminated_step] = 0
                    posterior_preci = torch.sum(h_return_preci, dim=0, keepdim=True)            # [1, N]
                    weights = h_return_preci / posterior_preci                                  # [H, N]
                    
                    # Weighted sum of h-step means is the posterior mean
                    mve_return_mean = torch.sum(h_return_mean * weights, dim=0)                 # [N]

                    # Posterior variance is the inverse of sum of precisions
                    mve_return_var = 1 / posterior_preci[0]                                     # [N]
            
            elif self.weighting == 'fixed':
                # Compute the weighted return of fixed weights 
                # weights:  [H, 1] 
                weights = self.lamb_vec[:, None] if self.lamb is not None else ptu.ones(horizon + 1, 1) / (horizon + 1)

                # (2) Get M x K samples of h-step returns
                R_returns_h = R_returns_h.reshape(self.num_qfs * num_particles, horizon + 1, batch_size)    # [MK, H, N]
                
                # (3) Get M x K samples of weighted returns
                mve_returns = torch.sum(R_returns_h * weights.unsqueeze(0), dim=1)               # [MK, N]

                # (4) Get the mean and variance of the weighted returns
                mve_return_var, mve_return_mean = torch.var_mean(mve_returns, dim=0)

            else:
                raise NotImplementedError           
            
            # (5) log the expected horizon for analysis
            horizon_vec = ptu.tensor(list(range(horizon+1))).float().unsqueeze(dim=1)
            expected_horizon = torch.sum(horizon_vec * weights, dim=0)
            gt.stamp('mve', unique=False)

        # Record the diagnostic information
        if self._need_to_update_eval_statistics:
            self.eval_statistics.update(create_stats_ordered_dict(
                'MVE Returns',
                ptu.get_numpy(mve_return_mean)
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'MVE expected horizon',
                ptu.get_numpy(expected_horizon)
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'MVE Return Var',
                ptu.get_numpy(mve_return_var)
            ))
            if self.use_t_dist:
                self.eval_statistics.update(create_stats_ordered_dict(
                    'MVE Return Var Gaussian',
                    ptu.get_numpy(mve_return_var_gaussian)
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'MVE Return Var Additional',
                    ptu.get_numpy(mve_return_var_additional)
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'MVE Return Var Ratio (Gaussian/Additional)',
                    ptu.get_numpy(mve_return_var_gaussian.mean()) / (ptu.get_numpy(mve_return_var_additional.mean()) + 1e-8)
                ))
            mve_target = mve_return_mean - self.lcb_coeff * torch.sqrt(mve_return_var)
            delta = (mve_return_mean - self.lcb_coeff * torch.sqrt(mve_return_var)) - \
                        torch.min(h_return_mean if self.weighting == 'adaptive' else mve_returns, dim=0)[0]
            self.eval_statistics.update(create_stats_ordered_dict(
                'MVE LCB Minus Min',
                ptu.get_numpy(delta)
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'MVE Target',
                ptu.get_numpy(mve_target)
            ))

        return mve_return_mean.unsqueeze(1), mve_return_var.unsqueeze(1)

    def get_target_from_mve_independent_sampling(
            self,
            obs,
            actions,
    ):
        """Computes the target Q value using MVE and the target Q function at (obs, actions).

        There are three weighting schemes:
            (1) weighting=='fixed' and lamb is None: uses a fixed 1/H as the weight for each n-step return
            (2) weighting=='fixed' and lamb is not None: fixed lambda weighting (geometric series)
            (3) weighting=='adaptive': adaptive weights computed via Bayesian posterior over the true return

        Args:
            obs (torch.Tensor)
            actions (torch.Tensor)
        """
        env = self.model_env
        batch_size, num_particles, horizon = obs.size(0), self.num_particles, self.horizon

        # Repeat the obs tensor by `num_particles * horizon`
        initial_obs_batch = obs.repeat(num_particles * (horizon + 1), 1)          # [KHN, O]
        env.reset(initial_obs_batch, return_as_np=False)

        with torch.no_grad():
            # This tensor records which particles have terminated.
            terminated = ptu.zeros(num_particles * (horizon + 1) * batch_size, 1, dtype=torch.bool)         # [KHN, 1]
            X_returns_h = ptu.zeros(num_particles, horizon + 1, batch_size)                     # Model-based rollout returns [K, H, N]
            Q_term_vals_h = ptu.zeros(self.num_qfs, num_particles, horizon + 1, batch_size)     # Bootstrapped Q values [M, K, H, N]
            aggr_obs_t = obs.clone().expand(horizon + 1, -1, -1)                    # [H, N, O]

            for t in range(horizon + 1):
                # Get the action to sample next state and reward (Note: make sure actions are within the support!)
                if self.on_policy:
                    # on-policy action sampling
                    if t == 0:
                        act_t = actions.clone()                     # Initial action is given
                        action_batch = act_t.repeat(num_particles * (horizon + 1), 1)   # [KHN, A]
                    else:
                        dist = self.policy(env._obs)              # Assuming TanhNormal policy
                        action_batch = dist.sample()

                else:
                    # averaged trajectory action sampling
                    if t == 0:
                        act_t = actions.clone().repeat(num_particles * (horizon + 1), 1) # [KHN, A]
                        action_batch = act_t
                    else:
                        dist = self.policy(aggr_obs_t)              # Assuming TanhNormal policy
                        act_t = dist.sample()                       # [H, N, A]
                        act_t = act_t[None].repeat(num_particles, 1, 1, 1) # [K, H, N, A]
                        action_batch = act_t.view(num_particles * (horizon + 1) * batch_size, -1)   # [KHN, A]
                    
                # (2) update Q(s)
                term_val_at_t = self.target_qfs(                   # [M, KHN, 1]
                    torch.cat([env._obs.clone(), action_batch], dim=-1).unsqueeze(0).expand(self.num_qfs, -1, -1),
                    use_propagation=False,
                    only_elite=True,
                )
                term_val_at_t[terminated.unsqueeze(0).expand(self.num_qfs, -1, -1)] = 0             # Terminated states have 0 terminal value
                term_val_at_t = term_val_at_t.view(self.num_qfs, num_particles, horizon + 1, batch_size)      # [M, K, H, N]
                Q_term_vals_h[..., t, :] = term_val_at_t.clone()[..., t, :] * self.discount ** t

                if t < horizon:
                    # Take a step in the model environment
                    _, rew_t, dones, _ = env.step(
                        action_batch, sample=True, # batch_size=num_particles * batch_size (using smaller batch size slows down)
                    )
                    rew_t[terminated] = 0               # Reward of already terminated states set to 0
                    terminated |= dones                 # Once terminated, should stay terminated
                    
                    rew_t_per_rollout = rew_t.clone().view(num_particles, horizon + 1, batch_size)      # [K, H, N]
                    obs_t_per_rollout = env._obs.clone().view(num_particles, horizon + 1, batch_size, -1)   # [K, H, N, O]
                    # Update the aggregated obs per rollout
                    aggr_obs_t = torch.mean(obs_t_per_rollout, dim=0)                                       # [H, N, O]

                    #################################################################
                    # Update h-step returns
                    # R0 =                                                      Q(s0)
                    # R1 = r(s0) +                                          g x Q(s1)
                    # R2 = r(s0) + g x r(s1) +                            g^2 x Q(s2)
                    # ...
                    # RH = r(s0) + g x r(s1) + ... + g^(H-1) x r(s_H-1) + g^H x Q(sH)
                    #
                    # At each rollout step h, update r(s) for Rh ~ RH, and Q(s) for Rh
                    #################################################################
                    # (1) update r(s)
                    X_returns_h[:, t+1, :] += rew_t_per_rollout[:, t+1, :] * (self.discount ** t)
            
            #################################################################
            # Compute MVE (fixed, lambda, adaptive)
            # Target: get returns for each of the rollouts [N]
            # Methodology: assume each h-return in a rollout (composed by K particles)
            #              follows a Gaussian, then aggregate h-returns with SB/MB
            #################################################################
            # (2) Compute R_h = X_h + Q_h
            R_returns_h = X_returns_h.unsqueeze(0) + Q_term_vals_h          # [M, K, H, N]
            
            if self.weighting == 'adaptive':                
                # (3) Compute the mean and variance from the whole samples directly
                h_return_var, h_return_mean = torch.var_mean(R_returns_h, dim=(0, 1))       # [H, N]
                h_return_var[h_return_var < 1e-5] = 1e-5        # Note: due to limited precision, var can be too small

                # (4) Bayesian posterior computation
                if self.use_t_dist:
                    # Student t posterior
                    # the scale parameter being... dof/((dof + H) * sum(preci)) + 1 / (dof + H) * weighted_sample_variance
                    h_return_preci = self.t_dist_dof / (self.t_dist_dof - 1) / h_return_var
                    sum_of_preci = torch.sum(h_return_preci, dim=0, keepdim=True)
                    weights = h_return_preci / sum_of_preci

                    # Posterior mean
                    mve_return_mean = torch.sum(h_return_mean * weights, dim=0)

                    # Posterior variance
                    deviation = h_return_mean - mve_return_mean[None, ...]                                      # [H, N]
                    weighted_sample_var = torch.sum(weights * torch.pow(deviation, 2), dim=0)                   # [N]
                    mve_return_var_additional = 1 / (self.t_dist_dof + self.horizon + 1) * weighted_sample_var
                    mve_return_var_gaussian = self.t_dist_dof / (self.t_dist_dof + self.horizon + 1) / sum_of_preci[0]   # [N]
                    mve_return_var = mve_return_var_gaussian + mve_return_var_additional                                 # [N]
                else:
                    # Gaussian posterior
                    h_return_preci = 1 / h_return_var                                   # rho_n for all n [H, N]
                    posterior_preci = torch.sum(h_return_preci, dim=0, keepdim=True)            # [1, N]
                    weights = h_return_preci / posterior_preci                                  # [H, N]

                    # Weighted sum of h-step means is the posterior mean
                    mve_return_mean = torch.sum(h_return_mean * weights, dim=0)                 # [N]

                    # Posterior variance is the inverse of sum of precisions
                    mve_return_var = 1 / posterior_preci[0]                                     # [N]
            
            elif self.weighting == 'fixed':
                # Compute the weighted return of fixed weights 
                # weights:  [H, 1] 
                weights = self.lamb_vec[:, None] if self.lamb is not None else ptu.ones(horizon + 1, 1) / (horizon + 1)
                
                # New code
                # (2) Get M x K samples of h-step returns
                R_returns_h = R_returns_h.view(self.num_qfs * num_particles, horizon + 1, batch_size)    # [MK, H, N]

                # (3) Get M x K samples of weighted returns
                mve_returns = torch.sum(R_returns_h * weights.unsqueeze(0), dim=1)              # [MK, N]

                # (4) Get the mean and variance of the weighted returns
                mve_return_var, mve_return_mean = torch.var_mean(mve_returns, dim=0)

            else:
                raise NotImplementedError
            
            # (5) log the expected horizon for analysis
            horizon_vec = ptu.tensor(list(range(horizon+1))).float().unsqueeze(dim=1)
            expected_horizon = torch.sum(horizon_vec * weights, dim=0)
            gt.stamp('mve', unique=False)

        # Record the diagnostic information
        if self._need_to_update_eval_statistics:
            self.eval_statistics.update(create_stats_ordered_dict(
                'MVE Returns',
                ptu.get_numpy(mve_return_mean)
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'MVE expected horizon',
                ptu.get_numpy(expected_horizon)
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'MVE Return Var',
                ptu.get_numpy(mve_return_var)
            ))
            if self.use_t_dist:
                self.eval_statistics.update(create_stats_ordered_dict(
                    'MVE Return Var Gaussian',
                    ptu.get_numpy(mve_return_var_gaussian)
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'MVE Return Var Additional',
                    ptu.get_numpy(mve_return_var_additional)
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'MVE Return Var Ratio (Gaussian/Additional)',
                    ptu.get_numpy(mve_return_var_gaussian) / (ptu.get_numpy(mve_return_var_additional) + 1e-8)
                ))
            mve_target = mve_return_mean - self.lcb_coeff * torch.sqrt(mve_return_var)
            delta = (mve_return_mean - self.lcb_coeff * torch.sqrt(mve_return_var)) - \
                        torch.min(h_return_mean if self.weighting == 'adaptive' else mve_returns, dim=0)[0]
            self.eval_statistics.update(create_stats_ordered_dict(
                'MVE LCB Minus Min',
                ptu.get_numpy(delta)
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'MVE Target',
                ptu.get_numpy(mve_target)
            ))

        return mve_return_mean.unsqueeze(1), mve_return_var.unsqueeze(1)

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return self.eval_statistics

    def start_epoch(self, epoch):
        super().start_epoch(epoch)
        self._n_train_steps_this_epoch = 0

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    def configure_logging(self, **kwargs):
        import wandb
        nets_updated = [self.policy, self.qfs]
        for i, net in enumerate(nets_updated):
            wandb.watch(net, idx=i, **kwargs)

    def get_snapshot(self):
        snapshot = dict(
            qfs=self.qfs.state_dict(),
            policy=self.policy.state_dict(),
        )
        return snapshot

    def load(self, state_dict, prefix=''):
        for name, network in self.network_dict.items():
            name = f"{prefix}/{name}" if prefix != '' else name
            if name in state_dict:
                try:
                    network.load_state_dict(state_dict[name])
                except RuntimeError:
                    print(f"Failed to load state_dict[{name}]")
