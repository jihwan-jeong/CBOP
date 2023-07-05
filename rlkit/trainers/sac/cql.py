from collections import OrderedDict
from typing import Tuple, Optional
import omegaconf

import gtimer as gt
import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.util.eval_util import create_stats_ordered_dict
from rlkit.core.rl_algorithms.torch_rl_algorithm import TorchTrainer
from rlkit.examples.algorithms.actor_critic.wrapper import CQLAgent
from rlkit.torch.distributions import TanhNormal
from rlkit.policies.gaussian_policy import TanhGaussianPolicy
from rlkit.trainers.sac.util import *


class CQLTrainer(TorchTrainer):
    def __init__(
            self, 
            env,
            agent: CQLAgent,
            qf1,
            qf2,
            target_qf1,
            target_qf2,

            discount=0.99,
            reward_scale=1.0,
            init_alpha=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            max_grad_norm: Optional[omegaconf.dictconfig.DictConfig] = None,
            track_grad_norm: bool = False,
            policy_eval_start=0,

            # CQL
            min_q_version=3,
            temp=1.0,
            min_q_weight=1.0,

            ## sort of backup
            max_q_backup=False,
            deterministic_backup=True,
            num_random=10,
            with_lagrange=False,
            lagrange_thresh=0.0,
    ):
        super().__init__()
        self.env = env
        self.policy = agent.policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2

        # Sync the target network parameters initially
        ptu.copy_model_params_from_to(self.target_qf1, qf1)
        ptu.copy_model_params_from_to(self.target_qf2, qf2)

        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.alpha = init_alpha
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item() 
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )
        
        self.with_lagrange = with_lagrange
        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh
            self.log_alpha_prime = ptu.zeros(1, requires_grad=True)
            self.alpha_prime_optimizer = optimizer_class(
                [self.log_alpha_prime],
                lr=qf_lr,
            )

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._max_grad_norm = max_grad_norm
        self._track_grad_norm = track_grad_norm
        self._n_train_steps_this_epoch = 0
        self._need_to_update_eval_statistics = True

        self.policy_eval_start = policy_eval_start
        
        self._curr_epoch = 0
        self._policy_update_ctr = 0
        self._num_policy_updates_made = 0

        ## min Q
        self.temp = temp
        self.min_q_version = min_q_version
        self.min_q_weight = min_q_weight

        self.softmax = torch.nn.Softmax(dim=1)
        self.softplus = torch.nn.Softplus(beta=self.temp, threshold=20)

        self.max_q_backup = max_q_backup
        self.deterministic_backup = deterministic_backup
        self.num_random = num_random

        # For implementation on the 
        self.discrete = False

        self.name = 'cql'

        network_dict = dict(
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
            policy=self.policy,
        )
        for key, module in network_dict.items():
            self.register_module(key, module)

    def train_from_torch(self, batch):
        """
        Modified to follow SAC PyTorch (https://github.com/Xingyu-Lin/mbpo_pytorch/blob/main/sac/sac.py) implementation.
        """
        gt.blank_stamp()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        QF Loss
        """
        with torch.no_grad():
            next_dist: TanhNormal = self.policy(next_obs)
            new_next_actions, new_log_pi = next_dist.sample_and_logprob()       # No gradient is computed
            new_log_pi = new_log_pi.unsqueeze(-1)
            if not self.max_q_backup:
                target_q_values = torch.min(
                    self.target_qf1(next_obs, new_next_actions),
                    self.target_qf2(next_obs, new_next_actions),
                )
                if not self.deterministic_backup:
                    target_q_values = target_q_values - self.alpha * new_log_pi
            else:
                """When using max q backup"""
                next_actions_temp, _ = get_policy_actions(next_obs, num_actions=10, network=self.policy, detach=True)
                target_qf1_values = get_tensor_values(
                    next_obs, next_actions_temp, network=self.target_qf1).squeeze(0).max(1)[0].view(-1, 1)
                target_qf2_values = get_tensor_values(
                    next_obs, next_actions_temp, network=self.target_qf2).squeeze(0).max(1)[0].view(-1, 1)
                target_q_values = torch.min(target_qf1_values, target_qf2_values)
            q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        qf1_loss = self.qf_criterion(q1_pred, q_target)
        qf2_loss = self.qf_criterion(q2_pred, q_target)

        """Add CQL"""
        with torch.no_grad():
            random_actions_tensor = ptu.empty(len(obs) * self.num_random, actions.shape[-1]).uniform_(-1, 1)
            curr_actions_tensor, curr_log_pis = get_policy_actions(
                obs,
                num_actions=self.num_random,
                network=self.policy,
                detach=True,
            )
            new_actions_tensor, new_log_pis = get_policy_actions(
                next_obs,
                num_actions=self.num_random,
                network=self.policy,
                detach=True,
            )
        q1_rand = get_tensor_values(obs, random_actions_tensor, network=self.qf1).squeeze(0)
        q2_rand = get_tensor_values(obs, random_actions_tensor, network=self.qf2).squeeze(0)
        q1_curr_actions = get_tensor_values(obs, curr_actions_tensor, network=self.qf1).squeeze(0)
        q2_curr_actions = get_tensor_values(obs, curr_actions_tensor, network=self.qf2).squeeze(0)
        q1_next_actions = get_tensor_values(obs, new_actions_tensor, network=self.qf1).squeeze(0)
        q2_next_actions = get_tensor_values(obs, new_actions_tensor, network=self.qf2).squeeze(0)

        cat_q1 = torch.cat(
            [q1_rand, q1_pred.unsqueeze(1), q1_next_actions, q1_curr_actions], dim=1
        )
        cat_q2 = torch.cat(
            [q2_rand, q2_pred.unsqueeze(1), q2_next_actions, q2_curr_actions], dim=1
        )
        std_q1 = torch.std(cat_q1, dim=1)
        std_q2 = torch.std(cat_q2, dim=1)

        if self.min_q_version == 3:
            # importance sammpled version
            random_density = np.log(0.5 ** curr_actions_tensor.shape[-1])
            cat_q1 = torch.cat(
                [q1_rand - random_density, q1_next_actions - new_log_pis,
                 q1_curr_actions - curr_log_pis], 1
            )
            cat_q2 = torch.cat(
                [q2_rand - random_density, q2_next_actions - new_log_pis,
                 q2_curr_actions - curr_log_pis], 1
            )

        min_qf1_loss = torch.logsumexp(cat_q1 / self.temp, dim=1, ).mean() * self.min_q_weight * self.temp
        min_qf2_loss = torch.logsumexp(cat_q2 / self.temp, dim=1, ).mean() * self.min_q_weight * self.temp

        """Subtract the log likelihood of data"""
        min_qf1_loss = min_qf1_loss - q1_pred.mean() * self.min_q_weight
        min_qf2_loss = min_qf2_loss - q2_pred.mean() * self.min_q_weight

        qf1_loss = qf1_loss + min_qf1_loss
        qf2_loss = qf2_loss + min_qf2_loss

        if self.with_lagrange:
            alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1000000.0)
            min_qf1_loss = alpha_prime * (min_qf1_loss - self.target_action_gap)
            min_qf2_loss = alpha_prime * (min_qf2_loss - self.target_action_gap)

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-min_qf1_loss - min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()

        """
        Policy and Alpha Loss
        # """
        dist: TanhNormal = self.policy(obs)
        policy_mean = dist.mean
        policy_log_std = dist.logstd
        new_obs_actions, log_pi = dist.rsample_and_logprob()
        log_pi = log_pi.unsqueeze(-1)

        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions),
            self.qf2(obs, new_obs_actions),
        )
        policy_loss = (self.alpha * log_pi - q_new_actions).mean()

        """
        Update networks
        """
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update the Q-functions
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        (qf1_loss + qf2_loss).backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self._num_policy_updates_made += 1

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1.0

        """
        Soft Updates
        """
        try_update_target_networks(
            self._n_train_steps_this_epoch,
            self.target_update_period,
            self.soft_target_tau,
            ((self.qf1, self.target_qf1), (self.qf2, self.target_qf2)),
        )
        self._n_train_steps_this_epoch += 1

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['min QF1 Loss'] = np.mean(ptu.get_numpy(min_qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['min QF2 Loss'] = np.mean(ptu.get_numpy(min_qf2_loss))

            if not self.discrete:
                self.eval_statistics['Std QF1 values'] = np.mean(ptu.get_numpy(std_q1))
                self.eval_statistics['Std QF2 values'] = np.mean(ptu.get_numpy(std_q2))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'QF1 in-distribution values',
                    ptu.get_numpy(q1_curr_actions),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'QF2 in-distribution values',
                    ptu.get_numpy(q2_curr_actions),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'QF1 random values',
                    ptu.get_numpy(q1_rand),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'QF2 random values',
                    ptu.get_numpy(q2_rand),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'QF1 next_actions values',
                    ptu.get_numpy(q1_next_actions),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'QF2 next_actions values',
                    ptu.get_numpy(q2_next_actions),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'actions',
                    ptu.get_numpy(actions)
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'rewards',
                    ptu.get_numpy(rewards)
                ))

            self.eval_statistics['Num Policy Updates'] = self._num_policy_updates_made
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            if not self.discrete:
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Policy mu',
                    ptu.get_numpy(policy_mean),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Policy log std',
                    ptu.get_numpy(policy_log_std),
                ))

            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()

            if self.with_lagrange:
                self.eval_statistics['Alpha_prime'] = alpha_prime.item()
                self.eval_statistics['min_q1_loss'] = ptu.get_numpy(min_qf1_loss).mean()
                self.eval_statistics['min_q2_loss'] = ptu.get_numpy(min_qf2_loss).mean()
                self.eval_statistics['threshold action gap'] = self.target_action_gap
                self.eval_statistics['alpha prime loss'] = alpha_prime_loss.item()

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._curr_epoch = epoch
        self._need_to_update_eval_statistics = True

    def configure_logging(self, **kwargs):
        import wandb
        nets_updated = [self.policy, self.qf1, self.qf2]
        for i, net in enumerate(nets_updated):
            wandb.watch(net, idx=i, **kwargs)

    def load(self, state_dict, prefix=''):
        for name, network in self.network_dict.items():
            name = f"{prefix}/{name}" if prefix != '' else name
            if name in state_dict:
                try:
                    network.load_state_dict(state_dict[name])
                except RuntimeError:
                    print(f"Failed to load state_dict[{name}]")
