from collections import OrderedDict, namedtuple
import gtimer as gt
from typing import Tuple, Optional
import omegaconf

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.models.ensemble import FlattenEnsembleMLP
from rlkit.util.eval_util import create_stats_ordered_dict
from rlkit.core.rl_algorithms.torch_rl_algorithm import TorchTrainer
from rlkit.core.logging.logging import add_prefix
from rlkit.examples.algorithms.actor_critic.wrapper import SACAgent
from rlkit.trainers.sac.util import *


class SACTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            agent: SACAgent,
            qfs: FlattenEnsembleMLP,
            target_qfs: FlattenEnsembleMLP,
            num_qfs: int,

            discount: float = 0.99,
            reward_scale: float = 1.0,
            init_alpha: float = 1,

            policy_lr: float = 3e-4,
            qf_lr: float = 3e-4,
            optimizer_class=optim.Adam,

            soft_target_tau: float = 5e-3,
            target_update_period: int = 1,
            render_eval_paths: bool = False,

            use_automatic_entropy_tuning: bool = True,
            target_entropy=None,

            max_q_backup: bool = False,
            deterministic_backup: bool = False,
            eta: float = -1.0,

            max_grad_norm: Optional[omegaconf.dictconfig.DictConfig] = None,
            track_grad_norm: bool = False,
    ):
        super().__init__()
        self.env = env
        self.policy = agent.policy
        self.qfs = qfs
        self.target_qfs = target_qfs
        self.num_qfs = num_qfs

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
                lr=policy_lr,
            )
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss(reduction='none')        # `reduction` should be `none`
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qfs_optimizer = optimizer_class(
            self.qfs.parameters(),
            lr=qf_lr,
        )
        
        self.discount = discount
        self.reward_scale = reward_scale
        self.max_q_backup = max_q_backup
        self.deterministic_backup = deterministic_backup
        self.eta = eta

        self.eval_statistics = OrderedDict()

        self._max_grad_norm = max_grad_norm
        self._track_grad_norm = track_grad_norm
        self._n_train_steps_this_epoch = 0
        self._need_to_update_eval_statistics = True
        self.name = 'sac'

        network_dict = dict(
            qfs=self.qfs,
            target_qfs=self.target_qfs,
            policy=self.policy,
        )
        for key, module in network_dict.items():
            self.register_module(key, module)

    def train_from_torch(self, batch):
        """
        Follows the original rlkit implementation, while the Q ensemble part and the diversified
        gradients are adapted from EDAC (An et al., 2021). 
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
            alpha = 1

        # Conservative Q evaluation via Q ensemble (if num_qs=2, it's just SAC)
        q_new_actions = self.qfs.sample(obs, new_obs_actions, sampling_method='min')
        policy_loss = (alpha * log_pi - q_new_actions).mean()

        """
        QF Loss
        """
        qs_pred = self.qfs(obs, actions)        # (num_qs, batch_size, output_size)

        next_dist = self.policy(next_obs)
        new_next_actions, new_log_pi = next_dist.sample_and_logprob()
        new_log_pi = new_log_pi.unsqueeze(-1)

        # If max Q backup is used... take the maximizing action from sampled actions
        # Then, take the min among the target values from the Q ensemble
        if self.max_q_backup:
            next_actions_temp, _ = get_policy_actions(
                next_obs, num_actions=10, network=self.policy
            )
            target_q_values = get_tensor_values(
                next_obs, next_actions_temp, network=self.qfs
            )
            target_q_values = target_q_values.max(2)[0].min(0)[0]
        
        else:
            target_q_values = self.target_qfs.sample(next_obs, new_next_actions, sampling_method='min')
            if not self.deterministic_backup:
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
        if self.use_automatic_entropy_tuning and not self.deterministic_backup:
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
            eval_statistics = OrderedDict()
            eval_statistics['QFs Loss'] = np.mean(ptu.get_numpy(qfs_loss) / self.num_qfs)
            if self.eta > 0:
                eval_statistics['Q Grad Loss'] = np.mean(ptu.get_numpy(
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
            policy_statistics = add_prefix(dist.get_diagnostics(), "policy/")
            eval_statistics.update(policy_statistics)

            if self.use_automatic_entropy_tuning:
                eval_statistics['Alpha'] = alpha.item()
                eval_statistics['Alpha Loss'] = alpha_loss.item()
                
            self.eval_statistics = eval_statistics
            # Compute statistics using only one batch per epoch
            self._need_to_update_eval_statistics = False

        gt.stamp('sac training', unique=False)

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def start_epoch(self, epoch):
        self._n_train_steps_this_epoch = 0

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    def configure_logging(self, **kwargs):
        import wandb
        nets_updated = [self.policy, self.qfs]
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
