from collections import OrderedDict
from typing import Optional
import omegaconf

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.util.eval_util import create_stats_ordered_dict
from rlkit.core.rl_algorithms.torch_rl_algorithm import TorchTrainer
from rlkit.examples.algorithms.actor_critic.wrapper import TD3Agent


class TD3Trainer(TorchTrainer):
    """
    Twin Delayed Deep Deterministic policy gradients
    """

    def __init__(
            self,
            agent: TD3Agent,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            target_policy,
            target_policy_noise=0.2,
            target_policy_noise_clip=0.5,

            discount=0.99,
            reward_scale=1.0,

            policy_learning_rate=3e-4,
            qf_learning_rate=3e-4,
            policy_and_target_update_period=2,
            tau=0.005,
            qf_criterion=None,
            optimizer_class=optim.Adam,
            max_grad_norm: Optional[omegaconf.dictconfig.DictConfig] =None,
            track_grad_norm: bool = False,
    ):
        super().__init__()
        if qf_criterion is None:
            qf_criterion = nn.MSELoss()
        self.qf1 = qf1
        self.qf2 = qf2
        self.policy = agent.eval_policy
        self.target_policy = target_policy
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.target_policy_noise = target_policy_noise
        self.target_policy_noise_clip = target_policy_noise_clip

        self.discount = discount
        self.reward_scale = reward_scale

        self.policy_and_target_update_period = policy_and_target_update_period
        self.tau = tau
        self.qf_criterion = qf_criterion

        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_learning_rate,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_learning_rate,
        )
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_learning_rate,
        )

        self.eval_statistics = OrderedDict()
        self._max_grad_norm = max_grad_norm
        self._track_grad_norm = track_grad_norm
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.name = 'td3'

        network_dict = dict(
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
            policy=self.policy,
            target_policy=self.target_policy,
        )
        for key, module in network_dict.items():
            self.register_module(key, module)

    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Critic operations.
        """

        next_actions = self.target_policy(next_obs)
        noise = ptu.randn(next_actions.shape) * self.target_policy_noise
        noise = torch.clamp(
            noise,
            -self.target_policy_noise_clip,
            self.target_policy_noise_clip
        )
        noisy_next_actions = next_actions + noise

        target_q1_values = self.target_qf1(next_obs, noisy_next_actions)
        target_q2_values = self.target_qf2(next_obs, noisy_next_actions)
        target_q_values = torch.min(target_q1_values, target_q2_values)
        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        q_target = q_target.detach()

        q1_pred = self.qf1(obs, actions)
        bellman_errors_1 = (q1_pred - q_target) ** 2
        qf1_loss = bellman_errors_1.mean()

        q2_pred = self.qf2(obs, actions)
        bellman_errors_2 = (q2_pred - q_target) ** 2
        qf2_loss = bellman_errors_2.mean()

        """
        Update Networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        if self._need_to_update_eval_statistics and self._track_grad_norm:
            self.log_grad_norm(self.qf1, self.eval_statistics, 'qf1')
        if self._max_grad_norm and self._max_grad_norm.get('qf1', False):
            nn.utils.clip_grad_norm_(self.qf1.parameters(), self._max_grad_norm.qf1)
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        if self._need_to_update_eval_statistics and self._track_grad_norm:
            self.log_grad_norm(self.qf2, self.eval_statistics, 'qf2')
        if self._max_grad_norm and self._max_grad_norm.get('qf2', False):
            nn.utils.clip_grad_norm_(self.qf2.parameters(), self._max_grad_norm.qf2)
        self.qf2_optimizer.step()

        policy_actions = policy_loss = None
        if self._n_train_steps_total % self.policy_and_target_update_period == 0:
            policy_actions = self.policy(obs)
            q_output = self.qf1(obs, policy_actions)
            policy_loss = - q_output.mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            if self._need_to_update_eval_statistics and self._track_grad_norm:
                self.log_grad_norm(self.policy, self.eval_statistics, 'policy')
            if self._max_grad_norm and self._max_grad_norm.get('policy', False):
                nn.utils.clip_grad_norm_(self.policy.parameters(), self._max_grad_norm.policy)
            self.policy_optimizer.step()

            ptu.soft_update_from_to(self.policy, self.target_policy, self.tau)
            ptu.soft_update_from_to(self.qf1, self.target_qf1, self.tau)
            ptu.soft_update_from_to(self.qf2, self.target_qf2, self.tau)

        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            if policy_loss is None:
                policy_actions = self.policy(obs)
                q_output = self.qf1(obs, policy_actions)
                policy_loss = - q_output.mean()

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
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
                'Bellman Errors 1',
                ptu.get_numpy(bellman_errors_1),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Bellman Errors 2',
                ptu.get_numpy(bellman_errors_2),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy Action',
                ptu.get_numpy(policy_actions),
            ))
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
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
