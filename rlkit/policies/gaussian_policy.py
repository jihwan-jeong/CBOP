"""
This file has been moved from rlkit/torch/sac/policies to here.
TanhGaussian policy is used by SAC, but can be utilized by other algorithms as well.
"""

import numpy as np
import torch
from rlkit.util.common import LOG_STD_MAX, LOG_STD_MIN
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.distributions import TanhNormal
from rlkit.torch.models.networks import Mlp
from rlkit.policies.base.base import (
    TorchStochasticPolicy,
)


class TanhGaussianPolicy(Mlp, TorchStochasticPolicy):
    """
    Usage:
    ```
    policy = TanhGaussianPolicy(...)
    """

    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            log_std_bounds=None,
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.log_std = None
        self.std = std
        if log_std_bounds is None:
            self.log_std_bounds = [LOG_STD_MIN, LOG_STD_MAX]
        else:
            self.log_std_bounds = log_std_bounds

        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            log_std_min, log_std_max = self.log_std_bounds
            assert log_std_min <= self.log_std <= log_std_max

    def forward(self, obs, **kwargs) -> TanhNormal:
        # Input normalization
        h = (obs - self.input_mu) / self.input_std

        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)

        if self.std is None:
            log_std_min, log_std_max = self.log_std_bounds
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, log_std_min, log_std_max)
            std = torch.exp(log_std)
        else:
            std = torch.from_numpy(np.array([self.std, ])).float().to(
                ptu.device)

        return TanhNormal(mean, std)

    def logprob(self, action, mean, std):
        tanh_normal = TanhNormal(mean, std)
        log_prob = tanh_normal.log_prob(
            action,
        )
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return log_prob

    def eval_score(self, x, y, **kwargs):
        dist = self(x)
        return - dist.log_prob(y)

    def get_loss(self, x, y, *args, **kwargs):
        dist = self(x)
        return - dist.log_prob(y).mean(), None
