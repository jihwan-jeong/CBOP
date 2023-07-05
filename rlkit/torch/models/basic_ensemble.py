import torch
import torch.nn as nn

import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.models.util import swish
from rlkit.torch.models.ensemble import Ensemble
from pathlib import Path
from typing import Union, Optional, Tuple

class DeterministicEnsemble(Ensemble):

    def __init__(
            self,
            ensemble_size,          # Number of members in ensemble
            input_dim,              # Input dim to the model (same as obs_dim unless processed)
            obs_dim,                # Observation dim of the environment
            action_dim,             # Action dim of environment
            hidden_sizes,           # Hidden sizes for each model
            learn_reward,           # Whether reward function is learned or not
            spectral_norm=False,    # Apply spectral norm to every hidden layer
            **kwargs
    ):
        super().__init__(
            ensemble_size=ensemble_size,
            hidden_sizes=hidden_sizes,
            input_size=input_dim,
            output_size=obs_dim + 1 if learn_reward else obs_dim,  # We predict next_state and reward (optional)
            spectral_norm=spectral_norm,
            **kwargs
        )
        self.is_probabilistic = False
        self.learn_reward = learn_reward

        self.obs_dim, self.input_dim, self.action_dim = obs_dim, input_dim, action_dim

    def sample(
            self,
            x,
            deterministic: bool = False,
            rng: Optional[torch.Generator] = None,
            **kwargs,
    ):
        preds = super().sample(x, rng=rng, deterministic=deterministic)[0]
        obs_preds = preds[:, :-1] if self.learn_reward else preds
        rewards = preds[:, -1:] if self.learn_reward else None
        return obs_preds, rewards
