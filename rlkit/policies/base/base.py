import abc

from torch import nn as nn
from rlkit.torch.core import torch_ify, elem_or_tuple_to_numpy
from rlkit.torch.distributions import Delta
from rlkit.torch.models.stochastic.distribution_generator import (
    DistributionGenerator
)
from typing import Tuple, Dict
import numpy as np


class Policy(object, metaclass=abc.ABCMeta):
    """
    General policy interface.
    """
    @abc.abstractmethod
    def get_action(self, observation) -> Tuple[np.ndarray, Dict]:
        """
        :param observation:
        :return: action, debug_dictionary
        """
        pass

    @abc.abstractmethod
    def get_actions(self, obs, **kwargs):
        raise NotImplementedError

    def plan(self, observation):
        pass

    def reset(self, *args, **kwargs):
        pass


class ExplorationPolicy(Policy, metaclass=abc.ABCMeta):
    def set_num_steps_total(self, t):
        pass


class TorchStochasticPolicy(
    DistributionGenerator,
    ExplorationPolicy, metaclass=abc.ABCMeta
):
    def get_action(self, obs_np, ):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs_np, **kwargs):
        dist = self._get_dist_from_np(obs_np)
        actions = dist.sample()
        return elem_or_tuple_to_numpy(actions)

    def _get_dist_from_np(self, *args, **kwargs):
        torch_args = tuple(torch_ify(x) for x in args)
        torch_kwargs = {k: torch_ify(v) for k, v in kwargs.items()}
        dist = self(*torch_args, **torch_kwargs)
        return dist


class MakeDeterministic(TorchStochasticPolicy):
    def __init__(
            self,
            action_distribution_generator: DistributionGenerator,
    ):
        super().__init__()
        self._action_distribution_generator = action_distribution_generator

    def forward(self, *args, **kwargs):
        dist = self._action_distribution_generator.forward(*args, **kwargs)
        return Delta(dist.mle_estimate())
