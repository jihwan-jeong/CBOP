import hydra.utils
import omegaconf
import torch
import torch.nn as nn
import numpy as np

from rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from rlkit.policies.base.base import MakeDeterministic
from rlkit.policies.gaussian_policy import TanhGaussianPolicy
from rlkit.torch.models.networks import TanhMlpPolicy
from rlkit.torch.distributions import Distribution
import rlkit.torch.pytorch_util as ptu
from typing import Tuple, Any


class SACAgent:
    """A wrapper class of SAC"""
    def __init__(
            self,
            policy: TanhGaussianPolicy,
            f_checkpoint: str = None,
            load_param: bool = False,
            **kwargs
    ):
        self.policy = policy
        self.eval_policy = MakeDeterministic(self.policy)

        if load_param:
            from pathlib import Path
            assert f_checkpoint is not None, "To load learned policy parameters, you need to provide the file path!"
            checkpoint = torch.load(f_checkpoint, map_location=ptu.device)
            self.policy.load_state_dict(checkpoint['trainer/policy_state_dict'])


class CQLAgent:
    """A wrapper class of CQL."""
    def __init__(
            self,
            policy: TanhGaussianPolicy,
            max_q_learning: bool = True,
            **kwargs,
    ):
        self.policy = policy
        eval_policy = MakeDeterministic(self.policy)

        if max_q_learning:
            assert 'qf' in kwargs
            qf = kwargs['qf']

            def eval_with_max_q_learning(obs):
                with torch.no_grad():
                    obs_ = ptu.from_numpy(obs.reshape(1, -1)).repeat(10, 1)
                    dist = self.policy(obs_)
                    action = dist.sample()
                    q_values = qf(obs_, action)
                    ind = q_values.max(0)[1]
                return ptu.get_numpy(action[ind]).flatten()
            self.eval_policy = eval_with_max_q_learning
        else:
            self.eval_policy = eval_policy


class TD3Agent:
    """A wrapper class of TD3"""
    def __init__(
            self,
            expl_policy: PolicyWrappedWithExplorationStrategy,
            eval_policy: TanhMlpPolicy,
            **kwargs,
    ):
        self.expl_policy = expl_policy
        self.eval_policy = eval_policy


class BehaviorCloningPolicy(nn.Module):
    """A wrapper class for behavior cloning policy.
    """
    def __init__(
            self,
            policy,
            cfg: omegaconf.DictConfig,
    ):
        super().__init__()
        self.policy = policy

        self.trained = False  # Whether the model has already been trained
        self._apply_output_transforms: bool = False

        # Load trained policy parameters if provided
        if cfg.cache_dir is not None:
            from rlkit.core.logging.logging import logger
            from pathlib import Path
            cwd = Path(hydra.utils.get_original_cwd())
            cache_dir = cwd / cfg.cache_dir
            if cfg.max_size is not None:
                cache_dir = cache_dir / f'{int(cfg.max_size)}'
            if not cache_dir.is_dir():
                logger.log(f'cache_dir is provided in config, but it is not a valid directory!\ncache_dir: {cache_dir}')
            elif (cache_dir / 'behavior_clone_checkpoint.pth').is_file():
                f_path = str(cache_dir / 'behavior_clone_checkpoint.pth')
                logger.log(f"Found 'behavior_clone_checkpoint.pth' from {cache_dir}. Loading parameters...")
                self.load(f_path)
                if self.trained:
                    logger.log("Policy parameters successfully loaded! (behavior cloning step will be skipped)")
                else:
                    logger.log("Failed to load policy parameters!")

    def forward(self, obs):
        """
        Behavior cloning policy maps a state to an action in a deterministic manner. If the linked policy
        is stochastic, we simply take the mean output of the predicted distribution.
        """
        if isinstance(self.policy, TanhGaussianPolicy):
            dist: Distribution = self.policy(obs)
            return dist.mle_estimate()
        else:
            raise NotImplementedError(
                "Forward pass for policies other than ``TanhGaussianPolicy`` has not been implemented.."
            )

    def get_loss(
            self, x, y, train=True, **kwargs,
    ) -> Tuple[torch.Tensor, Any]:
        """
        Computes the MSE loss

        Args:
            x (torch.Tensor): Input tensor corresponding to observations
            y (torch.Tensor): Target tensor corresponding to actions

        Returns:
            total_loss (torch.Tensor): The loss to which gradients of parameters are going to be computed
            mse_loss (torch.Tensor): Per model MSE loss returned for evaluation purpose
        """
        y_pred = self(x)
        mse_loss = torch.mean(torch.mean(torch.square(y_pred - y), dim=1), dim=0)
        return mse_loss, None

    def set_output_transforms(self, use_output_transform: bool):
        self._apply_output_transforms = use_output_transform

    def fit_output_stats(self, data, mask=None):
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        std[std < 1e-12] = 1.0
        if mask is not None:
            mean *= mask
            std *= mask

        self.policy.output_mu = nn.Parameter(
            ptu.from_numpy(mean), requires_grad=False).float()
        self.policy.output_std = nn.Parameter(
            ptu.from_numpy(std), requires_grad=False).float()
        return mean, std

    def fit_input_stats(self, data, mask=None):
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        std[std < 1e-12] = 1.0
        if mask is not None:
            mean *= mask
            std *= mask

        self.policy.input_mu.data = ptu.from_numpy(mean)
        self.policy.input_std.data = ptu.from_numpy(std)

    def eval_score(self, x_eval: torch.Tensor, y_eval: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Computes the MSE losses of a given batch for each and every model in the ensemble.
        """
        with torch.no_grad():
            mse_loss, _ = self.get_loss(x_eval, y_eval, **kwargs)
            return mse_loss

    @property
    def trained(self):
        return self._trained

    @trained.setter
    def trained(self, trained: bool):
        self._trained = trained

    def load(self, model_dir: str, key='behavior_clone_state_dict'):
        try:
            state_dict = torch.load(model_dir, map_location=ptu.device)
            try:
                self.load_state_dict(state_dict)
                self.trained = True
            except RuntimeError:
                self.load_state_dict(state_dict[key])
                self.trained = True

        except FileNotFoundError:
            print(f"{model_dir} not found!")
        except Exception as e:
            print(f"Failed loading saved parameters in {model_dir}")
            print(e)
