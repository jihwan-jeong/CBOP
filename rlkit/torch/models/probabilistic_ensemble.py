from typing import Tuple, Optional, Union

import torch
from torch import nn as nn
from torch.nn import functional as F

from rlkit.torch import pytorch_util as ptu
from rlkit.torch.models.ensemble import Ensemble
from rlkit.torch.models.util import swish, identity

class ProbabilisticEnsemble(Ensemble):

    """
    Probabilistic ensemble (Chua et al. 2018).
    Implementation is parallelized such that every model uses one forward call.
    This class is generic such that it can be used by for dynamics modeling or for value function, etc.
    Each model outputs the mean and variance given an input.
    Sampling is done either uniformly or via trajectory sampling.
    """

    def __init__(
            self,
            ensemble_size: int,                 # Number of members in ensemble
            hidden_sizes: list,                 # Hidden sizes for each model
            input_size: int,                    # Input dim to the model
            output_size: int,                   # Output dimension
            learn_logstd_min_max: bool = True,  # Whether to learn the min and max of logstd
            hidden_activation: callable = swish,
            output_activation: callable = identity,
            b_init_value: float = 0.0,
            propagation_method: str = 'fixed_model',
            use_decay: bool = False,
            **kwargs
    ):
        super().__init__(
            ensemble_size=ensemble_size,
            hidden_sizes=hidden_sizes,
            input_size=input_size,
            output_size=output_size,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            b_init_value=b_init_value,
            propagation_method=propagation_method,
            use_decay=use_decay,
            **kwargs
        )

        self.is_probabilistic = True

        # Note: we can learn max/min logstd here as in PETS; some implementations might not
        self.learn_logstd_min_max = learn_logstd_min_max
        self.max_logvar = nn.Parameter(
            ptu.ones(self.output_size//2)/2., requires_grad=learn_logstd_min_max)
        self.min_logvar = nn.Parameter(
            -ptu.ones(self.output_size//2) * 10., requires_grad=learn_logstd_min_max)

        # Output normalization
        self.output_mu = nn.Parameter(
            ptu.zeros(1, self.output_size//2), requires_grad=False).float()
        self.output_std = nn.Parameter(
            ptu.ones(1, self.output_size//2), requires_grad=False).float()

        # Register min_logvar and max_logvar as learnable Parameters if they need to be learned
        if learn_logstd_min_max:
            self.min_max_logvar = [self.min_logvar, self.max_logvar]

    def _default_forward(
            self, x: torch.Tensor, only_elite: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        output, _ = super()._default_forward(x, only_elite=only_elite)
        mean, logvar = torch.chunk(output, 2, dim=-1)

        # Variance clamping to prevent poor numerical predictions
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        return mean, logvar

    def get_loss(
            self, x, y, inc_var_loss=False, train=True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the negative log-likelihood loss, assuming the output distribution is diagonal Gaussian.
        The means and the variances are outputted by the ensemble network.

        Args:
            x (torch.Tensor):       Input tensor
            y (torch.Tensor):       Target tensor
            inc_var_loss (bool):    Whether to include the log-likelihood term due to the variance
                                    If False, only the mean squared error is returned.

        Returns:
            total_loss (torch.Tensor): The loss to which gradients of parameters are going to be computed
            mse_loss (torch.Tensor): Per model MSE loss returned for evaluation purpose
        """
        assert x.ndim == y.ndim
        if x.ndim == 2:             # This case can occur only when self.ensemble_size == 1
            assert len(self) == 1   # for testing..
            x.unsqueeze_(0)
            y.unsqueeze_(0)

        mean, logvar = self(x, use_propagation=False)           # Propagation only used for model rollouts

        # Compute the precision of Gaussian
        inv_var = torch.exp(-logvar)
        squared_error = torch.square(mean - y)
        if inc_var_loss:
            mse_loss = torch.mean(torch.mean(squared_error * inv_var, dim=-1), dim=-1)
            var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)
            total_loss = mse_loss.sum() + var_loss.sum()
        else:
            mse_loss = torch.mean(squared_error, dim=(1, 2))
            total_loss = mse_loss.sum()

        if self.use_decay and train:
            total_loss += self.get_decay_loss()
        return total_loss, mse_loss

    def eval_score(self, x_eval: torch.Tensor, y_eval: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Computes the MSE losses of a given batch for each and every model in the ensemble.
        """
        return super().eval_score(x_eval, y_eval, **kwargs)
