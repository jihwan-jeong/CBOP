import abc

from torch import nn

from rlkit.torch.distributions import (
    Bernoulli,
    Beta,
    Distribution,
    Independent,
    GaussianMixture as GaussianMixtureDistribution,
    GaussianMixtureFull as GaussianMixtureFullDistribution,
    MultivariateDiagonalNormal,
    TanhNormal,
)

class DistributionGenerator(nn.Module, metaclass=abc.ABCMeta):
    def forward(self, *input, **kwarg) -> Distribution:
        raise NotImplementedError

