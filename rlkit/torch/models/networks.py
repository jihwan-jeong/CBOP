"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import abc

import numpy as np
import torch
from torch import nn as nn

from rlkit.policies.base.base import Policy
from rlkit.torch.distributions import Distribution
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import eval_np
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.models.util import identity
from rlkit.torch.modules import LayerNorm


class Mlp(nn.Module):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=torch.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.,
            layer_norm=False,
            layer_norm_kwargs=None,
            final_init_scale=None,
            **kwargs,
    ):
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        if final_init_scale is None:
            self.last_fc.weight.data.uniform_(-init_w, init_w)
            self.last_fc.bias.data.uniform_(-init_w, init_w)
        else:
            ptu.ortho_init(self.last_fc.weight, final_init_scale)
            self.last_fc.bias.data.fill_(0)

        # data normalization
        self.input_mu = nn.Parameter(
            torch.zeros(1, input_size), requires_grad=False).float()
        self.input_std = nn.Parameter(
            torch.ones(1, input_size), requires_grad=False).float()
        self.output_mu = nn.Parameter(
            torch.zeros(1, ), requires_grad=False).float()
        self.output_std = nn.Parameter(
            torch.ones(1, ), requires_grad=False).float()

        self.mse_criterion = nn.MSELoss()

    def forward(self, input, return_preactivations=False, **kwargs):
        # Input normalization
        h = (input - self.input_mu) / self.input_std

        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output

    def get_loss(self, x, y, *args, **kwargs):
        y_pred = self(x)
        loss = self.mse_criterion(y_pred, y)
        return loss

    def save(self, path):
        """Saves the model to the given path"""
        torch.save(self.state_dict(), path)

    def load(self, state_dict, key=''):
        """Loads the model from the given path"""
        self.load_state_dict(state_dict[key])
    
    def fit_input_stats(self, data, mask=None):
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        std[std < 1e-12] = 1.0
        if mask is not None:
            mean *= mask
            std *= mask
        assert mean.shape == self.input_mu.shape and std.shape == self.input_std.shape
        self.input_mu.data = ptu.from_numpy(mean)
        self.input_std.data = ptu.from_numpy(std)

    def fit_output_stats(self, data, mask=None):
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        std[std < 1e-12] = 1.0
        if mask is not None:
            mean *= mask
            std *= mask

        self.output_mu = nn.Parameter(
            ptu.from_numpy(mean), requires_grad=False).float()
        self.output_std = nn.Parameter(
            ptu.from_numpy(std), requires_grad=False).float()
        return mean, std


class FlattenMlp(Mlp):
    """
    Flatten inputs along dimension 1 and then pass through MLP.
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=-1)
        return super().forward(flat_inputs, **kwargs)


class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            input_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.input_normalizer = input_normalizer

    def forward(self, x, **kwargs):
        """
        Args:
            x (torch.Tensor)  Could be an observation array or (observation, goal) array in GCRL
        Returns:
            (torch.Tensor) Action
        """
        if self.input_normalizer:
            x = self.input_normalizer.normalize(x)
        return super().forward(x, **kwargs)

    def get_action(self, x: np.ndarray, **kwargs):
        actions = self.get_actions(x[None])
        return actions[0, :], {}

    def get_actions(self, x, **kwargs):
        return eval_np(self, x, **kwargs)


class TanhMlpPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, output_activation=torch.tanh, **kwargs)


class DistributionGenerator(nn.Module, metaclass=abc.ABCMeta):
    def forward(self, *input, **kwarg) -> Distribution:
        raise NotImplementedError


class MlpAutoEncoder(nn.Module):
    def __init__(
            self,
            hidden_sizes,
            embed_size,
            input_size,
    ):
        super().__init__()
        original_dim = input_size
        self.encoder = Mlp(
            hidden_sizes=hidden_sizes,
            input_size=original_dim,
            output_size=embed_size,
        )
        self.decoder = Mlp(
            hidden_sizes=hidden_sizes,
            input_size=embed_size,
            output_size=original_dim,
        )
        self.trained = False

    def forward(self, inputs):
        latent_embedding = self.encoder(inputs)
        reconstruct = self.decoder(latent_embedding)
        return latent_embedding, reconstruct

    def eval_score(self, x, y):
        """Computes the prediction score (loss) given input and target"""
        _, y_pred = self(x)
        loss = (y - y_pred).pow(2).mean()
        return loss
