from typing import Optional, Tuple, Union, List, cast, Sequence

import numpy as np
import torch
from torch import nn as nn

import rlkit.types
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.models.util import swish, identity
from rlkit.util.math import truncated_normal_
from rlkit.core.logging.logging import logger
from collections import OrderedDict


def truncated_normal_init(m: nn.Module, b_init_value=0.0):
    if isinstance(m, nn.Linear):
        input_dim = m.in_features
        stddev = 1 / (2 * np.sqrt(input_dim))
        truncated_normal_(m.weight, std=stddev)
        m.bias.data.fill_(b_init_value)
    if isinstance(m, EnsembleFCLayer):
        input_dim = m.input_size
        # num_members, input_dim, _ = m.weight.data.shape
        stddev = 1 / (2 * np.sqrt(input_dim))
        # for i in range(num_members):
        truncated_normal_(m.weight, std=stddev)
        m.bias.data.fill_(b_init_value)


class EnsembleFCLayer(nn.Module):

    def __init__(
        self,
        ensemble_size,
        input_size,
        output_size,
        init_w=3e-3,
        init_func=ptu.fanin_init,
        b_init_value=0.0,
        weight_decay: float = 0.,
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.input_size = input_size
        self.output_size = output_size

        # Random initialization
        w_init = torch.randn((ensemble_size, input_size, output_size))
        init_func(w_init)
        self.weight_decay = weight_decay
        self.weight = nn.Parameter(
            w_init, requires_grad=True
        )

        # constant initialization
        b_init = torch.zeros((ensemble_size, 1, output_size)).float()
        b_init += b_init_value
        self.bias = nn.Parameter(b_init, requires_grad=True)

        # Elite models
        self.elite_models: List[int] = None
        self.use_only_elite = False

    def forward(self, x):
        if self.use_only_elite:
            xw = x.matmul(self.weight[self.elite_models, ...])
            return xw + self.bias[self.elite_models, ...]
        else:
            xw = x.matmul(self.weight)
            return xw + self.bias

    def set_elite(self, elite_models: Sequence[int]):
        self.elite_models = list(elite_models)

    def toggle_use_only_elite(self):
        self.use_only_elite = not self.use_only_elite

class Ensemble(nn.Module):

    def __init__(
            self,
            ensemble_size,                      # The number of models in the ensemble
            hidden_sizes,
            input_size,                         # The input dimension to the ensemble network
            output_size,                        # The output dimension of the ensemble network
            hidden_activation=swish,
            output_activation=identity,
            w_init_method=truncated_normal_init,
            b_init_value=0.0,
            propagation_method='fixed_model',   # Propagation method: 'fixed_model', 'random_model', 'expectation'
            use_decay=False,
            num_elites=None,
            **kwargs,
    ):
        super().__init__()

        self.is_probabilistic = False

        self.ensemble_size = ensemble_size
        self.input_size = input_size
        self.output_size = output_size

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        # data normalization
        self.input_mu = nn.Parameter(
            torch.zeros(1, input_size), requires_grad=False).float()
        self.input_std = nn.Parameter(
            torch.ones(1, input_size), requires_grad=False).float()
        self.output_mu = nn.Parameter(
            torch.zeros(1, output_size), requires_grad=False).float()
        self.output_std = nn.Parameter(
            torch.ones(1, output_size), requires_grad=False).float()

        self.fcs = []
        self.use_decay = use_decay

        in_size = input_size
        weight_decay_list = [0.000025, 0.00005, 0.000075, 0.000075, 0.0001]     # From PETS, also used in pytorch-mbpo
        for i, next_size in enumerate(hidden_sizes):
            # layer_size = (ensemble_size, in_size, next_size)
            fc = EnsembleFCLayer(
                ensemble_size, in_size, next_size,
                b_init_value=b_init_value,
                weight_decay=weight_decay_list[i],
            )
            self.__setattr__('ensemble_fc%d'% i, fc)
            self.fcs.append(fc)
            in_size = next_size

        self.ensemble_last_fc = EnsembleFCLayer(
            ensemble_size, in_size, output_size,
            b_init_value=b_init_value,
            weight_decay=weight_decay_list[min(len(hidden_sizes), len(weight_decay_list)-1)],
        )

        # Truncated normal initialization
        init_func = lambda m: w_init_method(m, b_init_value=b_init_value)
        self.apply(init_func)

        self.mse_criterion = nn.MSELoss()
        self.num_elites = num_elites
        if num_elites is None:
            self.num_elites = self.ensemble_size
        self.elite_models: List[int] = None

        self._propagation_method = propagation_method
        self._propagation_indices: torch.Tensor = None

        self._trained = False       # Whether the model has already been trained
        self._apply_output_transforms: bool = False

    def set_output_transforms(self, use_output_transform: bool):
        self._apply_output_transforms = use_output_transform

    def _maybe_toggle_layers_use_only_elite(self, only_elite: bool):
        if self.elite_models is None or (len(self.elite_models) == self.ensemble_size):
            return
        if self.ensemble_size > 1 and only_elite:
            for layer in self.fcs:
                layer.set_elite(self.elite_models)
                layer.toggle_use_only_elite()
            self.ensemble_last_fc.set_elite(self.elite_models)
            self.ensemble_last_fc.toggle_use_only_elite()

    def _default_forward(
            self, x: torch.Tensor, only_elite: bool = False, **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Use elite models? Toggle it on
        self._maybe_toggle_layers_use_only_elite(only_elite)

        # input normalization
        h = (x - self.input_mu) / self.input_std

        # standard feedforward network
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            h = self.hidden_activation(h)
        preactivation = self.ensemble_last_fc(h)
        output = self.output_activation(preactivation)

        # Toggle off
        self._maybe_toggle_layers_use_only_elite(only_elite)
        return output, None

    def _forward_from_indices(
            self, x: torch.Tensor, propagation_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, batch_size, _ = x.size()
        num_models = (
            len(self.elite_models) if self.elite_models is not None else self.ensemble_size
        )
        assert batch_size % num_models == 0

        # Shuffle the input tensor and let them go through the model
        shuffled_x = x[:, propagation_indices, ...].view(num_models, batch_size // num_models, -1)
        mean, logvar = self._default_forward(shuffled_x, only_elite=True)

        # Undo shuffling
        mean = mean.view(batch_size, -1)
        mean[propagation_indices] = mean.clone()

        if logvar is not None:
            logvar = logvar.view(batch_size, -1)
            logvar[propagation_indices] = logvar.clone()

        return mean, logvar

    def set_propagation_method(self, propagation_method: Optional[str] = None):
        self._propagation_method = propagation_method

    def get_propagation_method(self):
        return self._propagation_method

    def _forward_ensemble(
            self,
            x: torch.Tensor,
            rng: Optional[torch.Generator] = None,
    ):
        if self._propagation_method is None:
            mean, logvar = self._default_forward(x, only_elite=True)
            if self.model_len == 1:
                mean = mean[0]
                logvar = logvar[0] if logvar is not None else None
            return mean, logvar

        assert x.ndim == 2
        if self._propagation_method == "all":
            x = x.view(self.model_len, -1, self.input_size)       # [K, N, d]
            mean, logvar = self._default_forward(x, only_elite=True)
            if logvar is None:
                mean = mean.view(-1, self.output_size)
            else:
                mean = mean.view(-1, self.output_size // 2)
                logvar = logvar.view(-1, self.output_size // 2)
            return mean, logvar

        if x.size(0) % self.model_len != 0:
            raise ValueError(
                f"The ensemble model requires batch size to be a multiple of the "
                f"number of models. Current batch size is {x.size(0)} for "
                f"{self.model_len} models."
            )
        x = x.unsqueeze(0)
        if self._propagation_method == "random_model":
            propagation_indices = ptu.randperm(x.size(1))
            return self._forward_from_indices(x, propagation_indices)
        if self._propagation_method == "fixed_model":
            return self._forward_from_indices(x, self._propagation_indices)
        if self._propagation_method == "expectation":
            x = x.view(self.model_len, -1, self.input_size)
            mean, logvar = self._default_forward(x, only_elite=True)
            return mean.mean(dim=0), logvar.mean(dim=0) if logvar is not None else None
        raise ValueError(f"Invalid propagation method {self._propagation_method}.")

    def forward(
            self,
            x: torch.Tensor,
            rng: Optional[torch.Generator] = None,
            use_propagation: bool = True,
            **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Computes the predictions for the given input. If self.is_probabilistic is ``True``, mean and logvar
        predictions will be computed.

        When ``self.ensemble_size > 1``, the model supports uncertainty propagation options
        that can be used to aggregate the outputs of the different models in the ensemble.
        Valid propagation options are:
            - "random_model": for each output in the batch a model will be chosen at random.
              This corresponds to TS1 propagation in the PETS paper.
            - "fixed_model": for output j-th in the batch, the model will be chosen according to
              the model index in `propagation_indices[j]`. This can be used to implement TSinf
              propagation, described in the PETS paper.
            - "expectation": the output for each element in the batch will be the mean across
              models.

        Args:
            x (torch.Tensor): the input to the model. When ``self.propagation is None``,
                the shape must be ``E x B x d`` or ``B x Id``, where ``E``, ``B``
                and ``d`` represent ensemble size, batch size, and input dimension, respectively.
                In this case, each model in the ensemble will get one slice
                from the first dimension (e.g., the i-th ensemble member gets ``x[i]``).
                For other values of ``self.propagation`` (and ``use_propagation=True``),
                the shape must be ``B x Id``.
            rng (torch.Generator, optional): random number generator to use for "random_model"
                propagation.
            use_propagation (bool): if ``False``, the propagation method will be ignored
                and the method will return outputs for all models. Defaults to ``True``.

        Returns:
            (tuple of two tensors): the predicted mean and log variance of the output. If the model is
            a deterministic ensemble, then ``logvar`` will be ``None``. If ``propagation is not None``,
            the output will be 2-D (batch size, and output dimension). Otherwise, the outputs will have shape
            ``E x B x Od``, where ``Od`` represents output dimension.
        """
        if use_propagation:
            return self._forward_ensemble(x, rng=rng)
        return self._default_forward(x, **kwargs)

    def sample(
            self,
            x: torch.Tensor,
            deterministic: bool = False,
            rng: Optional[torch.Generator] = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if deterministic or not self.is_probabilistic:
            samples, _ = self(x, rng=rng, **kwargs)

            # Output transformed back to the original scale
            if self._apply_output_transforms:
                samples = samples * (self.output_std + 1e-8) + self.output_mu
                # TODO: is masking out very small values necessary?
            return samples, None

        assert rng is not None
        mean, logvar = self(x, rng=rng, **kwargs)
        var = torch.exp(logvar)
        std = torch.sqrt(var)
        samples = torch.normal(mean, std, generator=rng)

        # Output transformed back to the original scale
        if self._apply_output_transforms:
            samples = samples * (self.output_std + 1e-8) + self.output_mu
        return samples, logvar

    def fit_input_stats(self, data, mask=None):
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        std[std < 1e-12] = 1.0
        if mask is not None:
            mean *= mask
            std *= mask

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

    def get_loss(
            self, x, y, train=True, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the MSE loss

        Args:
            x (torch.Tensor): Input tensor
            y (torch.Tensor): Target tensor
            train (bool): Whether training mode or evaluation mode

        Returns:
            total_loss (torch.Tensor): The loss to which gradients of parameters are going to be computed
            mse_loss (torch.Tensor): Per model MSE loss returned for evaluation purpose
        """
        assert x.ndim == y.ndim
        if x.ndim == 2:             # This case can occur only when self.ensemble_size == 1
            assert len(self) == 1   # for testing..
            x.unsqueeze_(0)
            y.unsqueeze_(0)

        y_pred, *_ = self(x, use_propagation=False)     # Propagation only used for model rollouts

        mse_loss = torch.mean(torch.square(y_pred - y), dim=(1, 2))     # mse loss per model
        total_loss = mse_loss.sum()                                     # Total loss

        if self.use_decay and train:
            total_loss += self.get_decay_loss()
        return total_loss, mse_loss

    def eval_score(self, x_eval: torch.Tensor, y_eval: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Computes the MSE losses of a given batch for each and every model in the ensemble.
        """
        with torch.no_grad():
            _, mse_loss = self.get_loss(x_eval, y_eval, **kwargs)
            return mse_loss

    def get_decay_loss(self):
        decay_loss = 0.
        for m in self.children():
            if isinstance(m, EnsembleFCLayer):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.
        return decay_loss

    def _sample_propagation_indices(
            self, batch_size: int, _rng: torch.Generator
    ):
        """Returns a random permutation of integers in [0, ``batch_size``)."""
        if batch_size % self.model_len != 0:
            raise ValueError(
                f"To use the ensemble propagation, the batch size ({batch_size}) must "
                f"be a multiple of the number of models in the ensemble ({self.model_len})."
            )
        self.propagation_indices = ptu.randperm(batch_size, torch_device=torch.device('cpu'))

    @property
    def propagation_indices(self):
        return self._propagation_indices.to(ptu.device)         # move these to gpu when requested

    @propagation_indices.setter
    def propagation_indices(self, ind: torch.Tensor):
        self._propagation_indices = ind

    def save(self, path):
        """Saves the model to the given path"""
        torch.save(self.state_dict(), path)

    def reset(
            self, x: torch.Tensor, rng: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Initializes any internal dependent state when using the model for simulation.

        Initializes model indices for "fixed_model" propagation method of a bootstrapped ensemble
        with TSinf propagation.
        """
        assert rng is not None              # actually, rng is not used due to a bug of ``torch.randperm``
        self._sample_propagation_indices(x.size(0), rng)
        return x

    def __len__(self):
        return self.ensemble_size

    # Load the model parameters if possible
    def load(self, path: str, key=None):
        try:
            state_dict = torch.load(path, map_location=ptu.device)
            try:
                self.load_state_dict(state_dict)
                if 'elite_models' in state_dict:
                    self.elite_models = state_dict['elite_models']
                    logger.log(f"Elite models loaded to {type(self)}")
                else:
                    logger.log(f"Elite model is not specified to {type(self)}... skip loading")
                self.trained = True
            except RuntimeError:
                self.load_state_dict(state_dict[key])
                if f'{key}_elite_models' in state_dict:
                    self.elite_models = state_dict[f'{key}_elite_models']
                    logger.log(f"Elite models loaded to {type(self)}")
                else:
                    logger.log(f"Elite model is not specified to {type(self)}... skip loading")
                self.trained = True
            
        except FileNotFoundError:
            logger.log(f"{path} not found!")
        except Exception as e:
            logger.log(f"Failed loading saved parameters in {path}")
            logger.log(str(e))

    @property
    def model_len(self):
        return len(self.elite_models) if self.elite_models is not None else self.ensemble_size

    @property
    def trained(self):
        return self._trained

    @trained.setter
    def trained(self, trained: bool):
        self._trained = trained

    @property
    def elite_models(self):
        return self._elite_models
    
    @elite_models.setter
    def elite_models(self, elite_models):
        self._elite_models = elite_models
    
    def set_elite(self, elite_indices: Sequence[int]):
        """For ensemble models, indicates if some models should be considered elites"""
        if len(elite_indices) != self.ensemble_size:
            self.elite_models = list(elite_indices)

    def reset_elite_models(self):
        self.elite_models: List[int] = None


class FlattenEnsembleMLP(Ensemble):
    """Implements the ensemble network to be used for Q functions
    
    Note: the code is adapted from the official EDAC (An et al., 2021) repo.
    """
    def __init__(
            self,
            ensemble_size,                      # The number of models in the ensemble
            hidden_sizes,
            input_size,                         # The input dimension to the ensemble network
            output_size,                        # The output dimension of the ensemble network
            hidden_activation=swish,
            output_activation=identity,
            w_init_method=truncated_normal_init,
            b_init_value=0.0,
            sampling_method='min',                # Sampling method: 'min' or 'mean' or 'lcb'
            use_decay=False,
            num_elites=None,
            final_init_scale=None,              # From EDAC
            init_w=3e-3,                        # From EDAC
            **kwargs,
    ):
        super().__init__(
            ensemble_size=ensemble_size,
            hidden_sizes=hidden_sizes,
            input_size=input_size,
            output_size=output_size,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            w_init_method=w_init_method,
            b_init_value=b_init_value,
            use_decay=use_decay,
            num_elites=num_elites,
            **kwargs,
        )
        self.sampling_method = sampling_method

        if kwargs.get('lcb_coeff', False):
            self.lcb_coeff = kwargs['lcb_coeff']
        
        if final_init_scale is None:
            self.ensemble_last_fc.weight.data.uniform_(-init_w, init_w)
            self.ensemble_last_fc.bias.data.uniform_(-init_w, init_w)
        else:
            for j in range(self.ensemble_size - 1):
                ptu.ortho_init(self.ensemble_last_fc[j], final_init_scale)
                self.ensemble_last_fc.bias[j].data.fill_(0)
    
    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=-1)

        dim = len(flat_inputs.shape)

        # --- Below followed EDAC implementation ---
        # repeat h to make amenable to parallelization
        # if dim = 3, then we probably already did this somewhere else
        # (e.g. bootstrapping in training optimization)
        if dim < 3:
            flat_inputs = flat_inputs.unsqueeze(0)
            if dim == 1:
                flat_inputs = flat_inputs.unsqueeze(0)
            flat_inputs = flat_inputs.repeat(self.ensemble_size, 1, 1)
        
        output, _ = super()._default_forward(flat_inputs, **kwargs)
        
        # if original dim was 1D, squeeze the extra created layer
        if dim == 1:
            output = output.squeeze(1)

        # output is (ensemble_size, batch_size, output_size)
        return output
        
    
    def sample(self, *inputs, sampling_method: Optional[str] = None):
        preds = self.forward(*inputs)

        if sampling_method is None:
            sampling_method = self.sampling_method

        if sampling_method == 'min':
            return torch.min(preds, dim=0)[0]
        elif sampling_method == 'mean':
            return torch.mean(preds, dim=0)
        elif sampling_method == 'lcb':
            std, mean = torch.std_mean(preds, dim=0, unbiased=True)
            lcb_coeff = 1.0
            if hasattr(self, 'lcb_coeff'):
                lcb_coeff = self.lcb_coeff
            return mean - lcb_coeff * std

    def get_loss(
            self, x, y, train=True, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the MSE loss

        Args:
            x (torch.Tensor): Input tensor
            y (torch.Tensor): Target tensor
            train (bool): Whether training mode or evaluation mode

        Returns:
            total_loss (torch.Tensor): The loss to which gradients of parameters are going to be computed
            mse_loss (torch.Tensor): Per model MSE loss returned for evaluation purpose
        """
        assert x.ndim == y.ndim
        if x.ndim == 2:             # This case can occur only when self.ensemble_size == 1
            assert len(self) == 1   # for testing..
            x.unsqueeze_(0)
            y.unsqueeze_(0)

        y_pred = self(x, use_propagation=False)     # Propagation only used for model rollouts

        mse_loss = torch.mean(torch.square(y_pred - y), dim=(1, 2))     # mse loss per model
        total_loss = mse_loss.sum()                                     # Total loss

        if self.use_decay and train:
            total_loss += self.get_decay_loss()
        return total_loss, mse_loss
