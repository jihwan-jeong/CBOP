import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rlkit.torch.pytorch_util as ptu
from typing import Union, Optional, Tuple, List, Callable, Sequence
from pathlib import Path

import rlkit.types
from ..ensemble import Ensemble
from ..probabilistic_ensemble import ProbabilisticEnsemble


class Model(nn.Module):
    """A base abstract class used by all dynamics models

    """

class DynamicsModel(nn.Module):
    """Wrapper class for dynamics models.

    Following the convention in PETS (Chua et al., 2018), PDDM (Nagabandi et al., 2020),
    and MBOP (Argenson & Dulac-Arnold, 2021), the dynamics model of the environment can be modeled as a probabilistic
    ensemble network. This wrapper class provides the option to use a deterministic ensemble network as well.

    The specifics of model construction and forward passes are defined in more generic modules in
    :class:`rlkit.torch.models.ensemble.Ensemble` and
    in :class:`rlkit.torch.models.probabilistic_ensemble.ProbabilisticEnsemble`.
    The ensemble classes defined there can not only be used for dynamics models,
    but also for value ensemble models, etc. In contrast, DynamicsModel class defines specific methods and attributes
    necessary for using such ensembles as transition and reward functions of an environment.
    """
    def __init__(
            self,
            model: Union[Ensemble, ProbabilisticEnsemble],  # The ensemble model to use
            input_dim: int,
            obs_dim: int,                   # The observation dimension in the environment
            action_dim: int,                # The action dimension in the environment
            learn_reward: bool,             # Whether to learn the reward function as well
            learn_logstd_min_max: bool,     # Whether to learn the logstd_min and logstd_max parameters as well
            is_probabilistic: bool,  # Whether the network is probabilistic or deterministic
            normalize_outputs: bool,
            normalize_inputs: bool,
            num_elites: Optional[int] = None,
            use_true_reward: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.model = model
        self.input_dim = input_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.learn_reward = learn_reward
        self.use_true_reward = use_true_reward
        self.learn_logstd_min_max = learn_logstd_min_max
        self.is_probabilistic = is_probabilistic
        self.normalize_outputs = normalize_outputs
        self.normalize_inputs = normalize_inputs
        self.separate_reward_func = None
        self.ensemble_size = len(self.model)
        self.num_particle = None
        self.num_elites = num_elites
        if num_elites is None and isinstance(self.model, Ensemble):
            self.num_elites = self.model.ensemble_size
        self.elite_models: List[int] = (
            list(range(self.model.ensemble_size)) if isinstance(self.model, Ensemble)
            else None
        )

        self.trained = False       # Whether the model has already been trained
        
    def set_reward_func(self, reward_func: Optional[Callable] = None):
        self.separate_reward_func = reward_func

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Calls the forward method of the ensemble model to get the predicted next observations
        and (optionally) rewards. If a separate reward network should be used, the case should be handled separately.
        """
        if self.separate_reward_func is None:
            return self.model.forward(x, **kwargs)
        else:
            preds = self.model.forward(x, **kwargs)[0]
            x_ = x.expand(self.separate_reward_func.ensemble_size, -1, -1)
            rew_preds = self.separate_reward_func.forward(x_, **kwargs)
            rew_preds = rew_preds.mean(dim=0)
            output = torch.cat([preds, rew_preds], dim=-1)
            return (output, None)

    def get_loss(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor,
            **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        if self.separate_reward_func is None:
            total_loss, mse_loss = self.model.get_loss(inputs, targets, **kwargs)
            if self.learn_logstd_min_max and isinstance(self.model, ProbabilisticEnsemble):
                total_loss += 0.01 * torch.sum(self.model.max_logvar) - 0.01 * torch.sum(self.model.min_logvar)
            return total_loss, mse_loss
        else:
            rew_targets, trans_targets = targets[:, :, :1], targets[:, :, 1:]
            trans_total_loss, trans_mse_loss = self.model.get_loss(inputs, trans_targets, **kwargs)
            rew_total_loss, rew_mse_loss = self.separate_reward_func.get_loss(inputs, rew_targets, **kwargs)
            if self.learn_logstd_min_max and isinstance(self.model, ProbabilisticEnsemble):
                trans_total_loss += 0.01 * torch.sum(self.model.max_logvar) - 0.01 * torch.sum(self.model.min_logvar)
                rew_total_loss += 0.01 * torch.sum(self.separate_reward_func.max_logvar) \
                            - 0.01 * torch.sum(self.separate_reward_func.min_logvar)
            return (rew_total_loss, rew_mse_loss), (trans_total_loss, trans_mse_loss)

    def eval_score(
            self,
            x_eval: torch.Tensor, y_eval: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the MSE losses of a given batch for each and every model in the ensemble.
        """
        with torch.no_grad():
            if self.separate_reward_func is None:
                _, mse_loss = self.model.get_loss(x_eval, y_eval, train=False,
                                                  inc_var_loss=False)
            else:
                y_eval_rew, y_eval_trans = y_eval[:, :, :1], y_eval[:, :, 1:]
                _, trans_mse_loss = self.model.get_loss(x_eval, y_eval_trans, train=False,
                                                        inc_var_loss=False)
                _, rew_mse_loss = self.separate_reward_func.get_loss(x_eval, y_eval_rew, train=False,
                                                                    inc_var_loss=False)
                mse_loss = (trans_mse_loss * self.obs_dim + rew_mse_loss) / (self.obs_dim + 1)  # averaged over dimensions
            return mse_loss

    def fit_input_stats(self, data, mask=None):
        self.model.fit_input_stats(data, mask)
        if self.separate_reward_func is not None:
            self.separate_reward_func.fit_input_stats(data, mask)

    def fit_output_stats(self, data, mask=None):
        return self.model.fit_output_stats(data, mask)

    def sample(
            self,
            x,
            deterministic: bool = False,
            rng: Optional[torch.Generator] = None,
            return_logvar: bool = False,
            **kwargs,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Samples next observations and rewards from the underlying model(s).

        This wrapper assumes that the underlying model's sample method returns a tuple with just one tensor,
        which concatenates next_observation and reward.

        Args:
            x (torch.Tensor): The input (obs, action) tensor
            deterministic (bool): If `True`, the model returns a deterministic "sample" (e.g., the mean prediction).
                Defaults to `False`.
            rng (torch.Generator): A rng to use for sampling.
            return_logvar (bool): If `True`, the model additionally returns the log variance of (s', r) prediction

        Returns:
            (tuple of torch.Tensor) Predicted next_obs (o_{t+1}) and rewards (r_{t}) (and logvar optionally).
        """
        # Prepare inputs to the dynamics model
        assert x.ndim == 2
        model_in = x

        """Use the underlying ensemble models for prediction"""
        # Scale samples properly
        assert hasattr(self.model, 'set_output_transforms')
        self.model.set_output_transforms(True)

        # Sample next state predictions (and rewards) from the dynamics model
        preds, logvar = self.model.sample(
            model_in, deterministic=deterministic, rng=rng,
        )
        if not self.use_true_reward and self.learn_reward and self.separate_reward_func is None:
            reward, next_obs = preds[:, :1], preds[:, 1:]
        else:
            next_obs = preds
            if not self.use_true_reward and self.separate_reward_func is not None:
                """
                MBOP uses the average reward with a separate reward function (Argenson & Dulac-Arnold, 2021).
                """
                # Scale outputs properly
                assert hasattr(self.separate_reward_func, 'set_output_transforms')
                self.separate_reward_func.set_output_transforms(True)

                # Sample rewards from the reward model
                if self.separate_reward_func.get_propagation_method() is None:
                    model_in_rew = model_in[None, :].expand(self.separate_reward_func.model_len, -1, -1)
                    preds_rew, logvar_rew = self.separate_reward_func.sample(
                        model_in_rew, deterministic=deterministic, rng=rng
                    )
                    assert preds_rew.ndim == 3 and preds_rew.size(0) == self.separate_reward_func.model_len
                    reward = preds_rew.mean(dim=0)  # Take the average among ensemble models
                    if logvar is not None and logvar_rew is not None:
                        logvar_rew = logvar_rew.mean(dim=0)
                        logvar = torch.cat((logvar_rew, logvar), dim=-1)
                else:
                    model_in_rew = model_in
                    reward, logvar_rew = self.separate_reward_func.sample(
                        model_in_rew, deterministic=deterministic, rng=rng
                    )
                    if logvar is not None and logvar_rew is not None:
                        logvar = torch.cat((logvar_rew, logvar), dim=-1)
                self.separate_reward_func.set_output_transforms(False)
            else:
                reward = None
        self.model.set_output_transforms(False)
        return (reward, next_obs) if not return_logvar else (reward, next_obs, logvar)

    def set_num_particle(self, num_particle: int):
        self.num_particle = num_particle

    ## TODO: if separate reward function is used, its parameters should also be saved/loaded
    def load(self, model_path: str, key='dynamics'):
        try:
            state_dict = torch.load(model_path, map_location=ptu.device)
            try:
                self.load_state_dict(state_dict)
                if self.elite_models is not None:
                    self.set_elite(state_dict[f'{key}_elite_models'])
                self.trained = True
            except RuntimeError:
                self.load_state_dict(state_dict[key])
                if self.elite_models is not None:
                    self.set_elite(state_dict[f'{key}_elite_models'])
                self.trained = True
        except FileNotFoundError:
            print(f"{model_path} not found!")
        except Exception as e:
            print(f"Failed loading saved parameters in {model_path}")
            print(e)

    def reset(
            self, obs: rlkit.types.TensorType, rng: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Calls :meth:`Ensemble.reset`

        Args:
             obs (torch.Tensor or np.ndarray): the initial input to the model
             rng (torch.Generator): a rng to use for sampling the model indices

        Returns:
              (torch.Tensor) The output of the underlying model
        """
        if isinstance(obs, np.ndarray):
            obs = ptu.from_numpy(obs).float()
        obs = self.model.reset(obs, rng=rng)
        if self.separate_reward_func is not None:
            self.separate_reward_func.reset(obs, rng=rng)
        return obs

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
        self.elite_models = list(elite_indices)
        self.model.set_elite(elite_indices)
        if self.separate_reward_func is not None:
            self.separate_reward_func.set_elite(elite_indices)

    def reset_elite_models(self):
        self.elite_models = (
            list(range(self.model.ensemble_size)) if isinstance(self.model, Ensemble)
            else None
        )
        self.model.reset_elite_models()
        if self.separate_reward_func is not None:
            self.separate_reward_func.reset_elite_models()

    @property
    def model_len(self):
        return self.model.model_len

    @property
    def config(self):
        config = \
            f"{'_prob' if self.is_probabilistic else ''}" + \
            f"{'_nin' if self.normalize_inputs else ''}" + \
            f"{'_nout' if self.normalize_outputs else ''}" + \
            f"{'_rew' if self.learn_reward else ''}" + \
            f"{'_sep' if self.separate_reward_func is not None else ''}"
        return config.strip('_')


class LatentDynamicsModel(DynamicsModel):
    """Wrapper class for latent dynamics models"""
    def __init__(
        self,
        state_autoencoder,
        model: Union[Ensemble, ProbabilisticEnsemble], 
        input_dim: int, 
        obs_dim: int, 
        latent_dim: int,
        action_dim: int, 
        learn_reward: bool, 
        learn_logstd_min_max: bool, 
        is_probabilistic: bool = True, 
        num_elites: Optional[int] = None, 
        **kwargs
    ):
        super().__init__(
            model, 
            input_dim, 
            obs_dim,
            action_dim, 
            learn_reward, 
            learn_logstd_min_max, 
            is_probabilistic, 
            num_elites, 
            **kwargs
        )

        self.latent_dim = latent_dim
        self.state_autoencoder = state_autoencoder
        self._is_ae_fixed = False

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Take the observation tensor 
        obs = x[..., :-self.action_dim]
        act = x[..., -self.action_dim:]

        # First, get the latent embedding using the trained encoder
        embedding = self.state_autoencoder.encoder(obs).detach()
        z_act = torch.cat([embedding, act], dim=-1)

        if self.separate_reward_func is None:
            # Then, the latent ensemble model takes the forward step
            z_prime, log_z_prime = self.model.forward(z_act, **kwargs)
            assert log_z_prime is None, "Currently, do not support ProbabilisticEnsemble models"

            # Next latent state decoded back to next state prediction
            preds = self.state_autoencoder.decoder(z_prime)
            return (preds, None)

        else:
            raise NotImplementedError
            # # Then, the latent ensemble model takes the forward step
            # z_prime = self.model.forward(z_act, **kwargs)[0]

            # # Next latent state decoded back to next state prediction
            # preds = self.autoencoder.decoder(z_prime)

            # # Reward prediction is made separately
            # rew_input = z_act.expand(self.separate_reward_func.ensemble_size, -1, -1)
            # rew_preds = self.separate_reward_func.forward(rew_input, **kwargs)
            # rew_preds = rew_preds.mean(dim=0)
            # output = torch.cat([preds, rew_preds], dim=-1)
            # return (output, None)
    
    def get_loss_from_latent(
        self,
        z_act_input: torch.Tensor,
        z_target: torch.Tensor, 
        **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        self._detach_autoencoder()
        train = kwargs.get('train', True)

        assert z_act_input.ndim == z_target.ndim
        if z_act_input.ndim == 2:
            assert len(self.model) == 1
            z_act_input.unsqueeze_(0)
            z_target.unsqueeze_(0)
        
        z_pred, *_ = self.model(z_act_input, use_propagation=False)
        mse_loss = torch.mean(torch.square(z_pred - z_target), dim=(1, 2))
        total_loss = mse_loss.sum()

        if self.model.use_decay and train:
            total_loss += self.model.get_decay_loss()
        return total_loss, mse_loss
        
    def get_loss(self, inputs: torch.Tensor, targets: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, ...]:
        self._detach_autoencoder()
        train = kwargs.get('train', True)

        # Take the observation tensor 
        obs = inputs[..., :-self.action_dim]
        act = inputs[..., -self.action_dim:]

        # First, get the latent embedding using the trained encoder
        embedding = self.state_autoencoder.encoder(obs).detach()
        z_act = torch.cat([embedding, act], dim=-1)

        if self.separate_reward_func is None:
            # Due to latent decoding, cannot directly use `self.model.get_loss`
            assert z_act.ndim == targets.ndim
            if z_act.ndim == 2:
                assert len(self.model) == 1
                z_act.unsqueeze_(0)
                targets.unsqueeze_(0)
            
            z_pred, *_ = self.model(z_act, use_propagation=False)
            y_pred = self.state_autoencoder.decoder(z_pred)
            mse_loss = torch.mean(torch.square(y_pred, targets), dim=(1, 2))
            total_loss = mse_loss.sum()

            if self.model.use_decay and train:
                total_loss += self.model.get_decay_loss()
            return total_loss, mse_loss
        else:
            raise NotImplementedError

    def _detach_autoencoder(self):
        # Encoder / decoder parameters should not be updated
        if not self._is_ae_fixed:
            for param in self.state_autoencoder.parameters():
                param.requires_grad = False
            self._is_ae_fixed = True

    def _train_autoencoder(self):
        if self._is_ae_fixed:
            for param in self.state_autoencoder.parameters():
                param.requires_grad = True
            self._is_ae_fixed = False

    def sample(
        self, 
        x, 
        deterministic: bool = False, 
        rng: Optional[torch.Generator] = None, 
        return_logvar: bool = False, 
        latent: bool = True,
        **kwargs
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Given the (obs, act) tuple x, map it to the latent space and sample the next state transition"""
        assert not return_logvar

        if not latent:    
            assert x.size(-1) == self.obs_dim + self.action_dim

            obs = x[..., :self.obs_dim]
            act = x[..., self.obs_dim:]

            self._detach_autoencoder()
            latent_obs = self.state_autoencoder.encoder(obs)
            z_act_tuple = torch.cat([latent_obs, act], dim=-1)
            rew, next_z = super().sample(z_act_tuple, deterministic, rng, return_logvar, **kwargs)
            next_obs = self.state_autoencoder.decoder(next_z)
            return (rew, next_obs)
        else:
            assert x.size(-1) == self.latent_dim + self.action_dim
            z_act_tuple = x
            rew, next_z = super().sample(z_act_tuple, deterministic, rng, return_logvar, **kwargs)
            return (rew, next_z)
