import abc
from collections import OrderedDict
import copy

from typing import Optional, Dict, Tuple, Callable
from torch import nn as nn
import torch
import numpy as np

import gtimer as gt
from rlkit.core.rl_algorithms.offline.offline_pretrain_algorithm import OfflinePretrainAlgorithm
from rlkit.data_management.replay_buffers.replay_buffer import ReplayBuffer
from rlkit.policies.gaussian_policy import TanhGaussianPolicy
from rlkit.util.common import get_epoch_timings
from rlkit.core.logging.logging import logger
from rlkit.core.rl_algorithms.batch.batch_rl_algorithm import BatchRLAlgorithm
from rlkit.core.rl_algorithms.batch.mb_batch_rl_algorithm import MBBatchRLAlgorithm
from rlkit.core.rl_algorithms.offline.offline_rl_algorithm import OfflineRLAlgorithm
from rlkit.core.rl_algorithms.online.online_rl_algorithm import OnlineRLAlgorithm
from rlkit.core.rl_algorithms.online.mbrl_algorithm import MBRLAlgorithm
from rlkit.core.trainer import Trainer
from rlkit.torch.core import np_to_pytorch_batch
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.models.ensemble import Ensemble
from rlkit.torch.models.dynamics_models.model import DynamicsModel


class TorchOnlineRLAlgorithm(OnlineRLAlgorithm):
    def configure_logging(self, **kwargs):
        self.trainer.configure_logging(**kwargs)

    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)


class TorchBatchRLAlgorithm(BatchRLAlgorithm):
    def configure_logging(self, **kwargs):
        self.trainer.configure_logging(**kwargs)

    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)


class TorchMBRLAlgorithm(MBRLAlgorithm):
    def configure_logging(self, **kwargs):
        self.trainer.configure_logging(**kwargs)
        self.model_trainer.configure_logging(**kwargs)

    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)
        for net in self.model_trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)
        for net in self.model_trainer.networks:
            net.train(mode)


class TorchMBBatchRLAlgorithm(MBBatchRLAlgorithm):
    def configure_logging(self, **kwargs):
        self.trainer.configure_logging(**kwargs)
        self.model_trainer.configure_logging(**kwargs)

    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)
        for net in self.model_trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)
        for net in self.model_trainer.networks:
            net.train(mode)


class TorchOfflineRLAlgorithm(OfflineRLAlgorithm):
    def configure_logging(self, **kwargs):
        self.trainer.configure_logging(**kwargs)

    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)


class TorchOfflinePretrainAlgorithm(OfflinePretrainAlgorithm):
    def configure_logging(self, **kwargs):
        self.trainer.configure_logging(**kwargs)

    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)
        
    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)


class TorchTrainer(Trainer, metaclass=abc.ABCMeta):
    def __init__(self):
        self._num_train_total_steps = 0
        self._num_timesteps_this_epoch = 0
        self._num_train_steps_this_epoch = 0
        self._network_dict: Dict[str, nn.Module] = None

    def train(self, np_batch):
        batch = np_to_pytorch_batch(np_batch)
        self.train_from_torch(batch)
        self._num_train_total_steps += 1
        self._num_timesteps_this_epoch += 1

    def start_epoch(self, epoch):
        self._num_timesteps_this_epoch = 0
        self._num_train_steps_this_epoch = 0

    def get_diagnostics(self):
        return OrderedDict([
            ('num train calls', self._num_train_total_steps),
        ])

    def log_grad_norm(self, net, stats: OrderedDict, key: str):
        if isinstance(net, nn.Module):
            param_list = list(filter(lambda p: p.grad is not None, net.parameters()))
        elif isinstance(net, torch.Tensor):
            param_list = [net]
        else:
            raise ValueError
        norm = [None] * len(param_list)
        for i, p in enumerate(param_list):
            norm[i] = p.grad.data.norm(2).item()
        stats[f"{key} Grad Norm"] = np.mean(norm)

    def to(self, device):
        for net in self.networks:
            net.to(device)

    @abc.abstractmethod
    def train_from_torch(self, batch):
        pass

    @property
    def networks(self):
        if self._network_dict is None:
            raise NotImplementedError("Trainer class should specify associated torch modules")
        return [net for net in self._network_dict.values() if isinstance(net, nn.Module)]

    @property
    def network_dict(self):
        if self._network_dict is None or len(self._network_dict) == 0:
            raise NotImplementedError("Trainer class should specify associated torch modules")
        return self._network_dict

    def register_module(self, key: str, net):
        if self._network_dict is None:
            self._network_dict = {}
        self._network_dict[key] = net

    def get_snapshot(self):
        return {
            key: module.state_dict() if hasattr(module, 'state_dict') and isinstance(module.state_dict, Callable) 
                else module
            for key, module in self.network_dict.items()
        }

    @abc.abstractmethod
    def load(self, state_dict, prefix=''):
        pass

    def normalize_states(self, replay_buffer) -> Dict[str, np.ndarray]:
        pass
    
    def train_from_buffer(
            self,
            replay_buffer,
            *args,
            **kwargs
    ):
        pass

    def evaluate(
            self,
            model,
            x_eval: np.ndarray,
            y_eval: np.ndarray,
            batch_size: int = 1024,
    ):
        """
        Returns the MSE losses computed over {x_eval, y_eval} dataset for all ensemble models.
        """
        model.eval()
        is_ensemble = isinstance(model, Ensemble) or \
                        (isinstance(model, DynamicsModel) and isinstance(model.model, Ensemble))
        with torch.no_grad():
            batch_scores_list = []
            num_batches = int(np.ceil(x_eval.shape[0] / batch_size))

            for b in range(num_batches):
                x_batch, y_batch = x_eval[b * batch_size: (b + 1) * batch_size],\
                                   y_eval[b * batch_size: (b + 1) * batch_size]
                x_batch, y_batch = ptu.from_numpy(x_batch), ptu.from_numpy(y_batch)
                if is_ensemble:
                    x_batch, y_batch = x_batch.expand(model.ensemble_size, -1, -1), \
                                       y_batch.expand(model.ensemble_size, -1, -1)
                # Compute the squared loss
                batch_score = model.eval_score(x_batch, y_batch)   # Shape: (n_e, B, d)
                batch_scores_list.append(batch_score.unsqueeze(0) if is_ensemble else batch_score)

            # Concatenate along the batch axis
            batch_scores = torch.cat(batch_scores_list, dim=0)

            assert (batch_scores.ndim == 2 and is_ensemble) or (batch_scores.ndim == 1 and not is_ensemble),\
                "For debugging purpose"
            mean_axis = 0  # 1 if batch_scores.ndim == 2 else (1, 2)
            batch_scores = batch_scores.mean(dim=mean_axis)         # Shape: (n_e, )
            return batch_scores
        
    def _log_stats(self, epoch=None, prefix=''):
        if epoch is not None:
            logger.log("{} Epoch {} finished".format(prefix, epoch), with_timestamp=True)

        # Log some statistics to csv file
        logger.record_dict(
            self.get_diagnostics(), prefix=prefix
        )

        """
        Misc
        """
        logger.record_dict(get_epoch_timings())
        if epoch is not None:
            logger.record_tabular(f'{prefix}Epoch' if prefix else 'Epoch', epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
        gt.stamp('logging', unique=False)

    def maybe_get_best_weights(
            self,
            model,
            best_val_score: torch.Tensor,
            val_score: torch.Tensor,
            prev_params = None,
            threshold: float = 0.01,
    ) -> Tuple[Optional[Dict], torch.Tensor]:
        """
        Returns the current model state dict if the validation score improves.
        For ensembles, this checks the validation for each ensemble member separately.

        Args:
            best_val_score (tensor): the current best validation losses per model.
            val_score (tensor): the new validation loss per model.
            threshold (float): the threshold for relative improvement.
            prev_params (OrderedDict)
        
        Returns:
            (dict, optional): if the validation score's relative improvement over the
            best validation score is higher than the threshold, returns the state dictionary
            of the stored model, otherwise returns ``None``.
        """
        improvement = (best_val_score - val_score) / torch.abs(best_val_score)
        improved = (improvement > threshold)           # any improvements?
        best_val_score[improved] = val_score[improved]
        new_params_dict = copy.deepcopy(model.state_dict())
        # For ensemble models
        if best_val_score.squeeze().ndim >= 1 and len(best_val_score.squeeze()) > 1:
            assert prev_params is not None
            for key in model.state_dict():
                prev_param = prev_params[key]
                if prev_param.ndim > 2 and improved.any():
                    new_params_dict[key][~improved] = prev_param[~improved]
                    prev_param[improved] = new_params_dict[key][improved]
        return copy.deepcopy(new_params_dict) if improved.any() else None, best_val_score

    def _set_maybe_best_weights(
            self,
            model,
            best_weights: Optional[dict],
            best_val_score: torch.Tensor
    ):
        if best_weights is not None:
            model.load_state_dict(best_weights)

    @abc.abstractmethod
    def configure_logging(self, **kwargs):
        pass
