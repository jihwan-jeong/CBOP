from collections import OrderedDict
from typing import List
from os import path as osp
import gtimer as gt

from torch import nn as nn

from rlkit.core.rl_algorithms.torch_rl_algorithm import TorchTrainer
from rlkit.core.logging.logging import logger


class TrainerWrapper(TorchTrainer):
    """
    A wrapper trainer class that can include multiple trainers. 
    Currently used for pre-training in offline MBRL. 
    So, `train_from_torch` is disabled and you should use `train_from_buffer`.

    The trained model checkpoints are saved to `cache_dir`, which should be specified.
    """
    def __init__(
            self,
            trainers: list,
            save_dir,
            **kwargs
    ):
        super().__init__()
        self.trainers = trainers
        self.save_dir = save_dir

        self._need_to_update_eval_statistics = True
        self.name = 'wrapper'

        self._network_dict = {}

    def train_from_buffer(
        self, 
        replay_buffer, 
        holdout_pct=0.2, 
        max_epochs_since_last_update=100, 
        num_total_epochs=None, 
        improvement_threshold: float = 0.01, 
        use_best_parameters: bool = False, 
        *args, 
        **kwargs
    ):

        for trainer in self.trainers:
            logger.set_tabular_output(osp.join(logger.log_dir, f'{trainer.name}.csv'))
            orig_snapshot_dir = logger.get_snapshot_dir()
            orig_snapshot_mode = logger.get_snapshot_mode()
            logger.set_snapshot_mode(None)                      # Note: we do not store intermediate parameters
            trainer.train_from_buffer(
                replay_buffer=replay_buffer,
                holdout_pct=holdout_pct,
                max_epochs_since_last_update=max_epochs_since_last_update,
                num_total_epochs=num_total_epochs,
                improvement_threshold=improvement_threshold,
                use_best_parameters=use_best_parameters,
                *args,
                **kwargs
            )
            # Save the trained parameters
            snapshot = trainer.get_snapshot()
            logger.set_snapshot_dir(self.save_dir)
            logger.set_snapshot_mode('last')                    # Only store the final best parameters
            logger.save_itr_params(0, snapshot, prefix=trainer.name)
            logger.set_snapshot_dir(orig_snapshot_dir)
            logger.set_snapshot_mode(orig_snapshot_mode)

            gt.stamp(f'{trainer.name}', unique=True)

    def train_from_torch(self, batch):
        raise NotImplementedError

    def normalize_states(self, replay_buffer):
        for trainer in self.trainers:
            trainer.normalize_states(replay_buffer)

    def get_diagnostics(self):
        for trainer in self.trainers:
            self.eval_statistics.update(trainer.eval_statistics)
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True
        for trainer in self.trainers:
            trainer.end_epoch(epoch)

    @property
    def networks(self):
        networks = []
        for trainer in self.trainers:
            networks += trainer.networks
        return networks

    def get_snapshot(self):
        snapshot = dict()
        for trainer in self.trainers:
            snapshot.update(trainer.get_snapshot())
        return snapshot

    @property
    def trainers(self) -> List[TorchTrainer]:
        return self._trainers

    @trainers.setter
    def trainers(self, trainers: List[TorchTrainer]):
        assert isinstance(trainers, list)
        self._trainers = []
        for trainer in trainers:
            if trainer is not None:
                self._trainers.append(trainer)

    def configure_logging(self, **kwargs):
        for trainer in self.trainers:
            trainer.configure_logging(**kwargs)

    def load(self, state_dict, prefix=''):
        for trainer in self.trainers:
            trainer.load(state_dict, prefix=prefix)

    @property
    def network_dict(self):
        for trainer in self.trainers:
            self._network_dict.update(trainer.network_dict)
        return self._network_dict
