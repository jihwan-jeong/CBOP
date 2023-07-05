from typing import cast

import abc

from rlkit.core.rl_algorithms.offline.offline_rl_algorithm import OfflineRLAlgorithm
from rlkit.core.logging.logging import logger
from os import path as osp


class OfflinePretrainAlgorithm(OfflineRLAlgorithm, metaclass=abc.ABCMeta):
    """Implements the learning routine for pretraining some models
    
    self.trainer , which are then trained sequentially.
    Examples include 
        (1) MBRLTrainer which trains the dynamics model (+ reward)
        (2) Policy evaluation (evaluate the value function given the offline data)
        (3) Policy prior (the behavior cloned policy of the given offline data)
    """
    def __init__(
            self,
            trainer,
            replay_buffer,
            batch_size: int,
            timeout=None,
            use_best_parameters: bool = False,
            save_best_parameters: bool = False,
            save_snapshot_gap: int = 10,
            num_total_epochs: int = None,
            max_grad_steps: int = None,
            holdout_pct: float = 0.2,
            max_logging: int = int(1e3),
            max_epochs_since_last_update: int = 5,
            improvement_threshold: float = 0.01,
            *args,
            **kwargs,
    ):
        self.trainer = trainer
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.use_best_parameters = use_best_parameters
        self.save_best_parameters = save_best_parameters
        self.num_total_epochs = num_total_epochs
        self.max_grad_steps = max_grad_steps
        self.holdout_pct = holdout_pct
        self.max_logging = max_logging
        self.max_epochs_since_last_update = max_epochs_since_last_update
        self.improvement_threshold = improvement_threshold
        logger.set_snapshot_gap(save_snapshot_gap)

        self.timeout = timeout
        
    def _train(self):
        logger.log(
            "Start pretraining from the following trainers...\n" +\
            f"\t{[trainer.name for trainer in self.trainer.trainers]}"
        )
        self.trainer.train_from_buffer(
            replay_buffer=self.replay_buffer,
            holdout_pct=self.holdout_pct,
            max_grad_steps=self.max_grad_steps,
            max_epochs_since_last_update=self.max_epochs_since_last_update,
            num_total_epochs=self.num_total_epochs,
            improvement_threshold=self.improvement_threshold,
            use_best_parameters=self.use_best_parameters,
            max_logging=self.max_logging,
        )

    def get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        return snapshot
