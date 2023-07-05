import abc
import numpy as np
from collections import OrderedDict

import gtimer as gt
import time
from typing import Optional, Union

from rlkit.core.logging.logging import logger
from rlkit.util import eval_util
from rlkit.util.common import get_epoch_timings
from rlkit.data_management.replay_buffers.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import DataCollector

class BaseRLAlgorithm(object, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: DataCollector,
            evaluation_data_collector: DataCollector,
            replay_buffer: ReplayBuffer,
            save_snapshot_gap=10,
            save_best_parameters=False,
            timeout=None,
    ):
        self.trainer = trainer
        self.expl_env = exploration_env
        self.eval_env = evaluation_env
        self.expl_data_collector = exploration_data_collector
        self.eval_data_collector = evaluation_data_collector
        self.replay_buffer = replay_buffer
        self._start_epoch = 0
        self.save_snapshot_gap = save_snapshot_gap
        self.save_best_parameters = save_best_parameters
        logger.set_snapshot_gap(save_snapshot_gap)

        self._best_avg_return = -9e10
        self.post_epoch_funcs = []
        self.timeout = timeout
        self._start_time = None

    def train(self, start_epoch=0):
        self._start_epoch = start_epoch
        self._train()

    def _train(self):
        """
        Train model.
        """
        raise NotImplementedError('_train must implemented by inherited class')

    def _start_training(self):
        self._start_time = time.time()

    def end_epoch(self, epoch):
        """Save snapshots"""
        snapshot = self._get_snapshot()

        # Firstly, save the best parameters
        if self.save_best_parameters:
            eval_paths = self.eval_data_collector.get_epoch_paths()
            if len(eval_paths) > 0:
                returns = [sum(path['rewards']) for path in eval_paths]
                avg_return = np.mean(returns)
                if avg_return > self._best_avg_return:
                    self._best_avg_return = avg_return
                    logger.set_snapshot_mode('last')
                    logger.save_itr_params(epoch, snapshot, prefix='expert')

        # Also, periodically save the parameters
        logger.set_snapshot_mode('gap_and_last')
        logger.save_itr_params(epoch, snapshot, prefix=f'{self.trainer.name}')
        gt.stamp('saving', unique=False)

        self._log_stats(epoch)

        self.expl_data_collector.end_epoch(epoch)
        self.eval_data_collector.end_epoch(epoch)
        self.replay_buffer.end_epoch(epoch)
        self.trainer.end_epoch(epoch)

    def start_epoch(self, epoch):
        self.replay_buffer.start_epoch(epoch)
        self.trainer.start_epoch(epoch)

    def check_last_epoch(self, epoch):
        is_last_epoch = False
        cur_time = time.time()
        if self.timeout is not None and cur_time - self._start_time > self.timeout:
            is_last_epoch = True
        is_last_epoch = (epoch == self.num_epochs - 1) or is_last_epoch
        return is_last_epoch

    def _get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        # Not sure why below is necessary... policy is cached via trainer and no clear reason to save env
        # for k, v in self.expl_data_collector.get_snapshot().items():
        #     snapshot['exploration/' + k] = v
        # for k, v in self.eval_data_collector.get_snapshot().items():
        #     snapshot['evaluation/' + k] = v
        for k, v in self.replay_buffer.get_snapshot().items():
            snapshot['replay_buffer/' + k] = v
        return snapshot

    def _log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

        """
        Replay Buffer
        """
        logger.record_dict(
            self.replay_buffer.get_diagnostics(),
            prefix='replay_buffer/'
        )

        """
        Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/')

        """
        Exploration
        """
        logger.record_dict(
            self.expl_data_collector.get_diagnostics(),
            prefix='exploration/'
        )
        expl_paths = self.expl_data_collector.get_epoch_paths()
        if hasattr(self.expl_env, 'get_diagnostics'):
            logger.record_dict(
                self.expl_env.get_diagnostics(expl_paths),
                prefix='exploration/',
            )
        if len(expl_paths) > 0:
            logger.record_dict(
                eval_util.get_generic_path_information(expl_paths),
                prefix="exploration/",
            )

        """
        Evaluation
        """
        logger.record_dict(
            self.eval_data_collector.get_diagnostics(),
            prefix='evaluation/',
        )
        eval_paths = self.eval_data_collector.get_epoch_paths()
        if hasattr(self.eval_env, 'get_diagnostics'):
            logger.record_dict(
                self.eval_env.get_diagnostics(eval_paths),
                prefix='evaluation/',
            )
        logger.record_dict(
            eval_util.get_generic_path_information(eval_paths),
            prefix="evaluation/",
        )

        """
        Misc
        """
        gt.stamp('logging', unique=False)
        logger.record_dict(get_epoch_timings())
        logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass
