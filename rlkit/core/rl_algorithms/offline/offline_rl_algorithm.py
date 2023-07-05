import gtimer as gt

import abc
import numpy as np
import os.path as osp
from rlkit.core.logging.logging import logger
from rlkit.core.rl_algorithms.rl_algorithm import get_epoch_timings
from rlkit.util import eval_util
from typing import Optional
import time

class OfflineRLAlgorithm(object, metaclass=abc.ABCMeta):

    def __init__(
            self,
            trainer,
            evaluation_policy,
            evaluation_env,
            evaluation_data_collector,
            replay_buffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_trains_per_train_loop,
            train_at_start=False,                    # Flag for debugging
            num_train_loops_per_epoch=1,
            save_snapshot_gap=10,
            save_best_parameters=False,
            timeout=None,
            normalize_states=False,
            **kwargs,
    ):
        self.trainer = trainer
        self.eval_policy = evaluation_policy
        self.eval_env = evaluation_env
        self.eval_data_collector = evaluation_data_collector
        self.replay_buffer = replay_buffer

        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.train_at_start = train_at_start
        self.save_best_parameters = save_best_parameters
        logger.set_snapshot_gap(save_snapshot_gap)

        self.normalize_states = normalize_states

        self._start_epoch = 0
        self.timeout = timeout
        self._start_time = None
        self._best_avg_return = -9e10
        self.post_epoch_funcs = []

    def _train(self):
        output_csv = logger.get_tabular_output()

        if self.normalize_states:
            self.trainer.normalize_states(self.replay_buffer)
            
        # Pretrain the behavior-cloning policy of the offline dataset
        if self.train_at_start:
            logger.log("Direct pretraining within OfflineRLAlgorithm class is deprecated...\n"
                "In order to use pretrained models, run the algorithm `pretrain` with appropriate configs."
            )
            raise DeprecationWarning("Direct pretraining within OfflineMBRLAlgorithm class is deprecated...\n"
                "In order to use pretrained models, run the algorithm `pretrain` with appropriate configs.")

        # Return the output csv back to the main one
        logger.set_tabular_output(output_csv)

        # Record the initial evaluation results
        self.eval_data_collector.collect_new_paths(
            self.max_path_length,
            self.num_eval_steps_per_epoch,
            discard_incomplete_paths=True,
        )
        gt.stamp('evaluation sampling', unique=False)
        self.end_epoch(0)

        for epoch in gt.timed_for(
            range(self._start_epoch, self.num_epochs),
            save_itrs=True,
        ):
            self.start_epoch(epoch)

            self.training_mode(True)
            for _ in range(self.num_train_loops_per_epoch):
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(self.batch_size)
                    self.trainer.train(train_data)
            self.training_mode(False)
            gt.stamp('training')

            # In every epoch, evaluate the current policy
            is_last_epoch = self.check_last_epoch(epoch)

            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
                last_epoch=is_last_epoch,
            )
            gt.stamp('Evaluation sampling', unique=False)

            self.end_epoch(epoch + 1)
            if is_last_epoch:
                break

    def train(self, start_epoch=0):
        self._start_epoch = start_epoch
        self._train()

    def _start_training(self):
        self._start_time = time.time()

        # Below handles MOReL
        if hasattr(self.trainer, 'model_env'):
            from rlkit.envs.model_env import ModelEnv
            model_env: ModelEnv = self.trainer.model_env
            sampling_mode, sampling_cfg = model_env.sampling_mode, model_env.sampling_cfg
            if sampling_mode == "disagreement":
                assert 'type' in sampling_cfg.keys()
                sampling_type = sampling_cfg['type']
                if sampling_type == "mean":
                    assert 'pessimism_coeff' in sampling_cfg and 'truncate_reward' in sampling_cfg, \
                        "Pessimism coefficient and truncate reward should be specified!"
                    pessimism_coeff, truncate_reward = sampling_cfg.pessimism_coeff, sampling_cfg.truncate_reward

                    # Compute the maximum model disagreement first
                    obs, act = self.replay_buffer._observations.copy(), self.replay_buffer._actions.copy()
                    model_len = model_env.dynamics_model.model_len
                    num_samples = obs.shape[0]
                    obs, act = tuple(
                        map(lambda x: np.tile(x, (model_len, 1)), [obs, act])
                    )

                    model_env.reset(obs, return_as_np=True)
                    next_obs, *_ = model_env.sample(act, sample=False)
                    next_obs = next_obs.reshape(model_len, num_samples, -1)
                    max_err = np.zeros((num_samples, 1))
                    batch_size = 1024
                    num_batches = int(np.ceil(num_samples / batch_size))
                    for i in range(next_obs.shape[0]):
                        pred_1 = next_obs[i, :]
                        for j in range(i+1, next_obs.shape[0]):
                            pred_2 = next_obs[j, :]
                            error = np.linalg.norm((pred_1 - pred_2), axis=-1, keepdims=True)
                            for b in range(num_batches):
                                max_err[b * batch_size: (b+1) * batch_size] = np.maximum(
                                    max_err[b * batch_size: (b+1) * batch_size], error[b * batch_size: (b+1) * batch_size]
                                )
                    truncate_thresh = (1 / pessimism_coeff) * np.max(max_err)
                    logger.log(
                        f"Maximum error before truncation (i.e. unknown region threshold) = {truncate_thresh}"
                    )
                    sampling_cfg.update({'truncate_thresh': float(truncate_thresh)})

    def start_epoch(self, epoch):
        self.eval_data_collector.start_epoch(epoch)
        self.trainer.start_epoch(epoch)

    def check_last_epoch(self, epoch):
        is_last_epoch = False
        cur_time = time.time()
        if self.timeout is not None and cur_time - self._start_time > self.timeout:
            is_last_epoch = True
        is_last_epoch = (epoch == self.num_epochs - 1) or is_last_epoch
        return is_last_epoch

    def end_epoch(self, epoch):
        """Save snapshots"""
        snapshot = self.get_snapshot()
        orig_snapshot_mode = logger.get_snapshot_mode()

        # Firstly, save the best parameters
        if self.save_best_parameters:
            eval_paths = self.eval_data_collector.get_epoch_paths()
            if len(eval_paths) > 0:
                returns = [sum(path["rewards"]) for path in eval_paths]
                avg_return = np.mean(returns)
                if avg_return > self._best_avg_return:
                    self._best_avg_return = avg_return
                    logger.set_snapshot_mode('last' if orig_snapshot_mode else None)
                    logger.save_itr_params(epoch, snapshot, prefix='best_offline_algo')
                    logger.set_snapshot_mode(orig_snapshot_mode)

        # Also, periodically save the parameters
        logger.set_snapshot_mode(orig_snapshot_mode if orig_snapshot_mode else None)
        logger.save_itr_params(epoch, snapshot, prefix='offline_algo')
        gt.stamp('saving', unique=False)

        self._log_stats(epoch)

        self.eval_data_collector.end_epoch(epoch)
        self.trainer.end_epoch(epoch)

        if hasattr(self.eval_policy, 'end_epoch'):
            self.eval_policy.end_epoch(epoch)

    def get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        # for k, v in self.eval_data_collector.get_snapshot().items():      # TODO: Commented out for now since 
        #     snapshot['evaluation/' + k] = v                               # torch.save emits an error when trying to save environment
        for k, v in self.replay_buffer.get_snapshot().items():
            snapshot['replay_buffer/' + k] = v
        return snapshot

    def _get_trainer_diagnostics(self):
        return self.trainer.get_diagnostics()

    def _get_training_diagnostics_dict(self):
        # No training occurs when the trainer is MPCTrainer
        if not hasattr(self.trainer, "no_training"):
            return {'policy_trainer': self._get_trainer_diagnostics()}
        else:
            return dict()

    def _log_stats(self, epoch: Optional[int] = None):
        """
        Logs stat info. During training, this should be called after each epoch to record the progress within epochs.
        When evaluating, epoch is set to None so that only the necessary stats (path evaluation) can be recorded.
        """

        """
        Replay Buffer (do we need any information from the buffer?)
        """
        # Skip logging for initial evaluation (just check the log file)
        # if epoch == 0:
        #     return

        """
        Trainer
        """
        if epoch is not None:
            logger.log("Epoch {} finished".format(epoch), with_timestamp=True)
            logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/')

        """
        Evaluation
        """
        if self.num_eval_steps_per_epoch > 0:
            logger.record_dict(
                self.eval_data_collector.get_diagnostics(),
                prefix='evaluation/',
            )
            eval_paths = self.eval_data_collector.get_epoch_paths()
            if hasattr(self.eval_env, 'get_diagnostics'):               # Use-case?
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
        if epoch:
            logger.record_dict(get_epoch_timings())
            logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
        gt.stamp('logging', unique=False)

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass
