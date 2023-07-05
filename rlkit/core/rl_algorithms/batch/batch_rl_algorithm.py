import abc
from rlkit.torch import pytorch_util as ptu

import gtimer as gt
from rlkit.core.rl_algorithms.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffers.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector
from rlkit.samplers.data_collector.path_collector import MdpPathCollector
import numpy as np
from typing import cast, Optional
import time


def get_flat_params(model):
    params = []
    for param in model.parameters():
        # import ipdb; ipdb.set_trace()
        params.append(param.data.cpu().numpy().reshape(-1))
    return np.concatenate(params)

def set_flat_params(model, flat_params, trainable_only=True):
    idx = 0
    # import ipdb; ipdb.set_trace()
    for p in model.parameters():
        flat_shape = int(np.prod(list(p.data.shape)))
        flat_params_to_assign = flat_params[idx:idx+flat_shape]
      
        if len(p.data.shape):
            p.data = ptu.tensor(flat_params_to_assign.reshape(*p.data.shape))
        else:
            p.data = ptu.tensor(flat_params_to_assign[0])
        idx += flat_shape  
    return model

class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: MdpPathCollector,
            evaluation_data_collector: MdpPathCollector,
            replay_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            rand_init_policy_before_training=False,
            save_snapshot_gap=100,
            timeout: Optional[float] = None,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
            save_snapshot_gap=save_snapshot_gap,
            timeout=timeout,
        )

        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.rand_init_policy_before_training = rand_init_policy_before_training
        self.eval_data_collector = cast(MdpPathCollector, self.eval_data_collector)
        self.expl_data_collector = cast(MdpPathCollector, self.expl_data_collector)

    def _train(self):
        # Start training
        self._start_training()

        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
                initial_expl=self.rand_init_policy_before_training,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

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

            for _ in range(self.num_train_loops_per_epoch):
                # Sample new paths
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )
                gt.stamp('exploration sampling', unique=False)

                self.replay_buffer.add_paths(new_expl_paths)
                gt.stamp('data storing', unique=False)

                self.training_mode(True)
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size)
                    self.trainer.train(train_data)
                gt.stamp('training', unique=False)
                self.training_mode(False)

            # At every epoch, evaluate the current policy
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp('Evaluation sampling', unique=False)

            is_last_epoch = self.check_last_epoch(epoch)
            self.end_epoch(epoch + 1)
            if is_last_epoch:
                break
