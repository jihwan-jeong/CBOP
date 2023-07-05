import abc

import gtimer as gt

from rlkit.core.rl_algorithms.batch.batch_rl_algorithm import BatchRLAlgorithm


class MBBatchRLAlgorithm(BatchRLAlgorithm, metaclass=abc.ABCMeta):

    def __init__(
            self,
            model_trainer,
            model_max_grad_steps=int(1e3),
            num_model_learning_epochs=None,
            max_epochs_since_last_update=5,
            use_best_parameters=False,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.model_trainer = model_trainer
        self.model_max_grad_steps = model_max_grad_steps
        self.num_model_learning_epochs = num_model_learning_epochs
        self.max_epochs_since_last_update = max_epochs_since_last_update
        self.use_best_parameters = use_best_parameters

    def _train(self):
        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
                initial_expl=self.rand_init_policy_before_training,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

            gt.stamp('initial exploration', unique=False)

        self._start_training()

        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self.start_epoch(epoch)

            # Model training happens once in each epoch
            self.training_mode(True)
            if self.replay_buffer.num_steps_can_sample() > 0:
                self.model_trainer.train_from_buffer(
                    self.replay_buffer,
                    max_grad_steps=self.model_max_grad_steps,
                    max_epochs_since_last_update=self.max_epochs_since_last_update,
                    use_best_parameters=self.use_best_parameters,
                    num_total_epochs=self.num_model_learning_epochs,
                )
            gt.stamp('model training', unique=False)

            for _ in range(self.num_train_loops_per_epoch):
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

            # At the end of every epoch, evaluate the current policy
            is_last_epoch = self.check_last_epoch(epoch)

            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
                last_epoch=is_last_epoch,
            )
            gt.stamp('evaluation sampling', unique=False)

            self.end_epoch(epoch + 1)
            if is_last_epoch:
                break

    def _get_training_diagnostics_dict(self):
        training_diagnostics = super()._get_training_diagnostics_dict()
        training_diagnostics['model_trainer'] = self.model_trainer.get_diagnostics()
        return training_diagnostics

    def _get_snapshot(self):
        snapshot = super()._get_snapshot()
        for k, v in self.model_trainer.get_snapshot().items():
            snapshot['model/' + k] = v
        return snapshot

    def _end_epochs(self, epoch):
        super().end_epoch(epoch)
        self.model_trainer.end_epoch(epoch)
