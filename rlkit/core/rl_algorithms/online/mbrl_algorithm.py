import gtimer as gt

from rlkit.core.rl_algorithms.rl_algorithm import BaseRLAlgorithm
from rlkit.samplers.data_collector.step_collector import MdpStepCollector
from rlkit.samplers.data_collector.path_collector import MdpPathCollector
from typing import cast

class MBRLAlgorithm(BaseRLAlgorithm):

    def __init__(
            self,
            trainer,                                # Policy trainer; it does nothing for MPC
            model_trainer,                          # Trainer for environment models: dynamics and reward (optional)
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,                          # The replay buffer which stores historical data
            batch_size,                             #
            max_path_length,                        # The max length of a trajectory
            num_epochs,
            num_eval_steps_per_epoch,               # Number of full paths for evaluation in each training epoch
            num_expl_steps_per_train_loop,          # In each epoch and train loop, how many exploration steps to take.
                                                    # num_trains_per_expl_step = num_trains_per_train_loop // num_expl_steps_per_train_loop
            num_trains_per_train_loop,              # How many iterations in one epoch
            model_train_period=None,                  # How often will the model be trained
            num_train_loops_per_epoch=1,            # The loop within the epoch loop
            min_num_steps_before_training=0,        # Take this number of steps using expl_data_collector initially
            rand_init_policy_before_training=False,
            initial_training_steps=0,               # when epoch == 0 and t == 0, train the policy (not the model) with this number of steps
            eval_initial_policy=True,
            max_epochs_since_last_update=5,
            num_model_learning_epochs=None,          # The maximum number of epochs for model learning
            model_max_grad_steps=int(1e7),
            save_snapshot_gap=100,
            use_best_parameters=False,
            save_best_parameters=False,
            timeout=None,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
            save_snapshot_gap=save_snapshot_gap,
            save_best_parameters=save_best_parameters,
            timeout=timeout
        )

        self.model_trainer = model_trainer
        self.use_best_parameters = use_best_parameters
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.model_train_period = model_train_period
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.rand_init_policy_before_training = rand_init_policy_before_training
        self.initial_training_steps = initial_training_steps
        self.num_model_learning_epochs = num_model_learning_epochs
        self.max_epochs_since_last_update = max_epochs_since_last_update
        self.model_max_grad_steps = model_max_grad_steps
        self.eval_data_collector = cast(MdpPathCollector, self.eval_data_collector)
        self.expl_data_collector = cast(MdpStepCollector, self.expl_data_collector)

    def _get_training_diagnostics_dict(self):
        training_diagnostics = super()._get_training_diagnostics_dict()
        training_diagnostics['model_trainer'] = self.model_trainer.get_diagnostics()
        return training_diagnostics

    def _get_snapshot(self):
        snapshot = super()._get_snapshot()
        for k, v in self.model_trainer.get_snapshot().items():
            snapshot['model/' + k] = v
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        return snapshot

    def end_epoch(self, epoch):
        self.model_trainer.end_epoch(epoch)
        self.trainer.end_epoch(epoch)
        return super().end_epoch(epoch)

    def _train(self):
        self.training_mode(False)

        # Start training
        self._start_training()

        # Collect some experience with the random policy initially
        if self.min_num_steps_before_training > 0:
            self.expl_data_collector.collect_new_steps(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
                initial_expl=self.rand_init_policy_before_training,
            )
            init_expl_paths = self.expl_data_collector.get_epoch_paths()
            self.replay_buffer.add_paths(init_expl_paths)
            # self.expl_data_collector.end_epoch(-1)  # _epoch_steps = [] initialized
            gt.stamp('initial exploration', unique=False)

        # Record the initial evaluation
        self.eval_data_collector.collect_new_paths(
            self.max_path_length,
            self.num_eval_steps_per_epoch,
            discard_incomplete_paths=True,
        )
        gt.stamp('evaluation sampling', unique=False)
        self.end_epoch(0)

        num_trains_per_expl_step = self.num_trains_per_train_loop // self.num_expl_steps_per_train_loop
        model_train_period = self.model_train_period

        # Main training loop for online MBRL
        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self.start_epoch(epoch)

            # Another iteration loop within each epoch (default=1)
            for _ in range(self.num_train_loops_per_epoch):
                # How many exploration steps per train loop
                for t in range(self.num_expl_steps_per_train_loop):
                    # Train the environment model periodically
                    self.training_mode(True)
                    if t > 0 and t % model_train_period == 0:
                        self.model_trainer.train_from_buffer(
                            self.replay_buffer,
                            max_grad_steps=self.model_max_grad_steps,
                            max_epochs_since_last_update=self.max_epochs_since_last_update,
                            num_total_epochs=self.num_model_learning_epochs,
                            use_best_parameters=self.use_best_parameters,
                        )
                        gt.stamp('model training', unique=False)

                    # If initial policy training should be done
                    if (epoch == 0 and t == 0) and self.initial_training_steps > 0:
                        for _ in range(self.initial_training_steps):
                            train_data = self.replay_buffer.random_batch(
                                self.batch_size)
                            self.trainer.train(train_data)
                        gt.stamp('initial policy training', unique=False)

                    # Take exploration steps and store in the buffer
                    s, a, r, d, ns, info = self.expl_data_collector.collect_one_step(
                        self.max_path_length,
                        discard_incomplete_paths=False,
                    )
                    gt.stamp('exploration sampling', unique=False)
                    self.replay_buffer.add_sample(s, a, r, d, ns, env_info=info)
                    gt.stamp('data storing', unique=False)

                    # Main policy training loop
                    for _ in range(num_trains_per_expl_step):
                        train_data = self.replay_buffer.random_batch(
                            self.batch_size)
                        self.trainer.train(train_data)
                    gt.stamp('policy training', unique=False)
                    self.training_mode(False)

            # At the end of every epoch, evaluate the current policy
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp('evaluation sampling', unique=False)

            is_last_epoch = self.check_last_epoch(epoch)
            self.end_epoch(epoch + 1)
            if is_last_epoch:
                break
