from collections import OrderedDict
import gtimer as gt
import numpy as np

from rlkit.core.rl_algorithms.torch_rl_algorithm import TorchTrainer
from rlkit.util.eval_util import create_stats_ordered_dict
from rlkit.samplers.util.model_rollout_functions import rollout_model_and_get_paths
from rlkit.torch.models.dynamics_models.model import DynamicsModel
from rlkit.envs.model_env import ModelEnv
import rlkit.util.math
from rlkit.trainers.mbpo.util import maybe_replace_model_buffer, get_policy_training_batch
from rlkit.core.logging.logging import logger

class MBPOTrainer(TorchTrainer):

    """
    Model-based Policy Optimization (Janner et al. 2019).
    Policy optimization using synthetic model-based rollouts.
    Supports various types of policy optimization procedures using a model.

    This also supports offline model-based policy optimization methods
    such as MOPO (Yu et al., 2021) and MOReL (Kidambi et al., 2021).
    """

    def __init__(
            self,
            policy_trainer,                     # Associated policy trainer to learn from generated data
            dynamics_model: DynamicsModel,      # Note that MBPOTrainer is not responsible for training this
            model_env: ModelEnv,
            rollout_schedule: list,
            batch_size: int,                    # Batch size for policy training
            model_train_period: int = 250,
            rollout_batch_size: int = int(1e5),
            max_path_length: int = 1000,                # Epoch length of the task
            num_policy_updates_per_step=40,             # Number of policy updates per env timestep (should be > 1)
            num_max_policy_updates_per_step=5,
            policy_updates_every_steps=1,      # The period of how often the policy is trained w.r.t. environment step
            num_epochs_to_retain_model_buffer=1,
            real_data_ratio=0.05,                       # Percentage of real data used for policy training
            log_policy_training_period=100,             # How often the policy training progress is to be logged
            start_model_rollouts_from_init_dist: bool = False,     # Whether start model rollouts from initial state distribution
            **kwargs
    ):
        super().__init__()

        self.policy_trainer = policy_trainer
        self.policy = policy_trainer.policy
        self.dynamics_model = dynamics_model
        self.model_env = model_env
        self.replay_buffer = None
        self.model_data_buffer = None

        self.batch_size = batch_size
        self.policy_updates_every_steps = policy_updates_every_steps
        self.rollout_schedule = rollout_schedule
        self.model_train_period = model_train_period
        self.num_policy_updates_per_step = num_policy_updates_per_step

        # Note: below differs from what MOPO does; set this way due to mbpo_pytorch repo
        self.num_max_policy_updates_per_step = num_max_policy_updates_per_step

        self.max_path_length = max_path_length
        self.rollout_batch_size = int(rollout_batch_size)
        self.rollouts_per_epoch = self.rollout_batch_size * self.max_path_length / self.model_train_period
        self.num_epochs_to_retain_model_buffer = num_epochs_to_retain_model_buffer
        self.real_data_ratio = real_data_ratio

        self._num_total_policy_updates_made = 0
        self._need_to_update_eval_statistics = True
        self._start_model_rollouts_from_init_dist = start_model_rollouts_from_init_dist
        self.log_policy_training_period = log_policy_training_period
        self._curr_epoch = 0
        self._num_policy_updates_made = 0
        self.eval_statistics = OrderedDict()

        self.name = 'mbpo'
        self._network_dict = {}

    def set_replay_buffer(self, replay_buffer):
        self.replay_buffer = replay_buffer

    def train_from_torch(self, batch):
        # We only use the original batch to get the batch size for policy training
        assert self.replay_buffer is not None, "Make sure to associate the replay buffer with MBPOTrainer!"

        """
        Generate synthetic data using dynamics model
        """
        rollout_horizon = self.rollout_schedule if isinstance(self.rollout_schedule, int) else None
        if rollout_horizon is None:
            rollout_horizon = int(rlkit.util.math.truncated_linear(
                *(self.rollout_schedule + [self._curr_epoch])
            ))
        model_steps_per_epoch = int(rollout_horizon * self.rollouts_per_epoch)
        new_buffer_size = self.num_epochs_to_retain_model_buffer * model_steps_per_epoch
        self.model_data_buffer = maybe_replace_model_buffer(
            self.model_data_buffer,
            self.replay_buffer.env,
            new_buffer_size,
        )

        # Model is only occasionally updated; once it happens, generate all samples to be used until the next iteration
        # For MOPO and MOReL, the model is trained only once at the beginning; still, model rollouts should be
        # occasionally generated as the policy is keep changing.
        if self._num_train_total_steps > 0 and \
                self._num_train_total_steps % self.model_train_period == 0:
            rollout_diagnostics = OrderedDict()
            paths = rollout_model_and_get_paths(
                model_env=self.model_env,
                replay_buffer=self.replay_buffer,
                initial_obs=self.initial_obs,
                total_size=self.rollout_batch_size,
                agent=self.policy,
                deterministic=False,
                rollout_horizon=rollout_horizon,
                diagnostics=rollout_diagnostics,
            )
            batch = {}
            for i, key in enumerate(paths[0].keys()):
                if 'info' not in key:
                    batch[key] = np.concatenate([p[key] for p in paths], axis=0)
            self.model_data_buffer.add_batch(batch)
            # self.model_data_buffer.add_paths(paths)
            # self.model_data_buffer._paths = paths
            rollout_diagnostics.update(
                create_stats_ordered_dict(
                    'Path Length',
                    np.array([p['observations'].shape[0] for p in paths])
            ))
            logger.record_dict(
                rollout_diagnostics,
                prefix='Model rollout/'
            )
            gt.stamp('Model rollouts', unique=False)

        # Below conditions follow mbpo-pytorch and has shown good results on MuJoCo tasks
        # if self._num_train_total_steps % self.policy_updates_every_steps == 0:
        #     if self._num_total_policy_updates_made <= self.num_max_policy_updates_per_step * self._num_train_total_steps:
        # MOPO instead is based on softlearning repo and uses the following conditions
        if self._num_timesteps_this_epoch % self.policy_updates_every_steps == 0:
            if self._num_train_steps_this_epoch <= self.num_max_policy_updates_per_step * self._num_timesteps_this_epoch:
                for i in range(self.num_policy_updates_per_step):
                    # If model buffer is not yet filled, skip the training (when ``real_data_ratio==0``)
                    if self.real_data_ratio == 0.0 and len(self.model_data_buffer) == 0:
                        break
                    train_data = get_policy_training_batch(
                        self.batch_size,
                        self.real_data_ratio,
                        self.replay_buffer,
                        self.model_data_buffer,
                    )
                    self.policy_trainer.train(train_data)
                    self._num_total_policy_updates_made += 1
                    self._num_train_steps_this_epoch += 1
                    if self._num_total_policy_updates_made % self.log_policy_training_period == 0:
                        self.policy_trainer._log_stats(prefix='Policy Trainer/')
                        self.policy_trainer.end_epoch(-1)

        """
        Save some statistics for evaluation
        """
        if self._need_to_update_eval_statistics and self._num_train_total_steps % self.model_train_period == 0:
            self._need_to_update_eval_statistics = False

            self.eval_statistics['MBPO Rollout Length'] = rollout_horizon
            self.eval_statistics['Model Buffer Size'] = len(self.model_data_buffer)

    def get_diagnostics(self):
        self.eval_statistics.update(self.policy_trainer.eval_statistics)
        return self.eval_statistics

    def start_epoch(self, epoch):
        super().start_epoch(epoch)

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True
        self._curr_epoch = epoch
        self.policy_trainer.end_epoch(epoch)

    @property
    def networks(self):
        networks = list(self._network_dict.values())
        networks += self.policy_trainer.networks
        return networks

    @property
    def initial_obs(self):
        if self._start_model_rollouts_from_init_dist:
            assert hasattr(self.replay_buffer, '_paths')
            paths = self.replay_buffer._paths
            init_states_buffer = np.concatenate([p['observations'][0][None] for p in paths], axis=0)
            buffer_rand_idx = np.random.choice(
                len(init_states_buffer),
                size=self.rollout_batch_size,
                replace=True)
            initial_obs = init_states_buffer[buffer_rand_idx]
            return initial_obs
        else:
            return None

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot.update(self.policy_trainer.get_snapshot())
        return snapshot

    def configure_logging(self, **kwargs):
        self.policy_trainer.configure_logging(**kwargs)
