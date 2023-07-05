from collections import deque, OrderedDict

import numpy as np

from rlkit.util.eval_util import create_stats_ordered_dict
from rlkit.util import eval_util
from rlkit.data_management.util.path_builder import PathBuilder
from rlkit.samplers.data_collector.base import StepCollector


class MdpStepCollector(StepCollector):
    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs

        self._num_steps_total = 0
        self._num_paths_total = 0
        self._obs = None  # cache variable

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._obs = None

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        return dict(
            env=self._env,
            policy=self._policy,
        )

    def collect_new_steps(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
            initial_expl=False,
    ):
        for _ in range(num_steps):
            self.collect_one_step(max_path_length, discard_incomplete_paths, initial_expl)

    def collect_one_step(
            self,
            max_path_length,
            discard_incomplete_paths,
            initial_expl=False,
    ):
        if self._obs is None:
            self._start_new_rollout()

        # Select uniform random actions within bounds initially
        if initial_expl:
            action = self._env.action_space.sample()
            agent_info = dict()
        # Otherwise, use the policy to select an action
        else:
            action, agent_info = self._policy.get_action(self._obs)
        next_ob, reward, terminal, env_info = (
            self._env.step(action)
        )
        env_transition = (
            self._obs, action, reward, terminal, next_ob, env_info
        )

        if self._render:
            self._env.render(**self._render_kwargs)
        terminal = np.array([terminal])
        reward = np.array([reward])
        # store path obs
        self._current_path_builder.add_all(
            observations=self._obs,
            actions=action,
            rewards=reward,
            next_observations=next_ob,
            terminals=terminal,
            agent_infos=agent_info,
            env_infos=env_info,
        )
        if terminal or (max_path_length is not None and
                        len(self._current_path_builder) >= max_path_length):
            self._handle_rollout_ending(max_path_length,
                                        discard_incomplete_paths)
            self._start_new_rollout()
        else:
            self._obs = next_ob
        return env_transition

    def _start_new_rollout(self):
        self._current_path_builder = PathBuilder()
        self._obs = self._env.reset()

    def _handle_rollout_ending(
            self,
            max_path_length,
            discard_incomplete_paths
    ):
        if len(self._current_path_builder) > 0:
            path = self._current_path_builder.get_all_stacked()
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                return
            self._epoch_paths.append(path)
            self._num_paths_total += 1
            self._num_steps_total += path_len
            self._policy.reset()
