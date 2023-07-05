from collections import deque, OrderedDict
from typing import Callable

from rlkit.util.eval_util import create_stats_ordered_dict
from rlkit.samplers.util.rollout_functions import rollout, function_rollout
from rlkit.samplers.data_collector.base import PathCollector
import numpy as np


class MdpPathCollector(PathCollector):
    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=None,
            render=False,
            sparse_reward=False,
            render_kwargs=None,
            rollout_fn=rollout,
            save_env_in_snapshot=True,
            is_offline=False,
            post_epoch_funcs=[],
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs
        self._rollout_fn = rollout_fn

        self._num_steps_total = 0
        self._num_paths_total = 0
        self._sparse_reward = sparse_reward
        self.is_offline= is_offline

        self.diagnostics = OrderedDict()
        self.post_epoch_funcs = post_epoch_funcs
        self._save_env_in_snapshot = save_env_in_snapshot

    def update_policy(self, new_policy):
        self._policy = new_policy
    
    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
            policy_fn=None,
            initial_expl=False,
            last_epoch=False
    ):
        # Random initial exploration if specified
        if initial_expl:
            from rlkit.policies.base.simple import RandomPolicy
            policy = RandomPolicy(self._env.action_space)
        else:
            policy = self._policy

        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            path = self._rollout_fn(
                self._env,
                policy,
                max_path_length=max_path_length_this_loop,
                render=self._render,
                render_kwargs=self._render_kwargs,
            )
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len

            ## Used to sparsify reward
            if self._sparse_reward:
                random_noise = np.random.normal(size=path['rewards'].shape)
                path['rewards'] = path['rewards'] + 1.0*random_noise 

            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)

        """If `post_epoch_func` is defined, evaluate the evaluation trajectories with the function"""
        if last_epoch and len(self.post_epoch_funcs) > 0:
            for func in self.post_epoch_funcs:
                self.diagnostics.update(
                    func(self._epoch_paths)
                )
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self.diagnostics = OrderedDict()

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
        if self.is_offline:
            returns = [sum(path["rewards"]) for path in self._epoch_paths]
            try:
                normalized_eval_returns = list(map(lambda x: self._env.get_normalized_score(x) * 100, returns))
                stats.update(
                    create_stats_ordered_dict(
                        "normalized score",
                        normalized_eval_returns,
                        always_show_all_stats=True,
                    )
                )
            except:
                pass
        stats.update(self.diagnostics)
        return stats

    def get_snapshot(self):
        return dict(
            env=self._env,
            policy=self._policy,
        )

class CustomMDPPathCollector(PathCollector):
    def __init__(
        self,
        env,
        max_num_epoch_paths_saved=None,
        render=False,
        render_kwargs=None,
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs

        self._num_steps_total = 0
        self._num_paths_total = 0

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
            policy_fn=None,
        ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            path = function_rollout(
                self._env,
                policy_fn,
                max_path_length=max_path_length_this_loop,
            )
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths
    
    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

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
        )
