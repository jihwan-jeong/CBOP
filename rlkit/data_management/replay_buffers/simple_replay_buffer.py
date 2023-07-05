from collections import OrderedDict
import warnings

import numpy as np
import torch

from rlkit.data_management.replay_buffers.replay_buffer import ReplayBuffer
import rlkit.torch.pytorch_util as ptu
import rlkit.types
from typing import Optional

def load_replay_buffer_from_snapshot(new_replay, snapshot, force_terminal_false=False):
    for t in range(len(snapshot['replay_buffer/actions'])):
        sample = dict(env_info=dict())
        for k in ['observation', 'action', 'reward',
                  'terminal', 'next_observation', 'env_info']:
            if len(snapshot['replay_buffer/%ss' % k]) == 0:
                continue
            if force_terminal_false and k == 'terminal':
                sample[k] = [False]
            else:
                sample[k] = snapshot['replay_buffer/%ss' % k][t]
        new_replay.add_sample(**sample)


class SimpleReplayBuffer(ReplayBuffer):

    def __init__(
        self,
        max_replay_buffer_size,
        observation_dim,
        action_dim,
        env_info_sizes,
        replace=True,
        rng=None,
        **kwargs,
    ):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size, observation_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((max_replay_buffer_size, observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')
        self._logprobs = np.zeros((max_replay_buffer_size, 1))
        # Additionally paths can be saved (currently used for offline RL only)
        self._paths = None
        # Define self._env_infos[key][i] to be the return value of env_info[key]
        # at time i
        self._env_infos = {}
        for key, size in env_info_sizes.items():
            self._env_infos[key] = np.zeros((max_replay_buffer_size, size))
        self._env_info_keys = env_info_sizes.keys()

        self._replace = replace

        self._top = 0
        self._size = 0

        self.total_entries = 0

        if rng is None:
            self._rng: np.random.Generator = np.random.default_rng()
        else:
            self._rng: np.random.Generator = rng

    def obs_preproc(self, obs):
        return obs

    def obs_postproc(self, obs: np.ndarray) -> np.ndarray:
        return obs

    def add_sample(self, observation, action, reward, next_observation,
                   terminal, env_info, **kwargs):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation

        for key in self._env_info_keys:
            self._env_infos[key][self._top] = env_info[key]
        self._advance()

    def add_batch(self, batch, **kwargs):
        observation = batch['observations'],
        action = batch['actions'],
        reward = batch['rewards'],
        next_observation = batch['next_observations'],
        terminal = batch['terminals'],
        def copy_from_to(buffer_start, batch_start, how_many):
            buffer_slice = slice(buffer_start, buffer_start + how_many)
            batch_slice = slice(batch_start, batch_start + how_many)
            np.copyto(self._observations[buffer_slice], observation[batch_slice])
            np.copyto(self._actions[buffer_slice], action[batch_slice])
            np.copyto(self._rewards[buffer_slice], reward[batch_slice])
            np.copyto(self._next_obs[buffer_slice], next_observation[batch_slice])
            np.copyto(self._terminals[buffer_slice], terminal[batch_slice])

        _batch_start = 0
        buffer_end = self.top() + len(observation)
        if buffer_end > self.max_replay_buffer_size():
            copy_from_to(self.top(), _batch_start, self.max_replay_buffer_size() - self.top())
            _batch_start = self.max_replay_buffer_size() - self.top()
            self._top = 0

        _how_many = len(observation) - _batch_start
        copy_from_to(self.top(), _batch_start, _how_many)
        self._top = (self.top() + _how_many) % self.max_replay_buffer_size()
        self._size = min(self.max_replay_buffer_size(), self._size + len(observation))
        self.total_entries += len(observation)

    def add_sample_with_logprob(self, observation, action, reward, next_observation,
                                terminal, env_info, logprob, **kwargs):
        self._logprobs[self._top] = logprob
        self.add_sample(observation, action, reward, next_observation, terminal, env_info, **kwargs)

    def get_transitions(self):
        return np.concatenate([
            self._observations[:self._size],
            self._actions[:self._size],
            self._rewards[:self._size],
            self._terminals[:self._size],
            self._next_obs[:self._size],
        ], axis=1)

    def get_transitions_dict(self):
        return dict(
            observations=self._observations[:self._size],
            actions=self._actions[:self._size],
            rewards=self._rewards[:self._size],
            terminals=self._terminals[:self._size],
            next_observations=self._next_obs[:self._size],
        )

    def get_logprobs(self):
        return self._logprobs[:self._size].copy()

    def relabel_rewards(self, rewards):
        self._rewards[:len(rewards)] = rewards

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1
        self.total_entries += 1

    def random_batch(self, batch_size, min_pct=0, max_pct=1, include_logprobs=False, return_indices=False):
        indices = self._rng.choice(self._size, size=batch_size, replace=self._replace or self._size < batch_size)
        if not self._replace and self._size < batch_size:
            warnings.warn(
                'Replace was set to false, but is temporarily set to true because batch size is larger than current size of replay.')
        batch = dict(
            observations=self.obs_postproc(self._observations[indices]),
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self.obs_postproc(self._next_obs[indices]),
        )
        if include_logprobs:
            batch['logprobs'] = self._logprobs[indices]
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        if return_indices:
            return batch, indices
        else:
            return batch

    def construct_nstep_input_target_dataset(
            self,
            nsteps: int,
            vf: Optional[torch.nn.Module] = None,
            qf: Optional[torch.nn.Module] = None,
            use_bootstrap: bool = True,
            discount: float = 0.99,
            obs_preproc: Optional[rlkit.types.ObsProcessFnType] = None,
            batch_size: int = 512,
            sampling_method: str = 'mean',
    ):
        """Constructs the dataset consisting of {(s, a), y} input-target pairs where y is the n-step return target.
        The value function (either state or state-action) should be provided so that at the value of the nth state
        can be included in the targets.

        Args:
            nsteps          (int): Number of steps to look into future
            vf              (torch.nn.Module): State value function
            qf              (torch.nn.Module): Action value function
            use_bootstrap   (bool): Whether a value function is used to bootstrap the return target at the n-th state.
            discount        (float) The discount factor
            obs_preproc     (ProxyEnv.obs_preproc) Pre(post)-processing of obs data
            batch_size      (int) The batch size to be used when passing through a target network
        """
        assert (vf is None and qf is None and not use_bootstrap) or \
               (vf is not None and qf is None and use_bootstrap) or \
               (vf is None and qf is not None and use_bootstrap), \
               "None or either *one* of state or state-action value functions can be provided"
        assert (not use_bootstrap and discount == 1) or use_bootstrap, "When not bootstrapping, do not discount!"

        # Indices of states which become the n-th states (and we look backwards from there)
        indices = np.arange(self._size)
        term = np.zeros(indices.shape, dtype=bool)
        num_batches = int(np.ceil(self._size / batch_size))

        # Initialize the target values
        targets = np.zeros_like(indices, dtype=float)

        # Retrieve the nth transition tuples
        nth_obs = self._observations[indices]        # Shape: (N, d_s)
        if obs_preproc:
            nth_obs_ = obs_preproc(nth_obs)
        else:
            nth_obs_ = nth_obs.copy()
        nth_act = self._actions[indices]                                # Shape: (N, d_a)

        # Bootstrap the return target using the (action) value function (use mini-batches to prevent memory issues)
        if use_bootstrap:
            value_func = vf if vf else qf
            model_in = nth_obs_ if vf else np.concatenate([nth_obs_, nth_act], axis=-1)

            with torch.no_grad():
                val_n = []
                for b in range(num_batches):
                    input_b = model_in[b * batch_size: (b+1) * batch_size, :]
                    input_b = ptu.from_numpy(input_b)
                    val_n_b = value_func.sample(input_b, sampling_method=sampling_method)
                    val_n.append(val_n_b)
                val_n = torch.cat(val_n, dim=0).squeeze()
                val_n = ptu.get_numpy(val_n)

            # Add the value to the target
            targets += val_n

        # Accumulate rewards r_{n-1}, r_{n-2}, ..., r_1, r_0
        cur_obs = nth_obs.copy()  # s_{t+n}
        input_obs, input_act = None, None
        for j in range(1, nsteps + 1):
            # Retrieve transition tuples of index (i-j)
            ind_i__j = np.roll(indices, j)
            obs_i__j = self._observations[ind_i__j]         # s_{t+n-j-1}
            next_obs_i__j = self._next_obs[ind_i__j]        # s_{t+n-j}
            act_i__j = self._actions[ind_i__j]              # a_{t+n-j-1}
            rew_i__j = self._rewards[ind_i__j]              # r_{t+n-j-1}
            terminals_i__j = self._terminals[ind_i__j]      # d_{t+n-j-1}

            # Determine whether there was an end of episode
            """
            Compare s_{t+n-j} from ``_observations`` and s_{t+n-j} from ``_next_obs`` 
            That is, we compare ``_observation[t+n-j]`` vs. ``_next_obs[t+n-j-1]``. If the episode was continued from
            s_{t+n-j-1} through to s_{t+n-j}, these two values should match. Otherwise, we conclude that a new episode 
            started from the index [t+n-j]. In such cases, we discard those episodes in the dataset by masking them in
            such event.
            
            Below, ``check`` list contains whether 's_{t+n-j} == s_{t+n-j}' holds for a mini-batch. 
            The boolean value of an element in the `term` array will become True for the first time when s_{t+n-j+1} 
            is the beginning of a new episode (determined by `check` as well as `terminals_i__j`). 
            Once the value became True, it will remain as is. 
            """
            check, res = [], []
            for b in range(num_batches):
                check.append((cur_obs[b * batch_size: (b+1) * batch_size] !=
                              next_obs_i__j[b * batch_size: (b+1) * batch_size]).any(1))
                res.append(np.where(term[b * batch_size: (b+1) * batch_size] |
                                    check[b] |
                                    terminals_i__j[b * batch_size: (b+1) * batch_size].squeeze().astype(bool),
                                    np.ones(check[b].size, dtype=bool),
                                    term[b * batch_size: (b+1) * batch_size]))
            term = np.concatenate(res)

            # Update the targets if an episode continues; otherwise, keep the previous target values (will be discarded)
            target_lst = []
            for b in range(num_batches):
                target_lst.append(np.where(term[b * batch_size: (b+1) * batch_size],
                                            targets[b * batch_size: (b+1) * batch_size],
                                            rew_i__j[b * batch_size: (b+1) * batch_size].squeeze() +
                                            discount * targets[b * batch_size: (b+1) * batch_size]
                                            )
                                    )
            targets = np.concatenate(target_lst)
            cur_obs = obs_i__j
            input_obs, input_act = obs_i__j.copy(), act_i__j.copy()

        # Only include transitions that are longer than ``n``
        idx_to_include = np.where(~term)[0]

        # Now, construct the dataset
        data = dict(
            obs=input_obs[idx_to_include],
            actions=input_act[idx_to_include],
            targets=targets[idx_to_include],
        )
        return data

    def get_snapshot(self):
        return dict(
            observations=self._observations[:self._size],
            actions=self._actions[:self._size],
            rewards=self._rewards[:self._size],
            terminals=self._terminals[:self._size],
            next_observations=self._next_obs[:self._size],
            env_infos=self._env_infos,
        )

    def load_snapshot(self, snapshot):
        prev_info = snapshot['env_info']
        for t in range(snapshot['observations'].shape[0]):
            env_info = {key: prev_info[key][t] for key in prev_info}
            self.add_sample(
                observation=snapshot['observations'][t],
                action=snapshot['actions'][t],
                reward=snapshot['rewards'][t],
                next_observation=snapshot['next_observations'][t],
                terminal=snapshot['terminals'][t],
                env_info=env_info,
            )

    def rebuild_env_info_dict(self, idx):
        return {
            key: self._env_infos[key][idx]
            for key in self._env_info_keys
        }
    
    def batch_env_info_dict(self, indices):
        return {
            key: self._env_infos[key][indices]
            for key in self._env_info_keys
        }
    
    def top(self):
        return self._top

    def num_steps_can_sample(self):
        return self._size

    def max_replay_buffer_size(self):
        return self._max_replay_buffer_size

    def obs_dim(self):
        return self._observation_dim

    def action_dim(self):
        return self._action_dim

    def get_diagnostics(self):
        return OrderedDict([
            ('size', self._size)
        ])

    def __len__(self):
        return self._size
