import gym
import gym.wrappers
import numpy as np
import torch

from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.util import eval_util
from rlkit.core.logging.logging import logger


def make_env(env_name, terminates=True, **kwargs):
    env = None
    env_info = dict(
        mujoco=False,
    )

    """
    Episodic RL
    """
    env_name_split = env_name.split('-')
    if env_name_split[0].lower() in ('halfcheetah', 'hopper', 'walker', 'invertedpendulum'):
        env = gym.make(env_name)
        env_info['mujoco'] = True
        if not isinstance(env.action_space, gym.spaces.Discrete):
            env = NormalizedBoxEnv(env)

    if env is None:
        raise NameError('env_name not recognized')

    return env, env_info

"""
# print some statistics
returns = np.array([np.sum(p['rewards']) for p in raw_paths])
num_samples = np.sum([p['rewards'].shape[0] for p in raw_paths])
print("Number of samples collected = %i" % num_samples)
print("Collected trajectory return mean, std, min, max = %.2f , %.2f , %.2f, %.2f" % \
       (np.mean(returns), np.std(returns), np.min(returns), np.max(returns)) )
"""

def offline_data_to_paths(dataset):
    """Converts a d4rl dataset to paths (list of dictionaries)

    This is adapted from mjrl repo.

    Args:
        dataset (dict): The dataset in d4rl format

    Returns:
        (list of dict) paths objects in a list
    """
    assert 'timeouts' in dataset.keys()
    num_samples = dataset['observations'].shape[0]

    # If timeout occurred or terminal state reached at index t, record t+1 in ``timeouts``
    timeouts = [t+1 for t in range(num_samples) if (dataset['timeouts'][t] or dataset['terminals'][t])]

    # Add the last index of data to ``timeouts``
    if timeouts[-1] != dataset['observations'].shape[0]: timeouts.append(dataset['observations'].shape[0])
    timeouts.insert(0, 0)
    paths = []
    for idx in range(len(timeouts) - 1):
        path = dict()
        for key in dataset.keys():
            if 'metadata' not in key:
                val = dataset[key][timeouts[idx]: timeouts[idx+1]]
                if key in ['rewards', 'terminals', 'timeouts']:
                    val = val.reshape(-1, 1)
                path[key] = val
        paths.append(path)
    return paths


def get_offline_data(env_name, env, replay_buffer, max_size, rng, **kwargs):
    env_name_split = env_name.split('-')
    kwargs.update(dict(
        antmaze=True if env_name_split[0].lower() == 'antmaze' else False
    ))
    if env_name_split[0].lower() in ['halfcheetah', 'hopper', 'walker2d', 'antmaze', 'pen', 'hammer', 'door', 'relocate']:
        import d4rl
        if hasattr(env.env, 'wrapped_env'):
            assert isinstance(env.env.wrapped_env, d4rl.offline_env.OfflineEnv)
        elif hasattr(env.env, 'env'):
            assert isinstance(env.env.env, d4rl.offline_env.OfflineEnv)
        else:
            raise TypeError(f"Unsupported env {env.env}")
        paths = load_data_to_buffer(d4rl.qlearning_dataset(env), replay_buffer, max_size, rng, **kwargs)
    else:
        paths = load_data_to_buffer(env.get_dataset(), replay_buffer, max_size, rng, **kwargs)
    return paths


def load_data_to_buffer(dataset, replay_buffer, max_size, rng, **kwargs):
    # TODO: define a method for sampling from offline datasets
    # paths = offline_data_to_paths(dataset)
    # if max_size is not None:
    #     paths, dataset = sample_paths_within_max_size(dataset, paths, max_size, rng)
    paths = None


    observations, actions, rewards, terminals, next_observations = tuple(
        map(lambda x: dataset.get(x), ['observations', 'actions', 'rewards', 'terminals', 'next_observations'])
    )
    if kwargs.get('antmaze', False) and kwargs.get('reward_normalize', False):
        # Center reward for Ant-Maze
        # rewards = (np.expand_dims(rewards, 1) - 0.5) * 4.0
        rewards = np.expand_dims(np.squeeze(rewards), 1) - 1
    else:
        rewards = np.expand_dims(np.squeeze(rewards), 1)
    terminals = np.expand_dims(np.squeeze(terminals), 1)
    size = len(observations)

    logger.log("\nRewards stats before preprocessing")
    logger.log('mean: {:.4f}'.format(rewards.mean()))
    logger.log('std: {:.4f}'.format(rewards.std()))
    logger.log('max: {:.4f}'.format(rewards.max()))
    logger.log('min: {:.4f}'.format(rewards.min()))

    if kwargs.get('reward_normalize', False) and not kwargs.get('antmaze', False):
        # Centering
        rewards -= rewards.mean()
        # std normalization
        rewards_mean = rewards.mean()
        rewards = (rewards - rewards_mean) / rewards.std() + rewards_mean
        logger.log("\nRewards stats before preprocessing")
        logger.log(f'mean: {rewards.mean():.4f}')
        logger.log(f'std: {rewards.std():.4f}')
        logger.log(f'max: {rewards.max():.4f}')
        logger.log(f'min: {rewards.min():.4f}')

    replay_buffer._observations = observations
    replay_buffer._actions = actions
    replay_buffer._rewards = rewards
    replay_buffer._terminals = terminals
    replay_buffer._next_obs = next_observations
    replay_buffer._paths = paths
    replay_buffer._size = size
    replay_buffer._top = replay_buffer._size
    if replay_buffer.max_replay_buffer_size() < replay_buffer._size:
        replay_buffer._max_replay_buffer_size = replay_buffer._size
    logger.log(f'\nReplay buffer size : {replay_buffer._size}')
    logger.log(f"obs dim            : ", observations.shape)
    logger.log(f"action dim         : ", actions.shape)
    logger.log(f'# terminals: {replay_buffer._terminals.sum()}')
    logger.log(f'Mean rewards       : {replay_buffer._rewards.mean():.2f}')

    return paths

def sample_paths_within_max_size(dataset, paths, max_size, rng):
    idxs = np.arange(len(paths))
    idxs = rng.permutation(idxs)

    ret = []
    i = 0
    size = 0
    while size < max_size:
        path_i = paths[idxs[i]]
        ret.append(path_i)
        i += 1
        size += path_i['observations'].shape[0]

    dataset = dict(
        observations=np.zeros((0, dataset['observations'].shape[1])),
        actions=np.zeros((0, dataset['actions'].shape[1])),
        next_observations=np.zeros((0, dataset['next_observations'].shape[1])),
        rewards=np.zeros((0, 1)),
        terminals=np.zeros((0, 1)),
        timeouts=np.zeros((0, 1)),
    )

    for path in ret:
        for key in dataset:
            dataset[key] = np.concatenate([dataset[key], path[key]], axis=0)

    return ret, dataset


class DefaultEnvProc:
    @staticmethod
    def obs_preproc(obs):
        return obs

    @staticmethod
    def obs_postproc(obs, pred):
        """
        pred is the estimated difference between the current obs and the next_obs; hence, next_obs = obs + pred
        """
        return obs + pred

    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs


class HalfCheetahEnvProc(DefaultEnvProc):
    @staticmethod
    def obs_preproc(obs):
        if isinstance(obs, np.ndarray):
            if obs.ndim == 1:
                obs = obs.reshape(1, -1)
            if obs.shape[1] == 17:
                return np.concatenate([obs[:, 0:1], np.sin(obs[:, 1:2]), np.cos(obs[:, 1:2]), obs[:, 2:]], axis=1)
            elif obs.shape[1] == 18:
                return np.concatenate([obs[:, 1:2], np.sin(obs[:, 2:3]), np.cos(obs[:, 2:3]), obs[:, 3:]], axis=1)
            else:
                raise ValueError("Wrong observation dimension")
        elif isinstance(obs, torch.Tensor):
            if obs.ndim == 1:
                obs = obs.view(1, -1)
            if obs.size(1) == 17:
                return torch.cat([obs[:, 0:1], torch.sin(obs[:, 1:2]), torch.cos(obs[:, 1:2]), obs[:, 2:]], dim=1)
            elif obs.size(1) == 18:
                return torch.cat([obs[:, 1:2], torch.sin(obs[:, 2:3]), torch.cos(obs[:, 2:3]), obs[:, 3:]], dim=1)
            else:
                raise ValueError("Wrong observation dimension")
        else:
            raise ValueError("Wrong observation type provided")
