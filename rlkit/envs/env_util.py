import os

from typing import Union, Optional
from typing import Tuple as TupleType
import hydra.utils
import omegaconf
from omegaconf import OmegaConf

import rlkit.types
from rlkit.envs.wrappers import NormalizedBoxEnv, ProxyEnv
from rlkit.core.logging.logging import logger

import gym
import gym.wrappers
from gym.spaces import Box, Discrete, Tuple

ENV_ASSET_DIR = os.path.join(os.path.dirname(__file__), 'assets')


def get_asset_full_path(file_name):
    return os.path.join(ENV_ASSET_DIR, file_name)


def get_dim(space):
    if isinstance(space, Box):
        return space.low.size
    elif isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Tuple):
        return sum(get_dim(subspace) for subspace in space.spaces)
    elif hasattr(space, 'flat_dim'):
        return space.flat_dim
    else:
        raise TypeError("Unknown space: {}".format(space))


def mode(env, mode_type):
    try:
        getattr(env, mode_type)()
    except AttributeError:
        pass


def make_env_from_cfg(
    cfg: Union[omegaconf.ListConfig, omegaconf.DictConfig],
    is_eval: bool = False,
) -> TupleType[ProxyEnv, rlkit.types.TermFuncType, Optional[rlkit.types.RewardFuncType]]:
    """Creates an environment from a given OmegaConf configuration object.
    This method expects the configuration, ``cfg``,
    to have the following attributes (some are optional):
        - ``cfg.overrides.env``: the string description of the environment.
          Valid options are:
          - "dmcontrol___<domain>--<task>": a Deep-Mind Control suite environment
            with the indicated domain and task (e.g., "dmcontrol___cheetah--run".
          - "gym___<env_name>": a Gym environment (e.g., "gym___HalfCheetah-v2").
          - "<mujoco_env_name>": a Gym MuJoCo environment (e.g., walker2d, used for D4RL tasks)
          - "cartpole_continuous": a continuous version of gym's Cartpole environment.
          - "pets_halfcheetah": the implementation of HalfCheetah used in Chua et al.,
            PETS paper.
          - "ant_truncated_obs": the implementation of Ant environment used in Janner et al.,
            MBPO paper.
          - "humanoid_truncated_obs": the implementation of Humanoid environment used in
            Janner et al., MBPO paper.
          - "pointmaze": the PointMaze environment following the implementation in L3P (Zhang et al., 2021)
        - ``cfg.overrides.term_func``: (only for dmcontrol and gym environments) a string
          indicating the environment's termination function to use when simulating the
          environment with the model. It should correspond to the name of a function in
          :mod:`rlkit.envs.termination_funcs` (note there's also no_termination func).
        - ``cfg.overrides.reward_func``: (only for dmcontrol and gym environments)
          a string indicating the environment's reward function to use when simulating the
          environment with the model. If not present, it will try to use ``cfg.overrides.term_func``.
          If that's not present either, it will return a ``None`` reward function.
          If provided, it should correspond to the name of a function in
          :mod:`rlkit.envs.reward_funcs`.

        - ``cfg.overrides.learn_reward``: (optional) if present indicates that
          the reward function will be learned, in which case the method will return
          a ``None`` reward function.

        TODO: trial_length should be set here?
        - ``cfg.overrides.trial_length``: (optional) if presents indicates the maximum length
          of trials. Defaults to 1000.
    Args:
        cfg (omegaconf.DictConf): the configuration to use.
    Returns:
        (tuple of env, termination function, reward function): returns the new environment,
        the termination function to use, and the reward function to use (or ``None`` if
        ``cfg.learned_rewards == True``).
    """
    # Specify termination and reward functions
    import rlkit.envs
    if 'term_func' in cfg.overrides:
        term_func = hydra.utils.get_method(cfg.overrides.term_func)
    else:
        term_func = hydra.utils.get_method(cfg.env.term_func)

    if hasattr(cfg.overrides, "reward_func") and cfg.overrides.reward_func is not None:
        reward_func = hydra.utils.get_method(
            cfg.overrides.reward_func)  # getattr(rlkit.envs.reward_funcs, cfg.overrides.reward_func)
    else:
        reward_func = None

    env_name = cfg.env.name

    # Update the config such that `env_cfg.is_eval` is created or overwritten
    if 'env_cfg' in cfg.env and 'is_eval' in cfg.env.env_cfg:
        cfg.env.env_cfg.is_eval = is_eval
    elif 'env_cfg' in cfg.env:
        env_cfg = OmegaConf.to_container(cfg.env.env_cfg, resolve=True)
        env_cfg['is_eval'] = is_eval
        cfg.env.env_cfg = OmegaConf.create(env_cfg)
    else:
        env_ = OmegaConf.to_container(cfg.env, resolve=True)
        env_['env_cfg'] = dict(is_eval=is_eval)
        cfg.env = OmegaConf.create(env_)
    
    # Instantiate the environment object
    if cfg.is_offline:
        # For offline RL, load the dataset from D4RL
        if env_name.lower() in ['hopper', 'walker2d', 'halfcheetah', 'pen', 'hammer', 'door', 'relocate', 'antmaze']:
            import d4rl
            env = gym.make("{}-{}".format(env_name.lower(), cfg.overrides.d4rl_config))

        else:
            raise NotImplementedError(f"The environment '{env_name}' is not recognized")

    elif "gym___" in cfg.overrides.env:
        import gym.envs.mujoco as mujoco_env
        env_name = cfg.overrides.env.split("___")[1].lower()

        if env_name == 'halfcheetah':
            env = mujoco_env.HalfCheetahEnv()
        elif env_name == "hopper":
            env = mujoco_env.HopperEnv()
        elif env_name == 'walker2d':
            env = mujoco_env.Walker2dEnv()

    elif env_name in ['halfcheetah', 'walker2d', 'hopper']:
        ENV_NAME_MAP = {'halfcheetah': 'HalfCheetah-v2', 'walker2d': 'Walker2d-v2', 'hopper': 'Hopper-v2'}
        env = gym.make(f'{ENV_NAME_MAP[env_name]}')
        
    else:
        # Custom MuJoCo environment implemented in rlkit.envs
        import rlkit.envs.mujoco_env
        if cfg.overrides.env == "cartpole_continuous":
            env = rlkit.envs.cartpole_continuous.CartPoleEnv()

        elif cfg.overrides.env == "pets_halfcheetah":
            env = rlkit.envs.mujoco_env.PETSHalfCheetahEnv()

        else:
            try:
                env = hydra.utils.instantiate(cfg.env.classdef)
            except:
                raise ValueError("Invalid environment string.")
        # elif cfg.overrides.env == "ant_truncated_obs":
        #     env = rlkit.envs.mujoco_env.AntTruncatedObsEnv()
        #     term_func = rlkit.envs.termination_funcs.ant
        #     reward_func = None
        # elif cfg.overrides.env == "humanoid_truncated_obs":
        #     env = rlkit.envs.mujoco_env.HumanoidTruncatedObsEnv()
        #     term_func = rlkit.envs.termination_funcs.ant
        #     reward_func = None
        # else:

    if not isinstance(env.action_space, gym.spaces.Discrete):
        env = NormalizedBoxEnv(env)
    else:
        env = ProxyEnv(env)

    if "dynamics" in cfg.algorithm.keys():
        learn_reward = cfg.algorithm.dynamics.get("learn_reward", True)
        use_true_reward = cfg.algorithm.dynamics.get("use_true_reward", False)
        if learn_reward and not use_true_reward:
            logger.log("DynamicsModel will use the learned reward function")
            reward_func = None
        else:
            logger.log("DynamicsModel will use the true reward function")

    if cfg.seed is not None:
        env.seed(cfg.seed)
        env.observation_space.seed(cfg.seed + 1)
        env.action_space.seed(cfg.seed + 1)

    return env, term_func, reward_func


def make_env_from_str(
        env_name: str, is_offline: Optional[bool] = False, d4rl_config: Optional[str] = None,
) -> gym.Env:
    """Creates a new gym environment from the name of the environment.

    Args:
        env_name (str): The string description of the environment.
        is_offline (bool): Whether an offline environment is used
        d4rl_config (str): When `is_offline=True`, this should be specified. An example would be
            `medium-expert-v2`.

    Returns:
        (gym.Env) The created environment.
    """
    if "dmcontrol___" in env_name:
        import rlkit.third_party.dmc2gym as dmc2gym
        domain, task = env_name.split("___")[1].split("--")
        env = dmc2gym.make(domain_name=domain, task_name=task)
    elif "gym___" in env_name:
        env = gym.make(env_name.split("___")[1])
    elif is_offline:
        assert d4rl_config is not None
        import d4rl
        env = gym.make(f"{env_name.lower()}-{d4rl_config}")
    else:
        # Custom MuJoCo environment implemented in rlkit.envs
        import rlkit.envs.mujoco_env
        if env_name == "cartpole_continuous":
            env = rlkit.envs.cartpole_continuous.CartPoleEnv()

        elif env_name == "pets_halfcheetah":
            env = rlkit.envs.mujoco_env.PETSHalfCheetahEnv()

        else:
            try:
                env = hydra.utils.instantiate(env_name)
            except:
                raise ValueError("Invalid environment string.")
        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)

    return env

def get_current_state(
        env: Union[gym.wrappers.TimeLimit, ProxyEnv],
):
    """Returns the internal state of the environment.

    """
    if isinstance(env, ProxyEnv):
        env = env.wrapped_env
    if "mujoco" in env.spec.entry_point:
        state = (
            env.env.data.qpos.ravel().copy(),
            env.env.data.qvel.ravel().copy(),
        )
        elapsed_steps = env._elapsed_steps
        return state, elapsed_steps
    elif "rlkit.third_party.dmc2gym" in env.env.__class__.__module__:
        state = env.env._env.physics.get_state().copy()
        elapsed_steps = env._elapsed_steps
        step_count = env.env._env._step_count
        return state, elapsed_steps, step_count
    else:
        raise NotImplementedError(
            "Only gym mujoco and dm_control environments supported."
        )

def set_env_state(state: Tuple, env: Union[gym.wrappers.TimeLimit, ProxyEnv]):
    """Sets the state of the environment.

    This function assumes `state` was generated using :func:`get_current_state`,
    and only works with mujoco gym and dm_control environments.

    Args:
        state (tuple): See :func:`get_current_state` for description.
        env (TimeLimit or ProxyEnv): The environment.
    """
    if isinstance(env, ProxyEnv):
        env = env.wrapped_env
    if "mujoco" in env.spec.entry_point:
        env.set_state(*state[0])
        env._elapsed_steps = state[1]
    elif "rlkit.third_party.dmc2gym" in env.env.__class__.__module__:
        with env.env._env.physics.reset_context():
            env.env._env.physics.set_state(state[0])
            env._elapsed_steps = state[1]
            env.env._env._step_count = state[2]
    else:
        raise NotImplementedError(
            "Only gym mujoco and dm_control environments supported."
        )
