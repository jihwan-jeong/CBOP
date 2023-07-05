import multiprocessing as mp

import gym.wrappers
import numpy as np
import torch

from rlkit.envs import env_util, wrappers, env_util as env_util
from rlkit.torch import pytorch_util as ptu
from typing import Optional, Union, Tuple, Callable, cast, Sequence
import gym

from rlkit.samplers import util as sampler_util

env__: Union[gym.Env, wrappers.ProxyEnv] = None
lock__ = None

def init(env_name: str, seed: int, is_offline: bool, d4rl_config: str, lock: Optional = None):
    global env__
    global lock__
    env__ = env_util.make_env_from_str(env_name, is_offline, d4rl_config)
    env__.seed(seed)
    lock__ = lock

def step_env(action: np.ndarray):
    global env__
    return env__.step(action)


def evaluate_plan_fn(
    plan: np.ndarray,
    current_state: Tuple,
    behavior_clone: Optional[Callable] = None,
) -> float:
    global env__
    gpu_mode = torch.cuda.is_available()
    ptu.set_gpu_mode(gpu_mode)
    # obs0__ is not used (only here for compatibility with rollout_env)
    obs0 = env__.observation_space.sample()
    env = cast(gym.wrappers.TimeLimit, env__)
    env_util.set_env_state(current_state, env)
    _, rewards_, _ = sampler_util.rollout_plan_mujoco_env(
        env, obs0, -1, agent=None, plan=plan, behavior_clone=behavior_clone,
    )
    return rewards_.sum().item()


def evaluate_all_plans(
    plans: Sequence[Sequence[np.ndarray]],
    pool: mp.Pool,
    current_state: Tuple,
    behavior_clone: Optional[Callable] = None,
) -> torch.Tensor:

    res_objs = [
        pool.apply_async(evaluate_plan_fn, (plan, current_state, behavior_clone)) for plan in plans
    ]
    res = [res_obj.get() for res_obj in res_objs]
    return torch.tensor(res, dtype=torch.float32)