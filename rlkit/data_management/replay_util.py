from typing import Union, Optional, Callable

import numpy as np
import omegaconf

from rlkit.envs.wrappers import ProxyEnv
from rlkit.data_management.replay_buffers.env_replay_buffer import EnvReplayBuffer


def make_buffer_from_cfg(
        env: ProxyEnv,
        cfg: Union[omegaconf.ListConfig, omegaconf.DictConfig],
        rng: np.random.Generator = np.random.default_rng(),
        reward_func: Optional[Callable] = None,
):
    replay_buffer = EnvReplayBuffer(
        max_replay_buffer_size=int(cfg.algorithm.replay_buffer_size),
        env=env,
        rng=rng,
    )
    return replay_buffer
