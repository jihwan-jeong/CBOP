import importlib
import numpy as np

import omegaconf
import hydra

import rlkit.types
import rlkit.envs.wrappers
from rlkit.core.rl_algorithms import (
    TorchMBRLAlgorithm, TorchOfflineRLAlgorithm, TorchMBBatchRLAlgorithm,
    TorchBatchRLAlgorithm, TorchOnlineRLAlgorithm,
)
from rlkit.data_management.replay_buffers.env_replay_buffer import EnvReplayBuffer
from typing import Optional, Union


def get_algorithm(
        cfg: omegaconf.DictConfig,
        expl_env: rlkit.envs.wrappers.ProxyEnv,
        eval_env: rlkit.envs.wrappers.ProxyEnv,
        replay_buffer: EnvReplayBuffer,
        reward_func: Optional[rlkit.types.RewardFuncType],
        term_func: Optional[rlkit.types.TermFuncType],
        rng: Optional[np.random.Generator] = None,
) -> Union[TorchMBRLAlgorithm,
           TorchOnlineRLAlgorithm,
           TorchMBBatchRLAlgorithm,
           TorchBatchRLAlgorithm,
           TorchOfflineRLAlgorithm,
           ]:
    """
    Given the configuration specifying the environment,

    Args:
        cfg             (omegaconf.DictConfig)
        expl_env        (ProxyEnv)
        eval_env        (ProxyEnv)
        replay_buffer   (EnvReplayBuffer)
        reward_func     (RewardFuncType)
        term_func       (TermFuncType)
        rng             (np.random.Generator) Numpy RNG

    Returns:            (dict)
    """
    # Instantiate algorithm-specific models/trainers/etc
    get_modules = hydra.utils.get_method(cfg.algorithm.get_module_path)
    get_algorithms = hydra.utils.get_method(cfg.algorithm.get_algo_path)

    dict_modules = get_modules(cfg,
                               eval_env,
                               reward_func,
                               term_func,
                               rng,
                               )

    algo = get_algorithms(cfg,
                          expl_env=expl_env,
                          eval_env=eval_env,
                          replay_buffer=replay_buffer,
                          **dict_modules)

    return algo
