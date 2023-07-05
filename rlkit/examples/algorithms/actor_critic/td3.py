import hydra
import omegaconf
from omegaconf import OmegaConf

import numpy as np
import torch

from typing import Optional, Any
import rlkit.types

import rlkit.torch.pytorch_util as ptu
from rlkit.core.rl_algorithms.torch_rl_algorithm import TorchTrainer
from rlkit.samplers.data_collector import MdpPathCollector, MdpStepCollector
from rlkit.envs.wrappers import ProxyEnv
from rlkit.envs.env_util import get_dim
from rlkit.data_management.replay_buffers.env_replay_buffer import EnvReplayBuffer
from rlkit.examples.algorithms.actor_critic.wrapper import TD3Agent
from rlkit.policies.base.base import MakeDeterministic
from rlkit.torch.models.networks import TanhMlpPolicy
from rlkit.policies.util import complete_agent_cfg
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
import rlkit.util.pythonplusplus as ppp
from rlkit.torch.models.networks import FlattenMlp

def prepare_models_and_trainers(
        cfg: omegaconf.DictConfig,
        eval_env: ProxyEnv,
        *args,
        **kwargs
):
    """
    Prepares models and trainers to be used by an RlAlgorithm class

    Args:
        cfg         (omegaconf.DictConfig) A configuration related to this specific algorithm
        eval_env    (ProxyEnv) The wrapper environment

    Return:         (dict) A dictionary containing instantiated objects to be passed to
                prepare_generic_rl_algorithm function
    """
    from rlkit.trainers.td3 import TD3Trainer

    cache_dir = cfg.cache_dir
    obs_dim = get_dim(eval_env.observation_space)
    action_dim = get_dim(eval_env.action_space)
    input_dim = obs_dim + action_dim

    torch_generator = torch.Generator(device=ptu.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    """
    Create a stochastic policy
    """
    complete_agent_cfg(eval_env, cfg.algorithm.agent)
    hidden_sizes = cfg.algorithm.agent.hidden_sizes
    policy = TanhMlpPolicy(
        input_size=input_dim - action_dim,
        output_size=action_dim,
        hidden_sizes=hidden_sizes,
    )
    target_policy = TanhMlpPolicy(
        input_size=input_dim - action_dim,
        output_size=action_dim,
        hidden_sizes=hidden_sizes,
    )
    es = GaussianStrategy(
        action_space=eval_env.action_space,
        max_sigma=0.1,
        min_sigma=0.1,  # Constant sigma
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    """
    Create Q functions and their target functions.
    """
    qf1, qf2, target_qf1, target_qf2 = ppp.group_init(
        4,
        FlattenMlp,
        input_size=input_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )

    """
    Set up the SAC agent
    """
    agent: TD3Agent
    agent = hydra.utils.instantiate(cfg.algorithm.agent,
                                    expl_policy=exploration_policy,
                                    eval_policy=policy,
                                    _recursive_=False,
                                    )

    """
    Set up the trainer of SAC
    """
    trainer = TD3Trainer(
        agent=agent,
        qf1=qf1,
        qf2=qf2,
        target_policy=target_policy,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **cfg.overrides.trainer_cfg,
    )

    """
    Store all modules to a dictionary
    """
    cfg_dict = dict(
        trainer=trainer,
        agent=agent,
    )
    return cfg_dict

def prepare_generic_rl_algorithm(
        cfg: omegaconf.DictConfig,
        expl_env: ProxyEnv,
        eval_env: ProxyEnv,
        replay_buffer: EnvReplayBuffer,
        agent: TD3Agent,
        trainer: TorchTrainer,
        **kwargs):

    # Set up sample collectors for exploration / evaluation
    exploration_policy = agent.expl_policy
    evaluation_policy = agent.eval_policy
    if cfg.is_offline:
        expl_path_collector = None
    elif cfg.algorithm.collector_type == 'step':
        expl_path_collector = MdpStepCollector(expl_env, exploration_policy)
    elif cfg.algorithm.collector_type == 'path':
        expl_path_collector = MdpPathCollector(expl_env, exploration_policy)
    else:
        raise NotImplementedError('collector_type of experiment not recognized')
    eval_path_collector = MdpPathCollector(eval_env, evaluation_policy)

    # Update the configuration when doing the offline RL
    if cfg.is_offline:
        offline_cfg = OmegaConf.to_container(cfg.overrides.offline_cfg, resolve=True)
        algorithm_cfg = OmegaConf.to_container(cfg.algorithm.algorithm_cfg, resolve=True)
        algorithm_cfg.update(offline_cfg)
        cfg.algorithm.algorithm_cfg = OmegaConf.create(algorithm_cfg)

    # Algorithm class: either TorchMBRLAlgorithm or TorchOfflineMBRLAlgorithm
    from rlkit.core.rl_algorithms.torch_rl_algorithm import (
        TorchBatchRLAlgorithm, TorchOfflineRLAlgorithm, TorchOnlineRLAlgorithm
    )
    if cfg.is_offline:
        algorithm = TorchOfflineRLAlgorithm(
            trainer=trainer,
            evaluation_policy=evaluation_policy,
            evaluation_env=eval_env,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            **cfg.algorithm.algorithm_cfg,
         )
    elif cfg.algorithm.collector_type == 'step':
        algorithm = TorchOnlineRLAlgorithm(
            trainer=trainer,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            **cfg.algorithm.algorithm_cfg,
        )
    else:
        algorithm = TorchBatchRLAlgorithm(
            trainer=trainer,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            **cfg.algorithm.algorithm_cfg,
        )

    return algorithm
