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
from rlkit.examples.algorithms.actor_critic.wrapper import CQLAgent
from rlkit.policies.base.base import MakeDeterministic
from rlkit.policies.gaussian_policy import TanhGaussianPolicy
from rlkit.policies.util import complete_agent_cfg

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
    from rlkit.trainers.sac import CQLTrainer

    cache_dir = cfg.cache_dir
    obs_dim = get_dim(eval_env.observation_space)
    action_dim = get_dim(eval_env.action_space)
    input_dim = obs_dim + action_dim

    complete_agent_cfg(eval_env, cfg.algorithm.agent)
    M = cfg.algorithm.agent.layer_size
    num_hidden_layer = cfg.algorithm.agent.num_hidden_layer

    """
    Create Q functions and their target functions.
    """
    qf1, qf2, target_qf1, target_qf2 = ppp.group_init(
        4,
        FlattenMlp,
        input_size=input_dim,
        output_size=1,
        hidden_sizes=[M] * num_hidden_layer,
    )

    """
    Create a stochastic policy
    """
    policy = TanhGaussianPolicy(
        obs_dim=input_dim - action_dim,
        action_dim=action_dim,
        hidden_sizes=[M] * num_hidden_layer,
    )


    """
    Set up the SAC agent
    """
    agent: CQLAgent
    agent = hydra.utils.instantiate(cfg.algorithm.agent,
                                    policy=policy,
                                    qf=qf1,
                                    _recursive_=False,
                                    )

    """
    Set up the trainer of SAC
    """
    trainer = CQLTrainer(
        env=eval_env,
        agent=agent,
        qf1=qf1,
        qf2=qf2,
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
        agent: CQLAgent,
        trainer: TorchTrainer,
        **kwargs):

    # Set up sample collectors for exploration / evaluation
    exploration_policy = agent.policy
    evaluation_policy = agent.eval_policy
    if cfg.is_offline:
        expl_path_collector = None
    elif cfg.algorithm.collector_type == 'step':
        expl_path_collector = MdpStepCollector(expl_env, exploration_policy)
    elif cfg.algorithm.collector_type == 'path':
        expl_path_collector = MdpPathCollector(expl_env, exploration_policy)
    else:
        raise NotImplementedError('collector_type of experiment not recognized')
    eval_path_collector = MdpPathCollector(eval_env, evaluation_policy, is_offline=True)

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
