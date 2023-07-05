import hydra
import omegaconf
from omegaconf import OmegaConf

import numpy as np
import torch
import os.path as osp

from typing import Optional, Any
import rlkit.types

import rlkit.torch.pytorch_util as ptu
from rlkit.util.common import create_dynamics_and_reward_model
from rlkit.samplers.data_collector import MdpPathCollector, MdpStepCollector
from rlkit.envs.wrappers import ProxyEnv
from rlkit.envs.model_env import ModelEnv
from rlkit.envs.env_util import get_dim
from rlkit.data_management.replay_buffers.env_replay_buffer import EnvReplayBuffer
from rlkit.examples.algorithms.actor_critic.wrapper import SACAgent
from rlkit.policies.gaussian_policy import TanhGaussianPolicy
from rlkit.policies.util import complete_agent_cfg

import rlkit.util.pythonplusplus as ppp
from rlkit.torch.models.networks import FlattenMlp
from rlkit.trainers.mbpo import MBPOTrainer
from rlkit.trainers.mbrl import MBRLTrainer
from rlkit.trainers.sac import SACTrainer
from rlkit.trainers.behavior_clone import BehaviorCloneTrainer


def prepare_models_and_trainers(
        cfg: omegaconf.DictConfig,
        eval_env: ProxyEnv,
        reward_func: Optional[rlkit.types.RewardFuncType],
        term_func: Optional[rlkit.types.TermFuncType],
        rng: Optional[np.random.Generator] = None,
        *args,
        **kwargs
):
    """Prepares models and trainers to be used by an RlAlgorithm class.

    Note that this file can instantiate training modules for MBPO (default), MOPO, and MOReL,
    depending on configurations given in the file.

    Args:
        cfg         (omegaconf.DictConfig) A configuration related to this specific algorithm
        eval_env    (ProxyEnv) The wrapper environment
        reward_func (RewardFuncType) (optional) The true reward function of the environment
        term_func   (TermFuncType) (optional) The termination function of the environment
        rng         (np.random.Generator) Numpy RNG

    Return:         (dict) A dictionary containing instantiated objects to be passed to
                prepare_generic_rl_algorithm function
    """
    cache_dir = cfg.cache_dir
    obs_dim = get_dim(eval_env.observation_space)
    action_dim = get_dim(eval_env.action_space)

    input_dim = obs_dim + action_dim
    target_dim = obs_dim
    obs_preproc = hydra.utils.get_method(cfg.overrides.get('obs_preproc')) \
        if cfg.overrides.get('obs_preproc', False) else None
    obs_postproc = hydra.utils.get_method(cfg.overrides.get('obs_postproc')) \
        if cfg.overrides.get('obs_postproc', False) else None
    targ_proc = hydra.utils.get_method(cfg.overrides.get('targ_proc')) \
        if cfg.overrides.get('targ_proc', False) else None
    if obs_preproc is not None:
        input_dim = action_dim + obs_preproc(np.zeros((1, obs_dim))).shape[-1]
    if targ_proc is not None:
        target_dim = targ_proc(np.zeros((1, obs_dim)), np.zeros((1, obs_dim))).shape[-1]

    torch_generator = torch.Generator(device=ptu.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    """
    Create a stochastic policy
    """
    complete_agent_cfg(eval_env, cfg.algorithm.agent)
    M = cfg.algorithm.agent.layer_size
    num_hidden_layer = cfg.algorithm.agent.num_hidden_layer
    hidden_init = hydra.utils.get_method(cfg.algorithm.agent.get('hidden_init')) if cfg.algorithm.agent.get('hidden_init', False) else ptu.fanin_init
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M] * num_hidden_layer,
        log_std_bounds=[-5, 2],
        hidden_init=hidden_init,
        b_init_value=0.,
    )

    """
    Create Q functions and their target functions.
    """
    qf1, qf2, target_qf1, target_qf2 = ppp.group_init(
        4,
        FlattenMlp,
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M] * num_hidden_layer,
        hidden_init=hidden_init,
        b_init_value=0.,
    )

    """
    Set up the SAC agent
    """
    agent: SACAgent
    agent = hydra.utils.instantiate(cfg.algorithm.agent,
                                    policy=policy,
                                    _recursive_=False,
                                    )

    """
    Set up the trainer of SAC
    """
    policy_trainer = SACTrainer(
        env=eval_env,
        agent=agent,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **cfg.overrides.trainer_cfg.policy_cfg,
    )

    """
    Model-based reinforcement learning (MBRL) dynamics model
    """
    dynamics_model = create_dynamics_and_reward_model(
        cfg=cfg,
        obs_dim=obs_dim,
        input_dim=input_dim,
        action_dim=action_dim,
        target_dim=target_dim,
    )

    model_trainer = MBRLTrainer(
        dynamics_model=dynamics_model,
        obs_preproc=obs_preproc,
        targ_proc=targ_proc,
        rng=rng,
        **cfg.algorithm.dynamics,
    )

    model_env = ModelEnv(
        eval_env,
        dynamics_model,
        termination_func=term_func,
        reward_func=reward_func,
        generator=torch_generator,
        obs_preproc=obs_preproc,
        obs_postproc=obs_postproc,
        sampling_mode=cfg.algorithm.sampling_mode if 'sampling_mode' in cfg.algorithm else None,
        sampling_cfg=cfg.algorithm.sampling_cfg if 'sampling_cfg' in cfg.algorithm else None,
        clip_obs=cfg.overrides.trainer_cfg.get('clip_obs', False),
    )

    """
    Set up the MBPO trainer
    """
    trainer = MBPOTrainer(
        policy_trainer=policy_trainer,
        dynamics_model=dynamics_model,
        model_env=model_env,
        **cfg.overrides.trainer_cfg,
    )

    """
    (Optional) Set up behavior cloning trainer
    """
    bc_trainer = None
    if cfg.is_offline and 'offline_cfg' in cfg.overrides and cfg.overrides.offline_cfg.get('use_behavior_clone', False):
        from rlkit.examples.algorithms.actor_critic.wrapper import BehaviorCloningPolicy
        offline_cfg = cfg.overrides.offline_cfg
        bc_policy = BehaviorCloningPolicy(
            policy=policy,
            cfg=cfg,
        )
        bc_trainer = BehaviorCloneTrainer(
            model=bc_policy,
            obs_dim=obs_dim,
            action_dim=action_dim,
            obs_preproc=obs_preproc,
            **offline_cfg.prior_model,
        )

    """
    Store all modules to a dictionary
    """
    cfg_dict = dict(
        trainer=trainer,
        model_trainer=model_trainer,
        model_env=model_env,
        behavior_clone_trainer=bc_trainer,
        agent=agent,
    )
    return cfg_dict

def prepare_generic_rl_algorithm(
        cfg: omegaconf.DictConfig,
        expl_env: ProxyEnv,
        eval_env: ProxyEnv,
        replay_buffer: EnvReplayBuffer,
        agent: SACAgent,
        trainer: MBPOTrainer,
        model_trainer: MBRLTrainer,
        model_env: ModelEnv,
        behavior_clone_trainer: Optional[BehaviorCloneTrainer] = None,
        **kwargs):

    # Set up sample collectors for exploration / evaluation
    exploration_policy = agent.policy
    evaluation_policy = agent.eval_policy

    # Link the trainer with the replay buffer
    trainer.set_replay_buffer(replay_buffer)

    if cfg.is_offline:
        expl_path_collector = None
    elif cfg.algorithm.collector_type == 'step':
        expl_path_collector = MdpStepCollector(expl_env, exploration_policy)
    elif cfg.algorithm.collector_type == 'path':
        expl_path_collector = MdpPathCollector(expl_env, exploration_policy)
    else:
        raise NotImplementedError('collector_type of experiment not recognized')

    # Get post epoch functions
    post_epoch_funcs = []
    if cfg.algorithm.get('eval_transition_model', False):
        from rlkit.util.eval_util import eval_dynamics
        def eval_func(eval_paths):
            return eval_dynamics(model_env.dynamics_model,
                                 eval_paths,
                                 obs_preproc=model_env.obs_preproc,
                                 obs_postproc=model_env.obs_postproc,
                                 )
        post_epoch_funcs.append(eval_func)

    eval_path_collector = MdpPathCollector(
        eval_env,
        evaluation_policy,
        is_offline=cfg.is_offline,
        post_epoch_funcs=post_epoch_funcs
    )

    # Update the configuration when doing the offline RL
    if cfg.is_offline:
        offline_cfg = OmegaConf.to_container(cfg.overrides.offline_cfg, resolve=True)
        algorithm_cfg = OmegaConf.to_container(cfg.algorithm.algorithm_cfg, resolve=True)
        algorithm_cfg.update(offline_cfg)
        cfg.algorithm.algorithm_cfg = OmegaConf.create(algorithm_cfg)

    # Algorithm class: either TorchMBRLAlgorithm or TorchOfflineMBRLAlgorithm
    from rlkit.core.rl_algorithms.torch_rl_algorithm import (
        TorchMBBatchRLAlgorithm, TorchOfflineMBRLAlgorithm, TorchMBRLAlgorithm
    )
    if cfg.is_offline:
        algorithm = TorchOfflineMBRLAlgorithm(
            trainer=trainer,
            model_trainer=model_trainer,
            evaluation_policy=evaluation_policy,
            evaluation_env=eval_env,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            behavior_clone_trainer=behavior_clone_trainer,
            **cfg.algorithm.algorithm_cfg,
        )
    elif cfg.algorithm.collector_type == 'step':
        algorithm = TorchMBRLAlgorithm(
            trainer=trainer,
            model_trainer=model_trainer,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            **cfg.algorithm.algorithm_cfg,
        )
    else:
        algorithm = TorchMBBatchRLAlgorithm(
            trainer=trainer,
            model_trainer=model_trainer,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            **cfg.algorithm.algorithm_cfg,
        )
    return algorithm
