from pathlib import Path
import hydra
import omegaconf
from omegaconf import OmegaConf

import numpy as np
import torch
import torch.nn as nn
import copy
from os import path as osp
from typing import Optional
from rlkit.core.logging.logging import logger

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.models.ensemble import EnsembleFCLayer
from rlkit.trainers.wrapper import TrainerWrapper
from rlkit.util.common import create_dynamics_and_reward_model
from rlkit.envs.wrappers import ProxyEnv
from rlkit.envs.env_util import get_dim
from rlkit.data_management.replay_buffers.env_replay_buffer import EnvReplayBuffer

from rlkit.trainers.mbrl import MBRLTrainer
from rlkit.trainers.behavior_clone import BehaviorCloneTrainer
from rlkit.trainers.pol_eval import PolEvalTrainer


def prepare_models_and_trainers(
        cfg: omegaconf.DictConfig,
        eval_env: ProxyEnv,
        *args,
        **kwargs
):
    """Prepares the dynamics model and trainer to be used by an RlAlgorithm class.

    Note that this file can instantiate training modules for MBPO (default), MOPO, and MOReL,
    depending on configurations given in the file.

    Args:
        cfg         (omegaconf.DictConfig) A configuration related to this specific algorithm
        eval_env    (ProxyEnv) The wrapper environment
        reward_func (RewardFuncType) (optional) The true reward function of the environment
        term_func   (TermFuncType) (optional) The termination function of the environment

    Return:         (dict) A dictionary containing instantiated objects to be passed to
                prepare_generic_rl_algorithm function
    """
    assert cfg.is_offline

    cache_dir = cfg.cache_dir
    save_dir = cfg.save_dir
    assert save_dir is not None, "You must provide the directory to which learned parameters are saved"
    
    cwd = Path(hydra.utils.get_original_cwd())
    
    if cfg.max_size is None:
        cache_dir = cwd / cache_dir if cache_dir is not None else cache_dir
        save_dir = cwd / save_dir
    else:
        cache_dir = cwd / cache_dir / f'{int(cfg.max_size)}' if cache_dir is not None else cache_dir
        save_dir = cwd / save_dir / f'{int(cfg.max_size)}'
    
    if not save_dir.exists():
        logger.log(f"Save dir {str(save_dir)} does not exist and is created")
        save_dir.mkdir(parents=True)

    obs_dim = get_dim(eval_env.observation_space)
    action_dim = get_dim(eval_env.action_space)

    input_dim = obs_dim + action_dim
    target_dim = obs_dim
    obs_preproc = hydra.utils.get_method(cfg.overrides.get('obs_preproc')) \
        if cfg.overrides.get('obs_preproc', False) else None
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
    Dynamics model for model-based reinforcement learning (MBRL) 
    """
    dynamics_model, dynamics_trainer = None, None
    if cfg.algorithm.dynamics.train:
        dynamics_model = create_dynamics_and_reward_model(
            cfg=cfg,
            obs_dim=obs_dim,
            input_dim=input_dim,
            action_dim=action_dim,
            target_dim=target_dim,
            cache_dir=cache_dir,
        )

        dynamics_trainer = MBRLTrainer(
            dynamics_model=dynamics_model,
            obs_preproc=obs_preproc,
            targ_proc=targ_proc,
            **cfg.algorithm.dynamics,
        )

    """
    Set up behavior cloning trainer (used for MBOP)
    """
    bc_prior, bc_trainer = None, None
    if cfg.algorithm.bc_prior.train:
        bc_cfg = cfg.algorithm.bc_prior
        hidden_activation = hydra.utils.get_method(bc_cfg.get('activation_func')) \
            if bc_cfg.get('activation_func', False) else torch.relu

        bc_prior = hydra.utils.instantiate(
            bc_cfg,
            hidden_sizes=[bc_cfg.layer_size] * bc_cfg.num_hidden_layer,
            obs_dim=input_dim - action_dim,
            action_dim=action_dim,
            hidden_activation=hidden_activation,
        )

        # Set up the trainer
        bc_trainer = BehaviorCloneTrainer(
            model=bc_prior,
            obs_dim=obs_dim,
            action_dim=action_dim,
            obs_preproc=obs_preproc,
            **bc_cfg,
        )

    """
    Set up the value function and its trainer (set to None if not used)
    """
    value_func, pol_eval_trainer = None, None
    if cfg.algorithm.policy_eval.train:
        policy_eval_cfg = cfg.algorithm.policy_eval
        M = policy_eval_cfg['layer_size']
        num_hidden_layer = policy_eval_cfg['num_hidden_layer']
        hidden_activation = hydra.utils.get_method(policy_eval_cfg.get('activation_func')) \
            if policy_eval_cfg.get('activation_func', False) else torch.relu

        def fanin_init(m: nn.Module, b_init_value=0.0):
            if isinstance(m, EnsembleFCLayer):
                for i in range(m.ensemble_size):
                    ptu.fanin_init(m.weight[i])
                    m.bias[i].data.fill_(b_init_value)
                    
        value_func = hydra.utils.instantiate(
            policy_eval_cfg,
            input_size=input_dim if policy_eval_cfg.get('is_qf') else input_dim - action_dim,
            output_size=1,
            b_init_value=0.1,
            w_init_method=fanin_init,
            hidden_sizes=[M] * num_hidden_layer,
            hidden_activation=hidden_activation,
        )

        target_value_func = copy.deepcopy(value_func) if policy_eval_cfg.get('use_target_network', False) else None

        pol_eval_trainer = PolEvalTrainer(
            value_func=value_func,
            target_value_func=target_value_func,
            obs_preproc=obs_preproc,
            **policy_eval_cfg,
        )

    """
    Instantiate the TrainerWrapper
    """
    trainers = [dynamics_trainer, bc_trainer, pol_eval_trainer]
    trainer = TrainerWrapper(
        trainers=trainers,
        save_dir=save_dir,
    )

    """
    Store all modules to a dictionary
    """
    cfg_dict = dict(
        trainer=trainer,
    )
    return cfg_dict


def prepare_generic_rl_algorithm(
        cfg: omegaconf.DictConfig,
        replay_buffer: EnvReplayBuffer,
        trainer: TrainerWrapper,
        **kwargs
    ):
    import gtimer as gt
    import os

    assert cfg.is_offline, "Pretraining models can only be done for offline RL environments"
    
    # Instantiate the algorithm class
    from rlkit.core.rl_algorithms.torch_rl_algorithm import TorchOfflinePretrainAlgorithm
    algorithm = TorchOfflinePretrainAlgorithm(
        trainer=trainer,
        replay_buffer=replay_buffer,
        **cfg.algorithm.algorithm_cfg,
    )
    
    return algorithm
