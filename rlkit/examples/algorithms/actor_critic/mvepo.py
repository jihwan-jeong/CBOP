import hydra
from pathlib import Path
import omegaconf
from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Any
from rlkit.policies.util import complete_agent_cfg
from rlkit.trainers.mvepo.mvepo import MVEPOTrainer

import rlkit.types
import rlkit.torch.pytorch_util as ptu
from rlkit.util.common import create_dynamics_and_reward_model
from rlkit.torch.models.ensemble import EnsembleFCLayer, FlattenEnsembleMLP
from rlkit.core.rl_algorithms.torch_rl_algorithm import TorchTrainer
from rlkit.samplers.data_collector import MdpPathCollector, MdpStepCollector
from rlkit.envs.wrappers import ProxyEnv
from rlkit.envs.model_env import ModelEnv
from rlkit.envs.env_util import get_dim
from rlkit.data_management.replay_buffers.env_replay_buffer import SimpleReplayBuffer
from rlkit.policies.gaussian_policy import TanhGaussianPolicy
from rlkit.core.logging.logging import logger
import rlkit.util.pythonplusplus as ppp


def prepare_models_and_trainers(
        cfg: omegaconf.DictConfig,
        eval_env: ProxyEnv,
        reward_func: Optional[rlkit.types.RewardFuncType],
        term_func: Optional[rlkit.types.TermFuncType],
        rng: Optional[np.random.Generator] = None,
):
    """
    Prepares models and trainers to be used by an RlAlgorithm class

    Args:
        cfg         (omegaconf.DictConfig) A configuration related to this specific algorithm
        eval_env    (ProxyEnv) The wrapper environment
        reward_func (RewardFuncType) (optional) The true reward function of the environment
        term_func   (TermFuncType) (optional) The termination function of the environment
        rng         (np.random.Generator) Numpy RNG

    Return:         (dict) A dictionary containing instantiated objects to be passed to
                prepare_generic_rl_algorithm function
    """
    from rlkit.trainers.mbrl import MBRLTrainer

    eval_only = cfg.eval_policy

    # Get the cache directory
    cache_dir = cfg.cache_dir
    assert cache_dir is not None, "You should provide the directory at which the learend parameters are saved!"
    cwd = Path(hydra.utils.get_original_cwd())    
    if cfg.max_size is None:
        cache_dir = cwd / cfg.cache_dir
    else:
        cache_dir = cwd / cfg.cache_dir / f'{int(cfg.max_size)}'

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
    # if obs_preproc is not None:
    #     input_dim = action_dim + obs_preproc(np.zeros((1, obs_dim))).shape[-1]
    if targ_proc is not None:
        target_dim = targ_proc(np.zeros((1, obs_dim)), np.zeros((1, obs_dim))).shape[-1]

    torch_generator = torch.Generator(device=ptu.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)
    
    """
    Set up the dynamics model and the model environment
    """
    dynamics_model = create_dynamics_and_reward_model(
        cfg=cfg,
        obs_dim=obs_dim,
        input_dim=action_dim + obs_preproc(np.zeros((1, obs_dim))).shape[-1] if obs_preproc is not None else input_dim,
        action_dim=action_dim,
        target_dim=target_dim,
        cache_dir=cache_dir,
    )
    if cfg.is_offline:
        assert dynamics_model.trained
    
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
        clip_obs=cfg.algorithm.dynamics.clip_obs,
    )
    
    """
    Set up the actor and the Q ensemble function
    """
    # Instantiate Q ensemble model and load parameters
    complete_agent_cfg(eval_env, cfg.algorithm.agent)
    M = cfg.algorithm.agent.layer_size
    num_hidden_layer = cfg.algorithm.agent.num_hidden_layer
    num_qfs = cfg.algorithm.agent.num_qfs
    
    # Weight initialization method
    w_init_method_name = cfg.algorithm.agent.get('w_init_method', None)
    if w_init_method_name == 'uniform':
        def w_init_method(m: nn.Module, b_init_value: float = 0.1):
            if isinstance(m, nn.Linear) or isinstance(m, EnsembleFCLayer):
                ptu.uniform_init(m.weight, init_w=1)
                m.bias.data.fill_(b_init_value)
    else:
        def fanin_init(m: nn.Module, b_init_value=0.0):
            if isinstance(m, EnsembleFCLayer):
                for i in range(m.ensemble_size):
                    ptu.fanin_init(m.weight[i])
                    m.bias[i].data.fill_(b_init_value)
        w_init_method = fanin_init
        
    qfs, target_qfs = ppp.group_init(
        2,
        FlattenEnsembleMLP,
        ensemble_size=num_qfs,
        input_size=input_dim,
        output_size=1,
        hidden_sizes=[M] * num_hidden_layer,
        hidden_activation=torch.relu,
        b_init_value=0.1,
        w_init_method=w_init_method,
    )

    """
    Set up the policy
    """
    policy = TanhGaussianPolicy(
        obs_dim=input_dim - action_dim,
        action_dim=action_dim,
        hidden_sizes=[M] * num_hidden_layer,
    )
    
    """Load pretrained policy/qf parameters"""
    cpt_type = cfg.overrides.offline_cfg.get('checkpoint_type', 'behavior')     # 'behavior' or 'learned'
    f_checkpoint = cfg.overrides.offline_cfg.get('checkpoint_name', None)
    if cpt_type == 'behavior':
        # Load QFs parameters
        qf_cpt = cache_dir / 'policy_eval_checkpoint.pth'
        try:
            qfs.load(qf_cpt, key='value_func')
            if qfs.trained:
                logger.log(f"Successfully loaded Q ensemble parameters from {qf_cpt}!")
            else:
                logger.log("Failed to load Q ensemble parameters... Check configuration!")
        except Exception as e:
            logger.log(f"Failed to load pretrained Q parameters from {qf_cpt}")
            logger.log(f"Exception: {e}")
            logger.log("...Q function will be learned from scratch.")
        
        # Load policy parameters
        actor_cpt = cache_dir / 'behavior_clone_checkpoint.pth'
        try:
            state_dict = torch.load(actor_cpt, map_location=ptu.device)
            policy.load(state_dict, key='behavior_clone')
            logger.log(f"Successfully loaded policy parameters from {actor_cpt}!")
        except Exception as e:
            logger.log(f"Failed to load policy parameters from {str(actor_cpt)}...")
            logger.log(f"Exception: {e}")
            logger.log("Policy will be learned from scratch.")
    
    elif cpt_type == 'learned':
        if f_checkpoint is not None:
            logger.log(f"Loading QFs / actor from checkpoint {f_checkpoint}")
            file_path = cache_dir / f_checkpoint
            # Load QFs parameters
            try:
                qfs.load(file_path, key='trainer/qfs')
                if qfs.trained:
                    logger.log(f"Successfully loaded Q ensemble parameters from {file_path}!")
                else:
                    logger.log("Failed to load Q ensemble parameters... Check configuration!")
            except:
                logger.log(f"Failed to load Q ensemble parameters from {file_path}...\nQ function will be learned from scratch.")
            # Load policy parameters
            try:
                state_dict = torch.load(file_path, map_location=ptu.device)
                policy.load(state_dict, key='trainer/policy')
                logger.log(f"Successfully loaded policy parameters from {file_path}!")
            except Exception as e:
                logger.log(f"Failed to load policy parameters from {str(file_path)}...")
                logger.log(f"Exception: {e}")
                logger.log("Policy will be learned from scratch.")
    else:
        raise ValueError("Checkpoint type not recognized... exiting")
    
    """
    Set up the actor critic agent
    """
    agent = hydra.utils.instantiate(cfg.algorithm.agent,
                                    policy=policy,
                                    f_checkpoint=f_checkpoint,
                                    load_param=eval_only,   # Load learned parameters if only used for evaluation
                                    _recursive_=False,
    )

    """
    Set up the trainer
    """
    trainer = MVEPOTrainer(
        env=eval_env,
        policy=policy,
        qfs=qfs,
        target_qfs=target_qfs,
        dynamics_model=dynamics_model,
        model_env=model_env,
        **cfg.overrides.trainer_cfg,
    )
    
    """
    Store all modules to a dictionary
    """
    cfg_dict = dict(
        trainer=trainer,
        model_trainer=model_trainer,
        model_env=model_env,
        agent=agent,
    )

    return cfg_dict


def prepare_generic_rl_algorithm(
        cfg: omegaconf.DictConfig,
        expl_env: ProxyEnv,
        eval_env: ProxyEnv,
        replay_buffer: SimpleReplayBuffer,
        agent,
        trainer: TorchTrainer,
        model_env: ModelEnv,
        model_trainer: TorchTrainer = None,
        **kwargs):
    # For evaluation only
    if cfg.eval_policy:
        cfg.algorithm.algorithm_cfg.num_epochs = 0
        if 'min_num_steps_before_training' in cfg.algorithm.algorithm_cfg:
            cfg.algorithm.algorithm_cfg.min_num_steps_before_training = 0

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
    eval_path_collector = MdpPathCollector(eval_env, evaluation_policy, is_offline=cfg.is_offline)

    # Update the configuration when doing the offline RL
    if cfg.is_offline:
        try:
            offline_cfg = OmegaConf.to_container(cfg.overrides.offline_cfg, resolve=True)
            algorithm_cfg = OmegaConf.to_container(cfg.algorithm.algorithm_cfg, resolve=True)
            algorithm_cfg.update(offline_cfg)
            cfg.algorithm.algorithm_cfg = OmegaConf.create(algorithm_cfg)
        except Exception as e:
            logger.log("Could not import cfg.overrides.offline_cfg")       

    # Algorithm class: either TorchMBRLAlgorithm or TorchOfflineMBRLAlgorithm
    from rlkit.core.rl_algorithms.torch_rl_algorithm import (
        TorchMBRLAlgorithm, TorchOfflineRLAlgorithm
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
    else:
        algorithm = TorchMBRLAlgorithm(
            trainer=trainer,
            model_trainer=model_trainer,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            num_model_learning_epochs=cfg.overrides.dynamics.num_model_learning_epochs,
            **cfg.algorithm.algorithm_cfg,
        )

    return algorithm
