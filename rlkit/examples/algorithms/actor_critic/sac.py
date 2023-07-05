import hydra
import omegaconf
from omegaconf import OmegaConf
from pathlib import Path
import torch
import torch.nn as nn

from rlkit.torch import pytorch_util as ptu
from rlkit.torch.models.ensemble import EnsembleFCLayer, FlattenEnsembleMLP
from rlkit.core.rl_algorithms.torch_rl_algorithm import TorchTrainer
from rlkit.samplers.data_collector import MdpPathCollector, MdpStepCollector
from rlkit.envs.wrappers import ProxyEnv
from rlkit.envs.env_util import get_dim
from rlkit.data_management.replay_buffers.env_replay_buffer import EnvReplayBuffer
from rlkit.examples.algorithms.actor_critic.wrapper import SACAgent
from rlkit.policies.gaussian_policy import TanhGaussianPolicy
from rlkit.policies.util import complete_agent_cfg
from rlkit.core.logging.logging import logger
import rlkit.util.pythonplusplus as ppp


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
    from rlkit.trainers.sac import SACTrainer

    cache_dir = cfg.cache_dir
    eval_only = cfg.eval_policy
    if eval_only:
        assert cfg.f_checkpoint_name is not None
        f_checkpoint = Path.cwd() / cache_dir / cfg.f_checkpoint_name
    else:
        f_checkpoint = None

    obs_dim = get_dim(eval_env.observation_space)
    action_dim = get_dim(eval_env.action_space)
    input_dim = obs_dim + action_dim

    complete_agent_cfg(eval_env, cfg.algorithm.agent)
    M = cfg.algorithm.agent.layer_size
    num_hidden_layer = cfg.algorithm.agent.num_hidden_layer
    num_qfs = cfg.algorithm.agent.num_qfs

    """
    Create Q functions and their target functions.
    """
    def fanin_init(m: nn.Module, b_init_value=0.0):
        if isinstance(m, EnsembleFCLayer):
            for i in range(m.ensemble_size):
                ptu.fanin_init(m.weight[i])
                m.bias[i].data.fill_(b_init_value)
        
    qfs, target_qfs = ppp.group_init(
        2,
        FlattenEnsembleMLP,
        ensemble_size=num_qfs,
        input_size=input_dim,
        output_size=1,
        hidden_sizes=[M] * num_hidden_layer,
        hidden_activation=torch.relu,
        b_init_value=0.1,
        w_init_method=fanin_init,
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
    agent: SACAgent
    agent = hydra.utils.instantiate(cfg.algorithm.agent,
                                    policy=policy,
                                    f_checkpoint=f_checkpoint,
                                    load_param=eval_only,   # Load learned parameters if only used for evaluation
                                    _recursive_=False,
                                    )

    """
    Set up the trainer of SAC
    """
    trainer = SACTrainer(
        env=eval_env,
        agent=agent,
        qfs=qfs,
        target_qfs=target_qfs,
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
        agent: SACAgent,
        trainer: TorchTrainer,
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
