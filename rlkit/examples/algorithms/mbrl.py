import hydra
import omegaconf
from omegaconf import OmegaConf

import numpy as np
import torch
import copy
from os import path as osp
from typing import Optional
from rlkit.core.logging.logging import logger

import rlkit.torch.pytorch_util as ptu
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
    Model-based reinforcement learning (MBRL) dynamics model
    """
    dynamics_model = create_dynamics_and_reward_model(
        cfg=cfg,
        obs_dim=obs_dim,
        input_dim=input_dim,
        action_dim=action_dim,
        target_dim=target_dim,
    )

    trainer = MBRLTrainer(
        dynamics_model=dynamics_model,
        obs_preproc=obs_preproc,
        targ_proc=targ_proc,
        **cfg.algorithm.dynamics,
    )

    """
    Set up behavior cloning trainer (used for MBOP)
    """
    behavior_clone, behavior_clone_trainer = None, None
    if cfg.overrides.get('use_behavior_clone', False):
        bc_cfg = cfg.overrides.prior_model
        hidden_activation = hydra.utils.get_method(bc_cfg.get('activation_func')) \
            if bc_cfg.get('activation_func', False) else torch.relu

        behavior_clone = hydra.utils.instantiate(
            bc_cfg,
            hidden_sizes=[bc_cfg.layer_size] * bc_cfg.num_hidden_layer,
            input_size=input_dim if bc_cfg.include_prev_action_as_input else input_dim - action_dim,
            output_size=action_dim,
            hidden_activation=hidden_activation,
        )

        # Set up the trainer
        behavior_clone_trainer = BehaviorCloneTrainer(
            model=behavior_clone,
            obs_dim=obs_dim,
            action_dim=action_dim,
            obs_preproc=obs_preproc,
            **bc_cfg,
        )

    """
    Set up the value function and its trainer (set to None if not used)
    """
    value_func, pol_eval_trainer = None, None
    if cfg.algorithm.use_value_func:
        value_func_cfg = cfg.overrides.value_func_cfg
        M = value_func_cfg.get('layer_size', cfg.algorithm.dynamics.ensemble_model.layer_size)
        num_hidden_layer = value_func_cfg.get('num_hidden_layer',
                                            cfg.algorithm.dynamics.ensemble_model.num_hidden_layer)
        hidden_activation = hydra.utils.get_method(value_func_cfg.get('activation_func')) \
            if value_func_cfg.get('activation_func', False) else torch.relu

        value_func = hydra.utils.instantiate(
            cfg.overrides.value_func_cfg,
            input_size=input_dim if value_func_cfg.get('is_qf') else input_dim - action_dim,
            hidden_sizes=[M] * num_hidden_layer,
            hidden_activation=hidden_activation,
        )

        target_value_func = copy.deepcopy(value_func) if value_func_cfg.get('use_target_network', False) else None

        pol_eval_trainer = PolEvalTrainer(
            value_func=value_func,
            target_value_func=target_value_func,
            obs_preproc=obs_preproc,
            **value_func_cfg,
        )

    """
    Store all modules to a dictionary
    """
    cfg_dict = dict(
        trainer=trainer,
        model_trainer=trainer,
        behavior_clone_trainer=behavior_clone_trainer,
        pol_eval_trainer=pol_eval_trainer,
    )
    return cfg_dict

def prepare_generic_rl_algorithm(
        cfg: omegaconf.DictConfig,
        replay_buffer: EnvReplayBuffer,
        model_trainer: MBRLTrainer,
        behavior_clone_trainer: Optional[BehaviorCloneTrainer] = None,
        pol_eval_trainer: Optional[PolEvalTrainer] = None,
        **kwargs):

    assert cfg.is_offline, "Pretraining models can only be done for offline RL environments"
    import gtimer as gt
    import os

    # Update the configuration when doing the offline RL
    algorithm_cfg = cfg.algorithm.algorithm_cfg
    num_model_learning_epochs = algorithm_cfg.num_model_learning_epochs
    model_max_grad_steps = algorithm_cfg.get('model_max_grad_steps', int(1e7))
    max_epochs_since_last_update = algorithm_cfg.max_epochs_since_last_update
    use_best_parameters = algorithm_cfg.use_best_parameters

    """Define the training procedure"""
    class MBRL:
        def __init__(self):
            self.model_trainer = model_trainer
            self.behavior_clone_trainer = behavior_clone_trainer
            self.pol_eval_trainer = pol_eval_trainer

        def train(self):
            output_csv = logger.get_tabular_output()

            # Prepare directory and file name to which parameters will be saved
            cwd = hydra.utils.get_original_cwd()
            if cfg.overrides.get('d4rl_config', None) is not None:
                f_dir = osp.join(cwd, f"data/{cfg.env.name.lower()}/{'-'.join(cfg.overrides.save_dir.split('-')[:-1])}")
            else:
                raise ValueError("Unrecognized task information provided! Cannot train a model..exiting...")

            if cfg.max_size is not None:
                f_dir = osp.join(f_dir, f"{int(cfg.max_size)}")
            os.makedirs(f_dir, exist_ok=True)

            if behavior_clone_trainer is not None:
                logger.set_tabular_output(osp.join(logger.log_dir, 'behavior_clone_learning.csv'))
                bc_cfg = cfg.overrides.prior_model
                max_behavior_clone_gradient_steps = bc_cfg.get('max_behavior_clone_gradient_steps', int(1e7))
                num_behavior_clone_learning_epochs = bc_cfg.get('num_behavior_clone_learning_epochs', 10)

                behavior_clone_trainer.train_from_buffer(
                    replay_buffer=replay_buffer,
                    max_grad_steps=max_behavior_clone_gradient_steps,
                    max_epochs_since_last_update=max_epochs_since_last_update,
                    num_total_epochs=num_behavior_clone_learning_epochs,
                    use_best_parameters=use_best_parameters,
                )
                f_bc_name = osp.join(f_dir, 'behavior_clone_checkpoint.pth')
                torch.save(behavior_clone_trainer.get_snapshot(), f_bc_name)
                gt.stamp('Behavior clone training', unique=False)

            if pol_eval_trainer is not None:
                logger.set_tabular_output(osp.join(logger.log_dir, 'pol_eval_learning.csv'))
                num_value_learning_epochs = pol_eval_trainer.kwargs.get('num_value_learning_epochs', 40)
                max_value_learning_gradient_steps = algorithm_cfg.get('max_value_learning_gradient_steps', int(1e7))
                num_repeat = pol_eval_trainer.kwargs.get('num_value_learning_repeat', 10)
                pol_eval_trainer.train_from_buffer(
                    replay_buffer=replay_buffer,
                    max_grad_steps=max_value_learning_gradient_steps,
                    max_epochs_since_last_update=max_epochs_since_last_update,
                    num_total_epochs=num_value_learning_epochs,
                    use_best_parameters=use_best_parameters,
                    num_repeat=num_repeat,
                )
                f_vf_name = osp.join(f_dir, 'policy_eval_checkpoint.pth')
                torch.save(pol_eval_trainer.get_snapshot(), f_vf_name)
                gt.stamp('Policy evaluation', unique=False)

            assert model_trainer is not None
            logger.set_tabular_output(osp.join(logger.log_dir, 'model_learning.csv'))

            model_trainer.train_from_buffer(
                replay_buffer=replay_buffer,
                max_grad_steps=model_max_grad_steps,
                max_epochs_since_last_update=max_epochs_since_last_update,
                num_total_epochs=num_model_learning_epochs,
                use_best_parameters=use_best_parameters,
            )
            #       combined   /   normalize output   /   learn reward
            #   v1     o                  x                    o
            #   v2     o       /          o                    o
            #   v3     x       /          x                    o
            #   v4     x       /          o                    o
            #   v5     x       /          x                    x
            #   v6     x       /          o                    x
            if cfg.algorithm.dynamics.learn_reward:
                if cfg.overrides.dynamics.separate_reward_func and cfg.overrides.dynamics.normalize_outputs:
                    f_m_name = 'model_checkpoint_v4.pth'
                elif cfg.overrides.dynamics.separate_reward_func:
                    f_m_name = 'model_checkpoint_v3.pth'
                elif cfg.overrides.dynamics.normalize_outputs:
                    f_m_name = 'model_checkpoint_v2.pth'
                else:
                    f_m_name = 'model_checkpoint_v1.pth'
            else:
                if not cfg.overrides.dynamics.normalize_outputs:
                    f_m_name = 'model_checkpoint_v5.pth'
                else:
                    f_m_name = 'model_checkpoint_v6.pth'

            f_m_name = osp.join(f_dir, f_m_name)
            torch.save(model_trainer.get_snapshot(), f_m_name)
            gt.stamp('Model training', unique=False)

            logger.set_tabular_output(output_csv)

        def to(self, device):
            for net in self.model_trainer.networks:
                net.to(device)
            if self.pol_eval_trainer is not None:
                for net in self.pol_eval_trainer.networks:
                    net.to(device)
            if self.behavior_clone_trainer is not None:
                for net in self.behavior_clone_trainer.networks:
                    net.to(device)

        def configure_logging(self, **kwargs):
            self.model_trainer.configure_logging(**kwargs)
            if self.pol_eval_trainer is not None:
                self.pol_eval_trainer.configure_logging(**kwargs)
            if self.behavior_clone_trainer is not None:
                self.behavior_clone_trainer.configure_logging(**kwargs)

    algorithm = MBRL()
    return algorithm
