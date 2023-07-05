from pathlib import Path
from typing import Union, Optional
import hydra
import omegaconf
from omegaconf import OmegaConf
import gtimer as gt
from collections import OrderedDict
from rlkit.core.logging.logging import logger
import torch.nn as nn

from rlkit.torch.models.ensemble import EnsembleFCLayer, FlattenEnsembleMLP, truncated_normal_init
import rlkit.torch.pytorch_util as ptu

LOG_STD_MAX = 2
LOG_STD_MIN = -5


def load_hydra_cfg(results_dir: Union[str, Path]) -> omegaconf.DictConfig:
    """
    Loads a Hydra configuration from the given directory path.
    Tries to load the configuration from "results_dir/.hydra/config.yaml".

    Args:
        results_dir (str or Path): the path to the directory containing the config.

    Returns:
        (omegaconf.DictConfig): the loaded configuration.
    """
    results_dir = Path(results_dir)
    cfg_file = results_dir / ".hydra" / "config.yaml"
    cfg = omegaconf.OmegaConf.load(cfg_file)
    if not isinstance(cfg, omegaconf.DictConfig):
        raise RuntimeError("Configuration format not a omegaconf.DictConf")
    return cfg


def assert_arguments_not_none(*args):
    for i, item in enumerate(args):
        assert item is not None, f"The {i}th argument is None! Double-check proper arguments are passed"


def get_epoch_timings():
    times_itrs = gt.get_times().stamps.itrs
    times = OrderedDict()
    epoch_time = 0
    for key in sorted(times_itrs):
        time = times_itrs[key][-1]
        epoch_time += time
        times['time/{} (s)'.format(key)] = time
    times['time/epoch (s)'] = epoch_time
    times['time/total (s)'] = gt.get_times().total
    return times
    

def create_dynamics_and_reward_model(
        cfg: omegaconf.DictConfig,
        obs_dim: int,
        input_dim: int,
        action_dim: int,
        target_dim: int,
        cache_dir=None,
):
    """
    Creates a transition dynamics model of an environment given the .yaml configuration file.
    Note that an ensemble model (either probabilistic or deterministic) will be instantiated, followed by the
    instantiation of the wrapper class :class:`rlkit.torch.models.dynamics_models.model.DynamicsModel`. The ensemble
    model is agnostic to an RL environment, while the wrapper class handles all necessary operations to be compatible
    with the RL environment.

    Args:
        cfg (DictConfig):          The main configuration file
        obs_dim (int):             The dimensionality of the observation space of the environment
        input_dim (int):           The input dimensionality to the dynamics model, i.e., (s, a)
        action_dim (int):          The action dimension of the environment
        target_dim (int):
        cache_dir (str or PosixPath, Optional)

    """
    model_cfg = cfg.algorithm.dynamics
    from rlkit.torch.models.dynamics_models.model import DynamicsModel
    model_cls = DynamicsModel
    
    M = model_cfg.ensemble_model.layer_size
    num_layer = model_cfg.ensemble_model.num_hidden_layer
    hidden_sizes = [M] * num_layer
    hidden_activation = hydra.utils.get_method(model_cfg.ensemble_model.get('activation_func', 'torch.relu'))

    is_probabilistic = 'probabilistic' in str(model_cfg.ensemble_model._target_).lower()
    separate_reward_func = model_cfg.separate_reward_func and model_cfg.learn_reward

    if is_probabilistic:
        if model_cfg.learn_reward and not separate_reward_func:
            output_dim = target_dim * 2 + 2
        else:
            output_dim = target_dim * 2
    else:
        if model_cfg.learn_reward and not separate_reward_func:
            output_dim = target_dim + 1
        else:
            output_dim = target_dim

    # Weight initialization method
    w_init_method_name = cfg.get('w_init_method', None)
    if w_init_method_name == 'uniform':
        def w_init_method(m: nn.Module, b_init_value: float = 0.1):
            if isinstance(m, nn.Linear) or isinstance(m, EnsembleFCLayer):
                ptu.uniform_init(m.weight, init_w=1)
                m.bias.data.fill_(b_init_value)
    elif w_init_method_name == 'truncated_normal':
        w_init_method = truncated_normal_init
    else:
        def fanin_init(m: nn.Module, b_init_value=0.0):
            if isinstance(m, EnsembleFCLayer):
                for i in range(m.ensemble_size):
                    ptu.fanin_init(m.weight[i])
                    m.bias[i].data.fill_(b_init_value)
        w_init_method = fanin_init

    ensemble = hydra.utils.instantiate(
        model_cfg.ensemble_model,
        hidden_sizes=hidden_sizes,
        input_size=input_dim,
        output_size=output_dim,
        learn_logstd_min_max=model_cfg.learn_logstd_min_max,
        hidden_activation=hidden_activation,
        b_init_value=0.1,
        w_init_method=w_init_method
    )

    # If needed, instantiate a separate ensemble network for the reward function
    if separate_reward_func:
        rew_ensemble = hydra.utils.instantiate(
            model_cfg.ensemble_model,
            hidden_sizes=hidden_sizes,
            input_size=input_dim,
            output_size=2 if is_probabilistic else 1,
            learn_logstd_min_max=model_cfg.learn_logstd_min_max,
            hidden_activation=hidden_activation,
            propagation_method=model_cfg.get("reward_propagation_method"),
            b_init_value=0.1,
            w_init_method=w_init_method,
        )
    else:
        rew_ensemble = None

    dynamics_model = model_cls(
        model=ensemble,
        input_dim=input_dim,
        obs_dim=obs_dim,
        action_dim=action_dim,
        is_probabilistic=is_probabilistic,
        **model_cfg,
    )
    dynamics_model.set_reward_func(rew_ensemble)

    # Load learned parameters
    if cache_dir is not None:
        f_name = f'dynamics_model_{dynamics_model.config}_checkpoint.pth'

        if not cache_dir.is_dir():
            logger.log(f'cache_dir is provided in config, but it is not a valid directory!\ncache_dir: {str(cache_dir)}')
        elif (cache_dir / f_name).is_file():
            f_name = str(cache_dir / f_name)
            logger.log(f"Found {f_name}. Loading model parameters...")
            dynamics_model.load(model_path=f_name)
            if dynamics_model.trained:
                logger.log("Model parameters successfully loaded! (model training step will be skipped)")
            else:
                logger.log("Failed to load model parameters... Check configuration!")
                raise ValueError("Failed to load model parameters... Check configuration!")
        else:
            logger.log(f"[Dynamics] Cannot find {f_name} in {str(cache_dir)}; no pretrained parameters loaded")
    return dynamics_model
