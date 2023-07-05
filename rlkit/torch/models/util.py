from typing import Optional, Union
from pathlib import Path
import hydra, omegaconf
from omegaconf import OmegaConf
import rlkit.torch.pytorch_util as ptu
import torch

def create_dynamics_model(
        cfg: omegaconf.DictConfig,
        input_dim: int,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: list,
        model_dir: Optional[Union[str, Path]] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
):
    """Creates a dynamics model (optionally including the reward model) from a given configuration.
    Original code from 'facebookresearch/mbrl-lib'

    This function creates a new model from the given configuration

    The configuration should be structured as follows::
        - cfg
            - algorithm
                - dynamics
                    - _target_ (str): model Python class
                    - model_arg_1
                        ...
                    - model_arg_n
                    - learn_reward (bool): whether rewards should be learned or not
                    - learn_logstd_min_max (bool): whether the bounds of logstd should be learned

    The model will be instantiated using :func:`hydra.utils.instantiate` function.

    Args:
        cfg (omegaconf.DictConfig): the configuration to read.
        input_dim (int)
        obs_dim (int)
        action_dim (int)
        hidden_sizes (list)
        model_dir (str or Path): If provided, the model will attempt to load its
            weights and normalization information from "model_dir / model.pth" and
            "model_dir / env_stats.pickle", respectively.

    Returns:
        (:class:`rlkit.torch.models.probabilistic_ensemble.ProbabilisticEnsemble`): the model created.
    """
    kwargs = OmegaConf.to_container(cfg, resolve=True)

    # Instantiate the model
    model = hydra.utils.instantiate(
        cfg,
        input_dim=input_dim,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
        hidden_activation=hydra.utils.get_method(cfg.get('activation_func', 'torch.relu')),
        **kwargs
    )
    model.to(ptu.device)

    # Load model parameters if the file path is provided
    if model_dir:
        model.load(model_dir)
    elif checkpoint_dir:
        checkpoint = torch.load(str(checkpoint_dir), map_location=ptu.device)
        model.load_state_dict(checkpoint['model_state_dict'])

    return model


def swish(x):
    return x * torch.sigmoid(x)


def identity(x):
    return x