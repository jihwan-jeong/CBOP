from typing import Union
import omegaconf, hydra
import gym, torch
from .base.base import Policy
from pathlib import Path


def complete_agent_cfg(
    env, agent_cfg: omegaconf.DictConfig,
):
    """
    Completes an agent's configuration given information from the environment.

    The goal of this function is to complete the information about state and action shapes and ranges,
    without requiring the user to manually enter this into the Omegaconf configuration object.
    It will check for and complete any of the following keys:
       - "obs_dim": set to env.observation_space.shape
       - "action_dim": set to env.action_space.shape
    Note:
       If the user provides any of these values in the Omegaconf configuration object, these
       *will not* be overridden by this function.
    """
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    agent_cfg.obs_dim = obs_shape[0]
    agent_cfg.action_dim = act_shape[0]
    return agent_cfg


# TODO: How to load model-free agents?
def load_agent(agent_path: Union[str, Path], env: gym.Env) -> Policy:
    """Loads an agent from a Hydra config file at the given path.
    For agent of type "actor_critic.ActorCriticPolicy", the directory
    must contain the following files:
        - ".hydra/config.yaml": the Hydra configuration for the agent.
        - "critic.pth": the saved checkpoint for the critic.
        - "actor.pth": the saved checkpoint for the actor.
    Args:
        agent_path (str or Path): a path to the directory where the agent is saved.
        env (gym.Env): the environment on which the agent will operate (only used to complete
            the agent's configuration).
    Returns:
        (Policy): the new agent.
    """
    agent_path = Path(agent_path)
    cfg = omegaconf.OmegaConf.load(agent_path / ".hydra" / "config.yaml")

    if (
        cfg.algorithm.agent._target_
        == "rlkit.policies.actor_critic.ActorCriticPolicy"
    ):
        import rlkit.policies.actor_critic as pytorch_sac

        from .sac_wrapper import SACAgent

        complete_agent_cfg(env, cfg.algorithm.agent)
        agent: pytorch_sac.SACAgent = hydra.utils.instantiate(cfg.algorithm.agent)
        agent.critic.load_state_dict(torch.load(agent_path / "critic.pth"))
        agent.actor.load_state_dict(torch.load(agent_path / "actor.pth"))
        return SACAgent(agent)
    else:
        raise ValueError("Invalid agent configuration.")