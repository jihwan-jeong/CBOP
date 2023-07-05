from collections import OrderedDict
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.replay_buffers.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.model_env import ModelEnv
from rlkit.policies.base.base import Policy
from rlkit.util import eval_util

"""
Interface for generating rollouts from a dynamics model. Designed to support many
types of rollouts, and measuring different metrics. Three main types of rollouts
are (planned to be) supported, depending on GPU memory and other use cases:
    1) (standard_)rollout: keep everything in GPU memory, return all paths
    2) online_rollout: keep everything in GPU memory, but only store current
                       transitions, so only return final/cumulative info in
                       the torch format
    3) np_rollout: only use GPU as necessary, store everything in numpy, return
                   all paths (which are in numpy)
"""


def _create_full_tensors(init_obs, max_path_length, obs_dim, action_dim):
    num_rollouts = init_obs.shape[0]
    observations = ptu.zeros((num_rollouts, max_path_length + 1, obs_dim))
    observations[:, 0] = ptu.from_numpy(init_obs)
    actions = ptu.zeros((num_rollouts, max_path_length, action_dim))
    rewards = ptu.zeros((num_rollouts, max_path_length, 1))
    terminals = ptu.zeros((num_rollouts, max_path_length, 1), dtype=torch.bool)
    return observations, actions, rewards, terminals


def _sample_from_model(dynamics_model, state_actions, t):
    return dynamics_model.sample(state_actions)


def _get_prediction(sample_from_model, dynamics_model, states, actions, t, terminal_cutoff=0.5):
    state_actions = torch.cat([states, actions], dim=-1)
    transitions = sample_from_model(dynamics_model, state_actions, t)
    if (transitions != transitions).any():
        print('WARNING: NaN TRANSITIONS IN DYNAMICS MODEL ROLLOUT')
        transitions[transitions != transitions] = 0

    rewards = transitions[:,:1]
    dones = (transitions[:,1:2] > terminal_cutoff).float()
    delta_obs = transitions[:,2:]

    return rewards, dones, delta_obs


def _create_paths(obs, act, rew, dones, horizon):
    obs_np = ptu.get_numpy(obs)
    act_np = ptu.get_numpy(act)
    rew_np = ptu.get_numpy(rew)
    dones_np = ptu.get_numpy(dones)
    assert all(map(lambda x: x.ndim == 3, [obs, act, rew, dones]))

    paths = [None] * len(obs_np)
    for i in range(len(obs_np)):
        rollout_len = 1
        while rollout_len < horizon and dones[i, rollout_len - 1, 0] < 0.5:  # just check 0 or 1
            rollout_len += 1
        paths[i] = dict(
            observations=obs_np[i, :rollout_len],
            actions=act_np[i, :rollout_len],
            rewards=rew_np[i, :rollout_len],
            next_observations=obs_np[i, 1:rollout_len + 1],
            terminals=dones_np[i, :rollout_len],
            agent_infos=[[] for _ in range(rollout_len)],
            env_infos=[[] for _ in range(rollout_len)],
        )
    return paths


"""
Methods for generating actions from states
"""


def _get_policy_actions(states, t, action_kwargs):
    policy = action_kwargs['policy']
    actions, *_ = policy.forward(states)
    return actions


def _get_policy_latent_actions(states, t, action_kwargs):
    latents = action_kwargs['latents']
    state_latents = torch.cat([states, latents], dim=-1)
    return _get_policy_actions(state_latents, t, action_kwargs)


def _get_policy_latent_prior_actions(states, t, action_kwargs):
    latent_prior = action_kwargs['latent_prior']
    latents, *_ = latent_prior(states)
    state_latents = torch.cat([states, latents], dim=-1)
    return _get_policy_actions(state_latents, t, action_kwargs)


def _get_open_loop_actions(states, t, action_kwargs):
    actions = action_kwargs['actions']
    return actions[:,t]


"""
Base classes for doing rollout work
TODO: support recurrent dynamics models?
"""


def _model_rollout(
        dynamics_model,                             # torch dynamics model: (s, a) --> (r, d, s')
        start_states,                               # numpy array of states: (num_rollouts, obs_dim)
        get_action,                                 # method for getting action
        action_kwargs=None,                         # kwargs for get_action (ex. policy or actions)
        max_path_length=1000,                       # maximum rollout length (if not terminated)
        terminal_cutoff=0.5,                        # output Done if model pred > terminal_cutoff
        create_full_tensors=_create_full_tensors,
        sample_from_model=_sample_from_model,
        get_prediction=_get_prediction,
        create_paths=_create_paths,
        *args,
        **kwargs,
):
    if action_kwargs is None:
        action_kwargs = dict()
    if terminal_cutoff is None:
        terminal_cutoff = 1e6
    if max_path_length is None:
        raise ValueError('Must specify max_path_length in rollout function')

    obs_dim = dynamics_model.input_dim
    action_dim = dynamics_model.action_dim

    s, a, r, d = create_full_tensors(start_states, max_path_length, obs_dim, action_dim)
    for t in range(max_path_length):
        a[:,t] = get_action(s[:,t], t, action_kwargs)
        r[:,t], d[:,t], delta_t = get_prediction(
            sample_from_model,
            dynamics_model,
            s[:,t], a[:,t], t,
            terminal_cutoff=terminal_cutoff,
        )
        s[:,t+1] = s[:,t] + delta_t

    paths = create_paths(s, a, r, d, max_path_length)

    return paths


# TODO: _model_online_rollout


# TODO: _model_np_rollout


"""
Interface for rollout functions for other classes to use
"""


def policy(dynamics_model, policy, start_states, **kwargs):
    return _model_rollout(
        dynamics_model,
        start_states,
        _get_policy_actions,
        action_kwargs=dict(policy=policy),
        **kwargs,
    )


def open_loop_actions(dynamics_model, actions, start_states, **kwargs):
    return _model_rollout(
        dynamics_model,
        start_states,
        _get_open_loop_actions,
        action_kwargs=dict(actions=actions),
        **kwargs,
    )


def policy_latent(dynamics_model, policy, start_states, latents, **kwargs):
    return _model_rollout(
        dynamics_model,
        start_states,
        _get_policy_latent_actions,
        action_kwargs=dict(policy=policy, latents=latents),
        **kwargs,
    )


def policy_latent_prior(dynamics_model, policy, latent_prior, start_states, **kwargs):
    return _model_rollout(
        dynamics_model,
        start_states,
        _get_policy_latent_prior_actions,
        action_kwargs=dict(policy=policy, latent_prior=latent_prior),
        **kwargs,
    )


def _rollout_with_disagreement(base_rollout_func, *args, **kwargs):
    disagreement_type = kwargs.get('disagreement_type', 'mean')

    disagreements = []

    def sample_with_disagreement(dynamics_model, state_actions, t):
        # note that disagreement has shape (num_rollouts, 1), e.g. it is unsqueezed
        transitions, disagreement = dynamics_model.sample_with_disagreement(
            state_actions, disagreement_type=disagreement_type)
        disagreements.append(disagreement)
        return transitions

    paths = base_rollout_func(sample_from_model=sample_with_disagreement, *args, **kwargs)
    disagreements = torch.cat(disagreements, dim=-1)

    return paths, disagreements


def policy_with_disagreement(*args, **kwargs):
    return _rollout_with_disagreement(policy, *args, **kwargs)


def policy_latent_with_disagreement(*args, **kwargs):
    return _rollout_with_disagreement(policy_latent, *args, **kwargs)


def policy_latent_prior_with_disagreement(*args, **kwargs):
    return _rollout_with_disagreement(policy_latent_prior, *args, **kwargs)


def open_loop_with_disagreement(*args, **kwargs):
    return _rollout_with_disagreement(open_loop_actions, *args, **kwargs)


def rollout_model_and_get_paths(
    model_env: ModelEnv,
    replay_buffer: EnvReplayBuffer,
    total_size: int,
    agent: Policy,
    deterministic: bool,
    rollout_horizon: int,
    diagnostics: OrderedDict,
    initial_obs: Optional[np.ndarray] = None,
    discount: float = 1.0,
) -> List[Dict]:
    """
    Rollout in the model environment starting from ``start_states``. If ``start_states`` is None, d

    Args:
        model_env (ModelEnv): The environment defined by the learned dynamics model
        replay_buffer (EnvReplayBuffer): The replay buffer with true transitions
        total_size (int): The total number of rollouts to be sampled
        agent (Policy): The policy used to select actions
        deterministic (bool): Whether next state transition is deterministic or stochastic
        rollout_horizon (int): How long into the future do we want to look ahead using the model
        diagnostics (OrderedDict): Any diagnostic information can be saved
        initial_obs (np.ndarray, optional): States from which rollouts begin 
        discount (float): The discount factor to be used for return estimation (default at 1)

    Returns:
        paths (list of dict)
    """
    # Set the initial states to begin rollouts
    if initial_obs is None:
        initial_obs = replay_buffer.random_batch(total_size)['observations']
    else:
        assert initial_obs.shape[0] == total_size
    assert initial_obs.ndim == 2
    obs_dim, action_dim = initial_obs.shape[1], model_env.action_space.low.size

    initial_obs = model_env.reset(initial_obs, return_as_np=True)

    # Pre-allocate memory for rollouts
    s, a, r, d = _create_full_tensors(
        init_obs=initial_obs,
        max_path_length=rollout_horizon,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    # Rollout the dynamics model for ``rollout_horizon`` steps (actions chosen by given policy)
    batch_size = 256
    num_batches = int(np.ceil(total_size / batch_size))
    for t in range(rollout_horizon):
        act_t = []
        for b in range(num_batches):
            act_t.append(agent.get_actions(s[b * batch_size: (b+1) * batch_size, t], deterministic=deterministic))
        act_t = np.concatenate(act_t, axis=0)
        a[:, t] = ptu.from_numpy(act_t)
        s[:, t+1], r[:, t], d[:, t] = \
            tuple(
                map(lambda x: ptu.from_numpy(x), model_env.step(act_t,
                                                                sample=not deterministic,
                                                                diagnostics=diagnostics)[:3])
            )
        # Once terminated, should remain terminated
        d[:, t] |= d[:, t-1]
        nd = ~d[:, t]
        if nd.clone().type(torch.float).sum() == 0:
            break

    paths = _create_paths(s, a, r, d, rollout_horizon)
    diagnostics.update(
        eval_util.get_generic_path_information(paths)
    )
    return paths

def rollout_model_env(
        model_env: ModelEnv,
        initial_obs: Optional[np.ndarray] = None,
        plan: Optional[np.ndarray] = None,
        agent: Optional[Policy] = None,
        num_samples: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rolls out an environment model.

    Executes a plan on a dynamics model.

     Args:
         model_env      (:class:`rlkit.envs.model_env.ModelEnv`): the dynamics model environment to simulate.
         initial_obs    (np.ndarray, optional): initial observation to start episodes (a single observation)
         plan           (np.ndarray, optional): sequence of actions to execute.
         agent          (:class:`rlkit.policies.Policy`): an agent to generate a plan before
                    execution starts (as in `agent.plan(initial_obs)`). If given, takes precedence over ``plan``.
         num_samples    (int): how many samples to take from the model (i.e., independent rollouts).
                    Defaults to 1.

    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals
    """
    observations = []
    rewards = []
    if agent:
        plan = agent.plan(initial_obs)
    o = model_env.reset(initial_obs_batch=np.tile(initial_obs, [num_samples, 1]))
    observations.append(o)

    for action in plan:
        next_o, r, d, _ = model_env.step(
            np.tile(action, [num_samples, 1]), sample=False
        )
        observations.append(next_o)
        rewards.append(r)

    if len(plan.shape) == 1:
        plan = np.expand_dims(plan, 1)
    observations = np.stack(observations)
    rewards = np.stack(rewards)
    return observations, rewards, plan