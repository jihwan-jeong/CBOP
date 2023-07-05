import numpy as np
from typing import Optional, cast
from rlkit.policies.base.base import Policy
import rlkit.torch.pytorch_util as ptu
import torch.nn as nn
import copy


def rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        preprocess_obs_for_policy_fn=None,
        get_action_kwargs=None,
        return_dict_obs=False,
        full_o_postprocess_func=None,
        reset_callback=None,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    if preprocess_obs_for_policy_fn is None:
        preprocess_obs_for_policy_fn = lambda x: x
    raw_obs = []
    raw_next_obs = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    next_observations = []
    path_length = 0
    agent.reset()
    o = env.reset()
    if reset_callback:
        reset_callback(env, agent, o)
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        raw_obs.append(o)
        o_for_agent = preprocess_obs_for_policy_fn(o)
        a, agent_info = agent.get_action(o_for_agent, **get_action_kwargs)

        if full_o_postprocess_func:
            full_o_postprocess_func(env, agent, o)

        next_o, r, d, env_info = env.step(copy.deepcopy(a))
        if render:
            env.render(**render_kwargs)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        next_observations.append(next_o)
        raw_next_obs.append(next_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    if return_dict_obs:
        observations = raw_obs
        next_observations = raw_next_obs
    rewards = np.array(rewards)
    if len(rewards.shape) == 1:
        rewards = rewards.reshape(-1, 1)
    return dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        full_observations=raw_obs,
        full_next_observations=raw_obs,
    )

def rollout_plan_mujoco_env(
        env,
        initial_obs: np.ndarray,
        lookahead: int,
        agent: Optional[Policy] = None,
        plan: Optional[np.ndarray] = None,
        behavior_clone: Optional[nn.Module] = None,
):
    """Runs the environment for some number of steps then returns it to its original state.
        Works with mujoco gym and dm_control environments
        (with `dmc2gym <https://github.com/denisyarats/dmc2gym>`_).
        Args:
            env             (:class:`gym.wrappers.TimeLimit`): the environment.
            initial_obs     (np.ndarray): the latest observation returned by the environment (only
                        needed when ``agent is not None``, to get the first action).
            lookahead       (int): the number of steps to run. If ``plan is not None``,
                        it is overridden by `len(plan)`.
            agent           (:class:`rlkit.policies.base.Policy`, optional): if given, an agent to obtain actions.
            plan            (sequence of TensorType, optional): if given, a sequence of actions to execute.
                        Takes precedence over ``agent`` when both are given.
            behavior_clone (Callable): If provided, given `plan` will be treated as a random noise, while `behavior_clone`
                            provides the mean of the plan.

        Returns:
            (tuple of np.ndarray): the observations, rewards, and actions observed, respectively.
    """
    from rlkit.envs.mujoco_env import freeze_mujoco_env
    import gym.wrappers
    import torch

    actions = []
    real_obses = []
    rewards = []

    # Initialize some variables used when `behavior_clone` is not None
    past_action = ptu.zeros(env.action_space.low.size)
    upper_bound = torch.from_numpy(env.action_space.high).to('cpu')
    lower_bound = torch.from_numpy(env.action_space.low).to('cpu')

    # Use a frozen environment for rollouts given plan or agent (and/or behavior_clone)
    with freeze_mujoco_env(cast(gym.wrappers.TimeLimit, env)):
        current_obs = initial_obs.copy()
        real_obses.append(current_obs)
        if plan is not None:
            lookahead = len(plan)
        for i in range(lookahead):
            # Get action
            if plan is not None:
                a = plan[i]
                # Handle behavior_clone:
                if behavior_clone is not None:
                    with torch.no_grad():
                        obs_tensor = ptu.from_numpy(current_obs).view(1, -1)
                        input_to_bc = torch.cat([obs_tensor, past_action.view(1, -1)], dim=-1) \
                            if behavior_clone.include_prev_action_as_input \
                            else obs_tensor
                        input_to_bc = input_to_bc.expand(behavior_clone.ensemble_size, -1, -1)
                        assert hasattr(behavior_clone, 'set_output_transforms')
                        behavior_clone.set_output_transforms(True)
                        a_bc = behavior_clone(input_to_bc, use_propagation=False)[0].mean(dim=0).view(plan[i].size())
                        behavior_clone.set_output_transforms(False)
                        plan[i] += a_bc.cpu()              # in-place modification of sampled actions

                        # Clip actions to reside between bounds (in-place operation)
                        plan[i] = torch.where(
                            plan[i] > upper_bound, upper_bound, plan[i]
                        )
                        plan[i] = torch.where(
                            plan[i] < lower_bound, lower_bound, plan[i]
                        )

                        a = plan[i]
                        past_action = a.clone().to(ptu.device)
            else:
                a = agent.get_action(current_obs)

            if isinstance(a, torch.Tensor):
                a = ptu.get_numpy(a)
            next_obs, reward, done, _ = env.step(a)
            actions.append(a)
            real_obses.append(next_obs)
            rewards.append(reward)
            if done:
                break
            current_obs = next_obs

        # Incorporate the value function if specified (optional)

    return np.stack(real_obses), np.stack(rewards), np.stack(actions)

def function_rollout(
        env,
        agent_fn,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    env_infos = []
    o = env.reset()
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        a = agent_fn(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        env_infos=env_infos,
    )
