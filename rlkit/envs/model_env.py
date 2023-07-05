"""
This file implements ModelEnv class which uses a learned dynamics model and a (learned) reward model
to sample in the *imaginary* environment.

Adapted from the code in facebookresearch/mbrl-lib/mbrl/models/model_env.py
"""

from typing import Dict, Optional, Tuple, Callable
import omegaconf

import numpy as np
import torch
import torch.nn as nn
from rlkit.torch.distributions import Distribution
from rlkit.torch.models.ensemble import Ensemble

import rlkit.types
import rlkit.envs
from rlkit.envs.wrappers import ProxyEnv

from rlkit.util.eval_util import create_stats_ordered_dict
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.models.probabilistic_ensemble import ProbabilisticEnsemble
from rlkit.torch.models.dynamics_models.model import DynamicsModel, LatentDynamicsModel


class ModelEnv:
    """Wraps a dynamics model into a gym-like environment.

    This class can wrap a dynamics model to be used as an environment.

    The only requirement
    to use this class is for the model to use this wrapper is to have a method called
    ``predict()``
    with signature `next_observs, rewards = model.predict(obs,actions, sample=, rng=)`

    Args:
        env (ProxyEnv): the original gym environment for which the model was trained.
        model (:class:`rlkit.torch.models.dynamics_models.probabilistic_ensemble.ProbabilisticEnsemble`): the model to wrap.
        termination_func (callable): a function that receives actions and observations, and
            returns a boolean flag indicating whether the episode should end or not.
        reward_func (callable, optional): a function that receives actions and observations
            and returns the value of the resulting reward in the environment.
            Defaults to ``None``, in which case predicted rewards will be used.
        generator (torch.Generator, optional): a torch random number generator (must be in the
            same device as the given model). If None (default value), a new generator will be
            created using the default torch seed.
        env_proc (:class:`rlkit.envs.env_processor.DefaultConfig`, optional): Handles processing of observations and
            predictions
    """

    def __init__(
            self,
            env: ProxyEnv,
            model: DynamicsModel,
            termination_func: rlkit.types.TermFuncType,
            reward_func: Optional[rlkit.types.RewardFuncType] = None,
            sampling_mode: Optional[str] = None,
            sampling_cfg: Optional[omegaconf.DictConfig] = None,
            generator: Optional[torch.Generator] = None,
            obs_preproc: Optional[Callable] = None,
            obs_postproc: Optional[Callable] = None,
            clip_obs: bool = False,
            use_lamb_return: bool = False,
            lamb: float = 1.0, 
            discount: float = 1.0,
    ):
        self.dynamics_model = model
        self.termination_func = termination_func
        self.reward_func = reward_func
        self.device = ptu.device

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self._true_env = env
        self._obs: torch.Tensor = None
        self._prev_obs: torch.Tensor = None
        self._propagation_method: Optional[str] = None
        self._model_indices = None
        if generator:
            self._rng = generator
        else:
            self._rng = torch.Generator(device=self.device)
        self.obs_preproc = obs_preproc
        self.obs_postproc = obs_postproc

        # Set up for computing the lambda returns
        self.use_lamb_return = use_lamb_return
        self.lamb = lamb
        self.discount = discount
        # self.gamma_vec = None
        # self.lamb_vec = None

        self.sampling_mode = sampling_mode
        self.sampling_cfg = sampling_cfg

        self._return_as_np = True
        self._clip_obs = clip_obs

    def reset(
            self,
            initial_obs_batch,
            return_as_np: bool = True,
    ) -> rlkit.types.TensorType:
        """Resets the model environment
        The current observation of the model environment is set to be the provided initial_obs.

        Args:
            initial_obs_batch (np.ndarray): a batch of initial observations. One episode for
                each observation will be run in parallel. Shape must be ``B x d``, where
                ``B`` is batch size, and ``d`` is the observation dimension.
            return_as_np (bool): if ``True``, this method and :meth:`step` will return
                numpy arrays, otherwise it returns torch tensors in the same device as the
                model. Defaults to ``True``.
        Returns:
            (torch.Tensor or np.ndarray): the initial observation
        """
        assert initial_obs_batch.ndim == 2      # (B, d)
        self._obs: torch.Tensor = self.dynamics_model.reset(initial_obs_batch, rng=self._rng)
        self._return_as_np = return_as_np
        if self._return_as_np:
            return ptu.get_numpy(self._obs)
        return self._obs

    def step(
            self,
            actions: rlkit.types.TensorType,
            sample: bool = False,
            batch_size: int = -1,
            diagnostics: Optional[Dict] = None,
    ) -> Tuple[rlkit.types.TensorType, rlkit.types.TensorType, np.ndarray, Dict]:
        """
        Takes a single step within the model environment given the *batch* of actions.

        Args:
            actions (np.ndarray): the actions for each episode to rollout.
                Shape should be ``B x d_a``, where ``B`` is the batch size (i.e., number of episodes) and
                ``d_a`` is the action dimension.
                When the environment is reset,
            sample (bool): True if model predictions are stochastic. Defaults to False.
            batch_size (int): The batch size for forward propagation

        Returns:
            (tuple): (next_obs, reward, done, env_info)
        """
        assert actions.ndim == 2, "Should be provided with a batch of actions"
        if self.sampling_mode == "disagreement":
            next_obs, rewards, dones, _ = self.sample_with_disagreement(actions, sample, batch_size, diagnostics)
        else:
            next_obs, rewards, dones, _ = self.sample(actions, sample, batch_size, diagnostics)

        # if self._clip_obs:
        #     next_obs = self.clip(next_obs, self.observation_space)

        return next_obs, rewards, dones, {}

    def clip(self, arr, space):
        """Clips values in ``space`` to be within the upper and lower bounds."""
        low, high = space.low, space.high
        if isinstance(arr, np.ndarray):
            arr = np.where(arr < low, low, arr)
            arr = np.where(arr > high, high, arr)
        elif isinstance(arr, torch.Tensor):
            low, high = ptu.from_numpy(low), ptu.from_numpy(high)
            arr = torch.where(arr < low, low, arr)
            arr = torch.where(arr > high, high, arr)
        return arr

    def sample_with_disagreement(
            self,
            actions: rlkit.types.TensorType,
            sample: bool = False,
            batch_size: int = 500,
            diagnostics: Optional[Dict] = None,
    ) -> Tuple[rlkit.types.TensorType, rlkit.types.TensorType, np.ndarray, Dict]:
        """Samples next states while rewards are penalized for model disagreement (MOPO and MOReL).

        MOPO uses estimated variances of next state transitions to determine the penalty, while MOReL uses the maximum
        L2 discrepancy between estimated next state transitions for the purpose (penalty is only given when the
        disagreement exceeds a pre-defined threshold).
        """
        # For MOPO and MOReL
        model_len = self.dynamics_model.model_len

        batch_size = batch_size // model_len
        num_batches = int(np.ceil(self._obs.shape[0] / batch_size))

        rewards = ptu.zeros(self._obs.shape[0], 1).float()
        observations = ptu.zeros_like(self._obs).float()
        dones = ptu.zeros(self._obs.shape[0], 1, dtype=torch.bool)

        with torch.no_grad():
            # Prepare the input array: (N, d_s + d_a). Use mini-batches for efficiency
            for b in range(num_batches):
                obs = self._obs[b * batch_size: (b + 1) * batch_size]
                act = actions[b * batch_size: (b + 1) * batch_size]
                N = obs.size(0)

                # Repeat the input tensor for `model_len` number of times
                obs_ = obs.clone()
                if self.obs_preproc:
                    obs_ = self.obs_preproc(obs.clone())
                if isinstance(act, np.ndarray):
                    act = ptu.from_numpy(act)
                inputs = torch.cat([obs_, act], dim=-1)
                inputs = torch.tile(inputs, (model_len, 1))

                rew, obs_preds, logvar = self.dynamics_model.sample(
                    inputs,
                    deterministic=not sample,
                    rng=self._rng,
                    return_logvar=True
                )
                next_obs = self.obs_postproc(torch.tile(obs, (model_len, 1)), obs_preds)
                rew = (
                    rew if self.reward_func is None
                    else self.reward_func(torch.tile(obs, (model_len, 1)), torch.tile(act, (model_len, 1)), next_obs)
                )
                d = self.termination_func(next_obs).type_as(dones)

                # Now, compute the disagreement and penalize in rewards
                next_obs, rew, d, logvar = tuple(
                    map(lambda x: x.view(model_len, -1, x.size(-1)) if x is not None else None,
                        [next_obs, rew, d, logvar])
                )
                rew, d = self.compute_disagreement_and_penalize_reward(next_obs, rew, d, logvar, N, diagnostics)

                # Randomly select models for state trajectory propagation
                model_inds, batch_inds = ptu.randint(model_len, (N, ), generator=self._rng), ptu.arange(N)
                next_obs, rew, d = tuple(
                    map(lambda x: x[model_inds, batch_inds], [next_obs, rew, d])
                )

                observations[b * batch_size: (b + 1) * batch_size] = next_obs
                rewards[b * batch_size: (b + 1) * batch_size] = rew
                dones[b * batch_size: (b + 1) * batch_size] = d

            # For long horizon rollouts, terminated states may diverge and produce NaNs if left as-is; just set to prev obs
            self._obs = torch.where(dones, self._obs, observations)
            next_obs = self._obs.clone()

            if self._return_as_np:
                next_obs = ptu.get_numpy(next_obs)
                rewards = ptu.get_numpy(rewards)
                dones = ptu.get_numpy(dones)

        return next_obs, rewards, dones, {}

    def compute_disagreement_and_penalize_reward(
            self,
            next_obs: torch.Tensor,
            rewards: torch.Tensor,
            dones: torch.Tensor,
            logvar: torch.Tensor,
            batch_size: int,
            diagnostics: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # MOPO
        if self.sampling_cfg['type'] == 'var':
            assert isinstance(self.dynamics_model.model, ProbabilisticEnsemble)
            penalty_coeff = self.sampling_cfg.get('penalty_coeff')
            std = torch.sqrt(torch.exp(logvar))
            penalty = torch.amax(torch.linalg.norm(std, dim=2), dim=0).unsqueeze(-1)
            rewards[:] -= penalty_coeff * penalty

        # MOReL
        elif self.sampling_cfg['type'] == 'mean':
            truncate_thresh = self.sampling_cfg.truncate_thresh
            truncate_reward = self.sampling_cfg.truncate_reward
            max_err = np.zeros((batch_size, 1))
            next_obs = ptu.get_numpy(next_obs)
            for i in range(next_obs.shape[0]):
                pred_1 = next_obs[i, :]
                for j in range(i+1, next_obs.shape[0]):
                    pred_2 = next_obs[j, :]
                    error = np.linalg.norm((pred_1-pred_2), axis=-1, keepdims=True)
                    max_err = np.maximum(max_err, error)
            violations = ptu.from_numpy(max_err > truncate_thresh).type_as(dones)
            dones[:] |= violations
            rewards[:, violations] += truncate_reward
        return rewards, dones

    def sample(
            self,
            actions: rlkit.types.TensorType,
            sample: bool = False,
            batch_size: int = -1,
            diagnostics: Optional[Dict] = None,
    ) -> Tuple[rlkit.types.TensorType, rlkit.types.TensorType, np.ndarray, Dict]:
        """Samples next states in an ordinary manner."""
        # num_batches = int(np.ceil(self._obs.shape[0] / batch_size))
        if batch_size == -1 or self._obs.shape[0] // batch_size == 0:
            num_batches = 1
        else:
            assert self._obs.shape[0] % batch_size == 0
            num_batches = self._obs.shape[0] // batch_size
        rewards = ptu.zeros(self._obs.shape[0], 1).float()
        observations = ptu.zeros_like(self._obs).float()
        dones = ptu.zeros(self._obs.shape[0], 1, dtype=torch.bool)

        with torch.no_grad():
            # Prepare the input array: (N, d_s + d_a). Use mini-batches for efficiency
            for b in range(num_batches):
                obs = self._obs[b::num_batches]
                act = actions[b::num_batches]
                obs_ = obs.clone()
                if self.obs_preproc:
                    obs_ = self.obs_preproc(obs.clone())
                if isinstance(act, np.ndarray):
                    act = ptu.from_numpy(act)
                inputs = torch.cat([obs_, act], dim=-1)
                rew, obs_preds = self.dynamics_model.sample(
                    inputs,
                    deterministic=not sample,
                    rng=self._rng,
                )
                next_obs = self.obs_postproc(obs, obs_preds)
                rew = (
                    rew if self.reward_func is None
                    else self.reward_func(obs, act, next_obs)
                )
                d = self.termination_func(next_obs).type_as(dones)
                observations[b::num_batches] = next_obs
                rewards[b::num_batches] = rew
                dones[b::num_batches] = d
            # For long horizon rollouts, terminated states may diverge and produce NaNs if left as-is; just set to prev obs
            self._obs = torch.where(dones, self._obs, observations)
            if self._clip_obs:
                self._obs = self.clip(self._obs, self.observation_space)
            next_obs = self._obs.clone()

            if self._return_as_np:
                next_obs = ptu.get_numpy(next_obs)
                rewards = ptu.get_numpy(rewards)
                dones = ptu.get_numpy(dones)

        return next_obs, rewards, dones, {}

    def render(self, ):
        pass

    def evaluate_plans(
            self,
            plans: torch.Tensor,
            initial_state: np.ndarray,
            num_particles: int,
            past_action: torch.Tensor,
            prev_plan: torch.Tensor,
            iteration: int,
            mu: float = 0.0,
            value_func: Optional[nn.Module] = None,
            policy_prior: Optional[nn.Module] = None,
            diagnostics: Optional[Dict] = None,
            include_prev_action_as_input: bool = False,
            value_sampling_method: Optional[str] = None,
            std=None,
            **kwargs,
    ) -> torch.Tensor:
        """Evaluates a batch of action sequences using the model.

        Args:
            plans (torch.Tensor): A batch of action sequences to evaluate. The shape must be
                ``B x H x d``, where B, H, A are batch size, horizon, and action dimension, respectively.
            initial_state (np.ndarray): The initial state of the trajectories to be rolled out.
            num_particles (int): The number of times each plan is replicated. That is, per each trajectory that will be
                defined by an action sequence, ``num_particles`` number of replicates will be made.
            past_action (torch.Tensor): The action taken in the previous step.
            prev_plan (torch.Tensor): The previously optimized plan.
            iteration (int): The optimization iteration counter.
            mu (float): The coefficient for the lambda mixture with previous plan
            value_func (ValueEnsemble, optional): If provided, use the value ensemble to get the value of the final state.
            policy_prior (Module, optional): If provided, gives the distribution over actions a_H ~ \pi(a | s_H)

        Returns:
            (torch.Tensor) The accumulated reward for each action sequence, averaged over its particles.
        """
        # Sampling actions from a behavior clone can only be done when ``num_particles == 1``
        # assert (policy_prior is None) or (policy_prior is not None and num_particles == 1)

        # Check the tensor shape
        assert plans.ndim == 3      # (N, H, d)
        horizon = plans.size(1)

        if self.use_lamb_return:
            self.gamma_vec = ptu.tensor([self.discount ** h for h in range(horizon + 1)])
            return self.evaluate_plans_with_adaptive_lamd_return(
                plans,
                initial_state,
                num_particles,
                past_action,
                prev_plan,
                iteration,
                mu,
                value_func,
                policy_prior,
                diagnostics,
                include_prev_action_as_input=include_prev_action_as_input,
                value_sampling_method=value_sampling_method,
                std=std,
                **kwargs,
            )
        else:
            return self.evaluate_plans_default(
                plans,
                initial_state,
                num_particles,
                past_action,
                prev_plan,
                iteration,
                mu,
                value_func,
                policy_prior,
                diagnostics,
                include_prev_action_as_input=include_prev_action_as_input,
                value_sampling_method=value_sampling_method,
                std=std,
                **kwargs,
            )


    def evaluate_plans_with_lamb_return(
            self,
            plans: torch.Tensor,
            initial_state: np.ndarray,
            num_particles: int,
            past_action: torch.Tensor,
            prev_plan: torch.Tensor,
            iteration: int,
            mu: float = 0.0,
            value_func: Optional[nn.Module] = None,
            policy_prior: Optional[nn.Module] = None,
            diagnostics: Optional[Dict] = None,
    ) -> torch.Tensor:
        pass

    def evaluate_plans_with_adaptive_lamd_return(
            self,
            plans: torch.Tensor,
            initial_state: np.ndarray,
            num_particles: int,
            past_action: torch.Tensor,
            prev_plan: torch.Tensor,
            iteration: int,
            mu: float = 0.0,
            value_func: Optional[nn.Module] = None,
            policy_prior: Optional[nn.Module] = None,
            diagnostics: Optional[Dict] = None,
            include_prev_action_as_input: bool = False,
            value_sampling_method: Optional[str] = None,
            std=None,
            **kwargs,
    ) -> torch.Tensor:
        """
        Implements the adaptive lamba return for the values of trajectories.
        
        Notations:
            N: population_size
            K: num_particles = num_elites <= num_F_ensembles
            M: num_qfs
            H: horizon
            B: batch_size = N * K
            O/A: observation/action space size

        """
        assert value_func is not None, "MVE(lambda) requires specifiying value function!"
        assert isinstance(value_func, Ensemble)
        num_qfs = value_func.ensemble_size
        population_size, horizon, action_dim = plans.size()

        # Set the observation tensor
        initial_obs_batch = np.tile(                            # [KN, O]
            initial_state, (num_particles * population_size, 1)
        ).astype(np.float32)
        self.reset(initial_obs_batch, return_as_np=False)       # Update self._obs
        if policy_prior is not None:
            policy_prior.reset(self._obs, rng=self._rng)

        batch_size = initial_obs_batch.shape[0]
        terminated = ptu.zeros(num_particles * population_size, 1, dtype=torch.bool)
        # pre_return_lamb = ptu.zeros(batch_size, horizon)        # Stores n-step returns for n \in [1, horizon]
        # rewards = ptu.zeros(batch_size, horizon)                # Stores immediate rewards at each time step
        X_returns_h = ptu.zeros(population_size, num_particles, horizon + 1)                        # [N, K, H]
        Q_term_vals_h = ptu.zeros(num_qfs, population_size, num_particles, horizon + 1)     # [M, N, K, H]
        # R_h = ptu.zeros(num_Q_ensembles, population_size, num_particles, horizon)               # [M, N, K, H]

        lower_bound = ptu.from_numpy(np.tile(self.action_space.low, (population_size, 1)))
        upper_bound = ptu.from_numpy(np.tile(self.action_space.high, (population_size, 1)))

        past_action = past_action.clone().view(1, -1).expand(population_size, -1)   # [N, A]
        aggr_obs_t = ptu.from_numpy(np.tile(                    # [N, O], the average trajectory, for getting action only
            initial_state, (population_size, 1)
        ).astype(np.float32))

        for t in range(horizon + 1):
            if t == horizon:
                act_t = ptu.normal(0, std=std.expand(population_size, -1))
            else:
                act_t = plans[:, t, :]      # [population_size, horizon, action_dim] -> [N, A]

            # If `policy_prior` is given, use it to sample actions
            act_base = self.sample_from_prior(
                obs=aggr_obs_t,
                prior=policy_prior,
                include_prev_action_as_input=include_prev_action_as_input,
                past_action=past_action,
            )
            act_t[:] += act_base                # in place modification of action samples
            past_action = act_t.clone()         # a_t = f(s_t, a_{t-1}) + eps

            # TODO: ugly fix
            if t < horizon:
                # If the action from previous plan is mixed together with the current action input
                act_t[:] = (1 - mu) * act_t[:] + mu * prev_plan[t].expand(act_t.size(0), -1)

            # Clip actions to reside between bounds (in-place operation)
            act_t[:] = torch.where(
                act_t > upper_bound, upper_bound, act_t
            )
            act_t[:] = torch.where(
                act_t < lower_bound, lower_bound, act_t
            )

            # Repeat the action tensor for ``num_particles`` number of times
            action_batch = torch.repeat_interleave(         # [KN, A]
                act_t, num_particles, dim=0
            )

            # Compute the Q(s_t, a_t) from here for t > 0
            term_val_at_t = value_func(                                         # [M, KN, 1]
                torch.cat([self._obs.clone(), action_batch], dim=-1).unsqueeze(0).expand(value_func.ensemble_size, -1, -1),
                use_propagation=False,
                only_elite=True,
            )
            term_val_at_t[terminated.unsqueeze(0).expand(num_qfs, -1, -1)] = 0
            term_val_at_t = term_val_at_t.view(num_qfs, population_size, num_particles, 1)      # [M, N, K, 1]
            Q_term_vals_h[..., t: t + 1] = term_val_at_t.clone() * self.discount ** t

            if t < horizon:
                _, rew_t, dones, _ = self.step(action_batch, sample=True, batch_size=num_particles * population_size)    # Take a step in the model environment
                rew_t[terminated] = 0
                terminated |= dones

                # TODO: rewards of already terminated trajectory should be set to zero
                
                rew_t_per_rollout = rew_t.clone().view(population_size, num_particles, -1)
                obs_t_per_rollout = self._obs.clone().view(population_size, num_particles, -1)
                aggr_obs_t = torch.mean(obs_t_per_rollout, dim=1)           # Update the aggregated obs per rollout

                # Update cached reward matrix
                # rewards[:, t:t+1] = rew_t

                #################################################################
                # Update h-step returns
                # R0 =                                                      Q(s0)
                # R1 = r(s0) +                                          g x Q(s1)
                # R2 = r(s0) + g x r(s1) +                            g^2 x Q(s2)
                # ...
                # RH = r(s0) + g x r(s1) + ... + g^(H-1) x r(s_H-1) + g^H x Q(sH)
                #
                # At each rollout step h, update r(s) for Rh ~ RH, and Q(s) for Rh
                #################################################################
                # (1) update r(s)
                X_returns_h[..., t+1:] += rew_t_per_rollout * (self.discount ** t)
                # R_h[..., t:] += rew_t_per_rollout * (self.gamma ** t)       # [N, K, 1] -> [M, N, K, H]
                # (2) update Q(s)
                # term_val_at_t = value_func(self._obs.clone())               # [M, KN, 1]
                # term_val_at_t = term_val_at_t.view(num_qfs, population_size, num_particles, 1)  # [M, N, K, 1]
                # Q_term_vals_h[..., t] = term_val_at_t.clone()

        #################################################################
        # Compute MVE(lambda)-adaptive
        # Target: get returns for each of the rollouts [N]
        # Methodology: assume each h-return in a rollout (composed by K particles)
        #              follows a Gaussian, then aggregate h-returns with SB/MB
        #################################################################
        # (0) Update Q(s_H, a_H)... we don't have a_H from the given plan -> get from policy_prior
        #       - First use the aggregated obs to sample a_H
        #       - Followed by Q computation
        # act_H = self.sample_from_prior(
        #     obs=aggr_obs_t,
        #     prior=policy_prior,
        #     include_prev_action_as_input=include_prev_action_as_input,
        #     past_action=past_action,
        # )
        # noise_H = ptu.normal(0, std=std.expand(act_H.size(0), -1))
        # act_H += noise_H
        # action_batch = torch.repeat_interleave(         # [KN, A]
        #     act_H, num_particles, dim=0
        # )
        # term_val_at_H = value_func(                                         # [M, KN, 1]
        #     torch.cat([self._obs.clone(), action_batch], dim=-1).unsqueeze(0).expand(value_func.ensemble_size, -1, -1),
        #     use_propagation=False,
        #     only_elite=True,
        # )
        # term_val_at_H = term_val_at_H.view(num_qfs, population_size, num_particles, 1) * self.discount ** horizon      # [M, N, K, 1]
        # Q_term_vals_h[..., -1:] = term_val_at_H.clone()
        
        # (1) permute R_h axis and reshape
        X_returns_h = X_returns_h.permute(1, 2, 0)                      # [N, K, H] -> [K, H, N]
        Q_term_vals_h = Q_term_vals_h.permute(0, 2, 3, 1)               # [M, N, K, H] -> [M, K, H, N]
        # R_h = R_h.permute(0, 2, 3, 1)                                   # [M, N, K, H] -> [M, K, H, N]
        # R_h = R_h.view(num_Q_ensembles * num_particles, horizon, population_size)       # [MK, H, N]
        # (2) get mean
        X_var, X_mean = torch.var_mean(X_returns_h, dim=0)
        Q_var, Q_mean = torch.var_mean(Q_term_vals_h, dim=0)
        h_return_mean = X_mean + torch.mean(Q_mean, dim=0)                              # [H, N]
        # (3) get var
        E_of_Ey_sq = torch.mean(torch.square(Q_mean), dim=0)
        sq_of_E_Ey = torch.square(torch.mean(Q_mean, dim=0))
        h_return_var = torch.mean(Q_var, dim=0) + X_var + E_of_Ey_sq - sq_of_E_Ey
        # h_return_mean, h_return_var = torch.var_mean(R_h, dim=0)                        # [H, N]
        # (4) SB fusion
        h_return_preci = 1 / h_return_var
        # numerator_log = torch.sum(h_return_var_log, dim=0, keepdim=True).expand(horizon, -1)
        # numerator_log = numerator_log - h_return_var_log
        # numerator = torch.exp(numerator_log)
        # denominator = torch.sum(numerator, dim=0, keepdim=True)
        # assert torch.isinf(denominator).sum() == 0, '[ModelEnv.adaptive_lambda_return] Numerical precision issue.'
        # weights = numerator / denominator
        numerator_preci = torch.sum(h_return_preci, dim=0, keepdim=True).expand(horizon + 1, -1)
        weights = h_return_preci / numerator_preci
        adap_lambda_return = torch.sum(h_return_mean * weights, dim=0)                  # [N]
        # (5) log the expected horizon for analysis
        horizon_vec = ptu.tensor(list(range(horizon+1))).float().unsqueeze(dim=1)
        expected_horizon = torch.sum(horizon_vec * weights, dim=0)

        # Record the diagnostic information
        if diagnostics is not None:
            diagnostics.update(create_stats_ordered_dict(
                f'TrajOpt/Iteration {iteration} MVE Returns',
                ptu.get_numpy(adap_lambda_return)
            ))
            diagnostics.update(create_stats_ordered_dict(
                f'TrajOpt/Iteration {iteration} expected horizon',
                ptu.get_numpy(expected_horizon)
            ))

        return adap_lambda_return

    def sample_from_prior(
            self,
            obs,
            prior,
            include_prev_action_as_input: bool,
            past_action: torch.Tensor = None,
            **kwargs,
    ): 
        if prior is None:
            return ptu.zeros(obs.size(0), self.action_space.low.size)
        
        obs = obs.clone()
        if self.obs_preproc:
            obs = self.obs_preproc(obs)
        
        # In case policy prior is an ensemble...
        prior_len = prior.model_len if hasattr(prior, 'model_len') else 1
        if include_prev_action_as_input:
            input_to_bc = torch.cat([obs, past_action], dim=-1)
        else:
            input_to_bc = obs
        
        if hasattr(prior, 'set_output_transforms'):
            prior.set_output_transforms(True)

        if isinstance(prior, Ensemble):
            # Expand the aggregated observation tensor along the ensemble model dimension
            input_to_bc = input_to_bc.unsqueeze(0).expand(prior_len, -1, -1)
            act_base = prior.sample(input_to_bc, use_propagation=False, only_elite=True)[0]
            act_base = act_base.mean(dim=0)
        else:
            act_base = prior(input_to_bc)
            if isinstance(act_base, Distribution):
                act_base = act_base.sample()
            else:
                raise NotImplementedError("Prior type unrecognized!")
        
        if hasattr(prior, 'set_output_transforms'):
            prior.set_output_transforms(False)
        return act_base
    
    def evaluate_plans_default(
            self,
            plans: torch.Tensor,
            initial_state: np.ndarray,
            num_particles: int,
            past_action: torch.Tensor,
            prev_plan: torch.Tensor,
            iteration: int,
            mu: float = 0.0,
            value_func: Optional[nn.Module] = None,
            policy_prior: Optional[nn.Module] = None,
            diagnostics: Optional[Dict] = None,
            include_prev_action_as_input: bool = False,
            value_sampling_method: Optional[str] = None,
            **kwargs,
    ) -> torch.Tensor:
        population_size, horizon, action_dim = plans.size()

        # Set the observation tensor
        initial_obs_batch = np.tile(
            initial_state, (num_particles * population_size, 1)
        ).astype(np.float32)
        self.reset(initial_obs_batch, return_as_np=False)
        if policy_prior is not None:
            policy_prior.reset(self._obs, rng=self._rng)

        batch_size = initial_obs_batch.shape[0]
        total_returns = ptu.zeros(batch_size, 1)
        terminated = ptu.zeros(batch_size, 1, dtype=torch.bool)

        lower_bound = ptu.from_numpy(np.tile(self.action_space.low, (num_particles * population_size, 1)))
        upper_bound = ptu.from_numpy(np.tile(self.action_space.high, (num_particles * population_size, 1)))

        past_action = past_action.clone().view(1, -1).expand(population_size, -1)
        for t in range(horizon):
            act_t = plans[:, t, :]

            # If `policy_prior` is given, use it to sample actions
            act_base = self.sample_from_prior(
                obs=self._obs,
                prior=policy_prior,
                include_prev_action_as_input=include_prev_action_as_input,
                past_action=past_action,
            )
            act_t[:] += act_base        # in place modification of action samples
            past_action = act_t.clone()         # a_t = f(s_t, a_{t-1}) + eps

            # If the action from previous plan is mixed together with the current action input
            act_t[:] = (1 - mu) * act_t[:] + mu * prev_plan[t].expand(act_t.size(0), -1)

            # Clip actions to reside between bounds (in-place operation)
            act_t[:] = torch.where(
                act_t > upper_bound, upper_bound, act_t
            )
            act_t[:] = torch.where(
                act_t < lower_bound, lower_bound, act_t
            )

            # Repeat the action tensor for ``num_particles`` number of times
            action_batch = torch.repeat_interleave(
                act_t, num_particles, dim=0
            )
            _, rewards, dones, _ = self.step(action_batch, sample=True)     # Take a step in the model environment
            rewards[terminated] = 0                         # If already terminated, do not accumulate reward
            terminated |= dones
            total_returns += rewards * (self.discount ** t)

        """Include the value estimate at s_H"""
        if value_func is not None:
            obs_at_H = self._obs.clone()
            if self.obs_preproc:
                obs_at_H = self.obs_preproc(obs_at_H)

            # Option 1: use state-value function to compute V(s_H)
            if not value_func.is_qf:
                input_at_H = obs_at_H

            # Option 2: compute V_H = E_{a ~ \pi(a|s_H)} [ Q(s_H, a) ] ~= 1/N * Q(s_H, a_i) where a_i ~ \pi(a|s_H)
            # elif policy_prior is not None and value_func.is_qf:
            #     # TODO... handle action sampling
            #     obs_at_H_ = obs_at_H[None, :].expand(10, -1, -1)
            #     act_H = policy_prior.sample(obs_at_H, deterministic=False, rng=self._rng)
            #     input_at_H = torch.cat((obs_at_H_, act_H), dim=-1)

            # Option 3: use Q(s_t, a_{t-1}) as in MBOP.. Note this isn't an ordinary action value function
            else:
                act_H_1 = plans[:, -1, :]
                action_batch = torch.repeat_interleave(
                    act_H_1, num_particles, dim=0
                )
                input_at_H = torch.cat((obs_at_H, action_batch), dim=-1)
            # else:
            #     raise ValueError


            # Take the average (or min) of outputs generated by value ensemble models 
            # note: if `set_output_transforms` method exists, then values are scaled back to the original value scale
            if hasattr(value_func, 'set_output_transforms'):
                value_func.set_output_transforms(True)
            
            # Compute the terminal value
            term_values = value_func.sample(input_at_H, sampling_method=value_sampling_method)
            # term_values.squeeze_(dim=0)
            assert term_values.ndim == 2 and term_values.size(1) == 1
            term_values[terminated] = 0
            
            # Add the discounted terminal value to the total return
            total_returns += term_values * (self.discount ** horizon)
            
            if hasattr(value_func, 'set_output_transforms'):
                value_func.set_output_transforms(False)

            # Record the diagnostic information
            if diagnostics is not None:
                diagnostics.update(create_stats_ordered_dict(
                    f'TrajOpt/Iteration {iteration} Terminal Values',
                    ptu.get_numpy(term_values),
                ))

        total_returns = total_returns.reshape(-1, num_particles)
        return total_returns.mean(dim=1)
