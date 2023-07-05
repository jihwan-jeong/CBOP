from collections import OrderedDict
from typing import Callable, Optional, Sequence, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn

from rlkit.torch import pytorch_util as ptu
from rlkit.util import math

class Optimizer:
    def __init__(self):
        pass

    def optimize(
            self,
            obj_fun: Callable[[torch.Tensor, torch.Tensor, int, Optional[Dict]], torch.Tensor],
            x0: Optional[torch.Tensor] = None,
            **kwargs
    ) -> torch.Tensor:
        """Runs optimization.

        Args:
            obj_fun (callable(tensor, tensor, int, dict) -> tensor): objective function to maximize.
            x0 (tensor, optional): initial solution, if necessary.
        Returns:
            (torch.Tensor): the best solution found.
        """
        pass


class RSOptimizer(Optimizer):

    def __init__(
            self,
            population_size: int,
            max_iters: int,
            upper_bound: Sequence[Sequence[float]],
            lower_bound: Sequence[Sequence[float]],
            device: torch.device,
            **kwargs,
    ):
        super().__init__()
        self.horizon = len(lower_bound)
        self.population_size = population_size
        self.device = device if device is not None else ptu.device
        self.upper_bound = ptu.tensor(upper_bound, dtype=torch.float32,
                                      torch_device=device)
        self.lower_bound = ptu.tensor(lower_bound, dtype=torch.float32,
                                      torch_device=device)

        self.num_max_iters = max_iters
        self.use_policy_prior = False

    def optimize(
            self,
            obj_fun: Callable[[torch.Tensor, torch.Tensor, int, Optional[Dict]], torch.Tensor],
            x0: Optional[torch.Tensor] = None,
            **kwargs
    ) -> Tuple[torch.Tensor, OrderedDict]:
        """
        Optimizes the cost function and returns the corresponding argmin action sequence.

        Args:
            obj_fun (Callable): The objective function to maximize.
            x0 (tensor): The initial
        """

        # Create N random sequences of actions --- size is (N, H * d)
        plans = torch.distributions.uniform.Uniform(
            self.lower_bound[0], self.upper_bound[0]
        ).sample((self.population_size, ) + x0.shape)

        values = obj_fun(plans, x0, 0, 0, None)
        return plans[torch.argmin(values)], OrderedDict()

class CEMOptimizer(RSOptimizer):

    def __init__(
            self,
            population_size: int,
            max_iters: int,
            upper_bound: Sequence[Sequence[float]],
            lower_bound: Sequence[Sequence[float]],
            elite_ratio: float,
            device: torch.device = None,
            epsilon=1e-3,
            polyak=0.2,
            return_mean_elites: bool = False,       # If `False`, a single plan with the highest value returned
            **kwargs,
    ):
        super().__init__(
            population_size,
            max_iters=max_iters,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            device=device,
            **kwargs,
        )
        self.polyak = polyak
        self.epsilon = epsilon
        self.elite_ratio = elite_ratio
        self.return_mean_elites = return_mean_elites
        self.num_elites = np.ceil(self.population_size * self.elite_ratio).astype(np.int32)
        self.init_var = ((self.upper_bound - self.lower_bound) ** 2) / 16

    def optimize(
            self,
            obj_fun: Callable[[torch.Tensor, torch.Tensor, int, Optional[Dict]], torch.Tensor],
            x0: Optional[torch.Tensor] = None,
            callback: Optional[Callable[[torch.Tensor, torch.Tensor, int], None]] = None,
            **kwargs
    ) -> Tuple[torch.Tensor, OrderedDict]:
        """
        Optimizes the objective function by iteratively updating the mean (iterative random-shooting).

        Args:
            obj_fun (Callable): The objective function to maximize.
            x0 (tensor, optional): The initial mean for the population. Must
                be consistent with lower/upper bounds.
            callback (callable(tensor, tensor, int) -> any, optional): If given, this
                function will be called after every iteration, passing it as input the full
                population tensor, its corresponding objective function values, and
                the index of the current iteration. This can be used for logging and plotting
                purposes.

        Returns:
            (torch.Tensor): the best solution found.
        """
        mu = x0.clone()
        var = self.init_var.clone()

        best_plan = torch.empty_like(mu)
        best_value = -np.inf
        population = ptu.zeros((self.population_size, ) + x0.shape,
                               torch_device=self.device)

        diagnostics = OrderedDict()

        for i in range(self.num_max_iters):
            lb_dist = mu - self.lower_bound
            ub_dist = self.upper_bound - mu
            mv = torch.min(torch.square(lb_dist / 2), torch.square(ub_dist / 2))
            constrained_var = torch.min(mv, var)

            population = math.truncated_normal_(population)
            population = population * torch.sqrt(constrained_var) + mu

            values = obj_fun(population, x0, i, torch.sqrt(constrained_var[-1:, :]), diagnostics)

            if callback is not None:
                callback(population, values, i)

            # Filter out NaN values
            values[torch.isnan(values)] = -1e10
            best_values, elite_idx = values.topk(self.num_elites)
            elite = population[elite_idx]

            # Update new mean and variance of actions
            new_mu = torch.mean(elite, dim=0)
            new_var = torch.var(elite, unbiased=False, dim=0)
            mu = self.polyak * mu + (1 - self.polyak) * new_mu
            var = self.polyak * var + (1 - self.polyak) * new_var

            diagnostics[f'TrajOpt/Iteration {i} Average Mean'] = mu.mean()
            diagnostics[f'TrajOpt/Iteration {i} Std of Mean'] = mu.std()
            diagnostics[f'TrajOpt/Iteration {i} Variance Mean'] = var.mean()
            diagnostics[f'TrajOpt/Iteration {i} Variance Std'] = var.std()

            if best_values[0] > best_value:
                best_value = best_values[0]
                best_plan = population[elite_idx[0]].clone()

        return (mu, diagnostics) if self.return_mean_elites else (best_plan, diagnostics)

class MPPIOptimizer(RSOptimizer):

    def __init__(
            self,
            population_size: int,
            max_iters: int,
            upper_bound: Sequence[Sequence[float]],
            lower_bound: Sequence[Sequence[float]],
            device: torch.device,
            sigma: float,
            kappa: float,
            beta: float,
            **kwargs,
    ):
        super().__init__(
            population_size,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            max_iters=max_iters,
            device=device,
            **kwargs,
        )
        self.action_dim = len(lower_bound[0])
        self.mean = ptu.zeros(
            (self.horizon, self.action_dim), dtype=torch.float32,
            torch_device=device,
        )
        self.kappa = kappa
        self.var = sigma ** 2 * ptu.ones_like(self.lower_bound, torch_device=device)
        self.beta = beta

    def optimize(
            self,
            obj_fun: Callable[[torch.Tensor, torch.Tensor, int, Optional[Dict]], torch.Tensor],
            x0: Optional[torch.Tensor] = None,
            callback: Optional[Callable[[torch.Tensor, torch.Tensor, int], None]] = None,
            use_policy_prior: bool = False,
            **kwargs,
    ) -> Tuple[torch.Tensor, OrderedDict]:
        """Implementation of MPPI planner.
        Args:
            obj_fun (callable(tensor, tensor, int, dict) -> tensor): objective function to maximize.
            x0 (tensor, optional): Not required
            callback (callable(tensor, tensor, int) -> any, optional): if given, this
                function will be called after every iteration, passing it as input the full
                population tensor, its corresponding objective function values, and
                the index of the current iteration. This can be used for logging and plotting
                purposes.
            use_policy_prior (bool): Whether to use a policy prior for action sampling.

        Returns:
            (torch.Tensor): the best solution found.
        """
        past_action = self.mean[0]
        self.mean[:-1] = self.mean[1:].clone()

        if not use_policy_prior:
            return self._default_optimize(obj_fun, x0, past_action, callback=callback, **kwargs)
        else:
            return self._optimize_with_policy_prior(obj_fun, x0, callback=callback, **kwargs)

    def _optimize_with_policy_prior(
            self,
            obj_fun: Callable[[torch.Tensor, torch.Tensor, int, Optional[Dict]], torch.Tensor],
            x0: Optional[torch.Tensor] = None,
            callback: Optional[Callable[[torch.Tensor, torch.Tensor, int], None]] = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, OrderedDict]:

        diagnostics = OrderedDict()

        values = None
        for k in range(self.num_max_iters):
            # Sample noise and update constrained variances
            noise = ptu.empty(
                size=(
                    self.population_size,
                    self.horizon,
                    self.action_dim,
                ),
                torch_device=self.device,
            )
            nn.init.normal_(noise)                                  # e ~ N(0, 1)
            samples = noise.clone() * torch.sqrt(self.var)          # e ~ N(0, sigma**2)

            # noise = math.truncated_normal_(noise)
            #
            # lb_dist = self.mean - self.lower_bound
            # ub_dist = self.upper_bound - self.mean
            # mv = torch.min(torch.square(lb_dist / 2), torch.square(ub_dist / 2))
            # constrained_var = torch.min(mv, self.var)
            # samples = noise.clone() * torch.sqrt(constrained_var)

            # Compute the values of action samples..
            # Note: at this stage, ``samples`` is just a random noise;
            # within :meth:`rlkit.envs.model_env.ModelEnv.evaluate_plans`, a policy_prior is used to sample
            # actions, a_t = f(s_t, a_{t-1}) or f(s_t). This ``samples`` object will directly be modified such that
            # we can get access to the updated action values.
            values = obj_fun(samples, x0, k, torch.sqrt(self.var[-1:, :]), diagnostics)
            values[torch.isnan(values)] = -1e-10
            if callback is not None:
                callback(samples, values, k)

            # Weight actions (MPPI Update)
            weights = torch.reshape(
                torch.exp(self.kappa * (values - values.max())),
                (self.population_size, 1, 1),
            )
            norm = torch.sum(weights) + 1e-10
            weighted_actions = samples * weights
            self.mean = torch.sum(weighted_actions, dim=0) / norm

            # Log
            diagnostics[f'TrajOpt/Iteration {k} Average Mean'] = self.mean.mean()
            diagnostics[f'TrajOpt/Iteration {k} Std of Mean'] = self.mean.std()
            diagnostics[f'TrajOpt/Iteration {k} Value Mean'] = values.mean()
            diagnostics[f'TrajOpt/Iteration {k} Value Std'] = values.std()

        return self.mean.clone(), diagnostics

    def _default_optimize(
            self,
            obj_fun: Callable[[torch.Tensor, torch.Tensor, int, Optional[Dict]], torch.Tensor],
            x0: Optional[torch.Tensor],
            past_action: torch.Tensor,
            callback: Optional[Callable[[torch.Tensor, torch.Tensor, int], None]] = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, OrderedDict]:
        """
        Optimizes the cost function via the ordinary MPPI.

        Args:
            obj_func (callable(torch.Tensor, torch.Tensor, int, dict) -> torch.Tensor): The objective function to maximize.
            callback (callable(torch.Tensor, torch.Tensor, int) -> any, optional): if given, this
                function will be called after every iteration, passing it as input the full
                population tensor, its corresponding objective function values, and
                the index of the current iteration. This can be used for logging and plotting
                purposes.
        Returns:
            (torch.Tensor): The best solution found.
        """
        diagnostics = OrderedDict()

        values = None
        for k in range(self.num_max_iters):
            # Sample noise and update constrained variances
            noise = ptu.empty(
                size=(
                    self.population_size,
                    self.horizon,
                    self.action_dim,
                ),
                torch_device=self.device,
            )
            noise = math.truncated_normal_(noise)

            lb_dist = self.mean - self.lower_bound
            ub_dist = self.upper_bound - self.mean
            mv = torch.min(torch.square(lb_dist / 2), torch.square(ub_dist / 2))
            constrained_var = torch.min(mv, self.var)
            samples = noise.clone() * torch.sqrt(constrained_var)

            # Smoothed actions with noise
            samples[:, 0, :] = (
                self.beta * (self.mean[0, :] + noise[:, 0, :])
                + (1 - self.beta) * past_action
            )
            for i in range(max(self.horizon - 1, 0)):
                samples[:, i+1, :] = (
                    self.beta * (self.mean[i+1] + noise[:, i+1, :])
                    + (1 - self.beta) * samples[:, i, :]
                )

            # Clip actions to reside within the bounds
            samples = torch.where(
                samples > self.upper_bound, self.upper_bound, samples
            )
            samples = torch.where(
                samples < self.lower_bound, self.lower_bound, samples
            )

            # Compute the values of action samples
            values = obj_fun(samples, x0, k, torch.sqrt(constrained_var[-1:, :]), diagnostics)
            values[torch.isnan(values)] = -1e-10
            if callback is not None:
                callback(samples, values, k)

            # Weight actions
            weights = torch.reshape(
                torch.exp(self.kappa * (values - values.max())),
                (self.population_size, 1, 1),
            )
            norm = torch.sum(weights) + 1e-10
            weighted_actions = samples * weights
            self.mean = torch.sum(weighted_actions, dim=0) / norm

            # Log
            diagnostics[f'TrajOpt/Iteration {k} Average Mean'] = self.mean.mean()
            diagnostics[f'TrajOpt/Iteration {k} Std of Mean'] = self.mean.std()
            diagnostics[f'TrajOpt/Iteration {k} Value Mean'] = values.mean()
            diagnostics[f'TrajOpt/Iteration {k} Value Std'] = values.std()

        return self.mean.clone(), diagnostics
