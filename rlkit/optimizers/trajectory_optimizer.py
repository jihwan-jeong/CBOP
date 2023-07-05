"""
This code is adapted from https://github.com/facebookresearch/mbrl-lib/blob/master/mbrl/planning/trajectory_opt.py.
"""
from typing import Callable, Optional, Tuple, Dict

import numpy as np
from rlkit.torch import pytorch_util as ptu

import hydra
import hydra.utils
import omegaconf
import torch.distributions


class TrajectoryOptimizer:
    """Class for using generic optimizers on trajectory optimization problems.

    This is a convenience class that sets up an optimization problem for trajectories, given only
    action bounds and the length of the horizon. Using this class, the concern of handling
    appropriate tensor shapes for the optimization problem is hidden from the users, which only
    need to provide a function that is capable of evaluating trajectories of actions.

    It also takes care of shifting previous solution for the next optimization call, if the user desires.
    The optimization variables for the problem will have shape ``H x A``, where ``H`` and ``A``
    represent planning horizon and action dimension, respectively. The initial solution for the
    optimizer will be computed as (action_ub - action_lb) / 2, for each time step.

    Args:
        optimizer_cfg (omegaconf.DictConfig): the configuration of the optimizer to use.
        action_lb (np.ndarray): the lower bound for actions.
        action_ub (np.ndarray): the upper bound for actions.
        planning_horizon (int): the length of the trajectories that will be optimized.
        replan_freq (int): the frequency of re-planning. This is used for shifting the previous
        solution for the next time step, when ``keep_last_solution == True``. Defaults to 1.
        keep_last_solution (bool): if ``True``, the last solution found by a call to
            :meth:`optimize` is kept as the initial solution for the next step. This solution is
            shifted ``replan_freq`` time steps, and the new entries are filled using th3 initial
            solution. Defaults to ``True``.
    """

    def __init__(
            self,
            optimizer_cfg: omegaconf.DictConfig,
            action_lb: np.ndarray,
            action_ub: np.ndarray,
            horizon: int,
            plan_every: int = 1,
            keep_last_plan: bool = True,
    ):
        optimizer_cfg.lower_bound = np.tile(action_lb, (horizon, 1)).tolist()
        optimizer_cfg.upper_bound = np.tile(action_ub, (horizon, 1)).tolist()
        self.optimizer = hydra.utils.instantiate(optimizer_cfg)
        self.init_plan = (
            ((ptu.from_numpy(action_lb) + ptu.from_numpy(action_ub)) / 2)
        )
        self.action_lb, self.action_ub = action_lb, action_ub
        self.init_plan = self.init_plan.repeat((horizon, 1))
        self.prev_plan = self.init_plan.clone()
        self.plan_every = plan_every
        self.keep_last_plan = keep_last_plan
        self.horizon = horizon
        self.use_beta_mix = optimizer_cfg.get('use_beta_mix', False)
        self.use_policy_prior = False

    def optimize(
            self,
            trajectory_eval_fn: Callable[[torch.Tensor], torch.Tensor],
            callback: Optional[Callable] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Runs the trajectory optimization.

        Args:
            trajectory_eval_fn (callable(tensor) -> tensor): A function that receives a batch
                of action sequences and returns a batch of objective function values (e.g.,
                accumulated reward for each sequence). The shape of the action sequence tensor
                will be ``B x H x A``, where ``B``, ``H``, and ``A`` represent batch size,
                planning horizon, and action dimension, respectively.
            callback (callable, optional): a callback function
                to pass to the optimizer.

        Returns:
            (tuple of np.ndarray and float): the best action sequence.
        """
        best_sol, optim_diagnostics = self.optimizer.optimize(
            trajectory_eval_fn,
            x0=self.prev_plan,
            callback=callback,
            use_policy_prior=self.use_policy_prior,
        )

        if self.keep_last_plan:
            self.prev_plan = best_sol.roll(-self.plan_every, dims=0)
            # Note that initial_solution[i] is the same for all values of [i],
            # so just pick i = 0
            self.prev_plan[-self.plan_every:] = self.init_plan[0]
        return best_sol.cpu().numpy(), optim_diagnostics

    def reset(self):
        """Resets the previous solution cache to the initial solution."""
        self.prev_plan = self.init_plan.clone()

    def set_horizon(self, horizon: int):
        self.horizon = horizon
        self.init_plan = (
            ((ptu.from_numpy(self.action_lb) + ptu.from_numpy(self.action_ub)) / 2)
        )
        self.init_plan = self.init_plan.repeat((horizon, 1))

    def set_use_policy_prior(self, use_policy_prior):
        self.use_policy_prior = use_policy_prior
