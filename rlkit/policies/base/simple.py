from rlkit.policies.base.base import Policy

class RandomPolicy(Policy):
    """
    Policy that always outputs zero.
    """

    def __init__(self, action_space, **kwargs):
        self.action_space = action_space
        if 'rng' in kwargs:
            self._rng = kwargs['rng']
        else:
            import numpy as np
            self._rng = np.random.default_rng(0)

    def get_action(self, *args, **kwargs):
        return self.action_space.sample(), {}

    def get_actions(self, obs_np, **kwargs):
        assert obs_np.ndim == 2
        N = obs_np.shape[0]
        d = self.action_space.low.size
        lb, ub = self.action_space.low, self.action_space.high
        return self._rng.uniform(lb, ub, size=(N, d))

    def to(self, *args, **kwargs):
        pass

    def __call__(self, obs):
        if obs.ndim == 1:
            obs = obs.reshape(-1, 1)
        return self.get_actions(obs)
