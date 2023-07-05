import os
import numpy as np
import gym.wrappers

ENV_ASSET_DIR = os.path.join(os.path.dirname(__file__), 'assets')


"""
Below are classes / functions copied from facebookresearch/mbrl-lib 
"""
class freeze_mujoco_env:
    """Provides a context to freeze a Mujoco environment.
    This context allows the user to manipulate the state of a Mujoco environment and return it
    to its original state upon exiting the context.
    Works with mujoco gym and dm_control environments
    (with `dmc2gym <https://github.com/denisyarats/dmc2gym>`_).
    Example usage:
    .. code-block:: python
       env = gym.make("HalfCheetah-v2")
       env.reset()
       action = env.action_space.sample()
       # o1_expected, *_ = env.step(action)
       with freeze_mujoco_env(env):
           step_the_env_a_bunch_of_times()
       o1, *_ = env.step(action) # o1 will be equal to what o1_expected would have been
    Args:
        env (:class:`gym.wrappers.TimeLimit`): the environment to freeze.
    """

    def __init__(self, env: gym.wrappers.TimeLimit):
        self._env = env
        self._init_state: np.ndarray = None
        self._elapsed_steps = 0
        self._step_count = 0

        if "gym.envs.mujoco" in self._env.env.__class__.__module__:
            self._enter_method = self._enter_mujoco_gym
            self._exit_method = self._exit_mujoco_gym
        elif "rlkit.third_party.dmc2gym" in self._env.env.__class__.__module__:
            self._enter_method = self._enter_dmcontrol
            self._exit_method = self._exit_dmcontrol
        elif "d4rl.utils.wrappers" in self._env.env.__class__.__module__:
            self._enter_method = self._enter_mujoco_gym
            self._exit_method = self._exit_mujoco_gym
        else:
            raise RuntimeError("Tried to freeze an unsupported environment.")

    def _enter_mujoco_gym(self):
        self._init_state = (
            self._env.env.data.qpos.ravel().copy(),
            self._env.env.data.qvel.ravel().copy(),
        )
        self._elapsed_steps = self._env._elapsed_steps

    def _exit_mujoco_gym(self):
        self._env.set_state(*self._init_state)
        self._env._elapsed_steps = self._elapsed_steps

    def _enter_dmcontrol(self):
        self._init_state = self._env.env._env.physics.get_state().copy()
        self._elapsed_steps = self._env._elapsed_steps
        self._step_count = self._env.env._env._step_count

    def _exit_dmcontrol(self):
        with self._env.env._env.physics.reset_context():
            self._env.env._env.physics.set_state(self._init_state)
            self._env._elapsed_steps = self._elapsed_steps
            self._env.env._env._step_count = self._step_count

    def __enter__(self):
        return self._enter_method()

    def __exit__(self, *_args):
        return self._exit_method()
