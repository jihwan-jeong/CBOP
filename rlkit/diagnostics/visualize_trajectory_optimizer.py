"""
This file is used for diagnosing a trajectory optimizer such as MPPI or CEM.
To this end, a true dynamics (and reward) of the environment is used. So, this file
is really for double-checking whether the trajectory optimizer can really optimize
a given objective function (i.e., the cumulative reward for a fixed steps).

By setting `render=True`, a video will be saved. Otherwise, return values will simply
be saved in a numpy array.

This file is adapted from https://github.com/facebookresearch/mbrl-lib/blob/master/mbrl/diagnostics/control_env.py.
"""
import argparse
import multiprocessing as mp
from pathlib import Path
import time
from typing import cast

import gym.wrappers
import hydra.utils
import numpy as np
import omegaconf
import skvideo.io
import torch

from rlkit.torch import pytorch_util as ptu
from rlkit.envs.env_processor import DefaultEnvProc
import rlkit.envs.env_util as env_util
from rlkit.envs.wrappers import ProxyEnv
from rlkit.optimizers.trajectory_optimizer import TrajectoryOptimizer
from rlkit.diagnostics.util import init, evaluate_all_plans

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="halfcheetah")
    parser.add_argument("--is_offline", action='store_true')
    parser.add_argument("--d4rl_config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--control_horizon", type=int, default=30)
    parser.add_argument("--use_behavior_clone", action='store_true')
    parser.add_argument("--include_prev_action_as_input", action='store_false')
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--samples_per_process", type=int, default=64)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--optimizer_type", choices=["cem", "mppi"], default="cem")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    mp.set_start_method("spawn")
    eval_env = env_util.make_env_from_str(args.env, args.is_offline, args.d4rl_config)
    eval_env.seed(args.seed)
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    current_obs = eval_env.reset()

    gpu_mode = torch.cuda.is_available()
    ptu.set_gpu_mode(gpu_mode)

    # Instantiate a behavior clone model and load its saved parameters (or set to None)
    behavior_clone = None
    if args.use_behavior_clone:
        assert args.model_dir is not None
        file_path = Path(args.model_dir) / 'behavior_clone_checkpoint.pth'
        assert Path.is_file(file_path)

        obs_dim = env_util.get_dim(eval_env.observation_space)
        action_dim = env_util.get_dim(eval_env.action_space)

        input_dim = obs_dim + action_dim
        bc_model_cfg = omegaconf.OmegaConf.create(
            {
                "_target_": "rlkit.torch.models.ensemble.Ensemble",
                "ensemble_size": 3,
                "layer_size": 500,
                "num_hidden_layer": 2,
                "activation_func": "torch.relu",
                "include_prev_action_as_input": args.include_prev_action_as_input,
            }
        )
        obs_preproc = DefaultEnvProc.obs_preproc
        if obs_preproc is not None:
            input_dim = action_dim + obs_preproc(np.zeros((1, obs_dim))).shape[-1]
        hidden_activation = hydra.utils.get_method(bc_model_cfg.get('activation_func'))

        behavior_clone = hydra.utils.instantiate(
            bc_model_cfg,
            hidden_sizes=[bc_model_cfg.layer_size] * bc_model_cfg.num_hidden_layer,
            input_size=input_dim if bc_model_cfg.include_prev_action_as_input else input_dim - action_dim,
            output_size=action_dim,
            hidden_activation=hidden_activation,
            propagation_method='expectation',
        )
        behavior_clone.load(path=str(file_path), key='behavior_clone_state_dict')
        behavior_clone.include_prev_action_as_input = bc_model_cfg.include_prev_action_as_input

    if args.optimizer_type.lower() == "cem":
        optimizer_cfg = omegaconf.OmegaConf.create(
            {
                "_target_": "rlkit.optimizers.CEMOptimizer",
                "max_iters": 5,
                "elite_ratio": 0.1,
                "population_size": args.num_processes * args.samples_per_process,
                "polyak": 0.1,
                "device": "cpu",
                "lower_bound": "???",
                "upper_bound": "???",
            }
        )
    elif args.optimizer_type.lower() == "mppi":
        optimizer_cfg = omegaconf.OmegaConf.create(
            {
                "_target_": "rlkit.optimizers.MPPIOptimizer",
                "max_iters": 5 if behavior_clone is None else 1,
                "population_size": args.num_processes * args.samples_per_process,
                "sigma": 0.95 if behavior_clone is None else 0.05,
                "kappa": 1.0,
                "beta": 0.1,
                "lamb": 0.0,
                "device": "cpu",
                "lower_bound": "???",
                "upper_bound": "???",
            }
        )
    else:
        raise ValueError

    controller = TrajectoryOptimizer(
        optimizer_cfg,
        eval_env.action_space.low,
        eval_env.action_space.high,
        args.control_horizon,
    )
    controller.set_use_behavior_clone(use_behavior_clone=behavior_clone is not None)

    lock = mp.Lock()
    with mp.Pool(
        processes=args.num_processes, initializer=init, initargs=[args.env, args.seed, args.is_offline, args.d4rl_config, lock]
    ) as pool__:

        total_reward__ = 0
        frames = []
        value_history = np.zeros(
            (
                args.num_steps,
                optimizer_cfg.population_size,
                optimizer_cfg.max_iters,
            )
        )

        for t in range(args.num_steps):
            if args.render:
                frames.append(eval_env.render(mode="rgb_array"))
            start = time.time()

            if isinstance(eval_env, ProxyEnv):
                current_state__ = env_util.get_current_state(eval_env)
            else:
                current_state__ = env_util.get_current_state(
                    cast(gym.wrappers.TimeLimit, eval_env)
                )

            def trajectory_eval_fn(plans, prev_plan, iteration, diagnostics=None):
                return evaluate_all_plans(
                    plans,
                    pool__,
                    current_state__,
                    behavior_clone=behavior_clone,
                )

            best_value = [0]

            def compute_population_stats(_population, values, opt_step):
                value_history[t, :, opt_step] = values.numpy()
                best_value[0] = max(best_value[0], values.max().item())

            plan, _ = controller.optimize(
                trajectory_eval_fn, callback=compute_population_stats
            )
            action__ = plan[0]
            next_obs__, reward__, done__, _ = eval_env.step(action__)

            total_reward__ += reward__

            print(
                f"step: {t}, time: {time.time() - start: .3f}, "
                f"reward: {reward__: .3f}, pred_value: {best_value[0]: .3f}, total_reward: {total_reward__: .3f}"
            )

        output_dir = Path(args.output_dir)
        Path.mkdir(output_dir, parents=True, exist_ok=True)

        if args.render:
            frames_np = np.stack(frames)
            writer = skvideo.io.FFmpegWriter(
                output_dir / f"control_{args.env}_video.mp4", verbosity=1
            )
            for i in range(len(frames_np)):
                writer.writeFrame(frames_np[i, :, :, :])
            writer.close()

        print("total_reward: ", total_reward__)
        np.save(output_dir / "value_history.npy", value_history)
