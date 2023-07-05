import hydra
from hydra.utils import get_original_cwd, to_absolute_path

import numpy as np
import random
import omegaconf
from omegaconf import OmegaConf
import torch
import wandb
import os.path as osp
import rlkit.torch.pytorch_util as ptu
from rlkit.envs.env_util import make_env_from_cfg
from rlkit.data_management.replay_util import make_buffer_from_cfg
from rlkit.envs.env_processor import get_offline_data
from rlkit.examples.algorithms import loader
from rlkit.core.logging.logging import logger
from rlkit.core.logging.logging_setup import setup_logger
from rlkit.util import eval_util
from pathlib import Path


@hydra.main(config_path="conf", config_name="config")
def run(cfg: omegaconf.DictConfig):
    # Set up the gpu mode
    if cfg.device == 'cpu':
        gpu_mode = False
    else:
        gpu_mode = torch.cuda.is_available()
    
    gpu_id = 0
    if gpu_mode and cfg.device:
        gpu_id = cfg.device.split(':')[-1]
    ptu.set_gpu_mode(gpu_mode, gpu_id)

    output_dir = Path.cwd()

    # Set up the name of the experiment
    exp_name = "{}_{}".format(cfg.algorithm.name, cfg.env.name)
    if cfg.include_date:
        from datetime import datetime
        timestamp = datetime.now().strftime('%m-%d')
        exp_name = '%s-%s' % (timestamp, exp_name)

    # Set up logger and pring out the current configuration
    setup_logger(
        exp_prefix=exp_name,
        cfg=cfg,
        log_to_wandb=cfg.log_to_wandb,
        snapshot_mode=cfg.snapshot_mode,
    )
    logger.log(f"Configuration: \n{OmegaConf.to_yaml(cfg, resolve=True)}", with_timestamp=True)

    # To use cached parameters
    orig_cwd = get_original_cwd()
    cache_dir = cfg.cache_dir
    if cache_dir:
        cache_dir = osp.join(orig_cwd, cache_dir)
        if osp.isdir(cache_dir):
            cfg.cache_dir = cache_dir
        else:
            logger.log(f"cache_dir {cfg.cache_dir} is provided but it is not a valid directory, so it's ignored..")
            cfg.cache_dir = None

    # Create the gym environment for exploration
    expl_env, term_func, reward_func = make_env_from_cfg(cfg)

    # Seed
    rng = None
    if cfg.seed is not None:
        seed = cfg.seed
        rng = np.random.default_rng(seed=seed)
        torch.manual_seed(seed)
        if gpu_mode:
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)

    # Create the replay buffer
    replay_buffer = make_buffer_from_cfg(
        env=expl_env,
        cfg=cfg,
        rng=rng,
        reward_func=reward_func,
    )

    # Create the evaluation environment (can be different from the exploration environment)
    eval_env, _, _ = make_env_from_cfg(cfg, is_eval=True)

    # Handle the offline case (load the data and report statistics to the log)
    if cfg.is_offline:
        paths = get_offline_data(
            cfg.env.name.lower(),
            eval_env.wrapped_env,
            replay_buffer,
            cfg.max_size,
            rng,
            reward_normalize=cfg.env.get('reward_normalize', False)
        )

        # Print dataset statistics
        if paths:
            logger.log("Compute dataset statistics for offline RL")
            logger.record_dict(
                eval_util.get_generic_path_information(paths)
            )
        logger.dump_tabular(log_only=True)

    # Instantiate the main RL algorithm class
    algorithm = loader.get_algorithm(cfg, expl_env, eval_env, replay_buffer, reward_func, term_func, rng=rng)
    algorithm.to(ptu.device)
    if cfg.log_to_wandb and cfg.debug_mode:
        algorithm.configure_logging(log_freq=10, log=cfg.log_option)

    # Run the main training loop
    algorithm.train()

    # Finish the current run and upload all logged data to wandb server
    wandb.finish()

if __name__ == "__main__":
    run()
