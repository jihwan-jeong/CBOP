import datetime
import json
import os
from os import path as osp
from pathlib import Path
from collections import namedtuple
import dateutil.tz
import wandb

from omegaconf import OmegaConf
from rlkit.core.logging.logging import logger
from rlkit.launchers import conf
from typing import Optional

GitInfo = namedtuple(
    'GitInfo',
    [
        'directory',
        'code_diff',
        'code_diff_staged',
        'commit_hash',
        'branch_name',
    ],
)
#
# def create_log_dir(
#         exp_prefix,
#         exp_id=0,
#         seed=0,
#         base_log_dir=None,
#         include_exp_prefix_sub_dir=True,
# ):
#     """
#     Creates and returns a unique log directory.
#
#     :param exp_prefix: All experiments with this prefix will have log
#     directories be under this directory.
#     :param exp_id: The number of the specific experiment run within this
#     experiment.
#     :param base_log_dir: The directory where all log should be saved.
#     :return:
#     """
#     exp_name = create_exp_name(exp_prefix, exp_id=exp_id,
#                                seed=seed)
#     if base_log_dir is None:
#         base_log_dir = osp.join(os.getcwd(), 'exp')
#     if include_exp_prefix_sub_dir:
#         log_dir = osp.join(base_log_dir, exp_prefix.replace("_", "-"), exp_name)
#     else:
#         log_dir = osp.join(base_log_dir, exp_name)
#     if osp.exists(log_dir):
#         print("WARNING: Log directory already exists {}".format(log_dir))
#     os.makedirs(log_dir, exist_ok=True)
#     return log_dir


def get_git_infos(dirs):
    try:
        import git
        git_infos = []
        for directory in dirs:
            # Idk how to query these things, so I'm just doing try-catch
            try:
                repo = git.Repo(directory)
                try:
                    branch_name = repo.active_branch.name
                except TypeError:
                    branch_name = '[DETACHED]'
                git_infos.append(GitInfo(
                    directory=directory,
                    code_diff=repo.git.diff(None),
                    code_diff_staged=repo.git.diff('--staged'),
                    commit_hash=repo.head.commit.hexsha,
                    branch_name=branch_name,
                ))
            except git.exc.InvalidGitRepositoryError as e:
                print("Not a valid git repo: {}".format(directory))
    except ImportError:
        git_infos = None
    return git_infos
#
#
# def create_exp_name(exp_prefix, exp_id=0, seed=0):
#     """
#     Create a semi-unique experiment name that has a timestamp
#     :param exp_prefix:
#     :param exp_id:
#     :return:
#     """
#     now = datetime.datetime.now(dateutil.tz.tzlocal())
#     timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
#     return "%s_%s_%04d--s-%d" % (exp_prefix, timestamp, exp_id, seed)


def setup_logger(
        exp_prefix,
        cfg,
        text_log_file="debug.log",
        tabular_log_file="progress.csv",
        log_to_wandb=False,
        snapshot_mode="last",
        snapshot_gap=1,
        log_tabular_only=False,
        log_dir: Optional[Path] =None,
        git_infos=None,
        script_name=None,
):
    """
    Set up logger to have some reasonable default settings.
    Note that the logging directory is already set up automatically by Hydra.
    Hence, the log_dir would be:

        exp/algorithm/env/yyyy-mm-dd/hhmmss/

    Args:
        exp_prefix:
        cfg:
        text_log_file:
        tabular_log_file:
        snapshot_mode:
        log_tabular_only:
        snapshot_gap:
        log_dir             (str, optional)
        git_infos:
        script_name: If set, save the script name to this.

    Returns:

    """
    if git_infos is None:
        git_infos = get_git_infos(conf.CODE_DIRS_TO_MOUNT)

    log_dir = os.getcwd()
    first_time = len(logger.log_dir) == 0 or logger.log_dir != log_dir
    logger.log_dir = log_dir

    tabular_log_path = osp.join(log_dir, tabular_log_file)
    text_log_path = osp.join(log_dir, text_log_file)

    logger.add_text_output(text_log_path)
    if first_time:
        logger.add_tabular_output(tabular_log_path)                     # File opened as 'a' mode
    else:
        logger._add_output(tabular_log_path, logger._tabular_outputs,
                           logger._tabular_fds, mode='a')
        for tabular_fd in logger._tabular_fds:
            logger._tabular_header_written.add(tabular_fd)
    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode(snapshot_mode)
    logger.set_snapshot_gap(snapshot_gap)
    logger.set_log_tabular_only(log_tabular_only)
    tmp = log_dir.split("/")
    idx = tmp.index('exp')
    exp_name = '-'.join([tmp[idx+1]] + tmp[idx+3:]).replace('.', '-')
    logger.push_prefix("[%s] " % exp_name)

    if log_to_wandb:
        logger.log_to_wandb = True
        upload_to_wandb = cfg.get('upload_to_wandb', False)
        name = '-'.join(exp_name.split('-')[2:])
        conf_dict = OmegaConf.to_container(cfg, resolve=True)
        if cfg.group_name is None:
            group = '-'.join(exp_name.split('-')[:2])       # default: ${algorithm}-${env}
        elif cfg.sweep:                                     # Assuming a single variable (indep_var) matters in sweeping
            try:
                sweep_exp_name = "-".join(map(lambda x: str(x), OmegaConf.to_container(cfg.overrides.exp_name, resolve=True)))
                group = f"{cfg.group_name}-{sweep_exp_name}"
            except:
                group = f"{cfg.group_name}"
        else:
            group = f"{cfg.group_name}"
        if cfg.entity is None:
            run_ = wandb.init(name=name, config=conf_dict,
                       project=cfg.WANDB_PROJECT, group=group,)
                       # settings=wandb.Settings(start_method="fork", console="off"))
        else:
            run_ = wandb.init(name=name, config=conf_dict,
                              project=cfg.WANDB_PROJECT, group=group, entity=cfg.entity)
        if upload_to_wandb:
            logger.set_snapshot_dir(wandb.run.dir)

    # if git_infos is not None:
    #     for (
    #         directory, code_diff, code_diff_staged, commit_hash, branch_name
    #     ) in git_infos:
    #         directory = directory.replace('.', '')
    #         if directory[-1] == '/':
    #             directory = directory[:-1]
    #         diff_file_name = directory[1:].replace("/", "-") + ".patch"
    #         diff_staged_file_name = (
    #             directory[1:].replace("/", "-") + "_staged.patch"
    #         )
    #         if code_diff is not None and len(code_diff) > 0:
    #             with open(osp.join(log_dir, diff_file_name), "w") as f:
    #                 f.write(code_diff + '\n')
    #         if code_diff_staged is not None and len(code_diff_staged) > 0:
    #             with open(osp.join(log_dir, diff_staged_file_name), "w") as f:
    #                 f.write(code_diff_staged + '\n')
    #         with open(osp.join(log_dir, "git_infos.txt"), "a") as f:
    #             f.write("directory: {}\n".format(directory))
    #             f.write("git hash: {}\n".format(commit_hash))
    #             f.write("git branch name: {}\n\n".format(branch_name))
    if script_name is not None:
        with open(osp.join(log_dir, "script_name.txt"), "w") as f:
            f.write(script_name)
    return log_dir

#
# def dict_to_safe_json(d):
#     new_d = {}
#     for key, item in d.items():
#         if safe_json(item):
#             new_d[key] = item
#         else:
#             if isinstance(item, dict):
#                 new_d[key] = dict_to_safe_json(item)
#             else:
#                 new_d[key] = str(item)
#     return new_d

# def safe_json(data):
#     if data is None:
#         return True
#     elif isinstance(data, (bool, int, float)):
#         return True
#     elif isinstance(data, (tuple, list)):
#         return all(safe_json(x) for x in data)
#     elif isinstance(data, dict):
#         return all(isinstance(k, str) and safe_json(v) for k, v in data.items())
#     return False