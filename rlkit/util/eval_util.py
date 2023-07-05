"""
Common evaluation utilities.
"""

from collections import OrderedDict
from numbers import Number

import numpy as np
import torch
from typing import List, Dict
import rlkit.util.pythonplusplus as ppp
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.models.dynamics_models.model import DynamicsModel


def get_generic_path_information(paths, stat_prefix=''):
    """
    Get an OrderedDict with a bunch of statistic names and values.
    """

    statistics = OrderedDict()
    returns = np.vstack([sum(path["rewards"].flatten()) for path in paths])

    # Reward / returns
    rewards = np.concatenate([path["rewards"].flatten() for path in paths])
    statistics.update(create_stats_ordered_dict('Rewards', rewards,
                                                stat_prefix=stat_prefix))
    statistics.update(create_stats_ordered_dict('Returns', returns,
                                                stat_prefix=stat_prefix))

    # Action
    actions = [path["actions"].squeeze() for path in paths]
    if len(actions[0].shape) == 1:
        actions = np.hstack([path["actions"].squeeze() for path in paths])
    else:
        actions = np.vstack([path["actions"].squeeze() for path in paths])
    statistics.update(create_stats_ordered_dict(
        'Actions', actions, stat_prefix=stat_prefix
    ))

    # Paths
    num_paths = len(paths)
    statistics['Num Paths'] = num_paths
    path_lengths = np.vstack([len(path['observations'].squeeze()) for path in paths])
    statistics.update(
        create_stats_ordered_dict(
            'Path Length', path_lengths, stat_prefix=stat_prefix
        ))

    return statistics
    #
    # for info_key in ['env_infos', 'agent_infos']:
    #     if info_key in paths[0]:
    #         all_env_infos = [
    #             ppp.list_of_dicts__to__dict_of_lists(p[info_key])
    #             for p in paths
    #         ]
    #         for k in all_env_infos[0].keys():
    #             final_ks = np.array([info[k][-1] for info in all_env_infos])
    #             first_ks = np.array([info[k][0] for info in all_env_infos])
    #             all_ks = np.concatenate([info[k] for info in all_env_infos])
    #             statistics.update(create_stats_ordered_dict(
    #                 stat_prefix + k,
    #                 final_ks,
    #                 stat_prefix='{}/final/'.format(info_key),
    #             ))
    #             statistics.update(create_stats_ordered_dict(
    #                 stat_prefix + k,
    #                 first_ks,
    #                 stat_prefix='{}/initial/'.format(info_key),
    #             ))
    #             statistics.update(create_stats_ordered_dict(
    #                 stat_prefix + k,
    #                 all_ks,
    #                 stat_prefix='{}/'.format(info_key),
    #             ))
    #
    # return statistics


def get_average_returns(paths):
    returns = [sum(path["rewards"].squeeze()) for path in paths]
    return np.mean(returns)


def create_stats_ordered_dict(
        name,
        data,
        stat_prefix=None,
        always_show_all_stats=True,
        exclude_max_min=False,
):
    if stat_prefix is not None:
        name = f"{stat_prefix}{name}"
    if isinstance(data, Number):
        return OrderedDict({name: data})

    if len(data) == 0:
        return OrderedDict()

    if isinstance(data, tuple):
        ordered_dict = OrderedDict()
        for number, d in enumerate(data):
            sub_dict = create_stats_ordered_dict(
                f"{name}_{number}",
                d,
            )
            ordered_dict.update(sub_dict)
        return ordered_dict

    if isinstance(data, list):
        try:
            iter(data[0])
        except TypeError:
            pass
        else:
            data = np.concatenate(data)

    if (isinstance(data, np.ndarray) and data.size == 1
            and not always_show_all_stats):
        return OrderedDict({name: float(data)})

    stats = OrderedDict([
        (name + ' Mean', np.mean(data)),
        (name + ' Std', np.std(data)),
    ])
    if not exclude_max_min:
        stats[name + ' Max'] = np.max(data)
        stats[name + ' Min'] = np.min(data)
    return stats

def eval_dynamics(
        dynamics_model: DynamicsModel,
        paths: List[Dict],
        obs_preproc,
        obs_postproc
):
    """
    Computes the accumulated error between actual state transitions in the environment and the predicted next states.
    """
    diagnostics = OrderedDict()
    errors = [0] * len(paths)
    with torch.no_grad():
        for i, path in enumerate(paths):
            obs = path['observations']
            act = path['actions']
            next_obs = path['next_observations']

            # Create tensors
            obs, act = list(
                map(lambda x: ptu.from_numpy(x).view(x.shape[0], -1), [obs, act])
            )
            obs_ = obs.clone()
            if obs_preproc is not None:
                obs_ = obs_preproc(obs_)

            # Prepare input to the dynamics model (assuming it's an ensemble)
            num_models = dynamics_model.model_len
            state_action_pair = torch.cat([obs_, act], dim=-1)
            state_action_pair = torch.tile(state_action_pair, [num_models, 1])

            # Make prediction (average prediction is used)
            orig_prop_method = dynamics_model.model.get_propagation_method()
            dynamics_model.model.set_propagation_method("expectation")
            _, preds = dynamics_model.sample(
                x=state_action_pair,
                deterministic=True,
            )
            dynamics_model.model.set_propagation_method(orig_prop_method)
            if obs_postproc is not None:
                next_obs_preds = obs_postproc(obs, preds)      # e.g. when model outputs a delta state

            # Compute L2 error
            next_obs_preds = ptu.get_numpy(next_obs_preds)
            error = np.linalg.norm(next_obs_preds - next_obs)

            errors[i] = error
    # Record in a dictionary and return it
    diagnostics.update(create_stats_ordered_dict(
        "L2 Transition Errors",
        np.array(errors),
        always_show_all_stats=True,
    ))
    return diagnostics