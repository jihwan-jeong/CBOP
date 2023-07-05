# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
from typing import Union

from . import termination_funcs
import rlkit.torch.pytorch_util as ptu

from d4rl.hand_manipulation_suite.pen_v0 import ADD_BONUS_REWARDS


def cartpole(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    return (~termination_funcs.cartpole(next_obs)).float().view(-1, 1)


def inverted_pendulum(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    return (~termination_funcs.inverted_pendulum(next_obs)).float().view(-1, 1)


def pets_halfcheetah(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    reward_ctrl = -0.1 * act.square().sum(dim=1)
    reward_run = next_obs[:, 0] - 0.0 * next_obs[:, 2].square()
    return (reward_run + reward_ctrl).view(-1, 1)

def halfcheetah(obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, dt=0.05) -> torch.Tensor:
    assert len(next_obs.shape) == len(action.shape) == 2

    reward_ctrl = - 0.1 * action.square().sum(dim=1)
    reward_run = (next_obs[:, 0] - obs[:, 0]) / dt
    return reward_run + reward_ctrl

def hopper(obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, dt=0.05) -> torch.Tensor:
    assert len(next_obs.shape) == len(action.shape) == 2

    alive_bonus = 1.0
    reward_ctrl = -1e-3 * action.square().sum(dim=1)
    reward_run = (next_obs[:, 0] - obs[:, 0]) / dt          # Warning: this assumes the first state feature is the position
    return reward_run + alive_bonus + reward_ctrl

def walker2d(obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, dt=0.05) -> torch.Tensor:
    assert len(next_obs.shape) == len(action.shape) == 2

    alive_bonus = 1.0
    reward_run = (next_obs[:, 0] - obs[:, 0]) / dt          # Warning: this assumes the first state feature is the position
    reward_ctrl = -1e-3 * action.square().sum(dim=1)
    return reward_run + alive_bonus + reward_ctrl

def pen(obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    """
    See d4rl.d4rl.hand_manipulation_suite.pen_v0 for details.

    qp = env.data.qpos                                      [30,]
    obj_pos = env.data.body_xpos[env.obj_bid]               [3,]
    obj_vel = env.data.qvel[-6:]                            [6,]
    obj_orien = (env.data.site_xpos[env.obj_t_sid] - env.data.site_xpos[env.obj_b_sid])/env.pen_length      [3,]
    desired_orien = (env.data.site_xpos[env.tar_t_sid] - env.data.site_xpos[env.tar_b_sid])/env.tar_length  [3,]
    desired_pos = env.data.site_xpos[env.eps_ball_sid]      [3,]    (called desired_loc in step())

    obs_space: [qp[:-6], obj_pos, obj_vel, obj_orien, desired_orien
                obj_pos-desired_pos, obj_orien-desired_orien]  ->  [45,]
    """
    # act_mid = np.array([-0.1745, -0.09  ,  0.    ,  0.8   ,  0.8   ,  0.8   ,  0.    ,
    #                     0.8   ,  0.8   ,  0.8   ,  0.    ,  0.8   ,  0.8   ,  0.8   ,
    #                     0.35  ,  0.    ,  0.8   ,  0.8   ,  0.8   ,  0.    ,  0.65  ,
    #                     0.    ,  0.    , -0.7855])
    # act_rng = np.array([0.3495, 0.7   , 0.44  , 0.8   , 0.8   , 0.8   , 0.44  , 0.8   ,
    #                     0.8   , 0.8   , 0.44  , 0.8   , 0.8   , 0.8   , 0.35  , 0.44  ,
    #                     0.8   , 0.8   , 0.8   , 1.    , 0.65  , 0.26  , 0.52  , 0.7855])

    obj_pos = next_obs[:, 24:27]
    det_pos = next_obs[:, 39:42]
    desired_loc = obj_pos - det_pos
    obj_orien = next_obs[:, 33:36]
    desired_orien = next_obs[:, 36:39]

    # a = np.clip(action, -1., 1.)
    # try:
    #     starting_up = False
    #     a = act_mid + a*act_rng # mean center and scale
    # except:
    #     starting_up = True
    #     a = a 

    # pos cost
    dist = torch.linalg.norm(obj_pos - desired_loc, dim=1, keepdims=True)
    reward = -dist
    # orien cost
    orien_similarity = torch.sum(obj_orien * desired_orien, dim=1).unsqueeze(1)
    reward += orien_similarity

    if ADD_BONUS_REWARDS:
        # bonus for being close to desired orientation
        reward[torch.logical_and(dist < 0.075, orien_similarity > 0.9)] += 10
        reward[torch.logical_and(dist < 0.075, orien_similarity > 0.95)] += 50
        # if dist < 0.075 and orien_similarity > 0.9:
        #     reward += 10
        # if dist < 0.075 and orien_similarity > 0.95:
        #     reward += 50

    # penalty for dropping the pen
    reward[obj_pos[:, 2] < 0.075] -= 5

    return reward

def hammer(obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    """
    See d4rl.d4rl.hand_manipulation_suite.hammer_v0 for details.

    qp = env.data.qpos                                      [33,]
    qv = env.data.qvel                                      [33,]
    palm_pos = env.data.site_xpos[env.S_grasp_sid]          [3,]
    obj_pos = env.data.body_xpos[env.obj_bid]               [3,]
    obj_rot = quat2euler(env.data.body_xquat[env.obj_bid])  [4,] -> [3,]
    target_pos = env.data.site_xpos[env.target_obj_sid]     [3,]
    nail_impact = env.sim.data.sensordata[env.sim.model.sensor_name2id('S_nail')]   [1,]

    obs_space: [qp[:-6], qv[-6:], palm_pos, obj_pos, obj_rot,
                target_pos, np.array([nail_impact])]  ->  [46,]
    act_space: [26,]
    """
    # Missing: env.data.site_xpos[self.tool_sid] & env.data.site_xpos[self.goal_sid]
    raise NotImplementedError("Hammer env reward func: obs is not enough for calculating the reward.")

def door(obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    """
    See d4rl.d4rl.hand_manipulation_suite.door_v0 for details.

    qp = env.data.qpos                                      [30,]
    latch_pos = qp[-1]                                      [1,]
    door_pos = env.data.body_xpos[env.door_hinge_did]       [1,]
    palm_pos = env.data.site_xpos[env.grasp_sid]            [3,]
    handle_pos = env.data.site_xpos[env.handle_sid]         [3,]
    door_open: bool                                         [1,]

    obs_space: [qp[1:-2], [latch_pos], door_pos, palm_pos, handle_pos,
                palm_pos-handle_pos, [door_open]]  ->  [39,]
    act_space: [28,]
    """
    palm_pos = obs[:, 29:32]
    handle_pos = obs[:, 32:35]
    det_pos = obs[:, 35:38]
    # Missing: env.data.qvel
    raise NotImplementedError("Door env reward func: obs is not enough for calculating the reward.")

def relocate(obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    """
    See d4rl.d4rl.hand_manipulation_suite.relocate_v0 for details.

    qp = env.data.qpos                                      [36,]
    obj_pos = env.data.body_xpos[env.obj_bid]               [3,]
    palm_pos = env.data.site_xpos[env.S_grasp_sid]          [3,]
    target_pos = env.data.site_xpos[env.target_obj_sid]     [3,]

    obs_space: [qp[:-6], palm_pos-obj_pos, palm_pos-target_pos, obj_pos-target_pos]  ->  [39,]
    act_space: [30,]
    """
    palm_obj = obs[:, 30:33]
    palm_target = obs[:, 33:36]
    obj_target = obs[:, 36:]
    # Missing: don't know what exactly the obj_pos[2] is
    raise NotImplementedError("Door env reward func: obs is not enough for calculating the reward.")

def antmaze(obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("Antmaze env reward func: obs is not enough for calculating the reward.")

def pusher(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    goal_pos = torch.tensor([0.45, -0.05, -0.323]).to(next_obs.device)

    to_w, og_w = 0.5, 1.25
    tip_pos, obj_pos = next_obs[:, 14:17], next_obs[:, 17:20]

    tip_obj_dist = (tip_pos - obj_pos).abs().sum(axis=1)
    obj_goal_dist = (goal_pos - obj_pos).abs().sum(axis=1)
    obs_cost = to_w * tip_obj_dist + og_w * obj_goal_dist

    act_cost = 0.1 * (act ** 2).sum(axis=1)

    return -(obs_cost + act_cost).view(-1, 1)
