# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import numpy as np
import torch
from typing import Union

import rlkit.torch.pytorch_util as ptu


def hopper(next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done = (
        torch.isfinite(next_obs).all(-1)
        * (next_obs[:, 1:].abs() < 100).all(-1)
        * (height > 0.7)
        * (angle.abs() < 0.2)
    )

    done = ~not_done
    done = done[:, None]
    return done


def cartpole(next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    x, theta = next_obs[:, 0], next_obs[:, 2]

    x_threshold = 2.4
    theta_threshold_radians = 12 * 2 * math.pi / 360
    not_done = (
        (x > -x_threshold)
        * (x < x_threshold)
        * (theta > -theta_threshold_radians)
        * (theta < theta_threshold_radians)
    )
    done = ~not_done
    done = done[:, None]
    return done


def inverted_pendulum(next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    not_done = torch.isfinite(next_obs).all(-1) * (next_obs[:, 1].abs() <= 0.2)
    done = ~not_done

    done = done[:, None]

    return done


def no_termination(next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    done = torch.Tensor([False]).repeat(len(next_obs)).bool().to(next_obs.device)
    done = done[:, None]
    return done


def walker2d(next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done = (height > 0.8) * (height < 2.0) * (angle > -1.0) * (angle < 1.0)
    done = ~not_done
    done = done[:, None]
    return done

def pen(next_obs: torch.Tensor) -> torch.Tensor:
    """
    See corresponding reward_func for details.
    """
    obj_pos = next_obs[:, 24:27]
    done = ptu.zeros(next_obs.size()[0], 1, dtype=torch.bool)
    done[obj_pos[:,2] < 0.075] = True
    return done

def ant(next_obs: torch.Tensor):
    assert len(next_obs.shape) == 2

    x = next_obs[:, 0]
    not_done = torch.isfinite(next_obs).all(-1) * (x >= 0.2) * (x <= 1.0)

    done = ~not_done
    done = done[:, None]
    return done

def antmaze(next_obs, goal_pos, radius):
    device = next_obs.device
    goal_pos = torch.tensor(goal_pos).repeat(next_obs.shape[0], 1).to(device)

    dist = torch.linalg.norm(next_obs[:, :2] - goal_pos, dim=-1)
    done = dist < radius
    done = done[:, None]
    return done

def antmaze_umaze(next_obs: torch.Tensor):
    goal = (0.75, 8.75)
    radius = 0.5
    return antmaze(next_obs, goal, radius)

def antmaze_medium(next_obs: torch.Tensor):
    goal = (20.75, 20.75)
    radius = 0.5
    return antmaze(next_obs, goal, radius)

def antmaze_large(next_obs: torch.Tensor):
    goal = (32.75, 24.75)
    radius = 0.5
    return antmaze(next_obs, goal, radius)

def humanoid(next_obs: torch.Tensor):
    assert len(next_obs.shape) == 2

    z = next_obs[:, 0]
    done = (z < 1.0) + (z > 2.0)

    done = done[:, None]
    return done
