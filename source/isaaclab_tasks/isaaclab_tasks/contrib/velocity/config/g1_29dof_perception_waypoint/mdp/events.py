# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset the robot at the start of its waypoint episode."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .waypoint import WaypointCommand


def reset_root_at_waypoint_start(env: ManagerBasedRLEnv, env_ids: torch.Tensor, command_name: str) -> None:
    """Draw the coming episode and place the robot at rest on the ground at its start, with its start yaw."""
    command: WaypointCommand = env.command_manager.get_term(command_name)
    env_ids = command._as_env_ids(env_ids)
    command.sample_episode(env_ids)
    robot = command.robot
    default_pose = robot.data.default_root_pose.torch[env_ids]
    position = command.start_pos_w[env_ids] + default_pose[:, :3]
    yaw = command.start_yaw[env_ids]
    rotation = math_utils.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    orientation = math_utils.quat_mul(rotation, default_pose[:, 3:7])
    robot.write_root_pose_to_sim_index(root_pose=torch.cat((position, orientation), dim=-1), env_ids=env_ids)
    robot.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids
    )
