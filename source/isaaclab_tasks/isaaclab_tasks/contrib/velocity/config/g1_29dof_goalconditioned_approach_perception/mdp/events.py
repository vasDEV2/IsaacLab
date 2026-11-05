# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset the robot at rest at its random spawn."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .approach import ApproachMotionCommand


def reset_at_spawn(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str,
    position_range: tuple[float, float] = (-0.1, 0.1),
) -> None:
    """Draw the coming episode and place the robot upright and at rest at its spawn, with its spawn yaw.

    Joints start in the default pose plus uniform noise of ``position_range`` [rad], clipped to the soft limits.
    """
    command: ApproachMotionCommand = env.command_manager.get_term(command_name)
    env_ids = command._as_env_ids(env_ids)
    command.resample_time_steps(env_ids)
    robot = command.robot

    joint_pos = robot.data.default_joint_pos.torch[env_ids].clone()
    joint_pos += math_utils.sample_uniform(*position_range, joint_pos.shape, env.device)
    limits = robot.data.soft_joint_pos_limits.torch[env_ids]
    joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])
    robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos), env_ids=env_ids)

    default_pose = robot.data.default_root_pose.torch[env_ids]
    position = command.spawn_pos_w[env_ids] + default_pose[:, :3]
    yaw = command.spawn_yaw[env_ids]
    rotation = math_utils.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    orientation = math_utils.quat_mul(rotation, default_pose[:, 3:7])
    robot.write_root_pose_to_sim_index(root_pose=torch.cat((position, orientation), dim=-1), env_ids=env_ids)
    robot.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids
    )
