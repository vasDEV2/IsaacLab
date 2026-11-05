# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset the robot onto the reference for imitation episodes or near a standing start for task episodes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .motion import GoalMotionCommand


def reset_goal_joint_state(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str,
    position_range: tuple[float, float] = (-0.1, 0.1),
) -> None:
    """Write reference joints for imitation episodes and the default pose at rest for task episodes.

    ``position_range`` [rad] adds uniform noise to every joint before clipping to the soft limits.
    """
    command: GoalMotionCommand = env.command_manager.get_term(command_name)
    env_ids = command._as_env_ids(env_ids)
    command.resample_time_steps(env_ids)
    robot = command.robot
    task = (~command.is_imitation[env_ids]).unsqueeze(-1)
    joint_pos = torch.where(task, robot.data.default_joint_pos.torch[env_ids], command.joint_pos[env_ids])
    joint_pos += math_utils.sample_uniform(*position_range, joint_pos.shape, env.device)
    limits = robot.data.soft_joint_pos_limits.torch[env_ids]
    joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])
    joint_vel = command.joint_vel[env_ids] * ~task
    robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)


def reset_goal_root_state(env: ManagerBasedRLEnv, env_ids: torch.Tensor, command_name: str) -> None:
    """Write the reference root for imitation episodes and a fixed standing start for task episodes.

    Task starts move ``task_start_offset`` from the first reference root in its yaw frame, with small XY and yaw
    noise, upright and at rest. Task difficulty comes from the stair height of the tile, not from the start.
    """
    command: GoalMotionCommand = env.command_manager.get_term(command_name)
    env_ids = command._as_env_ids(env_ids)
    command.resample_time_steps(env_ids)
    cfg = command.cfg
    position = command.root_pos_w[env_ids]
    orientation = command.root_quat_w[env_ids]
    lin_vel = command.root_lin_vel_w[env_ids]
    ang_vel = command.root_ang_vel_w[env_ids]

    task = ~command.is_imitation[env_ids]
    count = int(task.sum())
    if count > 0:
        reference_quat = orientation[task]
        reference_yaw = math_utils.yaw_quat(reference_quat)
        offset = torch.tensor(cfg.task_start_offset, device=env.device).repeat(count, 1)
        offset[:, :2] += math_utils.sample_uniform(
            -cfg.task_start_position_noise, cfg.task_start_position_noise, (count, 2), env.device
        )
        position[task] += math_utils.quat_apply(reference_yaw, offset)
        yaw = math_utils.euler_xyz_from_quat(reference_quat)[2]
        yaw += math_utils.sample_uniform(-cfg.task_start_yaw_noise, cfg.task_start_yaw_noise, count, env.device)
        zeros = torch.zeros_like(yaw)
        orientation[task] = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
        lin_vel[task] = 0.0
        ang_vel[task] = 0.0

    command.robot.write_root_pose_to_sim_index(root_pose=torch.cat((position, orientation), dim=-1), env_ids=env_ids)
    command.robot.write_root_velocity_to_sim_index(root_velocity=torch.cat((lin_vel, ang_vel), dim=-1), env_ids=env_ids)
