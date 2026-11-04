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

_VELOCITY_AXES = ("x", "y", "z", "roll", "pitch", "yaw")


def reset_goal_joint_state(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str,
    position_range: tuple[float, float] = (-0.1, 0.1),
) -> None:
    """Write reference joints, blended toward the default pose and rest by the task difficulty.

    Imitation episodes keep the reference state. ``position_range`` [rad] adds uniform noise to every
    joint before clipping to the soft limits.
    """
    command: GoalMotionCommand = env.command_manager.get_term(command_name)
    env_ids = command._as_env_ids(env_ids)
    command.resample_time_steps(env_ids)
    robot = command.robot
    blend = (command.reset_difficulty(env_ids) * ~command.is_imitation[env_ids]).unsqueeze(-1)
    joint_pos = torch.lerp(command.joint_pos[env_ids], robot.data.default_joint_pos.torch[env_ids], blend)
    joint_pos += math_utils.sample_uniform(*position_range, joint_pos.shape, env.device)
    limits = robot.data.soft_joint_pos_limits.torch[env_ids]
    joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])
    joint_vel = command.joint_vel[env_ids] * (1.0 - blend)
    robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)


def reset_goal_root_state(env: ManagerBasedRLEnv, env_ids: torch.Tensor, command_name: str) -> None:
    """Write the reference root, moved toward an upright standing start for task episodes.

    Task starts move ``task_start_offset`` from the first reference root in its yaw frame, with XY and yaw
    noise, and blend roll, pitch, and velocity to zero, all scaled by the task difficulty. Above
    difficulty 0.8, a fraction of task starts receive root velocity noise.
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
    task_ids = env_ids[task]
    count = len(task_ids)
    if count > 0:
        difficulty = command.reset_difficulty(task_ids)
        reference_quat = orientation[task]
        reference_yaw = math_utils.yaw_quat(reference_quat)
        offset = torch.tensor(cfg.task_start_offset, device=env.device).repeat(count, 1) * difficulty[:, None]
        offset[:, :2] += math_utils.sample_uniform(
            -cfg.initial_position_noise, cfg.initial_position_noise, (count, 2), env.device
        ) * difficulty.unsqueeze(-1)
        position[task] += math_utils.quat_apply(reference_yaw, offset)
        yaw_noise = math_utils.sample_uniform(-cfg.initial_yaw_noise, cfg.initial_yaw_noise, count, env.device)
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(reference_quat)
        orientation[task] = math_utils.quat_from_euler_xyz(
            roll * (1.0 - difficulty), pitch * (1.0 - difficulty), yaw + yaw_noise * difficulty
        )

        ranges = torch.tensor(
            [cfg.task_velocity_range.get(axis, (0.0, 0.0)) for axis in _VELOCITY_AXES], device=env.device
        )
        samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (count, 6), env.device)
        moving = torch.rand(count, device=env.device) < cfg.task_velocity_probability
        # perturb velocities only once most of the reference momentum is removed
        scale = ((difficulty - 0.8) / 0.2).clamp(0.0, 1.0)
        if not cfg.task_reset_curriculum:
            scale = torch.zeros_like(difficulty)
        samples *= (scale * moving).unsqueeze(-1)
        keep = (1.0 - difficulty).unsqueeze(-1)
        lin_vel[task] = lin_vel[task] * keep + math_utils.quat_apply(reference_yaw, samples[:, :3])
        ang_vel[task] = ang_vel[task] * keep + math_utils.quat_apply(reference_yaw, samples[:, 3:])

    command.robot.write_root_pose_to_sim_index(root_pose=torch.cat((position, orientation), dim=-1), env_ids=env_ids)
    command.robot.write_root_velocity_to_sim_index(root_velocity=torch.cat((lin_vel, ang_vel), dim=-1), env_ids=env_ids)
