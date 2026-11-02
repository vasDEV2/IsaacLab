# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain-relative rewards for velocity tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def compute_standing_contact_penalty(
    foot_contact: torch.Tensor,
    velocity_command: torch.Tensor,
    base_lin_vel_b: torch.Tensor,
    base_ang_vel_b: torch.Tensor,
    projected_gravity_b: torch.Tensor,
    command_threshold: float = 0.05,
    recovery_linear_velocity: float = 0.2,
    recovery_angular_velocity: float = 0.5,
    recovery_tilt: float = 0.2,
) -> torch.Tensor:
    """Compute a standing contact penalty from tensors, without accessing the environment.

    This is a helper for reward terms, not a reward-manager term itself.
    ``velocity_command`` contains at least ``(vx, vy, wz)`` per environment.
    Base linear/angular velocities and projected gravity have shape ``(num_envs, 3)``
    and are expressed in the robot base frame.

    ``foot_contact`` has shape ``(num_envs, num_feet)`` and uses true/nonzero for contact.
    Linear and yaw commands must each be below ``command_threshold`` to enable the term.
    The penalty fades exponentially with measured base linear speed [m/s], angular
    speed [rad/s], and tilt from upright [rad], using the positive recovery scales.
    This permits recovery steps at a reduced cost; it does not guarantee push recovery.
    """
    if min(recovery_linear_velocity, recovery_angular_velocity, recovery_tilt) <= 0.0:
        raise ValueError("Standing contact penalty recovery scales must be positive.")
    standing = (torch.linalg.vector_norm(velocity_command[:, :2], dim=-1) < command_threshold) & (
        velocity_command[:, 2].abs() < command_threshold
    )
    base_tilt = torch.atan2(torch.linalg.vector_norm(projected_gravity_b[:, :2], dim=-1), -projected_gravity_b[:, 2])
    recovery = (
        torch.sum(base_lin_vel_b.square(), dim=-1) / recovery_linear_velocity**2
        + torch.sum(base_ang_vel_b.square(), dim=-1) / recovery_angular_velocity**2
        + (base_tilt / recovery_tilt).square()
    )
    unsupported_feet = (~foot_contact.bool()).sum(dim=-1)
    return unsupported_feet * standing * torch.exp(-recovery)


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    tanh_mult: float,
    asset_cfg: SceneEntityCfg,
    standing_position_foot_z: float = 0.039,
    height_sensor_cfg: SceneEntityCfg | None = None,
    ground_height_offset: float = 0.0,
    command_name: str | None = None,
    command_threshold: float = 0.05,
    foot_speed_threshold: float = 0.05,
) -> torch.Tensor:
    """Reward foot clearance relative to the mean scanned ground height.

    Args:
        env: Environment instance.
        target_height: Desired clearance above the standing foot height [m].
        std: Exponential denominator [m²], preserving the existing clearance reward.
        tanh_mult: Horizontal foot-speed multiplier [s/m].
        asset_cfg: Robot and foot bodies to measure.
        standing_position_foot_z: Standing foot body-origin height above the ground [m].
        height_sensor_cfg: Ray caster providing world-space ground hits. Without a sensor,
            the ground reference is world Z = 0, matching the existing reward.
        ground_height_offset: Optional offset to the ground height [m]. This is useful for accounting for additional layers
            such as sand or other terrain features above the base ground level.
        command_name: Optional velocity command name. When set, the reward is 0 while a non-zero planar velocity is
            commanded and all feet are standing still, and is unchanged otherwise. When ``None``, no gating is applied.
        command_threshold: Commanded planar speed above which the command counts as non-zero [m/s].
        foot_speed_threshold: Planar foot speed at or below which a foot counts as still [m/s].

    Returns:
        Velocity-weighted clearance reward, shape ``(num_envs,)``.
    """
    asset = env.scene[asset_cfg.name]
    ground_height = 0.0
    if height_sensor_cfg is not None:
        sensor = env.scene[height_sensor_cfg.name]
        ground_height = sensor.data.ray_hits_w.torch[..., 2].mean(dim=1, keepdim=True) + ground_height_offset
    height_error = torch.square(
        asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]
        - ground_height
        - (target_height + standing_position_foot_z)
    )
    velocity_weight = torch.tanh(
        tanh_mult * torch.linalg.vector_norm(asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2], dim=-1)
    )
    reward = torch.exp(-torch.sum(height_error * velocity_weight, dim=1) / std)
    if command_name is not None:
        command_speed = torch.linalg.vector_norm(env.command_manager.get_command(command_name)[:, :2], dim=-1)
        foot_speed = torch.linalg.vector_norm(asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2], dim=-1)
        commanded_to_move = command_speed > command_threshold
        feet_still = (foot_speed <= foot_speed_threshold).all(dim=-1)
        reward = reward * ~(commanded_to_move & feet_still)
    return reward
