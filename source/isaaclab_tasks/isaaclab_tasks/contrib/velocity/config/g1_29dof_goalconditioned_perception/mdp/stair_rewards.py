# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Alternative terrain references and clearance incentives for stair experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _local_terrain_samples(
    points_w: torch.Tensor, hits_w: torch.Tensor, local_radius: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select finite scanner hits within ``local_radius`` of each queried body in XY.

    Queries have shape (B, bodies, 3); hits have shape (B, rays, 3). All-miss scans
    return no valid samples. A query near a stair edge is limited by grid resolution.
    """
    if local_radius <= 0.0:
        raise ValueError(f"local_radius must be positive, got {local_radius}.")
    distances = (points_w[:, :, None, :2] - hits_w[:, None, :, :2]).square().sum(dim=-1)
    valid = torch.isfinite(hits_w).all(dim=-1)[:, None, :] & (distances <= local_radius**2)
    return torch.where(valid[..., None], hits_w[:, None, :, :], 0.0), valid


def _local_ground_height(points_w: torch.Tensor, hits_w: torch.Tensor, local_radius: float) -> torch.Tensor:
    """Return the mean finite terrain height in a local XY neighborhood.

    Queries have shape (B, bodies, 3); hits have shape (B, rays, 3). An all-miss
    or empty local neighborhood falls back to world Z=0.
    """
    samples, valid = _local_terrain_samples(points_w, hits_w, local_radius)
    count = valid.sum(dim=-1).clamp_min(1)
    return samples[..., 2].sum(dim=-1) / count


def local_base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    local_radius: float = 0.3,
) -> torch.Tensor:
    """Penalize root height relative to nearby scanned terrain."""
    root = env.scene[asset_cfg.name].data.root_pos_w.torch
    ground = _local_ground_height(root[:, None], env.scene[sensor_cfg.name].data.ray_hits_w.torch, local_radius)[:, 0]
    return (root[:, 2] - ground - target_height).square()


def feet_stumble(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    horizontal_vertical_ratio: float = 4.0,
    min_horizontal_force: float = 1.0,
) -> torch.Tensor:
    """Penalize a foot contact dominated by horizontal force, as in AME_Locomotion.

    Compare world XY and world Z components of the normal contact force [N].
    The force floor suppresses near-zero sensor noise. Stair-edge contacts produce
    horizontal surface-normal forces; ordinary ground support predominantly acts
    along world Z. The result is one penalty per environment if either foot trips.
    """
    forces = env.scene.sensors[sensor_cfg.name].data.net_normal_forces_w.torch[:, sensor_cfg.body_ids]
    horizontal = torch.linalg.vector_norm(forces[..., :2], dim=-1)
    tripped = (horizontal > horizontal_vertical_ratio * forces[..., 2].abs()) & (horizontal > min_horizontal_force)
    return tripped.any(dim=-1).float()


def local_stance_foot_angle_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg,
    angle_tolerance: float = 0.0,
    local_radius: float = 0.3,
) -> torch.Tensor:
    """Penalize stance-foot pitch relative to a plane fitted from nearby scanner hits."""
    feet = env.scene[asset_cfg.name].data.body_pos_w.torch[:, asset_cfg.body_ids]
    foot_quat = env.scene[asset_cfg.name].data.body_quat_w.torch[:, asset_cfg.body_ids]
    hits = env.scene[height_sensor_cfg.name].data.ray_hits_w.torch
    samples, valid_hits = _local_terrain_samples(feet, hits, local_radius)
    count = valid_hits.sum(dim=-1, keepdim=True)
    mean = samples.sum(dim=-2, keepdim=True) / count.clamp_min(1).unsqueeze(-1)
    centered = torch.where(valid_hits.unsqueeze(-1), samples - mean, 0.0)
    covariance = centered.transpose(-1, -2) @ centered
    xx, xy, xz = covariance[..., 0, :].unbind(dim=-1)
    yy, yz = covariance[..., 1, 1], covariance[..., 1, 2]
    determinant = xx * yy - xy.square()
    valid_plane = (count[..., 0] >= 3) & (determinant > (1.0e-6 * xx * yy).clamp_min(1.0e-12))
    denominator = determinant.clamp_min(1.0e-12)
    slope = torch.stack(((yy * xz - xy * yz) / denominator, (xx * yz - xy * xz) / denominator), dim=-1)

    forward_axis = torch.zeros_like(foot_quat[..., :3])
    forward_axis[..., 0] = 1.0
    forward = quat_apply(foot_quat, forward_axis)
    horizontal_length = torch.linalg.vector_norm(forward[..., :2], dim=-1)
    heading = forward[..., :2] / horizontal_length.clamp_min(1.0e-6).unsqueeze(-1)
    terrain_angle = torch.atan((slope * heading).sum(dim=-1))
    foot_angle = torch.atan2(forward[..., 2], horizontal_length)
    error = (foot_angle - terrain_angle).abs()
    penalty = (error - angle_tolerance).clamp_min(0.0).square()

    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contact = contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids] > 0.0
    valid = valid_plane & torch.isfinite(penalty)
    return torch.where(contact & valid, penalty, 0.0).sum(dim=-1)


def stair_foot_clearance(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    tanh_mult: float,
    asset_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg,
    standing_position_foot_z: float = 0.039,
    ground_height_offset: float = 0.0,
    command_name: str = "base_velocity",
    local_radius: float = 0.3,
    motion_gate: bool = True,
) -> torch.Tensor:
    """Reward foot clearance relative to the nearby scanned terrain.

    ``std`` retains the original reward's squared-meter denominator. Motion gating
    makes motionless feet earn zero instead of maximum reward. Terrain height is
    averaged only from finite scanner hits within ``local_radius`` of each foot.
    """
    asset = env.scene[asset_cfg.name]
    feet = asset.data.body_pos_w.torch[:, asset_cfg.body_ids]
    hits = env.scene[height_sensor_cfg.name].data.ray_hits_w.torch
    ground = _local_ground_height(feet, hits, local_radius)
    error = (feet[..., 2] - ground - ground_height_offset - target_height - standing_position_foot_z).square()
    speed = torch.linalg.vector_norm(asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2], dim=-1)
    speed_weight = torch.tanh(tanh_mult * speed)
    reward = torch.exp(-(error * speed_weight).sum(dim=1) / std)
    if motion_gate:
        command_speed = torch.linalg.vector_norm(env.command_manager.get_command(command_name)[:, :2], dim=-1)
        reward = reward * speed_weight.mean(dim=1) * (command_speed > 0.05)
    return reward


def world_height_convergence(
    env: ManagerBasedRLEnv,
    target_root_height: float = 0.75,
    height_scale: float = 0.1,
    max_vertical_speed: float = 1.0,
    command_name: str = "base_velocity",
    command_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the rate at which the pelvis closes its world-Z distance to the target [m/s].

    With ``error = root_z - target_root_height`` [m], the reward is
    ``-tanh(error / height_scale) * clip(root_vz, +-max_vertical_speed)``: positive while
    descending from above the target (pyramid top, +1.2 m) or rising from below it
    (inverted pyramid bottom, -1.2 m), negative when moving away, and zero when standing
    or walking at constant height. ``target_root_height`` is the standing pelvis height
    above the surface [m], so Z = 0 is the flat border between tiles; use 0.0 to drive
    the pelvis itself to Z = 0. The vertical speed is clipped so a fall off an edge cannot
    pay more than a controlled descent, and the reward is zero for near-zero commands.
    """
    asset = env.scene[asset_cfg.name]
    error = asset.data.root_pos_w.torch[:, 2] - target_root_height
    vertical_speed = asset.data.root_lin_vel_w.torch[:, 2].clamp(-max_vertical_speed, max_vertical_speed)
    command_speed = torch.linalg.vector_norm(env.command_manager.get_command(command_name)[:, :2], dim=-1)
    reward = -torch.tanh(error / height_scale) * vertical_speed * (command_speed > command_threshold)
    return torch.where(torch.isfinite(reward), reward, 0.0)


def world_height_error(
    env: ManagerBasedRLEnv,
    target_root_height: float = 0.75,
    max_error: float = 2.0,
    command_name: str = "base_velocity",
    command_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the pelvis world-Z distance [m] to ``target_root_height``, for use with a negative weight.

    Pair with :func:`world_height_convergence`: that term pays for closing the distance, this one
    charges every step the distance is maintained. The error is clipped at ``max_error`` [m] and is
    zero for near-zero commands, where the robot is told to stand and cannot fix its height.
    """
    asset = env.scene[asset_cfg.name]
    error = (asset.data.root_pos_w.torch[:, 2] - target_root_height).abs().clamp(max=max_error)
    command_speed = torch.linalg.vector_norm(env.command_manager.get_command(command_name)[:, :2], dim=-1)
    return torch.where(torch.isfinite(error), error, 0.0) * (command_speed > command_threshold)


def root_height_world_l2(
    env: ManagerBasedRLEnv,
    target_height: float = 0.75,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared error [m^2] between the root world Z and ``target_height``, for use with a negative weight."""
    root_z = env.scene[asset_cfg.name].data.root_pos_w.torch[:, 2]
    return torch.square(root_z - target_height)
