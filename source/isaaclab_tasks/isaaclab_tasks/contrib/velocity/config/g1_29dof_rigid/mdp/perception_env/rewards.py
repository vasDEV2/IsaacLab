# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


"""
whole-body centroidal momentum penalties.
"""


def _feet_rpy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Compute the yaw angles of feet.

    Args:
    env: The environment.
    asset_cfg: Configuration for the asset.
    feet_index: Optional list of indices specifying which feet to consider.
            If None, all bodies specified in asset_cfg.body_ids are used.

    Returns:
    torch.Tensor: Yaw angles of feet in radians.
    """
    # Get the entity
    entity = env.scene[asset_cfg.name]

    # Get the body IDs to use
    feet_quat = entity.data.body_quat_w.torch[:, asset_cfg.body_ids, :]
    # feet_quat = entity.data.body_quat_w.torch[:, feet_index, :]
    original_shape = feet_quat.shape
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(feet_quat.reshape(-1, 4))

    roll = (roll + torch.pi) % (2 * torch.pi) - torch.pi
    pitch = (pitch + torch.pi) % (2 * torch.pi) - torch.pi
    # yaw = (yaw + torch.pi) % (2*torch.pi) - torch.pi

    return roll.reshape(original_shape[0], -1), pitch.reshape(original_shape[0], -1), yaw.reshape(original_shape[0], -1)


def reward_feet_pitch_contact_hybrid(
    env: ManagerBasedRLEnv,
    rigid_contact_sensor_cfg: SceneEntityCfg,
    soft_contact_sensor_name: str = "physics_callback",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize foot pitch angle only at the moment of first contact.

    During swing the penalty is zero, so knee flexion is not penalized.
    At touchdown, pitch² is penalized to encourage flat foot landings.
    """
    rigid_contact_sensor: ContactSensor = env.scene.sensors[rigid_contact_sensor_cfg.name]
    soft_contact_sensor = env.action_manager.get_term(soft_contact_sensor_name).contact_solver

    rigid_first_contact = rigid_contact_sensor.compute_first_contact(env.step_dt)[
        :, rigid_contact_sensor_cfg.body_ids
    ]  # [B, N]
    soft_first_contact = soft_contact_sensor.compute_first_contact(env.step_dt)  # [B, N]
    first_contact = torch.where(soft_contact_sensor.data.is_sensor_active, soft_first_contact, rigid_first_contact)

    _, feet_pitch, _ = _feet_rpy(env, asset_cfg=asset_cfg)  # [B, N_feet]

    # penalize pitch² only on the landing step
    return torch.sum(torch.square(feet_pitch) * first_contact.float(), dim=-1)


def feet_air_time_positive_biped_hybrid(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    rigid_contact_sensor_cfg: SceneEntityCfg,
    soft_contact_sensor_name: str = "physics_callback",
    velocity_threshold: float = 0.05,
) -> torch.Tensor:
    """Hybrid biped air time reward selecting timing from the authoritative contact solver.

    Uses is_sensor_active (computed from force history in physics_callback) to pick
    rigid vs. soft solver per foot, avoiding stale zeros from the inactive solver.
    """
    rigid_contact_sensor: ContactSensor = env.scene.sensors[rigid_contact_sensor_cfg.name]
    soft_contact_sensor = env.action_manager.get_term(soft_contact_sensor_name).contact_solver

    # per-solver timings
    air_time_rigid = rigid_contact_sensor.data.current_air_time.torch[:, rigid_contact_sensor_cfg.body_ids]
    air_time_soft = soft_contact_sensor.data.current_air_time
    contact_time_rigid = rigid_contact_sensor.data.current_contact_time.torch[:, rigid_contact_sensor_cfg.body_ids]
    contact_time_soft = soft_contact_sensor.data.current_contact_time

    # select timing from the authoritative solver
    air_time = torch.where(soft_contact_sensor.data.is_sensor_active, air_time_soft, air_time_rigid)
    contact_time = torch.where(soft_contact_sensor.data.is_sensor_active, contact_time_soft, contact_time_rigid)

    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)

    linear_norm = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    angular_norm = torch.abs(env.command_manager.get_command(command_name)[:, 2])
    total_norm = linear_norm + angular_norm
    reward *= total_norm > velocity_threshold

    return reward


def feet_slide_hybrid(
    env: ManagerBasedRLEnv,
    rigid_contact_sensor_cfg: SceneEntityCfg,
    soft_contact_sensor_name: str = "physics_callback",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    rigid_contact_threshold: float = 1.0,
    soft_contact_threshold: float = 5.0,
) -> torch.Tensor:
    """Hybrid feet slide penalty selecting contact from the authoritative solver."""
    rigid_contact_sensor: ContactSensor = env.scene.sensors[rigid_contact_sensor_cfg.name]
    soft_contact_sensor = env.action_manager.get_term(soft_contact_sensor_name).contact_solver

    # for contact detection, use latest data
    rigid_contact = (
        rigid_contact_sensor.data.net_forces_w.torch[:, rigid_contact_sensor_cfg.body_ids, 2] > rigid_contact_threshold
    )
    soft_contact = soft_contact_sensor.data.net_forces_w[:, :, 2] > soft_contact_threshold

    # select contact from authoritative solver
    contacts = torch.where(soft_contact_sensor.data.is_sensor_active, soft_contact, rigid_contact)

    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
    return torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)


def no_fly_hybrid(
    env: ManagerBasedRLEnv,
    rigid_contact_sensor_cfg: SceneEntityCfg,
    soft_contact_sensor_name: str = "physics_callback",
    rigid_contact_threshold: float = 5.0,
    soft_contact_threshold: float = 5.0,
    command_name: str = "base_velocity",
    velocity_threshold: float = 1.5,
) -> torch.Tensor:
    """Hybrid no-fly penalty selecting contact from the authoritative solver."""
    rigid_contact_sensor: ContactSensor = env.scene.sensors[rigid_contact_sensor_cfg.name]
    soft_contact_sensor = env.action_manager.get_term(soft_contact_sensor_name).contact_solver

    # for contact detection, use latest data
    rigid_contact = (
        rigid_contact_sensor.data.net_forces_w.torch[:, rigid_contact_sensor_cfg.body_ids, 2] > rigid_contact_threshold
    )
    soft_contact = soft_contact_sensor.data.net_forces_w[:, :, 2] > soft_contact_threshold

    # select contact from authoritative solver
    is_contact = torch.where(soft_contact_sensor.data.is_sensor_active, soft_contact, rigid_contact)

    linear_norm = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    is_active = (linear_norm < velocity_threshold).float()

    reward = (torch.sum(is_contact, dim=-1) < 0.5).float()
    return reward * is_active


def reward_soft_landing_hybrid(
    env: ManagerBasedRLEnv,
    rigid_contact_sensor_cfg: SceneEntityCfg,
    soft_contact_sensor_name: str = "physics_callback",
    command_name: str = "base_velocity",
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Hybrid soft landing penalty selecting forces from the authoritative solver.

    Penalize high impact forces at landing to encourage soft footfalls.
    """
    rigid_contact_sensor: ContactSensor = env.scene.sensors[rigid_contact_sensor_cfg.name]
    soft_contact_sensor = env.action_manager.get_term(soft_contact_sensor_name).contact_solver

    # per-solver forces
    rigid_forces = rigid_contact_sensor.data.net_forces_w.torch[:, rigid_contact_sensor_cfg.body_ids, :]  # [B, N, 3]
    soft_forces = soft_contact_sensor.data.net_forces_w  # [B, N, 3]

    # per-solver first_contact
    rigid_first_contact = rigid_contact_sensor.compute_first_contact(env.step_dt)[
        :, rigid_contact_sensor_cfg.body_ids
    ]  # [B, N]
    soft_first_contact = soft_contact_sensor.compute_first_contact(env.step_dt)  # [B, N]

    is_soft = soft_contact_sensor.data.is_sensor_active  # [B, N_feet]

    # select forces and first_contact from authoritative solver
    forces = torch.where(is_soft.unsqueeze(-1), soft_forces, rigid_forces)
    force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
    first_contact = torch.where(is_soft, soft_first_contact, rigid_first_contact)
    landing_impact = force_magnitude * first_contact.float()  # [B, N]
    cost = torch.sum(landing_impact, dim=1)  # [B]

    num_landings = torch.sum(first_contact.float())
    mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
    env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force

    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        linear_norm = torch.norm(command[:, :2], dim=1)
        angular_norm = torch.abs(command[:, 2])
        active = ((linear_norm + angular_norm) > command_threshold).float()
        cost = cost * active
    return cost
