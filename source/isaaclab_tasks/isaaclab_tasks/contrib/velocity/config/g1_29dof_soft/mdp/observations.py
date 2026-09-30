# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
from isaaclab_newton.sensors import NewtonRaycastSensor

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera, ContactSensor

# from isaaclab.sensors import Camera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.envs.utils.io_descriptors import (
    generic_io_descriptor,
    record_body_names,
    record_dtype,
    record_shape,
)

"""
body kinematics.
"""


@generic_io_descriptor(observation_type="BodyState", on_inspect=[record_shape, record_dtype, record_body_names])
def foot_air_time_hybrid(
    env: ManagerBasedRLEnv,
    rigid_contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    soft_contact_sensor_name: str = "physics_callback",
) -> torch.Tensor:
    """Hybrid foot air time observation selecting from the authoritative solver."""
    rigid_contact_sensor: ContactSensor = env.scene.sensors[rigid_contact_sensor_cfg.name]
    soft_contact_sensor = env.action_manager.get_term(soft_contact_sensor_name).contact_solver

    air_time_rigid = rigid_contact_sensor.data.current_air_time.torch[:, rigid_contact_sensor_cfg.body_ids]
    air_time_soft = soft_contact_sensor.data.current_air_time

    return torch.where(soft_contact_sensor.data.is_sensor_active, air_time_soft, air_time_rigid)


def foot_contact_hybrid(
    env: ManagerBasedRLEnv,
    rigid_contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    soft_contact_sensor_name: str = "physics_callback",
    rigid_force_threshold: float = 1.0,
    soft_force_threshold: float = 1.0,
) -> torch.Tensor:
    """Hybrid foot contact observation selecting from the authoritative solver."""
    rigid_contact_sensor: ContactSensor = env.scene.sensors[rigid_contact_sensor_cfg.name]
    soft_contact_sensor = env.action_manager.get_term(soft_contact_sensor_name).contact_solver

    rigid_contact_forces = rigid_contact_sensor.data.net_forces_w.torch[:, rigid_contact_sensor_cfg.body_ids, :]
    soft_contact_forces = soft_contact_sensor.contact_wrench[:, :, :3]
    rigid_contact = (torch.norm(rigid_contact_forces, dim=-1) > rigid_force_threshold).float()
    soft_contact = (torch.norm(soft_contact_forces, dim=-1) > soft_force_threshold).float()

    return torch.where(soft_contact_sensor.data.is_sensor_active, soft_contact, rigid_contact)


def randome_obs(env: ManagerBasedRLEnv):
    return torch.ones((env.num_envs, 10))


def mid360_lidar_ranges(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("mid360_lidar"),
    clip_max: float | None = None,
) -> torch.Tensor:
    """Return ranges [m], shape (num_envs, num_rays); clip_max [m] defaults to sensor range."""
    sensor: NewtonRaycastSensor = env.scene.sensors[sensor_cfg.name]
    distances = sensor.data.ray_distances  # (N, B), inf where no hit
    max_distance = sensor.cfg.max_distance if clip_max is None else clip_max
    # Misses use the configured range; changing the sensor also changes preprocessing.
    distances = torch.nan_to_num(distances, nan=max_distance, posinf=max_distance, neginf=0.0)
    distances = torch.clamp(distances, min=0.0, max=max_distance)
    return distances


def d435_rgb(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("d435_camera"),
    normalize: bool = True,
) -> torch.Tensor:
    """RGB image from the D435, shape (num_envs, H, W, 3)."""
    sensor: Camera = env.scene.sensors[sensor_cfg.name]
    rgb = sensor.data.output["rgb"][..., :3].clone()
    if normalize:
        rgb = rgb.float() / 255.0
    return rgb


def d435_depth(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("d435_camera"),
    clip_max: float | None = None,
) -> torch.Tensor:
    """Return depth [m], shape (num_envs, H, W, 1); clip_max [m] defaults to the camera far plane."""
    sensor: Camera = env.scene.sensors[sensor_cfg.name]
    if clip_max is None:
        clip_max = sensor.cfg.spawn.clipping_range[1]
    depth = sensor.data.output["depth"].clone()
    # replace inf/NaN misses (rays that hit nothing) with max range
    depth = torch.nan_to_num(depth, posinf=clip_max, nan=clip_max)
    depth = torch.clamp(depth, max=clip_max)
    return depth


def foot_contact_forces_hybrid(
    env: ManagerBasedRLEnv,
    rigid_contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    soft_contact_sensor_name: str = "physics_callback",
    rigid_force_filter_threshold: float = 1.0,
    soft_force_filter_threshold: float = 1.0,
) -> torch.Tensor:
    """Hybrid foot contact forces observation selecting from the authoritative solver."""
    rigid_contact_sensor: ContactSensor = env.scene.sensors[rigid_contact_sensor_cfg.name]
    soft_contact_sensor = env.action_manager.get_term(soft_contact_sensor_name).contact_solver

    max_force = 1000.0
    rigid_contact_forces = rigid_contact_sensor.data.net_forces_w.torch[:, rigid_contact_sensor_cfg.body_ids, :].clamp(
        -max_force, max_force
    )
    soft_contact_forces = soft_contact_sensor.contact_wrench[:, :, :3].clamp(-max_force, max_force)

    is_soft = soft_contact_sensor.data.is_sensor_active  # [B, N_feet]

    forces = torch.where(is_soft.unsqueeze(-1), soft_contact_forces, rigid_contact_forces)

    threshold = (
        torch.where(is_soft, soft_force_filter_threshold, rigid_force_filter_threshold)
        .unsqueeze(-1)
        .expand(-1, -1, 3)
        .reshape(env.num_envs, -1)
    )
    forces = forces.reshape(env.num_envs, -1)
    forces = forces * (forces > threshold).float()

    return torch.sign(forces) * torch.log1p(torch.abs(forces))


def foot_contact_forces_raw_hybrid(
    env: ManagerBasedRLEnv,
    rigid_contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    soft_contact_sensor_name: str = "physics_callback",
    rigid_force_filter_threshold: float = 1.0,
    soft_force_filter_threshold: float = 1.0,
) -> torch.Tensor:
    """Hybrid foot contact forces observation selecting from the authoritative solver."""
    rigid_contact_sensor: ContactSensor = env.scene.sensors[rigid_contact_sensor_cfg.name]
    soft_contact_sensor = env.action_manager.get_term(soft_contact_sensor_name).contact_solver

    rigid_contact_forces = rigid_contact_sensor.data.net_forces_w.torch[:, rigid_contact_sensor_cfg.body_ids, :]
    soft_contact_forces = soft_contact_sensor.contact_wrench[:, :, :3]

    is_soft = soft_contact_sensor.data.is_sensor_active  # [B, N_feet]

    forces = torch.where(is_soft.unsqueeze(-1), soft_contact_forces, rigid_contact_forces)

    return forces


_SHAPE_ID_CACHE: dict[tuple[str, str], torch.Tensor] = {}
"""Backend shape indices per (asset, body selection), resolved once on first use."""


def _mean_body_friction(env: ManagerBasedRLEnv, asset: Articulation, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Mean contact friction coefficient [-] of the collision shapes attached to ``asset_cfg.body_ids``.

    Newton stores a single friction coefficient per collision shape, so the friction is read from
    the articulation view instead of the PhysX material buffer (which had separate static and
    dynamic friction columns).
    """
    from isaaclab_newton.physics.newton_manager import NewtonManager  # noqa: PLC0415

    shape_ids = _resolve_shape_ids(env, asset, asset_cfg)
    friction = wp.to_torch(asset.root_view.get_attribute("shape_material_mu", NewtonManager.get_model())[:, 0])
    return friction[:, shape_ids].mean(dim=-1).to(env.device)


def _resolve_shape_ids(env: ManagerBasedRLEnv, asset: Articulation, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Resolve the backend-ordered shape indices of the bodies selected by ``asset_cfg``."""
    cache_key = (asset_cfg.name, str(asset_cfg.body_names), str(asset_cfg.body_ids))
    shape_ids = _SHAPE_ID_CACHE.get(cache_key)
    if shape_ids is None:
        # the manager only resolves entity configs passed as term parameters, so a default one
        # still carries body names instead of ids
        if asset_cfg.body_names is not None and isinstance(asset_cfg.body_ids, slice):
            asset_cfg.resolve(env.scene)
        num_shapes_per_body = asset.backend_num_shapes_per_body
        shape_ids_list: list[int] = []
        for body_id in asset.map_body_ids_to_backend(asset_cfg.body_ids):
            start_idx = sum(num_shapes_per_body[:body_id])
            shape_ids_list.extend(range(start_idx, start_idx + num_shapes_per_body[body_id]))
        shape_ids = torch.tensor(shape_ids_list, dtype=torch.long)
        _SHAPE_ID_CACHE[cache_key] = shape_ids
    return shape_ids


def terrain_material_parameters_all_hybrid(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
    rigid_contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    soft_contact_sensor_name: str = "physics_callback",
) -> torch.Tensor:
    """Hybrid terrain material parameters observation selecting from the authoritative solver."""
    asset = env.scene[asset_cfg.name]
    soft_contact_sensor = env.action_manager.get_term(soft_contact_sensor_name).contact_solver

    on_soft_ground = soft_contact_sensor.data.is_sensor_active.any(dim=-1).float()

    mu_rigid = 0.9
    friction_rigid = _mean_body_friction(env, asset, asset_cfg)  # friction of the ankles
    rho_c_rigid = 3000.0
    rho_c_max = 3000.0

    friction_coef = soft_contact_sensor.terrain_friction * on_soft_ground + (1 - on_soft_ground) * friction_rigid
    rho_c = soft_contact_sensor.terrain_density * on_soft_ground + (1 - on_soft_ground) * rho_c_rigid
    mu_int = soft_contact_sensor.terrain_stiffness * on_soft_ground + (1 - on_soft_ground) * mu_rigid

    return torch.stack([friction_coef, rho_c / rho_c_max, mu_int], dim=-1)


def terrain_material_parameters_hybrid(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
    rigid_contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    soft_contact_sensor_name: str = "physics_callback",
) -> torch.Tensor:
    """Hybrid terrain material parameters observation selecting from the authoritative solver."""
    asset = env.scene[asset_cfg.name]
    term = env.action_manager.get_term(soft_contact_sensor_name)
    soft_contact_sensor = term.contact_solver

    on_soft_ground = soft_contact_sensor.data.is_sensor_active.any(dim=-1).float()

    if term.backend in ["2D", "2D-warp", "3D", "3D-warp"]:
        mu_rigid = 0.9
        friction_rigid = _mean_body_friction(env, asset, asset_cfg)  # friction of the ankles
        rho_c_rigid = 3000.0
        rho_c_max = 3000.0

        friction_coef = soft_contact_sensor.terrain_friction * on_soft_ground + (1 - on_soft_ground) * friction_rigid
        rho_c = soft_contact_sensor.terrain_density * on_soft_ground + (1 - on_soft_ground) * rho_c_rigid
        mu_int = soft_contact_sensor.terrain_stiffness * on_soft_ground + (1 - on_soft_ground) * mu_rigid

        # NOTE: since both rho_c and mu_int affect stiffness, we use media dependent scaling factor as observation
        g = 9.81
        xi = rho_c * g * (894.0 * (mu_int**3.0) - 386.0 * (mu_int**2.0) + 89.0 * mu_int)
        xi_max = rho_c_max * g * (894.0 * (mu_rigid**3.0) - 386.0 * (mu_rigid**2.0) + 89.0 * mu_rigid)
        stiffness = xi / xi_max  # normalize

        # return torch.stack([friction_coef, rho_c / rho_c_max, mu_int], dim=-1)
        # return torch.stack([friction_coef, stiffness], dim=-1)
        return stiffness.view(-1, 1)

    elif term.backend in ["cone-drft", "cone-drft-multipoint"]:
        sigma_flat_rigid = 10.0e6
        sigma_cone_rigid = 0.6e6
        friction_rigid = _mean_body_friction(env, asset, asset_cfg)  # friction of the ankles
        rho_c_rigid = 3000.0
        rho_c_max = 3000.0

        friction_coef = soft_contact_sensor.terrain_friction * on_soft_ground + (1 - on_soft_ground) * friction_rigid
        rho_c = soft_contact_sensor.terrain_density * on_soft_ground + (1 - on_soft_ground) * rho_c_rigid

        sigma_flat = soft_contact_sensor.terrain_sigma_flat * on_soft_ground + (1 - on_soft_ground) * sigma_flat_rigid
        sigma_cone = soft_contact_sensor.terrain_sigma_cone * on_soft_ground + (1 - on_soft_ground) * sigma_cone_rigid

        return torch.stack(
            [friction_coef, rho_c / rho_c_max, sigma_flat / sigma_cone_rigid, sigma_cone / sigma_cone_rigid], dim=-1
        )
        # return sigma_flat/sigma_cone_rigid
    else:
        raise ValueError(f"Unsupported backend: {term.backend}")
