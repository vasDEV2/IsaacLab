# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LiDAR observations expressed in the sensor frame at the measurement time."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import base_ang_vel as root_base_ang_vel
from isaaclab.envs.utils.io_descriptors import generic_io_descriptor, record_dtype, record_shape
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors import patterns
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@generic_io_descriptor(
    units="rad/s", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def finite_base_ang_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return root angular velocity with non-finite components replaced by zero."""
    return torch.nan_to_num(root_base_ang_vel(env, asset_cfg), nan=0.0, posinf=0.0, neginf=0.0)


class lidar_point_cloud(ManagerTermBase):
    """Select a fixed angular subset of hits and return ``(B, N, 4)`` XYZ + validity.

    XYZ is in meters relative to the sensor origin and axes at the last scan.
    Misses have zero XYZ and validity 0; hits have validity 1. This uses the common
    ray-caster data API, including its cached measurement pose, so a moving robot
    does not rotate an old scan using its newer pose between sensor updates.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._sensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        num_points = cfg.params["num_points"]
        if not 0 < num_points <= self._sensor.num_rays:
            raise ValueError(f"num_points must be in [1, {self._sensor.num_rays}], got {num_points}.")
        if self._sensor.cfg.ray_alignment != "base":
            raise ValueError("LiDAR point clouds require ray_alignment='base'.")
        self._indices = torch.linspace(0, self._sensor.num_rays - 1, num_points, device=env.device).long()

    def __call__(self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, num_points: int) -> torch.Tensor:
        data = self._sensor.data
        hits = data.ray_hits_w.torch[:, self._indices]
        relative = hits - data.pos_w.torch[:, None, :]
        valid = torch.isfinite(relative).all(dim=-1)
        valid &= torch.linalg.vector_norm(relative, dim=-1) <= self._sensor.cfg.max_distance
        relative = torch.where(valid.unsqueeze(-1), relative, 0.0)
        quat = data.quat_w.torch[:, None, :].expand(-1, num_points, -1)
        points = quat_apply_inverse(quat.reshape(-1, 4), relative.reshape(-1, 3)).reshape(-1, num_points, 3)
        return torch.cat((points, valid.unsqueeze(-1).to(points.dtype)), dim=-1)


class elevation_map(ManagerTermBase):
    """XYZ grid in meters relative to the scanner origin, with yaw-only axes.

    For this task the scanner origin is the torso-link origin. Grid axis 0 is
    forward X and axis 1 is left Y; Z is world-up height relative to that origin.
    Missed rays keep their grid XY and use ``missing_height`` for Z. The sensor's
    ray starts may be elevated above its origin without changing this reference.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._sensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        pattern = self._sensor.cfg.pattern_cfg
        if (
            self._sensor.cfg.ray_alignment != "yaw"
            or not isinstance(pattern, patterns.GridPatternCfg)
            or pattern.ordering != "yx"
            or tuple(pattern.direction) != (0.0, 0.0, -1.0)
        ):
            raise ValueError("Elevation maps require a yaw-aligned, downward grid with ordering='yx'.")
        starts, _ = pattern.func(pattern, env.device)
        self._shape = (starts[:, 0].unique().numel(), starts[:, 1].unique().numel(), 3)
        if starts.shape[0] != self._sensor.num_rays:
            raise ValueError("Elevation-map grid must match the sensor ray count.")
        self._missing = starts.clone()
        self._missing[:, 2] = cfg.params["missing_height"]
        self._shape_reported = False

    def __call__(self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, missing_height: float) -> torch.Tensor:
        data = self._sensor.data
        hits = data.ray_hits_w.torch
        sensor_pos = data.pos_w.torch
        sensor_quat = data.quat_w.torch
        quat_norm = torch.linalg.vector_norm(sensor_quat, dim=-1)
        valid_pose = (
            torch.isfinite(sensor_pos).all(dim=-1)
            & torch.isfinite(sensor_quat).all(dim=-1)
            & torch.isfinite(quat_norm)
            & (quat_norm > 1.0e-6)
        )
        valid = torch.isfinite(hits).all(dim=-1) & valid_pose.unsqueeze(-1)
        safe_pos = torch.where(valid_pose.unsqueeze(-1), sensor_pos, torch.zeros_like(sensor_pos))
        relative = torch.where(valid.unsqueeze(-1), hits - safe_pos[:, None, :], 0.0)
        valid &= torch.isfinite(relative).all(dim=-1)
        relative = torch.where(valid.unsqueeze(-1), relative, 0.0)
        identity_quat = torch.zeros_like(sensor_quat)
        identity_quat[:, 3] = 1.0
        normalized_quat = sensor_quat / quat_norm.clamp_min(1.0e-6).unsqueeze(-1)
        safe_quat = torch.where(valid_pose.unsqueeze(-1), normalized_quat, identity_quat)
        quat = yaw_quat(safe_quat)[:, None, :].expand(-1, hits.shape[1], -1)
        points = quat_apply_inverse(quat.reshape(-1, 4), relative.reshape(-1, 3)).reshape_as(hits)
        valid &= torch.isfinite(points).all(dim=-1)
        points = torch.where(valid.unsqueeze(-1), points, self._missing.unsqueeze(0))
        elevation_map_tensor = points.reshape(-1, *self._shape)
        if not self._shape_reported:
            print("[G1 perception array sizes] elevation-map observation pipeline:")
            for name, tensor in (
                ("ray_hits_w", hits),
                ("sensor_position_w", data.pos_w.torch),
                ("sensor_quaternion_w", data.quat_w.torch),
                ("valid_ray_mask", valid),
                ("relative_hits", relative),
                ("transformed_points", points),
                ("elevation_map", elevation_map_tensor),
            ):
                size_bytes = tensor.numel() * tensor.element_size()
                print(
                    f"  {name}: shape={tuple(tensor.shape)}, numel={tensor.numel()}, "
                    f"dtype={tensor.dtype}, device={tensor.device}, size={size_bytes / 1024**2:.2f} MiB"
                )
            self._shape_reported = True
        return elevation_map_tensor
