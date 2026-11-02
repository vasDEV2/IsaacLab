# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Idealized Mid-360 and D435 sensors; mounting poses require hardware calibration."""

import math

import torch
from isaaclab_newton.renderers import NewtonWarpRendererCfg

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg, RayCasterCfg, patterns

# A regular angular grid approximates the field of view, not the Mid-360 scan timing/pattern.
MID360_LIDAR_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*torso_link",
    update_period=0.1,
    max_distance=40.0,
    ray_alignment="base",
    mesh_prim_paths=["/World/ground"],
    global_world_only=False,
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
    pattern_cfg=patterns.LidarPatternCfg(
        channels=40,
        vertical_fov_range=(-7.0, 52.0),
        horizontal_fov_range=(0.0, 360.0),
        horizontal_res=1.0,
    ),
    debug_vis=False,
)

# Attach to an existing moving body, rather than a nonexistent d435_link.
D435_CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso_link/d435_camera",
    update_period=1.0 / 30.0,
    width=640,
    height=480,
    data_types=["rgb", "depth"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=1.93,
        horizontal_aperture=3.68,
        clipping_range=(0.1, 10.0),
    ),
    offset=CameraCfg.OffsetCfg(pos=(0.15, 0.0, 0.1), convention="world"),
    renderer_cfg=NewtonWarpRendererCfg(),
)

# PHP-inspired training camera. World convention: +X forward, +Z up; positive Y pitch looks down.
# The torso-relative mount is illustrative and must be calibrated for hardware deployment.
DEPTH_CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso_link/depth_camera",
    update_period=0.02,
    width=106,
    height=60,
    data_types=["depth"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=1.0,
        horizontal_aperture=2.0 * math.tan(math.radians(89.5 / 2)),
        clipping_range=(0.01, 3.0),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.15, 0.0, 0.1),
        rot=(0.0, math.sin(math.radians(27.0 / 2)), 0.0, math.cos(math.radians(27.0 / 2))),
        convention="world",
    ),
    renderer_cfg=NewtonWarpRendererCfg(depth_clipping_behavior="max", enable_textures=False),
)


def elevation_grid_pattern(cfg: patterns.GridPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Cast from 20 m above the torso in the yaw frame, independent of torso pitch/roll."""
    starts, directions = patterns.grid_pattern(cfg, device)
    starts[:, 2] = 20.0
    return starts, directions
