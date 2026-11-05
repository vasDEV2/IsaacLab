# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stair terrains: a flat/up/down pyramid curriculum, and the staircase paired with a motion-library clip."""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import trimesh

from isaaclab.terrains import (
    MeshInvertedPyramidStairsTerrainCfg,
    MeshPlaneTerrainCfg,
    MeshPyramidStairsTerrainCfg,
    SubTerrainBaseCfg,
)
from isaaclab.terrains.trimesh.mesh_terrains import (
    flat_terrain,
    inverted_pyramid_stairs_terrain,
    pyramid_stairs_terrain,
)
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils.configclass import configclass


def max_stair_steps(cfg: StairsCurriculumTerrainCfg) -> int:
    """Number of rises from the ground to the central platform of the full-size pyramid.

    ``pyramid_stairs_terrain`` builds ``num_steps`` rings and a central platform one rise above the last ring,
    so the robot climbs ``num_steps + 1`` rises.
    """
    inner = cfg.size[0] - 2 * cfg.border_width
    return int((inner - cfg.platform_width) // (2 * cfg.step_width) + 1) + 1


def num_stair_levels(cfg: StairsCurriculumTerrainCfg) -> int:
    """Number of curriculum levels: one flat level plus an up and a down level per step count."""
    return 1 + 2 * max_stair_steps(cfg)


def stairs_curriculum_terrain(
    difficulty: float, cfg: StairsCurriculumTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Level 0 is flat, then level 1 = up 1 rise, 2 = down 1 rise, 3 = up 2 rises, 4 = down 2 rises, ...

    ``difficulty`` is the generator's ``(row + U[0, 1)) / num_rows``, so the row (level) is recovered by
    flooring ``difficulty * num_levels``. This requires ``TerrainGeneratorCfg.num_rows == num_stair_levels(cfg)``.
    """
    num_levels = num_stair_levels(cfg)
    level = min(int(difficulty * num_levels + 1e-6), num_levels - 1)
    if level == 0:
        return flat_terrain(0.0, MeshPlaneTerrainCfg(size=cfg.size))

    num_rises = (level + 1) // 2
    num_steps = num_rises - 1  # rings built by the base functions; the central platform adds the last rise
    go_up = level % 2 == 1
    # the base functions derive the ring count from platform_width; pick one in the middle of the interval
    # that yields exactly ``num_steps`` (the central platform is size - 2 * num_steps * step_width wide).
    inner = cfg.size[0] - 2 * cfg.border_width
    platform_width = inner - (2 * num_steps - 1) * cfg.step_width
    if go_up:
        stair_cfg = MeshPyramidStairsTerrainCfg(
            size=cfg.size,
            border_width=cfg.border_width,
            step_height_range=cfg.step_height_range,
            step_width=cfg.step_width,
            platform_width=platform_width,
        )
        return pyramid_stairs_terrain(0.0, stair_cfg)
    stair_cfg = MeshInvertedPyramidStairsTerrainCfg(
        size=cfg.size,
        border_width=cfg.border_width,
        step_height_range=cfg.step_height_range,
        step_width=cfg.step_width,
        platform_width=platform_width,
    )
    return inverted_pyramid_stairs_terrain(0.0, stair_cfg)


@configclass
class StairsCurriculumTerrainCfg(SubTerrainBaseCfg):
    """Flat -> up 1 -> down 1 -> up 2 -> down 2 ... up to the full pyramid (``platform_width`` sets the max steps)."""

    function = stairs_curriculum_terrain

    border_width: float = 0.0
    step_height_range: tuple[float, float] = (0.10, 0.10)
    step_width: float = 0.30
    platform_width: float = 1.2


def motion_stairs_height_scale(cfg: MotionStairsTerrainCfg, level: int) -> float:
    """Height scale of the clip's terrain boxes on curriculum ``level`` (row); 1 without ``height_scale_range``."""
    if cfg.height_scale_range is None:
        return 1.0
    low, high = cfg.height_scale_range
    return low + (high - low) * level / max(cfg.num_levels - 1, 1)


def motion_stairs_terrain(difficulty: float, cfg: MotionStairsTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Ground plane plus the terrain boxes stored with the motion clip, in the clip's own frame.

    The returned origin is the clip frame's origin (on the ground), so reference root poses map to the
    world by adding the environment origin only. Without ``clip_offset``, the boxes' XY bounds are centered
    in the tile. Without ``height_scale_range``, ``difficulty`` is ignored and the clip fixes the geometry.
    Otherwise the row ``floor(difficulty * num_levels)`` scales every box height (and so every rise) by
    :func:`motion_stairs_height_scale`, keeping the footprint, so clip start poses still fit the tile.
    This requires ``TerrainGeneratorCfg.num_rows == num_levels``.
    """
    with np.load(cfg.motion_file, allow_pickle=False) as data:
        positions = np.array(data["terrain_position"], dtype=float)
        dimensions = np.array(data["terrain_dimension"], dtype=float)
        quats = data.get("terrain_quat")
        ground = float(data["ground_height"]) if "ground_height" in data else 0.0
    level = min(int(difficulty * cfg.num_levels + 1e-6), cfg.num_levels - 1)
    scale = motion_stairs_height_scale(cfg, level)
    # boxes rest on the ground; scale their heights about it
    positions[:, 2] = ground + (positions[:, 2] - ground) * scale
    dimensions[:, 2] *= scale
    if cfg.clip_offset is None:
        lower = (positions[:, :2] - dimensions[:, :2] / 2.0).min(axis=0)
        upper = (positions[:, :2] + dimensions[:, :2] / 2.0).max(axis=0)
        offset = -(lower + upper) / 2.0
    else:
        offset = np.asarray(cfg.clip_offset)
    origin = np.array((cfg.size[0] / 2.0 + offset[0], cfg.size[1] / 2.0 + offset[1], 0.0))
    meshes = [make_plane(cfg.size, ground, center_zero=False)]
    for i, (position, dimension) in enumerate(zip(positions, dimensions)):
        transform = trimesh.transformations.translation_matrix(origin + position)
        if quats is not None:
            x, y, z, w = quats[i]
            transform = transform @ trimesh.transformations.quaternion_matrix((w, x, y, z))
        meshes.append(trimesh.creation.box(dimension, transform))
    return meshes, origin


@configclass
class MotionStairsTerrainCfg(SubTerrainBaseCfg):
    """Staircase reproduced from a motion-library clip's terrain boxes."""

    function = motion_stairs_terrain

    motion_file: str = MISSING
    """Path to the motion NPZ holding ``terrain_position`` and ``terrain_dimension`` box arrays."""

    clip_offset: tuple[float, float] | None = None
    """XY placement [m] of the clip frame origin relative to the sub-terrain center; None centers the boxes."""

    height_scale_range: tuple[float, float] | None = None
    """Box height scale on the first and last curriculum row; None keeps the clip's heights on every row."""

    num_levels: int = 1
    """Curriculum rows of the generator, used to decode the row from ``difficulty``."""
