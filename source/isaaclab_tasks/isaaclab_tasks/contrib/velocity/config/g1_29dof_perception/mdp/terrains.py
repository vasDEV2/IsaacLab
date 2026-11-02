# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Staircase terrain whose row index selects flat, then alternating up/down pyramids with more steps."""

from __future__ import annotations

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
