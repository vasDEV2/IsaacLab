# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stair terrains: the perception up/down curriculum, and a pyramid shaped like the reference clips' staircase."""

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


def clip_pyramid_stairs_terrain(
    difficulty: float, cfg: ClipPyramidStairsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """A four-sided pyramid shaped like the reference clips' staircase, with randomized rises and widths.

    The pyramid is ``num_rises`` stacked square boxes centered in the tile on a ground plane at Z = 0; the top box
    is a ``landing_width`` square landing. The mean rise is ``rise_range`` interpolated by ``difficulty``, and each
    rise adds uniform ``rise_jitter`` noise; the tile draws one step width from ``step_width_range`` and each step
    adds uniform ``step_width_jitter`` noise. The irregular steps follow the clips, whose rises span 0.16-0.23 m and
    step widths 0.18-0.23 m. The origin is the landing center, so the tile origin's Z is the landing height.
    """
    rng = np.random
    mean_rise = cfg.rise_range[0] + difficulty * (cfg.rise_range[1] - cfg.rise_range[0])
    rises = np.clip(mean_rise + rng.uniform(-cfg.rise_jitter, cfg.rise_jitter, cfg.num_rises), cfg.min_rise, None)
    width = rng.uniform(*cfg.step_width_range)
    widths = np.clip(
        width + rng.uniform(-cfg.step_width_jitter, cfg.step_width_jitter, cfg.num_rises - 1), cfg.min_step_width, None
    )
    heights = np.cumsum(rises)
    # box i reaches heights[i] and extends past the landing by the widths of the steps above it
    half_widths = cfg.landing_width / 2.0 + np.concatenate((np.cumsum(widths[::-1])[::-1], [0.0]))
    if half_widths[0] > min(cfg.size) / 2.0:
        raise ValueError("The pyramid base is wider than the sub-terrain; enlarge size or shrink the widths.")
    center = np.array((cfg.size[0] / 2.0, cfg.size[1] / 2.0))
    meshes = [make_plane(cfg.size, 0.0, center_zero=False)]
    for half_width, height in zip(half_widths, heights):
        transform = trimesh.transformations.translation_matrix((center[0], center[1], height / 2.0))
        meshes.append(trimesh.creation.box((2.0 * half_width, 2.0 * half_width, height), transform))
    return meshes, np.array((center[0], center[1], heights[-1]))


@configclass
class ClipPyramidStairsTerrainCfg(SubTerrainBaseCfg):
    """Pyramid stairs matching the reference clips' structure, harder (taller rises) with difficulty.

    The defaults reproduce the clips' layout: six rises to a 2.5 m square landing (base 4.5 m wide), with the
    clip's mean rise (0.19 m) at difficulty 0.8 and its step width (0.20 m) at the middle of ``step_width_range``.
    """

    function = clip_pyramid_stairs_terrain

    num_rises: int = 6
    """Rises from the ground to the landing; the clips have five steps and a landing."""

    rise_range: tuple[float, float] = (0.05, 0.22)
    """Mean rise [m] at difficulty 0 and 1."""

    rise_jitter: float = 0.03
    """Uniform per-rise noise [m]."""

    min_rise: float = 0.02

    step_width_range: tuple[float, float] = (0.18, 0.26)
    """Range [m] of the per-tile step width."""

    step_width_jitter: float = 0.02
    """Uniform per-step width noise [m]."""

    min_step_width: float = 0.15

    landing_width: float = 2.5
    """Side [m] of the square top landing."""
