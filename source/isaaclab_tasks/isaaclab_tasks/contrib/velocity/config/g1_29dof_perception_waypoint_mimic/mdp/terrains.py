# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pyramid stairs placed where the motion clips' pyramid stands: a parametric one, and the clip's own pyramid with
its top width and step heights varied by a curriculum."""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.velocity.config.g1_29dof_goalconditioned_perception.mdp.terrains import (
    MotionStairsTerrainCfg,
)


def load_clip_boxes(motion_file: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Terrain box centers and sizes [m] in the clip frame, and the ground height [m], of a motion clip."""
    with np.load(motion_file, allow_pickle=False) as data:
        positions = np.array(data["terrain_position"], dtype=float)
        dimensions = np.array(data["terrain_dimension"], dtype=float)
        ground = float(data["ground_height"]) if "ground_height" in data else 0.0
    return positions, dimensions, ground


"""
Pyramid curriculum.
"""


@configclass
class PyramidCurriculumCfg:
    """Grid of pyramid tiles for the curriculum regime, shared by the terrain and the command.

    Columns ``[0, num_clip_cols)`` hold the clip's own pyramid (the earlier regime); the others hold the clip's
    pyramid with every box widened by ``delta`` (the top width changes by ``delta``, the step depths do not) and every
    box height scaled by ``scale``. Each curriculum column has a fixed value ``c`` in [-1, 1], spread evenly, and a
    fixed shuffled value ``r`` in [-1, 1]. Row (level) 0 is the clip's pyramid; rows ``1..width_levels`` widen or
    narrow the top by ``c`` times a growing fraction of :attr:`top_width_delta_range`; the next ``height_levels`` rows
    take the top width ``r`` across the full range and scale the step heights by ``c`` times a growing fraction of
    :attr:`step_scale_range`.
    """

    start_step: int = 4001 * 24
    """Control step from which episodes are split between the earlier regime and the curriculum."""

    curriculum_probability: float = 0.5
    """Fraction of episodes reset after :attr:`start_step` that run the curriculum; the rest run the earlier regime
    (approach, then imitate) on the clip's pyramid."""

    replay_probability: float = 0.2
    """Fraction of curriculum episodes placed on a random level at or below the robot's own."""

    promotion_probability: float = 0.25
    """Chance that a curriculum episode reaching the top waypoint moves the robot up a level, to keep it slow."""

    demotion_phase: float = 0.5
    """A curriculum episode that ends before this fraction of the clip without reaching the top moves down a
    level."""

    num_clip_cols: int = 4
    """Columns of the clip's own pyramid."""

    num_cols: int = 20
    """Columns in total."""

    width_levels: int = 8
    """Levels over which the top width range opens."""

    height_levels: int = 8
    """Levels over which the step height range opens, after the width levels."""

    top_width_delta_range: tuple[float, float] = (-1.5, 1.5)
    """Narrowest and widest change of the top width [m] (the clip's top is 2.5 m wide)."""

    step_scale_range: tuple[float, float] = (0.63, 1.58)
    """Lowest and highest step height scale (the clip's steps average 0.19 m)."""

    seed: int = 0
    """Seed of the shuffled top widths of the height levels."""

    @property
    def num_rows(self) -> int:
        return 1 + self.width_levels + self.height_levels


def pyramid_tile_geometry(cfg: PyramidCurriculumCfg, row: int, col: int) -> tuple[float, float]:
    """Top width change [m] and step height scale of the curriculum tile at ``row`` and ``col``."""
    num = cfg.num_cols - cfg.num_clip_cols
    if col < cfg.num_clip_cols or num < 1:
        return 0.0, 1.0
    j = col - cfg.num_clip_cols
    spread = -1.0 + 2.0 * j / max(num - 1, 1)
    shuffled = -1.0 + 2.0 * np.random.default_rng(cfg.seed).permutation(num)[j] / max(num - 1, 1)
    low, high = cfg.top_width_delta_range
    low_scale, high_scale = cfg.step_scale_range

    def width(value: float) -> float:
        return value * (high if value > 0.0 else -low)

    if row <= cfg.width_levels:
        return row / max(cfg.width_levels, 1) * width(spread), 1.0
    fraction = (row - cfg.width_levels) / max(cfg.height_levels, 1)
    scale = spread * (high_scale - 1.0 if spread > 0.0 else 1.0 - low_scale)
    return width(shuffled), 1.0 + fraction * scale


def curriculum_pyramid_terrain(
    difficulty: float, cfg: CurriculumPyramidTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """The clip's pyramid with the top width and step heights of its curriculum tile, in the clip frame.

    Every box grows by the tile's ``delta`` in both horizontal sizes and keeps its center, so the steps keep their
    depths; box heights scale by the tile's ``scale``. The pyramid stays centered in the tile and the returned origin
    is the clip frame's origin, as for :class:`MotionStairsTerrainCfg`.
    """
    num_rows = cfg.curriculum.num_rows
    row = min(int(difficulty * num_rows + 1e-6), num_rows - 1)
    delta, scale = pyramid_tile_geometry(cfg.curriculum, row, cfg.column)
    positions, dimensions, ground = load_clip_boxes(cfg.motion_file)
    lower = (positions[:, :2] - dimensions[:, :2] / 2.0).min(axis=0)
    upper = (positions[:, :2] + dimensions[:, :2] / 2.0).max(axis=0)
    origin = np.array((cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.0))
    origin[:2] -= (lower + upper) / 2.0
    positions[:, 2] = ground + (positions[:, 2] - ground) * scale
    dimensions[:, 2] *= scale
    dimensions[:, :2] += delta
    meshes = [make_plane(cfg.size, ground, center_zero=False)]
    for position, dimension in zip(positions, dimensions):
        meshes.append(trimesh.creation.box(dimension, trimesh.transformations.translation_matrix(origin + position)))
    return meshes, origin


@configclass
class CurriculumPyramidTerrainCfg(SubTerrainBaseCfg):
    """One curriculum column of :class:`PyramidCurriculumCfg`; ``difficulty`` selects the row."""

    function = curriculum_pyramid_terrain

    motion_file: str = MISSING
    """Motion NPZ holding the clip's terrain boxes."""

    curriculum: PyramidCurriculumCfg = MISSING

    column: int = MISSING
    """Column of the generator this configuration fills."""


def pyramid_curriculum_sub_terrains(cfg: PyramidCurriculumCfg, motion_file: str) -> dict[str, SubTerrainBaseCfg]:
    """One sub-terrain per column, in column order, for a curriculum :class:`~isaaclab.terrains.TerrainGenerator`.

    The generator must have ``curriculum=True``, ``num_rows == cfg.num_rows`` and ``num_cols == cfg.num_cols``.
    """
    sub_terrains = {}
    for col in range(cfg.num_cols):
        if col < cfg.num_clip_cols:
            sub_terrains[f"clip_{col:02d}"] = MotionStairsTerrainCfg(proportion=1.0, motion_file=motion_file)
        else:
            sub_terrains[f"curriculum_{col:02d}"] = CurriculumPyramidTerrainCfg(
                proportion=1.0, motion_file=motion_file, curriculum=cfg, column=col
            )
    return sub_terrains


"""
Parametric pyramid.
"""


def uniform_pyramid_stairs_terrain(
    difficulty: float, cfg: UniformPyramidStairsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Ground plane plus ``num_steps`` square steps of equal rise and depth, in the motion clip's frame.

    The pyramid is centered where the clip's own pyramid is, and the returned origin is the clip frame's origin,
    as for the clip's terrain, so clip poses (spawns and waypoints) map to the same world positions.
    ``difficulty`` is ignored.
    """
    with np.load(cfg.motion_file, allow_pickle=False) as data:
        positions = np.array(data["terrain_position"], dtype=float)
        dimensions = np.array(data["terrain_dimension"], dtype=float)
        ground = float(data["ground_height"]) if "ground_height" in data else 0.0
    # the clip pyramid's XY center, in the clip frame
    lower = (positions[:, :2] - dimensions[:, :2] / 2.0).min(axis=0)
    upper = (positions[:, :2] + dimensions[:, :2] / 2.0).max(axis=0)
    center = (lower + upper) / 2.0
    # the clip frame origin, with the clip pyramid centered in the tile
    origin = np.array((cfg.size[0] / 2.0 - center[0], cfg.size[1] / 2.0 - center[1], 0.0))
    meshes = [make_plane(cfg.size, ground, center_zero=False)]
    for i in range(cfg.num_steps):
        width = cfg.top_width + 2.0 * cfg.step_width * (cfg.num_steps - 1 - i)
        height = cfg.step_height * (i + 1)
        position = (origin[0] + center[0], origin[1] + center[1], ground + height / 2.0)
        meshes.append(
            trimesh.creation.box((width, width, height), trimesh.transformations.translation_matrix(position))
        )
    return meshes, origin


@configclass
class UniformPyramidStairsTerrainCfg(SubTerrainBaseCfg):
    """Square pyramid of equal steps, centered on a motion clip's pyramid.

    The footprint is ``top_width + 2 * step_width * (num_steps - 1)`` and the top is ``num_steps * step_height``
    above the ground. The defaults approximate the clips' pyramid (4.5 m footprint, 1.14 m top).
    """

    function = uniform_pyramid_stairs_terrain

    motion_file: str = MISSING
    """Motion NPZ whose terrain boxes give the pyramid's center and the clip frame."""

    num_steps: int = 6
    """Number of steps, the top landing included."""

    step_height: float = 0.19
    """Rise of every step [m]."""

    step_width: float = 0.2
    """Depth of every step below the top landing [m]."""

    top_width: float = 2.5
    """Side length of the square top landing [m]."""
