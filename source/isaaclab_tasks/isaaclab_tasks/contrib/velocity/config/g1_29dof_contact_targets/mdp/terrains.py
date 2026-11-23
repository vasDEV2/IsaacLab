# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain with one column per motion clip, each tile holding that clip's terrain boxes (or flat ground)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.velocity.config.g1_29dof_goalconditioned_perception.mdp.terrains import (
    MotionStairsTerrainCfg,
)


def motion_flat_terrain(difficulty: float, cfg: MotionFlatTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Flat ground at the clip's ground height, with the clip's root path centered in the tile.

    The returned origin is the clip frame's origin (on the ground), so reference root poses map to the world by
    adding the environment origin only. ``difficulty`` is ignored.
    """
    with np.load(cfg.motion_file, allow_pickle=False) as data:
        root_xy = data["body_pos_w"][:, 0, :2]
        ground = float(data["ground_height"]) if "ground_height" in data else 0.0
    offset = -(root_xy.min(axis=0) + root_xy.max(axis=0)) / 2.0
    origin = np.array((cfg.size[0] / 2.0 + offset[0], cfg.size[1] / 2.0 + offset[1], 0.0))
    return [make_plane(cfg.size, ground, center_zero=False)], origin


@configclass
class MotionFlatTerrainCfg(SubTerrainBaseCfg):
    """Flat ground for a motion-library clip without terrain boxes."""

    function = motion_flat_terrain

    motion_file: str = MISSING
    """Path to the motion NPZ whose root path (body 0 of ``body_pos_w``) is centered in the tile."""


def motion_terrain_generator(motion_files: Sequence[str], num_rows: int = 1, size: float = 10.0) -> TerrainGeneratorCfg:
    """A curriculum-ordered generator whose column ``i`` holds the boxes of ``motion_files[i]``.

    The tile origin is the clip frame's origin, with the boxes (or, for a clip without boxes, its root path) centered
    in the tile, so clip poses map to the world by adding the environment origin. Rows are identical copies.
    """
    return TerrainGeneratorCfg(
        seed=42,
        size=(size, size),
        num_rows=num_rows,
        num_cols=len(motion_files),
        border_width=5.0,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        curriculum=True,  # columns follow the sub-terrain order
        use_cache=False,
        sub_terrains={f"clip_{i:02d}": _clip_terrain(path) for i, path in enumerate(motion_files)},
    )


def _clip_terrain(path: str) -> SubTerrainBaseCfg:
    """The clip's own terrain boxes, or flat ground for a clip without them."""
    with np.load(path, allow_pickle=False) as data:
        has_boxes = "terrain_position" in data and len(data["terrain_position"]) > 0
    if has_boxes:
        return MotionStairsTerrainCfg(proportion=1.0, motion_file=path)
    return MotionFlatTerrainCfg(proportion=1.0, motion_file=path)
