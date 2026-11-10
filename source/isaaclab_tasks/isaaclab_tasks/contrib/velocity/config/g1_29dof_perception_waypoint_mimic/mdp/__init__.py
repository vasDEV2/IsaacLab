# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pyramid-base waypoint command with clip imitation, and its observation and reward terms."""

from .command import (
    APPROACH,
    IMITATE,
    IMITATE_LOOSE,
    MODE_NAMES,
    WALK,
    PyramidMimicCommand,
    PyramidMimicCommandCfg,
    PyramidMimicSplitCommand,
    PyramidMimicSplitCommandCfg,
)
from .curriculums import pyramid_tile_curriculum
from .events import reset_joints_to_reference, reset_root_walk_or_reference
from .observations import mirror_waypoint_xy, waypoint_xy_b
from .terrains import (
    CurriculumPyramidTerrainCfg,
    PyramidCurriculumCfg,
    UniformPyramidStairsTerrainCfg,
    curriculum_pyramid_terrain,
    pyramid_curriculum_sub_terrains,
    pyramid_tile_geometry,
    uniform_pyramid_stairs_terrain,
)
from .rewards import (
    backward_velocity_l2,
    base_reached_bonus,
    excess_speed_l2,
    loose_anchor_ori_exp,
    loose_anchor_pos_exp,
    loose_body_ori_exp,
    loose_body_pos_exp,
    top_position_after_approach_exp,
)
from .terminations import bad_anchor_height_decaying, bad_anchor_tilt_decaying, bad_body_height_decaying
