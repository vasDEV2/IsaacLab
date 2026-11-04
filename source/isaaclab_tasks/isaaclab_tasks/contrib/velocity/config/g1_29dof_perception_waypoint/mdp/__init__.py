# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Waypoint command, reset, reward, and curriculum terms."""

from .curriculums import terrain_levels_waypoint
from .events import reset_root_at_waypoint_start
from .observations import root_pos_tile_b
from .rewards import (
    waypoint_bearing_linear,
    waypoint_height_exp,
    waypoint_position_exp,
    waypoint_progress,
    waypoint_stand_still,
)
from .symmetry import mirror_waypoint
from .waypoint import WaypointCommand, WaypointCommandCfg, WaypointGaitCommand, WaypointGaitCommandCfg
