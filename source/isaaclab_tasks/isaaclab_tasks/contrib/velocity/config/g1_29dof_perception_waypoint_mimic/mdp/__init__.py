# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pyramid-base waypoint command with clip imitation, and its observation and reward terms."""

from .command import (
    APPROACH,
    IMITATE,
    MODE_NAMES,
    WALK,
    PyramidMimicCommand,
    PyramidMimicCommandCfg,
    PyramidMimicSplitCommand,
    PyramidMimicSplitCommandCfg,
)
from .events import reset_joints_to_reference, reset_root_walk_or_reference
from .observations import mirror_waypoint_xy, waypoint_xy_b
from .rewards import backward_velocity_l2, base_reached_bonus, excess_speed_l2, top_position_after_approach_exp
from .terminations import bad_anchor_height_decaying, bad_anchor_tilt_decaying, bad_body_height_decaying
