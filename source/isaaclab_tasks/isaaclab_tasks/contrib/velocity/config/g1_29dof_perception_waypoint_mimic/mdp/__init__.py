# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pyramid-base waypoint command with clip imitation, and its observation and reward terms."""

from .command import APPROACH, IMITATE, MODE_NAMES, WALK, PyramidMimicCommand, PyramidMimicCommandCfg
from .observations import mirror_waypoint_xy, waypoint_xy_b
from .rewards import backward_velocity_l2, base_reached_bonus
from .terminations import bad_anchor_height_decaying, bad_anchor_tilt_decaying, bad_body_height_decaying
