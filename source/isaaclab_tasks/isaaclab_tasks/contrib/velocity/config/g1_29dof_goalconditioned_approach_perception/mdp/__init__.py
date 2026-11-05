# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Approach-phase command, reset, termination, curriculum, and observation terms."""

from .approach import APPROACH, CLIMB, PHASE_NAMES, WALK, ApproachMotionCommand, ApproachMotionCommandCfg
from .curriculums import approach_stats, climbed_task_terrain_levels
from .events import reset_at_spawn
from .observations import approach_phase, mirror_heading, root_pos_tile_b, target_heading
from .terminations import approach_time_out, climb_time_out
