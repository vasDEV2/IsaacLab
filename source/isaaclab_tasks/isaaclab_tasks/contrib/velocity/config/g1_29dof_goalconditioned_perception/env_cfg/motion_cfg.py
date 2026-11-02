# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stair-climbing reference clip shared by the command, observations, rewards, and terrain."""

import os
from pathlib import Path

# Kimodo ``supplied_stairs``: 174 frames at 50 Hz (3.48 s), six ~0.19 m rises over ~0.2 m treads, 1.2 m wide,
# climbed along +X at ~0.33 m/s from the ground to a 1.15 m platform. The clip data is not vendored; by default it
# is read from a motion_library checkout next to IsaacLab (https://github.com/jnskkmhr/motion_library).
STAIR_MOTION_FILE = os.environ.get(
    "G1_STAIR_MOTION_FILE",
    str(Path(__file__).resolve().parents[8].parent / "motion_library/data/kimodo/supplied_stairs/motion.npz"),
)

STAIR_MOTION_DURATION_S = 3.48
"""Clip length [s]; episodes end on the clip's last frame before this time-out."""

STAIR_MOTION_FORWARD_SPEED = 0.33
"""Mean horizontal pelvis speed [m/s] along +X over the clip."""
