# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reference clips used by the motion-matching style reward."""

from pathlib import Path

# The goal-conditioned task's Kimodo pyramid climbs: one climb rotated onto the four faces of a pyramid with a
# 2.5 m landing, then a synthesized settle and standing hold (374 frames at 50 Hz). The rotations share their
# joint trajectories, so the style library deduplicates them.
STYLE_MOTION_DIR = (
    Path(__file__).resolve().parents[2]
    / "g1_29dof_goalconditioned_perception"
    / "data"
    / "supplied_stairs_pyramid_stand"
)
STYLE_MOTION_FILES = [str(STYLE_MOTION_DIR / f"motion_yaw_{yaw:03d}.npz") for yaw in (0, 90, 180, 270)]

STYLE_CLIP_FRAMES = 174
"""Captured climb frames per clip. Later frames settle into the standing pose, which a robot at rest would match."""
