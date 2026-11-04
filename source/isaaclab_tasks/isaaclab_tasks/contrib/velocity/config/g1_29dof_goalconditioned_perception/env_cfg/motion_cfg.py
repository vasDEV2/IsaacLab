# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pyramid stair clips and body sets shared by the command, observations, rewards, and terrain."""

from pathlib import Path

STAIR_MOTION_DIR = Path(__file__).resolve().parents[1] / "data" / "supplied_stairs_pyramid_stand"

# Kimodo ``supplied_stairs`` rotated to the four sides of one pyramid (2.5 m square top landing), each followed by
# a two-second settling transition and a two-second standing hold: 374 frames at 50 Hz (7.48 s). All four clips
# share the pyramid geometry stored in their ``terrain_*`` fields.
STAIR_MOTION_FILES = [str(STAIR_MOTION_DIR / f"motion_yaw_{yaw:03d}.npz") for yaw in (0, 90, 180, 270)]

TRACKED_BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]
"""Reference bodies tracked by the imitation rewards and observed by the critic."""

FOOT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
END_EFFECTOR_BODY_NAMES = [*FOOT_BODY_NAMES, "left_wrist_yaw_link", "right_wrist_yaw_link"]
