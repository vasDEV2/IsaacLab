# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from ..mdp.motion import GoalMotionCommandCfg
from .motion_cfg import FOOT_BODY_NAMES, STAIR_MOTION_FILES, TRACKED_BODY_NAMES


@configclass
class G1CommandsCfg:
    """A destination root pose, with imitation and task episodes sampled on a schedule."""

    motion = GoalMotionCommandCfg(
        asset_name="robot",
        motion_files=STAIR_MOTION_FILES,
        body_names=TRACKED_BODY_NAMES,
        anchor_body_name="torso_link",
        foot_body_names=FOOT_BODY_NAMES,
        # imitation only for 5,000 iterations, then down to half the episodes over the next 5,000
        warmup_steps=120_000,
        transition_steps=120_000,
        final_imitation_probability=0.5,
        # expand toward standing starts only after successful, non-failing task episodes
        task_reset_curriculum=True,
        task_start_offset=(-0.9, 0.0, 0.0),
        initial_position_noise=0.3,
        initial_yaw_noise=0.3,
        # move the reference endpoint safely into the 2.5 m square landing
        task_goal_offset_initial=(0.2, 0.0, 0.0),
        task_goal_offset=(0.75, 0.0, 0.0),
        goal_position_noise=0.35,
        task_velocity_probability=0.5,
        task_velocity_range={"x": (-0.15, 0.15), "y": (-0.15, 0.15), "yaw": (-0.2, 0.2)},
    )
