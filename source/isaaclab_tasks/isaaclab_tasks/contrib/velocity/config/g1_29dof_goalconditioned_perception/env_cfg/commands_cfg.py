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
        # imitation only for 2,000 iterations (24 steps each), then down to half the episodes over the next 5,000
        warmup_steps=48_000,
        transition_steps=120_000,
        final_imitation_probability=0.5,
        # imitation on the clips' staircase; task episodes on height-scaled copies, one row up per success
        imitation_terrain="motion_stairs",
        task_terrain="task_stairs",
        task_terrain_curriculum=True,
        task_max_init_level=0,
        # task episodes always start standing at rest in front of the stairs
        task_start_offset=(-0.5, 0.0, 0.0),
        task_start_position_noise=0.1,
        task_start_yaw_noise=0.1,
        # move the reference endpoint safely into the 2.5 m square landing
        task_goal_offset=(0.75, 0.0, 0.0),
        goal_position_noise=0.35,
    )
