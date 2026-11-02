# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp as g1_mdp

from ..mdp.motion import StairMotionCommandCfg
from .motion_cfg import STAIR_MOTION_FILE, STAIR_MOTION_FORWARD_SPEED


@configclass
class G1CommandsCfg:
    """Command specifications for the MDP."""

    # Declared first: its reset writes the reference state that the velocity command's heading reset reads.
    motion = StairMotionCommandCfg(
        asset_name="robot",
        motion_file=STAIR_MOTION_FILE,
        min_remaining_time_s=1.0,
        root_y_range=(-0.1, 0.1),
    )
    # The clip climbs straight along +X at a fixed pace: no standing, lateral, or turning commands.
    base_velocity = g1_mdp.UniformVelocityYawCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=g1_mdp.UniformVelocityYawCommandCfg.Ranges(
            lin_vel_x=(STAIR_MOTION_FORWARD_SPEED, STAIR_MOTION_FORWARD_SPEED),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
    )
