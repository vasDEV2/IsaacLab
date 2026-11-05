# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.velocity.config.g1_29dof_perception_waypoint.mdp import WaypointGaitCommandCfg

from ..mdp import PyramidGoalCommandCfg


@configclass
class G1CommandsCfg:
    """The landing-center goal, and the walking intent derived from it for gait terms.

    There is no velocity command. Gait terms that gate on a velocity command read ``waypoint_gait``, which
    reports 0.8 m/s toward the goal and zero within its reach radius; it is neither observed nor tracked.
    """

    waypoint = PyramidGoalCommandCfg(debug_vis=True)
    waypoint_gait = WaypointGaitCommandCfg(waypoint_command_name="waypoint")
