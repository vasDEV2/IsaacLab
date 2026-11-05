# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.velocity.config.g1_29dof_perception_waypoint.mdp as waypoint_mdp
import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp as g1_mdp


@configclass
class G1CurriculumCfg:
    """Curriculum terms for the MDP."""

    # promote after reaching the landing; demote after ending over half the start distance away
    terrain_levels = CurrTerm(func=waypoint_mdp.terrain_levels_waypoint, params={"command_name": "waypoint"})
    feet_air_time_weight = CurrTerm(
        func=g1_mdp.ramp_reward_weight,
        params={"term_name": "feet_air_time", "weight_0": 2.0, "weight_1": 5.0, "step_0": 0, "step_1": 15000 * 24},
    )
    foot_clearance_weight = CurrTerm(
        func=g1_mdp.ramp_reward_weight,
        params={"term_name": "foot_clearance", "weight_0": 5.0, "weight_1": 10.0, "step_0": 0, "step_1": 15000 * 24},
    )
