# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AME PPO for the waypoint-then-imitation task: the waypoint task's networks and settings."""

from isaaclab.utils.configclass import configclass

from ...g1_29dof_perception_waypoint.agents.compositional_ppo_cfg import G1RoughPerceptionWaypointPPORunnerCfg


@configclass
class G1RoughPerceptionWaypointMimicPPORunnerCfg(G1RoughPerceptionWaypointPPORunnerCfg):
    """Waypoint AME PPO; ``num_steps_per_env`` must match ``env_cfg.STEPS_PER_ITERATION``."""

    experiment_name = "g1_29dof_perception_ame_waypoint_mimic"


@configclass
class G1RoughPerceptionWaypointMimicSplitPPORunnerCfg(G1RoughPerceptionWaypointMimicPPORunnerCfg):
    """The same AME PPO for the split task, logged separately."""

    experiment_name = "g1_29dof_perception_ame_waypoint_mimic_split"
