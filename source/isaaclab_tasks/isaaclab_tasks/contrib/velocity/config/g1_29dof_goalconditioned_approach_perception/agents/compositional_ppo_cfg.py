# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AME PPO for the approach task: the goal-conditioned task's networks and PPO settings."""

from isaaclab.utils.configclass import configclass

from ...g1_29dof_goalconditioned_perception.agents.compositional_ppo_cfg import (
    G1RoughGoalConditionedPerceptionPPORunnerCfg,
)


@configclass
class G1RoughGoalConditionedApproachPerceptionPPORunnerCfg(G1RoughGoalConditionedPerceptionPPORunnerCfg):
    """Goal-conditioned AME PPO; ``num_steps_per_env`` must match ``env_cfg.STEPS_PER_ITERATION``."""

    experiment_name = "g1_29dof_goalconditioned_approach_perception_ame_stairs"
