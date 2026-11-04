# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 goal-conditioned stair climbing with AME height-map encoding, from imitation and task episodes."""

import gymnasium as gym

for suffix, cfg_suffix in (("", ""), ("-Play", "_PLAY")):
    gym.register(
        id=f"IsaacContrib-Velocity-Rough-G1-29dof-GoalConditioned-Perception{suffix}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1RoughGoalConditionedPerceptionEnvCfg{cfg_suffix}",
            "rsl_rl_cfg_entry_point": (
                f"{__name__}.agents.compositional_ppo_cfg:G1RoughGoalConditionedPerceptionPPORunnerCfg"
            ),
        },
    )
