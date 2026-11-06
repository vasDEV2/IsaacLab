# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 AME perception that reaches the base of the motion clips' pyramid, then imitates the clip up it."""

import gymnasium as gym

for suffix, cfg_suffix in (("", ""), ("-Play", "_PLAY")):
    gym.register(
        id=f"IsaacContrib-Velocity-Rough-G1-29dof-Perception-Waypoint-Mimic{suffix}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:G1RoughPerceptionWaypointMimicEnvCfg{cfg_suffix}",
            "rsl_rl_cfg_entry_point": (
                f"{__name__}.agents.compositional_ppo_cfg:G1RoughPerceptionWaypointMimicPPORunnerCfg"
            ),
        },
    )
