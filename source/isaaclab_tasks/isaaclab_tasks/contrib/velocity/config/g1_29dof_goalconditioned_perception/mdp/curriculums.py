# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-environment task stair height driven by goal-reaching success."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .motion import GoalMotionCommand


def task_terrain_levels(env: ManagerBasedRLEnv, env_ids: torch.Tensor, command_name: str) -> torch.Tensor:
    """Move an environment's task tile one row up (taller stairs) after it reaches the goal and survives.

    Any other finished task episode moves it one row down. Imitation episodes and the first reset leave it
    unchanged. The level applies at the environment's next task episode. It is not saved with checkpoints, so a
    resumed run restarts from the initial rows.
    """
    command: GoalMotionCommand = env.command_manager.get_term(command_name)
    completed = (env.episode_length_buf[env_ids] > 0) & ~command.is_imitation[env_ids]
    success = command.goal_reached[env_ids] & ~env.termination_manager.terminated[env_ids]
    change = torch.where(success, 1, -1) * completed
    command.task_level[env_ids] = (command.task_level[env_ids] + change).clamp(0, command.num_levels - 1)
    return command.task_level.float().mean()
