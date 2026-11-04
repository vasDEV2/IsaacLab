# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-environment task start difficulty driven by goal-reaching success."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .motion import GoalMotionCommand


def task_reset_difficulty(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str,
    success_step: float = 0.02,
    failure_step: float = 0.05,
) -> torch.Tensor:
    """Raise an environment's difficulty after it reaches the goal and survives a task episode.

    Any other finished task episode lowers it. Imitation episodes and the first reset leave it unchanged.
    The difficulty is not saved with checkpoints, so a resumed run restarts from zero.
    """
    command: GoalMotionCommand = env.command_manager.get_term(command_name)
    completed = (env.episode_length_buf[env_ids] > 0) & ~command.is_imitation[env_ids]
    success = command.goal_reached[env_ids] & ~env.termination_manager.terminated[env_ids]
    change = torch.where(success, success_step, -failure_step) * completed
    command.task_reset_difficulty[env_ids] = (command.task_reset_difficulty[env_ids] + change).clamp(0.0, 1.0)
    return command.task_reset_difficulty.mean()
