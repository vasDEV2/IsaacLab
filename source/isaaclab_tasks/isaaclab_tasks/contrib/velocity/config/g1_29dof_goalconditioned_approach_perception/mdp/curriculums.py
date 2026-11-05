# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task stair height curriculum on climbed episodes, and phase logging."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from .approach import CLIMB, PHASE_NAMES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .approach import ApproachMotionCommand


def climbed_task_terrain_levels(env: ManagerBasedRLEnv, env_ids: torch.Tensor, command_name: str) -> torch.Tensor:
    """Move a task tile one row up after the task episode reaches its destination and survives, else one row down.

    As :func:`~isaaclab_tasks.contrib.velocity.config.g1_29dof_goalconditioned_perception.mdp.task_terrain_levels`,
    but only task episodes that reached the waypoint and climbed count; failing the approach does not change
    the stair height.
    """
    command: ApproachMotionCommand = env.command_manager.get_term(command_name)
    completed = (env.episode_length_buf[env_ids] > 0) & ~command.is_imitation[env_ids]
    completed &= command.episode_phase[env_ids] == CLIMB
    success = command.goal_reached[env_ids] & ~env.termination_manager.terminated[env_ids]
    change = torch.where(success, 1, -1) * completed
    command.task_level[env_ids] = (command.task_level[env_ids] + change).clamp(0, command.num_levels - 1)
    return command.task_level.float().mean()


def approach_stats(env: ManagerBasedRLEnv, env_ids: Sequence[int], command_name: str) -> dict:
    """Log the fraction of environments in each phase and the imitation probability of new episodes."""
    command: ApproachMotionCommand = env.command_manager.get_term(command_name)
    stats = {
        f"{name}_fraction": (command.episode_phase == phase).float().mean().item()
        for phase, name in enumerate(PHASE_NAMES)
    }
    stats["climbing_enabled"] = float(command.climbing_enabled)
    stats["imitation_probability"] = command.sampling_probability
    return stats
