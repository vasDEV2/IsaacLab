# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum term placing each resetting robot on the pyramid tile of its coming episode."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .command import PyramidMimicSplitCommand


def pyramid_tile_curriculum(env: ManagerBasedRLEnv, env_ids: torch.Tensor, command_name: str) -> dict[str, float]:
    """Update levels and assign tiles through :meth:`~.command.PyramidMimicSplitCommand.assign_tiles`.

    Returns the mean and highest curriculum level and the fraction of curriculum episodes, for logging.
    """
    command: PyramidMimicSplitCommand = env.command_manager.get_term(command_name)
    command.assign_tiles(env_ids)
    level = command.curriculum_level.float()
    return {
        "level_mean": level.mean().item(),
        "level_max": level.max().item(),
        "curriculum_fraction": command.loose.float().mean().item(),
    }
