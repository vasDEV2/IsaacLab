# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase logging."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from .approach import PHASE_NAMES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .approach import ApproachMotionCommand


def approach_stats(env: ManagerBasedRLEnv, env_ids: Sequence[int], command_name: str) -> dict:
    """Log the fraction of environments in each phase and whether new episodes climb."""
    command: ApproachMotionCommand = env.command_manager.get_term(command_name)
    stats = {
        f"{name}_fraction": (command.episode_phase == phase).float().mean().item()
        for phase, name in enumerate(PHASE_NAMES)
    }
    stats["climbing_enabled"] = float(command.climbing_enabled)
    return stats
