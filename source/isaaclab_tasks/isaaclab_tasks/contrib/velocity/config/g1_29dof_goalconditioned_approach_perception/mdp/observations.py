# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Privileged phase observation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .approach import PHASE_NAMES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .approach import ApproachMotionCommand


def approach_phase(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """One-hot phase ``(walk, approach, climb)``, which decides the reward. Shape is (num_envs, 3)."""
    command: ApproachMotionCommand = env.command_manager.get_term(command_name)
    return torch.nn.functional.one_hot(command.episode_phase, len(PHASE_NAMES)).float()
