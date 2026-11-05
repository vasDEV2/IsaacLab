# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Approach time-out."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .approach import APPROACH

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .approach import ApproachMotionCommand


def approach_time_out(env: ManagerBasedRLEnv, command_name: str, max_time_s: float) -> torch.Tensor:
    """Approach episodes that have not reached the waypoint within ``max_time_s`` [s]."""
    command: ApproachMotionCommand = env.command_manager.get_term(command_name)
    return (command.episode_phase == APPROACH) & (env.episode_length_buf * env.step_dt >= max_time_s)
