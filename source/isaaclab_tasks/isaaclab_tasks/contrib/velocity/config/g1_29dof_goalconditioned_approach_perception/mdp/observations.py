# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tile position and privileged phase observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils import math as math_utils

from .approach import PHASE_NAMES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .approach import ApproachMotionCommand


def approach_phase(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """One-hot phase ``(walk, approach, climb)``, which decides the reward. Shape is (num_envs, 3)."""
    command: ApproachMotionCommand = env.command_manager.get_term(command_name)
    return torch.nn.functional.one_hot(command.episode_phase, len(PHASE_NAMES)).float()


def root_pos_tile_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Root position [m] in the root yaw frame: XY from the tile center, Z above the flat ground. Shape is (N, 3).

    This is the waypoint task's ``root_pos_tile_b``. There the environment origin is the tile center; here it is
    the clip frame's origin, 2.3 m from the pyramid center, so the tile center comes from the motion command.
    """
    command: ApproachMotionCommand = env.command_manager.get_term(command_name)
    data = command.robot.data
    position = data.root_pos_w.torch - command.tile_center_w
    return math_utils.quat_apply_inverse(math_utils.yaw_quat(data.root_quat_w.torch), position)


def target_heading(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Sine and cosine of the yaw error to the active target's heading. Shape is (num_envs, 2).

    The actor's command is position only, as in the waypoint task; this gives the critic the heading that the
    climb's ``goal_heading`` reward scores.
    """
    command: ApproachMotionCommand = env.command_manager.get_term(command_name)
    error = command.heading_error_target
    return torch.stack((error.sin(), error.cos()), dim=-1)


def mirror_heading(data: torch.Tensor) -> torch.Tensor:
    """Reflect a ``(sin, cos)`` yaw error across the root XZ plane."""
    return data * data.new_tensor((-1.0, 1.0))
