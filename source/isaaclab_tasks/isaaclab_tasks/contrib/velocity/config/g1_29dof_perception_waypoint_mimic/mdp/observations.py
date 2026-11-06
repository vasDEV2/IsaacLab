# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Horizontal error to the waypoint, the policy's only positional input."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .command import PyramidMimicCommand


def waypoint_xy_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Waypoint XY relative to the root, in the root yaw frame [m]. Shape is (num_envs, 2)."""
    command: PyramidMimicCommand = env.command_manager.get_term(command_name)
    return command.command[:, :2]


def mirror_waypoint_xy(data: torch.Tensor) -> torch.Tensor:
    """Reflect :func:`waypoint_xy_b` across the root XZ plane: negate local Y."""
    return data * data.new_tensor((1.0, -1.0))
