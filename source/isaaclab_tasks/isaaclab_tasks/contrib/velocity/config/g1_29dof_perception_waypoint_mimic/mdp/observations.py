# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Waypoint, imitation state, and root position on the pyramid tile."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .command import PyramidMimicCommand


def waypoint_pose(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Waypoint relative to the root in the root yaw frame [m], then sine and cosine of the heading error.

    Shape is (num_envs, 5).
    """
    command: PyramidMimicCommand = env.command_manager.get_term(command_name)
    heading = command.heading_error.unsqueeze(-1)
    return torch.cat((command.command, heading.sin(), heading.cos()), dim=-1)


def mirror_waypoint_pose(data: torch.Tensor) -> torch.Tensor:
    """Reflect :func:`waypoint_pose` across the root XZ plane: negate local Y and the heading-error sine."""
    return data * data.new_tensor((1.0, -1.0, 1.0, -1.0, 1.0))


def imitation_state(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Whether the clip is playing, and its phase in [0, 1] (zero before). Shape is (num_envs, 2)."""
    command: PyramidMimicCommand = env.command_manager.get_term(command_name)
    imitating = command.is_imitation.float()
    return torch.stack((imitating, command.phase.clamp_min(0.0) * imitating), dim=-1)


def root_pos_pyramid_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Root position [m] in the root yaw frame: XY from the pyramid center, Z above the ground (world Z)."""
    command: PyramidMimicCommand = env.command_manager.get_term(command_name)
    data = command.robot.data
    position = data.root_pos_w.torch.clone()
    position[:, :2] -= command.pyramid_center_w
    return math_utils.quat_apply_inverse(math_utils.yaw_quat(data.root_quat_w.torch), position)
