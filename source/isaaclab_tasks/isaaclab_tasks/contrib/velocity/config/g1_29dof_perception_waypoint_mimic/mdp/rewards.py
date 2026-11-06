# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Waypoint heading reward."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .command import PyramidMimicCommand


def waypoint_heading_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """``exp(-e_yaw^2 / std^2)`` of the yaw error [rad] to the clip's start heading, within the reach radius."""
    command: PyramidMimicCommand = env.command_manager.get_term(command_name)
    return torch.exp(-command.heading_error.square() / std**2) * command.at_waypoint
