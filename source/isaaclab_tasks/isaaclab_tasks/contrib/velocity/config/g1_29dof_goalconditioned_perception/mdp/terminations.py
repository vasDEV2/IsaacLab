# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reference terminations, which apply only to imitation episodes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .motion import GoalMotionCommand


def motion_end(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """End imitation episodes on the clip's last frame; task episodes continue past it."""
    command: GoalMotionCommand = env.command_manager.get_term(command_name)
    return command.has_reached_end & command.is_imitation


def bad_anchor_height(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """Anchor height error [m] from the reference above ``threshold``."""
    command: GoalMotionCommand = env.command_manager.get_term(command_name)
    error = (command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2]).abs()
    return (error > threshold) & command.is_imitation


def bad_anchor_tilt(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """Difference of the anchor-frame gravity Z components between reference and robot above ``threshold``."""
    command: GoalMotionCommand = env.command_manager.get_term(command_name)
    gravity = torch.zeros_like(command.anchor_pos_w)
    gravity[:, 2] = -1.0
    reference = quat_apply_inverse(command.anchor_quat_w, gravity)[:, 2]
    robot = quat_apply_inverse(command.robot_anchor_quat_w, gravity)[:, 2]
    return ((reference - robot).abs() > threshold) & command.is_imitation


def bad_body_height(env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str]) -> torch.Tensor:
    """Height error [m] of any of ``body_names`` from the anchor-aligned reference above ``threshold``."""
    command: GoalMotionCommand = env.command_manager.get_term(command_name)
    ids = [command.cfg.body_names.index(name) for name in body_names]
    error = (command.body_pos_relative_w[:, ids, 2] - command.robot_body_pos_w[:, ids, 2]).abs()
    return (error > threshold).any(dim=-1) & command.is_imitation
