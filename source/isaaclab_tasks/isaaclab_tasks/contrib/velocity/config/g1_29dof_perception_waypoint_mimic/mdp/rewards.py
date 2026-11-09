# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Penalties on walking backward and too fast, the bonus for reaching the base waypoint, and the top waypoint reward."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .command import PyramidMimicCommand, PyramidMimicSplitCommand


def backward_velocity_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Squared backward root speed [m^2/s^2]: ``min(v_x, 0)^2`` of the body-frame forward velocity."""
    return env.scene[asset_cfg.name].data.root_lin_vel_b.torch[:, 0].clamp_max(0.0).square()


def excess_speed_l2(
    env: ManagerBasedRLEnv, max_speed: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Squared horizontal root speed above ``max_speed`` [m^2/s^2]: ``max(|v_xy| - max_speed, 0)^2``."""
    speed = torch.linalg.norm(env.scene[asset_cfg.name].data.root_lin_vel_w.torch[:, :2], dim=-1)
    return (speed - max_speed).clamp_min(0.0).square()


def base_reached_bonus(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """1 on the step after an approach episode first reaches the base waypoint, else 0 (never in walk)."""
    command: PyramidMimicCommand = env.command_manager.get_term(command_name)
    return command.base_reached_now.float()


def top_position_after_approach_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """``exp(-|e|^2 / std^2)`` of the root-to-waypoint error [m], only in imitations that followed an approach.

    The waypoint is then the clip's end on the top landing. Imitations started on the reference get zero.
    """
    command: PyramidMimicSplitCommand = env.command_manager.get_term(command_name)
    return torch.exp(-command.error_w.square().sum(dim=-1) / std**2) * command.from_approach
