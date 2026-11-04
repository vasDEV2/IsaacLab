# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rewards for reaching a waypoint root position."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .waypoint import WaypointCommand


def waypoint_position_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """``exp(-|e|^2 / std^2)`` of the 3D root-to-waypoint error [m], including the target height."""
    command: WaypointCommand = env.command_manager.get_term(command_name)
    return torch.exp(-command.error_w.square().sum(dim=-1) / std**2)


def waypoint_height_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """``exp(-e_z^2 / std^2)`` of the root height error [m], once within the reach radius."""
    command: WaypointCommand = env.command_manager.get_term(command_name)
    return torch.exp(-command.height_error.square() / std**2) * command.at_waypoint


def waypoint_bearing_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Face the waypoint while away from it: ``exp(-yaw_error^2 / std^2)`` [rad], zero within the reach radius."""
    command: WaypointCommand = env.command_manager.get_term(command_name)
    return torch.exp(-command.bearing_error.square() / std**2) * ~command.at_waypoint


def waypoint_stand_still(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Squared root planar speed [m^2/s^2] and yaw rate [rad^2/s^2] within the reach radius."""
    command: WaypointCommand = env.command_manager.get_term(command_name)
    data = command.robot.data
    speed = data.root_lin_vel_w.torch[:, :2].square().sum(dim=-1) + data.root_ang_vel_w.torch[:, 2].square()
    return speed * command.at_waypoint


class waypoint_progress(ManagerTermBase):
    """Rate [m/s] at which the horizontal distance to the waypoint shrinks, clipped to ``max_speed``.

    It is zero on the first step of an episode and once within the reach radius.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous = torch.zeros(env.num_envs, device=env.device)
        self._fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._fresh[env_ids] = True

    def __call__(self, env: ManagerBasedRLEnv, command_name: str, max_speed: float = 1.0) -> torch.Tensor:
        command: WaypointCommand = env.command_manager.get_term(command_name)
        distance = command.distance_xy
        rate = ((self._previous - distance) / env.step_dt).clamp(-max_speed, max_speed)
        rate = torch.where(self._fresh | command.at_waypoint, 0.0, rate)
        self._previous[:] = distance
        self._fresh[:] = False
        return rate
