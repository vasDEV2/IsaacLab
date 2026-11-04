# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stair-level curriculum driven by waypoint success."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .waypoint import WaypointCommand


def terrain_levels_waypoint(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor, command_name: str, demote_fraction: float = 0.5
) -> torch.Tensor:
    """Promote an environment that reached its waypoint; demote one that ended over ``demote_fraction`` of its
    starting distance away. The first reset changes nothing.
    """
    command: WaypointCommand = env.command_manager.get_term(command_name)
    terrain = env.scene.terrain
    finished = env.episode_length_buf[env_ids] > 0
    move_up = finished & command.reached[env_ids]
    remaining = command.distance_xy[env_ids]
    move_down = finished & ~move_up & (remaining > demote_fraction * command.initial_distance[env_ids])
    terrain.update_env_origins(env_ids, move_up, move_down)
    return terrain.terrain_levels.float().mean()
