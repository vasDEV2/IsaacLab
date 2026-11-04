# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Waypoint commands between the center and the flat border of each stair tile."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import SPHERE_MARKER_CFG
from isaaclab.utils import math as math_utils
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class WaypointCommand(CommandTerm):
    """One target root position per episode, on the other side of the tile's stairs from the start.

    Each episode starts either at the tile center (the top platform of a pyramid, the bottom of a pit, or flat
    ground) or on the flat border ring around the stairs, and its waypoint lies on the other one. The waypoint
    height is the ground height under it plus :attr:`WaypointCommandCfg.base_height`, so a robot that stands on
    level ground at the waypoint has zero height error.

    The command is the waypoint relative to the root in the root's yaw frame: ``(x, y, z)`` in meters.
    The start pose is drawn by :meth:`sample_episode`, which the reset event calls before placing the robot.
    """

    cfg: WaypointCommandCfg

    def __init__(self, cfg: WaypointCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.start_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.start_yaw = torch.zeros(self.num_envs, device=self.device)
        self.waypoint_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.start_at_center = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.initial_distance = torch.zeros(self.num_envs, device=self.device)
        self.reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._pending = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        for name in ("distance", "height_error", "success", "climb_height"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "WaypointCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tBorder ring (Chebyshev): {self.cfg.border_distance_range} m\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Waypoint relative to the root in the root yaw frame [m]. Shape is (num_envs, 3)."""
        return math_utils.quat_apply_inverse(self._root_yaw_quat, self.error_w)

    @property
    def error_w(self) -> torch.Tensor:
        """World-frame displacement from the root to the waypoint [m]."""
        return self.waypoint_pos_w - self.robot.data.root_pos_w.torch

    @property
    def distance_xy(self) -> torch.Tensor:
        return self.error_w[:, :2].norm(dim=-1)

    @property
    def height_error(self) -> torch.Tensor:
        return self.error_w[:, 2]

    @property
    def bearing_error(self) -> torch.Tensor:
        """Yaw from the root heading to the waypoint direction [rad], in [-pi, pi]."""
        error_b = self.command
        return torch.atan2(error_b[:, 1], error_b[:, 0])

    @property
    def at_waypoint(self) -> torch.Tensor:
        """Root within the horizontal reach radius."""
        return self.distance_xy < self.cfg.reach_radius

    @property
    def _root_yaw_quat(self) -> torch.Tensor:
        return math_utils.yaw_quat(self.robot.data.root_quat_w.torch)

    """
    Episode sampling
    """

    def sample_episode(self, env_ids: Sequence[int] | torch.Tensor | slice):
        """Draw the start pose and waypoint of the coming episode; later calls before the next reset do nothing."""
        env_ids = self._as_env_ids(env_ids)
        pending = env_ids[self._pending[env_ids]]
        count = len(pending)
        if count == 0:
            return
        cfg = self.cfg
        center = self._env.scene.env_origins[pending].clone()
        center[:, :2] += math_utils.sample_uniform(-cfg.center_noise, cfg.center_noise, (count, 2), self.device)
        # a point on a square ring around the tile center, where the generated stairs end in flat ground
        angle = math_utils.sample_uniform(-math.pi, math.pi, (count,), self.device)
        direction = torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
        direction /= direction.abs().amax(dim=-1, keepdim=True)
        border = self._env.scene.env_origins[pending].clone()
        border[:, :2] += direction * math_utils.sample_uniform(*cfg.border_distance_range, (count, 1), self.device)
        border[:, 2] = cfg.border_height

        at_center = torch.rand(count, device=self.device) < cfg.center_start_probability
        self.start_at_center[pending] = at_center
        self.start_pos_w[pending] = torch.where(at_center[:, None], center, border)
        goal = torch.where(at_center[:, None], border, center)
        goal[:, 2] += cfg.base_height
        self.waypoint_pos_w[pending] = goal
        self.start_yaw[pending] = math_utils.sample_uniform(*cfg.start_yaw_range, (count,), self.device)
        self.initial_distance[pending] = (goal[:, :2] - self.start_pos_w[pending, :2]).norm(dim=-1)
        self.reached[pending] = False
        self._pending[pending] = False

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)
        extras = super().reset(env_ids)
        env_ids = self._as_env_ids(env_ids)
        # the next reset event draws a new episode
        self._pending[env_ids] = True
        return extras

    def _as_env_ids(self, env_ids: Sequence[int] | torch.Tensor | slice) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)[env_ids]
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        self.reached |= self.at_waypoint & (self.height_error.abs() < self.cfg.height_tolerance)
        self.metrics["distance"][:] = self.distance_xy
        self.metrics["height_error"][:] = self.height_error.abs()
        self.metrics["success"][:] = self.reached.float()
        self.metrics["climb_height"][:] = self.waypoint_pos_w[:, 2] - self.cfg.base_height - self.start_pos_w[:, 2]

    def _resample_command(self, env_ids: Sequence[int]):
        # the reset event already drew the episode unless it is disabled
        self.sample_episode(env_ids)

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis and not hasattr(self, "waypoint_visualizer"):
            self.waypoint_visualizer = VisualizationMarkers(self.cfg.waypoint_visualizer_cfg)
        if hasattr(self, "waypoint_visualizer"):
            self.waypoint_visualizer.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        if self.robot.is_initialized:
            self.waypoint_visualizer.visualize(self.waypoint_pos_w)


@configclass
class WaypointCommandCfg(CommandTermCfg):
    """Configuration for :class:`WaypointCommand`."""

    class_type: type = WaypointCommand

    asset_name: str = "robot"

    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    """One waypoint per episode."""

    center_start_probability: float = 0.5
    """Fraction of episodes that start at the tile center and walk out to the border ring."""

    border_distance_range: tuple[float, float] = (2.4, 3.4)
    """Chebyshev distance [m] of border starts and waypoints from the tile center.

    The default tile has stairs within 2.0 m of its center and flat ground from 2.0 m to its 4.0 m edge.
    """

    border_height: float = 0.0
    """World ground height [m] of the border ring."""

    center_noise: float = 0.2
    """Uniform XY noise [m] of center starts and waypoints; stays on the smallest (1.0 m) top platform."""

    base_height: float = 0.75
    """Waypoint root height [m] above the ground under it."""

    start_yaw_range: tuple[float, float] = (-math.pi, math.pi)

    reach_radius: float = 0.3
    """Horizontal distance [m] within which the root counts as at the waypoint."""

    height_tolerance: float = 0.2
    """Root height error [m] allowed for the waypoint to count as reached."""

    waypoint_visualizer_cfg: VisualizationMarkersCfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Command/waypoint")
    """Marker drawn at the target root position."""


class WaypointGaitCommand(CommandTerm):
    """Walking intent derived from a :class:`WaypointCommand`, for gait terms that gate on velocity commands.

    The command is ``(vx, vy, 0)`` [m/s] toward the waypoint at :attr:`WaypointGaitCommandCfg.speed` in the root
    yaw frame, and zero within the waypoint's reach radius. It is not an observation and not tracked by any reward.
    """

    cfg: WaypointGaitCommandCfg

    def __init__(self, cfg: WaypointGaitCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._waypoint: WaypointCommand | None = None

    def __str__(self) -> str:
        return f"WaypointGaitCommand:\n\tFollows: {self.cfg.waypoint_command_name}\n"

    @property
    def command(self) -> torch.Tensor:
        if self._waypoint is None:
            self._waypoint = self._env.command_manager.get_term(self.cfg.waypoint_command_name)
        error_b = self._waypoint.command[:, :2]
        direction = error_b / error_b.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        velocity = direction * self.cfg.speed * ~self._waypoint.at_waypoint.unsqueeze(-1)
        return torch.cat((velocity, torch.zeros_like(velocity[:, :1])), dim=-1)

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _update_command(self):
        pass


@configclass
class WaypointGaitCommandCfg(CommandTermCfg):
    """Configuration for :class:`WaypointGaitCommand`."""

    class_type: type = WaypointGaitCommand

    waypoint_command_name: str = "waypoint"

    speed: float = 0.8
    """Nominal speed [m/s] reported while walking; below the ``fly`` reward's 1.0 m/s running threshold."""

    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
