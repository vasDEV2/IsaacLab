# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Goal-conditioned stair climbing that starts with a walk from a random spawn to the reference clip's start.

Every episode spawns the robot at rest at a random place on the flat ground in front of the clip's side of the
pyramid. Its first target is the first reference root pose (the waypoint). The episode is in one of three phases:

* :data:`WALK`: before :attr:`ApproachMotionCommandCfg.approach_steps`, episodes only walk to the waypoint and are
  rewarded like the perception waypoint task.
* :data:`APPROACH`: from then on, episodes walk to the same waypoint without reward.
* :data:`CLIMB`: an :data:`APPROACH` episode becomes one once the root reaches the waypoint. The goal-conditioned
  imitation or task episode then starts from the clip's first frame, with the goal-conditioned rewards.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.utils import math as math_utils
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.velocity.config.g1_29dof_goalconditioned_perception.mdp.motion import (
    GoalMotionCommand,
    GoalMotionCommandCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

WALK, APPROACH, CLIMB = 0, 1, 2
PHASE_NAMES = ("walk", "approach", "climb")


class ApproachMotionCommand(GoalMotionCommand):
    """:class:`GoalMotionCommand` whose episodes begin by walking from a random spawn to the clip's first root pose.

    The reference clock holds the first frame until the episode climbs, so imitation always starts from the clip's
    beginning where the robot arrived. :attr:`command` is the active target position in the root yaw frame, as in the
    waypoint task: the waypoint (first reference root position) before climbing, the destination after.

    The waypoint properties (:attr:`error_w`, :attr:`distance_xy`, :attr:`height_error`, :attr:`bearing_error`,
    :attr:`at_waypoint`) follow :class:`~isaaclab_tasks.contrib.velocity.config.g1_29dof_perception_waypoint.mdp.
    WaypointCommand`, so the waypoint task's rewards and gait command can read this term directly.

    :meth:`mask` lets :class:`~isaaclab_tasks.contrib.velocity.config.g1_29dof_perception_motion_style.mdp.ModeGated`
    gate terms by phase.
    """

    cfg: ApproachMotionCommandCfg

    def __init__(self, cfg: ApproachMotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.episode_phase = torch.full((self.num_envs,), WALK, dtype=torch.long, device=self.device)
        """Per-environment phase: :data:`WALK`, :data:`APPROACH`, or :data:`CLIMB`."""
        self.climb_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        """Control steps since the episode started climbing."""
        self.waypoint_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        """First reference root position [m] in the world frame."""
        self.waypoint_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.waypoint_quat_w[:, 3] = 1.0
        self.spawn_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        """Spawn position on the ground [m] in the world frame."""
        self.spawn_yaw = torch.zeros(self.num_envs, device=self.device)
        # all clips share one staircase, whose boxes are centered on the tile (see ``motion_stairs_terrain``)
        with np.load(cfg.motion_files[0], allow_pickle=False) as data:
            boxes = np.concatenate((data["terrain_position"], data["terrain_dimension"]), axis=-1)
            self.ground_height = float(data["ground_height"]) if "ground_height" in data else 0.0
            """Ground height [m] in the clip frame."""
        lower = (boxes[:, :2] - boxes[:, 3:5] / 2.0).min(axis=0)
        upper = (boxes[:, :2] + boxes[:, 3:5] / 2.0).max(axis=0)
        self.tile_center = torch.tensor((lower + upper) / 2.0, dtype=torch.float32, device=self.device)
        """Tile (pyramid) center XY [m] in the clip frame. Shape is (2,)."""
        self.waypoint_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._finished_phase = torch.full_like(self.episode_phase, WALK)
        for name in ("waypoint_distance", "waypoint_success"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = super().__str__()
        msg += f"\n\tUnrewarded approach and climbing from control step: {self.cfg.approach_steps}"
        return msg

    """
    Properties: phase and target.
    """

    @property
    def climbing_enabled(self) -> bool:
        """Whether new episodes approach and then climb instead of only walking to the waypoint."""
        return self._env.common_step_counter >= self.cfg.approach_steps

    def mask(self, phases: Sequence[int]) -> torch.Tensor:
        """Environments whose phase is in ``phases``. Shape is (num_envs,)."""
        return torch.isin(self.episode_phase, torch.as_tensor(phases, device=self.device))

    @property
    def command(self) -> torch.Tensor:
        """Active target position [m] relative to the root, in the root yaw frame. Shape is (N, 3).

        The target is the waypoint until the episode climbs and the destination afterwards, encoded like the
        waypoint task's command.
        """
        climbing = (self.episode_phase == CLIMB).unsqueeze(-1)
        target_pos = torch.where(climbing, self.goal_pos_w, self.waypoint_pos_w)
        root_yaw = math_utils.yaw_quat(self.robot.data.root_quat_w.torch)
        return math_utils.quat_apply_inverse(root_yaw, target_pos - self.robot.data.root_pos_w.torch)

    @property
    def heading_error_target(self) -> torch.Tensor:
        """Yaw [rad] from the root heading to the active target's heading, in [-pi, pi]. Shape is (N,)."""
        climbing = (self.episode_phase == CLIMB).unsqueeze(-1)
        target_quat = torch.where(climbing, self.goal_quat_w, self.waypoint_quat_w)
        return math_utils.wrap_to_pi(
            math_utils.euler_xyz_from_quat(target_quat)[2]
            - math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w.torch)[2]
        )

    @property
    def tile_center_w(self) -> torch.Tensor:
        """Tile center on the flat ground [m] in the world frame. Shape is (N, 3)."""
        center = self._env.scene.env_origins.clone()
        center[:, :2] += self.tile_center
        center[:, 2] += self.ground_height
        return center

    """
    Properties: waypoint, in the interface of the waypoint task's command.
    """

    @property
    def error_w(self) -> torch.Tensor:
        """World displacement [m] from the root to the waypoint. Shape is (N, 3)."""
        return self.waypoint_pos_w - self.robot.data.root_pos_w.torch

    @property
    def distance_xy(self) -> torch.Tensor:
        return self.error_w[:, :2].norm(dim=-1)

    @property
    def height_error(self) -> torch.Tensor:
        return self.error_w[:, 2]

    @property
    def bearing_error(self) -> torch.Tensor:
        """Yaw [rad] from the root heading to the waypoint direction, in [-pi, pi]."""
        error_b = math_utils.quat_apply_inverse(math_utils.yaw_quat(self.robot.data.root_quat_w.torch), self.error_w)
        return torch.atan2(error_b[:, 1], error_b[:, 0])

    @property
    def at_waypoint(self) -> torch.Tensor:
        """Root within the waypoint's horizontal reach radius."""
        return self.distance_xy < self.cfg.waypoint_reach_radius

    @property
    def at_waypoint_height(self) -> torch.Tensor:
        """Root within the reach radius and the height tolerance of the waypoint."""
        return self.at_waypoint & (self.height_error.abs() < self.cfg.waypoint_height_tolerance)

    """
    Operations.
    """

    def resample_time_steps(self, env_ids: Sequence[int] | torch.Tensor | slice):
        """Draw the episode as :class:`GoalMotionCommand` does, then its phase, spawn, and waypoint.

        The clip always starts on its first frame.
        """
        env_ids = self._as_env_ids(env_ids)
        pending = env_ids[self._resample_pending[env_ids]]
        if len(pending) == 0:
            return
        self._finished_phase[pending] = self.episode_phase[pending]
        super().resample_time_steps(pending)
        count = len(pending)
        self.time_steps[pending] = 0
        self.episode_phase[pending] = APPROACH if self.climbing_enabled else WALK
        self.climb_steps[pending] = 0
        self.waypoint_reached[pending] = False

        # the waypoint is the first reference root pose; spawns lie behind it in its yaw frame, on the ground
        first = self.motion.frame_indices(self.motion_ids[pending], torch.zeros_like(pending))
        origins = self._env.scene.env_origins[pending]
        ground = origins[:, 2] + self.ground_height
        self.waypoint_pos_w[pending] = self.motion.body_pos_w[first, self.root_index] + origins
        if self.cfg.waypoint_base_height is not None:
            # the clips start on the flat ground, so the waypoint height is measured like the waypoint task's
            self.waypoint_pos_w[pending, 2] = ground + self.cfg.waypoint_base_height
        waypoint_yaw = math_utils.yaw_quat(self.motion.body_quat_w[first, self.root_index])
        self.waypoint_quat_w[pending] = waypoint_yaw
        offset = torch.zeros(count, 3, device=self.device)
        offset[:, 0] = math_utils.sample_uniform(*self.cfg.spawn_x_range, (count,), self.device)
        offset[:, 1] = math_utils.sample_uniform(*self.cfg.spawn_y_range, (count,), self.device)
        spawn = self.waypoint_pos_w[pending] + math_utils.quat_apply(waypoint_yaw, offset)
        spawn[:, 2] = ground
        self.spawn_pos_w[pending] = spawn
        self.spawn_yaw[pending] = math_utils.sample_uniform(*self.cfg.spawn_yaw_range, (count,), self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)
        extras = super().reset(env_ids)
        finished = self._finished_phase[env_ids]
        approached = finished != WALK
        # share of finished approach episodes that reached the waypoint and climbed
        extras["approach_success_rate"] = (finished == CLIMB).sum() / approached.sum().clamp_min(1)
        return extras

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        super()._update_metrics()
        self.waypoint_reached |= self.at_waypoint_height
        self.metrics["waypoint_distance"][:] = self.distance_xy
        self.metrics["waypoint_success"][:] = self.waypoint_reached.float()

    def _update_command(self):
        # approaching episodes that reach the waypoint start climbing on the clip's first frame
        start = (self.episode_phase == APPROACH) & self.at_waypoint_height
        self.episode_phase[start] = CLIMB
        # the reference clock runs only while climbing; it holds the last frame, where ``motion_end`` ends imitation
        climbing = self.episode_phase == CLIMB
        self.time_steps = torch.where(
            climbing, torch.minimum(self.time_steps + 1, self.motion_lengths - 1), self.time_steps
        )
        self.climb_steps += climbing.long()
        self._update_relative_bodies()

    def _debug_vis_callback(self, event):
        if self.robot.is_initialized:
            climbing = (self.episode_phase == CLIMB).unsqueeze(-1)
            self.goal_visualizer.visualize(
                torch.where(climbing, self.goal_pos_w, self.waypoint_pos_w),
                torch.where(climbing, self.goal_quat_w, self.waypoint_quat_w),
            )


@configclass
class ApproachMotionCommandCfg(GoalMotionCommandCfg):
    """Configuration for :class:`ApproachMotionCommand`."""

    class_type: type = ApproachMotionCommand

    approach_steps: int = 1500 * 24
    """Control step from which new episodes approach without reward and then climb, instead of only walking."""

    full_clip_probability: float = 1.0
    """Imitation always starts on the first frame, where the approach ends."""

    spawn_x_range: tuple[float, float] = (-2.0, -0.5)
    """Spawn offset [m] from the waypoint along its heading; negative is away from the pyramid.

    The clips start at the foot of the pyramid, about 2.2 m from the edge of the 9 m tile.
    """

    spawn_y_range: tuple[float, float] = (-2.0, 2.0)
    """Spawn offset [m] from the waypoint across its heading; the pyramid is 4.5 m wide."""

    spawn_yaw_range: tuple[float, float] = (-math.pi, math.pi)
    """Spawn yaw [rad] in the world frame."""

    waypoint_base_height: float | None = 0.75
    """Waypoint root height [m] above the flat ground, as the waypoint task's ``base_height``.

    None keeps the clip's first root height (0.80 m).
    """

    waypoint_reach_radius: float = 0.3
    """Horizontal distance [m] within which the root counts as at the waypoint."""

    waypoint_height_tolerance: float = 0.2
    """Root height error [m] allowed for the waypoint to count as reached."""
