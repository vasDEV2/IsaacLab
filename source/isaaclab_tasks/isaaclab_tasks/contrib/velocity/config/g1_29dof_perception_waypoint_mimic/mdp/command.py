# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Waypoint at the base of a clip's pyramid, followed by imitation of the clip once the robot is there."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.utils import math as math_utils
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.velocity.config.g1_29dof_goalconditioned_perception.mdp.motion import StairMotionLibrary
from isaaclab_tasks.contrib.velocity.config.g1_29dof_perception_waypoint.mdp import WaypointCommand, WaypointCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

WALK, APPROACH, IMITATE = 0, 1, 2
"""Episode modes. ``WALK``: waypoint rewards. ``APPROACH``: the same waypoint without reward.
``IMITATE``: an ``APPROACH`` episode that reached the waypoint and now tracks the clip."""

MODE_NAMES = ("walk", "approach", "imitate")


class PyramidMimicCommand(WaypointCommand):
    """Waypoint at a clip's first root pose (the base of its pyramid), then imitation of that clip.

    Each episode picks a clip uniformly and spawns the robot at rest on the ground behind the clip's start, with a
    random yaw. The waypoint is the clip's first root position and heading. Episodes reset before
    :attr:`PyramidMimicCommandCfg.imitation_start_step` are :data:`WALK`; later ones are :data:`APPROACH` and switch
    to :data:`IMITATE` on reaching the waypoint, when the clip starts playing from its first frame and the waypoint
    moves to the clip's last root pose, standing on the pyramid's top landing.

    Besides the waypoint interface, this term exposes the reference interface (``anchor_pos_w``,
    ``body_pos_relative_w``, ``is_imitation``, ...) read by the goal-conditioned task's imitation rewards and
    terminations. The terrain origin is the clip frame's origin, so reference poses map to the world by adding
    the environment origin.
    """

    cfg: PyramidMimicCommandCfg

    def __init__(self, cfg: PyramidMimicCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        root_name = self.robot.body_names[0]
        if root_name not in cfg.body_names:
            raise ValueError(f"Tracked body_names must include the root body '{root_name}'.")
        self.motion = StairMotionLibrary(
            cfg.motion_files, self.robot.joint_names, cfg.body_names, env.step_dt, self.device
        )
        self.num_motions = len(cfg.motion_files)
        self.motion_root_index = cfg.body_names.index(root_name)
        self.motion_anchor_index = cfg.body_names.index(cfg.anchor_body_name)
        self.robot_anchor_index = self.robot.body_names.index(cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )
        # XY center of the pyramid's boxes in the clip frame
        with np.load(cfg.motion_files[0], allow_pickle=False) as data:
            lower = (data["terrain_position"][:, :2] - data["terrain_dimension"][:, :2] / 2.0).min(axis=0)
            upper = (data["terrain_position"][:, :2] + data["terrain_dimension"][:, :2] / 2.0).max(axis=0)
        self.pyramid_center = torch.tensor((lower + upper) / 2.0, dtype=torch.float32, device=self.device)

        self.mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.goal_yaw = torch.zeros(self.num_envs, device=self.device)
        self.top_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        """Clip's last root position [m], standing on the top landing; the waypoint while imitating."""
        self.top_yaw = torch.zeros(self.num_envs, device=self.device)
        self.top_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        num_bodies = len(cfg.body_names)
        self.body_pos_relative_w = torch.zeros(self.num_envs, num_bodies, 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, num_bodies, 4, device=self.device)
        self.body_quat_relative_w[..., 3] = 1.0
        del self.metrics["climb_height"]
        for name in ("heading_error", *MODE_NAMES, "imitation_phase", "top_reached"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "PyramidMimicCommand:\n"
        msg += f"\tClips: {self.num_motions}\n"
        msg += f"\tImitation from control step: {self.cfg.imitation_start_step}\n"
        return msg

    """
    Modes
    """

    def mask(self, modes: Sequence[int]) -> torch.Tensor:
        """Environments whose mode is one of ``modes``."""
        mask = torch.zeros_like(self.mode, dtype=torch.bool)
        for mode in modes:
            mask |= self.mode == mode
        return mask

    @property
    def is_imitation(self) -> torch.Tensor:
        return self.mode == IMITATE

    """
    Waypoint
    """

    @property
    def heading_error(self) -> torch.Tensor:
        """Yaw from the root heading to the clip's start heading [rad], in [-pi, pi]."""
        root_yaw = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w.torch)[2]
        return math_utils.wrap_to_pi(self.goal_yaw - root_yaw)

    @property
    def waypoint_reached(self) -> torch.Tensor:
        """Root within the reach radius and height tolerance of the waypoint; the heading is not checked."""
        return self.at_waypoint & (self.height_error.abs() < self.cfg.height_tolerance)

    @property
    def pyramid_center_w(self) -> torch.Tensor:
        """World XY of the pyramid center on each environment's tile [m]. Shape is (num_envs, 2)."""
        return self._env.scene.env_origins[:, :2] + self.pyramid_center

    """
    Reference
    """

    @property
    def _frames(self) -> torch.Tensor:
        return self.motion.frame_indices(self.motion_ids, self.time_steps)

    @property
    def phase(self) -> torch.Tensor:
        """Position in the clip, 0 on the first frame and 1 on the last."""
        return self.time_steps.float() / (self.motion.lengths[self.motion_ids] - 1).clamp_min(1)

    @property
    def has_reached_end(self) -> torch.Tensor:
        return self.time_steps >= self.motion.lengths[self.motion_ids] - 1

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self._frames] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self._frames]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self._frames]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self._frames]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self._frames, self.motion_anchor_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self._frames, self.motion_anchor_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self._frames, self.motion_anchor_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self._frames, self.motion_anchor_index]

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w.torch[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w.torch[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w.torch[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w.torch[:, self.robot_anchor_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w.torch[:, self.robot_anchor_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w.torch[:, self.robot_anchor_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w.torch[:, self.robot_anchor_index]

    """
    Episode sampling
    """

    def sample_episode(self, env_ids: Sequence[int] | torch.Tensor | slice):
        """Draw the clip, spawn, and waypoint of the coming episode; later calls before the next reset do nothing."""
        env_ids = self._as_env_ids(env_ids)
        pending = env_ids[self._pending[env_ids]]
        count = len(pending)
        if count == 0:
            return
        cfg = self.cfg
        motion_ids = torch.randint(self.num_motions, (count,), device=self.device)
        first = self.motion.frame_indices(motion_ids, torch.zeros_like(motion_ids))
        start = self.motion.body_pos_w[first, self.motion_root_index] + self._env.scene.env_origins[pending]
        yaw = math_utils.euler_xyz_from_quat(self.motion.body_quat_w[first, self.motion_root_index])[2]
        last = self.motion.frame_indices(motion_ids, self.motion.lengths[motion_ids] - 1)
        self.top_pos_w[pending] = (
            self.motion.body_pos_w[last, self.motion_root_index] + self._env.scene.env_origins[pending]
        )
        top_quat = self.motion.body_quat_w[last, self.motion_root_index]
        self.top_yaw[pending] = math_utils.euler_xyz_from_quat(top_quat)[2]
        self.top_reached[pending] = False
        # spawn behind the clip's start, in its heading frame, on the flat ground around the pyramid
        back = math_utils.sample_uniform(*cfg.spawn_back_range, (count,), self.device)
        lateral = math_utils.sample_uniform(*cfg.spawn_lateral_range, (count,), self.device)
        cos, sin = torch.cos(yaw), torch.sin(yaw)
        spawn = start.clone()
        spawn[:, 0] += -back * cos - lateral * sin
        spawn[:, 1] += -back * sin + lateral * cos
        spawn[:, 2] = cfg.border_height

        self.motion_ids[pending] = motion_ids
        self.time_steps[pending] = 0
        self.start_pos_w[pending] = spawn
        self.start_yaw[pending] = math_utils.sample_uniform(*cfg.start_yaw_range, (count,), self.device)
        self.start_at_center[pending] = False
        self.waypoint_pos_w[pending] = start
        self.goal_yaw[pending] = yaw
        self.initial_distance[pending] = (start[:, :2] - spawn[:, :2]).norm(dim=-1)
        self.reached[pending] = False
        imitate = self._env.common_step_counter >= cfg.imitation_start_step
        self.mode[pending] = APPROACH if imitate else WALK
        self._pending[pending] = False

    """
    Implementation specific functions.
    """

    def _print_errors(self, reached: torch.Tensor):
        """Print each environment's mode and its error to the current waypoint (world frame)."""
        error = (self.waypoint_pos_w - self.robot.data.root_pos_w.torch).tolist()
        heading = self.heading_error.tolist()
        lines = [f"[waypoint] step {self._env.common_step_counter}"]
        for i, (mode, (dx, dy, dz), yaw, done) in enumerate(zip(self.mode.tolist(), error, heading, reached.tolist())):
            lines.append(
                f"  env {i:3d} {MODE_NAMES[mode]:8s} dx {dx:+6.2f} dy {dy:+6.2f} dz {dz:+6.2f} m"
                f"  xy {math.hypot(dx, dy):5.2f} m  heading {yaw:+5.2f} rad{'  REACHED' if done else ''}"
            )
        print("\n".join(lines))

    def _update_metrics(self):
        reached = self.waypoint_reached
        if self.cfg.print_interval > 0 and self._env.common_step_counter % self.cfg.print_interval == 0:
            self._print_errors(reached)
        self.top_reached |= self.is_imitation & reached
        self.reached |= reached
        # approach episodes start the clip on reaching the waypoint; _update_command advances it to frame 0.
        # The waypoint then moves to the clip's end on the top landing.
        start = (self.mode == APPROACH) & reached
        self.mode[start] = IMITATE
        self.time_steps[start] = -1
        self.waypoint_pos_w[start] = self.top_pos_w[start]
        self.goal_yaw[start] = self.top_yaw[start]
        self.metrics["distance"][:] = self.distance_xy
        self.metrics["height_error"][:] = self.height_error.abs()
        self.metrics["heading_error"][:] = self.heading_error.abs()
        self.metrics["success"][:] = self.reached.float()
        for mode, name in enumerate(MODE_NAMES):
            self.metrics[name][:] = (self.mode == mode).float()
        self.metrics["imitation_phase"][:] = self.phase.clamp_min(0.0) * self.is_imitation
        self.metrics["top_reached"][:] = self.top_reached.float()

    def _update_command(self):
        last = self.motion.lengths[self.motion_ids] - 1
        self.time_steps = torch.where(self.is_imitation, torch.minimum(self.time_steps + 1, last), self.time_steps)
        # reference bodies moved onto the robot's anchor (XY and yaw), keeping the reference anchor height
        num_bodies = len(self.cfg.body_names)
        anchor_pos = self.anchor_pos_w[:, None, :].expand(-1, num_bodies, -1)
        anchor_quat = self.anchor_quat_w[:, None, :].expand(-1, num_bodies, -1)
        delta_pos = self.robot_anchor_pos_w[:, None, :].expand(-1, num_bodies, -1).clone()
        delta_pos[..., 2] = anchor_pos[..., 2]
        robot_quat = self.robot_anchor_quat_w[:, None, :].expand(-1, num_bodies, -1)
        delta_quat = math_utils.yaw_quat(math_utils.quat_mul(robot_quat, math_utils.quat_inv(anchor_quat)))
        self.body_quat_relative_w = math_utils.quat_mul(delta_quat, self.body_quat_w)
        self.body_pos_relative_w = delta_pos + math_utils.quat_apply(delta_quat, self.body_pos_w - anchor_pos)


@configclass
class PyramidMimicCommandCfg(WaypointCommandCfg):
    """Configuration for :class:`PyramidMimicCommand`."""

    class_type: type = PyramidMimicCommand

    motion_files: list[str] = MISSING
    """Clips climbing the same pyramid; the terrain must be built from the first one."""

    body_names: list[str] = MISSING
    """Tracked bodies, including the root body."""

    anchor_body_name: str = "torso_link"

    imitation_start_step: int = 3000 * 24
    """Control step (iteration times steps per environment) from which episodes approach without reward and imitate."""

    spawn_back_range: tuple[float, float] = (2.0, 4.0)
    """Spawn distance [m] behind the clip's start, along its start heading."""

    spawn_lateral_range: tuple[float, float] = (-1.0, 1.0)
    """Spawn offset [m] to the left of the clip's start heading."""

    print_interval: int = 0
    """Control steps between printouts of each environment's waypoint error; 0 disables them."""

    start_yaw_range: tuple[float, float] = (-math.pi, math.pi)
