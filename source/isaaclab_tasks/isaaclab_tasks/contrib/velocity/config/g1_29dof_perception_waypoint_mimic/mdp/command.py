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

from .terrains import PyramidCurriculumCfg, load_clip_boxes, pyramid_tile_geometry

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

WALK, APPROACH, IMITATE, IMITATE_LOOSE = 0, 1, 2, 3
"""Episode modes. ``WALK``: waypoint rewards. ``APPROACH``: the same waypoint without reward.
``IMITATE``: an ``APPROACH`` episode that reached the waypoint and now tracks the clip.
``IMITATE_LOOSE``: the same on a curriculum pyramid, with loose tracking rewards and terminations."""

MODE_NAMES = ("walk", "approach", "imitate", "imitate_loose")


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
        self._init_tile_fit(cfg.motion_files[0])

        self.mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.goal_yaw = torch.zeros(self.num_envs, device=self.device)
        self.top_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        """Clip's last root position [m], standing on the top landing; the waypoint while imitating."""
        self.top_yaw = torch.zeros(self.num_envs, device=self.device)
        self.top_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.base_reached_now = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        """Whether the base waypoint was first reached on the last command update (one step per episode)."""
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
        return (self.mode == IMITATE) | (self.mode == IMITATE_LOOSE)

    """
    Tile fit: the clip reference moved onto a pyramid with another top width or step height
    """

    def _init_tile_fit(self, terrain_motion_file: str):
        """Per-environment tile geometry, each clip's outward face direction, and the ground under its root."""
        self.loose = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        """Whether the episode runs on a curriculum pyramid; it then imitates in :data:`IMITATE_LOOSE`."""
        self.tile_delta = torch.zeros(self.num_envs, device=self.device)
        """Top width change [m] of the environment's pyramid relative to the clip's."""
        self.tile_scale = torch.ones(self.num_envs, device=self.device)
        """Step height scale of the environment's pyramid relative to the clip's."""
        self._ref_shift = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        """Fitted clip's first root position [m], at the pyramid base; the base waypoint, also when not observed."""
        positions, dimensions, ground = load_clip_boxes(terrain_motion_file)
        tops = positions[:, 2] + dimensions[:, 2] / 2.0 - ground
        root_xy = self.motion.body_pos_w[:, self.motion_root_index, :2].cpu().numpy()
        inside = np.all(np.abs(root_xy[:, None, :] - positions[None, :, :2]) <= dimensions[None, :, :2] / 2.0, axis=-1)
        under = np.where(inside, tops[None, :], 0.0).max(axis=-1)
        # smooth over each clip so the height fit does not jump at the step edges
        width = self.cfg.ground_smoothing_frames
        kernel = np.ones(2 * width + 1) / (2 * width + 1)
        smoothed = np.zeros_like(under)
        directions = np.zeros((self.num_motions, 2))
        center = self.pyramid_center.cpu().numpy()
        for motion_id in range(self.num_motions):
            start, length = int(self.motion.starts[motion_id]), int(self.motion.lengths[motion_id])
            clip = under[start : start + length]
            smoothed[start : start + length] = np.convolve(np.pad(clip, width, mode="edge"), kernel, mode="valid")
            # the face the clip climbs: the axis of its start's offset from the pyramid center
            offset = root_xy[start] - center
            axis = int(np.argmax(np.abs(offset)))
            directions[motion_id, axis] = np.sign(offset[axis])
        self._root_ground = torch.tensor(smoothed, dtype=torch.float32, device=self.device)
        """Clip pyramid height [m] under the reference root at each packed frame, smoothed."""
        self._face_direction = torch.tensor(directions, dtype=torch.float32, device=self.device)
        """Outward normal (XY) of the pyramid face each clip climbs."""

    def _ref_offset(self, frames: torch.Tensor, env_ids: torch.Tensor | slice = slice(None)) -> torch.Tensor:
        """Offset [m] from the clip pose to the fitted reference at ``frames``, shape (len(env_ids), 3).

        Horizontal: the clip's face moved out by half the top width change. Vertical: the clip pyramid's height under
        the root, scaled by the step height scale minus one.
        """
        offset = self._ref_shift[env_ids].clone()
        offset[:, 2] = (self.tile_scale[env_ids] - 1.0) * self._root_ground[frames]
        return offset

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
        frames = self._frames
        offset = self._env.scene.env_origins + self._ref_offset(frames)
        return self.motion.body_pos_w[frames] + offset[:, None, :]

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
        frames = self._frames
        return (
            self.motion.body_pos_w[frames, self.motion_anchor_index]
            + self._env.scene.env_origins
            + self._ref_offset(frames)
        )

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
        # the clip moved onto this tile's pyramid: its face out by half the top width change
        self._ref_shift[pending, :2] = self._face_direction[motion_ids] * (self.tile_delta[pending, None] / 2.0)
        start = (
            self.motion.body_pos_w[first, self.motion_root_index]
            + self._env.scene.env_origins[pending]
            + self._ref_offset(first, pending)
        )
        yaw = math_utils.euler_xyz_from_quat(self.motion.body_quat_w[first, self.motion_root_index])[2]
        last = self.motion.frame_indices(motion_ids, self.motion.lengths[motion_ids] - 1)
        self.top_pos_w[pending] = (
            self.motion.body_pos_w[last, self.motion_root_index]
            + self._env.scene.env_origins[pending]
            + self._ref_offset(last, pending)
        )
        top_quat = self.motion.body_quat_w[last, self.motion_root_index]
        self.top_yaw[pending] = math_utils.euler_xyz_from_quat(top_quat)[2]
        self.top_reached[pending] = False
        # spawn behind the clip's start, in its heading frame, on the flat ground around the pyramid
        back = math_utils.sample_uniform(*cfg.spawn_back_range, (count,), self.device)
        # a wider pyramid moves the start out: keep the spawn within max_spawn_radius of the pyramid center
        start_radius = (start[:, :2] - self.pyramid_center_w[pending]).norm(dim=-1)
        back = torch.minimum(back, (cfg.max_spawn_radius - start_radius).clamp_min(0.0))
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
        self.base_pos_w[pending] = start
        self.goal_yaw[pending] = yaw
        self.initial_distance[pending] = (start[:, :2] - spawn[:, :2]).norm(dim=-1)
        self.reached[pending] = False
        self.base_reached_now[pending] = False
        imitate = self._env.common_step_counter >= cfg.imitation_start_step
        self.mode[pending] = APPROACH if imitate else WALK
        if cfg.top_waypoint_only:
            # one walk straight to the top landing: no base waypoint and no clip
            self.mode[pending] = WALK
            self.waypoint_pos_w[pending] = self.top_pos_w[pending]
            self.goal_yaw[pending] = self.top_yaw[pending]
            self.initial_distance[pending] = (self.top_pos_w[pending, :2] - spawn[:, :2]).norm(dim=-1)
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

    def _base_arrival(self, reached: torch.Tensor) -> torch.Tensor:
        """Whether each episode is at its pyramid base, where an approach starts imitating: the waypoint here."""
        return reached

    def _on_imitation_start(self, start: torch.Tensor) -> None:
        """Called with the approach episodes that start imitating on this step, before their mode changes."""

    def _update_metrics(self):
        reached = self.waypoint_reached
        if self.cfg.print_interval > 0 and self._env.common_step_counter % self.cfg.print_interval == 0:
            self._print_errors(reached)
        self.top_reached |= self.is_imitation & reached
        # first arrival at the base waypoint of an approach episode; rewarded on the next step, once imitating
        # approach episodes start the clip on arriving at the base; _update_command advances it to frame 0
        start = (self.mode == APPROACH) & self._base_arrival(reached)
        self.base_reached_now = start
        self.reached |= reached
        self._on_imitation_start(start)
        self.mode[start] = torch.where(self.loose[start], IMITATE_LOOSE, IMITATE)
        self.time_steps[start] = -1
        # the waypoint then moves to the clip's end on the top landing; curriculum episodes keep theirs
        moved = start & ~self.loose
        self.waypoint_pos_w[moved] = self.top_pos_w[moved]
        self.goal_yaw[moved] = self.top_yaw[moved]
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

    max_spawn_radius: float = 6.4
    """Largest spawn distance [m] from the pyramid center along the clip's start heading; 0.6 m inside a 14 m tile.
    It only shortens spawns in front of pyramids wider than the clip's."""

    ground_smoothing_frames: int = 10
    """Half-width (frames) of the moving average of the clip pyramid height under the root, which lifts the
    reference on pyramids with other step heights."""

    top_waypoint_only: bool = False
    """Make every episode a :data:`WALK` episode whose waypoint is the clip's end on the top landing, skipping the
    base waypoint and imitation (for play)."""


class PyramidMimicSplitCommand(PyramidMimicCommand):
    """:class:`PyramidMimicCommand` with separate walk and imitation episodes after the switch.

    Episodes are drawn by the control step at their reset:

    * Before :attr:`PyramidMimicCommandCfg.imitation_start_step`: :data:`WALK`, as in the parent.
    * From there until :attr:`PyramidMimicSplitCommandCfg.approach_start_step`: :data:`IMITATE` with probability
      :attr:`PyramidMimicSplitCommandCfg.imitation_probability`, :data:`WALK` otherwise. Imitation episodes start on
      the reference (reference state initialization): at a uniformly drawn frame of the clip, or at its first frame
      with probability :attr:`PyramidMimicSplitCommandCfg.full_clip_probability`. Episodes at the first frame start
      from a state in which a walk episode reached the base waypoint, once enough have been recorded, instead of the
      clip's first pose. The waypoint of every imitation is the clip's last root pose. The reset events read
      :meth:`imitation_root_state` and :meth:`imitation_joint_state`.
    * From :attr:`PyramidMimicSplitCommandCfg.approach_start_step`: :data:`APPROACH`, switching to :data:`IMITATE`
      at the base waypoint, as in the parent. :attr:`from_approach` marks these imitations.
    """

    cfg: PyramidMimicSplitCommandCfg

    def __init__(self, cfg: PyramidMimicSplitCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.from_approach = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        """Whether the episode imitates after approaching the base waypoint (not from a reference start)."""
        self.from_walk_end = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        """Whether the imitation episode starts from a recorded end-of-walk state instead of the reference."""
        self._start_root_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._start_joint_pos = torch.zeros_like(self.robot.data.joint_pos.torch)
        self._start_joint_vel = torch.zeros_like(self.robot.data.joint_vel.torch)
        # end-of-walk states; the root is stored in the frame of the base waypoint and the clip's start heading
        size = cfg.walk_end_buffer_size
        self._walk_end_root = torch.zeros(size, 13, device=self.device)
        self._walk_end_joint_pos = torch.zeros(size, self._start_joint_pos.shape[1], device=self.device)
        self._walk_end_joint_vel = torch.zeros_like(self._walk_end_joint_pos)
        self._walk_end_count = 0
        self._walk_end_cursor = 0
        self.metrics["walk_end_start"] = torch.zeros(self.num_envs, device=self.device)
        # pyramid curriculum: each robot's level, and the geometry of every tile
        self.curriculum_level = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        """Curriculum level (row) of each robot; it moves only on curriculum episodes."""
        curriculum = cfg.pyramid_curriculum
        if curriculum is not None:
            if curriculum.start_step < cfg.approach_start_step:
                raise ValueError(
                    f"The pyramid curriculum starts at control step {curriculum.start_step}, before every episode "
                    f"approaches the base (approach_start_step={cfg.approach_start_step})."
                )
            geometry = [
                [pyramid_tile_geometry(curriculum, row, col) for col in range(curriculum.num_cols)]
                for row in range(curriculum.num_rows)
            ]
            geometry = torch.tensor(geometry, dtype=torch.float32, device=self.device)
            self._tile_delta_table, self._tile_scale_table = geometry[..., 0], geometry[..., 1]
            self.metrics["curriculum_episode"] = torch.zeros(self.num_envs, device=self.device)

    def sample_episode(self, env_ids: Sequence[int] | torch.Tensor | slice):
        env_ids = self._as_env_ids(env_ids)
        pending = env_ids[self._pending[env_ids]]
        if len(pending) == 0:
            return
        super().sample_episode(pending)
        self.from_approach[pending] = False
        self.from_walk_end[pending] = False
        step = self._env.common_step_counter
        # curriculum episodes: the policy's only waypoint is the center of the top landing; the base stays internal
        loose = pending[self.loose[pending]]
        if len(loose) > 0:
            self.waypoint_pos_w[loose, :2] = self.pyramid_center_w[loose]
            self.waypoint_pos_w[loose, 2] = self.top_pos_w[loose, 2]
            self.goal_yaw[loose] = self.top_yaw[loose]
            self.initial_distance[loose] = (self.waypoint_pos_w[loose, :2] - self.start_pos_w[loose, :2]).norm(dim=-1)
        if step >= self.cfg.approach_start_step or self.cfg.top_waypoint_only:
            return  # as drawn by the parent: approach, then imitate, or a walk to the top
        self.mode[pending] = WALK
        if step < self.cfg.imitation_start_step:
            return
        ids = pending[torch.rand(len(pending), device=self.device) < self.cfg.imitation_probability]
        if len(ids) == 0:
            return
        # a frame before the last, so the episode does not end on its first step
        last = self.motion.lengths[self.motion_ids[ids]] - 1
        frames = (torch.rand(len(ids), device=self.device) * last).long()
        full_clip = torch.rand(len(ids), device=self.device) < self.cfg.full_clip_probability
        self.mode[ids] = IMITATE
        self.time_steps[ids] = torch.where(full_clip, 0, frames)
        if self._walk_end_count >= self.cfg.walk_end_min_states:
            self._draw_walk_end_states(ids[full_clip])
        self.waypoint_pos_w[ids] = self.top_pos_w[ids]
        self.goal_yaw[ids] = self.top_yaw[ids]
        self.start_pos_w[ids] = self.imitation_root_state(ids)[:, :3]

    def _update_metrics(self):
        if not self.cfg.top_waypoint_only:  # walks then end on the top, not at the base
            self._record_walk_end_states()
        approaching = self.mode == APPROACH
        super()._update_metrics()
        self.from_approach |= approaching & self.is_imitation
        self.metrics["walk_end_start"][:] = self.from_walk_end.float()
        if "curriculum_episode" in self.metrics:
            self.metrics["curriculum_episode"][:] = self.loose.float()

    def _base_arrival(self, reached: torch.Tensor) -> torch.Tensor:
        """The waypoint for the earlier regimes; for curriculum episodes, whose waypoint is the top center, the root
        within the reach radius of the base line of the clip's face (anywhere along it) and the height tolerance of
        the internal base waypoint."""
        offset = self.robot.data.root_pos_w.torch - self.base_pos_w
        outward = (offset[:, :2] * self._face_direction[self.motion_ids]).sum(dim=-1)
        at_base = (outward < self.cfg.reach_radius) & (offset[:, 2].abs() < self.cfg.height_tolerance)
        return torch.where(self.loose, at_base, reached)

    def _on_imitation_start(self, start: torch.Tensor) -> None:
        """Move the clip of starting curriculum episodes sideways along its face onto the robot."""
        ids = (start & self.loose).nonzero(as_tuple=False).flatten()
        if len(ids) == 0:
            return
        normal = self._face_direction[self.motion_ids[ids]]
        offset = self.robot.data.root_pos_w.torch[ids, :2] - self.base_pos_w[ids, :2]
        lateral = offset - (offset * normal).sum(dim=-1, keepdim=True) * normal
        self._ref_shift[ids, :2] += lateral
        self.base_pos_w[ids, :2] += lateral

    def assign_tiles(self, env_ids: torch.Tensor) -> None:
        """Move the robots about to reset onto the tile of their coming episode, before it is sampled.

        Before :attr:`PyramidCurriculumCfg.start_step` every episode runs on a clip pyramid tile. From there, a
        robot's finished curriculum episode first updates its level: up (with
        :attr:`PyramidCurriculumCfg.promotion_probability`) when it reached the top waypoint, down when it neither
        did nor got past :attr:`PyramidCurriculumCfg.demotion_phase` of the clip. Then each coming episode runs the
        curriculum with :attr:`PyramidCurriculumCfg.curriculum_probability`, on its level or, with
        :attr:`PyramidCurriculumCfg.replay_probability`, on a random lower one, in a random curriculum column;
        the others run the earlier regime on a random clip pyramid tile.
        """
        cur = self.cfg.pyramid_curriculum
        env_ids = self._as_env_ids(env_ids)
        count = len(env_ids)
        if cur is None or count == 0:
            return
        rand = lambda: torch.rand(count, device=self.device)  # noqa: E731
        curriculum = torch.zeros(count, dtype=torch.bool, device=self.device)
        if self._env.common_step_counter >= cur.start_step:
            finished = self.loose[env_ids]
            top = self.top_reached[env_ids]
            up = finished & top & (rand() < cur.promotion_probability)
            down = finished & ~top & (self.phase[env_ids] < cur.demotion_phase)
            level = self.curriculum_level[env_ids] + up.long() - down.long()
            self.curriculum_level[env_ids] = level.clamp(0, cur.num_rows - 1)
            curriculum = rand() < cur.curriculum_probability
        level = self.curriculum_level[env_ids]
        replay = rand() < cur.replay_probability
        curriculum_row = torch.where(replay, (rand() * (level + 1)).long(), level)
        curriculum_col = cur.num_clip_cols + (rand() * (cur.num_cols - cur.num_clip_cols)).long()
        rows = torch.where(curriculum, curriculum_row, (rand() * cur.num_rows).long())
        cols = torch.where(curriculum, curriculum_col, (rand() * cur.num_clip_cols).long())
        terrain = self._env.scene.terrain
        terrain.terrain_levels[env_ids] = rows
        terrain.terrain_types[env_ids] = cols
        terrain.env_origins[env_ids] = terrain.terrain_origins[rows, cols]
        self.tile_delta[env_ids] = self._tile_delta_table[rows, cols]
        self.tile_scale[env_ids] = self._tile_scale_table[rows, cols]
        self.loose[env_ids] = curriculum

    def _record_walk_end_states(self):
        """Store the state of every walk episode that first reaches the base waypoint on this step."""
        ids = (((self.mode == WALK) & self.waypoint_reached & ~self.reached).nonzero(as_tuple=False)).flatten()
        count = len(ids)
        if count == 0:
            return
        data = self.robot.data
        yaw = self._yaw_quat(self.goal_yaw[ids])
        root = torch.cat(
            (
                math_utils.quat_apply_inverse(yaw, data.root_pos_w.torch[ids] - self.waypoint_pos_w[ids]),
                math_utils.quat_mul(math_utils.quat_inv(yaw), data.root_quat_w.torch[ids]),
                math_utils.quat_apply_inverse(yaw, data.root_lin_vel_w.torch[ids]),
                math_utils.quat_apply_inverse(yaw, data.root_ang_vel_w.torch[ids]),
            ),
            dim=-1,
        )
        size = self.cfg.walk_end_buffer_size
        slots = (self._walk_end_cursor + torch.arange(count, device=self.device)) % size
        self._walk_end_root[slots] = root
        self._walk_end_joint_pos[slots] = data.joint_pos.torch[ids]
        self._walk_end_joint_vel[slots] = data.joint_vel.torch[ids]
        self._walk_end_cursor = (self._walk_end_cursor + count) % size
        self._walk_end_count = min(self._walk_end_count + count, size)

    def _draw_walk_end_states(self, ids: torch.Tensor):
        """Give ``ids`` recorded end-of-walk states, moved to their clip's base waypoint and start heading."""
        if len(ids) == 0:
            return
        drawn = torch.randint(self._walk_end_count, (len(ids),), device=self.device)
        stored = self._walk_end_root[drawn]
        # called before the waypoint moves to the top landing
        yaw = self._yaw_quat(self.goal_yaw[ids])
        self._start_root_state[ids] = torch.cat(
            (
                self.waypoint_pos_w[ids] + math_utils.quat_apply(yaw, stored[:, :3]),
                math_utils.quat_mul(yaw, stored[:, 3:7]),
                math_utils.quat_apply(yaw, stored[:, 7:10]),
                math_utils.quat_apply(yaw, stored[:, 10:13]),
            ),
            dim=-1,
        )
        self._start_joint_pos[ids] = self._walk_end_joint_pos[drawn]
        self._start_joint_vel[ids] = self._walk_end_joint_vel[drawn]
        self.from_walk_end[ids] = True

    def _yaw_quat(self, yaw: torch.Tensor) -> torch.Tensor:
        return math_utils.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)

    def imitation_root_state(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Root state an imitation episode starts from: a recorded end-of-walk state or the reference's.

        Same layout as :meth:`reference_root_state`.
        """
        state = self.reference_root_state(env_ids)
        walk_end = self.from_walk_end[env_ids]
        state[walk_end] = self._start_root_state[env_ids[walk_end]]
        return state

    def imitation_joint_state(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Joint positions [rad] and velocities [rad/s] an imitation episode starts from, robot order."""
        joint_pos, joint_vel = self.reference_joint_state(env_ids)
        walk_end = self.from_walk_end[env_ids].unsqueeze(-1)
        joint_pos = torch.where(walk_end, self._start_joint_pos[env_ids], joint_pos)
        joint_vel = torch.where(walk_end, self._start_joint_vel[env_ids], joint_vel)
        return joint_pos, joint_vel

    def reference_root_state(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Reference root pose and velocity at each episode's current frame, in the world frame.

        Shape is (len(env_ids), 13): position [m], quaternion (x, y, z, w), linear [m/s] and angular [rad/s]
        velocity.
        """
        frames = self.motion.frame_indices(self.motion_ids[env_ids], self.time_steps[env_ids])
        root = self.motion_root_index
        return torch.cat(
            (
                self.motion.body_pos_w[frames, root]
                + self._env.scene.env_origins[env_ids]
                + self._ref_offset(frames, env_ids),
                self.motion.body_quat_w[frames, root],
                self.motion.body_lin_vel_w[frames, root],
                self.motion.body_ang_vel_w[frames, root],
            ),
            dim=-1,
        )

    def reference_joint_state(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference joint positions [rad] and velocities [rad/s] at each episode's current frame, robot order."""
        frames = self.motion.frame_indices(self.motion_ids[env_ids], self.time_steps[env_ids])
        return self.motion.joint_pos[frames], self.motion.joint_vel[frames]


@configclass
class PyramidMimicSplitCommandCfg(PyramidMimicCommandCfg):
    """Configuration for :class:`PyramidMimicSplitCommand`."""

    class_type: type = PyramidMimicSplitCommand

    imitation_probability: float = 0.5
    """Fraction of episodes reset between :attr:`imitation_start_step` and :attr:`approach_start_step` that imitate
    from a reference start; the rest walk."""

    approach_start_step: int = 5000 * 24
    """Control step from which every episode approaches the base waypoint and then imitates, as in the parent."""

    full_clip_probability: float = 0.2
    """Fraction of imitation episodes that start at the clip's first frame instead of a random one. These start from
    recorded end-of-walk states once at least :attr:`walk_end_min_states` exist."""

    walk_end_buffer_size: int = 4096
    """End-of-walk states kept; the oldest are overwritten."""

    walk_end_min_states: int = 64
    """Recorded states needed before they are used; until then the episode starts on the clip's first pose."""

    pyramid_curriculum: PyramidCurriculumCfg | None = None
    """Tiles of the pyramid curriculum regime, assigned by :meth:`PyramidMimicSplitCommand.assign_tiles`; the
    terrain generator must be built from the same configuration. None keeps every episode on the clip's pyramid."""
