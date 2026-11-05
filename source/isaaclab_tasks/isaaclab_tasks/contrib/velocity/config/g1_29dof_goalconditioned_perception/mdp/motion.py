# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Destination command trained with separate reference-imitation and goal-reaching episodes."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import math as math_utils
from isaaclab.utils.configclass import configclass

from .terrains import motion_stairs_height_scale

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class StairMotionLibrary:
    """Reference clips packed in articulation joint order and a common body order.

    Packed frame rows are storage indices; episode clocks stay local to each clip, so use
    :meth:`frame_indices` to look up a clip's frame. Clips must match the control rate and share
    their terrain, which is built from the first clip.
    """

    def __init__(
        self,
        motion_files: Sequence[str],
        joint_names: Sequence[str],
        body_names: Sequence[str],
        step_dt: float,
        device: str,
    ):
        if not motion_files:
            raise ValueError("At least one motion file is required.")
        arrays = {
            key: [] for key in ("joint_pos", "joint_vel", "body_pos", "body_quat", "body_lin_vel", "body_ang_vel")
        }
        lengths = []
        terrain = None
        for path in motion_files:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Motion file not found: {path}")
            with np.load(path, allow_pickle=False) as data:
                fps = float(np.asarray(data["fps"]).reshape(-1)[0])
                if not math.isclose(fps * step_dt, 1.0, rel_tol=1.0e-6):
                    raise ValueError(f"{path}: motion FPS {fps} must match the control rate {1.0 / step_dt} Hz.")
                if "quaternion_order" in data and str(np.asarray(data["quaternion_order"]).item()) != "xyzw":
                    raise ValueError(f"{path}: expected xyzw body quaternions.")
                clip_joints = data["joint_names"].tolist()
                clip_bodies = data["body_names"].tolist()
                if set(clip_joints) != set(joint_names):
                    raise ValueError(f"{path}: motion and robot joint names must cover the same joints.")
                missing = set(body_names) - set(clip_bodies)
                if missing:
                    raise ValueError(f"{path}: tracked bodies {sorted(missing)} are missing.")
                clip_terrain = np.concatenate((data["terrain_position"], data["terrain_dimension"]), axis=-1)
                if terrain is None:
                    terrain = clip_terrain
                elif clip_terrain.shape != terrain.shape or not np.allclose(clip_terrain, terrain):
                    raise ValueError(f"{path}: all clips must share the first clip's terrain.")
                joints = [clip_joints.index(name) for name in joint_names]
                bodies = [clip_bodies.index(name) for name in body_names]
                arrays["joint_pos"].append(data["joint_pos"][:, joints])
                arrays["joint_vel"].append(data["joint_vel"][:, joints])
                arrays["body_pos"].append(data["body_pos_w"][:, bodies])
                arrays["body_quat"].append(data["body_quat_w"][:, bodies])
                arrays["body_lin_vel"].append(data["body_lin_vel_w"][:, bodies])
                arrays["body_ang_vel"].append(data["body_ang_vel_w"][:, bodies])
                lengths.append(data["joint_pos"].shape[0])

        def pack(key: str) -> torch.Tensor:
            return torch.tensor(np.concatenate(arrays[key]), dtype=torch.float32, device=device)

        self.fps = fps
        self.body_names = list(body_names)
        self.joint_pos = pack("joint_pos")
        """Joint positions [rad] in articulation order. Shape is (frames, joints)."""
        self.joint_vel = pack("joint_vel")
        """Joint velocities [rad/s] in articulation order. Shape is (frames, joints)."""
        self.body_pos_w = pack("body_pos")
        """Body positions [m] in the clip frame. Shape is (frames, bodies, 3)."""
        self.body_quat_w = pack("body_quat")
        """Body orientations (x, y, z, w) in the clip frame. Shape is (frames, bodies, 4)."""
        self.body_lin_vel_w = pack("body_lin_vel")
        """Body linear velocities [m/s] in the clip frame. Shape is (frames, bodies, 3)."""
        self.body_ang_vel_w = pack("body_ang_vel")
        """Body angular velocities [rad/s] in the clip frame. Shape is (frames, bodies, 3)."""
        self.lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        self.terrain_top = float((terrain[:, 2] + terrain[:, 5] / 2.0).max())
        """Top of the clips' terrain boxes [m] in the clip frame, i.e. the landing height."""
        self.starts = self.lengths.cumsum(0) - self.lengths

    def frame_indices(self, motion_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        """Map clip IDs and clip-local frame numbers to packed rows."""
        return self.starts[motion_ids] + time_steps


class GoalMotionCommand(CommandTerm):
    """Destination root pose for the actor, with a reference clip for imitation episodes.

    Each reset samples an imitation or a task episode and one of the clips. Imitation episodes start
    on a random reference frame and their destination is the clip's final root pose. Task episodes start
    standing near the first reference root and their destination is shifted into the top landing with noise.
    The actor observes only :attr:`command`; the episode type and reference are privileged.

    With :attr:`GoalMotionCommandCfg.task_terrain` set, each reset also moves the environment to a terrain
    tile: imitation episodes to the clips' own staircase, task episodes to a copy whose rises are scaled by
    the row :attr:`task_level`, which :func:`~.curriculums.task_terrain_levels` raises after successes.
    Task destinations are lowered with the scaled landing.

    The episode draw happens in :meth:`resample_time_steps`, which the reset events call before the
    command manager resets, and runs at most once per reset.
    """

    cfg: GoalMotionCommandCfg

    def __init__(self, cfg: GoalMotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.motion = StairMotionLibrary(
            cfg.motion_files, self.robot.joint_names, cfg.body_names, env.step_dt, self.device
        )
        self.num_motions = len(cfg.motion_files)
        if self.robot.body_names[0] not in cfg.body_names:
            raise ValueError(f"The root body {self.robot.body_names[0]} must be a tracked body.")
        self.root_index = cfg.body_names.index(self.robot.body_names[0])
        self.anchor_index = cfg.body_names.index(cfg.anchor_body_name)
        self.body_ids = torch.tensor(
            self.robot.find_bodies(cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )
        self.robot_anchor_id = self.robot.body_names.index(cfg.anchor_body_name)
        self.contact_sensor = env.scene.sensors[cfg.contact_sensor_name]
        self.foot_ids = self.contact_sensor.find_bodies(cfg.foot_body_names, preserve_order=True)[0]

        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.is_imitation = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.goal_quat_w[:, 3] = 1.0
        self.goal_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._setup_terrain(env)
        self._finished_task = torch.zeros_like(self.is_imitation)
        self._finished_success = torch.zeros_like(self.is_imitation)
        self._resample_pending = torch.ones_like(self.is_imitation)
        num_bodies = len(cfg.body_names)
        self.body_pos_relative_w = torch.zeros(self.num_envs, num_bodies, 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, num_bodies, 4, device=self.device)
        self.body_quat_relative_w[..., 3] = 1.0

        for name in (
            "goal_distance",
            "goal_heading_error",
            "imitation_probability",
            "error_anchor_pos",
            "error_anchor_rot",
            "error_body_pos",
            "error_joint_pos",
            "phase",
        ):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "GoalMotionCommand:\n"
        msg += f"\tMotion files: {len(self.cfg.motion_files)}\n"
        msg += f"\tFrames per clip: {self.motion.lengths.tolist()} at {self.motion.fps} Hz\n"
        msg += f"\tFinal imitation probability: {self.cfg.final_imitation_probability}"
        return msg

    """
    Properties: destination.
    """

    @property
    def command(self) -> torch.Tensor:
        """Destination XYZ [m] in the robot yaw frame, and sine and cosine of the yaw error. Shape is (N, 5)."""
        yaw = math_utils.yaw_quat(self.robot.data.root_quat_w.torch)
        error_b = math_utils.quat_apply_inverse(yaw, self.goal_error)
        heading = self.heading_error
        return torch.cat((error_b, heading.sin().unsqueeze(-1), heading.cos().unsqueeze(-1)), dim=-1)

    @property
    def goal_error(self) -> torch.Tensor:
        """World displacement [m] from the robot root to its destination. Shape is (N, 3)."""
        return self.goal_pos_w - self.robot.data.root_pos_w.torch

    @property
    def heading_error(self) -> torch.Tensor:
        """Destination minus robot yaw [rad], wrapped to (-pi, pi]. Shape is (N,)."""
        goal_yaw = math_utils.euler_xyz_from_quat(self.goal_quat_w)[2]
        root_yaw = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w.torch)[2]
        return math_utils.wrap_to_pi(goal_yaw - root_yaw)

    @property
    def position_reached(self) -> torch.Tensor:
        """Whether the root is within the destination's horizontal and vertical tolerances. Shape is (N,)."""
        error = self.goal_error
        return (error[:, :2].norm(dim=-1) < self.cfg.position_tolerance) & (
            error[:, 2].abs() < self.cfg.height_tolerance
        )

    @property
    def feet_supported(self) -> torch.Tensor:
        """Whether a foot carries more than the contact threshold [N]. Shape is (N,)."""
        forces = self.contact_sensor.data.net_forces_w.torch[:, self.foot_ids]
        return (forces.norm(dim=-1) > self.cfg.contact_threshold).any(dim=-1)

    @property
    def sampling_probability(self) -> float:
        """Probability that a new episode imitates the reference at the current control step."""
        if self.cfg.imitation_probability is not None:
            return self.cfg.imitation_probability
        elapsed = (self._env.common_step_counter - self.cfg.warmup_steps) / self.cfg.transition_steps
        return 1.0 - min(max(elapsed, 0.0), 1.0) * (1.0 - self.cfg.final_imitation_probability)

    def task_height_scale(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Stair height scale of the task tile at each environment's :attr:`task_level`. Shape is (len(env_ids),)."""
        return self.level_height_scales[self.task_level[env_ids]]

    """
    Properties: reference at the current frame, in the world frame.
    """

    @property
    def reference_frame_indices(self) -> torch.Tensor:
        return self.motion.frame_indices(self.motion_ids, self.time_steps)

    @property
    def motion_lengths(self) -> torch.Tensor:
        return self.motion.lengths[self.motion_ids]

    @property
    def phase(self) -> torch.Tensor:
        """Clip progress in [0, 1]. Shape is (N,)."""
        return self.time_steps.float() / (self.motion_lengths - 1).clamp(min=1)

    @property
    def has_reached_end(self) -> torch.Tensor:
        return self.time_steps >= self.motion_lengths - 1

    @property
    def joint_pos(self) -> torch.Tensor:
        """Reference joint positions [rad] in articulation order. Shape is (N, J)."""
        return self.motion.joint_pos[self.reference_frame_indices]

    @property
    def joint_vel(self) -> torch.Tensor:
        """Reference joint velocities [rad/s] in articulation order. Shape is (N, J)."""
        return self.motion.joint_vel[self.reference_frame_indices]

    @property
    def body_pos_w(self) -> torch.Tensor:
        """Reference body positions [m]. Shape is (N, B, 3)."""
        return self.motion.body_pos_w[self.reference_frame_indices] + self._env.scene.env_origins[:, None]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.reference_frame_indices]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.reference_frame_indices]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.reference_frame_indices]

    @property
    def root_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.reference_frame_indices, self.root_index] + self._env.scene.env_origins

    @property
    def root_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.reference_frame_indices, self.root_index]

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.reference_frame_indices, self.root_index]

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.reference_frame_indices, self.root_index]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.reference_frame_indices, self.anchor_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.reference_frame_indices, self.anchor_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.reference_frame_indices, self.anchor_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.reference_frame_indices, self.anchor_index]

    """
    Properties: robot bodies matching the reference.
    """

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w.torch[:, self.body_ids]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w.torch[:, self.body_ids]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w.torch[:, self.body_ids]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w.torch[:, self.robot_anchor_id]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w.torch[:, self.robot_anchor_id]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w.torch[:, self.robot_anchor_id]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w.torch[:, self.robot_anchor_id]

    """
    Operations.
    """

    def resample_time_steps(self, env_ids: Sequence[int] | torch.Tensor | slice):
        """Draw the episode type, clip, start frame, and destination for ``env_ids``, once per reset.

        The reset events run before the command manager resets, so they call this to place the robot
        on the new episode's start; :meth:`_resample_command` re-arms the draw for the next reset.
        """
        env_ids = self._as_env_ids(env_ids)
        pending = env_ids[self._resample_pending[env_ids]]
        if len(pending) == 0:
            return
        count = len(pending)
        # keep the finished episode's outcome for logging before overwriting it
        self._finished_task[pending] = ~self.is_imitation[pending]
        self._finished_success[pending] = self.goal_reached[pending]
        self.goal_reached[pending] = False

        self.is_imitation[pending] = torch.rand(count, device=self.device) < self.sampling_probability
        self.motion_ids[pending] = torch.randint(self.num_motions, (count,), device=self.device)
        self._place_on_terrain(pending)
        # imitation starts on a random frame (or the first, for full clips); task episodes start on the first
        frames = (torch.rand(count, device=self.device) * self.motion_lengths[pending]).long()
        full_clip = torch.rand(count, device=self.device) < self.cfg.full_clip_probability
        self.time_steps[pending] = torch.where(self.is_imitation[pending] & ~full_clip, frames, 0)

        # destination: the clip's final root pose; task destinations move into the landing with noise
        last = self.motion.frame_indices(self.motion_ids[pending], self.motion_lengths[pending] - 1)
        self.goal_pos_w[pending] = self.motion.body_pos_w[last, self.root_index] + self._env.scene.env_origins[pending]
        self.goal_quat_w[pending] = math_utils.yaw_quat(self.motion.body_quat_w[last, self.root_index])
        task = (~self.is_imitation[pending]).float().unsqueeze(-1)
        offset = torch.tensor(self.cfg.task_goal_offset, device=self.device).expand(count, -1)
        self.goal_pos_w[pending] += math_utils.quat_apply(self.goal_quat_w[pending], offset) * task
        noise = math_utils.sample_uniform(
            -self.cfg.goal_position_noise, self.cfg.goal_position_noise, (count, 2), self.device
        )
        self.goal_pos_w[pending, :2] += noise * task
        # the task tile's landing sits lower or higher by the scaled box heights
        landing_shift = (self.task_height_scale(pending) - 1.0) * self.motion.terrain_top
        self.goal_pos_w[pending, 2] += landing_shift * task.squeeze(-1)
        self._resample_pending[pending] = False

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)
        extras = super().reset(env_ids)
        tasks = self._finished_task[env_ids]
        extras["task_episode_fraction"] = tasks.float().mean()
        extras["task_success_rate"] = (self._finished_success[env_ids] & tasks).sum() / tasks.sum().clamp_min(1)
        return extras

    """
    Implementation specific functions.
    """

    def _setup_terrain(self, env: ManagerBasedRLEnv):
        """Find the imitation and task tile columns and the task rows' stair height scales."""
        self.task_level = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        """Per-environment task tile row; see :func:`~.curriculums.task_terrain_levels`."""
        self.level_height_scales = torch.ones(1, device=self.device)
        """Stair height scale of each task tile row. Shape is (num_levels,)."""
        self.num_levels = 1
        if self.cfg.task_terrain is None:
            return
        terrain = env.scene.terrain
        generator = terrain.cfg.terrain_generator
        if generator is None or not generator.curriculum or terrain.terrain_origins is None:
            raise ValueError("GoalMotionCommand with task_terrain needs a curriculum terrain generator.")
        if tuple(generator.difficulty_range) != (0.0, 1.0):
            raise ValueError("GoalMotionCommand with task_terrain needs difficulty_range (0, 1).")
        task_cfg = generator.sub_terrains[self.cfg.task_terrain]
        if task_cfg.num_levels != generator.num_rows:
            raise ValueError(f"Sub-terrain '{self.cfg.task_terrain}' needs num_levels == num_rows.")
        # the generator assigns sub-terrains to columns by proportion (TerrainGenerator._generate_curriculum_terrains)
        names = list(generator.sub_terrains)
        proportions = np.array([generator.sub_terrains[name].proportion for name in names], dtype=float)
        cumulative = np.cumsum(proportions / proportions.sum())
        column_names = [
            names[np.min(np.where(col / generator.num_cols + 0.001 < cumulative)[0])]
            for col in range(generator.num_cols)
        ]
        self.terrain_columns: dict[bool, torch.Tensor] = {}
        """Tile columns of imitation (True) and task (False) episodes."""
        for is_imitation, name in ((True, self.cfg.imitation_terrain), (False, self.cfg.task_terrain)):
            cols = [col for col, column_name in enumerate(column_names) if column_name == name]
            if not cols:
                raise ValueError(f"No terrain column holds sub-terrain '{name}'.")
            self.terrain_columns[is_imitation] = torch.tensor(cols, dtype=torch.long, device=self.device)
        self.num_levels = generator.num_rows
        self.level_height_scales = torch.tensor(
            [motion_stairs_height_scale(task_cfg, level) for level in range(self.num_levels)], device=self.device
        )
        max_init_level = min(self.cfg.task_max_init_level, self.num_levels - 1)
        self.task_level[:] = torch.randint(max_init_level + 1, (self.num_envs,), device=self.device)

    def _place_on_terrain(self, env_ids: torch.Tensor):
        """Move imitation episodes to a clip-staircase tile and task episodes to a tile of their :attr:`task_level`."""
        if self.cfg.task_terrain is None:
            return
        if not self.cfg.task_terrain_curriculum:
            self.task_level[env_ids] = torch.randint(self.num_levels, (len(env_ids),), device=self.device)
        terrain = self._env.scene.terrain
        for is_imitation, cols in self.terrain_columns.items():
            ids = env_ids[self.is_imitation[env_ids] == is_imitation]
            if len(ids) == 0:
                continue
            terrain.terrain_types[ids] = cols[torch.randint(len(cols), (len(ids),), device=self.device)]
            # every imitation row holds the same staircase
            terrain.terrain_levels[ids] = (
                torch.randint(self.num_levels, (len(ids),), device=self.device)
                if is_imitation
                else self.task_level[ids]
            )
            terrain.env_origins[ids] = terrain.terrain_origins[terrain.terrain_levels[ids], terrain.terrain_types[ids]]

    def _as_env_ids(self, env_ids: Sequence[int] | torch.Tensor | slice) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)[env_ids]
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

    def _update_metrics(self):
        self.goal_reached |= (
            self.position_reached & (self.heading_error.abs() < self.cfg.heading_tolerance) & self.feet_supported
        )
        self.metrics["goal_distance"][:] = self.goal_error.norm(dim=-1)
        self.metrics["goal_heading_error"][:] = self.heading_error.abs()
        self.metrics["imitation_probability"][:] = self.sampling_probability
        self.metrics["error_anchor_pos"][:] = (self.anchor_pos_w - self.robot_anchor_pos_w).norm(dim=-1)
        self.metrics["error_anchor_rot"][:] = math_utils.quat_error_magnitude(
            self.anchor_quat_w, self.robot_anchor_quat_w
        )
        self.metrics["error_body_pos"][:] = (self.body_pos_relative_w - self.robot_body_pos_w).norm(dim=-1).mean(-1)
        self.metrics["error_joint_pos"][:] = (self.joint_pos - self.robot.data.joint_pos.torch).norm(dim=-1)
        self.metrics["phase"][:] = self.phase

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = self._as_env_ids(env_ids)
        if len(env_ids) == 0:
            return
        self.resample_time_steps(env_ids)
        self._resample_pending[env_ids] = True
        # terminations run before the first reference update; compare against the new start, not stale buffers
        self.body_pos_relative_w[env_ids] = self.body_pos_w[env_ids]
        self.body_quat_relative_w[env_ids] = self.body_quat_w[env_ids]

    def _update_command(self):
        # hold the last frame; ``motion_end`` ends imitation episodes there, task episodes continue
        self.time_steps = torch.minimum(self.time_steps + 1, self.motion_lengths - 1)

        # Move the reference bodies under the robot: robot anchor XY and yaw, reference anchor height.
        num_bodies = len(self.cfg.body_names)
        anchor_pos = self.anchor_pos_w
        delta_pos = self.robot_anchor_pos_w.clone()
        delta_pos[:, 2] = anchor_pos[:, 2]
        delta_quat = math_utils.yaw_quat(
            math_utils.quat_mul(self.robot_anchor_quat_w, math_utils.quat_inv(self.anchor_quat_w))
        )
        delta_quat = delta_quat[:, None].expand(-1, num_bodies, -1)
        self.body_quat_relative_w = math_utils.quat_mul(delta_quat, self.body_quat_w)
        self.body_pos_relative_w = delta_pos[:, None] + math_utils.quat_apply(
            delta_quat, self.body_pos_w - anchor_pos[:, None]
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis and not hasattr(self, "goal_visualizer"):
            self.goal_visualizer = VisualizationMarkers(self.cfg.goal_visualizer_cfg)
        if hasattr(self, "goal_visualizer"):
            self.goal_visualizer.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        if self.robot.is_initialized:
            self.goal_visualizer.visualize(self.goal_pos_w, self.goal_quat_w)


@configclass
class GoalMotionCommandCfg(CommandTermCfg):
    """Configuration for :class:`GoalMotionCommand`."""

    class_type: type = GoalMotionCommand

    asset_name: str = MISSING
    """Name of the robot asset."""

    motion_files: list[str] = MISSING
    """Reference NPZ files sharing one terrain, sampled uniformly at each reset."""

    body_names: list[str] = MISSING
    """Tracked reference bodies, including the root and anchor bodies."""

    anchor_body_name: str = "torso_link"
    """Body whose pose anchors the reference body comparison."""

    foot_body_names: list[str] = ["left_ankle_roll_link", "right_ankle_roll_link"]
    """Contact-sensor bodies that support the robot at the destination."""

    contact_sensor_name: str = "contact_forces"

    warmup_steps: int = 120_000
    """Imitation-only control steps (5,000 rollouts of 24 steps)."""

    transition_steps: int = 120_000
    """Control steps over which the imitation probability falls to its final value."""

    final_imitation_probability: float = 0.5
    """Probability that an episode imitates the reference after the transition."""

    imitation_probability: float | None = None
    """Fixed imitation probability overriding the schedule, e.g. zero for task evaluation."""

    full_clip_probability: float = 0.2
    """Fraction of imitation episodes that start on the first frame instead of a random one."""

    imitation_terrain: str = "motion_stairs"
    """Sub-terrain holding the clips' own staircase, used by imitation episodes when :attr:`task_terrain` is set."""

    task_terrain: str | None = None
    """Sub-terrain with height-scaled staircases (one scale per row) for task episodes.

    None leaves terrain placement to the caller and keeps the clips' heights for task destinations.
    """

    task_terrain_curriculum: bool = True
    """Place task episodes on their success-driven :attr:`GoalMotionCommand.task_level`; otherwise on a random row."""

    task_max_init_level: int = 0
    """Highest task tile row an environment starts on."""

    task_start_offset: tuple[float, float, float] = (-0.5, 0.0, 0.0)
    """Standing-start offset [m] from the first reference root, in its yaw frame."""

    task_start_position_noise: float = 0.1
    """Uniform task start XY noise [m] in the first reference root's yaw frame."""

    task_start_yaw_noise: float = 0.1
    """Uniform task start yaw noise [rad]."""

    task_goal_offset: tuple[float, float, float] = (0.75, 0.0, 0.0)
    """Task destination offset [m] from the final reference root, in its yaw frame."""

    goal_position_noise: float = 0.35
    """Uniform task destination XY noise [m] in world axes."""

    position_tolerance: float = 0.35
    """Destination XY tolerance [m]."""

    height_tolerance: float = 0.25
    """Destination height tolerance [m]."""

    heading_tolerance: float = 0.4
    """Destination yaw tolerance [rad]."""

    contact_threshold: float = 5.0
    """Foot force [N] that counts as support at the destination."""

    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    """Destinations are only resampled on reset."""

    goal_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/destination")
