# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stair-climbing motion reference: phase command, critic prior, imitation reward, and end-of-clip termination."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class StairMotionCommand(CommandTerm):
    """Time-indexed reference from one motion-library clip, aligned to the clip-paired stair terrain.

    The clip frame coincides with the terrain origin of each environment (see
    :class:`~..terrains.MotionStairsTerrainCfg`), so reference root poses only need the environment origin.
    On reset the term samples a start frame and writes the reference root and joint state to the robot
    (reference state initialization). It must be the first command term so that later terms that
    read the robot state on reset, such as the velocity command's heading target, see the written pose.
    The phase advances by one frame per control step and holds at the last frame;
    :func:`motion_clip_end` ends the episode there.
    """

    cfg: StairMotionCommandCfg

    def __init__(self, cfg: StairMotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]

        if not os.path.isfile(cfg.motion_file):
            raise FileNotFoundError(
                f"Motion file not found: {cfg.motion_file}. Clone https://github.com/jnskkmhr/motion_library"
                " next to IsaacLab or set G1_STAIR_MOTION_FILE."
            )
        with np.load(cfg.motion_file, allow_pickle=False) as data:
            self.fps = float(np.asarray(data["fps"]).reshape(-1)[0])
            if "quaternion_order" in data and str(np.asarray(data["quaternion_order"]).item()) != "xyzw":
                raise ValueError("Expected xyzw body quaternions in the motion file.")
            joint_names = data["joint_names"].tolist()
            root_index = data["body_names"].tolist().index(cfg.root_body_name)
            joint_pos = data["joint_pos"]
            joint_vel = data["joint_vel"]
            root_pos = data["body_pos_w"][:, root_index]
            root_quat = data["body_quat_w"][:, root_index]
            root_lin_vel = data["body_lin_vel_w"][:, root_index]
            root_ang_vel = data["body_ang_vel_w"][:, root_index]

        if set(joint_names) != set(self.robot.joint_names):
            raise ValueError("Motion and robot joint names must cover the same joints.")
        if not math.isclose(self.fps * env.step_dt, 1.0, rel_tol=1.0e-6):
            raise ValueError(f"Motion FPS {self.fps} must match the control rate {1.0 / env.step_dt} Hz.")

        # store the reference in articulation joint order
        order = [joint_names.index(name) for name in self.robot.joint_names]
        self.ref_joint_pos = torch.tensor(joint_pos[:, order], dtype=torch.float32, device=self.device)
        self.ref_joint_vel = torch.tensor(joint_vel[:, order], dtype=torch.float32, device=self.device)
        self.ref_root_pose = torch.tensor(
            np.concatenate((root_pos, root_quat), -1), dtype=torch.float32, device=self.device
        )
        self.ref_root_vel = torch.tensor(
            np.concatenate((root_lin_vel, root_ang_vel), -1), dtype=torch.float32, device=self.device
        )
        self.num_frames = self.ref_joint_pos.shape[0]
        min_remaining = round(cfg.min_remaining_time_s * self.fps)
        self.max_start_frame = max(self.num_frames - 1 - min_remaining, 0) if cfg.random_start else 0

        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["phase"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "StairMotionCommand:\n"
        msg += f"\tMotion file: {self.cfg.motion_file}\n"
        msg += f"\tFrames: {self.num_frames} at {self.fps} Hz\n"
        msg += f"\tStart frames: [0, {self.max_start_frame}]"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Reference joint positions [rad] in articulation order. Shape is (num_envs, num_joints)."""
        return self.ref_joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        """Reference joint velocities [rad/s] in articulation order. Shape is (num_envs, num_joints)."""
        return self.ref_joint_vel[self.time_steps]

    @property
    def phase(self) -> torch.Tensor:
        """Clip progress in [0, 1]. Shape is (num_envs,)."""
        return self.time_steps.float() / max(self.num_frames - 1, 1)

    @property
    def at_end(self) -> torch.Tensor:
        """Whether the reference has reached its last frame. Shape is (num_envs,)."""
        return self.time_steps >= self.num_frames - 1

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        error = self.command - self.robot.data.joint_pos.torch
        self.metrics["error_joint_pos"] = torch.linalg.vector_norm(error, dim=-1)
        self.metrics["phase"] = self.phase

    def _resample_command(self, env_ids: Sequence[int] | torch.Tensor | slice):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return
        frames = torch.randint(0, self.max_start_frame + 1, (env_ids.numel(),), device=self.device)
        self.time_steps[env_ids] = frames

        # reference state initialization in the clip-aligned terrain frame
        root_pose = self.ref_root_pose[frames].clone()
        root_pose[:, :3] += self._env.scene.env_origins[env_ids]
        root_pose[:, 1] += torch.empty(env_ids.numel(), device=self.device).uniform_(*self.cfg.root_y_range)
        joint_pos = self.ref_joint_pos[frames]
        joint_limits = self.robot.data.soft_joint_pos_limits.torch[env_ids]
        joint_pos = joint_pos.clamp(joint_limits[..., 0], joint_limits[..., 1])
        self.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=self.ref_root_vel[frames], env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=self.ref_joint_vel[frames], env_ids=env_ids)

    def _update_command(self):
        self.time_steps = torch.clamp(self.time_steps + 1, max=self.num_frames - 1)


@configclass
class StairMotionCommandCfg(CommandTermCfg):
    """Configuration for :class:`StairMotionCommand`."""

    class_type: type = StairMotionCommand

    asset_name: str = MISSING
    """Name of the robot asset."""

    motion_file: str = MISSING
    """Path to a motion-library NPZ with 29 named joints and xyzw body quaternions."""

    root_body_name: str = "pelvis"
    """Motion body that corresponds to the articulation root."""

    random_start: bool = True
    """Sample the start frame uniformly; otherwise always start at frame 0."""

    min_remaining_time_s: float = 1.0
    """Latest random start leaves at least this much clip time [s]."""

    root_y_range: tuple[float, float] = (0.0, 0.0)
    """Lateral reset offset [m]. The clip's stairs are 1.2 m wide, so small offsets keep the reference valid."""

    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    """The phase is only resampled on reset."""


def _motion_term(env: ManagerBasedRLEnv, command_name: str) -> StairMotionCommand:
    return env.command_manager.get_term(command_name)


def motion_ref_joint_pos_rel(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reference joint positions [rad] relative to the default pose, in ``asset_cfg`` joint order."""
    asset: Articulation = env.scene[asset_cfg.name]
    ref = _motion_term(env, command_name).command[:, asset_cfg.joint_ids]
    return ref - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]


def motion_ref_joint_vel(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reference joint velocities [rad/s], in ``asset_cfg`` joint order."""
    return _motion_term(env, command_name).joint_vel[:, asset_cfg.joint_ids]


def motion_joint_pos_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reference minus measured joint positions [rad], in ``asset_cfg`` joint order."""
    asset: Articulation = env.scene[asset_cfg.name]
    ref = _motion_term(env, command_name).command[:, asset_cfg.joint_ids]
    return ref - asset.data.joint_pos.torch[:, asset_cfg.joint_ids]


def motion_phase(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Clip progress in [0, 1]. Shape is (num_envs, 1)."""
    return _motion_term(env, command_name).phase.unsqueeze(-1)


def motion_joint_pos_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward joint positions that follow the reference: ``exp(-mean(err^2) / std^2)`` with ``std`` in [rad]."""
    error = motion_joint_pos_error(env, command_name, asset_cfg)
    return torch.exp(-error.square().mean(dim=-1) / std**2)


def motion_clip_end(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """End the episode once the reference reaches its last frame."""
    return _motion_term(env, command_name).at_end
