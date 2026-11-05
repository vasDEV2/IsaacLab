# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reference tracking rewards for imitation episodes and destination rewards for task episodes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_error_magnitude

from isaaclab_tasks.contrib.velocity.config.g1_29dof_perception_motion_style.mdp.motion_style import (
    MotionMatchingStyle,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .motion import GoalMotionCommand


def _command(env: ManagerBasedRLEnv, command_name: str) -> GoalMotionCommand:
    return env.command_manager.get_term(command_name)


def _body_indices(command: GoalMotionCommand, body_names: list[str] | None) -> list[int] | slice:
    if body_names is None:
        return slice(None)
    return [command.cfg.body_names.index(name) for name in body_names]


"""
Imitation episodes: track the reference. Each term is zero in task episodes.
"""


def motion_anchor_pos_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Track the reference anchor position in the world frame: ``exp(-|err|^2 / std^2)`` with ``std`` in [m]."""
    command = _command(env, command_name)
    error = (command.anchor_pos_w - command.robot_anchor_pos_w).square().sum(dim=-1)
    return torch.exp(-error / std**2) * command.is_imitation


def motion_anchor_ori_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Track the reference anchor orientation, including yaw, with ``std`` in [rad]."""
    command = _command(env, command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w).square()
    return torch.exp(-error / std**2) * command.is_imitation


def motion_anchor_lin_vel_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Track the reference anchor linear velocity in the world frame, with ``std`` in [m/s]."""
    command = _command(env, command_name)
    error = (command.anchor_lin_vel_w - command.robot_anchor_lin_vel_w).square().sum(dim=-1)
    return torch.exp(-error / std**2) * command.is_imitation


def motion_anchor_ang_vel_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Track the reference anchor angular velocity in the world frame, with ``std`` in [rad/s]."""
    command = _command(env, command_name)
    error = (command.anchor_ang_vel_w - command.robot_anchor_ang_vel_w).square().sum(dim=-1)
    return torch.exp(-error / std**2) * command.is_imitation


def motion_body_pos_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Track reference body positions moved under the robot anchor's XY and yaw, with ``std`` in [m].

    The reference keeps its own heights, so a crouched pose or a low foot still lowers the reward.
    """
    command = _command(env, command_name)
    ids = _body_indices(command, body_names)
    error = (command.body_pos_relative_w[:, ids] - command.robot_body_pos_w[:, ids]).square().sum(dim=-1)
    return torch.exp(-error.mean(dim=-1) / std**2) * command.is_imitation


def motion_body_ori_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Track reference body orientations rotated by the robot anchor's yaw, with ``std`` in [rad]."""
    command = _command(env, command_name)
    ids = _body_indices(command, body_names)
    error = quat_error_magnitude(command.body_quat_relative_w[:, ids], command.robot_body_quat_w[:, ids]).square()
    return torch.exp(-error.mean(dim=-1) / std**2) * command.is_imitation


def motion_body_lin_vel_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Track reference body linear velocities in the world frame, with ``std`` in [m/s]."""
    command = _command(env, command_name)
    ids = _body_indices(command, body_names)
    error = (command.body_lin_vel_w[:, ids] - command.robot_body_lin_vel_w[:, ids]).square().sum(dim=-1)
    return torch.exp(-error.mean(dim=-1) / std**2) * command.is_imitation


"""
Task episodes: reach the destination. Each term is zero in imitation episodes.
"""


def goal_position_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Reward root proximity to the destination: ``exp(-|err|^2 / std^2)`` with ``std`` in [m]."""
    command = _command(env, command_name)
    return torch.exp(-command.goal_error.square().sum(dim=-1) / std**2) * ~command.is_imitation


def goal_heading_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Reward the destination yaw once the root is within the destination position tolerances."""
    command = _command(env, command_name)
    reward = torch.exp(-command.heading_error.square() / std**2)
    return reward * command.position_reached * ~command.is_imitation


def goal_settle(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Squared root linear [m/s] and angular [rad/s] speed at the destination, to penalize."""
    command = _command(env, command_name)
    data = command.robot.data
    speed = data.root_lin_vel_w.torch.square().sum(dim=-1) + data.root_ang_vel_w.torch.square().sum(dim=-1)
    return speed * command.position_reached * ~command.is_imitation


def goal_joint_deviation_l1(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """L1 deviation [rad] from the default joint pose while standing at the destination, to penalize."""
    command = _command(env, command_name)
    standing = command.position_reached & command.feet_supported & ~command.is_imitation
    return mdp.joint_deviation_l1(env, asset_cfg) * standing


class TaskMotionMatchingStyle(MotionMatchingStyle):
    """:class:`MotionMatchingStyle` of the perception motion-style task, paid only in task episodes.

    The robot's joint state is matched to its nearest reference climb frame, so the term needs no reference
    phase or world pose and applies on every stair height. ``command_name`` names the motion command whose
    episode type gates the reward; the other parameters are those of :class:`MotionMatchingStyle`.
    """

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        motion_files: list[str],
        match_threshold: float,
        joint_groups: dict[str, dict],
        joint_vel_weight: float,
        min_joint_speed: float,
        frame_limit: int | None = None,
        include_mirrored: bool = True,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Return the motion-matching style score in [0, 1] in task episodes and zero in imitation episodes."""
        reward = super().__call__(
            env,
            motion_files,
            match_threshold,
            joint_groups,
            joint_vel_weight,
            min_joint_speed,
            frame_limit,
            include_mirrored,
            asset_cfg,
        )
        return reward * ~_command(env, command_name).is_imitation
