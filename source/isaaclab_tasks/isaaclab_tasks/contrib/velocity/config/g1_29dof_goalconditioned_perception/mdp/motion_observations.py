# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Privileged critic observations of the episode type, the reference, and the robot's body pose.

Reference observations are zeroed in task episodes, where no reference is tracked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .motion import GoalMotionCommand


def _command(env: ManagerBasedRLEnv, command_name: str) -> GoalMotionCommand:
    return env.command_manager.get_term(command_name)


def _rotation_6d(quat: torch.Tensor) -> torch.Tensor:
    """First two rotation-matrix columns, row-flattened. Shape is (..., 6)."""
    return matrix_from_quat(quat)[..., :2].flatten(-2)


def imitation_mode(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """One in imitation episodes and zero in task episodes. Shape is (N, 1)."""
    return _command(env, command_name).is_imitation.float().unsqueeze(-1)


def reference_joint_pos(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reference joint positions [rad] in ``asset_cfg`` joint order."""
    command = _command(env, command_name)
    return command.joint_pos[:, asset_cfg.joint_ids] * command.is_imitation.unsqueeze(-1)


def reference_phase(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Clip progress in [0, 1]. Shape is (N, 1)."""
    command = _command(env, command_name)
    return (command.phase * command.is_imitation).unsqueeze(-1)


def reference_anchor_pos_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Reference anchor position [m] in the robot anchor frame. Shape is (N, 3)."""
    command = _command(env, command_name)
    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w, command.robot_anchor_quat_w, command.anchor_pos_w, command.anchor_quat_w
    )
    return pos * command.is_imitation.unsqueeze(-1)


def reference_anchor_ori_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Reference anchor orientation in the robot anchor frame, as a 6D rotation. Shape is (N, 6)."""
    command = _command(env, command_name)
    _, quat = subtract_frame_transforms(
        command.robot_anchor_pos_w, command.robot_anchor_quat_w, command.anchor_pos_w, command.anchor_quat_w
    )
    return _rotation_6d(quat) * command.is_imitation.unsqueeze(-1)


def robot_body_pos_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Tracked body positions [m] in the robot anchor frame. Shape is (N, 3 * bodies)."""
    command = _command(env, command_name)
    num_bodies = len(command.cfg.body_names)
    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None].expand(-1, num_bodies, -1),
        command.robot_anchor_quat_w[:, None].expand(-1, num_bodies, -1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    return pos.flatten(1)


def robot_body_ori_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Tracked body orientations in the robot anchor frame, as 6D rotations. Shape is (N, 6 * bodies)."""
    command = _command(env, command_name)
    num_bodies = len(command.cfg.body_names)
    _, quat = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None].expand(-1, num_bodies, -1),
        command.robot_anchor_quat_w[:, None].expand(-1, num_bodies, -1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    return _rotation_6d(quat).flatten(1)
