# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resets for :class:`~.command.PyramidMimicSplitCommand`: walk episodes at rest, imitation episodes on the clip."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils import math as math_utils

from isaaclab_tasks.contrib.velocity.config.g1_29dof_perception_waypoint.mdp import reset_root_at_waypoint_start

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .command import PyramidMimicSplitCommand


def reset_root_walk_or_reference(env: ManagerBasedRLEnv, env_ids: torch.Tensor, command_name: str) -> None:
    """Draw the coming episode; walk episodes start at rest at their spawn, imitation episodes on the reference root."""
    command: PyramidMimicSplitCommand = env.command_manager.get_term(command_name)
    env_ids = command._as_env_ids(env_ids)
    command.sample_episode(env_ids)
    imitate = command.is_imitation[env_ids]
    walk_ids, imitation_ids = env_ids[~imitate], env_ids[imitate]
    if len(walk_ids) > 0:
        reset_root_at_waypoint_start(env, walk_ids, command_name)
    if len(imitation_ids) > 0:
        state = command.reference_root_state(imitation_ids)
        command.robot.write_root_pose_to_sim_index(root_pose=state[:, :7], env_ids=imitation_ids)
        command.robot.write_root_velocity_to_sim_index(root_velocity=state[:, 7:], env_ids=imitation_ids)


def reset_joints_to_reference(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str,
    position_range: tuple[float, float] = (-0.1, 0.1),
) -> None:
    """Write the reference joints of imitation episodes; walk episodes keep the preceding joint reset.

    ``position_range`` [rad] adds uniform noise to every joint before clipping to the soft limits.
    """
    command: PyramidMimicSplitCommand = env.command_manager.get_term(command_name)
    env_ids = command._as_env_ids(env_ids)
    command.sample_episode(env_ids)
    env_ids = env_ids[command.is_imitation[env_ids]]
    if len(env_ids) == 0:
        return
    robot = command.robot
    joint_pos, joint_vel = command.reference_joint_state(env_ids)
    joint_pos = joint_pos + math_utils.sample_uniform(*position_range, joint_pos.shape, env.device)
    limits = robot.data.soft_joint_pos_limits.torch[env_ids]
    joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])
    robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
