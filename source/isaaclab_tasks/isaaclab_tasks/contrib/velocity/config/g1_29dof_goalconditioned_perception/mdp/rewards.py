# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reference tracking rewards for imitation episodes and destination rewards for task episodes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, quat_error_magnitude, yaw_quat

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


class TaskMotionStyle(ManagerTermBase):
    """Match local limb motion to the reference in task episodes, independent of world pose.

    A hidden reference phase starts at the episode's first frame and advances each control step at the
    candidate rate in ``phase_rates`` (times the recorded rate) that best matches the robot. The match
    sums normalized errors of joint positions, rate-scaled joint velocities, and ``body_names`` positions
    relative to the root in its yaw frame; the reward is ``exp(-error)``. The phase never moves backward
    and clamps at the clip end. Stationary reference frames pay only within the destination tolerances,
    where a stationary final frame becomes the standing target. Neither the phase nor the reference
    features are observed.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.command = _command(env, cfg.params["command_name"])
        motion = self.command.motion
        root = self.command.root_index
        body_ids = [self.command.cfg.body_names.index(name) for name in cfg.params["body_names"]]
        self.robot_body_ids = self.command.body_ids[body_ids]
        root_yaw = yaw_quat(motion.body_quat_w[:, root])
        self.reference_body_pos = quat_apply_inverse(
            root_yaw[:, None], motion.body_pos_w[:, body_ids] - motion.body_pos_w[:, root, None]
        )
        phase_rates = cfg.params["phase_rates"]
        if not phase_rates or any(rate <= 0 for rate in phase_rates):
            raise ValueError("phase_rates must contain positive playback rates.")
        for name in ("joint_pos_std", "joint_vel_std", "body_pos_std"):
            if cfg.params[name] <= 0:
                raise ValueError(f"{name} must be positive.")
        self.phase_rates = torch.tensor(phase_rates, device=self.device)
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self._initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._reward = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Restart the phase from the new episode's frame on its first evaluation."""
        if env_ids is None:
            env_ids = slice(None)
        self._initialized[env_ids] = False
        self._last_step[env_ids] = -1
        self._reward[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        body_names: list[str],
        joint_pos_std: float,
        joint_vel_std: float,
        body_pos_std: float,
        phase_rates: tuple[float, ...],
        stationary_velocity_threshold: float,
    ) -> torch.Tensor:
        """Return the task-episode style score in [0, 1], updated once per control step."""
        command = self.command
        motion = command.motion
        data = command.robot.data
        phase = torch.where(self._initialized, self.phase, command.time_steps.float())
        last_frame = command.motion_lengths - 1
        candidates = torch.minimum(phase[:, None] + self.phase_rates * (env.step_dt * motion.fps), last_frame[:, None])
        final_rows = motion.frame_indices(command.motion_ids, last_frame)
        final_stationary = motion.joint_vel[final_rows].square().mean(dim=-1) < stationary_velocity_threshold**2
        settle = command.position_reached & final_stationary
        candidates = torch.where(settle[:, None], last_frame[:, None].float(), candidates)

        lower = candidates.long()
        upper = torch.minimum(lower + 1, last_frame[:, None])
        alpha = (candidates - lower)[..., None]
        lower = motion.frame_indices(command.motion_ids[:, None], lower)
        upper = motion.frame_indices(command.motion_ids[:, None], upper)
        joint_pos = torch.lerp(motion.joint_pos[lower], motion.joint_pos[upper], alpha)
        joint_vel = torch.lerp(motion.joint_vel[lower], motion.joint_vel[upper], alpha)
        body_pos = torch.lerp(self.reference_body_pos[lower], self.reference_body_pos[upper], alpha[..., None])
        robot_body_pos = quat_apply_inverse(
            yaw_quat(data.root_quat_w.torch)[:, None],
            data.body_pos_w.torch[:, self.robot_body_ids] - data.root_pos_w.torch[:, None],
        )
        # scale the reference velocity consistently with each candidate playback rate
        error = (data.joint_pos.torch[:, None] - joint_pos).square().mean(dim=-1) / joint_pos_std**2
        error += (data.joint_vel.torch[:, None] - joint_vel * self.phase_rates[None, :, None]).square().mean(
            dim=-1
        ) / joint_vel_std**2
        error += (robot_body_pos[:, None] - body_pos).square().sum(dim=-1).mean(dim=-1) / body_pos_std**2
        best = error.argmin(dim=-1, keepdim=True)
        next_phase = candidates.gather(1, best).squeeze(1)
        moving = joint_vel.square().mean(dim=-1).gather(1, best).squeeze(1) >= stationary_velocity_threshold**2
        eligible = moving | command.position_reached
        reward = torch.exp(-error.gather(1, best).squeeze(1)) * eligible * ~command.is_imitation

        update = self._last_step != env.common_step_counter
        self.phase[:] = torch.where(update, next_phase, self.phase)
        self._reward[:] = torch.where(update, reward, self._reward)
        self._last_step[:] = env.common_step_counter
        self._initialized[:] = True
        return self._reward.clone()
