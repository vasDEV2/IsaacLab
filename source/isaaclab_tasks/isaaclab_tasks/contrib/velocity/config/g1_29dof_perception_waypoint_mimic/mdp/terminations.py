# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The goal-conditioned task's reference terminations, with thresholds that tighten as training progresses.

Each threshold is ``threshold_start`` when imitation begins (the command's ``imitation_start_step``) and falls
linearly to ``threshold`` over the next ``decay_steps`` control steps of training, then stays there. The schedule
follows the environment's step counter, which restarts at zero on every launch. All terms are false outside
imitation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .command import PyramidMimicCommand


def _threshold(
    env: ManagerBasedRLEnv, command: PyramidMimicCommand, threshold_start: float, threshold: float, decay_steps: int
) -> float:
    """Threshold at the current training step, shared by all environments."""
    elapsed = env.common_step_counter - command.cfg.imitation_start_step
    progress = min(max(elapsed / max(decay_steps, 1), 0.0), 1.0)
    return threshold_start + (threshold - threshold_start) * progress


def bad_anchor_height_decaying(
    env: ManagerBasedRLEnv, command_name: str, threshold_start: float, threshold: float, decay_steps: int
) -> torch.Tensor:
    """Anchor height error [m] from the reference above the current threshold."""
    command: PyramidMimicCommand = env.command_manager.get_term(command_name)
    error = (command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2]).abs()
    return (error > _threshold(env, command, threshold_start, threshold, decay_steps)) & command.is_imitation


def bad_anchor_tilt_decaying(
    env: ManagerBasedRLEnv, command_name: str, threshold_start: float, threshold: float, decay_steps: int
) -> torch.Tensor:
    """Difference of the anchor-frame gravity Z components between reference and robot above the threshold."""
    command: PyramidMimicCommand = env.command_manager.get_term(command_name)
    gravity = torch.zeros_like(command.anchor_pos_w)
    gravity[:, 2] = -1.0
    reference = quat_apply_inverse(command.anchor_quat_w, gravity)[:, 2]
    robot = quat_apply_inverse(command.robot_anchor_quat_w, gravity)[:, 2]
    limit = _threshold(env, command, threshold_start, threshold, decay_steps)
    return ((reference - robot).abs() > limit) & command.is_imitation


def bad_body_height_decaying(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold_start: float,
    threshold: float,
    decay_steps: int,
    body_names: list[str],
) -> torch.Tensor:
    """Height error [m] of any of ``body_names`` from the anchor-aligned reference above the current threshold."""
    command: PyramidMimicCommand = env.command_manager.get_term(command_name)
    ids = [command.cfg.body_names.index(name) for name in body_names]
    error = (command.body_pos_relative_w[:, ids, 2] - command.robot_body_pos_w[:, ids, 2]).abs()
    limit = _threshold(env, command, threshold_start, threshold, decay_steps)
    return (error > limit).any(dim=-1) & command.is_imitation
