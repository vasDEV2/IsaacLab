# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion-matching style reward: look like the nearest reference frame once the robot is already close to one."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp.symmetry import mirror_g1_joints

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionMatchingStyle(ManagerTermBase):
    """Reward looking like the reference frame the robot's joint state is currently closest to.

    The reference library holds every frame of every clip (and, optionally, its left-right mirror) whose
    RMS joint speed is at least ``min_joint_speed``. Each step:

    1. Find the library frame nearest to the robot, by
       ``sqrt(mean(dq^2) + joint_vel_weight^2 * mean(dqd^2))``.
    2. If that distance is within ``match_threshold``, reward resembling that same frame. Each entry of
       ``joint_groups`` (for example legs and upper body) has its own RMS joint-position error ``e_g`` [rad] to the
       frame, its own ``inner_threshold`` and ``reward_std``, and the reward is
       ``exp(-sum_g (max(e_g - inner_threshold_g, 0) / reward_std_g)^2)``: ``1`` while every group is inside its
       inner margin. Otherwise the reward is zero.

    The reward is constant inside the inner margins, so the gait only has to look similar to the clip, not
    match it exactly; a wide leg margin lets the legs adapt to the stairs while a tight upper-body margin keeps
    the torso and arms close to the clip. Only the robot's own joint state is used: no world pose, root pose, or
    terrain, so the term applies on any staircase. Velocities enter the match only, to tell apart poses that look
    alike but move differently; set ``joint_vel_weight`` to zero to match on joint positions alone. The term keeps
    no state between steps; :attr:`nearest_clip`, :attr:`nearest_frame`, and :attr:`group_error` record the latest
    match for inspection.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        params = cfg.params
        if params["match_threshold"] <= 0.0:
            raise ValueError("match_threshold must be positive.")
        for group, group_cfg in params["joint_groups"].items():
            if group_cfg["inner_threshold"] < 0.0 or group_cfg["reward_std"] <= 0.0:
                raise ValueError(f"Joint group '{group}' needs inner_threshold >= 0 and reward_std > 0.")
        if params["joint_vel_weight"] < 0.0:
            raise ValueError("joint_vel_weight must be non-negative.")
        asset_cfg: SceneEntityCfg = params.get("asset_cfg", SceneEntityCfg("robot"))
        self.robot: Articulation = env.scene[asset_cfg.name]

        joint_names: list[str] | None = None
        seen, positions, velocities, clip_ids, frame_ids = [], [], [], [], []
        self.clip_names: list[str] = []
        """Library clip names; :attr:`nearest_clip` indexes this list."""
        for path in params["motion_files"]:
            with np.load(path, allow_pickle=False) as data:
                names = data["joint_names"].tolist()
                joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
                joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
            if joint_names is None:
                joint_names = names
            elif names != joint_names:
                raise ValueError(f"Joint names in '{path}' differ from the first motion file.")
            frame_limit = params.get("frame_limit")
            if frame_limit is not None:
                joint_pos, joint_vel = joint_pos[:frame_limit], joint_vel[:frame_limit]
            clips = [(Path(path).stem, joint_pos, joint_vel)]
            if params.get("include_mirrored", True):
                pos_t, vel_t = torch.from_numpy(joint_pos), torch.from_numpy(joint_vel)
                mirrored = (mirror_g1_joints(pos_t, names).numpy(), mirror_g1_joints(vel_t, names).numpy())
                clips.append((f"{Path(path).stem}_mirrored", *mirrored))
            for clip_name, clip_pos, clip_vel in clips:
                # clips that differ only by world yaw share their joint trajectories; keep one copy
                if any(s.shape == clip_pos.shape and np.allclose(s, clip_pos, atol=1e-4) for s in seen):
                    continue
                seen.append(clip_pos)
                keep = np.sqrt((clip_vel**2).mean(axis=-1)) >= params["min_joint_speed"]
                positions.append(clip_pos[keep])
                velocities.append(clip_vel[keep])
                clip_ids.append(np.full(int(keep.sum()), len(self.clip_names)))
                frame_ids.append(np.nonzero(keep)[0])
                self.clip_names.append(clip_name)
        if sum(len(p) for p in positions) == 0:
            raise ValueError("No reference frame passed the min_joint_speed filter.")

        self.joint_ids = self.robot.find_joints(joint_names, preserve_order=True)[0]
        self.library_pos = torch.tensor(np.concatenate(positions), device=self.device)
        self.library_vel = torch.tensor(np.concatenate(velocities), device=self.device)
        self.library_clip = torch.tensor(np.concatenate(clip_ids), device=self.device)
        self.library_frame = torch.tensor(np.concatenate(frame_ids), device=self.device)
        self.num_joints = self.library_pos.shape[-1]
        # each clip joint belongs to exactly one group; ``group_masks`` is (num_groups, num_joints)
        groups = params["joint_groups"]
        masks = [
            [re.fullmatch(cfg_g["joint_names"], name) is not None for name in joint_names] for cfg_g in groups.values()
        ]
        counts = np.sum(masks, axis=0)
        if (counts != 1).any():
            bad = [name for name, count in zip(joint_names, counts) if count != 1]
            raise ValueError(f"Each clip joint must match exactly one joint group; check {bad}.")
        self.group_names = list(groups)
        self.group_masks = torch.tensor(masks, dtype=torch.float32, device=self.device)
        self.group_inner = torch.tensor([g["inner_threshold"] for g in groups.values()], device=self.device)
        self.group_std = torch.tensor([g["reward_std"] for g in groups.values()], device=self.device)
        self.group_error = torch.zeros(self.num_envs, len(groups), device=self.device)
        """Latest RMS joint-position error [rad] per joint group to the matched frame."""
        self.vel_weight = float(params["joint_vel_weight"])
        self.library_features = self._features(self.library_pos, self.library_vel)

        self.matched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.match_distance = torch.zeros(self.num_envs, device=self.device)
        self.nearest_clip = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.nearest_frame = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _features(self, joint_pos: torch.Tensor, joint_vel: torch.Tensor) -> torch.Tensor:
        """Features whose Euclidean distance is the combined RMS match distance."""
        return torch.cat((joint_pos, joint_vel * self.vel_weight), dim=-1) / self.num_joints**0.5

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        motion_files: list[str],
        match_threshold: float,
        joint_groups: dict[str, dict],
        joint_vel_weight: float,
        min_joint_speed: float,
        frame_limit: int | None = None,
        include_mirrored: bool = True,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Match the current joint state to its nearest reference frame and reward resembling that frame."""
        joint_pos = self.robot.data.joint_pos.torch[:, self.joint_ids]
        joint_vel = self.robot.data.joint_vel.torch[:, self.joint_ids]

        distance, nearest = torch.cdist(self._features(joint_pos, joint_vel), self.library_features).min(dim=-1)
        self.matched[:] = distance < match_threshold
        self.match_distance[:] = distance
        self.nearest_clip[:] = self.library_clip[nearest]
        self.nearest_frame[:] = self.library_frame[nearest]

        # saturating per joint group: constant inside each group's inner margin, so similar is as good as exact
        squared = (joint_pos - self.library_pos[nearest]).square()
        self.group_error[:] = (squared @ self.group_masks.T / self.group_masks.sum(dim=-1)).sqrt()
        excess = (self.group_error - self.group_inner).clamp_min(0.0) / self.group_std
        return torch.exp(-excess.square().sum(dim=-1)) * self.matched
