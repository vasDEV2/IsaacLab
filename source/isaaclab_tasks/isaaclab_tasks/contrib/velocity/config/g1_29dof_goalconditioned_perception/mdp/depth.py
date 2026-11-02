# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-local metric depth processing inspired by PHP's low-resolution camera input."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class depth_image(ManagerTermBase):
    """Crop optical-axis depth and return a normalized ``(B, 1, H, W)`` image.

    Background, nonpositive and nonfinite measurements map to the far distance.
    Positive measurements are clipped to the configured metric range. Corruption
    approximates depth noise, calibration bias, missing patches and edge dropout;
    it does not model stereo matching. The observation manager owns latency.
    Repeated reads of one camera frame reuse the same corrupted image.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._sensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self._last_step = -1
        self._frame = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        self._image = torch.zeros((env.num_envs, 1, *cfg.params["resize"]), device=env.device)
        self._bias = torch.zeros((env.num_envs, 1, 1, 1), device=env.device)
        self.reset()

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        ids = slice(None) if env_ids is None else env_ids
        self._frame[ids] = -1
        self._bias[ids] = torch.randn_like(self._bias[ids]) * 0.03
        self._last_step = -1

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        resize: tuple[int, int],
        crop: tuple[int, int, int, int] = (2, 0, 4, 4),
        depth_range: tuple[float, float] = (0.3, 3.0),
        corrupt: bool = True,
    ) -> torch.Tensor:
        if self._last_step == env.common_step_counter:
            return self._image
        lo, hi = depth_range
        if not 0.0 < lo < hi:
            raise ValueError("Depth range must satisfy 0 < near < far.")
        # Reading data updates the camera before we inspect its frame counter.
        raw = self._sensor.data.output["depth"].torch
        frame = self._sensor.frame.torch
        top, bottom, left, right = crop
        if min(crop) < 0 or top + bottom >= raw.shape[1] or left + right >= raw.shape[2]:
            raise ValueError(f"Invalid depth crop {crop} for image shape {tuple(raw.shape)}.")
        image = torch.where(torch.isfinite(raw) & (raw > 0.0), raw, hi).clamp(lo, hi)
        image = image[:, top : -bottom if bottom else None, left : -right if right else None]
        image = F.interpolate(
            image.permute(0, 3, 1, 2), size=resize, mode="bicubic", align_corners=False, antialias=True
        ).clamp(lo, hi)
        if corrupt:
            # Drop pixels near depth discontinuities and spatially correlated patches.
            local_max = F.max_pool2d(image, 3, stride=1, padding=1)
            local_min = -F.max_pool2d(-image, 3, stride=1, padding=1)
            edge_drop = ((local_max - local_min) > 0.2) & (torch.rand_like(image) < 0.1)
            patches = (
                F.interpolate(torch.rand((env.num_envs, 1, 8, 12), device=image.device), size=resize, mode="nearest")
                < 0.02
            )
            image = image + torch.randn_like(image) * 0.03 + self._bias
            image = torch.where(edge_drop | patches, hi, image).clamp(lo, hi)
        image = (image - lo) / (hi - lo) - 0.5
        self._image = torch.where((frame != self._frame)[:, None, None, None], image, self._image)
        self._frame.copy_(frame)
        self._last_step = env.common_step_counter
        return self._image
