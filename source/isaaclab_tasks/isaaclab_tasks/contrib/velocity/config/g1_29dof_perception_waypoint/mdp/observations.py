# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot position on its stair tile."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def root_pos_tile_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root position [m] in the root yaw frame: XY from the tile center, Z above the flat ground around the stairs.

    The yaw frame keeps the term mirrorable as ``(x, -y, z)``. World Z is used for height because the tile
    center sits on a pyramid top or pit bottom, while the ground around every tile is at Z = 0.
    """
    data = env.scene[asset_cfg.name].data
    position = data.root_pos_w.torch.clone()
    position[:, :2] -= env.scene.env_origins[:, :2]
    return math_utils.quat_apply_inverse(math_utils.yaw_quat(data.root_quat_w.torch), position)
