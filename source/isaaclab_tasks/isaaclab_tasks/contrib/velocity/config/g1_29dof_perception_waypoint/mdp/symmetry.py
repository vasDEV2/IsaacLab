# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reflection of yaw-frame positions: the waypoint command and the root position on the tile."""

import torch


def mirror_waypoint(data: torch.Tensor) -> torch.Tensor:
    """Reflect an ``(x, y, z)`` yaw-frame position across the root XZ plane."""
    return data * data.new_tensor((1.0, -1.0, 1.0))
