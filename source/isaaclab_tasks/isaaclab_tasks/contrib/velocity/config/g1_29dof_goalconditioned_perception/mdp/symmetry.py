# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reflection of the AME task's structured XYZ elevation map and its destination command."""

import torch


def mirror_elevation_map(data: torch.Tensor) -> torch.Tensor:
    """Reflect ``(..., nx, ny, 3)`` points across the torso yaw-frame XZ plane.

    Reverse lateral grid cells and negate their Y coordinates. X and terrain Z
    retain their values; history/batch dimensions and missing-height markers are
    preserved. A spatial flip alone would associate points with the wrong XY cells.
    """
    if data.ndim < 3 or data.shape[-1] != 3:
        raise ValueError("Elevation-map reflection requires (..., nx, ny, 3) XYZ data.")
    return data.flip(-2) * data.new_tensor((1.0, -1.0, 1.0))


def mirror_goal(data: torch.Tensor) -> torch.Tensor:
    """Reflect the ``(x, y, z, sin, cos)`` destination command across the yaw-frame XZ plane.

    Negate the lateral offset and the sine of the yaw error.
    """
    if data.shape[-1] != 5:
        raise ValueError(f"Expected a five-component destination command, got {data.shape}.")
    return data * data.new_tensor((1.0, -1.0, 1.0, -1.0, 1.0))
