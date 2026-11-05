# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility imports for the former LiDAR/depth agent config entry points.

All registered perception tasks now train AME with elevation-map observations.
"""

from .compositional_ppo_cfg import G1FlatPerceptionPPORunnerCfg, G1RoughPerceptionPPORunnerCfg

G1RoughDepthPPORunnerCfg = G1RoughPerceptionPPORunnerCfg
G1FlatDepthPPORunnerCfg = G1FlatPerceptionPPORunnerCfg

__all__ = [
    "G1FlatDepthPPORunnerCfg",
    "G1FlatPerceptionPPORunnerCfg",
    "G1RoughDepthPPORunnerCfg",
    "G1RoughPerceptionPPORunnerCfg",
]
