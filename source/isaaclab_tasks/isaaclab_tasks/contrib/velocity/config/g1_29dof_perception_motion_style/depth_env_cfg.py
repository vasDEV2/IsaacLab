# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility aliases: former depth task IDs now use the AME height-map task.

No depth camera or LiDAR is created by these configurations.
"""

from .flat_env_cfg import G1FlatPerceptionEnvCfg, G1FlatPerceptionEnvCfg_PLAY
from .rough_env_cfg import G1RoughPerceptionEnvCfg, G1RoughPerceptionEnvCfg_PLAY

G1RoughDepthEnvCfg = G1RoughPerceptionEnvCfg
G1RoughDepthEnvCfg_PLAY = G1RoughPerceptionEnvCfg_PLAY
G1FlatDepthEnvCfg = G1FlatPerceptionEnvCfg
G1FlatDepthEnvCfg_PLAY = G1FlatPerceptionEnvCfg_PLAY
