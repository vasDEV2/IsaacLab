# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Compatibility imports; edit experiment defaults in env_cfg/experiment_cfg.py."""

from .experiment_cfg import (
    SOFT_CONTACT_THRESHOLD,
    G1RewardsCfg,
)

__all__ = ["SOFT_CONTACT_THRESHOLD", "G1RewardsCfg"]
