# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Compatibility imports; edit experiment defaults in env_cfg/experiment_cfg.py."""

from .experiment_cfg import (
    ACTIVE_JOINT,
    LEG_JOINT,
    SOFT_CONTACT_THRESHOLD,
    CriticCfg,
    CriticHistoryCfg,
    G1ObservationsCfg,
    LoggingObsCfg,
    PolicyCfg,
    PolicyHistoryCfg,
    PrivilegedHistoryCfg,
    PrivilegedObsCfg,
)

__all__ = [
    "SOFT_CONTACT_THRESHOLD",
    "ACTIVE_JOINT",
    "LEG_JOINT",
    "PolicyCfg",
    "CriticCfg",
    "PolicyHistoryCfg",
    "CriticHistoryCfg",
    "PrivilegedObsCfg",
    "PrivilegedHistoryCfg",
    "LoggingObsCfg",
    "G1ObservationsCfg",
]
