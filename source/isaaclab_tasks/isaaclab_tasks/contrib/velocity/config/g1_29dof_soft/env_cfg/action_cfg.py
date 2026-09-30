# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Compatibility imports; edit experiment defaults in env_cfg/experiment_cfg.py."""

from .experiment_cfg import (
    ACTIVE_JOINT,
    COLLIDER_SHAPE,
    SOFT_CONTACT_THRESHOLD,
    G1ActionsCfg,
    collider_cfg,
    contact_model,
)

__all__ = ["COLLIDER_SHAPE", "SOFT_CONTACT_THRESHOLD", "contact_model", "collider_cfg", "ACTIVE_JOINT", "G1ActionsCfg"]
