# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp as g1_mdp
import isaaclab_tasks.core.velocity.mdp as mdp


@configclass
class G1TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    base_too_low = TerminationTermCfg(
        func=g1_mdp.root_height_below_minimum_adaptive,
        params={
            # "minimum_height": 0.5,
            "minimum_height": 0.2,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    bad_orientation = TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": 0.8})
    terrain_out_of_bounds = TerminationTermCfg(
        func=mdp.terrain_out_of_bounds,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "distance_buffer": 1.0,
        },
    )
