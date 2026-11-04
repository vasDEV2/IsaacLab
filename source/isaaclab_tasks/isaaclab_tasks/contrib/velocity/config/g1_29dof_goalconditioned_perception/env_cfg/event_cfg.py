# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.mimic.mdp as mimic_mdp
import isaaclab_tasks.core.velocity.mdp as mdp

from ..mdp.events import reset_goal_joint_state, reset_goal_root_state


@configclass
class G1EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )
    add_joint_default_pos = EventTerm(
        func=mimic_mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
            "joint_action_name": "joint_pos",
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # reset: both events draw the new episode from the motion command before the command manager resets
    reset_joints = EventTerm(
        func=reset_goal_joint_state, mode="reset", params={"command_name": "motion", "position_range": (-0.1, 0.1)}
    )
    reset_base = EventTerm(func=reset_goal_root_state, mode="reset", params={"command_name": "motion"})
