# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab_tasks.contrib.soft_contact import (
    BoxColliderCfg,
    PhysicsCallbackActionCfg,
    PlaneColliderCfg,
    SphereColliderCfg,
)

"""
collider geometry
"""

COLLIDER_SHAPE = "box"  # "plane", "sphere"

if COLLIDER_SHAPE == "plane":
    collider_cfg = PlaneColliderCfg(
        contact_edge_x=(-0.065, 0.141),
        contact_edge_y=(-0.0368, 0.0368),
        contact_edge_z=(-0.03539, 0.0),
        resolution=(5, 5),
    )
elif COLLIDER_SHAPE == "box":
    collider_cfg = BoxColliderCfg(
        contact_edge_x=(-0.065, 0.141),
        contact_edge_y=(-0.0368, 0.0368),
        contact_edge_z=(-0.03539, 0.0),
        resolution=(5, 5),
    )
elif COLLIDER_SHAPE == "sphere":
    collider_cfg = SphereColliderCfg(
        radius=0.05,
        center=(0.0, 0.0, 0.0),
        resolution=(8, 8),
    )

SOFT_CONTACT_THRESHOLD = 40.0

contact_model = "3D-warp"
# contact_model = "2D-warp"
# contact_model = "cone-drft"
# contact_model = "cone-drft-multipoint"


ACTIVE_JOINT = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


@configclass
class G1ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ACTIVE_JOINT,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )

    """
    Contact solver.
    """
    physics_callback = PhysicsCallbackActionCfg(
        asset_name="robot",
        body_names=[".*_ankle_roll_link"],
        backend=contact_model,
        intruder_geometry_cfg=collider_cfg,
        enable_ema_filter=False,
        contact_threshold=SOFT_CONTACT_THRESHOLD,
        debug_vis=False,
        contact_data_history_length=10,  # logging interval = 0.005*10 = 0.05s, 10 history -> 0.5s
        history_logging_decimation=10,
        contact_vis_force_threshold=SOFT_CONTACT_THRESHOLD,
    )
