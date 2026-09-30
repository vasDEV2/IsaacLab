# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp as g1_mdp
import isaaclab_tasks.contrib.velocity.config.g1_29dof_soft.mdp.perception_env as g1_soft_mdp
import isaaclab_tasks.core.velocity.mdp as mdp

SOFT_CONTACT_THRESHOLD = 40.0
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
LEG_JOINT = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    # "left_ankle_pitch_joint",
    # "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    # "right_ankle_pitch_joint",
    # "right_ankle_roll_joint",
]


@configclass
class PolicyCfg(ObsGroup):
    """Observations for policy group."""

    # observation terms (order preserved)
    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel,
        noise=Unoise(n_min=-0.2, n_max=0.2),
        scale=0.25,
    )
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        noise=Unoise(n_min=-0.01, n_max=0.01),
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=ACTIVE_JOINT,
                preserve_order=True,
            ),
        },
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        noise=Unoise(n_min=-1.5, n_max=1.5),
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=ACTIVE_JOINT,
                preserve_order=True,
            ),
        },
        scale=0.05,
    )
    actions = ObsTerm(func=mdp.last_action)
    lidar = ObsTerm(
            func=g1_soft_mdp.mid360_lidar_ranges,
            params={"sensor_cfg": SceneEntityCfg("mid360_lidar")},
            clip=(0.0, 40.0),  # matches sensor max_distance
        )

    # rgb = ObsTerm(
    #         func=g1_soft_mdp.d435_rgb,
    #         params={"sensor_cfg": SceneEntityCfg("d435_camera")},
    #     )
    # depth = ObsTerm(
    #     func=g1_soft_mdp.d435_depth,
    #     params={"sensor_cfg": SceneEntityCfg("d435_camera"), "clip_max": 10.0},
    # )
    
    height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        noise=Unoise(n_min=-0.1, n_max=0.1),
        clip=(-1.0, 1.0),
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        self.history_length = 1


@configclass
class CriticCfg(ObsGroup):
    """Observations for policy group."""

    # observation terms (order preserved)
    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel,
        scale=0.25,
    )
    base_quat = ObsTerm(
        func=mdp.root_quat_w,
        params={"make_quat_unique": True},
    )
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=ACTIVE_JOINT,
                preserve_order=True,
            ),
        },
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=ACTIVE_JOINT,
                preserve_order=True,
            ),
        },
        scale=0.05,
    )
    actions = ObsTerm(func=mdp.last_action)
    
    height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        clip=(-1.0, 1.0),
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 1


@configclass
class PolicyHistoryCfg(PolicyCfg):
    def __post_init__(self):
        self.history_length = 10


@configclass
class CriticHistoryCfg(CriticCfg):
    def __post_init__(self):
        self.history_length = 10


@configclass
class PrivilegedObsCfg(ObsGroup):
    """Observations for policy group."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    foot_height = ObsTerm(
        func=g1_mdp.foot_height,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link")},
    )

    foot_contact = ObsTerm(
        func=g1_soft_mdp.foot_contact_hybrid,
        params={
            "rigid_contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "soft_contact_sensor_name": "physics_callback",
            "rigid_force_threshold": 5.0,
            "soft_force_threshold": SOFT_CONTACT_THRESHOLD,
        },
    )
    foot_contact_force = ObsTerm(
        func=g1_soft_mdp.foot_contact_forces_hybrid,
        params={
            "rigid_contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "soft_contact_sensor_name": "physics_callback",
            "rigid_force_filter_threshold": 5.0,
            "soft_force_filter_threshold": SOFT_CONTACT_THRESHOLD,
        },
    )
    foot_air_time = ObsTerm(
        func=g1_soft_mdp.foot_air_time_hybrid,
        params={
            "rigid_contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "soft_contact_sensor_name": "physics_callback",
        },
    )

    terrain_material_parameters = ObsTerm(
        # func=mdp.terrain_material_parameters_all_hybrid,
        func=g1_soft_mdp.terrain_material_parameters_hybrid,
        params={
            "rigid_contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "soft_contact_sensor_name": "physics_callback",
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 1


@configclass
class PrivilegedHistoryCfg(PrivilegedObsCfg):
    def __post_init__(self):
        self.history_length = 10


"""
obs for logging
"""


@configclass
class LoggingObsCfg(ObsGroup):
    """Observations for policy group."""

    base_pos = ObsTerm(func=mdp.root_pos_w)
    base_quat = ObsTerm(func=mdp.root_quat_w)
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    commands = ObsTerm(
        func=mdp.generated_commands,
        params={"command_name": "base_velocity"},
    )
    contact_forces = ObsTerm(
        func=g1_soft_mdp.foot_contact_forces_raw_hybrid,
        params={
            "rigid_contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "soft_contact_sensor_name": "physics_callback",
            "rigid_force_filter_threshold": 5.0,
            "soft_force_filter_threshold": 40.0,
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True

 
@configclass
class G1ObservationsCfg:
    """Observation specifications for the MDP."""

    # observation groups
    policy: PolicyHistoryCfg = PolicyHistoryCfg()
    critic: CriticHistoryCfg = CriticHistoryCfg()
    privileged: PrivilegedHistoryCfg = PrivilegedHistoryCfg()

    # logging: LoggingObsCfg = LoggingObsCfg()
