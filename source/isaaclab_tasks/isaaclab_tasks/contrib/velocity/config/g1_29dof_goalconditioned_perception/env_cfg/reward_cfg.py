# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp as g1_mdp
import isaaclab_tasks.contrib.velocity.config.vel_mdp as vel_mdp
import isaaclab_tasks.core.velocity.mdp as mdp

from ..mdp.motion import motion_joint_pos_exp
from ..mdp.stair_rewards import feet_stumble
from .observation_cfg import ACTIVE_JOINTS


@configclass
class G1RewardsCfg:
    """Reward terms for the MDP."""

    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=4.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.5)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.5)}
    )
    track_heading = RewTerm(
        func=g1_mdp.track_heading_world_exp,
        weight=4.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "base_velocity", "std": math.sqrt(0.5)},
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!.*ankle.*).*"), "threshold": 1.0},
    )

    action_rate_l2_lower_body = RewTerm(func=g1_mdp.action_rate_l2, weight=-0.01, params={"joint_idx": list(range(12))})
    action_rate_l2_upper_body = RewTerm(
        func=g1_mdp.action_rate_l2, weight=-0.05, params={"joint_idx": list(range(12, 29))}
    )
    energy = RewTerm(func=g1_mdp.energy, weight=-0.001)
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.0002)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-07)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot")})
    joint_deviation = RewTerm(
        func=g1_mdp.variable_posture_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "command_name": "base_velocity",
            "weight_standing": {
                ".*hip_pitch.*": 0.02,
                ".*hip_roll.*": 0.3,
                ".*hip_yaw.*": 0.08,
                ".*knee.*": 0.02,
                ".*ankle_pitch.*": 0.02,
                ".*ankle_roll.*": 0.02,
                ".*(waist|shoulder|elbow|wrist).*": 1.0,
            },
            "weight_walking": {
                ".*hip_pitch.*": 0.02,
                ".*hip_roll.*": 0.3,
                ".*hip_yaw.*": 0.08,
                ".*knee.*": 0.02,
                ".*ankle_pitch.*": 0.02,
                ".*ankle_roll.*": 0.02,
                ".*waist_yaw.*": 0.15,
                ".*waist_roll.*": 2.0,
                ".*waist_pitch.*": 2.0,
                ".*shoulder_pitch.*": 0.5,
                ".*elbow.*": 0.25,
                ".*shoulder_roll.*": 0.4,
                ".*shoulder_yaw.*": 0.35,
                ".*wrist.*": 0.5,
            },
            "weight_running": {
                ".*hip_pitch.*": 0.005,
                ".*hip_roll.*": 0.3,
                ".*hip_yaw.*": 0.08,
                ".*knee.*": 0.005,
                ".*ankle_pitch.*": 0.01,
                ".*ankle_roll.*": 0.01,
                ".*waist_yaw.*": 0.15,
                ".*waist_roll.*": 2.0,
                ".*waist_pitch.*": 2.0,
                ".*shoulder_pitch.*": 0.5,
                ".*elbow.*": 0.25,
                ".*shoulder_roll.*": 0.4,
                ".*shoulder_yaw.*": 0.35,
                ".*wrist.*": 0.5,
            },
            "walking_threshold": 0.05,
            "running_threshold": 2.0,
        },
    )

    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-10.0,
        params={"target_height": 0.75, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-10.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    feet_roll = RewTerm(
        func=g1_mdp.reward_feet_roll,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"], preserve_order=True)},
    )
    feet_roll_diff = RewTerm(
        func=g1_mdp.reward_feet_roll_diff,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"], preserve_order=True)},
    )
    stance_foot_angle_penalty = RewTerm(
        func=g1_mdp.stance_foot_angle_penalty,
        weight=-8.0,
        params={
            "height_sensor_cfg": SceneEntityCfg("height_scanner"),
            "angle_tolerance": 0.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*ankle_roll.*"], preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"], preserve_order=True),
        },
    )

    feet_air_time = RewTerm(
        func=g1_mdp.feet_air_time_positive_biped,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "threshold": 0.5,
            "velocity_threshold": 0.05,
        },
    )
    feet_air_time_variance = RewTerm(
        func=mdp.feet_air_time_variance,
        weight=-0.7,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "max_time": 0.5,
        },
    )
    feet_stumble = RewTerm(
        func=feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )
    break_contact = RewTerm(
        func=g1_mdp.break_contact_penalty,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "command_threshold": 0.05,
            "recovery_linear_velocity": 0.2,
            "recovery_angular_velocity": 0.5,
            "recovery_tilt": 0.2,
        },
    )
    no_fly = RewTerm(
        func=g1_mdp.fly,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "threshold": 5.0,
            "command_name": "base_velocity",
            "velocity_threshold": 1.0,
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    foot_distance = RewTerm(
        func=g1_mdp.reward_foot_lateral_symmetry,
        weight=-2.0,
        params={
            "ref_dist": 0.2,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link", preserve_order=True),
        },
    )
    contact_impulse = RewTerm(
        func=g1_mdp.reward_soft_landing,
        weight=-0.005,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "command_name": "base_velocity",
            "command_threshold": 0.05,
        },
    )

    foot_clearance = RewTerm(
        func=vel_mdp.foot_clearance_reward,
        weight=5.0,
        params={
            "target_height": 0.1,
            "std": 0.05,
            "tanh_mult": 2.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "standing_position_foot_z": 0.03539,
            "height_sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )
    # root_height_world (world Z held at 0.75 m) is removed: the clip climbs the pelvis to ~1.96 m.

    motion_joint_pos = RewTerm(
        func=motion_joint_pos_exp,
        weight=3.0,
        params={
            "command_name": "motion",
            "std": 0.25,
            "asset_cfg": SceneEntityCfg("robot", joint_names=ACTIVE_JOINTS, preserve_order=True),
        },
    )
