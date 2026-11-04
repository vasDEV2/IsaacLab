# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.velocity.mdp as mdp

from ..mdp import rewards as goal_rewards
from .motion_cfg import END_EFFECTOR_BODY_NAMES, FOOT_BODY_NAMES


@configclass
class G1RewardsCfg:
    """Reference tracking in imitation episodes, destination rewards in task episodes, and shared regularization."""

    # -- imitation episodes: track the reference
    motion_global_anchor_pos = RewTerm(
        func=goal_rewards.motion_anchor_pos_exp, weight=2.0, params={"command_name": "motion", "std": 0.3}
    )
    motion_global_anchor_ori = RewTerm(
        func=goal_rewards.motion_anchor_ori_exp, weight=2.0, params={"command_name": "motion", "std": 0.4}
    )
    motion_global_anchor_lin_vel = RewTerm(
        func=goal_rewards.motion_anchor_lin_vel_exp, weight=1.0, params={"command_name": "motion", "std": 1.0}
    )
    motion_global_anchor_ang_vel = RewTerm(
        func=goal_rewards.motion_anchor_ang_vel_exp, weight=1.0, params={"command_name": "motion", "std": 3.14}
    )
    motion_body_pos = RewTerm(
        func=goal_rewards.motion_body_pos_exp, weight=1.0, params={"command_name": "motion", "std": 0.3}
    )
    motion_body_ori = RewTerm(
        func=goal_rewards.motion_body_ori_exp, weight=1.0, params={"command_name": "motion", "std": 0.4}
    )
    motion_feet_lin_vel = RewTerm(
        func=goal_rewards.motion_body_lin_vel_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0, "body_names": FOOT_BODY_NAMES},
    )

    # -- task episodes: reach the destination with the reference's local limb motion
    task_motion_style = RewTerm(
        func=goal_rewards.TaskMotionStyle,
        weight=2.0,
        params={
            "command_name": "motion",
            "body_names": END_EFFECTOR_BODY_NAMES,
            "joint_pos_std": 0.35,
            "joint_vel_std": 1.5,
            "body_pos_std": 0.15,
            "phase_rates": (0.5, 0.75, 1.0, 1.25),
            "stationary_velocity_threshold": 0.1,
        },
    )
    goal_position = RewTerm(
        func=goal_rewards.goal_position_exp, weight=4.0, params={"command_name": "motion", "std": 2.0}
    )
    goal_position_fine = RewTerm(
        func=goal_rewards.goal_position_exp, weight=4.0, params={"command_name": "motion", "std": 0.5}
    )
    goal_heading = RewTerm(
        func=goal_rewards.goal_heading_exp, weight=2.0, params={"command_name": "motion", "std": 0.5}
    )
    goal_settle = RewTerm(func=goal_rewards.goal_settle, weight=-0.5, params={"command_name": "motion"})
    goal_joint_deviation = RewTerm(
        func=goal_rewards.goal_joint_deviation_l1,
        weight=-1.0,
        params={"command_name": "motion", "asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )

    # -- both episode types
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits, weight=-10.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
                    r"(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "threshold": 1.0,
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
