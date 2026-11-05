# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.velocity.mdp as mdp

from ..mdp import rewards as goal_rewards
from .motion_cfg import FOOT_BODY_NAMES, STAIR_MOTION_FILES, STYLE_CLIP_FRAMES


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

    # -- task episodes: reach the destination looking like the nearest reference frame (motion-style task's term)
    task_motion_style = RewTerm(
        func=goal_rewards.TaskMotionMatchingStyle,
        weight=2.0,
        params={
            "command_name": "motion",
            "motion_files": STAIR_MOTION_FILES,
            "frame_limit": STYLE_CLIP_FRAMES,
            "include_mirrored": True,
            # combined RMS distance sqrt(mean dq^2 + (0.1 s)^2 mean dqd^2); standing at rest is >= 0.32 away
            "match_threshold": 0.25,
            "joint_vel_weight": 0.1,
            # per-group RMS joint-position error [rad] below which the reward is constant, and the falloff beyond it:
            # loose legs to adapt to new stairs, a tight upper body
            "joint_groups": {
                "legs": {"joint_names": ".*_(hip|knee|ankle)_.*", "inner_threshold": 0.15, "reward_std": 0.15},
                "upper_body": {
                    "joint_names": ".*(waist|shoulder|elbow|wrist).*",
                    "inner_threshold": 0.04,
                    "reward_std": 0.06,
                },
            },
            "min_joint_speed": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
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
