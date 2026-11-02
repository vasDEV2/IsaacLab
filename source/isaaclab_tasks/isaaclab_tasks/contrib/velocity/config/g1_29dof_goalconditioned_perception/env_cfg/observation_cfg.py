# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_contrib.mdp import MirrorObservationTermCfg as MirrorObsTerm
from isaaclab_contrib.mdp import mirror_identity, mirror_quat, mirror_vec3

import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp as g1_mdp
import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp import symmetry

from ..mdp.depth import depth_image
from ..mdp.motion import motion_joint_pos_error, motion_phase, motion_ref_joint_pos_rel, motion_ref_joint_vel
from ..mdp.observations import elevation_map, finite_base_ang_vel, lidar_point_cloud
from ..mdp.symmetry import mirror_elevation_map
from .action_cfg import G1ActionsCfg

ACTIVE_JOINTS = [
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
class PolicyCfg(ObsGroup):
    """Observations for policy group."""

    base_ang_vel = MirrorObsTerm(
        func=finite_base_ang_vel,
        mirror=mirror_vec3,
        mirror_params={"axial": True},
        noise=Unoise(n_min=-0.2, n_max=0.2),
        scale=0.25,
    )
    projected_gravity = MirrorObsTerm(
        func=mdp.projected_gravity, mirror=mirror_vec3, noise=Unoise(n_min=-0.05, n_max=0.05)
    )
    joint_pos = MirrorObsTerm(
        func=mdp.joint_pos_rel,
        mirror=symmetry.mirror_g1_joints,
        mirror_params={"joint_names": ACTIVE_JOINTS},
        noise=Unoise(n_min=-0.01, n_max=0.01),
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTIVE_JOINTS, preserve_order=True)},
    )
    joint_vel = MirrorObsTerm(
        func=mdp.joint_vel_rel,
        mirror=symmetry.mirror_g1_joints,
        mirror_params={"joint_names": ACTIVE_JOINTS},
        noise=Unoise(n_min=-1.5, n_max=1.5),
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTIVE_JOINTS, preserve_order=True)},
        scale=0.05,
    )
    actions = MirrorObsTerm(
        func=mdp.last_action,
        mirror=symmetry.mirror_g1_joints,
        mirror_params={"joint_names": G1ActionsCfg().joint_pos.joint_names},
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        self.history_length = 1


@configclass
class CriticCfg(ObsGroup):
    """Observations for policy group."""

    base_ang_vel = MirrorObsTerm(
        func=finite_base_ang_vel, mirror=mirror_vec3, mirror_params={"axial": True}, scale=0.25
    )
    base_quat = MirrorObsTerm(func=mdp.root_quat_w, mirror=mirror_quat, params={"make_quat_unique": True})
    joint_pos = MirrorObsTerm(
        func=mdp.joint_pos_rel,
        mirror=symmetry.mirror_g1_joints,
        mirror_params={"joint_names": ACTIVE_JOINTS},
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTIVE_JOINTS, preserve_order=True)},
    )
    joint_vel = MirrorObsTerm(
        func=mdp.joint_vel_rel,
        mirror=symmetry.mirror_g1_joints,
        mirror_params={"joint_names": ACTIVE_JOINTS},
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTIVE_JOINTS, preserve_order=True)},
        scale=0.05,
    )
    actions = MirrorObsTerm(
        func=mdp.last_action,
        mirror=symmetry.mirror_g1_joints,
        mirror_params={"joint_names": G1ActionsCfg().joint_pos.joint_names},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 1


@configclass
class PolicyHistoryCfg(PolicyCfg):
    def __post_init__(self):
        self.history_length = 1


@configclass
class CriticHistoryCfg(CriticCfg):
    def __post_init__(self):
        self.history_length = 1


@configclass
class CommandCfg(ObsGroup):
    """Observations for command group."""

    velocity_commands = MirrorObsTerm(
        func=mdp.generated_commands, mirror=symmetry.mirror_velocity_heading, params={"command_name": "base_velocity"}
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class PrivilegedObsCfg(ObsGroup):
    """Observations for policy group."""

    base_lin_vel = MirrorObsTerm(func=mdp.base_lin_vel, mirror=mirror_vec3)
    foot_height = MirrorObsTerm(
        func=g1_mdp.foot_height,
        mirror=symmetry.mirror_foot_scalars,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link")},
    )
    foot_contact = MirrorObsTerm(
        func=g1_mdp.foot_contact,
        mirror=symmetry.mirror_foot_scalars,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"), "threshold": 5.0},
    )
    foot_contact_force = MirrorObsTerm(
        func=g1_mdp.foot_contact_forces,
        mirror=symmetry.mirror_foot_forces,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )
    foot_air_time = MirrorObsTerm(
        func=g1_mdp.foot_air_time,
        mirror=symmetry.mirror_foot_scalars,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class PrivilegedHistoryCfg(PrivilegedObsCfg):
    def __post_init__(self):
        self.history_length = 1


@configclass
class MotionPriorCfg(ObsGroup):
    """Privileged stair-climbing reference for the critic only; the actor never sees the clip."""

    ref_joint_pos = MirrorObsTerm(
        func=motion_ref_joint_pos_rel,
        mirror=symmetry.mirror_g1_joints,
        mirror_params={"joint_names": ACTIVE_JOINTS},
        params={
            "command_name": "motion",
            "asset_cfg": SceneEntityCfg("robot", joint_names=ACTIVE_JOINTS, preserve_order=True),
        },
    )
    ref_joint_vel = MirrorObsTerm(
        func=motion_ref_joint_vel,
        mirror=symmetry.mirror_g1_joints,
        mirror_params={"joint_names": ACTIVE_JOINTS},
        params={
            "command_name": "motion",
            "asset_cfg": SceneEntityCfg("robot", joint_names=ACTIVE_JOINTS, preserve_order=True),
        },
        scale=0.05,
    )
    joint_pos_error = MirrorObsTerm(
        func=motion_joint_pos_error,
        mirror=symmetry.mirror_g1_joints,
        mirror_params={"joint_names": ACTIVE_JOINTS},
        params={
            "command_name": "motion",
            "asset_cfg": SceneEntityCfg("robot", joint_names=ACTIVE_JOINTS, preserve_order=True),
        },
    )
    phase = MirrorObsTerm(func=motion_phase, mirror=mirror_identity, params={"command_name": "motion"})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class LidarCfg(ObsGroup):
    """One current scan; proprioceptive history is maintained in its own group."""

    points = ObsTerm(func=lidar_point_cloud, params={"sensor_cfg": SceneEntityCfg("mid360_lidar"), "num_points": 1024})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 0


@configclass
class TerrainCfg(ObsGroup):
    """Privileged ground-height measurements for the critic only."""

    height_scan = ObsTerm(
        func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner")}, clip=(-1.0, 1.0)
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ElevationMapCfg(ObsGroup):
    """Current XYZ map in the torso yaw frame; retain both grid dimensions."""

    map = MirrorObsTerm(
        func=elevation_map,
        mirror=mirror_elevation_map,
        params={"sensor_cfg": SceneEntityCfg("height_scanner"), "missing_height": -2.0},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.concatenate_dim = -1
        self.history_length = 0


@configclass
class G1PerceptionObservationsCfg:
    """Task-owned observation groups for the perception actor and privileged critic."""

    policy: PolicyHistoryCfg = PolicyHistoryCfg()
    critic: CriticHistoryCfg = CriticHistoryCfg()
    command: CommandCfg = CommandCfg()
    privileged: PrivilegedHistoryCfg = PrivilegedHistoryCfg()
    motion_prior: MotionPriorCfg = MotionPriorCfg()
    elevation_map: ElevationMapCfg = ElevationMapCfg()


@configclass
class DepthCfg(ObsGroup):
    """One delayed image; image dimensions are retained for RSL-RL's CNN actor."""

    image = ObsTerm(
        func=depth_image,
        params={"sensor_cfg": SceneEntityCfg("d435_camera"), "resize": (58, 87), "corrupt": True},
        delay_min_lag=3,
        delay_max_lag=4,
        delay_hold_prob=0.0,
    )

    def __post_init__(self):
        self.enable_corruption = False  # The depth term owns structured image corruption.
        self.concatenate_terms = True
        self.concatenate_dim = 0
        self.history_length = 0


@configclass
class G1DepthObservationsCfg(G1PerceptionObservationsCfg):
    """Depth actor observations and the task's privileged critic observations."""

    lidar: LidarCfg | None = None
    depth: DepthCfg = DepthCfg()
