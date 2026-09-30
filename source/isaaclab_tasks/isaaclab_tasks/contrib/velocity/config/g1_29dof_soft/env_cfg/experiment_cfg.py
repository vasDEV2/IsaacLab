# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Single editing location for G1 soft experiment choices.

Sections use Isaac Lab's native configuration types. Edit defaults here, then use
this file's training entry point or the standard Isaac Lab train/play commands.
Sensor distances/offsets use [m], periods use [s], angles in scan patterns use
[deg], and camera optical parameters retain Isaac Lab's native [mm] convention.
"""

from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_newton.sensors import NewtonRaycastSensorCfg

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, patterns
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp as g1_mdp
import isaaclab_tasks.contrib.velocity.config.g1_29dof_soft.mdp as g1_soft_mdp
import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab_tasks.contrib.soft_contact import (
    BoxColliderCfg,
    PhysicsCallbackActionCfg,
    PlaneColliderCfg,
    SphereColliderCfg,
)


@configclass
class G1SensorsCfg:
    """Perception sensor configurations; set a sensor to None to disable it."""

    mid360_lidar = NewtonRaycastSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*torso_link",
        update_period=0.1,
        max_distance=40.0,
        ray_alignment="base",
        global_world_only=False,
        mesh_prim_paths=[],
        offset=NewtonRaycastSensorCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),
        pattern_cfg=patterns.LidarPatternCfg(
            channels=40, vertical_fov_range=(-7.0, 52.0), horizontal_fov_range=(0.0, 360.0), horizontal_res=1.0
        ),
        debug_vis=True,
    )
    d435_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/d435_link/Camera",
        update_period=1 / 30,
        width=640,
        height=480,
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=1.93, horizontal_aperture=3.68, clipping_range=(0.1, 10.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0), convention="ros"),
        renderer_cfg=NewtonWarpRendererCfg(),
    )


@configclass
class G1PerceptionCfg:
    """Policy sensor selection and preprocessing; perception is opt-in."""

    lidar: bool = False
    rgb: bool = False
    depth: bool = False
    lidar_sensor: str = "mid360_lidar"
    camera_sensor: str = "d435_camera"
    normalize_rgb: bool = True
    lidar_clip_max: float | None = None
    "Optional observation-only range limit [m]; None uses the live sensor range."
    depth_clip_max: float | None = None
    "Optional observation-only depth limit [m]; None uses the live camera far plane."


SENSORS = G1SensorsCfg()
PERCEPTION = G1PerceptionCfg()
COLLIDER_SHAPE = "box"
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
    collider_cfg = SphereColliderCfg(radius=0.05, center=(0.0, 0.0, 0.0), resolution=(8, 8))
else:
    raise ValueError(f"Unsupported collider shape: {COLLIDER_SHAPE}")
SOFT_CONTACT_THRESHOLD = 40.0
contact_model = "3D-warp"
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
        asset_name="robot", joint_names=ACTIVE_JOINT, scale=0.25, use_default_offset=True, preserve_order=True
    )
    "\n    Contact solver.\n    "
    physics_callback = PhysicsCallbackActionCfg(
        asset_name="robot",
        body_names=[".*_ankle_roll_link"],
        backend=contact_model,
        intruder_geometry_cfg=collider_cfg,
        enable_ema_filter=False,
        contact_threshold=SOFT_CONTACT_THRESHOLD,
        debug_vis=False,
        contact_data_history_length=10,
        history_logging_decimation=10,
        contact_vis_force_threshold=SOFT_CONTACT_THRESHOLD,
    )


LEG_JOINT = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
]


@configclass
class PolicyCfg(ObsGroup):
    """Observations for policy group."""

    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), scale=0.25)
    projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        noise=Unoise(n_min=-0.01, n_max=0.01),
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTIVE_JOINT, preserve_order=True)},
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        noise=Unoise(n_min=-1.5, n_max=1.5),
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTIVE_JOINT, preserve_order=True)},
        scale=0.05,
    )
    actions = ObsTerm(func=mdp.last_action)
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

    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
    base_quat = ObsTerm(func=mdp.root_quat_w, params={"make_quat_unique": True})
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTIVE_JOINT, preserve_order=True)},
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTIVE_JOINT, preserve_order=True)},
        scale=0.05,
    )
    actions = ObsTerm(func=mdp.last_action)
    height_scan = ObsTerm(
        func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner")}, clip=(-1.0, 1.0)
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
        func=g1_mdp.foot_height, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link")}
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


@configclass
class LoggingObsCfg(ObsGroup):
    """Observations for policy group."""

    base_pos = ObsTerm(func=mdp.root_pos_w)
    base_quat = ObsTerm(func=mdp.root_quat_w)
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    contact_forces = ObsTerm(
        func=g1_soft_mdp.foot_contact_forces_raw_hybrid,
        params={
            "rigid_contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "soft_contact_sensor_name": "physics_callback",
            "rigid_force_filter_threshold": 5.0,
            "soft_force_filter_threshold": SOFT_CONTACT_THRESHOLD,
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class G1ObservationsCfg:
    """Observation specifications for the MDP."""

    policy: PolicyHistoryCfg = PolicyHistoryCfg()
    critic: CriticHistoryCfg = CriticHistoryCfg()
    privileged: PrivilegedHistoryCfg = PrivilegedHistoryCfg()
