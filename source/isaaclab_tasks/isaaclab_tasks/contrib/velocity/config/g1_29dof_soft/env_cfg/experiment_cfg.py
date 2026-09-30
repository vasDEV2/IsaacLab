# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Single editing location for G1 soft experiments.

Use the native Sensors, Perception, Actions, Observations, Rewards, Events,
Curriculum, Commands, Environment variants, and Learning sections below.
Offsets and ranges use [m], periods use [s], force thresholds use [N],
velocity ranges use [m/s] or [rad/s], and yaw uses [rad]. Sensor pattern
angles use the native [deg] convention and camera optics use [mm].
"""

import math

from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_newton.sensors import NewtonRaycastSensorCfg
from isaaclab_visualizers.kit import KitVisualizerCfg
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
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
from isaaclab_tasks.contrib.velocity.config.g1_29dof_soft.env_cfg.physics_cfg import G1PhysicsCfg
from isaaclab_tasks.core.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from isaaclab_assets import UNITREE_G1_29DOF_CFG


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
SOFT_CONTACT_THRESHOLD = 40.0  # [N], shared by actions, observations, and rewards.
RIGID_CONTACT_THRESHOLD = 5.0  # [N]
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
            "rigid_force_threshold": RIGID_CONTACT_THRESHOLD,
            "soft_force_threshold": SOFT_CONTACT_THRESHOLD,
        },
    )
    foot_contact_force = ObsTerm(
        func=g1_soft_mdp.foot_contact_forces_hybrid,
        params={
            "rigid_contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "soft_contact_sensor_name": "physics_callback",
            "rigid_force_filter_threshold": RIGID_CONTACT_THRESHOLD,
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
            "rigid_force_filter_threshold": RIGID_CONTACT_THRESHOLD,
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


@configclass
class G1RewardsCfg:
    """Reward terms for the MDP."""

    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=4.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.5)},
    )
    track_heading = RewTerm(
        func=g1_mdp.track_heading_world_exp,
        weight=4.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "base_velocity", "std": math.sqrt(0.5)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.5)}
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    action_rate_l2_lower_body = RewTerm(
        func=g1_mdp.action_rate_l2,
        weight=-0.01,
        params={
            "joint_idx": [
                i for i, name in enumerate(ACTIVE_JOINT) if any(part in name for part in ("_hip_", "_knee_", "_ankle_"))
            ]
        },
    )
    action_rate_l2_upper_body = RewTerm(
        func=g1_mdp.action_rate_l2,
        weight=-0.05,
        params={
            "joint_idx": [
                i
                for i, name in enumerate(ACTIVE_JOINT)
                if not any(part in name for part in ("_hip_", "_knee_", "_ankle_"))
            ]
        },
    )
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10, params={"target_height": 0.75})
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-10.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!.*ankle.*).*"), "threshold": 1.0},
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
            "weight_standing": {".*": 0.5},
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
            "walking_threshold": 0.05,
            "running_threshold": 2.0,
        },
    )
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
    feet_pitch = RewTerm(
        func=g1_mdp.reward_feet_pitch,
        weight=-4.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"], preserve_order=True)},
    )
    feet_pitch_diff = RewTerm(
        func=g1_mdp.reward_feet_pitch_diff,
        weight=-4.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"], preserve_order=True)},
    )
    feet_pitch_contact = RewTerm(
        func=g1_soft_mdp.reward_feet_pitch_contact_hybrid,
        weight=-4.0,
        params={
            "rigid_contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "soft_contact_sensor_name": "physics_callback",
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"], preserve_order=True),
        },
    )
    feet_air_time = RewTerm(
        func=g1_soft_mdp.feet_air_time_positive_biped_hybrid,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "threshold": 0.5,
            "rigid_contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "soft_contact_sensor_name": "physics_callback",
            "velocity_threshold": 0.05,
        },
    )
    no_fly = RewTerm(
        func=g1_soft_mdp.no_fly_hybrid,
        weight=-1.0,
        params={
            "rigid_contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "soft_contact_sensor_name": "physics_callback",
            "rigid_contact_threshold": RIGID_CONTACT_THRESHOLD,
            "soft_contact_threshold": SOFT_CONTACT_THRESHOLD,
            "command_name": "base_velocity",
            "velocity_threshold": 1.0,
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
    feet_slide = RewTerm(
        func=g1_soft_mdp.feet_slide_hybrid,
        weight=-0.25,
        params={
            "rigid_contact_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*ankle_roll.*", preserve_order=True
            ),
            "soft_contact_sensor_name": "physics_callback",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "rigid_contact_threshold": RIGID_CONTACT_THRESHOLD,
            "soft_contact_threshold": SOFT_CONTACT_THRESHOLD,
        },
    )
    contact_impulse = RewTerm(
        func=g1_soft_mdp.reward_soft_landing_hybrid,
        weight=-0.005,
        params={
            "rigid_contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "soft_contact_sensor_name": "physics_callback",
            "command_name": "base_velocity",
            "command_threshold": 0.05,
        },
    )
    foot_clearance = RewTerm(
        func=g1_mdp.foot_clearance_reward,
        weight=5.0,
        params={
            "target_height": 0.1,
            "std": 0.05,
            "tanh_mult": 2.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "standing_position_foot_z": 0.03539,
        },
    )


@configclass
class G1EventCfg:
    """Configuration for events."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.2, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (0.0, 5.0),
            "operation": "add",
        },
    )
    reset_base = EventTerm(
        func=g1_mdp.reset_root_state_uniform_on_ground,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
    )
    scale_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "operation": "scale",
            "distribution": "uniform",
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
        },
    )
    randomize_friction = EventTerm(
        func=g1_soft_mdp.randomize_terrain_friction,
        mode="reset",
        params={"friction_range": (0.1, 1.0), "contact_solver_name": "physics_callback"},
    )
    if contact_model == "3D-warp" or contact_model == "2D-warp":
        randomize_stiffness = EventTerm(
            func=g1_soft_mdp.randomize_terrain_stiffness,
            mode="reset",
            params={"stiffness_range": (0.2, 0.9), "contact_solver_name": "physics_callback"},
        )
    elif contact_model == "cone-drft" or contact_model == "cone-drft-multipoint":
        randomize_stiffness = EventTerm(
            func=g1_soft_mdp.randomize_cone_model_terrain_stiffness,
            mode="reset",
            params={
                "sigma_flat_range": (1000000.0, 10000000.0),
                "sigma_cone_range": (150000.0, 600000.0),
                "contact_solver_name": "physics_callback",
            },
        )
    randomize_material_density = EventTerm(
        func=g1_soft_mdp.randomize_material_density,
        mode="reset",
        params={
            "packing_ratio_range": (0.5, 1.0),
            "bulk_density_range": (1000.0, 3000.0),
            "contact_solver_name": "physics_callback",
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )


@configclass
class G1CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    command_vel = CurrTerm(
        func=g1_mdp.commands_vel,
        params={
            "command_name": "base_velocity",
            "velocity_stages": [
                {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
                {"step": 10000 * 24, "lin_vel_x": (-1.0, 1.7), "ang_vel_z": (-0.7, 0.7)},
                {"step": 15000 * 24, "lin_vel_x": (-1.0, 2.5), "ang_vel_z": (-1.0, 1.0)},
            ],
        },
    )
    track_lin_vel = CurrTerm(
        func=g1_mdp.ramp_reward_param,
        params={
            "term_name": "track_lin_vel_xy",
            "param_name": "std",
            "val_0": math.sqrt(1.0),
            "val_1": math.sqrt(0.1),
            "step_0": 0,
            "step_1": 15000 * 24,
        },
    )
    track_ang_vel = CurrTerm(
        func=g1_mdp.ramp_reward_param,
        params={
            "term_name": "track_ang_vel_z",
            "param_name": "std",
            "val_0": math.sqrt(1.0),
            "val_1": math.sqrt(0.1),
            "step_0": 0,
            "step_1": 15000 * 24,
        },
    )
    track_heading = CurrTerm(
        func=g1_mdp.ramp_reward_param,
        params={
            "term_name": "track_heading",
            "param_name": "std",
            "val_0": math.sqrt(1.0),
            "val_1": math.sqrt(0.1),
            "step_0": 0,
            "step_1": 15000 * 24,
        },
    )
    track_lin_vel_weight = CurrTerm(
        func=g1_mdp.ramp_reward_weight,
        params={"term_name": "track_lin_vel_xy", "weight_0": 4.0, "weight_1": 8.0, "step_0": 0, "step_1": 15000 * 24},
    )
    track_heading_weight = CurrTerm(
        func=g1_mdp.ramp_reward_weight,
        params={"term_name": "track_heading", "weight_0": 4.0, "weight_1": 8.0, "step_0": 0, "step_1": 15000 * 24},
    )


@configclass
class G1CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = g1_mdp.UniformVelocityYawCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=g1_mdp.UniformVelocityYawCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0)
        ),
    )


@configclass
class G1TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    base_too_low = TerminationTermCfg(
        func=g1_mdp.root_height_below_minimum_adaptive,
        params={"minimum_height": 0.2, "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link")},
    )
    bad_orientation = TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": 0.8})
    terrain_out_of_bounds = TerminationTermCfg(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"), "distance_buffer": 1.0},
    )


@configclass
class G1SceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    rigid_floor: TerrainImporterCfg = None
    visual_terrain: TerrainImporterCfg = None
    robot: ArticulationCfg = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    mid360_lidar = SENSORS.mid360_lidar
    d435_camera = SENSORS.d435_camera
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


VISUALIZER = "newton_gl"


@configclass
class G1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    sim: SimulationCfg = SimulationCfg(physics=G1PhysicsCfg())
    rewards: G1RewardsCfg = G1RewardsCfg()
    actions: G1ActionsCfg = G1ActionsCfg()
    observations: G1ObservationsCfg = G1ObservationsCfg()
    scene: G1SceneCfg = G1SceneCfg(num_envs=4096, env_spacing=2.5)
    terminations: G1TerminationsCfg = G1TerminationsCfg()
    curriculum: G1CurriculumCfg = G1CurriculumCfg()
    events: G1EventCfg = G1EventCfg()
    commands: G1CommandsCfg = G1CommandsCfg()
    seed: int = 42

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        if VISUALIZER == "newton_gl":
            self.sim.visualizer_cfgs = [NewtonGLVisualizerCfg(eye=(12.0, 0.0, 6.0), headless=True)]
            self.video_recorders = [
                VideoRecorderCfg(
                    source="visualizer:newton_gl", output_dir="videos/", video_length=200, video_interval=2000
                )
            ]
        elif VISUALIZER == "newton_rtx":
            self.sim.visualizer_cfgs = [NewtonRTXVisualizerCfg(eye=(12.0, 0.0, 6.0), headless=True)]
            self.video_recorders = [
                VideoRecorderCfg(
                    source="visualizer:newton_rtx", output_dir="videos/", video_length=200, video_interval=2000
                )
            ]
        elif VISUALIZER == "kit":
            self.sim.visualizer_cfgs = [KitVisualizerCfg(eye=(12.0, 0.0, 6.0), headless=True)]
            self.video_recorders = [
                VideoRecorderCfg(source="visualizer:kit", output_dir="videos/", video_length=200, video_interval=2000)
            ]


@configclass
class G1RoughEnvCfg_PLAY(G1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None


@configclass
class G1FlatEnvCfg(G1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 0.005
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.scene.terrain = g1_soft_mdp.CurriculumSoftTerrain.copy()
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.reset_robot_joints.params["position_range"] = (0.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)
        self.terminations.terrain_out_of_bounds = None


class G1FlatEnvCfg_PLAY(G1FlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.sim.dt = 1 / 200
        self.decimation = 4
        self.episode_length_s = 10.0
        self.scene.num_envs = 50
        self.scene.env_spacing = 0.0
        self.scene.terrain = g1_soft_mdp.SoftTerrain.copy()
        self.curriculum.terrain_levels = None
        self.curriculum.command_vel = None
        self.observations.policy.enable_corruption = False
        self.events.add_base_mass = None
        self.events.push_robot = None
        self.events.physics_material = None
        self.events.scale_actuator_gains = None
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (self.episode_length_s, self.episode_length_s)
        self.commands.base_velocity.debug_vis = True
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        if VISUALIZER == "newton_gl":
            self.sim.visualizer_cfgs = [NewtonGLVisualizerCfg(eye=(0.0, -4.0, 1.0))]
            self.video_recorders = []
        elif VISUALIZER == "newton_rtx":
            self.sim.visualizer_cfgs = [NewtonRTXVisualizerCfg(eye=(0.0, -4.0, 1.0))]
            self.video_recorders = []
        elif VISUALIZER == "kit":
            self.sim.visualizer_cfgs = [KitVisualizerCfg(eye=(0.0, -4.0, 1.0))]
            self.video_recorders = []
