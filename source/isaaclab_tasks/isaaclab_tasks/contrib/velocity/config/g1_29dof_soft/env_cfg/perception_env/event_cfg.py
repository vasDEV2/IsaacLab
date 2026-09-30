# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp as g1_mdp
import isaaclab_tasks.contrib.velocity.config.g1_29dof_soft.mdp.perception_env as g1_soft_mdp
import isaaclab_tasks.core.velocity.mdp as mdp

contact_model = "3D-warp"
# contact_model = "2D-warp"
# contact_model = "cone-drft"
# contact_model = "cone-drft-multipoint"


@configclass
class G1EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            # "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "static_friction_range": (0.6, 1.0),
            # "dynamic_friction_range": (0.4, 0.6),
            "dynamic_friction_range": (0.2, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,  # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (0.0, 5.0),
            "operation": "add",
        },
    )

    # base_com = EventTerm(
    #     func=mdp.randomize_rigid_body_com,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
    #         "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
    #     },
    # )

    """
    reset
    """

    reset_base = EventTerm(
        # func=mdp.reset_root_state_uniform,
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
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    scale_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,  # type: ignore
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "operation": "scale",
            "distribution": "uniform",
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            # "stiffness_distribution_params": (0.75, 1.25),
            # "damping_distribution_params": (0.75, 1.25),
        },
    )

    # randomize terrain friction
    randomize_friction = EventTerm(
        func=g1_soft_mdp.randomize_terrain_friction,
        mode="reset",
        params={
            "friction_range": (0.1, 1.0),
            "contact_solver_name": "physics_callback",
        },
    )

    # randomize terrain stiffness
    if contact_model == "3D-warp" or contact_model == "2D-warp":
        randomize_stiffness = EventTerm(
            func=g1_soft_mdp.randomize_terrain_stiffness,
            mode="reset",
            params={
                "stiffness_range": (0.2, 0.9),
                "contact_solver_name": "physics_callback",
            },
        )
    elif contact_model == "cone-drft" or contact_model == "cone-drft-multipoint":
        randomize_stiffness = EventTerm(
            func=g1_soft_mdp.randomize_cone_model_terrain_stiffness,
            mode="reset",
            params={
                "sigma_flat_range": (1.0e6, 10.0e6),
                "sigma_cone_range": (0.15e6, 0.6e6),
                # "sigma_flat_range": (1.0e6, 1.0e6),
                # "sigma_cone_range": (0.15e6, 0.15e6),
                # "sigma_flat_range": (10.0e6, 10.0e6),
                # "sigma_cone_range": (0.6e6, 0.6e6),
                "contact_solver_name": "physics_callback",
            },
        )

    # randomize material density
    randomize_material_density = EventTerm(
        func=g1_soft_mdp.randomize_material_density,
        mode="reset",
        params={
            "packing_ratio_range": (0.5, 1.0),
            "bulk_density_range": (1000.0, 3000.0),
            "contact_solver_name": "physics_callback",
        },
    )

    """
    interval
    """

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )
