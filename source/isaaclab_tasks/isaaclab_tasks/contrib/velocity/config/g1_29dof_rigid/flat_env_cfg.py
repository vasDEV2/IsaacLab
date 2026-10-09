# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import math

from isaaclab_visualizers.kit import KitVisualizerCfg
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

from isaaclab.utils.configclass import configclass

from . import mdp
from .rough_env_cfg import G1RoughEnvCfg, G1RoughEnvCfgPerception
from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.env_cfg.perception_env.scene_cfg import STAIRS_TERRAINS_CFG

VISUALIZER = "newton_gl"
# VISUALIZER = "newton_rtx"
# VISUALIZER = "kit"


@configclass
class G1FlatEnvCfg(G1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Optionally override physics speed
        self.sim.dt = 0.005  # 200Hz
        self.decimation = 4  # 50Hz
        self.sim.render_interval = self.decimation

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # curriculum settings
        self.curriculum.terrain_levels = None  # type: ignore
        # self.curriculum.command_vel = None  # no running

        # no height scan
        self.scene.height_scanner = None  # type: ignore
        self.observations.policy.height_scan = None  # type: ignore
        self.observations.critic.height_scan = None  # type: ignore

        # Randomization
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)


@configclass
class G1FlatEnvCfgPerception(G1RoughEnvCfgPerception):
    def __post_init__(self):
        super().__post_init__()

        self.sim.dt = 0.005
        self.decimation = 4
        self.sim.render_interval = self.decimation

        # change terrain to flat
        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None  # type: ignore
        
        self.scene.height_scanner = None  # type: ignore
        self.observations.policy.height_scan = None  # type: ignore
        self.observations.critic.height_scan = None  # type: ignore

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

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.heading_command = False

        self.terminations.terrain_out_of_bounds = None  # type: ignore


class G1FlatEnvCfg_PLAY(G1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # change timestep
        self.episode_length_s = 20.0

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # disable curriculum
        self.curriculum.terrain_levels = None  # type: ignore
        self.curriculum.command_vel = None  # type: ignore

        # disable randomization for play
        self.observations.policy.enable_corruption = False

        # remove random pushing
        self.events.add_base_mass = None  # type: ignore
        self.events.push_robot = None  # type: ignore
        self.events.physics_material = None  # type: ignore
        self.events.scale_actuator_gains = None  # type: ignore

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (2, 2)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (self.episode_length_s / 4, self.episode_length_s / 4)
        self.commands.base_velocity.debug_vis = False

        # Randomization
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                # "yaw": (-math.pi, math.pi),
                # "yaw": (-math.pi/2, -math.pi/2),
                "yaw": (0, 0),
            },
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
            self.sim.visualizer_cfgs = [
                NewtonGLVisualizerCfg(eye=(12.0, 0.0, 6.0)),
            ]

            self.video_recorders = []

        elif VISUALIZER == "newton_rtx":
            self.sim.visualizer_cfgs = [
                NewtonRTXVisualizerCfg(eye=(12.0, 0.0, 6.0)),
            ]

            self.video_recorders = []

        elif VISUALIZER == "kit":
            self.sim.visualizer_cfgs = [
                KitVisualizerCfg(eye=(12.0, 0.0, 6.0)),
            ]

            self.video_recorders = []
