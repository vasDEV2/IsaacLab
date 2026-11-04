# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone perceptive G1 environment configuration."""

from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from .env_cfg import (
    G1ActionsCfg,
    G1CommandsCfg,
    G1CurriculumCfg,
    G1EventCfg,
    G1PerceptionObservationsCfg,
    G1PerceptionSceneCfg,
    G1PhysicsCfg,
    G1RewardsCfg,
    G1TerminationsCfg,
)


@configclass
class G1RoughPerceptionEnvCfg(ManagerBasedRLEnvCfg):
    """AME height-map-driven locomotion on Vasu's pyramid stairs, with current G1 dynamics."""

    scene: G1PerceptionSceneCfg = G1PerceptionSceneCfg(num_envs=128, env_spacing=2.5)
    observations: G1PerceptionObservationsCfg = G1PerceptionObservationsCfg()
    sim: SimulationCfg = SimulationCfg(physics=G1PhysicsCfg())
    actions: G1ActionsCfg = G1ActionsCfg()
    commands: G1CommandsCfg = G1CommandsCfg()
    rewards: G1RewardsCfg = G1RewardsCfg()
    terminations: G1TerminationsCfg = G1TerminationsCfg()
    events: G1EventCfg = G1EventCfg()
    curriculum: G1CurriculumCfg = G1CurriculumCfg()
    seed: int = 42

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt
        # self.sim.visualizer_cfgs = [
        #     NewtonGLVisualizerCfg(eye=(12.0, 0.0, 6.0), headless=False),
        # # #     # NewtonRTXVisualizerCfg(eye=(12.0, 0.0, 6.0), headless=False),
        # ]

        self.sim.visualizer_cfgs = []
        # self.video_recorders = []


        self.video_recorders = [
            VideoRecorderCfg(
                source="visualizer:newton_gl",
                output_dir="/workspace/isaaclab/logs/videos",
                video_length=200,
                video_interval=2000,
            ),
        ]
        # self.scene.mid360_lidar.debug_vis = True
        # self.scene.height_scanner.debug_vis = True


@configclass
class G1RoughPerceptionEnvCfg_PLAY(G1RoughPerceptionEnvCfg):
    """Small, deterministic-command evaluation scene."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        # self.scene.terrain.terrain_generator.num_rows = 5
        # self.scene.terrain.terrain_generator.num_cols = 5
        # self.scene.terrain.terrain_generator.curriculum = False
        # self.scene.terrain.max_init_terrain_level = None
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.curriculum.terrain_levels = None

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(eye=(0.0, -4.0, 1.0)),
            # NewtonRTXVisualizerCfg(eye=(0.0, -4.0, 1.0)),
        ]
        self.video_recorders = []
