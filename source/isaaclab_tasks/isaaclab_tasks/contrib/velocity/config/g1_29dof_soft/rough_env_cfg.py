# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_visualizers.kit import KitVisualizerCfg
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
 
##
# Pre-defined configs
##
from .env_cfg import (
    G1ActionsCfg,
    G1CommandsCfg,
    G1CurriculumCfg,
    G1EventCfg,
    G1ObservationsCfg,
    G1PhysicsCfg,
    G1RewardsCfg,
    G1SceneCfg,
    G1TerminationsCfg,
)

VISUALIZER = "newton_gl"
# VISUALIZER = "newton_rtx"
# VISUALIZER = "kit"


@configclass
class G1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    sim: SimulationCfg = SimulationCfg(physics=G1PhysicsCfg())  # type: ignore
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
        # post init of parent
        super().__post_init__()

        # Randomization
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
            self.sim.visualizer_cfgs = [
                NewtonGLVisualizerCfg(eye=(12.0, 0.0, 6.0), headless=True),
            ]

            self.video_recorders = [
                VideoRecorderCfg(
                    source="visualizer:newton_gl", output_dir="videos/", video_length=200, video_interval=2000
                ),
            ]

        elif VISUALIZER == "newton_rtx":
            self.sim.visualizer_cfgs = [
                NewtonRTXVisualizerCfg(eye=(12.0, 0.0, 6.0), headless=True),
            ]

            self.video_recorders = [
                VideoRecorderCfg(
                    source="visualizer:newton_rtx", output_dir="videos/", video_length=200, video_interval=2000
                ),
            ]

        elif VISUALIZER == "kit":
            self.sim.visualizer_cfgs = [
                KitVisualizerCfg(eye=(12.0, 0.0, 6.0), headless=True),
            ]

            self.video_recorders = [
                VideoRecorderCfg(
                    source="visualizer:kit", output_dir="videos/", video_length=200, video_interval=2000
                ),
            ]


@configclass
class G1RoughEnvCfg_PLAY(G1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        # self.events.base_external_force_torque = None
        self.events.push_robot = None  # type: ignore
