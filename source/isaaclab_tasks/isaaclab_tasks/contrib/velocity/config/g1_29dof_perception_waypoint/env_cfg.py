# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 perception on the stair curriculum, commanded by a waypoint instead of a velocity."""

from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_contrib.mdp import MirrorObservationTermCfg as MirrorObsTerm

import isaaclab_tasks.core.velocity.mdp as mdp

from ..g1_29dof_perception.env_cfg import (
    G1CurriculumCfg,
    G1EventCfg,
    G1PerceptionObservationsCfg,
    G1RewardsCfg,
)
from ..g1_29dof_perception.env_cfg.observation_cfg import PolicyHistoryCfg
from ..g1_29dof_perception.rough_env_cfg import G1RoughPerceptionEnvCfg
from . import mdp as waypoint_mdp


@configclass
class G1WaypointCommandsCfg:
    """The waypoint, and the walking intent derived from it for gait terms."""

    waypoint = waypoint_mdp.WaypointCommandCfg(debug_vis=True)
    waypoint_gait = waypoint_mdp.WaypointGaitCommandCfg(waypoint_command_name="waypoint")


@configclass
class WaypointCommandObsCfg(ObsGroup):
    """Waypoint relative to the root, in the root yaw frame [m]."""

    waypoint = MirrorObsTerm(
        func=mdp.generated_commands, mirror=waypoint_mdp.mirror_waypoint, params={"command_name": "waypoint"}
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class WaypointPolicyCfg(PolicyHistoryCfg):
    """Perception proprioception followed by the root position on the tile."""

    root_position = MirrorObsTerm(
        func=waypoint_mdp.root_pos_tile_b, mirror=waypoint_mdp.mirror_waypoint, noise=Unoise(n_min=-0.02, n_max=0.02)
    )


@configclass
class G1WaypointObservationsCfg(G1PerceptionObservationsCfg):
    """Perception observations with the root position in ``policy`` and the waypoint in ``command``.

    Both groups feed the actor and the critic.
    """

    policy: WaypointPolicyCfg = WaypointPolicyCfg()
    command: WaypointCommandObsCfg = WaypointCommandObsCfg()


@configclass
class G1WaypointRewardsCfg(G1RewardsCfg):
    """Waypoint error rewards with the perception task's gait and regularization terms.

    Velocity tracking and the world-height target are removed. Gait terms that gate on a velocity
    command read ``waypoint_gait`` (see :class:`~.mdp.WaypointGaitCommand`).
    """

    track_lin_vel_xy = None
    track_ang_vel_z = None
    track_heading = None
    root_height_world = None
    joint_deviation = None

    waypoint_position = RewTerm(
        func=waypoint_mdp.waypoint_position_exp, weight=4.0, params={"command_name": "waypoint", "std": 2.0}
    )
    waypoint_position_fine = RewTerm(
        func=waypoint_mdp.waypoint_position_exp, weight=4.0, params={"command_name": "waypoint", "std": 0.5}
    )
    waypoint_height = RewTerm(
        func=waypoint_mdp.waypoint_height_exp, weight=2.0, params={"command_name": "waypoint", "std": 0.1}
    )
    waypoint_progress = RewTerm(
        func=waypoint_mdp.waypoint_progress, weight=2.0, params={"command_name": "waypoint", "max_speed": 1.0}
    )
    waypoint_bearing = RewTerm(
        func=waypoint_mdp.waypoint_bearing_exp, weight=1.0, params={"command_name": "waypoint", "std": 0.5}
    )
    waypoint_stand_still = RewTerm(
        func=waypoint_mdp.waypoint_stand_still, weight=-1.0, params={"command_name": "waypoint"}
    )

    def __post_init__(self):
        for term in vars(self).values():
            if isinstance(term, RewTerm) and term.params.get("command_name") == "base_velocity":
                term.params["command_name"] = "waypoint_gait"


@configclass
class G1WaypointCurriculumCfg(G1CurriculumCfg):
    """Promote stair levels on waypoint success; keep the gait weight ramps."""

    terrain_levels = CurrTerm(func=waypoint_mdp.terrain_levels_waypoint, params={"command_name": "waypoint"})
    track_lin_vel = None
    track_ang_vel = None
    track_heading = None


@configclass
class G1WaypointEventCfg(G1EventCfg):
    """Start each episode at rest at the waypoint command's start pose."""

    reset_base = EventTerm(
        func=waypoint_mdp.reset_root_at_waypoint_start, mode="reset", params={"command_name": "waypoint"}
    )


@configclass
class G1RoughPerceptionWaypointEnvCfg(G1RoughPerceptionEnvCfg):
    """Reach a waypoint across the stairs: climb down from pyramid tops and up out of pits, or the reverse.

    The terrain rows are the perception task's levels (flat, up 1 rise, down 1 rise, up 2, ...). Waypoints sit on
    the top platform or pit bottom at the tile center and on the flat border around it, at ``base_height`` above
    the ground under them.
    """

    observations: G1WaypointObservationsCfg = G1WaypointObservationsCfg()
    commands: G1WaypointCommandsCfg = G1WaypointCommandsCfg()
    rewards: G1WaypointRewardsCfg = G1WaypointRewardsCfg()
    events: G1WaypointEventCfg = G1WaypointEventCfg()
    curriculum: G1WaypointCurriculumCfg = G1WaypointCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # start on flat ground and the one-rise levels; success promotes from there
        self.scene.terrain.max_init_terrain_level = 2


@configclass
class G1RoughPerceptionWaypointEnvCfg_PLAY(G1RoughPerceptionWaypointEnvCfg):
    """Evaluate on every stair level, without promotion or pushes."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.sim.visualizer_cfgs = [NewtonGLVisualizerCfg(eye=(0.0, -6.0, 3.0))]
        self.video_recorders = []


@configclass
class G1FlatPerceptionWaypointEnvCfg(G1RoughPerceptionWaypointEnvCfg):
    """Plane variant: the same center and border waypoints, all at ground level."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None


@configclass
class G1FlatPerceptionWaypointEnvCfg_PLAY(G1RoughPerceptionWaypointEnvCfg_PLAY):
    """Plane evaluation variant."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
