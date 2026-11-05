# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Staged G1 perception: flat waypoints and reference stair climbing, then motion-style pyramid goals.

Until the switch (3000 iterations), half of the environments walk to a waypoint at the center of a flat tile with
this package's goal and gait rewards (without the style reward), and half purely imitate the reference clips on
their own pyramid, with the ``g1_29dof_goalconditioned_perception`` imitation rewards and terminations. Both run
10 s episodes. From the switch on, every environment moves at its next reset to a clip-shaped pyramid, where the
same waypoint command puts the goal on the landing, and trains like this package's motion-style task with the
motion-matching style reward.

Each source task's reward and termination terms are wrapped in :class:`~.mdp.ModeGated` under a ``flat_``,
``stairs_``, or ``pyramid_`` prefix, so they act only on environments of their mode.
"""

import copy

from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import MeshPlaneTerrainCfg, TerrainGeneratorCfg
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.mdp import MirrorObservationTermCfg as MirrorObsTerm
from isaaclab_contrib.mdp import mirror_identity, mirror_vec3

import isaaclab_tasks.contrib.velocity.config.g1_29dof_perception_waypoint.mdp as waypoint_mdp
import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp as g1_mdp
import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab_tasks.contrib.mimic.mdp.symmetry import mirror_rotation_6d
from isaaclab_tasks.contrib.velocity.config.g1_29dof_goalconditioned_perception import env_cfg as stairs_cfg
from isaaclab_tasks.contrib.velocity.config.g1_29dof_goalconditioned_perception.env_cfg.motion_cfg import (
    STAIR_MOTION_FILES,
)
from isaaclab_tasks.contrib.velocity.config.g1_29dof_goalconditioned_perception.mdp import (
    motion_observations,
)
from isaaclab_tasks.contrib.velocity.config.g1_29dof_goalconditioned_perception.mdp.events import (
    reset_goal_joint_state,
    reset_goal_root_state,
)
from isaaclab_tasks.contrib.velocity.config.g1_29dof_goalconditioned_perception.mdp.terrains import (
    MotionStairsTerrainCfg,
)
from isaaclab_tasks.contrib.velocity.config.g1_29dof_perception import env_cfg as flat_cfg
from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp import symmetry

from . import env_cfg as pyramid_cfg
from . import mdp as style_mdp
from .env_cfg.observation_cfg import ACTIVE_JOINTS, CriticHistoryCfg
from .mdp import FLAT, PYRAMID, STAIRS, ModeGated, ModeGatedEvent
from .mdp.terrains import ClipPyramidStairsTerrainCfg
from .rough_env_cfg import G1RoughPerceptionEnvCfg

SWITCH_ITERATION = 3000
STEPS_PER_ITERATION = 24
"""Must match the runner's ``num_steps_per_env``; the switch is counted in control steps."""


def _gated(func, func_params: dict, modes: tuple[int, ...]) -> dict:
    return {"func": func, "func_params": func_params, "modes": modes}


def _gated_terms(source, prefix: str, modes: tuple[int, ...], term_type: type, skip: tuple[str, ...] = ()) -> dict:
    """Copy every ``term_type`` term of a source config, wrapped in :class:`ModeGated` and renamed ``prefix_name``."""
    terms = {}
    for name, term in vars(source).items():
        if isinstance(term, term_type) and name not in skip:
            params = _gated(term.func, copy.deepcopy(term.params), modes)
            terms[f"{prefix}_{name}"] = term.replace(func=ModeGated, params=params)
    return terms


def _make_cfg(name: str, doc: str, terms: dict) -> type:
    namespace = {"__annotations__": {key: type(value) for key, value in terms.items()}, **terms}
    namespace.update({"__doc__": doc, "__module__": __name__})
    return configclass(type(name, (), namespace))


"""
Scene: flat, reference-pyramid, and clip-shaped pyramid columns on one curriculum generator.
"""

_TERRAIN_GENERATOR = TerrainGeneratorCfg(
    size=(9.0, 9.0),
    border_width=5.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        # flat waypoints: the goal is the tile center on the ground
        "flat": MeshPlaneTerrainCfg(proportion=0.25),
        # the reference clips' own pyramid, with the clip origin at the tile origin (goal-conditioned task)
        "motion_stairs": MotionStairsTerrainCfg(proportion=0.25, motion_file=STAIR_MOTION_FILES[0]),
        # clip-shaped pyramids whose rises grow with the row (motion-style task); every env ends up here
        "pyramid": ClipPyramidStairsTerrainCfg(proportion=0.5),
    },
)


@configclass
class G1StagedSceneCfg(pyramid_cfg.G1PerceptionSceneCfg):
    """The perception scene with all three tile types; :class:`~.mdp.EnvModeCommand` places each environment."""

    terrain = pyramid_cfg.G1PerceptionSceneCfg(num_envs=1, env_spacing=2.5).terrain.replace(
        terrain_generator=_TERRAIN_GENERATOR, max_init_terrain_level=None
    )


"""
Commands.
"""


@configclass
class G1StagedCommandsCfg:
    """The mode schedule and every mode's commands; each command runs for all environments, gated where used."""

    env_mode = style_mdp.EnvModeCommandCfg(switch_step=SWITCH_ITERATION * STEPS_PER_ITERATION)
    # stairs: pure imitation of the reference clips for the whole stage (no goal-conditioned task episodes)
    motion = stairs_cfg.G1CommandsCfg().motion.replace(imitation_probability=1.0, debug_vis=False, task_terrain=None)
    # flat and pyramid: the tile-center goal (on the ground for flat tiles, on the landing for pyramids)
    # and its gait intent
    waypoint = pyramid_cfg.G1CommandsCfg().waypoint.replace(debug_vis=False)
    waypoint_gait = pyramid_cfg.G1CommandsCfg().waypoint_gait


"""
Observations.
"""


@configclass
class StagedCommandObsCfg(ObsGroup):
    """Goal ``(x, y, z, sin, cos)``; stairs, flat, and pyramid goals share one encoding."""

    command = MirrorObsTerm(func=style_mdp.mode_command, mirror=style_mdp.mirror_mode_command)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class StagedCriticCfg(CriticHistoryCfg):
    """Perception critic plus the mode and the goal-conditioned reference state (zero outside stairs imitation)."""

    env_mode = MirrorObsTerm(func=mdp.generated_commands, mirror=mirror_identity, params={"command_name": "env_mode"})
    imitation_mode = MirrorObsTerm(
        func=ModeGated,
        mirror=mirror_identity,
        params=_gated(motion_observations.imitation_mode, {"command_name": "motion"}, (STAIRS,)),
    )
    reference_joint_pos = MirrorObsTerm(
        func=ModeGated,
        mirror=symmetry.mirror_g1_joints,
        mirror_params={"joint_names": ACTIVE_JOINTS},
        params=_gated(
            motion_observations.reference_joint_pos,
            {
                "command_name": "motion",
                "asset_cfg": SceneEntityCfg("robot", joint_names=ACTIVE_JOINTS, preserve_order=True),
            },
            (STAIRS,),
        ),
    )
    reference_phase = MirrorObsTerm(
        func=ModeGated,
        mirror=mirror_identity,
        params=_gated(motion_observations.reference_phase, {"command_name": "motion"}, (STAIRS,)),
    )
    reference_anchor_pos_b = MirrorObsTerm(
        func=ModeGated,
        mirror=mirror_vec3,
        params=_gated(motion_observations.reference_anchor_pos_b, {"command_name": "motion"}, (STAIRS,)),
    )
    reference_anchor_ori_b = MirrorObsTerm(
        func=ModeGated,
        mirror=mirror_rotation_6d,
        params=_gated(motion_observations.reference_anchor_ori_b, {"command_name": "motion"}, (STAIRS,)),
    )


@configclass
class G1StagedObservationsCfg(pyramid_cfg.G1PerceptionObservationsCfg):
    """Perception observation groups with the shared command and the mode-aware critic."""

    critic: StagedCriticCfg = StagedCriticCfg()
    command: StagedCommandObsCfg = StagedCommandObsCfg()


"""
Rewards and terminations: each source task's terms, gated by mode.
"""

G1StagedRewardsCfg = _make_cfg(
    "G1StagedRewardsCfg",
    "Waypoint and gait rewards on flat tiles, goal-conditioned rewards on the reference pyramid, motion-style on "
    "pyramids; each mode has its own copy of the terms.",
    {
        # the motion-style task's goal and gait terms, without the style reward
        **_gated_terms(pyramid_cfg.G1RewardsCfg(), "flat", (FLAT,), RewTerm, skip=("motion_style",)),
        **_gated_terms(stairs_cfg.G1RewardsCfg(), "stairs", (STAIRS,), RewTerm),
        **_gated_terms(pyramid_cfg.G1RewardsCfg(), "pyramid", (PYRAMID,), RewTerm),
    },
)

_SHARED_TERMINATIONS = ("time_out", "terrain_out_of_bounds")
G1StagedTerminationsCfg = _make_cfg(
    "G1StagedTerminationsCfg",
    "Shared time-outs and bounds, and each source task's failure terms gated by mode.",
    {
        "time_out": DoneTerm(func=mdp.time_out, time_out=True),
        "terrain_out_of_bounds": DoneTerm(func=mdp.terrain_out_of_bounds),
        # flat waypoint and reference climbing episodes last 10 s; pyramid episodes run the full 20 s
        "flat_stairs_time_out": DoneTerm(
            func=ModeGated,
            params=_gated(style_mdp.mode_time_out, {"max_episode_length_s": 10.0}, (FLAT, STAIRS)),
            time_out=True,
        ),
        **_gated_terms(flat_cfg.G1TerminationsCfg(), "flat", (FLAT,), DoneTerm, skip=_SHARED_TERMINATIONS),
        **_gated_terms(stairs_cfg.G1TerminationsCfg(), "stairs", (STAIRS,), DoneTerm, skip=_SHARED_TERMINATIONS),
        **_gated_terms(pyramid_cfg.G1TerminationsCfg(), "pyramid", (PYRAMID,), DoneTerm, skip=_SHARED_TERMINATIONS),
    },
)


"""
Events and curriculum.
"""


@configclass
class G1StagedEventCfg(pyramid_cfg.G1EventCfg):
    """Perception startup randomization; per-mode resets after the mode schedule, which must reset first."""

    # replaces the first reset term so that it runs before every other reset event
    reset_base = EventTerm(func=style_mdp.assign_env_modes, mode="reset")
    reset_robot_joints = EventTerm(
        func=ModeGatedEvent,
        mode="reset",
        params=_gated(
            mdp.reset_joints_by_scale, {"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)}, (FLAT, PYRAMID)
        ),
    )
    push_robot = EventTerm(
        func=ModeGatedEvent,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params=_gated(
            mdp.push_by_setting_velocity, {"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}}, (FLAT, PYRAMID)
        ),
    )
    stairs_reset_joints = EventTerm(
        func=ModeGatedEvent,
        mode="reset",
        params=_gated(reset_goal_joint_state, {"command_name": "motion", "position_range": (-0.1, 0.1)}, (STAIRS,)),
    )
    stairs_reset_root = EventTerm(
        func=ModeGatedEvent, mode="reset", params=_gated(reset_goal_root_state, {"command_name": "motion"}, (STAIRS,))
    )
    # flat and pyramid: start at rest on the ring 2.9-3.9 m from the tile center, facing the waypoint
    waypoint_reset_root = EventTerm(
        func=ModeGatedEvent,
        mode="reset",
        params=_gated(waypoint_mdp.reset_root_at_waypoint_start, {"command_name": "waypoint"}, (FLAT, PYRAMID)),
    )


def _ramp_weight(term_name: str, weight_0: float, weight_1: float) -> CurrTerm:
    return CurrTerm(
        func=g1_mdp.ramp_reward_weight,
        params={
            "term_name": term_name,
            "weight_0": weight_0,
            "weight_1": weight_1,
            "step_0": 0,
            "step_1": 15000 * STEPS_PER_ITERATION,
        },
    )


@configclass
class G1StagedCurriculumCfg:
    """Mode logging, the pyramid curriculum, and the gait weight ramps on each mode's prefixed terms."""

    env_modes = CurrTerm(func=style_mdp.env_mode_stats)
    # pyramid: promote after reaching the landing, demote after ending over half the start distance away
    pyramid_terrain_levels = CurrTerm(
        func=ModeGatedEvent,
        params=_gated(waypoint_mdp.terrain_levels_waypoint, {"command_name": "waypoint"}, (PYRAMID,)),
    )
    # flat: the motion-style task's gait weight ramps, on the flat copies of the terms
    flat_feet_air_time_weight = _ramp_weight("flat_feet_air_time", 2.0, 5.0)
    flat_foot_clearance_weight = _ramp_weight("flat_foot_clearance", 5.0, 10.0)
    # pyramid: the motion-style task's gait weight ramps
    pyramid_feet_air_time_weight = _ramp_weight("pyramid_feet_air_time", 2.0, 5.0)
    pyramid_foot_clearance_weight = _ramp_weight("pyramid_foot_clearance", 5.0, 10.0)


"""
Environments.
"""


@configclass
class G1StagedPerceptionEnvCfg(G1RoughPerceptionEnvCfg):
    """Flat waypoints and reference stair climbing until 3000 iterations, then motion-style pyramid goals."""

    scene: G1StagedSceneCfg = G1StagedSceneCfg(num_envs=128, env_spacing=2.5)
    observations: G1StagedObservationsCfg = G1StagedObservationsCfg()
    commands: G1StagedCommandsCfg = G1StagedCommandsCfg()
    rewards: G1StagedRewardsCfg = G1StagedRewardsCfg()
    terminations: G1StagedTerminationsCfg = G1StagedTerminationsCfg()
    events: G1StagedEventCfg = G1StagedEventCfg()
    curriculum: G1StagedCurriculumCfg = G1StagedCurriculumCfg()


@configclass
class G1StagedPerceptionEnvCfg_PLAY(G1StagedPerceptionEnvCfg):
    """Evaluate the final stage: every environment on a pyramid of any level, without promotion or pushes."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.commands.env_mode.switch_step = 0
        self.commands.env_mode.pyramid_max_init_level = self.scene.terrain.terrain_generator.num_rows - 1
        self.commands.waypoint.debug_vis = True
        self.curriculum.pyramid_terrain_levels = None
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.sim.visualizer_cfgs = [NewtonGLVisualizerCfg(eye=(0.0, -6.0, 3.0))]
        self.video_recorders = []
