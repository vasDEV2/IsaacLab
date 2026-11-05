# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Goal-conditioned stair climbing that starts with a walk from a random spawn to the reference clip's start.

* Until 1500 iterations (:data:`~.mdp.WALK`): every episode spawns at a random place in front of the pyramid and
  walks to the first reference root pose, with the ``g1_29dof_perception_waypoint`` rewards and terminations.
* From then on, episodes spawn the same way and walk to the same waypoint without reward (:data:`~.mdp.APPROACH`).
  Once there (:data:`~.mdp.CLIMB`), a ``g1_29dof_goalconditioned_perception`` imitation episode starts on the clip's
  first frame with that task's imitation rewards and terminations. There are no task episodes and no stair
  curriculum: every tile holds the clips' own pyramid.

Each source task's terms are wrapped in
:class:`~isaaclab_tasks.contrib.velocity.config.g1_29dof_perception_motion_style.mdp.ModeGated` under a ``walk_`` or
``climb_`` prefix, gated by the motion command's phase.
"""

import copy

from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_contrib.mdp import MirrorObservationTermCfg as MirrorObsTerm
from isaaclab_contrib.mdp import mirror_identity

import isaaclab_tasks.contrib.velocity.config.g1_29dof_perception_waypoint.mdp as waypoint_mdp
import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp as g1_mdp
import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab_tasks.contrib.velocity.config.g1_29dof_goalconditioned_perception import env_cfg as climb_cfg
from isaaclab_tasks.contrib.velocity.config.g1_29dof_goalconditioned_perception.env_cfg.motion_cfg import (
    FOOT_BODY_NAMES,
    STAIR_MOTION_FILES,
    TRACKED_BODY_NAMES,
)
from isaaclab_tasks.contrib.velocity.config.g1_29dof_goalconditioned_perception.env_cfg.observation_cfg import (
    CriticCfg,
)
from isaaclab_tasks.contrib.velocity.config.g1_29dof_goalconditioned_perception.mdp.terrains import (
    MotionStairsTerrainCfg,
)
from isaaclab_tasks.contrib.velocity.config.g1_29dof_goalconditioned_perception.rough_env_cfg import (
    G1RoughGoalConditionedPerceptionEnvCfg,
)
from isaaclab_tasks.contrib.velocity.config.g1_29dof_perception import env_cfg as walk_cfg
from isaaclab_tasks.contrib.velocity.config.g1_29dof_perception_motion_style.mdp import ModeGated
from isaaclab_tasks.contrib.velocity.config.g1_29dof_perception_waypoint.env_cfg import (
    G1WaypointRewardsCfg,
    WaypointCommandObsCfg,
    WaypointPolicyCfg,
)

from . import mdp as approach_mdp
from .mdp import APPROACH, CLIMB, WALK

STEPS_PER_ITERATION = 24
"""Must match the runner's ``num_steps_per_env``; the schedule is counted in control steps."""

CLIMB_ITERATION = 1500
"""Iteration from which new episodes approach without reward and then climb."""


def _gated(func, func_params: dict, phases: tuple[int, ...]) -> dict:
    return {"func": func, "func_params": func_params, "modes": phases, "mode_command": "motion"}


def _gated_terms(source, prefix: str, phases: tuple[int, ...], term_type: type, names=None) -> dict:
    """Copy ``term_type`` terms of a source config (all, or ``names``), gated by phase and renamed ``prefix_name``.

    Waypoint terms of the source read the motion command, which provides the waypoint interface.
    """
    terms = {}
    for name, term in vars(source).items():
        if not isinstance(term, term_type) or (names is not None and name not in names):
            continue
        params = copy.deepcopy(term.params)
        if params.get("command_name") == "waypoint":
            params["command_name"] = "motion"
        terms[f"{prefix}_{name}"] = term.replace(func=ModeGated, params=_gated(term.func, params, phases))
    return terms


def _make_cfg(name: str, doc: str, terms: dict) -> type:
    namespace = {"__annotations__": {key: type(value) for key, value in terms.items()}, **terms}
    namespace.update({"__doc__": doc, "__module__": __name__})
    return configclass(type(name, (), namespace))


"""
Commands and observations.
"""


@configclass
class G1ApproachCommandsCfg:
    """The goal-conditioned command with its approach phases, and the walking intent toward its waypoint."""

    motion = approach_mdp.ApproachMotionCommandCfg(
        asset_name="robot",
        motion_files=STAIR_MOTION_FILES,
        body_names=TRACKED_BODY_NAMES,
        anchor_body_name="torso_link",
        foot_body_names=FOOT_BODY_NAMES,
        approach_steps=CLIMB_ITERATION * STEPS_PER_ITERATION,
        # every climb imitates the clip; with no task terrain, episodes stay on their tile of the clips' pyramid
        imitation_probability=1.0,
        task_terrain=None,
    )
    # gait terms of the waypoint rewards gate on this velocity intent toward the waypoint
    waypoint_gait = waypoint_mdp.WaypointGaitCommandCfg(waypoint_command_name="motion")


@configclass
class ApproachPolicyCfg(WaypointPolicyCfg):
    """The waypoint task's actor proprioception, with the root position measured from the pyramid's tile center."""

    root_position = MirrorObsTerm(
        func=approach_mdp.root_pos_tile_b,
        mirror=waypoint_mdp.mirror_waypoint,
        noise=Unoise(n_min=-0.02, n_max=0.02),
        params={"command_name": "motion"},
    )


@configclass
class ApproachCommandObsCfg(WaypointCommandObsCfg):
    """The waypoint task's command: the active target relative to the root, in the root yaw frame [m]."""

    waypoint = MirrorObsTerm(
        func=mdp.generated_commands, mirror=waypoint_mdp.mirror_waypoint, params={"command_name": "motion"}
    )


@configclass
class ApproachCriticCfg(CriticCfg):
    """Goal-conditioned critic plus the phase, which decides the reward."""

    approach_phase = MirrorObsTerm(
        func=approach_mdp.approach_phase, mirror=mirror_identity, params={"command_name": "motion"}
    )


@configclass
class G1ApproachObservationsCfg(climb_cfg.G1PerceptionObservationsCfg):
    """The waypoint task's actor inputs; the command is the waypoint before climbing, the destination after."""

    policy: ApproachPolicyCfg = ApproachPolicyCfg()
    command: ApproachCommandObsCfg = ApproachCommandObsCfg()
    critic: ApproachCriticCfg = ApproachCriticCfg()


"""
Rewards and terminations: each source task's terms, gated by phase.
"""

_TASK_REWARDS = (
    "task_motion_style",
    "goal_position",
    "goal_position_fine",
    "goal_heading",
    "goal_settle",
    "goal_joint_deviation",
)
"""Goal-conditioned rewards of task episodes only, which this task does not run."""
_CLIMB_REWARDS = tuple(name for name in vars(climb_cfg.G1RewardsCfg()) if name not in _TASK_REWARDS)

G1ApproachRewardsCfg = _make_cfg(
    "G1ApproachRewardsCfg",
    "Waypoint task rewards while walking, goal-conditioned imitation rewards while climbing, none while approaching.",
    {
        **_gated_terms(G1WaypointRewardsCfg(), "walk", (WALK,), RewTerm),
        **_gated_terms(climb_cfg.G1RewardsCfg(), "climb", (CLIMB,), RewTerm, _CLIMB_REWARDS),
    },
)

_CLIMB_FAILURES = ("anchor_pos", "anchor_ori", "ee_body_pos")
_FALLS = ("base_too_low", "fallen", "base_contact", "extreme_action")
G1ApproachTerminationsCfg = _make_cfg(
    "G1ApproachTerminationsCfg",
    "Shared time-out and bounds, waypoint task failures while walking, goal-conditioned ones after.",
    {
        "time_out": DoneTerm(func=mdp.time_out, time_out=True),
        "terrain_out_of_bounds": DoneTerm(func=mdp.terrain_out_of_bounds),
        **_gated_terms(walk_cfg.G1TerminationsCfg(), "walk", (WALK,), DoneTerm, ("base_too_low", "bad_orientation")),
        # approach: falls only, and a time-out for not reaching the waypoint
        "approach_time_out": DoneTerm(
            func=approach_mdp.approach_time_out, params={"command_name": "motion", "max_time_s": 10.0}, time_out=True
        ),
        **_gated_terms(climb_cfg.G1TerminationsCfg(), "climb", (APPROACH, CLIMB), DoneTerm, _FALLS),
        # climb: the imitation episode ends, as a time-out, on the clip's last frame
        **_gated_terms(climb_cfg.G1TerminationsCfg(), "climb", (CLIMB,), DoneTerm, ("motion_end", *_CLIMB_FAILURES)),
    },
)


"""
Events and curriculum.
"""


@configclass
class G1ApproachEventCfg(climb_cfg.G1EventCfg):
    """Goal-conditioned startup randomization; every episode starts at rest at a random spawn."""

    reset_joints = None
    reset_base = EventTerm(
        func=approach_mdp.reset_at_spawn, mode="reset", params={"command_name": "motion", "position_range": (-0.1, 0.1)}
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
class G1ApproachCurriculumCfg:
    """Phase logging and the waypoint task's gait weight ramps; the stairs have no curriculum."""

    approach = CurrTerm(func=approach_mdp.approach_stats, params={"command_name": "motion"})
    walk_feet_air_time_weight = _ramp_weight("walk_feet_air_time", 2.0, 5.0)
    walk_foot_clearance_weight = _ramp_weight("walk_foot_clearance", 5.0, 10.0)


"""
Environments.
"""


def _robot_camera(headless: bool) -> NewtonGLVisualizerCfg:
    """Newton GL camera that follows env 0's robot; the default looks at the world origin, far from every tile."""
    return NewtonGLVisualizerCfg(
        headless=headless, origin_type="asset", origin_track_path="robot", eye=(-3.0, -3.0, 2.0), lookat=(0.0, 0.0, 0.3)
    )


@configclass
class G1RoughGoalConditionedApproachPerceptionEnvCfg(G1RoughGoalConditionedPerceptionEnvCfg):
    """Walk from a random spawn to the reference clip's start, then climb the pyramid as the goal-conditioned task."""

    observations: G1ApproachObservationsCfg = G1ApproachObservationsCfg()
    commands: G1ApproachCommandsCfg = G1ApproachCommandsCfg()
    rewards: G1ApproachRewardsCfg = G1ApproachRewardsCfg()
    terminations: G1ApproachTerminationsCfg = G1ApproachTerminationsCfg()
    events: G1ApproachEventCfg = G1ApproachEventCfg()
    curriculum: G1ApproachCurriculumCfg = G1ApproachCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # up to 10 s of approach and the 7.5 s clip; walking episodes run the full 20 s, as in the waypoint task
        self.episode_length_s = 20.0
        # every tile holds the clips' own pyramid at its recorded height, with no stair height levels
        generator = self.scene.terrain.terrain_generator
        generator.sub_terrains = {
            "motion_stairs": MotionStairsTerrainCfg(size=(9.0, 9.0), proportion=1.0, motion_file=STAIR_MOTION_FILES[0])
        }
        generator.curriculum = False
        self.scene.terrain.max_init_terrain_level = None
        # the camera follows env 0's robot (the default looks at the world origin, between tiles far from it);
        # a window opens if the goal-conditioned config enables one, otherwise it only serves recordings
        configured = self.sim.visualizer_cfgs
        configured = configured if isinstance(configured, list) else [configured]
        window = any(cfg.visualizer_type == "newton_gl" and not cfg.headless for cfg in configured)
        self.sim.visualizer_cfgs = [_robot_camera(headless=not window)]


@configclass
class G1RoughGoalConditionedApproachPerceptionEnvCfg_PLAY(G1RoughGoalConditionedApproachPerceptionEnvCfg):
    """Approach from a random spawn, then imitate the clip."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.observations.policy.enable_corruption = False
        self.commands.motion.approach_steps = 0
        self.commands.motion.debug_vis = True
        self.events.reset_base.params["position_range"] = (0.0, 0.0)
        self.sim.visualizer_cfgs = [_robot_camera(headless=False)]
        self.video_recorders = []
