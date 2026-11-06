# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The waypoint task on the motion clips' pyramid: reach the clip's start, then imitate the clip.

* Episodes reset before :data:`IMITATION_ITERATION` (:data:`~.mdp.WALK`) walk from a spawn behind the pyramid's base
  to the first root pose of a randomly chosen clip, with the waypoint task's rewards.
* Later episodes (:data:`~.mdp.APPROACH`) get the same waypoint without waypoint rewards. On reaching it
  (:data:`~.mdp.IMITATE`), the clip plays from its first frame with the goal-conditioned task's imitation rewards
  and terminations, and the episode ends on the clip's last frame.

There is no stair curriculum and no imitation sampling schedule.
"""

from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.mdp import MirrorObservationTermCfg as MirrorObsTerm

from isaaclab_tasks.contrib.velocity.config.g1_29dof_goalconditioned_perception.env_cfg.motion_cfg import (
    FOOT_BODY_NAMES,
    STAIR_MOTION_FILES,
    TRACKED_BODY_NAMES,
)
from isaaclab_tasks.contrib.velocity.config.g1_29dof_goalconditioned_perception.mdp import rewards as goal_rewards
from isaaclab_tasks.contrib.velocity.config.g1_29dof_goalconditioned_perception.mdp import (
    terminations as goal_terminations,
)
from isaaclab_tasks.contrib.velocity.config.g1_29dof_goalconditioned_perception.mdp.terrains import (
    MotionStairsTerrainCfg,
)
from isaaclab_tasks.contrib.velocity.config.g1_29dof_perception_motion_style.mdp import ModeGated

from ..g1_29dof_perception.env_cfg import G1TerminationsCfg
from ..g1_29dof_perception_waypoint import mdp as waypoint_mdp
from ..g1_29dof_perception_waypoint.env_cfg import (
    G1RoughPerceptionWaypointEnvCfg,
    G1WaypointCurriculumCfg,
    G1WaypointObservationsCfg,
    G1WaypointRewardsCfg,
    WaypointCommandObsCfg,
    WaypointPolicyCfg,
)
from . import mdp as mimic_mdp
from .mdp import APPROACH, IMITATE, WALK

STEPS_PER_ITERATION = 24
"""Must match the runner's ``num_steps_per_env``; the schedule is counted in control steps."""

IMITATION_ITERATION = 3000
"""Iteration from which new episodes approach without waypoint rewards and then imitate."""

TERMINATION_DECAY_ITERATION = 2000
"""Iterations after :data:`IMITATION_ITERATION` over which the reference termination thresholds tighten to their
final values."""

BASE_REACHED_BONUS = 0.0
"""Return paid once per approach episode on first reaching the base waypoint (not in walk)."""


@configclass
class G1WaypointMimicCommandsCfg:
    """The pyramid-base waypoint with its clip, and the walking intent toward it for gait terms."""

    waypoint = mimic_mdp.PyramidMimicCommandCfg(
        debug_vis=True,
        motion_files=STAIR_MOTION_FILES,
        body_names=TRACKED_BODY_NAMES,
        imitation_start_step=IMITATION_ITERATION * STEPS_PER_ITERATION,
    )
    waypoint_gait = waypoint_mdp.WaypointGaitCommandCfg(waypoint_command_name="waypoint")


@configclass
class WaypointMimicCommandObsCfg(WaypointCommandObsCfg):
    """Horizontal waypoint error in the root yaw frame: the policy's only positional input.

    There is no heading target, height error, or imitation state, so the policy cannot tell walking from imitating.
    """

    waypoint = MirrorObsTerm(
        func=mimic_mdp.waypoint_xy_b, mirror=mimic_mdp.mirror_waypoint_xy, params={"command_name": "waypoint"}
    )


@configclass
class WaypointMimicPolicyCfg(WaypointPolicyCfg):
    """Perception proprioception, without the root position on the tile."""

    root_position = None


@configclass
class G1WaypointMimicObservationsCfg(G1WaypointObservationsCfg):
    """The perception observations with the horizontal waypoint error in ``command`` and no root position."""

    policy: WaypointMimicPolicyCfg = WaypointMimicPolicyCfg()
    command: WaypointMimicCommandObsCfg = WaypointMimicCommandObsCfg()


_REGULARIZATION = (
    "termination_penalty",
    "undesired_contacts",
    "action_rate_l2_lower_body",
    "action_rate_l2_upper_body",
    "energy",
    "dof_vel_l2",
    "dof_acc_l2",
    "dof_pos_limits",
)
"""Waypoint task terms kept in every mode, including imitation."""

_IMITATION = (
    "motion_global_anchor_pos",
    "motion_global_anchor_ori",
    "motion_global_anchor_lin_vel",
    "motion_global_anchor_ang_vel",
    "motion_body_pos",
    "motion_body_ori",
    "motion_feet_lin_vel",
)
"""Goal-conditioned imitation terms; each is zero outside :data:`~.mdp.IMITATE`."""


@configclass
class G1WaypointMimicRewardsCfg(G1WaypointRewardsCfg):
    """Waypoint rewards while walking, gait terms until imitation, and pure imitation once the clip plays.

    * ``waypoint_*``: :data:`~.mdp.WALK` only.
    * Imitation terms (the goal-conditioned task's weights): :data:`~.mdp.IMITATE` only.
    * ``top_*``: the waypoint position rewards toward the top waypoint, :data:`~.mdp.IMITATE` only.
    * Regularization (:data:`_REGULARIZATION`): every mode.
    * ``backward_velocity`` and every other waypoint task penalty (negative weight): :data:`~.mdp.WALK` and
      :data:`~.mdp.APPROACH`.
    * Every other waypoint task reward (positive weight; gait, posture, and style): :data:`~.mdp.WALK` only, so
      :data:`~.mdp.APPROACH` has no positive reward until the robot reaches the base and imitation starts.
    * ``base_reached``: :data:`BASE_REACHED_BONUS`, once per :data:`~.mdp.APPROACH` episode on first reaching the
      base waypoint (paid on the next step, when the episode already imitates; never in :data:`~.mdp.WALK`).
      Its weight is set by the environment config, since the reward manager scales every term by the step time.
    """

    base_reached = RewTerm(
        func=mimic_mdp.base_reached_bonus, weight=BASE_REACHED_BONUS, params={"command_name": "waypoint"}
    )

    # walking backward toward the waypoint is penalized in walk and approach; the waypoint task's
    # bearing and facing-scaled progress rewards add to it in walk
    backward_velocity = RewTerm(func=mimic_mdp.backward_velocity_l2, weight=-2.0)

    # while imitating, the waypoint is the clip's end on the top landing
    top_position = RewTerm(
        func=waypoint_mdp.waypoint_position_exp, weight=4.0, params={"command_name": "waypoint", "std": 2.0}
    )
    top_position_fine = RewTerm(
        func=waypoint_mdp.waypoint_position_exp, weight=4.0, params={"command_name": "waypoint", "std": 0.5}
    )

    motion_global_anchor_pos = RewTerm(
        func=goal_rewards.motion_anchor_pos_exp, weight=2.0, params={"command_name": "waypoint", "std": 0.3}
    )
    motion_global_anchor_ori = RewTerm(
        func=goal_rewards.motion_anchor_ori_exp, weight=2.0, params={"command_name": "waypoint", "std": 0.4}
    )
    motion_global_anchor_lin_vel = RewTerm(
        func=goal_rewards.motion_anchor_lin_vel_exp, weight=1.0, params={"command_name": "waypoint", "std": 1.0}
    )
    motion_global_anchor_ang_vel = RewTerm(
        func=goal_rewards.motion_anchor_ang_vel_exp, weight=1.0, params={"command_name": "waypoint", "std": 3.14}
    )
    motion_body_pos = RewTerm(
        func=goal_rewards.motion_body_pos_exp, weight=1.0, params={"command_name": "waypoint", "std": 0.3}
    )
    motion_body_ori = RewTerm(
        func=goal_rewards.motion_body_ori_exp, weight=1.0, params={"command_name": "waypoint", "std": 0.4}
    )
    motion_feet_lin_vel = RewTerm(
        func=goal_rewards.motion_body_lin_vel_exp,
        weight=1.0,
        params={"command_name": "waypoint", "std": 1.0, "body_names": FOOT_BODY_NAMES},
    )

    def __post_init__(self):
        super().__post_init__()
        for name, term in vars(self).items():
            # the bonus is zero while imitating by construction, so it needs no gate
            skip = _REGULARIZATION + _IMITATION + ("base_reached",)
            if not isinstance(term, RewTerm) or term.func is ModeGated or name in skip:
                continue
            if name.startswith("top_"):
                modes = (IMITATE,)
            elif name.startswith("waypoint_") or term.weight > 0.0:
                # approach episodes get no positive reward before imitation
                modes = (WALK,)
            else:
                modes = (WALK, APPROACH)
            term.params = {"func": term.func, "func_params": term.params, "modes": modes, "mode_command": "waypoint"}
            term.func = ModeGated


@configclass
class G1WaypointMimicTerminationsCfg(G1TerminationsCfg):
    """Falls in every mode; the goal-conditioned task's reference terminations while imitating.

    The reference thresholds start loose at :data:`IMITATION_ITERATION` and tighten linearly to the goal-conditioned
    task's values over the next :data:`TERMINATION_DECAY_ITERATION` iterations.
    """

    # the clip has played out: a time-out, not a failure
    motion_end = DoneTerm(func=goal_terminations.motion_end, params={"command_name": "waypoint"}, time_out=True)
    anchor_pos = DoneTerm(
        func=mimic_mdp.bad_anchor_height_decaying,
        params={
            "command_name": "waypoint",
            "threshold_start": 0.6,
            "threshold": 0.25,
            "decay_steps": TERMINATION_DECAY_ITERATION * STEPS_PER_ITERATION,
        },
    )
    anchor_ori = DoneTerm(
        func=mimic_mdp.bad_anchor_tilt_decaying,
        params={
            "command_name": "waypoint",
            "threshold_start": 1.5,
            "threshold": 0.8,
            "decay_steps": TERMINATION_DECAY_ITERATION * STEPS_PER_ITERATION,
        },
    )
    ee_body_pos = DoneTerm(
        func=mimic_mdp.bad_body_height_decaying,
        params={
            "command_name": "waypoint",
            "threshold_start": 0.6,
            "threshold": 0.25,
            "decay_steps": TERMINATION_DECAY_ITERATION * STEPS_PER_ITERATION,
            "body_names": FOOT_BODY_NAMES,
        },
    )


@configclass
class G1WaypointMimicCurriculumCfg(G1WaypointCurriculumCfg):
    """The waypoint task's gait weight ramps; every tile holds the same pyramid, so there are no terrain levels."""

    terrain_levels = None


@configclass
class G1RoughPerceptionWaypointMimicEnvCfg(G1RoughPerceptionWaypointEnvCfg):
    """Reach the base of the clips' pyramid, then (after :data:`IMITATION_ITERATION`) imitate the clip up it."""

    observations: G1WaypointMimicObservationsCfg = G1WaypointMimicObservationsCfg()
    commands: G1WaypointMimicCommandsCfg = G1WaypointMimicCommandsCfg()
    rewards: G1WaypointMimicRewardsCfg = G1WaypointMimicRewardsCfg()
    terminations: G1WaypointMimicTerminationsCfg = G1WaypointMimicTerminationsCfg()
    curriculum: G1WaypointMimicCurriculumCfg = G1WaypointMimicCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # every tile holds the clips' pyramid (4.5 m base), centered; 14 m tiles leave flat ground for spawns
        # up to 4 m behind its base, with a margin to the neighbouring tile
        generator = self.scene.terrain.terrain_generator
        generator.size = (14.0, 14.0)
        generator.num_rows = 10
        generator.num_cols = 20
        generator.curriculum = False
        generator.sub_terrains = {
            "motion_pyramid": MotionStairsTerrainCfg(proportion=1.0, motion_file=STAIR_MOTION_FILES[0])
        }
        self.scene.terrain.max_init_terrain_level = None
        # the reward manager multiplies each term by the step time; pay the full bonus in one step
        self.rewards.base_reached.weight = BASE_REACHED_BONUS / (self.sim.dt * self.decimation)


@configclass
class G1RoughPerceptionWaypointMimicEnvCfg_PLAY(G1RoughPerceptionWaypointMimicEnvCfg):
    """Approach and imitate from the first step, without pushes."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.commands.waypoint.imitation_start_step = 0
        self.commands.waypoint.print_interval = 25  # every 0.5 s
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                origin_type="asset", origin_track_path="robot", eye=(-3.0, -3.0, 2.0), lookat=(0.0, 0.0, 0.3)
            )
        ]
        self.video_recorders = []
