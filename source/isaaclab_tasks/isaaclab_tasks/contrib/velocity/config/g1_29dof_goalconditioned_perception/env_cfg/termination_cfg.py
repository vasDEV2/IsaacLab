# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.mimic.mdp as mimic_mdp
import isaaclab_tasks.core.velocity.mdp as mdp

from ..mdp import terminations as goal_terminations
from .motion_cfg import FOOT_BODY_NAMES


@configclass
class G1TerminationsCfg:
    """Reference terminations for imitation episodes and physical failures for both episode types."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Imitation episodes end as a time-out on the clip's last frame; this is not task success.
    motion_end = DoneTerm(func=goal_terminations.motion_end, params={"command_name": "motion"}, time_out=True)

    # -- imitation episodes: leaving the reference
    anchor_pos = DoneTerm(
        func=goal_terminations.bad_anchor_height, params={"command_name": "motion", "threshold": 0.25}
    )
    anchor_ori = DoneTerm(func=goal_terminations.bad_anchor_tilt, params={"command_name": "motion", "threshold": 0.8})
    ee_body_pos = DoneTerm(
        func=goal_terminations.bad_body_height,
        params={"command_name": "motion", "threshold": 0.25, "body_names": FOOT_BODY_NAMES},
    )

    # -- both episode types: falls, measured from the base to the lowest foot
    base_too_low = DoneTerm(
        func=mimic_mdp.root_height_below_minimum_adaptive,
        params={"minimum_height": 0.2, "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link")},
    )
    fallen = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.2})
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["pelvis", "torso_link"]), "threshold": 1.0},
    )
    extreme_action = DoneTerm(
        func=mimic_mdp.ExtremeJointPositionAction,
        params={"action_name": "joint_pos", "torque_fraction": 5.0, "asset_cfg": SceneEntityCfg("robot")},
    )
