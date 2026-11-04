# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils.configclass import configclass

from ..mdp.curriculums import task_reset_difficulty


@configclass
class G1CurriculumCfg:
    """Curriculum terms for the MDP."""

    # Every tile holds the same pyramid, so there is no terrain curriculum.
    task_resets = CurrTerm(func=task_reset_difficulty, params={"command_name": "motion"})
