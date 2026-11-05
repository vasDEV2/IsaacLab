# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils.configclass import configclass

from ..mdp.curriculums import task_terrain_levels


@configclass
class G1CurriculumCfg:
    """Curriculum terms for the MDP."""

    # Task episodes climb taller stairs after each success; imitation tiles keep the clips' pyramid.
    task_terrain_levels = CurrTerm(func=task_terrain_levels, params={"command_name": "motion"})
