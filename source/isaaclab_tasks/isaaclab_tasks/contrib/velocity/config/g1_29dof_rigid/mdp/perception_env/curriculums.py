# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_friction_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    friction_range: tuple[float, float] = (0.1, 1.0),
    contact_solver_name: str = "physics_callback",
) -> torch.Tensor:
    """Curriculum based on the friction of the terrain."""
    terrain: TerrainImporter = env.scene.terrain

    contact_solver = env.action_manager.get_term(contact_solver_name).contact_solver

    terrain_level = terrain.terrain_levels.float()[env_ids]
    ratio = terrain_level / (terrain.max_terrain_level - 1)
    friction_mean = friction_range[1] + (friction_range[0] - friction_range[1]) * ratio
    scale_offset = 0.1
    friction_scale = math_utils.sample_uniform(
        1.0 - scale_offset, 1.0 + scale_offset, (len(env_ids),), device=env.device
    )
    friction_samples = friction_mean * friction_scale
    friction_samples = torch.clamp(friction_samples, friction_range[0], friction_range[1])

    contact_solver.update_friction_params(env_ids, friction_samples, friction_samples)

    return torch.mean(friction_mean)


def terrain_stiffness_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    stiffness_range: tuple[float, float] = (0.1, 1.0),
    contact_solver_name: str = "physics_callback",
) -> torch.Tensor:
    """Curriculum based on the stiffness of the terrain."""
    terrain: TerrainImporter = env.scene.terrain

    contact_solver = env.action_manager.get_term(contact_solver_name).contact_solver

    terrain_level = terrain.terrain_levels.float()[env_ids]
    ratio = terrain_level / (terrain.max_terrain_level - 1)
    stiffness_mean = stiffness_range[1] + (stiffness_range[0] - stiffness_range[1]) * ratio
    scale_offset = 0.1
    stiffness_scale = math_utils.sample_uniform(
        1.0 - scale_offset, 1.0 + scale_offset, (len(env_ids),), device=env.device
    )
    stiffness_samples = stiffness_mean * stiffness_scale
    stiffness_samples = torch.clamp(stiffness_samples, stiffness_range[0], stiffness_range[1])

    contact_solver.randomize_ground_stiffness(env_ids, stiffness_samples)

    return torch.mean(stiffness_mean)


def terrain_density_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    density_range: tuple[float, float] = (1000.0, 3000.0),
    packing_ratio_range: tuple[float, float] = (0.5, 1.0),
    contact_solver_name: str = "physics_callback",
) -> torch.Tensor:
    """Curriculum based on the density and packing ratio of the terrain."""
    terrain: TerrainImporter = env.scene.terrain

    contact_solver = env.action_manager.get_term(contact_solver_name).contact_solver

    terrain_level = terrain.terrain_levels.float()[env_ids]
    ratio = terrain_level / (terrain.max_terrain_level - 1)
    density_mean = density_range[1] + (density_range[0] - density_range[1]) * ratio
    packing_ratio_mean = packing_ratio_range[1] + (packing_ratio_range[0] - packing_ratio_range[1]) * ratio
    scale_offset = 0.1
    scale = math_utils.sample_uniform(1.0 - scale_offset, 1.0 + scale_offset, (len(env_ids),), device=env.device)
    density_samples = density_mean * scale
    density_samples = torch.clamp(density_samples, density_range[0], density_range[1])
    packing_ratio_samples = packing_ratio_mean * scale
    packing_ratio_samples = torch.clamp(packing_ratio_samples, packing_ratio_range[0], packing_ratio_range[1])

    contact_solver.update_material_density(env_ids, packing_ratio_samples, density_samples)

    return torch.mean(density_mean * packing_ratio_mean)
