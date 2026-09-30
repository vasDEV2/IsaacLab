# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable different events.

Events include anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_terrain_friction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    friction_range: tuple[float, float],
    contact_solver_name: str = "physics_callback",
) -> None:
    contact_solver = env.action_manager.get_term(contact_solver_name).contact_solver
    friction_samples = math_utils.sample_uniform(
        friction_range[0], friction_range[1], (len(env_ids),), device=env.device
    )
    contact_solver.update_friction_params(env_ids, friction_samples, friction_samples)
    if "log" in env.extras.keys():
        if "Events/terrain_friction" in env.extras["log"].keys():
            env.extras["log"]["Events/terrain_friction"] = friction_samples.mean()


def randomize_terrain_stiffness(
    env: ManagerBasedEnv,
    env_ids: Sequence[int],
    stiffness_range: tuple[float, float],
    contact_solver_name: str = "physics_callback",
) -> None:
    # extract the used quantities (to enable type-hinting)
    contact_solver = env.action_manager.get_term(contact_solver_name).contact_solver
    stiffness_samples = math_utils.sample_uniform(
        stiffness_range[0], stiffness_range[1], (len(env_ids),), device=env.device
    )
    contact_solver.randomize_ground_stiffness(env_ids, stiffness_samples)
    if "log" in env.extras.keys():
        if "Events/terrain_stiffness" in env.extras["log"].keys():
            env.extras["log"]["Events/terrain_stiffness"] = stiffness_samples.mean()


def randomize_cone_model_terrain_stiffness(
    env: ManagerBasedEnv,
    env_ids: Sequence[int],
    sigma_flat_range: tuple[float, float],
    sigma_cone_range: tuple[float, float],
    contact_solver_name: str = "physics_callback",
) -> None:
    # extract the used quantities (to enable type-hinting)
    contact_solver = env.action_manager.get_term(contact_solver_name).contact_solver  # type: ignore
    sigma_flat_samples = math_utils.sample_uniform(
        sigma_flat_range[0], sigma_flat_range[1], (len(env_ids),), device=env.device
    )
    sigma_cone_samples = math_utils.sample_uniform(
        sigma_cone_range[0], sigma_cone_range[1], (len(env_ids),), device=env.device
    )
    contact_solver.randomize_ground_stiffness(env_ids, sigma_flat_samples, sigma_cone_samples)
    if "log" in env.extras.keys():
        if "Events/terrain_stiffness" in env.extras["log"].keys():
            env.extras["log"]["Events/sigma_flat"] = sigma_flat_samples.mean()
        if "Events/sigma_cone" in env.extras["log"].keys():
            env.extras["log"]["Events/sigma_cone"] = sigma_cone_samples.mean()


def randomize_material_density(
    env: ManagerBasedEnv,
    env_ids: Sequence[int],
    packing_ratio_range: tuple[float, float],
    bulk_density_range: tuple[float, float],
    contact_solver_name: str = "physics_callback",
) -> None:
    # extract the used quantities (to enable type-hinting)
    contact_solver = env.action_manager.get_term(contact_solver_name).contact_solver
    packing_ratio = math_utils.sample_uniform(
        packing_ratio_range[0], packing_ratio_range[1], (len(env_ids),), device=env.device
    )
    bulk_density = math_utils.sample_uniform(
        bulk_density_range[0], bulk_density_range[1], (len(env_ids),), device=env.device
    )
    contact_solver.update_material_density(env_ids, packing_ratio, bulk_density)
    if "log" in env.extras.keys():
        if "Events/packing_ratio" in env.extras["log"].keys():
            env.extras["log"]["Events/packing_ratio"] = packing_ratio.mean()
        if "Events/bulk_density" in env.extras["log"].keys():
            env.extras["log"]["Events/bulk_density"] = bulk_density.mean()


def sample_terrain_property(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    friction_range: tuple[float, float],
    stiffness_range: tuple[float, float],
    packing_ratio_range: tuple[float, float],
    bulk_density_range: tuple[float, float],
    bin_size: float = 4.0,
    max_bins: int = 10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_solver_name: str = "physics_callback",
) -> None:
    """Set terrain properties (friction, stiffness, material density) based on robot x position.

    The terrain is divided into bins of ``bin_size`` metres along the x axis. The bin index is used
    to compute a ratio ``bin / max_bins`` which linearly interpolates each parameter from its upper
    bound (near x=0) down to its lower bound (at ``max_bins * bin_size`` metres away).

    The interpolation formula is:
        parameter = parameter_ub + (parameter_lb - parameter_ub) * ratio
    """
    # -- get per-env x position
    asset = env.scene[asset_cfg.name]
    x_pos = asset.data.root_pos_w.torch[env_ids, 0]

    # -- compute bin index and ratio, clamped to [0, max_bins]
    bin_idx = torch.floor(x_pos.abs() / bin_size).clamp(min=0, max=max_bins)
    # ratio = 1 - (bin_idx / max_bins * 2 - 1).abs()
    ratio = bin_idx / max_bins  # linearly decrease from 1 to 0 as bin_idx goes from 0 to max_bins

    # -- interpolate parameters: ub at ratio=0, lb at ratio=1
    friction_lb, friction_ub = friction_range
    stiffness_lb, stiffness_ub = stiffness_range
    packing_lb, packing_ub = packing_ratio_range
    density_lb, density_ub = bulk_density_range

    friction = friction_ub + (friction_lb - friction_ub) * ratio
    stiffness = stiffness_ub + (stiffness_lb - stiffness_ub) * ratio
    packing_ratio = packing_ub + (packing_lb - packing_ub) * ratio
    bulk_density = density_ub + (density_lb - density_ub) * ratio

    # -- apply to contact solver
    contact_solver = env.action_manager.get_term(contact_solver_name).contact_solver
    contact_solver.update_friction_params(env_ids, friction, friction)
    contact_solver.randomize_ground_stiffness(env_ids, stiffness)
    contact_solver.update_material_density(env_ids, packing_ratio, bulk_density)

    # -- logging
    if "log" in env.extras:
        log = env.extras["log"]
        if "Events/terrain_friction" in log:
            log["Events/terrain_friction"] = friction.mean()
        if "Events/terrain_stiffness" in log:
            log["Events/terrain_stiffness"] = stiffness.mean()
        if "Events/packing_ratio" in log:
            log["Events/packing_ratio"] = packing_ratio.mean()
        if "Events/bulk_density" in log:
            log["Events/bulk_density"] = bulk_density.mean()


def sample_terrain_property_linear(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    friction_range: tuple[float, float],
    stiffness_range: tuple[float, float],
    packing_ratio_range: tuple[float, float],
    bulk_density_range: tuple[float, float],
    contact_solver_name: str = "physics_callback",
) -> None:
    """Set terrain properties (friction, stiffness, material density) based on robot x position.

    The terrain is divided into bins of ``bin_size`` metres along the x axis. The bin index is used
    to compute a ratio ``bin / max_bins`` which linearly interpolates each parameter from its upper
    bound (near x=0) down to its lower bound (at ``max_bins * bin_size`` metres away).

    The interpolation formula is:
        parameter = parameter_ub + (parameter_lb - parameter_ub) * ratio
    """
    ratio = env.episode_length_buf / env.max_episode_length

    # -- interpolate parameters: ub at ratio=0, lb at ratio=1
    friction_lb, friction_ub = friction_range
    stiffness_lb, stiffness_ub = stiffness_range
    packing_lb, packing_ub = packing_ratio_range
    density_lb, density_ub = bulk_density_range

    friction = friction_ub + (friction_lb - friction_ub) * ratio
    stiffness = stiffness_ub + (stiffness_lb - stiffness_ub) * ratio
    packing_ratio = packing_ub + (packing_lb - packing_ub) * ratio
    bulk_density = density_ub + (density_lb - density_ub) * ratio

    # -- apply to contact solver
    contact_solver = env.action_manager.get_term(contact_solver_name).contact_solver
    contact_solver.update_friction_params(env_ids, friction, friction)
    contact_solver.randomize_ground_stiffness(env_ids, stiffness)
    contact_solver.update_material_density(env_ids, packing_ratio, bulk_density)

    # -- logging
    if "log" in env.extras:
        log = env.extras["log"]
        if "Events/terrain_friction" in log:
            log["Events/terrain_friction"] = friction.mean()
        if "Events/terrain_stiffness" in log:
            log["Events/terrain_stiffness"] = stiffness.mean()
        if "Events/packing_ratio" in log:
            log["Events/packing_ratio"] = packing_ratio.mean()
        if "Events/bulk_density" in log:
            log["Events/bulk_density"] = bulk_density.mean()
