# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-environment training modes for the staged task: flat waypoints, reference stair climbing, pyramid goals.

Before :attr:`EnvModeCommandCfg.switch_step`, each environment keeps the mode it was given at construction:
:data:`FLAT` (reach a waypoint at the center of a flat tile) or :data:`STAIRS` (the goal-conditioned imitation and
task episodes on the reference clips' pyramid). From the switch on, every environment becomes :data:`PYRAMID`
(reach the landing center of a clip-shaped pyramid with the motion-style rewards) at its next reset.

The mode decides the environment's terrain tile and, through :class:`ModeGated` and :class:`ModeGatedEvent`,
which reward, termination, observation, event, and curriculum terms act on it.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.managers import CommandTerm, CommandTermCfg, ManagerTermBase
from isaaclab.utils import math as math_utils
from isaaclab.utils.configclass import configclass
from isaaclab.utils.string import string_to_callable

from isaaclab_tasks.contrib.velocity.config.g1_29dof_goalconditioned_perception.mdp.symmetry import mirror_goal

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

FLAT, STAIRS, PYRAMID = 0, 1, 2
MODE_NAMES = ("flat", "stairs", "pyramid")


class EnvModeCommand(CommandTerm):
    """Holds each environment's mode and places it on a terrain tile of that mode.

    The command is the one-hot mode ``(flat, stairs, pyramid)``. :meth:`assign`, called by the first reset event,
    applies the schedule and moves environments whose mode changed to a random column of their sub-terrain. The
    terrain must be generated with ``curriculum=True``; flat and stairs tiles start on a random row, pyramid tiles on
    rows ``0..pyramid_max_init_level``, after which the pyramid curriculum moves them along their column.
    """

    cfg: EnvModeCommandCfg

    def __init__(self, cfg: EnvModeCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        terrain = env.scene.terrain
        generator = terrain.cfg.terrain_generator
        if generator is None or not generator.curriculum or terrain.terrain_origins is None:
            raise ValueError("EnvModeCommand needs a curriculum terrain generator.")
        # the generator assigns sub-terrains to columns by proportion (TerrainGenerator._generate_curriculum_terrains)
        names = list(generator.sub_terrains)
        proportions = np.array([generator.sub_terrains[name].proportion for name in names], dtype=float)
        cumulative = np.cumsum(proportions / proportions.sum())
        column_names = [
            names[np.min(np.where(col / generator.num_cols + 0.001 < cumulative)[0])]
            for col in range(generator.num_cols)
        ]
        self.columns: list[torch.Tensor] = []
        for mode, sub_terrain in enumerate((cfg.flat_terrain, cfg.stairs_terrain, cfg.pyramid_terrain)):
            cols = [col for col, name in enumerate(column_names) if name == sub_terrain]
            if not cols:
                raise ValueError(f"No terrain column holds sub-terrain '{sub_terrain}' for mode {MODE_NAMES[mode]}.")
            self.columns.append(torch.tensor(cols, dtype=torch.long, device=self.device))
        self.num_rows = generator.num_rows

        self.initial_mode = torch.full((self.num_envs,), FLAT, dtype=torch.long, device=self.device)
        stairs = torch.randperm(self.num_envs, device=self.device)[: round(cfg.stairs_fraction * self.num_envs)]
        self.initial_mode[stairs] = STAIRS
        self.mode = self.initial_mode.clone()
        self._placed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def __str__(self) -> str:
        msg = "EnvModeCommand:\n"
        msg += f"\tStairs fraction before the switch: {self.cfg.stairs_fraction}\n"
        msg += f"\tSwitch to pyramids at control step: {self.cfg.switch_step}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """One-hot mode ``(flat, stairs, pyramid)``. Shape is (num_envs, 3)."""
        return torch.nn.functional.one_hot(self.mode, len(MODE_NAMES)).float()

    @property
    def switched(self) -> bool:
        """Whether the schedule has reached the switch to pyramids."""
        return self._env.common_step_counter >= self.cfg.switch_step

    def mask(self, modes: Sequence[int]) -> torch.Tensor:
        """Environments whose mode is in ``modes``. Shape is (num_envs,)."""
        return torch.isin(self.mode, torch.as_tensor(modes, device=self.device))

    def assign(self, env_ids: torch.Tensor):
        """Apply the schedule to resetting environments and move those whose mode changed to a tile of it."""
        env_ids = env_ids.long()
        target = torch.full_like(env_ids, PYRAMID) if self.switched else self.initial_mode[env_ids]
        moved = (target != self.mode[env_ids]) | ~self._placed[env_ids]
        self.mode[env_ids] = target
        self._placed[env_ids] = True
        terrain = self._env.scene.terrain
        for mode in range(len(MODE_NAMES)):
            ids = env_ids[moved & (target == mode)]
            if len(ids) == 0:
                continue
            cols = self.columns[mode]
            max_row = min(self.cfg.pyramid_max_init_level, self.num_rows - 1) if mode == PYRAMID else self.num_rows - 1
            terrain.terrain_types[ids] = cols[torch.randint(len(cols), (len(ids),), device=self.device)]
            terrain.terrain_levels[ids] = torch.randint(max_row + 1, (len(ids),), device=self.device)
            terrain.env_origins[ids] = terrain.terrain_origins[terrain.terrain_levels[ids], terrain.terrain_types[ids]]

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _update_command(self):
        pass


@configclass
class EnvModeCommandCfg(CommandTermCfg):
    """Configuration for :class:`EnvModeCommand`."""

    class_type: type = EnvModeCommand

    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    stairs_fraction: float = 0.5
    """Fraction of environments in :data:`STAIRS` mode before the switch; the rest are :data:`FLAT`."""

    switch_step: int = 3000 * 24
    """Control step (iterations times steps per environment) from which resets move environments to pyramids."""

    flat_terrain: str = "flat"
    stairs_terrain: str = "motion_stairs"
    pyramid_terrain: str = "pyramid"
    """Sub-terrain names of the three modes."""

    pyramid_max_init_level: int = 2
    """Highest pyramid row an environment starts on after the switch."""


def assign_env_modes(env: ManagerBasedRLEnv, env_ids: torch.Tensor, command_name: str = "env_mode") -> None:
    """Reset event that applies the mode schedule; it must be the first reset event."""
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids if isinstance(env_ids, slice) else slice(None)]
    env.command_manager.get_term(command_name).assign(torch.as_tensor(env_ids, device=env.device))


def _resolve_term(cfg, env: ManagerBasedRLEnv):
    """Resolve the wrapped callable, and instantiate it with ``func_params`` if it is a class.

    The managers resolve scene entities throughout a term's parameters, including ``func_params``.
    """
    func = cfg.params["func"]
    if isinstance(func, str):
        func = string_to_callable(str(func))
    if inspect.isclass(func):
        func = func(cfg.replace(func=func, params=cfg.params["func_params"]), env)
    return func


class ModeGated(ManagerTermBase):
    """Reward, termination, or observation term ``func(env, **func_params)`` that is zero outside ``modes``.

    Class terms are instantiated with ``func_params`` and receive resets. Values outside ``modes`` are replaced
    (not multiplied) by zero or ``False``, so a non-finite value there cannot leak.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._term = _resolve_term(cfg, env)
        self._modes = cfg.params["modes"]

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if isinstance(self._term, ManagerTermBase):
            self._term.reset(env_ids=env_ids)

    def __call__(self, env: ManagerBasedRLEnv, func, func_params: dict, modes: Sequence[int], mode_command="env_mode"):
        value = self._term(env, **func_params)
        mask = env.command_manager.get_term(mode_command).mask(self._modes)
        mask = mask.view(-1, *([1] * (value.dim() - 1)))
        return torch.where(mask, value, torch.zeros_like(value))


class ModeGatedEvent(ManagerTermBase):
    """Event or curriculum term ``func(env, env_ids, **func_params)`` applied only to environments in ``modes``.

    As a curriculum term it returns the wrapped term's latest value, held while no environment of ``modes`` resets.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._term = _resolve_term(cfg, env)
        self._modes = cfg.params["modes"]
        self._last = None

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if isinstance(self._term, ManagerTermBase):
            self._term.reset(env_ids=env_ids)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        func,
        func_params: dict,
        modes: Sequence[int],
        mode_command="env_mode",
    ):
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device)
        env_ids = torch.as_tensor(env_ids, device=env.device)
        env_ids = env_ids[env.command_manager.get_term(mode_command).mask(self._modes)[env_ids]]
        if len(env_ids) > 0:
            self._last = self._term(env, env_ids, **func_params)
        return self._last


def mode_time_out(env: ManagerBasedRLEnv, max_episode_length_s: float) -> torch.Tensor:
    """Episode time-out at ``max_episode_length_s``, shorter than the environment's episode length."""
    return env.episode_length_buf * env.step_dt >= max_episode_length_s


def mode_command(
    env: ManagerBasedRLEnv,
    mode_command_name: str = "env_mode",
    motion_command_name: str = "motion",
    waypoint_command_name: str = "waypoint",
) -> torch.Tensor:
    """Actor goal shared by all modes: ``(x, y, z, sin, cos)``. Shape is (num_envs, 5).

    The destination in the root yaw frame [m] and the yaw error to the goal heading. Stairs use the motion command's
    destination; flat and pyramid environments use the waypoint (the tile center, on the ground for flat tiles and
    on the landing for pyramids) with the approach heading, as the reference clips end facing the direction they
    climbed. The actor does not see which mode it is in.
    """
    stairs = env.command_manager.get_term(mode_command_name).mask((STAIRS,)).unsqueeze(-1)
    motion_goal = env.command_manager.get_command(motion_command_name)
    waypoint = env.command_manager.get_term(waypoint_command_name)
    root_yaw = math_utils.euler_xyz_from_quat(waypoint.robot.data.root_quat_w.torch)[2]
    yaw_error = math_utils.wrap_to_pi(waypoint.goal_yaw - root_yaw).unsqueeze(-1)
    waypoint_goal = torch.cat((waypoint.command, yaw_error.sin(), yaw_error.cos()), dim=-1)
    return torch.where(stairs, motion_goal, waypoint_goal)


def mirror_mode_command(data: torch.Tensor) -> torch.Tensor:
    """Reflect :func:`mode_command` across the root XZ plane."""
    return mirror_goal(data)


def env_mode_stats(env: ManagerBasedRLEnv, env_ids: Sequence[int], command_name: str = "env_mode") -> dict:
    """Log the switch state, the fraction of environments in each mode, and the mean pyramid level."""
    modes: EnvModeCommand = env.command_manager.get_term(command_name)
    stats = {"switched": float(modes.switched)}
    for mode, name in enumerate(MODE_NAMES):
        stats[f"{name}_fraction"] = (modes.mode == mode).float().mean().item()
    pyramid = modes.mode == PYRAMID
    levels = env.scene.terrain.terrain_levels[pyramid].float()
    stats["pyramid_level"] = levels.mean().item() if len(levels) else 0.0
    return stats
