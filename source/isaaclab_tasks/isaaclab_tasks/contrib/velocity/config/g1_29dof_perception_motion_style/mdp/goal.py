# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A goal on the pyramid's top landing, approached from the flat ground around it."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.utils import math as math_utils
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.velocity.config.g1_29dof_perception_waypoint.mdp import WaypointCommand, WaypointCommandCfg


class PyramidGoalCommand(WaypointCommand):
    """:class:`WaypointCommand` that always starts on the flat ring and targets the landing center.

    The start yaw faces the goal with uniform ``start_yaw_noise``, as in the reference clips, which start at the
    foot of a pyramid face walking toward its center. The goal is the tile origin (the landing center for
    :class:`~.terrains.ClipPyramidStairsTerrainCfg`) plus ``base_height``. :attr:`goal_yaw` is the approach
    direction from the start to the goal, the heading the clips end with; no reward tracks it.
    """

    cfg: PyramidGoalCommandCfg

    def __init__(self, cfg: PyramidGoalCommandCfg, env):
        super().__init__(cfg, env)
        self.goal_yaw = torch.zeros(self.num_envs, device=self.device)
        """Approach direction [rad] from the episode start to the goal."""

    def sample_episode(self, env_ids: Sequence[int] | torch.Tensor | slice):
        env_ids = self._as_env_ids(env_ids)
        pending = env_ids[self._pending[env_ids]]
        super().sample_episode(pending)
        if len(pending) == 0:
            return
        to_goal = self.waypoint_pos_w[pending, :2] - self.start_pos_w[pending, :2]
        heading = torch.atan2(to_goal[:, 1], to_goal[:, 0])
        self.goal_yaw[pending] = heading
        noise = math_utils.sample_uniform(
            -self.cfg.start_yaw_noise, self.cfg.start_yaw_noise, heading.shape, self.device
        )
        self.start_yaw[pending] = math_utils.wrap_to_pi(heading + noise)


@configclass
class PyramidGoalCommandCfg(WaypointCommandCfg):
    """Configuration for :class:`PyramidGoalCommand`."""

    class_type: type = PyramidGoalCommand

    center_start_probability: float = 0.0
    """Every episode climbs from the ring to the landing."""

    border_distance_range: tuple[float, float] = (2.9, 3.9)
    """Chebyshev distance [m] of the start from the tile center.

    The widest default pyramid base reaches 2.65 m and the 9 m tile edge is at 4.5 m.
    """

    start_yaw_noise: float = 0.5
    """Uniform noise [rad] of the start yaw around the direction to the goal."""
