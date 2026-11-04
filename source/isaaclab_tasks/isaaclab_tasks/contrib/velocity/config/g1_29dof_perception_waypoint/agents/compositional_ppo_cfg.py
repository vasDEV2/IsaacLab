# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AME PPO for the waypoint task; the actor's ``command`` input is the waypoint."""

from isaaclab.utils.configclass import configclass

from ...g1_29dof_perception.agents.compositional_ppo_cfg import G1RoughPerceptionPPORunnerCfg


@configclass
class G1RoughPerceptionWaypointPPORunnerCfg(G1RoughPerceptionPPORunnerCfg):
    """The perception task's networks and PPO settings, without BatchNorm in the map encoder."""

    experiment_name = "g1_29dof_perception_ame_waypoint"

    def __post_init__(self):
        # BatchNorm statistics differ between rollouts and minibatch updates, which breaks the PPO ratio.
        self.networks.map_encoder.cnn_normalization = "none"


@configclass
class G1FlatPerceptionWaypointPPORunnerCfg(G1RoughPerceptionWaypointPPORunnerCfg):
    """Waypoint AME PPO on a plane."""

    experiment_name = "g1_29dof_perception_ame_waypoint_flat"
