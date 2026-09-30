# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 experiment configuration and perception contracts."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab_tasks.contrib.velocity.config.g1_29dof_soft.env_cfg.experiment_cfg import G1SensorsCfg
from isaaclab_tasks.contrib.velocity.config.g1_29dof_soft.env_cfg.scene_cfg import G1SceneCfg
from isaaclab_tasks.contrib.velocity.config.g1_29dof_soft.mdp.observations import d435_depth, mid360_lidar_ranges


def test_sensor_defaults_and_instance_isolation():
    first = G1SceneCfg(num_envs=1, env_spacing=2.5)
    second = G1SceneCfg(num_envs=1, env_spacing=2.5)
    defaults = G1SensorsCfg()
    assert first.mid360_lidar.to_dict() == defaults.mid360_lidar.to_dict()
    assert first.d435_camera.to_dict() == defaults.d435_camera.to_dict()
    assert first.mid360_lidar.max_distance == 40.0
    assert (first.d435_camera.width, first.d435_camera.height) == (640, 480)
    first.mid360_lidar.pattern_cfg.channels = 2
    assert second.mid360_lidar.pattern_cfg.channels == 40


@pytest.mark.parametrize("limit", [3.0, 17.0])
def test_perception_clipping_tracks_live_sensor_limits(limit):
    lidar = SimpleNamespace(
        cfg=SimpleNamespace(max_distance=limit),
        data=SimpleNamespace(ray_distances=torch.tensor([[1.0, float("inf"), float("nan"), 100.0]])),
    )
    camera = SimpleNamespace(
        cfg=SimpleNamespace(spawn=SimpleNamespace(clipping_range=(0.1, limit))),
        data=SimpleNamespace(output={"depth": torch.tensor([[[[1.0], [float("inf")], [float("nan")], [100.0]]]])}),
    )
    env = SimpleNamespace(scene=SimpleNamespace(sensors={"mid360_lidar": lidar, "d435_camera": camera}))
    expected = torch.tensor([[1.0, limit, limit, limit]])
    torch.testing.assert_close(mid360_lidar_ranges(env), expected)
    torch.testing.assert_close(d435_depth(env).flatten(1), expected)
    torch.testing.assert_close(d435_depth(env, clip_max=2.0).flatten(1), torch.tensor([[1.0, 2.0, 2.0, 2.0]]))


def test_joint_action_order_and_observation_defaults():
    from isaaclab_tasks.contrib.velocity.config.g1_29dof_soft.env_cfg.experiment_cfg import (
        ACTIVE_JOINT,
        SOFT_CONTACT_THRESHOLD,
        G1ActionsCfg,
        G1ObservationsCfg,
    )

    actions = G1ActionsCfg()
    observations = G1ObservationsCfg()
    assert len(ACTIVE_JOINT) == len(set(ACTIVE_JOINT)) == 29
    assert actions.joint_pos.joint_names == ACTIVE_JOINT
    assert actions.joint_pos.scale == 0.25
    for group in (observations.policy, observations.critic):
        assert group.joint_pos.params["asset_cfg"].joint_names == ACTIVE_JOINT
        assert group.joint_vel.params["asset_cfg"].joint_names == ACTIVE_JOINT
        assert group.history_length == 10
    assert observations.privileged.foot_contact.params["soft_force_threshold"] == SOFT_CONTACT_THRESHOLD
    assert actions.physics_callback.contact_threshold == SOFT_CONTACT_THRESHOLD


def test_raw_contact_logging_preserves_flat_shape():
    from isaaclab_tasks.contrib.velocity.config.g1_29dof_soft.mdp.observations import foot_contact_forces_raw_hybrid

    rigid = torch.arange(12.0).reshape(2, 2, 3)
    soft = torch.ones(2, 2, 6)
    solver = SimpleNamespace(
        contact_wrench=soft, data=SimpleNamespace(is_sensor_active=torch.tensor([[True, False], [False, True]]))
    )
    env = SimpleNamespace(
        num_envs=2,
        scene=SimpleNamespace(
            sensors={"contact_forces": SimpleNamespace(data=SimpleNamespace(net_forces_w=SimpleNamespace(torch=rigid)))}
        ),
        action_manager=SimpleNamespace(get_term=lambda name: SimpleNamespace(contact_solver=solver)),
    )
    result = foot_contact_forces_raw_hybrid(env)
    torch.testing.assert_close(result, torch.tensor([[1.0, 1.0, 1.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 1.0, 1.0, 1.0]]))
