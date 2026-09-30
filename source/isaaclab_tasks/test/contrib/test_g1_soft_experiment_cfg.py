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
