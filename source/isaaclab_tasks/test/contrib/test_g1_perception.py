# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coordinate, masking, and exported-policy contracts for G1 perception."""

import math
from types import SimpleNamespace

import torch
from tensordict import TensorDict

from isaaclab.managers import ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors import patterns

from isaaclab_tasks.contrib.velocity.config.g1_29dof_perception.agents.pointnet import PointNetMLPModel
from isaaclab_tasks.contrib.velocity.config.g1_29dof_perception.env_cfg.sensors_cfg import elevation_grid_pattern
from isaaclab_tasks.contrib.velocity.config.g1_29dof_perception.mdp.depth import depth_image
from isaaclab_tasks.contrib.velocity.config.g1_29dof_perception.mdp.observations import (
    elevation_map,
    finite_base_ang_vel,
    lidar_point_cloud,
)
from isaaclab_tasks.contrib.velocity.config.g1_29dof_perception.mdp.stair_rewards import (
    feet_stumble,
    local_base_height_l2,
    local_stance_foot_angle_penalty,
    stair_foot_clearance,
    world_height_convergence,
    world_height_error,
)
from isaaclab_tasks.contrib.velocity.config.g1_29dof_perception.mdp.symmetry import mirror_elevation_map
from isaaclab_tasks.contrib.velocity.config.vel_mdp.rewards import foot_clearance_reward


def test_finite_base_ang_vel_replaces_nonfinite_components():
    """The policy and critic receive finite angular velocity even if the simulator state is invalid."""
    angular_velocity = torch.tensor([[0.5, math.nan, math.inf], [-math.inf, -0.25, 1.0]])
    robot = SimpleNamespace(data=SimpleNamespace(root_ang_vel_b=SimpleNamespace(torch=angular_velocity)))
    env = SimpleNamespace(scene={"robot": robot})

    actual = finite_base_ang_vel(env)

    torch.testing.assert_close(actual, torch.tensor([[0.5, 0.0, 0.0], [0.0, -0.25, 1.0]]))


def test_elevation_map_mirror_moves_terrain_to_the_opposite_lateral_cell():
    """A raised left-hand cell becomes a raised right-hand cell with valid grid coordinates."""
    data = torch.tensor(
        [
            [
                [[0.0, -1.0, -0.8], [0.0, 0.0, -0.6], [0.0, 1.0, -0.4]],
                [[1.0, -1.0, -2.0], [1.0, 0.0, -0.6], [1.0, 1.0, -0.3]],
            ]
        ]
    )
    original = data.clone()
    mirrored = mirror_elevation_map(data)
    torch.testing.assert_close(mirrored[..., :2], data[..., :2])
    torch.testing.assert_close(mirrored[..., 2], torch.tensor([[[-0.4, -0.6, -0.8], [-0.3, -0.6, -2.0]]]))
    torch.testing.assert_close(mirror_elevation_map(mirrored), original)
    torch.testing.assert_close(data, original)


def test_stumble_penalty_distinguishes_stair_impacts_from_support_and_noise():
    """Horizontal impact on either foot is penalized; support, no contact and tiny noise are not."""
    forces = torch.tensor(
        [
            [[0.0, 0.0, 200.0], [0.0, 0.0, 0.0]],
            [[0.0, 60.0, 10.0], [0.0, 0.0, 200.0]],
            [[0.0, 0.0, 200.0], [-60.0, 0.0, -10.0]],
            [[0.01, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    sensor = SimpleNamespace(data=SimpleNamespace(net_normal_forces_w=SimpleNamespace(torch=forces)))
    env = SimpleNamespace(scene=SimpleNamespace(sensors={"contacts": sensor}))
    cfg = SceneEntityCfg("contacts", body_ids=[0, 1])
    torch.testing.assert_close(feet_stumble(env, cfg), torch.tensor([0.0, 1.0, 1.0, 0.0]))


def test_stair_rewards_use_local_treads_and_do_not_reward_motionless_feet():
    """Adjacent stair heights must not bias local clearance; standing cannot earn swing reward."""
    velocity = torch.zeros(1, 2, 3)
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=torch.tensor([[1.0, 0.0, 0.95]])),
            body_pos_w=SimpleNamespace(torch=torch.tensor([[[0.0, 0.0, 0.14], [1.0, 0.0, 0.34]]])),
            body_lin_vel_w=SimpleNamespace(torch=velocity),
        )
    )
    scanner = SimpleNamespace(
        data=SimpleNamespace(
            ray_hits_w=SimpleNamespace(
                torch=torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.2], [10.0, 0.0, 10.0], [math.nan] * 3]])
            )
        )
    )
    command = torch.tensor([[0.75, 0.0, 0.0]])
    env = SimpleNamespace(
        scene={"robot": robot, "scanner": scanner}, command_manager=SimpleNamespace(get_command=lambda name: command)
    )
    sensor_cfg = SceneEntityCfg("scanner")
    torch.testing.assert_close(local_base_height_l2(env, 0.75, sensor_cfg), torch.zeros(1))
    params = dict(
        target_height=0.1,
        std=0.05,
        tanh_mult=2.0,
        asset_cfg=SceneEntityCfg("robot", body_ids=[0, 1]),
        height_sensor_cfg=sensor_cfg,
        standing_position_foot_z=0.04,
    )
    torch.testing.assert_close(stair_foot_clearance(env, **params), torch.zeros(1))
    velocity[..., 0] = 10.0
    torch.testing.assert_close(stair_foot_clearance(env, **params), torch.ones(1))
    command.zero_()
    torch.testing.assert_close(stair_foot_clearance(env, **params), torch.zeros(1))


def test_stance_foot_angle_uses_scanner_hits_near_each_foot():
    """Feet aligned with distinct local slopes have no angle penalty."""
    slope_angle = math.atan(0.5)
    foot_quat = torch.tensor(
        [[[0.0, 0.0, 0.0, 1.0], [0.0, -math.sin(slope_angle / 2), 0.0, math.cos(slope_angle / 2)]]]
    )
    feet = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    offsets = ((-0.1, -0.1), (-0.1, 0.1), (0.1, -0.1), (0.1, 0.1))
    hits = [[x, y, 0.0] for x, y in offsets]
    hits.extend([[2.0 + x, y, 0.5 * x] for x, y in offsets])
    hits.extend([[10.0, -1.0, -10.0], [10.0, 1.0, 10.0]])
    robot = SimpleNamespace(
        data=SimpleNamespace(body_pos_w=SimpleNamespace(torch=feet), body_quat_w=SimpleNamespace(torch=foot_quat))
    )
    scanner = SimpleNamespace(data=SimpleNamespace(ray_hits_w=SimpleNamespace(torch=torch.tensor([hits]))))
    contact_sensor = SimpleNamespace(data=SimpleNamespace(current_contact_time=SimpleNamespace(torch=torch.ones(1, 2))))

    class Scene(dict):
        def __init__(self):
            super().__init__(robot=robot, scanner=scanner)
            self.sensors = {"contacts": contact_sensor}

    env = SimpleNamespace(scene=Scene(), device="cpu")
    asset_cfg = SceneEntityCfg("robot", body_ids=[0, 1])
    penalty = local_stance_foot_angle_penalty(
        env,
        sensor_cfg=SceneEntityCfg("contacts", body_ids=[0, 1]),
        asset_cfg=asset_cfg,
        height_sensor_cfg=SceneEntityCfg("scanner"),
        angle_tolerance=0.0,
        local_radius=0.3,
    )
    torch.testing.assert_close(penalty, torch.zeros(1), atol=1e-6, rtol=0.0)


def test_world_height_reward_pays_progress_toward_zero_not_standing_at_height():
    """Descending from a top or rising from a bottom pays; standing, moving away or an uncommanded env do not."""
    root_z = torch.tensor([1.95, -0.45, 1.95, -0.45, 1.95, 0.75, 1.95, 1.95])
    vertical = torch.tensor([-0.5, 0.5, 0.0, 0.0, 0.5, 0.5, -5.0, -0.5])
    velocity = torch.stack((torch.ones(8), torch.zeros(8), vertical), dim=-1)
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=torch.stack((torch.zeros(8), torch.zeros(8), root_z), dim=-1)),
            root_lin_vel_w=SimpleNamespace(torch=velocity),
        )
    )
    command = torch.tensor([[1.0, 0.0, 0.0]] * 7 + [[0.0, 0.0, 0.0]])
    env = SimpleNamespace(scene={"robot": robot}, command_manager=SimpleNamespace(get_command=lambda name: command))
    torch.testing.assert_close(
        world_height_convergence(env), torch.tensor([0.5, 0.5, 0.0, 0.0, -0.5, 0.0, 1.0, 0.0]), atol=1e-6, rtol=0.0
    )


def test_world_height_error_charges_distance_from_zero_for_commanded_envs_only():
    """The error is the clipped distance to the target height from either side, and zero when uncommanded."""
    root_z = torch.tensor([0.75, 1.95, -0.45, 9.0, 1.95])
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=torch.stack((torch.zeros(5), torch.zeros(5), root_z), dim=-1)),
        )
    )
    command = torch.tensor([[1.0, 0.0, 0.0]] * 4 + [[0.0, 0.0, 0.0]])
    env = SimpleNamespace(scene={"robot": robot}, command_manager=SimpleNamespace(get_command=lambda name: command))
    torch.testing.assert_close(world_height_error(env), torch.tensor([0.0, 1.2, 1.2, 2.0, 0.0]), atol=1e-6, rtol=0.0)


def test_lidar_uses_measurement_frame_and_masks_misses():
    """World translation/yaw must not leak into the sensor-frame cloud; misses stay finite."""
    sensor = SimpleNamespace(
        num_rays=5,
        cfg=SimpleNamespace(ray_alignment="base", max_distance=40.0),
        data=SimpleNamespace(
            pos_w=SimpleNamespace(torch=torch.tensor([[10.0, 20.0, 1.0]])),
            quat_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]])),
            ray_hits_w=SimpleNamespace(
                torch=torch.tensor(
                    [[[10.0, 22.0, 1.0], [9.0, 20.0, 1.0], [math.inf] * 3, [math.nan] * 3, [10.0, 61.0, 1.0]]]
                )
            ),
        ),
    )
    env = SimpleNamespace(scene=SimpleNamespace(sensors={"lidar": sensor}), device="cpu")
    cfg = ObservationTermCfg(func=lidar_point_cloud, params={"sensor_cfg": SceneEntityCfg("lidar"), "num_points": 5})
    term = lidar_point_cloud(cfg, env)
    actual = term(env, **cfg.params)
    expected = torch.tensor([[[2.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0] * 4, [0.0] * 4, [0.0] * 4]])
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_elevation_map_masks_nonfinite_sensor_pose():
    """An invalid scanner frame yields the configured finite missing-ray value."""
    pattern = patterns.GridPatternCfg(
        func=elevation_grid_pattern,
        resolution=0.2,
        size=(0.2, 0.2),
        ordering="yx",
    )
    starts, _ = pattern.func(pattern, "cpu")
    sensor = SimpleNamespace(
        num_rays=starts.shape[0],
        cfg=SimpleNamespace(ray_alignment="yaw", pattern_cfg=pattern),
        data=SimpleNamespace(
            pos_w=SimpleNamespace(torch=torch.tensor([[math.nan, 0.0, 1.0]])),
            quat_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.0, 1.0]])),
            ray_hits_w=SimpleNamespace(torch=torch.zeros(1, starts.shape[0], 3)),
        ),
    )
    env = SimpleNamespace(scene=SimpleNamespace(sensors={"height_scanner": sensor}), device="cpu")
    cfg = ObservationTermCfg(
        func=elevation_map,
        params={"sensor_cfg": SceneEntityCfg("height_scanner"), "missing_height": -2.0},
    )
    term = elevation_map(cfg, env)
    actual = term(env, **cfg.params)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual[..., 2], torch.full_like(actual[..., 2], -2.0))


def test_foot_clearance_ignores_nonfinite_height_scan_hits():
    """Missed and invalid rays cannot turn a zero-speed clearance reward into NaN."""
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_pos_w=SimpleNamespace(torch=torch.tensor([[[0.0, 0.0, 0.43539]], [[0.0, 0.0, 0.13539]]])),
            body_lin_vel_w=SimpleNamespace(torch=torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]])),
        )
    )
    scanner = SimpleNamespace(
        data=SimpleNamespace(
            ray_hits_w=SimpleNamespace(
                torch=torch.tensor(
                    [
                        [[0.0, 0.0, 0.2], [0.0, 0.0, 0.4], [math.inf] * 3, [math.nan] * 3],
                        [[math.inf] * 3, [math.nan] * 3, [math.inf] * 3, [math.nan] * 3],
                    ]
                )
            )
        )
    )
    env = SimpleNamespace(scene={"robot": robot, "height_scanner": scanner})
    reward = foot_clearance_reward(
        env,
        target_height=0.1,
        std=0.05,
        tanh_mult=2.0,
        asset_cfg=SceneEntityCfg("robot", body_ids=[0]),
        standing_position_foot_z=0.03539,
        height_sensor_cfg=SceneEntityCfg("height_scanner"),
    )
    assert torch.isfinite(reward).all()
    torch.testing.assert_close(reward, torch.ones(2))


def _make_model():
    torch.manual_seed(4)
    points = torch.randn(2, 8, 4)
    points[..., 3] = 1.0
    points[:, -2:, 3] = 0.0
    obs = TensorDict({"policy": torch.randn(2, 6), "lidar": points, "command": torch.randn(2, 3)}, batch_size=[2])
    model = PointNetMLPModel(
        obs,
        {"actor": ["policy", "lidar", "command"]},
        "actor",
        3,
        hidden_dims=(16,),
        pointnet_hidden_dims=(8,),
        pointnet_feature_dim=8,
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
    )
    return model, obs


def test_policy_is_point_order_invariant_and_ignores_padding():
    """Point normalization/pooling must not learn point order or treat misses as surfaces."""
    model, obs = _make_model()
    model.update_normalization(obs)
    model.eval()
    expected = model(obs)
    padded = obs.clone()
    padded["lidar"][:, -2:, :3] = math.nan
    padded["lidar"] = padded["lidar"][:, torch.randperm(8)]
    torch.testing.assert_close(model(padded), expected)
    empty = obs.clone()
    empty["lidar"][..., :3] = math.nan
    empty["lidar"][..., 3] = 0.0
    assert torch.isfinite(model(empty)).all()
    model.zero_grad()
    model(obs).square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.pointnet.parameters())
    assert any(torch.count_nonzero(p.grad) > 0 for p in model.pointnet.parameters())


def test_policy_checkpoint_and_exports_preserve_point_encoder(tmp_path):
    """Saving/exporting must include the encoder and trained normalization, not only the MLP head."""
    model, obs = _make_model()
    model.update_normalization(obs)
    model.eval()
    expected = model(obs)
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    restored, _ = _make_model()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    restored.eval()
    torch.testing.assert_close(restored(obs), expected)
    inputs = (torch.cat((obs["policy"], obs["command"]), dim=-1), obs["lidar"])
    scripted = torch.jit.script(restored.as_jit())
    scripted.save(str(tmp_path / "policy.pt"))
    torch.testing.assert_close(torch.jit.load(str(tmp_path / "policy.pt"))(*inputs), expected)
    exported = restored.as_onnx()
    torch.testing.assert_close(exported(*inputs), expected)
    # The RSL-RL exporter consumes these same two inputs.
    assert tuple(x.shape for x in exported.get_dummy_inputs()) == ((1, 9), (1, 8, 4))


def test_depth_crop_units_misses_and_measurement_cache():
    """Depth stays metric before normalization; repeated reads and partial resets preserve frame ownership."""
    raw = torch.full((2, 3, 7, 1), 100.0)
    raw[:, 1, 1:6, 0] = torch.tensor([0.3, 1.65, 3.0, 0.0, math.nan])
    sensor = SimpleNamespace(
        data=SimpleNamespace(output={"depth": SimpleNamespace(torch=raw)}),
        frame=SimpleNamespace(torch=torch.ones(2, dtype=torch.long)),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(sensors={"camera": sensor}), device="cpu", num_envs=2, common_step_counter=0
    )
    cfg = ObservationTermCfg(
        func=depth_image,
        params={
            "sensor_cfg": SceneEntityCfg("camera"),
            "resize": (1, 5),
            "crop": (1, 1, 1, 1),
            "corrupt": False,
        },
    )
    term = depth_image(cfg, env)
    expected = torch.tensor([[[[-0.5, 0.0, 0.5, 0.5, 0.5]]]]).expand(2, -1, -1, -1)
    torch.testing.assert_close(term(env, **cfg.params), expected)
    assert torch.isnan(raw[:, 1, 5, 0]).all()  # Sensor storage must not be modified.
    raw.fill_(3.0)
    torch.testing.assert_close(term(env, **cfg.params), expected)
    env.common_step_counter += 1
    torch.testing.assert_close(term(env, **cfg.params), expected)  # No new camera frame yet.
    sensor.frame.torch[0] += 1
    env.common_step_counter += 1
    updated = term(env, **cfg.params).clone()
    torch.testing.assert_close(updated[0], torch.full((1, 1, 5), 0.5))
    torch.testing.assert_close(updated[1], expected[1])
    # A reset can occur without advancing the global step or changing the camera's frame number.
    raw[0].fill_(0.3)
    term.reset(torch.tensor([0]))
    reset = term(env, **cfg.params)
    torch.testing.assert_close(reset[0], torch.full((1, 1, 5), -0.5))
    torch.testing.assert_close(reset[1], updated[1])


def test_elevation_map_preserves_yaw_frame_grid_and_missing_cells():
    """A translated/pitched scanner keeps XYZ relative to its origin and finite missing cells."""
    from isaaclab.sensors import patterns

    from isaaclab_tasks.contrib.velocity.config.g1_29dof_perception.env_cfg.sensors_cfg import elevation_grid_pattern
    from isaaclab_tasks.contrib.velocity.config.g1_29dof_perception.mdp.observations import elevation_map

    half_pitch = math.pi / 12
    half_yaw = math.pi / 4
    quat = torch.tensor(
        [
            [
                -math.sin(half_pitch) * math.sin(half_yaw),
                math.sin(half_pitch) * math.cos(half_yaw),
                math.cos(half_pitch) * math.sin(half_yaw),
                math.cos(half_pitch) * math.cos(half_yaw),
            ]
        ]
    )
    sensor = SimpleNamespace(
        num_rays=6,
        cfg=SimpleNamespace(
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(
                func=elevation_grid_pattern, resolution=1.0, size=(1.0, 2.0), ordering="yx"
            ),
        ),
        data=SimpleNamespace(
            pos_w=SimpleNamespace(torch=torch.tensor([[10.0, 20.0, 1.0]])),
            quat_w=SimpleNamespace(torch=quat),
            ray_hits_w=SimpleNamespace(
                torch=torch.tensor(
                    [
                        [
                            [11.0, 19.5, 0.5],
                            [10.0, 19.5, 0.6],
                            [9.0, 19.5, 0.7],
                            [11.0, 20.5, 0.8],
                            [math.nan] * 3,
                            [math.inf] * 3,
                        ]
                    ]
                )
            ),
        ),
    )
    env = SimpleNamespace(scene=SimpleNamespace(sensors={"map": sensor}), device="cpu")
    cfg = ObservationTermCfg(func=elevation_map, params={"sensor_cfg": SceneEntityCfg("map"), "missing_height": -2.0})
    term = elevation_map(cfg, env)
    expected = torch.tensor(
        [
            [
                [[-0.5, -1.0, -0.5], [-0.5, 0.0, -0.4], [-0.5, 1.0, -0.3]],
                [[0.5, -1.0, -0.2], [0.5, 0.0, -2.0], [0.5, 1.0, -2.0]],
            ]
        ]
    )
    torch.testing.assert_close(term(env, **cfg.params), expected, atol=1e-6, rtol=1e-6)
    assert torch.isnan(sensor.data.ray_hits_w.torch[0, 4]).all()
