# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PointNet actor using named observations, without manager or runner overrides."""

from __future__ import annotations

import copy

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, HiddenState
from tensordict import TensorDict
from torch import nn


class PointNetEncoder(nn.Module):
    """Encode XYZ + validity points with shared weights and masked max pooling."""

    def __init__(self, hidden_dims: tuple[int, ...], feature_dim: int, activation: str):
        super().__init__()
        self.point_mlp = MLP(3, feature_dim, hidden_dims, activation)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        valid = points[..., 3:4] > 0.5
        # Ignore invalid XYZ even if an external caller supplies NaN padding.
        xyz = torch.where(valid, points[..., :3], 0.0)
        features = self.point_mlp(xyz).masked_fill(~valid, -torch.inf)
        pooled = features.max(dim=1).values
        return torch.where(valid.any(dim=1), pooled, 0.0)


class PointNetMLPModel(MLPModel):
    """Encode one ``(B, N, 4)`` point group alongside flat proprioceptive groups.

    Only flat observations are empirically normalized. Point coordinates remain
    metric and share one encoder across points, preserving permutation invariance.
    PPO, distributions, normalization updates, and checkpointing use RSL-RL's
    existing model interface. No auxiliary objective is introduced here.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        activation: str = "elu",
        obs_normalization: bool = True,
        distribution_cfg: dict | None = None,
        point_cloud_group: str = "lidar",
        pointnet_hidden_dims: tuple[int, ...] = (64, 128),
        pointnet_feature_dim: int = 256,
    ):
        self.point_cloud_group = point_cloud_group
        self.pointnet_feature_dim = pointnet_feature_dim
        if point_cloud_group not in obs_groups[obs_set]:
            raise ValueError(f"Point-cloud group '{point_cloud_group}' must be selected in '{obs_set}'.")
        points = obs[point_cloud_group]
        if points.ndim != 3 or points.shape[-1] != 4 or points.shape[1] == 0:
            raise ValueError(f"Expected point cloud (batch, points, 4), got {tuple(points.shape)}.")
        self.num_points = points.shape[1]
        super().__init__(
            obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, distribution_cfg
        )
        self.pointnet = PointNetEncoder(pointnet_hidden_dims, pointnet_feature_dim, activation)

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        groups = [name for name in obs_groups[obs_set] if name != self.point_cloud_group]
        if not groups:
            raise ValueError("PointNetMLPModel requires at least one flat proprioceptive observation group.")
        return super()._get_obs_dim(obs, {obs_set: groups}, obs_set)

    def _get_latent_dim(self) -> int:
        return self.obs_dim + self.pointnet_feature_dim

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        proprioception = super().get_latent(obs, masks, hidden_state)
        return torch.cat((proprioception, self.pointnet(obs[self.point_cloud_group])), dim=-1)

    def as_jit(self) -> nn.Module:
        """Export deterministic inference with flat observations and points as separate inputs."""
        return _PointNetExport(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Export the same two-input inference graph, including the point encoder."""
        return _OnnxPointNetExport(self, verbose)


class _PointNetExport(nn.Module):
    """Inference graph shared by TorchScript and ONNX exports."""

    def __init__(self, model: PointNetMLPModel):
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.pointnet = copy.deepcopy(model.pointnet)
        self.mlp = copy.deepcopy(model.mlp)
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module() if model.distribution is not None else nn.Identity()
        )

    def forward(self, observations: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        latent = torch.cat((self.obs_normalizer(observations), self.pointnet(points)), dim=-1)
        return self.deterministic_output(self.mlp(latent))

    @torch.jit.export
    def reset(self) -> None:
        """No recurrent state is maintained."""


class _OnnxPointNetExport(_PointNetExport):
    """RSL-RL ONNX exporter metadata for the two input tensors."""

    is_recurrent = False

    def __init__(self, model: PointNetMLPModel, verbose: bool):
        super().__init__(model)
        self.verbose = verbose
        self.obs_dim = model.obs_dim
        self.num_points = model.num_points

    def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(1, self.obs_dim), torch.zeros(1, self.num_points, 4)

    @property
    def input_names(self) -> list[str]:
        return ["observations", "points"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
