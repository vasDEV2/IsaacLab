# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-local perception model loaded by RSL-RL's qualified class-name mechanism."""

from __future__ import annotations

import torch
from rsl_rl.models import CNNModel
from rsl_rl.utils import resolve_callable
from tensordict import TensorDict
from torch import nn


class G1VisionEncoder(nn.Module):
    """Adapt a flattened encoder to a fixed latent size and maintain its freeze state.

    Encoder factories follow the RSL-RL CNN contract: accept ``input_dim`` (H, W)
    and ``input_channels``, and expose ``output_dim`` and ``output_channels=None``.
    The projection belongs to the trainable fusion stage, even when the backbone
    is frozen. Encoder-specific checkpoint/pretrained arguments go in its kwargs.
    """

    def __init__(self, backbone: nn.Module, latent_dim: int, freeze_encoder: bool):
        super().__init__()
        if backbone.output_channels is not None:
            raise ValueError("G1 vision encoders must return flattened features (output_channels=None).")
        if latent_dim <= 0:
            raise ValueError("encoder_latent_dim must be positive.")
        self.backbone = backbone
        self.projection = nn.Linear(int(backbone.output_dim), latent_dim)
        self.output_dim = latent_dim
        self.output_channels = None
        self.freeze_encoder = freeze_encoder
        self.backbone.requires_grad_(not freeze_encoder)
        self.train()

    def train(self, mode: bool = True) -> G1VisionEncoder:
        """Keep frozen BatchNorm statistics and dropout in evaluation mode."""
        super().train(mode)
        if self.freeze_encoder:
            self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images of shape [N, C, H, W] into fusion features."""
        return self.projection(self.backbone(images))


class G1VisionActor(CNNModel):
    """Fuse separate image groups with proprioception using the standard RSL-RL head."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
        activation: str,
        obs_normalization: bool,
        distribution_cfg: dict | None,
        encoder_class_name: str,
        encoder_kwargs: dict,
        encoder_latent_dim: int,
        freeze_encoder: bool,
    ) -> None:
        factory = resolve_callable(encoder_class_name)
        encoders = {}
        for group in obs_groups[obs_set]:
            sample = obs[group]
            if sample.ndim == 4:
                backbone = factory(
                    input_dim=tuple(sample.shape[2:]), input_channels=sample.shape[1], **encoder_kwargs
                )
                encoders[group] = G1VisionEncoder(backbone, encoder_latent_dim, freeze_encoder)
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
            cnns=encoders,
        )
