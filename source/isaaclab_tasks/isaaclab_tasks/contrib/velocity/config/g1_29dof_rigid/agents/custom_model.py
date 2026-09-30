# my_project/models/pointnet_mlp_model.py
"""PointNet-augmented MLPModel for rsl_rl (Isaac Lab 3.0 external-actor interface).

Subclasses `rsl_rl.models.mlp_model.MLPModel` instead of reimplementing it. The
model expects the *last* `num_point_obs` entries of the (already concatenated,
already normalized) flat observation vector to be a flattened point cloud
(`num_points * point_dim` values). Those entries are reshaped into
`(B, num_points, point_dim)`, passed through a small PointNet-style encoder
(shared per-point MLP + max-pool), and the resulting fixed-size latent is
concatenated back onto the remaining observation entries before being fed into
the regular MLP head -- everything downstream of that (MLP head, distribution,
log-prob/KL/entropy, obs normalization, hidden-state no-ops) is inherited
verbatim from `MLPModel`.

Caveats:
  - `num_point_obs` must be a fixed size (`num_points * point_dim`) every step,
    same as the flat-obs assumption MLPModel already makes -- pad/subsample your
    point cloud to a constant count in the observation term.
  - Observation normalization (`obs_normalization=True`) normalizes the WHOLE
    concatenated vector elementwise, point-cloud entries included, exactly like
    a normal MLPModel would. That's usually fine (each point coordinate is just
    another scalar feature to EmpiricalNormalization) but be aware it does not
    do anything point-cloud-aware (no centering per point, no per-point-index
    grouping of stats).
  - `as_jit` / `as_onnx` are overridden (not inherited) -- the base MLPModel
    export wrappers only carry `obs_normalizer` + `mlp`, so without overriding
    them the exported graph would silently skip the PointNet encoder.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel

_ACTIVATIONS = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}


class _PointNetEncoder(nn.Module):
    """Shared-weight per-point MLP + max-pool symmetric function.

    Input:  (B, N, point_dim)
    Output: (B, feature_dim) -- permutation-invariant w.r.t. point order.
    """

    def __init__(
        self,
        point_dim: int,
        hidden_dims: tuple[int, ...] = (64, 128),
        feature_dim: int = 256,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        act = _ACTIVATIONS[activation]

        layers: list[nn.Module] = []
        in_dim = point_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), act()]
            in_dim = h
        layers += [nn.Linear(in_dim, feature_dim)]
        # Shared per-point MLP, applied identically to every point via reshape
        # (rather than nn.Conv1d) to keep TorchScript/ONNX export simple.
        self.point_mlp = nn.Sequential(*layers)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # points: (B, N, point_dim)
        b, n, d = points.shape
        feat = self.point_mlp(points.reshape(b * n, d))  # (B*N, feature_dim)
        feat = feat.reshape(b, n, -1)  # (B, N, feature_dim)
        global_feat, _ = feat.max(dim=1)  # (B, feature_dim)
        return global_feat


class PointNetMLPModel(MLPModel):
    """MLPModel whose last `num_point_obs` input entries are PointNet-encoded."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        num_point_obs: int = 0,
        point_dim: int = 3,
        pointnet_hidden_dims: tuple[int, ...] = (64, 128),
        pointnet_feature_dim: int = 256,
    ) -> None:
        """
        Args:
            obs, obs_groups, obs_set, output_dim, hidden_dims, activation,
            obs_normalization, distribution_cfg: same as `MLPModel`.
            num_point_obs: number of trailing flat-obs entries that make up the
                flattened point cloud (must equal `num_points * point_dim`).
            point_dim: per-point feature size (3 for xyz, more if you pack in
                normals/intensity/etc).
            pointnet_hidden_dims: hidden sizes of the shared per-point MLP.
            pointnet_feature_dim: size of the pooled point-cloud latent that
                gets concatenated back onto the remaining observations.
        """
        if num_point_obs % point_dim != 0:
            raise ValueError(f"num_point_obs ({num_point_obs}) must be a multiple of point_dim ({point_dim}).")

        # Set before calling super().__init__(): plain (non-Module) attributes can be
        # set on an nn.Module before nn.Module.__init__() has run, and _get_latent_dim()
        # (called from inside MLPModel.__init__) needs these already in place.
        self.num_point_obs = num_point_obs
        self.point_dim = point_dim
        self.num_points = num_point_obs // point_dim if num_point_obs > 0 else 0
        self._pointnet_hidden_dims = pointnet_hidden_dims
        self._pointnet_feature_dim = pointnet_feature_dim
        self._pointnet_activation = activation

        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )

        if self.num_point_obs > self.obs_dim:
            raise ValueError(f"num_point_obs ({self.num_point_obs}) exceeds total obs_dim ({self.obs_dim}).")

        # nn.Module.__init__() has now run (as the first statement inside MLPModel.__init__),
        # so assigning a submodule here registers it correctly.
        self.pointnet = _PointNetEncoder(
            point_dim=self.point_dim,
            hidden_dims=self._pointnet_hidden_dims,
            feature_dim=self._pointnet_feature_dim,
            activation=self._pointnet_activation,
        )

    # ------------------------------------------------------------------ #
    # Only these two are overridden; forward(), reset(), get_hidden_state(),
    # distribution properties, get_output_log_prob(), get_kl_divergence(),
    # update_normalization(), etc. are all inherited unchanged from MLPModel.
    # ------------------------------------------------------------------ #
    def _get_latent_dim(self) -> int:
        """MLP input size = remaining flat obs + pooled point-cloud feature."""
        return (self.obs_dim - self.num_point_obs) + self._pointnet_feature_dim

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state=None
    ) -> torch.Tensor:
        """Concatenate + normalize obs (base behavior), then PointNet-encode the tail."""
        latent = super().get_latent(obs, masks, hidden_state)  # (B, obs_dim), normalized as usual
        if self.num_point_obs == 0:
            return latent
        rest = latent[..., : -self.num_point_obs]
        points_flat = latent[..., -self.num_point_obs :]
        points = points_flat.reshape(points_flat.shape[0], self.num_points, self.point_dim)
        point_latent = self.pointnet(points)
        return torch.cat([rest, point_latent], dim=-1)

    # ------------------------------------------------------------------ #
    # Export: must be overridden since MLPModel.as_jit / as_onnx only carry
    # obs_normalizer + mlp and would otherwise drop the PointNet encoder.
    # ------------------------------------------------------------------ #
    def as_jit(self) -> nn.Module:
        return _TorchPointNetMLPModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxPointNetMLPModel(self, verbose)


class _TorchPointNetMLPModel(nn.Module):
    """Exportable version for JIT, mirroring MLPModel's `_TorchMLPModel` but with the
    PointNet split/encode step folded into `forward`."""

    def __init__(self, model: PointNetMLPModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.pointnet = copy.deepcopy(model.pointnet)
        self.mlp = copy.deepcopy(model.mlp)
        self.num_point_obs = model.num_point_obs
        self.num_points = model.num_points
        self.point_dim = model.point_dim
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference on pre-concatenated, pre-split observations."""
        x = self.obs_normalizer(x)
        if self.num_point_obs > 0:
            rest, points_flat = x[..., : -self.num_point_obs], x[..., -self.num_point_obs :]
            points = points_flat.reshape(points_flat.shape[0], self.num_points, self.point_dim)
            x = torch.cat([rest, self.pointnet(points)], dim=-1)
        out = self.mlp(x)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxPointNetMLPModel(nn.Module):
    """Exportable version for ONNX, mirroring MLPModel's `_OnnxMLPModel`."""

    is_recurrent: bool = False

    def __init__(self, model: PointNetMLPModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.pointnet = copy.deepcopy(model.pointnet)
        self.mlp = copy.deepcopy(model.mlp)
        self.num_point_obs = model.num_point_obs
        self.num_points = model.num_points
        self.point_dim = model.point_dim
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.input_size = model.obs_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        if self.num_point_obs > 0:
            rest, points_flat = x[..., : -self.num_point_obs], x[..., -self.num_point_obs :]
            points = points_flat.reshape(points_flat.shape[0], self.num_points, self.point_dim)
            x = torch.cat([rest, self.pointnet(points)], dim=-1)
        out = self.mlp(x)
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]