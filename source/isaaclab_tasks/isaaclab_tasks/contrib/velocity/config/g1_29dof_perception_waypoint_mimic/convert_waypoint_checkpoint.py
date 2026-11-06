# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Narrow a ``g1_29dof_perception_waypoint`` AME checkpoint to this task's observations.

The waypoint task's ``policy`` group ends with the 3-d root position on the tile, and its ``command`` group is the
3-d waypoint in the root yaw frame. This task drops the root position and the waypoint height, keeping the waypoint
XY in the same frame. The matching input columns are removed from the map encoder's state projection, the actor,
and the critic, together with their normalizer statistics and Adam moments. Every kept input means what it did, so
the converted policy is a close warm start, but it is no longer identical: the dropped inputs had nonzero weights.

Usage: ``uv run --no-sync python convert_waypoint_checkpoint.py SOURCE TARGET`` (see the task's README).
"""

from __future__ import annotations

import argparse

import torch

ROOT_POSITION_DIMS = 3
"""Root position at the end of the waypoint task's ``policy`` group."""

COMMAND_DIMS = 3
"""Waypoint XYZ in the waypoint task's ``command`` group; Z (the last) is dropped."""

_STATE = "map_encoder.encoder.state_projection.weight"
_ACTOR = "actor.mlp.0.weight"
_CRITIC = "critic.mlp.0.weight"


def _dropped_columns(networks: dict[str, torch.Tensor]) -> dict[str, list[int]]:
    """Input columns to remove from each network's first layer.

    The map encoder's state projection reads ``[policy, command]`` and the actor ``[policy, command, map_encoder]``,
    so both lose the end of ``policy`` and the command's Z. The critic reads ``[critic, privileged, command,
    map_encoder]`` and loses the command's Z only.
    """
    state = networks[f"terms.{_STATE}"].shape[-1]
    actor = networks[f"terms.{_ACTOR}"].shape[-1]
    critic = networks[f"terms.{_CRITIC}"].shape[-1]
    policy = state - COMMAND_DIMS
    map_features = actor - state
    root_position = list(range(policy - ROOT_POSITION_DIMS, policy))
    command_z = policy + COMMAND_DIMS - 1
    return {
        "map_encoder.": [*root_position, command_z],
        "actor.": [*root_position, command_z],
        "critic.": [critic - map_features - 1],
    }


def _drop(tensor: torch.Tensor, columns: list[int]) -> torch.Tensor:
    """Remove ``columns`` along the last dimension."""
    keep = [i for i in range(tensor.shape[-1]) if i not in set(columns)]
    return tensor[..., keep]


def _prefix(name: str, dropped: dict[str, list[int]]) -> str | None:
    return next((p for p in dropped if name.startswith(p)), None)


def _narrow_state_dict(state_dict: dict, dropped: dict[str, list[int]]):
    """Narrow first-layer weights and observation normalizers in place."""
    for key, value in list(state_dict.items()):
        name = key.removeprefix("terms.")
        prefix = _prefix(name, dropped)
        first_layer = name in (_STATE, _ACTOR, _CRITIC)
        normalizer = name.endswith(("obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std"))
        if prefix is not None and (first_layer or normalizer):
            state_dict[key] = _drop(value, dropped[prefix])


def convert(checkpoint: dict) -> dict:
    """Narrow the networks, the legacy actor and critic state dicts, and the PPO optimizer's Adam moments."""
    dropped = _dropped_columns(checkpoint["network_state_dict"])
    for key in ("network_state_dict", "actor_state_dict", "critic_state_dict"):
        if key in checkpoint:
            _narrow_state_dict(checkpoint[key], dropped)

    optimizer_state = checkpoint.get("optimizer_state_dict", {}).get("state", {})
    for i, name in enumerate(checkpoint.get("optimizer_parameter_names", [])):
        if name in (_STATE, _ACTOR, _CRITIC) and i in optimizer_state:
            state = optimizer_state[i]
            for moment in ("exp_avg", "exp_avg_sq"):
                if moment in state:
                    state[moment] = _drop(state[moment], dropped[_prefix(name, dropped)])
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", help="Checkpoint trained on the waypoint task.")
    parser.add_argument("target", help="Where to write the narrowed checkpoint.")
    args = parser.parse_args()

    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    networks = checkpoint["network_state_dict"]
    before = [networks[f"terms.{name}"].shape[-1] for name in (_STATE, _ACTOR, _CRITIC)]
    convert(checkpoint)
    after = [networks[f"terms.{name}"].shape[-1] for name in (_STATE, _ACTOR, _CRITIC)]
    torch.save(checkpoint, args.target)
    print(f"map encoder / actor / critic inputs {before} -> {after}; wrote {args.target}")


if __name__ == "__main__":
    main()
