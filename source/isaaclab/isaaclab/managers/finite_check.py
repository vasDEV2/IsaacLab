# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for opt-in finite-value diagnostics in manager pipelines."""

import logging

import torch

logger = logging.getLogger(__name__)


def check_finite_tensor(value: torch.Tensor, context: str) -> None:
    """Raise an informative error when a batched tensor contains non-finite values.

    Args:
        value: Tensor whose first dimension indexes environments.
        context: Name of the pipeline value being checked.

    Raises:
        ValueError: If any value is NaN or infinite.
    """
    finite = torch.isfinite(value)
    if finite.all():
        return
    if value.ndim == 0:
        raise ValueError(f"{context} contains non-finite values.")

    bad_envs = torch.nonzero(~finite.reshape(value.shape[0], -1).all(dim=1)).flatten()
    first_env = int(bad_envs[0].item())
    bad_indices = torch.nonzero(~finite[first_env].reshape(-1)).flatten()[:8].tolist()
    raise ValueError(
        f"{context} contains non-finite values for environment(s) {bad_envs[:8].tolist()}; "
        f"flattened feature indices for the first environment are {bad_indices}."
    )


def sanitize_finite_tensor(value: torch.Tensor, context: str, warned: set[str]) -> torch.Tensor:
    """Replace non-finite tensor entries with zero and warn once per pipeline context.

    Args:
        value: Tensor whose first dimension indexes environments.
        context: Name of the pipeline value being sanitized.
        warned: Contexts already reported by the owning manager.

    Returns:
        The original tensor when all values are finite, otherwise a sanitized copy.
    """
    finite = torch.isfinite(value)
    if finite.all():
        return value

    if context not in warned:
        warned.add(context)
        if value.ndim == 0:
            logger.warning("%s contains non-finite values; replacing them with zero.", context)
        else:
            bad_envs = torch.nonzero(~finite.reshape(value.shape[0], -1).all(dim=1)).flatten()
            first_env = int(bad_envs[0].item())
            bad_indices = torch.nonzero(~finite[first_env].reshape(-1)).flatten()[:8].tolist()
            logger.warning(
                "%s contains non-finite values for environment(s) %s; first environment feature indices %s. "
                "Replacing non-finite values with zero.",
                context,
                bad_envs[:8].tolist(),
                bad_indices,
            )
    return torch.where(finite, value, torch.zeros_like(value))
