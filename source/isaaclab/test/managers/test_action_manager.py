# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for action-manager diagnostics."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers.action_manager import ActionManager

pytestmark = pytest.mark.unit


class _ActionTerm:
    action_dim = 2

    def __init__(self, *, produce_nonfinite: bool):
        self.produce_nonfinite = produce_nonfinite
        self.raw_actions = torch.zeros(2, 2)
        self.processed_actions = torch.zeros(2, 2)

    def process_actions(self, actions: torch.Tensor) -> None:
        self.raw_actions[:] = actions
        self.processed_actions[:] = actions
        if self.produce_nonfinite:
            self.processed_actions[1, 0] = torch.inf


@pytest.mark.parametrize(
    ("nonfinite_input", "nonfinite_processed", "context"),
    [
        (True, False, "Incoming action tensor from the policy"),
        (False, True, "Action term 'joint_position' processed actions"),
    ],
)
def test_nonfinite_action_diagnostic_identifies_stage_and_environment(nonfinite_input, nonfinite_processed, context):
    """Finite checks distinguish a bad policy output from action-term processing."""
    action = torch.zeros(2, 2)
    if nonfinite_input:
        action[1, 0] = torch.nan

    manager = ActionManager.__new__(ActionManager)
    manager._check_finite = True
    manager._sanitize_nonfinite = False
    manager._total_action_dim = 2
    manager._action = torch.zeros_like(action)
    manager._prev_action = torch.zeros_like(action)
    manager._env = SimpleNamespace(device="cpu")
    manager._resolve_terms_handle = None
    manager._terms = {
        "joint_position": _ActionTerm(produce_nonfinite=nonfinite_processed),
    }

    with pytest.raises(ValueError, match=rf"{context}.*environment\(s\) \[1\]"):
        manager.process_action(action)


@pytest.mark.parametrize("nonfinite_stage", ["policy", "processed"])
def test_nonfinite_action_sanitization_zeros_invalid_values(nonfinite_stage, caplog):
    """Sanitization prevents non-finite policy or processed actions reaching simulation."""
    action = torch.zeros(2, 2)
    if nonfinite_stage == "policy":
        action[1, 0] = torch.nan

    term = _ActionTerm(produce_nonfinite=nonfinite_stage == "processed")
    manager = ActionManager.__new__(ActionManager)
    manager._check_finite = False
    manager._sanitize_nonfinite = True
    manager._nonfinite_warnings = set()
    manager._total_action_dim = 2
    manager._action = torch.zeros_like(action)
    manager._prev_action = torch.zeros_like(action)
    manager._env = SimpleNamespace(device="cpu")
    manager._resolve_terms_handle = None
    manager._terms = {"joint_position": term}

    manager.process_action(action)

    assert torch.isfinite(manager.action).all()
    assert torch.isfinite(term.raw_actions).all()
    assert torch.isfinite(term.processed_actions).all()
    assert term.processed_actions[1, 0] == 0.0
    assert "Replacing non-finite values with zero" in caplog.text
