# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the unified play entrypoint and print per-term rewards every step.

This wraps ``play.py`` without touching Isaac Lab core: it patches ``RslRlVecEnvWrapper.step`` at runtime to print the
total reward and each reward term for one environment. All normal ``play.py`` arguments are accepted, plus:

* ``--print_env_idx N``: environment to print (default 0).
* ``--print_every N``: print every N steps (default 1).

Example:
    ./isaaclab.sh -p scripts/reinforcement_learning/play_print_rewards.py \
        --task IsaacContrib-Velocity-Rough-G1-29dof-Perception-Play --print_every 10
"""

import sys
import traceback

import torch
import warp as wp

wp.config.enable_backward = False

from isaaclab_rl.entrypoints import run_play_cli  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

from isaaclab_tasks.contrib.velocity.config.g1_29dof_perception.mdp.stair_rewards import (  # noqa: E402
    _local_ground_height,
)


def _foot_clearance_report(env, env_idx: int, term_name: str = "foot_clearance") -> str:
    """Recompute every intermediate of the active foot clearance term for one env and format it.

    Supports ``vel_mdp.foot_clearance_reward`` (mean of all scanner hits as ground, no motion gate) and
    ``stair_foot_clearance`` (local ground height, optional motion gate), chosen by the configured function.
    """
    term = env.reward_manager.get_term_cfg(term_name)
    p = term.params
    func_name = getattr(term.func, "__name__", str(term.func))
    local = func_name == "stair_foot_clearance"
    asset = env.scene[p["asset_cfg"].name]
    ids = p["asset_cfg"].body_ids
    names = [asset.body_names[k] for k in ids] if not isinstance(ids, slice) else asset.body_names
    feet = asset.data.body_pos_w.torch[:, ids]
    hits = env.scene[p["height_sensor_cfg"].name].data.ray_hits_w.torch
    ground_offset = p.get("ground_height_offset", 0.0)
    radius = p.get("local_radius", 0.3)
    if local:
        ground = _local_ground_height(feet, hits, radius)
    else:
        # mean over every scanner ray, one value per env, broadcast to both feet
        ground = hits[..., 2].mean(dim=1, keepdim=True).expand(-1, feet.shape[1])
    target = p["target_height"]
    standing_z = p.get("standing_position_foot_z", 0.039)
    rel_z = feet[..., 2] - ground
    error = (rel_z - ground_offset - target - standing_z).square()
    speed = torch.linalg.vector_norm(asset.data.body_lin_vel_w.torch[:, ids, :2], dim=-1)
    speed_weight = torch.tanh(p["tanh_mult"] * speed)
    kernel = torch.exp(-(error * speed_weight).sum(dim=1) / p["std"])
    reward = kernel
    gate_text = "no motion gate"
    cmd = env.command_manager.get_command(p.get("command_name") or "base_velocity")[:, :2]
    cmd_speed = torch.linalg.vector_norm(cmd, dim=-1)
    if local and p.get("motion_gate", True):
        gate = speed_weight.mean(dim=1) * (cmd_speed > 0.05)
        reward = kernel * gate
        gate_text = f"gate(mean_w*[cmd>0.05])={gate[env_idx].item():.4f}"
    elif not local and p.get("command_name") is not None:
        # vel_mdp / g1_mdp foot_clearance_reward: zero when commanded to move but every foot is still
        commanded_to_move = cmd_speed > p.get("command_threshold", 0.05)
        feet_still = (speed <= p.get("foot_speed_threshold", 0.05)).all(dim=-1)
        zeroed = commanded_to_move & feet_still
        reward = kernel * ~zeroed
        gate_text = (
            f"zeroed(cmd>{p.get('command_threshold', 0.05)} & all feet<={p.get('foot_speed_threshold', 0.05)})"
            f"={bool(zeroed[env_idx].item())}"
        )
    i = env_idx
    out = [
        f"  [{term_name}] func={func_name} weight={term.weight:.4f} target={target} standing_z={standing_z}"
        f" std={p['std']} tanh_mult={p['tanh_mult']} cmd_speed={cmd_speed[i].item():.3f}"
        f" ground={'local r=' + str(radius) if local else 'mean of all scanner rays'}"
    ]
    for k, name in enumerate(names):
        line = (
            f"    {name:<22s} foot_z={feet[i, k, 2].item():+.4f} ground={ground[i, k].item():+.4f}"
            f" rel_z={rel_z[i, k].item():+.4f} (goal {target + standing_z + ground_offset:.4f})"
            f" err^2={error[i, k].item():.5f} speed={speed[i, k].item():.4f}"
            f" speed_w={speed_weight[i, k].item():.4f}"
        )
        if local:
            d2 = (feet[i, k, None, :2] - hits[i, :, :2]).square().sum(-1)
            n = ((d2 <= radius**2) & torch.isfinite(hits[i]).all(-1)).sum().item()
            line += f" hits={n}"
        out.append(line)
    out.append(
        f"    exponent=sum(err^2*w)={(error * speed_weight).sum(dim=1)[i].item():.5f}"
        f" kernel=exp(-exponent/std)={kernel[i].item():.4f} {gate_text}"
        f" -> reward={reward[i].item():.4f} x weight = {(reward[i] * term.weight).item():+.4f}"
    )
    return "\n".join(out)


def _pop_arg(argv: list[str], flag: str, default: int) -> int:
    """Remove ``flag value`` from ``argv`` in place and return the value."""
    if flag in argv:
        i = argv.index(flag)
        value = int(argv[i + 1])
        del argv[i : i + 2]
        return value
    return default


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    env_idx = _pop_arg(argv, "--print_env_idx", 0)
    every = _pop_arg(argv, "--print_every", 1)

    original_step = RslRlVecEnvWrapper.step
    counter = {"n": 0}

    def step_and_print(self, actions):
        result = original_step(self, actions)
        counter["n"] += 1
        if counter["n"] % every == 0:
            rew = result[1]
            manager = self.unwrapped.reward_manager
            terms = manager.get_active_iterable_terms(env_idx)
            lines = [f"[step {counter['n']}] env {env_idx} total reward: {rew[env_idx].item():+.4f}"]
            lines += [f"    {name:<30s} {values[0]:+.4f}" for name, values in terms]
            try:
                lines.append(_foot_clearance_report(self.unwrapped, env_idx))
            except Exception as exc:  # keep playback alive if the term is absent or changed
                lines.append(f"  [foot_clearance] report unavailable: {exc!r}\n{traceback.format_exc()}")
            print("\n".join(lines), flush=True)
        return result

    RslRlVecEnvWrapper.step = step_and_print
    return run_play_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
