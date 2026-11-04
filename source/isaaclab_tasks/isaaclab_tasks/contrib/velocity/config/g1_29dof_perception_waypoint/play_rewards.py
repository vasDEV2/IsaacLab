# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a waypoint checkpoint and print every reward term.

Takes the same arguments as ``isaaclab play --rl_library rsl_rl``, plus ``--print_env`` and
``--print_every``. Each printout shows the weighted per-step value (weight x term, before the
``step_dt`` scaling) of the printed environment, its episode sum so far (with ``step_dt``),
the mean over all environments, and the waypoint state. A summary prints when that environment's
episode ends.

.. code-block:: bash

    uv run --no-sync python -m \\
        isaaclab_tasks.contrib.velocity.config.g1_29dof_perception_waypoint.play_rewards \\
        --task IsaacContrib-Velocity-Rough-G1-29dof-Perception-Waypoint-Play \\
        --checkpoint /path/to/model.pt --print_env 0 --print_every 25
"""

import argparse
import contextlib
import logging
import os
import sys

import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.app import launch_simulation
from isaaclab.utils import to_dict

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from isaaclab_rl.entrypoints.backends.play_rsl_rl import _parse_args, _resolve_checkpoint
from isaaclab_rl.entrypoints.common import apply_env_overrides, close_env, create_isaaclab_env, run_playback
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_tasks.utils import resolve_task_config

logger = logging.getLogger(__name__)


def _split_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--print_env", type=int, default=0, help="Environment whose rewards are printed.")
    parser.add_argument("--print_every", type=int, default=25, help="Control steps between printouts.")
    return parser.parse_known_args(argv)


def _print_rewards(base, env_id: int, episode_sum: torch.Tensor, header: str) -> None:
    manager = base.reward_manager
    step = manager._step_reward
    command = base.command_manager.get_term("waypoint")
    levels = getattr(base.scene.terrain, "terrain_levels", None)
    level = f"  level {int(levels[env_id])}" if levels is not None else ""
    print(f"\n{header}  env {env_id}  t={float(base.episode_length_buf[env_id]) * base.step_dt:5.2f} s{level}")
    # after an episode ends the environment has already reset, so this is the next episode's waypoint
    label = "waypoint (next episode)" if "END" in header else "waypoint"
    print(
        f"  {label}: distance {float(command.distance_xy[env_id]):.2f} m,"
        f" height error {float(command.height_error[env_id]):+.2f} m,"
        f" bearing {float(command.bearing_error[env_id]):+.2f} rad,"
        f" reached {bool(command.reached[env_id])}"
    )
    print(f"  {'term':<28}{'weighted/step':>14}{'episode sum':>13}{'mean all envs':>15}")
    for index, name in enumerate(manager.active_terms):
        print(
            f"  {name:<28}{float(step[env_id, index]):>14.3f}{float(episode_sum[env_id, index]):>13.3f}"
            f"{float(step[:, index].mean()):>15.3f}"
        )
    print(
        f"  {'TOTAL':<28}{float(step[env_id].sum()):>14.3f}{float(episode_sum[env_id].sum()):>13.3f}"
        f"{float(step.sum(dim=1).mean()):>15.3f}"
    )


def main() -> None:
    extra, argv = _split_args(sys.argv[1:])
    args_cli = _parse_args(argv)
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=not args_cli.train_env_cfg)
    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        apply_env_overrides(args_cli, env_cfg)
        env_cfg.seed = agent_cfg.seed
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = _resolve_checkpoint(args_cli, agent_cfg, env_cfg, log_root_path)
        if resume_path is None:
            return
        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        cleanup.callback(lambda: close_env(env))
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        logger.info(f"Loading model checkpoint from: {resume_path}")
        runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        base = env.unwrapped
        env_id = extra.print_env
        if not 0 <= env_id < base.num_envs:
            raise ValueError(f"--print_env must be in [0, {base.num_envs}), got {env_id}.")
        episode_sum = torch.zeros(base.num_envs, len(base.reward_manager.active_terms), device=base.device)
        obs = env.get_observations()
        count = 0

        def step() -> None:
            nonlocal obs, count
            obs, _, dones, _ = env.step(policy(obs))
            policy.reset(dones)
            count += 1
            # the reward manager's step values are from the step that just ended, also for a reset environment
            episode_sum.add_(base.reward_manager._step_reward, alpha=base.step_dt)
            if dones[env_id]:
                _print_rewards(base, env_id, episode_sum, "=== EPISODE END")
            elif count % extra.print_every == 0:
                _print_rewards(base, env_id, episode_sum, f"--- step {count}")
            episode_sum[dones.bool()] = 0.0

        run_playback(step, dt=base.step_dt, args_cli=args_cli, env_cfg=env_cfg)


if __name__ == "__main__":
    main()
