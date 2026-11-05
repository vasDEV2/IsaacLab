# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play an approach checkpoint and print the reward terms it earns.

Takes the same arguments as ``isaaclab play --rl_library rsl_rl``, plus ``--print_env``, ``--print_every``,
``--all_terms``, and ``--walk``. The play task's episodes approach and climb; ``--walk`` plays the first stage's
rewarded walk to the waypoint instead. Each printout shows the printed environment's phase, its distance to the
active target (the waypoint before climbing, the destination after), and for each reward term the weighted
per-step value (weight x term, before the ``step_dt`` scaling), its episode sum so far (with ``step_dt``), and
the mean over all environments. Terms of the other phases are zero and hidden unless ``--all_terms`` is
passed. A summary prints when that environment's episode ends.

.. code-block:: bash

    uv run --no-sync python -m \\
        isaaclab_tasks.contrib.velocity.config.g1_29dof_goalconditioned_approach_perception.play_rewards \\
        --task IsaacContrib-Velocity-Rough-G1-29dof-GoalConditioned-Approach-Perception-Play \\
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

from .mdp import CLIMB, PHASE_NAMES

logger = logging.getLogger(__name__)


def _split_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--print_env", type=int, default=0, help="Environment whose rewards are printed.")
    parser.add_argument("--print_every", type=int, default=25, help="Control steps between printouts.")
    parser.add_argument("--all_terms", action="store_true", help="Also print the other phases' (zero) terms.")
    parser.add_argument("--walk", action="store_true", help="Play rewarded walk episodes instead of approach + climb.")
    return parser.parse_known_args(argv)


def _print_rewards(base, env_id: int, episode_sum: torch.Tensor, phase: int, header: str, all_terms: bool) -> None:
    manager = base.reward_manager
    step = manager._step_reward
    command = base.command_manager.get_term("motion")
    print(f"\n{header}  env {env_id}  t={float(base.episode_length_buf[env_id]) * base.step_dt:5.2f} s")
    # after an episode ends the environment has already reset, so the command state is the next episode's
    if "END" in header:
        print(f"  ended in phase {PHASE_NAMES[phase]}")
    elif phase == CLIMB:
        error = command.goal_error[env_id]
        kind = "imitation" if bool(command.is_imitation[env_id]) else "task"
        print(
            f"  phase climb ({kind}, frame {int(command.time_steps[env_id])}/{int(command.motion_lengths[env_id])}):"
            f" destination distance {float(error[:2].norm()):.2f} m, height error {float(error[2]):+.2f} m,"
            f" heading error {float(command.heading_error[env_id]):+.2f} rad"
        )
    else:
        print(
            f"  phase {PHASE_NAMES[phase]}: waypoint distance {float(command.distance_xy[env_id]):.2f} m,"
            f" height error {float(command.height_error[env_id]):+.2f} m,"
            f" bearing {float(command.bearing_error[env_id]):+.2f} rad,"
            f" reached {bool(command.waypoint_reached[env_id])}"
        )
    prefix = {0: "walk_", 2: "climb_"}.get(phase)
    print(f"  {'term':<34}{'weighted/step':>14}{'episode sum':>13}{'mean all envs':>15}")
    for index, name in enumerate(manager.active_terms):
        if not all_terms and (prefix is None or not name.startswith(prefix)) and not episode_sum[env_id, index]:
            continue
        print(
            f"  {name:<34}{float(step[env_id, index]):>14.3f}{float(episode_sum[env_id, index]):>13.3f}"
            f"{float(step[:, index].mean()):>15.3f}"
        )
    print(
        f"  {'TOTAL':<34}{float(step[env_id].sum()):>14.3f}{float(episode_sum[env_id].sum()):>13.3f}"
        f"{float(step.sum(dim=1).mean()):>15.3f}"
    )


def main() -> None:
    extra, argv = _split_args(sys.argv[1:])
    args_cli = _parse_args(argv)
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=not args_cli.train_env_cfg)
    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        apply_env_overrides(args_cli, env_cfg)
        if extra.walk:
            # climbing starts at this control step, which playback never reaches
            env_cfg.commands.motion.approach_steps = 2**62
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
        command = base.command_manager.get_term("motion")
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
            episode_sum.add_(base.reward_manager._step_reward, alpha=base.step_dt)
            # a finished episode has already reset; its phase is kept as the finished phase
            if dones[env_id]:
                phase = int(command._finished_phase[env_id])
                _print_rewards(base, env_id, episode_sum, phase, "=== EPISODE END", extra.all_terms)
            elif count % extra.print_every == 0:
                phase = int(command.episode_phase[env_id])
                _print_rewards(base, env_id, episode_sum, phase, f"--- step {count}", extra.all_terms)
            episode_sum[dones.bool()] = 0.0

        run_playback(step, dt=base.step_dt, args_cli=args_cli, env_cfg=env_cfg)


if __name__ == "__main__":
    main()
