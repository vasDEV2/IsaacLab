# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay the stair reference clips through the task and report the reward each term accumulates.

Episodes run in imitation mode from clip frame 0, on a randomly chosen one of the four pyramid
approaches. Each control step sends the next reference frame as the joint-position target, so the
rate penalties see the clip's own motion.

* ``teleport`` (default): before each step, write the previous reference frame's root and joint state.
  Physics then integrates one 20 ms step, so contact-based terms see real contacts while the robot
  cannot drift from the clip. This approximates the reward an ideal tracker would earn.
* ``track``: write only the initial state, then follow the reference with PD targets alone.

Pass ``--loop`` to replay until the viewer closes or Ctrl+C. Run from the IsaacLab root::

    uv run --no-sync python -m \\
        isaaclab_tasks.contrib.velocity.config.g1_29dof_goalconditioned_perception.replay_motion --viz newton_gl
"""

import argparse
import csv
import itertools
import sys

import warp as wp

wp.config.enable_backward = False

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli  # noqa: E402

TASK = "IsaacContrib-Velocity-Rough-G1-29dof-GoalConditioned-Perception-Play"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("teleport", "track"), default="teleport")
    parser.add_argument("--episodes", type=int, default=1, help="Clip passes to replay.")
    parser.add_argument("--loop", action="store_true", help="Replay until the viewer closes or Ctrl+C.")
    parser.add_argument("--csv", default=None, help="Optional per-step weighted reward log.")
    add_launcher_args(parser)
    args, hydra_args = setup_preset_cli(parser, sys.argv[1:])
    # remaining arguments are Hydra overrides of the env config, e.g. env.rewards.motion_joint_pos.weight=5.0
    sys.argv = [sys.argv[0]] + hydra_args

    env_cfg, _ = resolve_task_config(TASK, "")
    env_cfg.scene.num_envs = 1
    # imitate full clips without reset noise; the -Play task otherwise evaluates task episodes only
    env_cfg.commands.motion.imitation_probability = 1.0
    env_cfg.commands.motion.full_clip_probability = 1.0
    env_cfg.commands.motion.debug_vis = False
    # --viz arrives as a Hydra preset; run headless without it
    if not any(arg.startswith("visualizer=") for arg in hydra_args):
        env_cfg.sim.visualizer_cfgs = []

    with launch_simulation(env_cfg, args):
        env = gym.make(TASK, cfg=env_cfg)
        rows = []
        episodes = itertools.count() if args.loop else range(args.episodes)
        try:
            for episode in episodes:
                if not env.unwrapped.sim.is_running():
                    break
                replay_episode(env, args.mode, episode, rows)
        except KeyboardInterrupt:
            print("\nStopped.")

        if args.csv and rows:
            with open(args.csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f"Wrote {len(rows)} steps to {args.csv}")
        env.close()


def replay_episode(env: gym.Env, mode: str, episode: int, rows: list[dict]):
    """Replay one clip pass from frame 0, appending per-step rewards to ``rows`` and printing the totals."""
    base = env.unwrapped
    motion = base.command_manager.get_term("motion")
    robot = base.scene["robot"]
    rewards = base.reward_manager
    action_term = base.action_manager.get_term("joint_pos")
    joint_ids = action_term._joint_ids
    scale = action_term.cfg.scale
    dt = base.step_dt
    env_ids = torch.arange(base.num_envs, device=base.device)

    env.reset()
    approach = int(motion.motion_ids[0]) * 90
    names = list(rewards.active_terms)
    totals = dict.fromkeys(names, 0.0)
    ret, steps, reason = 0.0, 0, "none"
    while base.sim.is_running():
        frame = motion.time_steps.clone()
        if mode == "teleport":
            clip = motion.motion
            ref_rows = clip.frame_indices(motion.motion_ids, (frame - 1).clamp(min=0))
            root = motion.root_index
            root_pose = torch.cat(
                (clip.body_pos_w[ref_rows, root] + base.scene.env_origins, clip.body_quat_w[ref_rows, root]), -1
            )
            root_vel = torch.cat((clip.body_lin_vel_w[ref_rows, root], clip.body_ang_vel_w[ref_rows, root]), -1)
            robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
            robot.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)
            robot.write_joint_position_to_sim_index(position=clip.joint_pos[ref_rows], env_ids=env_ids)
            robot.write_joint_velocity_to_sim_index(velocity=clip.joint_vel[ref_rows], env_ids=env_ids)
        target = motion.joint_pos[:, joint_ids] - robot.data.default_joint_pos.torch[:, joint_ids]
        _, reward, terminated, truncated, _ = env.step(target / scale)

        # the per-term weighted values survive the auto-reset; scale by dt as training does
        step = {name: value[0] * dt for name, value in rewards.get_active_iterable_terms(0)}
        for name, value in step.items():
            totals[name] += value
        ret += reward[0].item()
        steps += 1
        rows.append({"episode": episode, "frame": int(frame[0]), **step})
        if terminated[0] or truncated[0]:
            tm = base.termination_manager
            reason = ",".join(n for n in tm.active_terms if tm.get_term(n)[0]) or "unknown"
            break
    if steps == 0:
        return

    print(
        f"\nEpisode {episode} ({mode}, {approach} deg approach): {steps} steps ({steps * dt:.2f} s), ended by {reason}"
    )
    print(f"{'term':<28}{'weight':>10}{'sum':>12}{'per s':>10}")
    for name in sorted(names, key=lambda n: totals[n]):
        weight = rewards.get_term_cfg(name).weight
        print(f"{name:<28}{weight:>10.4g}{totals[name]:>12.4f}{totals[name] / (steps * dt):>10.4f}")
    print(f"{'total return':<28}{'':>10}{ret:>12.4f}{ret / (steps * dt):>10.4f}")


if __name__ == "__main__":
    main()
