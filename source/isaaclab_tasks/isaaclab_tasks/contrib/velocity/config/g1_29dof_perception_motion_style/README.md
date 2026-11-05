# G1 perception stair climbing with a motion-matching style reward

`IsaacContrib-Velocity-Rough-G1-29dof-Perception-MotionStyle` is a copy of
[`g1_29dof_perception`](../g1_29dof_perception) (scene, sensors, AME networks, PPO settings, gait and
regularization rewards) with three changes:

1. No velocity command: the goal is the center of the pyramid's top landing.
2. The terrain is a pyramid shaped like the reference clips' staircase, randomized per tile.
3. A motion-matching style reward pays for looking like the reference climb, without requiring an exact match.

```bash
uv run --no-sync isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Rough-G1-29dof-Perception-MotionStyle --num_envs 4096
```

`-Play` evaluates every pyramid level without promotion or pushes. The `Flat` variants place the same goal
on a plane. The `-Depth` IDs are the same aliases as in the perception task.

## Goal and episodes

Each episode starts at rest on the flat ground around the pyramid, 2.9–3.9 m (Chebyshev) from the center,
facing the center within ±0.5 rad, as the clips start at the foot of a face walking toward the center. The goal
is the landing center (±0.2 m) at 0.75 m above the landing. The `command` observation (actor and critic) is the
goal relative to the root in the root yaw frame, `(x, y, z)` in meters.

Gait terms inherited from the perception task gate on a velocity command; they read `waypoint_gait`, which
reports 0.8 m/s toward the goal and zero within the 0.3 m reach radius. It is neither observed nor tracked.
The goal command, reset, goal rewards, and curriculum come from
[`g1_29dof_perception_waypoint`](../g1_29dof_perception_waypoint).

## Terrain

The clips' staircase has six rises (0.16–0.23 m, mean 0.19 m) and five step widths (0.18–0.23 m, mean
0.20 m) to a 2.5 m square landing, with a 4.5 m base. `ClipPyramidStairsTerrainCfg` builds the same pyramid
centered in each 9 m tile:

| Quantity | Value |
| --- | --- |
| Rises | 6, mean from 0.05 m (row 0) to 0.22 m (last row), ±0.03 m per rise |
| Step width | One draw per tile from 0.18–0.26 m, ±0.02 m per step |
| Landing | 2.5 m square |

Difficulty 0.8 matches the clips' mean rise. There are 10 rows; environments start on rows 0–2, move up a row
after reaching the goal (0.3 m horizontally, 0.2 m in height), and move down after ending over half their
start distance away.

## Rewards

The goal terms replace velocity and heading tracking and the world-height target:

| Term | Weight | Behavior |
| --- | ---: | --- |
| `waypoint_position`, `waypoint_position_fine` | 4, 4 | `exp(-‖e‖² / std²)` of the 3D error, std 2.0 m and 0.5 m |
| `waypoint_height` | 2 | `exp(-e_z² / 0.1²)` within the reach radius |
| `waypoint_progress` | 2 | Rate the horizontal distance shrinks while facing the goal, clipped to ±1 m/s |
| `waypoint_bearing` | 1 | `1 - abs(yaw error) / π` while away from the goal |
| `waypoint_stand_still` | -1 | Planar speed and yaw rate within the reach radius |
| `motion_style` | 2 | Motion-matching style, below |

The other perception rewards and the `feet_air_time` and `foot_clearance` weight ramps are kept.

### `motion_style`

[`MotionMatchingStyle`](mdp/motion_style.py) uses the goal-conditioned task's
[pyramid climbs](../g1_29dof_goalconditioned_perception/data/supplied_stairs_pyramid_stand). Its library is
the 174 captured climb frames, which are the same in all four rotated clips, plus their left-right mirror:
348 frames. The synthesized settle and standing hold that follow are excluded, because a robot at rest
would match them. Each step:

1. The robot is matched to its nearest library frame (which frame of which clip) by
   `sqrt(mean(Δq²) + 0.1² · mean(Δq̇²))`.
2. If that distance is within 0.25, the reward scores the RMS joint-position error `e_g` [rad] of each joint group to
   that same frame: `exp(-Σ_g (max(e_g - inner_g, 0) / std_g)²)`, which is `1` while every group is inside its
   inner margin. Otherwise it pays zero.

| Group | Joints | Inner margin | Falloff std |
| --- | --- | ---: | ---: |
| `legs` | hips, knees, ankles (12) | 0.15 rad | 0.15 rad |
| `upper_body` | waist, shoulders, elbows, wrists (17) | 0.04 rad | 0.06 rad |

The wide leg margin keeps the reward constant while the legs adapt their placement and lift to new stairs; the
tight upper-body margin keeps the torso and arm posture close to the clip. With the clip replayed plus noise, legs
at 0.10 rad RMS still score 1.0 and at 0.20 rad 0.88, while the upper body at 0.08 rad scores 0.66 and at 0.12 rad
0.23. Only joint state is used: no root pose, world pose, or terrain, so the
term applies on any staircase. A robot standing at rest is at least 0.32 from every climb frame, so the term
pays nothing until the policy moves like the clip.

## Staged task

`IsaacContrib-Velocity-Rough-G1-29dof-Perception-MotionStyle-Staged` trains in two stages with one policy
([`staged_env_cfg.py`](staged_env_cfg.py)):

| Stage | Environments | Terrain | Trains like |
| --- | --- | --- | --- |
| Iterations 0–3000 | Half `flat` | Flat tiles | This package without `motion_style`: walk to the tile center |
| Iterations 0–3000 | Half `stairs` | The reference clips' own pyramid | [`g1_29dof_goalconditioned_perception`](../g1_29dof_goalconditioned_perception): pure imitation |
| Iteration 3000 on | All `pyramid` | Clip-shaped pyramids, curriculum on rise | This package: the same goal, now on the landing, with `motion_style` |

Flat and pyramid environments share one waypoint command: start at rest 2.9–3.9 m from the tile center facing it,
goal 0.75 m above the tile origin. On a flat tile that is the ground at the center; on a pyramid it is the landing
center, so the switch only moves the goal on top of the pyramid.

`EnvModeCommand` ([`mdp/env_modes.py`](mdp/env_modes.py)) holds each environment's mode. Its reset event runs
first: before control step 3000 × 24 an environment keeps its initial mode, and from then on it becomes `pyramid`
at its next reset and moves to a pyramid tile on rows 0–2. One terrain generator holds all three tile types
(5 flat, 5 reference-pyramid, and 10 clip-shaped pyramid columns, 10 rows). `Curriculum/env_modes/*` logs the
switch, the fraction of environments in each mode, and the mean pyramid level.

Each mode has its own reward and termination terms, wrapped in `ModeGated` under a `flat_`, `stairs_`, or `pyramid_`
prefix and zero outside their mode: `flat_` rewards are this package's goal and gait terms without `motion_style`,
`flat_` terminations are the perception task's, and `stairs_` and `pyramid_` terms come from the goal-conditioned task
and this package. Resets, pushes, and curricula are gated the same way with `ModeGatedEvent`. The gait-weight ramps
apply to the `flat_` and `pyramid_` terms. Stairs
environments only imitate (`imitation_probability = 1.0`): every episode starts on a reference frame and tracks
the clip, so the goal-conditioned task-only rewards stay zero. Imitation episodes end on the clip's last frame;
the 10 s time-out is a fallback. Flat episodes also end at 10 s; pyramid episodes run 20 s.

The actor's `command` is the goal `(x, y, z, sin, cos)` in every mode: the destination in the root yaw frame and
the yaw error to the goal heading. There is no velocity command. Flat and pyramid goals use the approach direction
as heading, like the clips' endings, so the actor cannot tell the three goals apart and its goal-reaching transfers
at the switch. The critic also
sees the one-hot mode and the goal-conditioned reference state (zero outside stairs imitation). Actions, policy
observations, physics, and startup randomization are this package's (the perception task's).

The switch counts control steps, so it assumes `num_steps_per_env = 24`, and a resumed run restarts the count.
`-Play` starts every environment on a pyramid of any level.
