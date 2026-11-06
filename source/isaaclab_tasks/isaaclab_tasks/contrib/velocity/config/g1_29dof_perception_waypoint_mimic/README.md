# G1 perception waypoint to pyramid base, then imitation

`IsaacContrib-Velocity-Rough-G1-29dof-Perception-Waypoint-Mimic` is the
[`g1_29dof_perception_waypoint`](../g1_29dof_perception_waypoint) task on the pyramid of the
`supplied_stairs_pyramid_stand` clips (stored with
[`g1_29dof_goalconditioned_perception`](../g1_29dof_goalconditioned_perception/data/supplied_stairs_pyramid_stand)).
After 3000 iterations, reaching the waypoint starts imitation of the clip up the pyramid.

```bash
uv run --no-sync isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Rough-G1-29dof-Perception-Waypoint-Mimic --num_envs 4096
```

## Terrain and episodes

Every 9 m tile holds the clips' pyramid (six stacked boxes, 1.15 m high with a 2.5 m top landing),
centered, with flat ground around it. There is no terrain curriculum.

Each episode picks one of the four clips (one per side of the pyramid). The waypoint is the clip's
first root pose at the base of the pyramid: its position and its heading. The robot spawns at rest on the
ground 0.5–1.5 m behind that pose and ±1.0 m to the side, with a random yaw. The waypoint counts as reached
within 0.3 m horizontally, 0.2 m in height, and 0.4 rad in heading.

The mode of an episode is set when it resets:

| Mode | Episodes | Rewards |
| --- | --- | --- |
| `walk` | reset before iteration 3000 | waypoint task rewards, plus `waypoint_heading` (weight 2, std 0.5 rad, within the reach radius) |
| `approach` | reset from iteration 3000 | same waypoint, no waypoint rewards |
| `imitate` | an `approach` episode once it reaches the waypoint | the goal-conditioned task's imitation rewards, plus position rewards toward the top waypoint |

In `imitate` the clip plays from its first frame with the goal-conditioned task's imitation rewards
(anchor pose and velocity, body pose, and foot velocity) and reference terminations (anchor height,
anchor tilt, foot height). The episode ends as a time-out on the clip's last frame.

When imitation starts, the commanded waypoint moves to the top of the pyramid: the clip's last root pose,
standing on the top landing (position and heading). `top_position` and `top_position_fine` reward it
while imitating, with the waypoint task's position kernels (`exp(-‖e‖² / std²)` of the 3D error, std 2.0 m
and 0.5 m, weight 4 each), so the climb is imitation plus a pull toward the landing.

The waypoint task's gait, posture, and style terms are active in `walk` and `approach` only. Its
regularization terms (termination penalty, undesired contacts, action rate, energy, joint velocity,
acceleration, and limits) apply in every mode. The goal-conditioned task's imitation sampling schedule,
task episodes, and reset curriculum are not used.

`IMITATION_ITERATION` and `STEPS_PER_ITERATION` in `env_cfg.py` set the switch. The command counts control
steps, so `STEPS_PER_ITERATION` must match the runner's `num_steps_per_env` (24). The step counter starts
at zero on every launch, including resumed runs.

## Observations

The `command` group (actor and critic) is the waypoint relative to the root in the root yaw frame,
the sine and cosine of the heading error, whether the clip is playing, and the clip phase (0–1, zero
before imitation). The `policy` group ends with the root position relative to the pyramid center.

`Metrics/waypoint/{walk,approach,imitate}` are the fractions of ending episodes in each mode.
`Metrics/waypoint/imitation_phase` is the clip phase at the end; 1.0 means the clip played out.
`Metrics/waypoint/top_reached` is the fraction of ending episodes that reached the top waypoint.
`-Play` approaches and imitates from the first step, without pushes.
