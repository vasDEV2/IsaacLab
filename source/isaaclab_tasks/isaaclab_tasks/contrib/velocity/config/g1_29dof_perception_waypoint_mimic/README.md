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

### Starting from a waypoint-task checkpoint

This task drops two inputs the waypoint task has: the root position at the end of `policy` and the waypoint
height in `command`. A `g1_29dof_perception_waypoint` checkpoint therefore does not load directly. Narrow it
first. The kept inputs mean what they did, so the converted policy is a close warm start, though not identical,
since the dropped inputs had nonzero weights:

```bash
uv run --no-sync python \
  source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/g1_29dof_perception_waypoint_mimic/convert_waypoint_checkpoint.py \
  path/to/model_21000.pt path/to/model_21000_mimic.pt
```

Then pass the converted file with `--checkpoint`.

## Terrain and episodes

Every 14 m tile holds the clips' pyramid (six stacked boxes, 1.15 m high with a 2.5 m top landing),
centered, with flat ground around it. There is no terrain curriculum.

Each episode picks one of the four clips (one per side of the pyramid). The waypoint is the clip's
first root pose at the base of the pyramid: its position and its heading. The robot spawns at rest on the
ground 2–4 m behind that pose and ±1.0 m to the side, with a random yaw. The waypoint counts as reached
within 0.6 m horizontally and 0.2 m in height. The heading is not checked, since the policy does not observe it.

The mode of an episode is set when it resets:

| Mode | Episodes | Rewards |
| --- | --- | --- |
| `walk` | reset before iteration 3000 | waypoint task rewards |
| `approach` | reset from iteration 3000 | same waypoint, no positive reward at all: only penalties until the base is reached |
| `imitate` | an `approach` episode once it reaches the waypoint | the goal-conditioned task's imitation rewards, plus position rewards toward the top waypoint |

In `imitate` the clip plays from its first frame with the goal-conditioned task's imitation rewards
(anchor pose and velocity, body pose, and foot velocity) and reference terminations (anchor height,
anchor tilt, foot height). The episode ends as a time-out on the clip's last frame.

The reference terminations start loose when imitation is switched on (iteration 3000) and tighten linearly
over the next 2000 iterations (`TERMINATION_DECAY_ITERATION`), then keep the goal-conditioned task's values.
The schedule uses the step counter, so a resumed run starts loose again:

| Termination | Iteration 3000 | From iteration 5000 |
| --- | --- | --- |
| `anchor_pos` (pelvis height error) | 0.6 m | 0.25 m |
| `ee_body_pos` (foot height error) | 0.6 m | 0.25 m |
| `anchor_ori` (tilt) | 1.5 | 0.8 |

When imitation starts, the commanded waypoint moves to the top of the pyramid: the clip's last root pose,
standing on the top landing (position and heading). `top_position` and `top_position_fine` reward it
while imitating, with the waypoint task's position kernels (`exp(-‖e‖² / std²)` of the 3D error, std 2.0 m
and 0.5 m, weight 4 each), so the climb is imitation plus a pull toward the landing.

`base_reached` pays a one-off bonus of 100 (`BASE_REACHED_BONUS` in `env_cfg.py`) on first reaching the base
waypoint in `approach` (not in `walk`). It is the only positive reward before imitation. The reward
manager scales every term by the step time, so its weight is `100 / step_dt` (5000).

The waypoint task's gait, posture, and style penalties are active in `walk` and `approach` only, along with
`backward_velocity` (weight -2, `min(v_x, 0)²` of the body-frame forward velocity), which penalizes walking
backward, and `max_speed` (weight -2, `max(|v_xy| - 1, 0)²` of the horizontal root speed), which discourages
moving faster than 1 m/s. In `walk`, the waypoint task's `waypoint_bearing` and facing-scaled `waypoint_progress` also favour
walking forward toward the waypoint. The positive gait rewards (`feet_air_time`, `foot_clearance`) are active in
`walk` only. Its
regularization terms (termination penalty, undesired contacts, action rate, energy, joint velocity,
acceleration, and limits) apply in every mode. The goal-conditioned task's imitation sampling schedule,
task episodes, and reset curriculum are not used.

`IMITATION_ITERATION` and `STEPS_PER_ITERATION` in `env_cfg.py` set the switch. The command counts control
steps, so `STEPS_PER_ITERATION` must match the runner's `num_steps_per_env` (24). The step counter starts
at zero on every launch, including resumed runs.

## Observations

The only positional input is the `command` group (actor and critic): the waypoint's XY relative to the root,
in the root yaw frame. The `policy` group is the perception proprioception, without the waypoint task's root
position. The policy sees no target heading, waypoint height, or imitation state, so it cannot tell `walk`,
`approach`, and `imitate` apart. It sees only the waypoint, which jumps to the top when imitation starts.

`Metrics/waypoint/{walk,approach,imitate}` are the fractions of ending episodes in each mode.
`Metrics/waypoint/imitation_phase` is the clip phase at the end; 1.0 means the clip played out.
`Metrics/waypoint/top_reached` is the fraction of ending episodes that reached the top waypoint.
`-Play` approaches and imitates from the first step, without pushes.

## Split task

`IsaacContrib-Velocity-Rough-G1-29dof-Perception-Waypoint-Mimic-Split` uses the same terrain, observations, and
networks, in three stages by the iteration at which an episode resets. `-Split-Play` runs only the last stage
(approach, then imitate) from the first step, without pushes:

| Episodes reset | Mode | Start | Waypoint | Rewards |
| --- | --- | --- | --- | --- |
| Before iteration 3000 | `walk` | At rest, 2–4 m behind the clip's start | Pyramid base | Waypoint task, as in the base task's `walk` |
| 3000–5000, 50% | `walk` | At rest, 2–4 m behind the clip's start | Pyramid base | Waypoint task, as in the base task's `walk` |
| 3000–5000, 50% | `imitate` | A random frame of the clip, or the clip's beginning from a recorded end-of-walk state (`full_clip_probability`) | Clip's end on the top landing | Goal-conditioned task's imitation rewards |
| From 5000 | `approach`, then `imitate` | At rest, 2–4 m behind the clip's start | Pyramid base, then the clip's end | As in the base task: penalties and the 100 base bonus while approaching; then the imitation rewards and `top_position*` |

The imitation rewards are the goal-conditioned task's seven tracking terms and its own regularization
(`imitation_alive`, `imitation_joint_acc`, `imitation_joint_torque`, `imitation_action_rate_l2`,
`imitation_joint_limit`, `imitation_undesired_contacts`, `imitation_feet_slide`), with its weights, in every
imitation. `top_position` and `top_position_fine` pay only in imitations that followed an approach. Reference joints
get ±0.1 rad of noise at a reference start.

When a walk episode first reaches the base waypoint, its root and joint state are recorded relative to the waypoint
and the clip's start heading. Imitation episodes that start at the clip's first frame begin from one of these states,
moved to their own clip, so they begin as a walk ends. Until 64 states exist (`walk_end_min_states`) they use the
clip's first pose. `Metrics/waypoint/walk_end_start` is the fraction of ending episodes that began this way.

Imitation ends on the clip's last frame or at the reference thresholds:

| Iteration | Pelvis height | Tilt | Feet height |
| --- | --- | --- | --- |
| Before 5000 | 0.25 m | 0.8 | 0.25 m |
| 5000 | 0.6 m | 1.5 | 0.6 m |
| From 7000 | 0.25 m | 0.8 | 0.25 m |

Between 5000 and 7000 they tighten linearly, as in the base task after its switch.

`SPLIT_IMITATION_ITERATION` and `SPLIT_APPROACH_ITERATION` in `env_cfg.py` set the stages, and the command's
`imitation_probability` and `full_clip_probability` set the split. The step counter restarts at zero on every
launch, so a resumed run starts again from the first stage. Runs log to
`g1_29dof_perception_ame_waypoint_mimic_split`.

### Pyramid curriculum stage

From `SPLIT_CURRICULUM_ITERATION`, a fraction `CURRICULUM_PROBABILITY` of the resetting episodes run the pyramid
curriculum and the rest approach and imitate on the clip's pyramid as before. The terrain is a grid of 17 rows
(levels) by 20 columns: columns 0–3 hold the clip's pyramid, the others the clip's pyramid with every box widened by
the tile's top width change and its heights scaled by the tile's step scale (`PYRAMID_CURRICULUM` in `env_cfg.py`,
`PyramidCurriculumCfg` in `mdp/terrains.py`):

| Levels | Top width (clip: 2.5 m) | Step heights |
| --- | --- | --- |
| 0 | The clip's | The clip's |
| 1–8 | Spread around the clip's over a range that opens to 1.0–4.0 m | The clip's |
| 9–16 | Random over 1.0–4.0 m | Spread around the clip's over a range that opens to 0.63–1.58 times |

Each curriculum column has its own fixed place in the spread, so on every level some pyramids are smaller than the
clip's and some larger, and the differences grow with the level. A robot's level moves on its curriculum episodes
only: up, with probability 0.25, after reaching the top waypoint; down after ending before half the clip without
reaching it. `CURRICULUM_REPLAY_PROBABILITY` of curriculum episodes run on a random level at or below the robot's.
Levels restart at 0 on every launch. `Curriculum/pyramid_tiles/level_mean` and `level_max` log the levels.

A curriculum episode's only waypoint, the one the policy observes, is the center of the top landing (at the
clip's end height) from the start. The base waypoint is kept internally: when the root reaches the base line of
the face the clip climbs (within the reach radius of it, anywhere along it), the episode imitates in
`imitate_loose`, with the clip moved sideways along the face onto the robot. The clip is moved onto the
pyramid: out along its face by half the top width change, and up by its pyramid height under the root (smoothed)
times the step scale minus one. The strict tracking terms are off; the `loose_*` terms pay their full weight anywhere
within a tolerance of this reference (anchor 0.4 m and 0.5 rad, mean body 0.2 m and 0.5 rad), so tracking it more
closely pays nothing more. The reference terminations are fixed at 0.5 m pelvis height, 1.5 tilt, and 0.5 m feet
height. The `imitation_*` regularization terms and `top_position*` apply as in the strict imitation. Spawns in front
of wider pyramids are shortened to stay within 6.4 m of the pyramid center.
