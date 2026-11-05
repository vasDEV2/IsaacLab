# G1 goal-conditioned stair climbing with AME

A perceptive G1 policy that climbs a staircase to a destination root position and heading. The training
paradigm follows the goal-conditioned Mimic task on the
[`dokyun/multi-mimic` branch](https://github.com/jnskkmhr/IsaacLab/tree/dokyun/multi-mimic/source/isaaclab_tasks/isaaclab_tasks/contrib/mimic/config/g1_29dof_perception_goal_condition)
of jnskkmhr/IsaacLab, itself inspired by GfR. The actor keeps this package's AME height-map encoder.

Training mixes two kinds of episodes:

- **Imitation**: start on a random frame of a reference stair climb and track it.
- **Task**: start standing in front of the stairs and reach a destination on the top landing, on stairs
  whose height follows a success curriculum.

The actor never sees the reference or the episode type. It observes proprioception, the destination,
and the elevation map; imitation teaches it the reference's climbing motion, and task episodes teach it
to reach destinations with that motion.

## Data and terrain

The four clips in [`data/supplied_stairs_pyramid_stand`](data/supplied_stairs_pyramid_stand) are the
motion-library `kimodo/supplied_stairs` climb rotated onto the four sides of one pyramid with a 2.5 m
square top landing. Each clip adds a two-second settle into the default standing pose and a two-second
hold: 374 frames at 50 Hz (7.48 s). Each reset picks one of the four approaches uniformly.

The terrain has 11 rows of 9 m tiles. Four columns (`motion_stairs`) hold the clips' shared pyramid on every
row, built from their terrain boxes; six columns (`task_stairs`) hold the same boxes with their heights scaled
by row, from 0.2 (row 0) to 1.2 (row 10) in steps of 0.1. Row 8 is the clips' own pyramid. The footprint
never changes and the clip frame origin is the environment origin on every tile, so reference and start
poses need no other offset. Each reset moves the environment to a tile of its episode type: imitation
episodes to a random `motion_stairs` tile, task episodes to a `task_stairs` tile on their own level.

```bash
uv pip install --python .venv/bin/python -e ../compositional_rsl_rl

uv run --no-sync isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Rough-G1-29dof-GoalConditioned-Perception --num_envs 4096
```

`-Play` evaluates task episodes only, on random stair heights. Replay
the reference through the reward terms with
`python -m isaaclab_tasks.contrib.velocity.config.g1_29dof_goalconditioned_perception.replay_motion`.

## Episode schedule and resets

The `motion` command ([`mdp/motion.py`](mdp/motion.py)) draws each episode in the reset events:

| Phase | Imitation probability |
| --- | --- |
| First 48,000 control steps (2,000 iterations) | 1.0 |
| Next 120,000 steps | Linear from 1.0 to 0.5 |
| Afterwards | 0.5 (`final_imitation_probability`) |

This is a per-episode sampling probability, not a loss weight. Imitation episodes start on a uniformly
random frame (on the first frame for 20% of them), with ±0.1 rad joint noise, and their destination is
the clip's final root pose.

Task episodes start standing in the default pose at rest, upright, 0.5 m behind the first reference
root in its yaw frame, with ±0.1 m XY and ±0.1 rad yaw noise. Their destination is the final root pose
moved 0.75 m into the landing, with ±0.35 m XY noise, and lowered or raised with the tile's scaled landing.

Task difficulty is the stair height only. Each environment's task level starts on row 0, rises one row
after a task episode that reached the destination without a failure termination, and falls one row after
any other task episode. It is not saved in checkpoints. A destination counts as reached within 0.35 m
horizontally, 0.25 m vertically, and 0.4 rad of yaw, with a foot in contact.
`Metrics/motion/task_success_rate` and `Curriculum/task_terrain_levels` log the success rate and mean level.

## Rewards and terminations

Imitation-only terms (zero in task episodes), each `exp(-error² / std²)`:

| Term | Weight | std | Error |
| --- | ---: | --- | --- |
| `motion_global_anchor_pos` | 2 | 0.3 m | Torso position in the world |
| `motion_global_anchor_ori` | 2 | 0.4 rad | Torso orientation, including yaw |
| `motion_global_anchor_lin_vel` | 1 | 1.0 m/s | Torso linear velocity |
| `motion_global_anchor_ang_vel` | 1 | 3.14 rad/s | Torso angular velocity |
| `motion_body_pos` | 1 | 0.3 m | Body positions, with the reference moved under the robot torso's XY and yaw |
| `motion_body_ori` | 1 | 0.4 rad | Body orientations, rotated by the same yaw |
| `motion_feet_lin_vel` | 1 | 1.0 m/s | Ankle linear velocities |

Task-only terms (zero in imitation episodes):

| Term | Weight | Behavior |
| --- | ---: | --- |
| `goal_position`, `goal_position_fine` | 4, 4 | `exp(-distance² / std²)` with std 2.0 m and 0.5 m |
| `goal_heading` | 2 | Yaw alignment once within the position tolerance |
| `goal_settle` | -0.5 | Root speed once within the position tolerance |
| `goal_joint_deviation` | -1 | Distance from the default pose while standing at the destination |
| `task_motion_style` | 2 | Motion-matching style of the perception motion-style task |

`task_motion_style` is [`MotionMatchingStyle`](../g1_29dof_perception_motion_style/mdp/motion_style.py) from
[`g1_29dof_perception_motion_style`](../g1_29dof_perception_motion_style), with the same parameters, paid only
in task episodes. The robot's joint state is matched to its nearest frame among the 174 captured climb frames
and their mirror; within a match distance of 0.25 it scores per-group joint error with saturating margins
(legs 0.15 rad, upper body 0.04 rad). It uses no reference phase or world pose, so it applies on every stair
height, and it pays nothing at rest. Both episode types share `alive` (+1) and
small joint, action-rate, contact, and foot-slide penalties. There is no reward for approach speed.

Imitation episodes end as a time-out on the clip's last frame, or as a failure when the torso height
differs from the reference by over 0.25 m, the torso tilt by over 0.8, or an ankle height by over 0.25 m.
Task episodes run for 10 s. Both fail on a low base, a tilt over 1.2 rad, pelvis or torso contact, or
an extreme joint target.

## Observation and network contract

| Group | Shape per environment | Use |
| --- | --- | --- |
| `policy` | 93: angular velocity, gravity, joint positions and velocities, last action | Actor head and attention query |
| `command` | 5: destination XYZ in the yaw frame, sine and cosine of the yaw error | Both heads and attention query |
| `elevation_map` | `(8, 8, 3)` | Shared AME map encoder |
| `critic` | 406: proprioception, episode type, reference joints, phase and torso pose (zero in task episodes), body poses, base velocity | Critic head only |

AME turns the XYZ grid into CNN tokens and attends to them with the `policy` and `command` vectors;
one 64-dimensional map feature feeds both heads. Both heads are `[512, 256, 128]` ELU MLPs without
observation normalization. PPO uses symmetry augmentation and a mirror loss (coefficient 0.1).

## Differences from the source task

- The actor sees this package's AME elevation-map features instead of a flat height scan.
- The robot is the box-foot G1 used by this package; the source uses a ball-hand variant of it.
- Physics, sensors, and the scene layout come from this package.
- Task difficulty is a success curriculum on the stair height, with fixed standing starts: the source's
  start-randomization curriculum, time-based ramp, and generated-stair fine-tuning tasks are not ported.

Teleporting the robot onto clip frames 37–42 makes the right thigh and shank colliders overlap at the
reference's deep knee flexion, which the solver pushes apart. A tracking policy avoids this by flexing
slightly less; only a kinematic replay hits it.
