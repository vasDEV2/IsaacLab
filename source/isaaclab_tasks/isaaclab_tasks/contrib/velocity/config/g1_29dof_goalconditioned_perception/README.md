# G1 goal-conditioned stair climbing with AME

A perceptive G1 policy that climbs a staircase to a destination root position and heading. The training
paradigm follows the goal-conditioned Mimic task on the
[`dokyun/multi-mimic` branch](https://github.com/jnskkmhr/IsaacLab/tree/dokyun/multi-mimic/source/isaaclab_tasks/isaaclab_tasks/contrib/mimic/config/g1_29dof_perception_goal_condition)
of jnskkmhr/IsaacLab, itself inspired by GfR. The actor keeps this package's AME height-map encoder.

Training mixes two kinds of episodes:

- **Imitation**: start on a random frame of a reference stair climb and track it.
- **Task**: start near a standing pose in front of the stairs and reach a destination on the top landing.

The actor never sees the reference or the episode type. It observes proprioception, the destination,
and the elevation map; imitation teaches it the reference's climbing motion, and task episodes teach it
to reach destinations with that motion.

## Data and terrain

The four clips in [`data/supplied_stairs_pyramid_stand`](data/supplied_stairs_pyramid_stand) are the
motion-library `kimodo/supplied_stairs` climb rotated onto the four sides of one pyramid with a 2.5 m
square top landing. Each clip adds a two-second settle into the default standing pose and a two-second
hold: 374 frames at 50 Hz (7.48 s). Each reset picks one of the four approaches uniformly.

Every 9 m terrain tile holds the clips' shared pyramid, built from their terrain boxes and centered in
the tile; the clip frame origin is the environment origin, so reference poses need no other offset.

```bash
uv pip install --python .venv/bin/python -e ../compositional_rsl_rl

uv run --no-sync isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Rough-G1-29dof-GoalConditioned-Perception --num_envs 4096
```

`-Play` evaluates task episodes only, from the full range of standing starts and destinations. Replay
the reference through the reward terms with
`python -m isaaclab_tasks.contrib.velocity.config.g1_29dof_goalconditioned_perception.replay_motion`.

## Episode schedule and resets

The `motion` command ([`mdp/motion.py`](mdp/motion.py)) draws each episode in the reset events:

| Phase | Imitation probability |
| --- | --- |
| First 120,000 control steps (5,000 iterations) | 1.0 |
| Next 120,000 steps | Linear from 1.0 to 0.5 |
| Afterwards | 0.5 (`final_imitation_probability`) |

This is a per-episode sampling probability, not a loss weight. Imitation episodes start on a uniformly
random frame (on the first frame for 20% of them), with ±0.1 rad joint noise, and their destination is
the clip's final root pose.

Task episodes start on the first frame and their destination moves forward from the final root pose
into the landing. A per-environment difficulty `d` blends both toward harder settings:

| Quantity | `d = 0` | `d = 1` |
| --- | --- | --- |
| Start root | First reference root | 0.9 m behind it, ±0.3 m XY, ±0.3 rad yaw, upright |
| Start joints and velocities | Reference | Default standing pose at rest |
| Destination offset | 0.2 m | 0.75 m, ±0.35 m XY |
| Root velocity noise | None | Above `d = 0.8`, half the starts get up to ±0.15 m/s and ±0.2 rad/s |

`d` rises by 0.02 after a task episode that reached the destination without a failure termination and
falls by 0.05 after any other task episode. It is not saved in checkpoints. A destination counts as
reached within 0.35 m horizontally, 0.25 m vertically, and 0.4 rad of yaw, with a foot in contact.
`Metrics/motion/task_success_rate` and `Curriculum/task_resets` log the success rate and mean `d`.

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
| `task_motion_style` | 2 | Local limb motion against the reference at an adaptive playback rate |

`task_motion_style` keeps a hidden reference phase that advances at 0.5, 0.75, 1.0, or 1.25 times the
recorded rate, whichever best matches the robot's joint positions, velocities, and ankle and wrist
positions relative to the root. It ignores the world pose. Both episode types share `alive` (+1) and
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
- The success-curriculum reset is the only reset mode: the time-based ramp and the generated-stair
  fine-tuning tasks of the source are not ported.

Teleporting the robot onto clip frames 37–42 makes the right thigh and shank colliders overlap at the
reference's deep knee flexion, which the solver pushes apart. A tracking policy avoids this by flexing
slightly less; only a kinematic replay hits it.
