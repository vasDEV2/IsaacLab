# G1 contact targets (rigid)

`IsaacContrib-Velocity-G1-29dof-Rigid-ContactTargets` (and `-Play`) trains the 29-DoF G1 to follow the foot and
hand contacts of motion clips, on rigid terrain and without perception. It reuses the
[rigid task](../g1_29dof_rigid/README.md)'s robot, actions, physics, randomization, and rough-terrain rewards.

## Clips and terrain

Fifteen clips. Thirteen train, drawn per episode with stairs and walking in equal shares (`MOTION_WEIGHTS` in
`env_cfg.py`); two are held out as a test set (`TEST_MOTION_FILES`), never drawn in training and the only clips in
`-Play`:

| Clip | Content | Use |
| --- | --- | --- |
| `motion_yaw_000`, `_090`, `_180` | Pyramid stairs of the goal-conditioned task, approached from three sides, 7.5 s | Train |
| `motion_yaw_270` | The same pyramid climb from the fourth side | Test |
| `lafan_walk_00`–`_05`, `_07`, `_09`, `_11`, `_12` | Steady forward LAFAN1 walking on flat ground, 6 s | Train |
| `lafan_walk_08` | A straight LAFAN1 walk | Test |

The walking clips are 6 s windows of the public [Unitree G1 retargeting of LAFAN1](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset).
They were picked for steady forward walking at standing height, at 0.6–1 m/s, with under 35° of turning and within
a 7 m extent. They start mid-stride; the feet land 0.5 m apart (median; 0.33–0.70 m for 80% of footsteps). LAFAN1 is CC BY-NC-ND 4.0, so the clips
are not in this repository. Build them locally with [motion_library](https://github.com/jnskkmhr/motion_library)'s
LAFAN importer and `export_mimic.py` (box-foot G1, 50 Hz) into `~/motion_library/data/lafan/lafan_walk_NN/motion.npz`,
or set `LAFAN_MOTION_DIR` to the folder holding them. Clips `_06` and `_10` are left out: a heel-to-toe roll hides
one of their footsteps from the contact detection below.

`IsaacContrib-Velocity-G1-29dof-Rigid-ContactTargets-Walk` (and `-Walk-Play`) is the same task on the walking clips
only: the ten training walking clips drawn uniformly, and `lafan_walk_08` alone in `-Walk-Play`. Its runs log
to `g1_29dof_rigid_contact_targets_walk_ppo`.

The terrain has one column per clip, holding that clip's boxes (flat ground, with the root path centered, for the
walking clips); at reset each robot moves onto its clip's column
and starts on the clip's first pose, with ±0.05 rad joint noise. The clip then plays at the robot's pace
(`wait_for_contact`): it does not land the current footstep (see [Foot targets](#foot-targets)) until the robot's
stepping foot is down (contact force above 5 N) within 8 cm (`reach_radius`) of its target; when the robot lifts
the stepping foot (for at least 0.1 s, so the drop after a reset and bounces do not count) and puts it down within
8 cm of its target early, the clip jumps to that touchdown. In training the clip waits at most 1 s per footstep
(`max_wait_s`) and then plays on; in `-Play` and `-Walk-Play` it waits indefinitely. The `clip_wait` metric is 1
while the clip waits. The episode ends at the clip's last frame, at a fall, when the root is 1 m from the clip's
root, or after 20 s (30 s in Play).

## Contacts

Contacts are measured from each clip's geometry when it is loaded (`mdp/motion.py`). A foot is in contact when at
least two corners of its foot box are within 1 cm of the terrain and its ankle moves slower than 0.3 m/s (dragged
swing feet do not count); a hand when its sphere is within 1 cm and slower than 0.3 m/s. Contacts and gaps shorter
than three frames are dropped. On the clips that carry contact labels this agrees with them on 84% (stairs) and
89–94% (platform) of the frames. None of the clips uses the hands; the walking clips have no hand spheres, so their
hand bodies' origins stand in for them.

## Foot targets

The clip's footsteps are its foot contacts that land at least 5 cm (`step_min_distance`) from that foot's previous
footstep, or from its position in the first frame; nearer contacts are the foot settling in place. The feet then
move one footstep at a time. From the first frame, the foot that steps first targets where its footstep lands and
the other foot targets its starting position. When the footstep lands, the next footstep in the clip begins: its
foot targets its landing, and the other foot holds the footstep it just made. Footsteps alternate left and right in
all nine clips. A target is the mean ankle roll link position over that contact.

## Observations

The actor reads the rigid task's proprioception (no height scan) and, in place of the velocity command, the
`command` group: for the left foot, right foot, left hand, and right hand, the target position relative to the root
in its yaw frame, and whether the body should be planted (1 for the holding foot, 0 for the stepping foot; hands:
1 in contact), 16 values. Hands target their own contacts and never touch anything in these clips. The critic reads
the same group with the rigid task's critic and privileged observations. The left/right mirror swaps the bodies and
negates the lateral coordinate.

## Rewards

The rigid task's rough-terrain rewards without `track_lin_vel_xy`, `track_ang_vel_z`, and `track_heading`. The terms
that tell standing from walking (`joint_deviation`, `feet_air_time`, `break_contact`, `no_fly`, `contact_impulse`,
`standing_foot_motion`, `foot_clearance`) read the clip's root velocity as their command. Two penalties on the base:

- `base_overspeed` (weight -2.0): the squared horizontal base speed above the clip's root speed plus 0.2 m/s, so
  rushing or lunging is penalized but walking along with the clip is not.
- `base_ang_vel` (weight -0.5): the squared base angular velocity about all three axes, i.e. how fast the base's
  orientation changes (on top of the rigid task's `ang_vel_xy_l2`, weight -0.05).

Per foot, a penalty on the ankle's orientation:

- `*_foot_orientation` (weight -1.0): the squared angle [rad^2] between the robot's ankle roll link and the clip's
  at the same frame, in stance and swing, so the ankle follows the clip's roll, pitch, yaw, toe-off and heel strike.

Per foot, on its ankle roll link:

- `left/right_foot_contact_target` (weight 3.0): while that foot holds its target and the clip is not waiting, the
  mean of `exp(-d² / 0.05²)` and `exp(-d² / 0.15²)` of its distance `d` to the target (1 on the target, 0.63 at
  5 cm, 0.33 at 10 cm, 0.03 at 25 cm).
- `left/right_foot_step_target` (weight 1.5): while that foot steps, `exp(-d_xy² / 0.25²)` of its horizontal distance
  to its landing (0.85 at 10 cm, 0.37 at 25 cm, 0.02 at 50 cm).
- `left/right_foot_progress` (weight 1.0): while that foot steps, its horizontal speed toward its landing [m/s],
  clipped to ±1 (negative when it moves away), and zero within 2 cm of the landing.

The curriculum is empty.

## Training

```bash
uv run --with wandb isaaclab train --rl_library rsl_rl \
    --task IsaacContrib-Velocity-G1-29dof-Rigid-ContactTargets --num_envs 4096 --headless
```

Runs log to `g1_29dof_rigid_contact_targets_ppo`. In `-Play`, the viewer looks at env 0's clip terrain, and the
**Live Plots** section of its sidebar plots env 0's weighted reward terms (and the episode reward) every step. For the walking-only task, pass
`--task IsaacContrib-Velocity-G1-29dof-Rigid-ContactTargets-Walk`.

## Play

Two Play variants each, with the same Play settings (the clip waits indefinitely, no pushes or observation noise):

| Task | Clips |
| --- | --- |
| `...-ContactTargets-Play` | Held out, out of distribution: `motion_yaw_270` and `lafan_walk_08` |
| `...-ContactTargets-Play-Train` | Training clips, in distribution: `motion_yaw_000` and `lafan_walk_00` |
| `...-ContactTargets-Walk-Play` | Held out: `lafan_walk_08` |
| `...-ContactTargets-Walk-Play-Train` | A training clip: `lafan_walk_00` |

```bash
uv run isaaclab play --rl_library rsl_rl \
    --task IsaacContrib-Velocity-G1-29dof-Rigid-ContactTargets-Walk-Play-Train --checkpoint latest
```
