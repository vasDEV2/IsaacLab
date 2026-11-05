# G1 goal-conditioned stair climbing with an approach walk

`IsaacContrib-Velocity-Rough-G1-29dof-GoalConditioned-Approach-Perception` (and `-Play`) trains the
`g1_29dof_goalconditioned_perception` imitation climb, but every episode first walks from a random spawn to the
reference clip's first root pose. The climb rewards are only reachable from the foot of the pyramid, which the
policy learns to reach first.

## Terrain and spawns

Every tile holds the clips' own pyramid at its recorded height; there are no height-scaled copies and no stair
curriculum. Each episode picks one of the four clips (one per pyramid side). The
robot spawns at rest with a random yaw, 0.5–2.0 m behind the clip start and up to 2.0 m to either side, on the
flat ground of that side. The waypoint is the clip's first root position and heading, at 0.75 m above the ground.

## Schedule

| Iterations | Phase after spawn | Rewards and terminations |
| --- | --- | --- |
| 0–1,500 | `walk`: reach the waypoint for the whole 20 s episode | `g1_29dof_perception_waypoint` (`walk_*`) |
| 1,500 on | `approach`: reach the waypoint, at most 10 s | none, apart from fall terminations |
| | `climb`: from the waypoint, imitation of the clip from frame 0 | `g1_29dof_goalconditioned_perception` imitation terms (`climb_*`) |

Climbing always imitates the clip and ends on its last frame. The goal-conditioned task's task-episode rewards
(`goal_*` and `task_motion_style`) are left out.

## Observations

The actor has the `g1_29dof_perception_waypoint` inputs:

- `policy` (96): the waypoint task's proprioception, followed by the root position in the root yaw frame. XY is
  measured from the tile (pyramid) center and Z above the flat ground. The environment origin here is the clip
  frame's origin, 2.3 m from the pyramid center, so the tile center comes from the clips' terrain boxes.
- `command` (3): the active target relative to the root in the root yaw frame. Before climbing, the target is the
  waypoint at the clip start, 2.33 m from the pyramid center on the clip's side and 0.75 m above the ground (the
  waypoint task's `base_height`). Once climbing, it is the clip's final root position.
- The elevation map.

The critic keeps the goal-conditioned inputs and adds the one-hot phase.

## Printing rewards during play

`play_rewards.py` plays a checkpoint and prints the reward terms of one environment: the weighted per-step value,
the episode sum, and the mean over all environments, with its phase and distance to the active target.

```bash
uv run --no-sync python -m \
    isaaclab_tasks.contrib.velocity.config.g1_29dof_goalconditioned_approach_perception.play_rewards \
    --task IsaacContrib-Velocity-Rough-G1-29dof-GoalConditioned-Approach-Perception-Play \
    --checkpoint /path/to/model.pt --print_env 0 --print_every 25
```

Play episodes approach and climb; add `--walk` to play the rewarded walk of the first stage instead. Only the
current phase's terms are listed; `--all_terms` lists every term.
