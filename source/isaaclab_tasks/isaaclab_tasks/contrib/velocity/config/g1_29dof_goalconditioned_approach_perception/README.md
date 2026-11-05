# G1 goal-conditioned stair climbing with an approach walk

`IsaacContrib-Velocity-Rough-G1-29dof-GoalConditioned-Approach-Perception` (and `-Play`) trains the
`g1_29dof_goalconditioned_perception` climb, but every episode first walks from a random spawn to the reference
clip's first root pose. The climb rewards are only reachable from the foot of the pyramid, which the policy learns
to reach first.

## Terrain and spawns

The scene is the goal-conditioned task's: the clips' own pyramid for imitation episodes and height-scaled copies,
one stair height per row, for task episodes. Each episode picks one of the four clips (one per pyramid side). The
robot spawns at rest with a random yaw, 0.5–2.0 m behind the clip start and up to 2.0 m to either side, on the
flat ground of that side. The waypoint is the clip's first root position and heading, at 0.75 m above the ground.

## Schedule

| Iterations | Phase after spawn | Rewards and terminations |
| --- | --- | --- |
| 0–1,500 | `walk`: reach the waypoint for the whole 20 s episode | `g1_29dof_perception_waypoint` (`walk_*`) |
| 1,500 on | `approach`: reach the waypoint, at most 10 s | none, apart from fall terminations |
| | `climb`: from the waypoint, the goal-conditioned episode starts on clip frame 0 | `g1_29dof_goalconditioned_perception` (`climb_*`) |

During climbing, imitation episodes end on the clip's last frame and task episodes 10 s after reaching the
waypoint. Climbing is imitation only until 4,500 iterations. After that, the imitation probability falls to
0.5 over 5,000 iterations, and task episodes follow the goal-conditioned stair height curriculum. Only
task episodes that reached the waypoint move their stair height.

## Observations

The actor has the `g1_29dof_perception_waypoint` inputs:

- `policy` (96): the waypoint task's proprioception, followed by the root position in the root yaw frame. XY is
  measured from the tile (pyramid) center and Z above the flat ground. The environment origin here is the clip
  frame's origin, 2.3 m from the pyramid center, so the tile center comes from the clips' terrain boxes.
- `command` (3): the active target relative to the root in the root yaw frame. Before climbing, the target is the
  waypoint at the clip start, 2.33 m from the pyramid center on the clip's side and 0.75 m above the ground (the
  waypoint task's `base_height`). Once climbing, it is the destination.
- The elevation map.

The critic keeps the goal-conditioned inputs and adds the sine and cosine of the target heading error, which the
climb's `goal_heading` reward scores, plus the one-hot phase.
