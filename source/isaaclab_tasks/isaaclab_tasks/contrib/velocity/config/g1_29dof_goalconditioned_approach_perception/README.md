# G1 goal-conditioned stair climbing with an approach walk

`IsaacContrib-Velocity-Rough-G1-29dof-GoalConditioned-Approach-Perception` (and `-Play`) trains the
`g1_29dof_goalconditioned_perception` climb, but every episode first walks from a random spawn to the reference
clip's first root pose. The climb rewards are only reachable from the foot of the pyramid, which the policy learns
to reach first.

## Terrain and spawns

The scene is the goal-conditioned task's: the clips' own pyramid for imitation episodes and height-scaled copies,
one stair height per row, for task episodes. Each episode picks one of the four clips (one per pyramid side). The
robot spawns at rest with a random yaw, 0.5–2.0 m behind the clip start and up to 2.0 m to either side, on the
flat ground of that side. The waypoint is the clip's first root pose.

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

The actor observes one goal `(x, y, z, sin, cos)`: the waypoint before climbing, then the destination. The
critic also observes the one-hot phase.
