# G1 perception waypoint reaching

`IsaacContrib-Velocity-Rough-G1-29dof-Perception-Waypoint` reuses the
[`g1_29dof_perception`](../g1_29dof_perception) scene, robot, sensors, AME networks, and gait terms,
and replaces its velocity command with one waypoint per episode.

```bash
uv run --no-sync isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Rough-G1-29dof-Perception-Waypoint --num_envs 4096
```

`IsaacContrib-Velocity-Flat-G1-29dof-Perception-Waypoint` uses the same waypoints on a plane, where every
waypoint is at ground level and there is no terrain curriculum; use it to validate the pipeline. Both tasks
have `-Play` variants.

## Episodes

Terrain rows are the perception curriculum levels: flat, then up 1 rise, down 1 rise, up 2 rises, and so on
to the 6-rise pyramid and pit. Each tile has stairs within 2.0 m of its center and flat ground from 2.0 m to
its 4.0 m edge.

Half of the episodes start at the tile center (a pyramid top, a pit bottom, or flat ground) and target the
border ring; the other half start on the ring and target the center. The ring point is 2.4–3.4 m
(Chebyshev) from the center in a random direction, the center point has ±0.2 m noise, and the start yaw is
random. So a robot on a pyramid top climbs down, a robot in a pit climbs up, and a robot on the ring climbs
up a pyramid or down into a pit.

The waypoint height is the ground height under it plus 0.75 m (`base_height`), so the height target always
corresponds to standing on level ground there.

## Command and observations

The `command` observation (actor and critic) is the waypoint relative to the root in the root yaw frame,
`(x, y, z)` in meters. No velocity is commanded or observed.

The `policy` group ends with the root position on the tile, in the root yaw frame: XY from the tile center
and Z above the flat ground around the stairs (world Z), with ±0.02 m noise.

Gait terms inherited from the perception task gate on a velocity command. They read `waypoint_gait`, which
reports 0.8 m/s toward the waypoint and zero within the 0.3 m reach radius. It is neither observed nor tracked.

## Rewards and curriculum

| Term | Weight | Behavior |
| --- | ---: | --- |
| `waypoint_position`, `waypoint_position_fine` | 4, 4 | `exp(-‖e‖² / std²)` of the 3D error, std 2.0 m and 0.5 m |
| `waypoint_height` | 2 | `exp(-e_z² / 0.1²)` within the reach radius |
| `waypoint_progress` | 2 | Rate the horizontal distance shrinks, clipped to ±1 m/s |
| `waypoint_bearing` | 1 | Facing the waypoint while away from it |
| `waypoint_stand_still` | -1 | Planar speed and yaw rate within the reach radius |

Velocity and heading tracking, the world-height target, and the zero-weight posture term are removed; the
other perception rewards are kept. A waypoint counts as reached within 0.3 m horizontally and 0.2 m in
height. Environments start on levels 0–2, move up a level after reaching their waypoint, and move down
after ending over half their starting distance away. `Metrics/waypoint/success` and
`Curriculum/terrain_levels` track progress. `-Play` samples all levels without promotion.

The map encoder runs without BatchNorm, unlike the perception task's default.
