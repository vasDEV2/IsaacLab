* Added ``IsaacContrib-Velocity-Rough-G1-29dof-Perception-Waypoint-Mimic``: the AME G1 waypoint task on the
  pyramid of the stair motion clips, where the waypoint is a clip's first root pose at the pyramid base. The
  waypoint is rewarded for the first 3000 iterations; afterwards it is commanded without reward, and reaching
  it starts imitation of the clip with the goal-conditioned task's imitation rewards and terminations, while
  the commanded and rewarded waypoint moves to the clip's end on the top landing.
