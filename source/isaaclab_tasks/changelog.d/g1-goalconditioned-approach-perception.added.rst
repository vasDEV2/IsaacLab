* Added ``IsaacContrib-Velocity-Rough-G1-29dof-GoalConditioned-Approach-Perception``: the AME G1 goal-conditioned
  stair climb, where every episode first walks from a random spawn to the reference clip's start. The walk is
  rewarded like the waypoint task for the first 1500 iterations and unrewarded afterwards, when reaching the
  start begins the goal-conditioned imitation episode, and from 4500 iterations its task episodes and stair
  height curriculum. The actor has the waypoint task's inputs.
