# Kimodo pyramid stairs with a standing finish

Copied unchanged from `contrib/mimic/data/motions/data/kimodo/supplied_stairs_pyramid_stand` on the
[`dokyun/multi-mimic` branch of jnskkmhr/IsaacLab](https://github.com/jnskkmhr/IsaacLab/tree/dokyun/multi-mimic).

The source is the motion-library `kimodo/supplied_stairs` climb (174 frames at 50 Hz). Its one-sided
staircase is extended to a four-sided pyramid with a 2.5 m square top landing, and each file rotates the
reference by 0, 90, 180, or 270 degrees about the pyramid center. Each clip appends a two-second
procedural transition into the default standing pose and a two-second static hold: 374 frames (7.48 s).

The NPZ fields follow the motion-library schema: named joints and bodies, xyzw body quaternions, and the
pyramid as `terrain_position`, `terrain_dimension`, and `terrain_quat` boxes in the reference frame, with
`ground_height` at zero. `provenance.json` records the source hash and the final standing pose.
The finish is synthesized, not captured motion, and dynamic tracking of it has not been validated.
