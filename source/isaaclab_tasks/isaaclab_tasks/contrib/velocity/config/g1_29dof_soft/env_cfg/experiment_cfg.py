# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Single editing location for G1 soft experiment choices.

Sections use Isaac Lab's native configuration types. Edit defaults here, then use
this file's training entry point or the standard Isaac Lab train/play commands.
Sensor distances/offsets use [m], periods use [s], angles in scan patterns use
[deg], and camera optical parameters retain Isaac Lab's native [mm] convention.
"""

from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_newton.sensors import NewtonRaycastSensorCfg

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg, patterns
from isaaclab.utils.configclass import configclass


@configclass
class G1SensorsCfg:
    """Perception sensor configurations; set a sensor to None to disable it."""

    mid360_lidar = NewtonRaycastSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*torso_link",  # your mounting link
        update_period=0.1,  # ~10 Hz, matching the real Mid-360 frame rate
        max_distance=40.0,  # Mid-360 max range spec
        ray_alignment="base",  # rays follow full 6-DoF body pose (rigidly mounted)
        global_world_only=False,  # also hit shapes in this env, not just shared terrain
        mesh_prim_paths=[],  # unused on Newton (ignored, but field must be set)
        offset=NewtonRaycastSensorCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),  # identity, (x, y, z, w) order in 3.0
        ),
        pattern_cfg=patterns.LidarPatternCfg(
            channels=40,  # vertical "lines" approximating the non-repetitive scan
            vertical_fov_range=(-7.0, 52.0),  # Mid-360 spec: 59° vertical FOV
            horizontal_fov_range=(0.0, 360.0),  # full 360° horizontal
            horizontal_res=1.0,  # tune for point density vs. compute cost
        ),
        debug_vis=True,
    )
    # --- D435-like RGB-D camera, mounted on the robot ---
    d435_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/d435_link/Camera",
        update_period=1 / 30,  # D435 default depth/RGB stream rate
        width=640,
        height=480,  # matches common D435 depth stream (848x480 also common)
        data_types=["rgb", "depth"],  # only rgb/depth are supported by the Newton Warp renderer
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=1.93,  # D435 RGB module focal length (mm)
            horizontal_aperture=3.68,  # sized to give ~69° horizontal FOV (RGB spec)
            clipping_range=(0.1, 10.0),  # D435 depth range spec (usable accuracy ~0.3-3 m)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),  # identity, (x, y, z, w) order in 3.0
            convention="ros",  # +Z forward, -Y up — matches how D435 depth data is usually consumed
        ),
        renderer_cfg=NewtonWarpRendererCfg(),
    )


@configclass
class G1PerceptionCfg:
    """Policy sensor selection and preprocessing; perception is opt-in."""

    lidar: bool = False
    rgb: bool = False
    depth: bool = False
    lidar_sensor: str = "mid360_lidar"
    camera_sensor: str = "d435_camera"
    normalize_rgb: bool = True
    lidar_clip_max: float | None = None
    """Optional observation-only range limit [m]; None uses the live sensor range."""
    depth_clip_max: float | None = None
    """Optional observation-only depth limit [m]; None uses the live camera far plane."""


SENSORS = G1SensorsCfg()
PERCEPTION = G1PerceptionCfg()
