# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns, CameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab_newton.sensors import NewtonRaycastSensorCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg

##
# Pre-defined configs
##
from isaaclab_assets import UNITREE_G1_29DOF_CFG

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


@configclass
class G1SceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # Optional rigid floor at the center
    rigid_floor: TerrainImporterCfg = None  # type: ignore
    visual_terrain: TerrainImporterCfg = None  # type: ignore

    # robots
    robot: ArticulationCfg = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore

    # sensors
    height_scanner = RayCasterCfg(
        # the URDF importer nests bodies along the kinematic tree, so match the leaf name anywhere
        prim_path="{ENV_REGEX_NS}/Robot/.*torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    mid360_lidar = NewtonRaycastSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*torso_link",   # your mounting link
    update_period=0.1,          # ~10 Hz, matching the real Mid-360 frame rate
    max_distance=40.0,          # Mid-360 max range spec
    ray_alignment="base",       # rays follow full 6-DoF body pose (rigidly mounted)
    global_world_only=False,    # also hit shapes in this env, not just shared terrain
    mesh_prim_paths=[],         # unused on Newton (ignored, but field must be set)
    offset=NewtonRaycastSensorCfg.OffsetCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),   # identity, (x, y, z, w) order in 3.0
    ),
    pattern_cfg=patterns.LidarPatternCfg(
        channels=40,                       # vertical "lines" approximating the non-repetitive scan
        vertical_fov_range=(-7.0, 52.0),   # Mid-360 spec: 59° vertical FOV
        horizontal_fov_range=(0.0, 360.0), # full 360° horizontal
        horizontal_res=1.0,                # tune for point density vs. compute cost
    ),
    debug_vis=True,
)
    # --- D435-like RGB-D camera, mounted on the robot ---
    d435_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/d435_link/Camera",
        update_period=1 / 30,          # D435 default depth/RGB stream rate
        width=640,
        height=480,                    # matches common D435 depth stream (848x480 also common)
        data_types=["rgb", "depth"],   # only rgb/depth are supported by the Newton Warp renderer
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=1.93,             # D435 RGB module focal length (mm)
            horizontal_aperture=3.68,      # sized to give ~69° horizontal FOV (RGB spec)
            clipping_range=(0.1, 10.0),    # D435 depth range spec (usable accuracy ~0.3-3 m)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),      # identity, (x, y, z, w) order in 3.0
            convention="ros",              # +Z forward, -Y up — matches how D435 depth data is usually consumed
        ),
        renderer_cfg=NewtonWarpRendererCfg(),
    )
        # multi-body contact reporting
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
