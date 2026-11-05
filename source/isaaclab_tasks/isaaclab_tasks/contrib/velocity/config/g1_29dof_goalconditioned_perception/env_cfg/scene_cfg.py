# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone G1 scene with the motion clip's staircase, LiDAR, and an optional RGB-D camera."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_assets import UNITREE_G1_29DOF_BOX_FOOT_CFG

from ..mdp.terrains import MotionStairsTerrainCfg
from .motion_cfg import STAIR_MOTION_FILES, TASK_STAIR_HEIGHT_SCALES, TASK_STAIR_LEVELS
from .sensors_cfg import DEPTH_CAMERA_CFG, elevation_grid_pattern

# Every tile holds the clips' shared 4.5 m pyramid, centered with a margin for standing starts up to 1.2 m outside
# it, so reference root poses are valid at every environment origin.
_MOTION_STAIRS_CFG = MotionStairsTerrainCfg(size=(9.0, 9.0), proportion=0.4, motion_file=STAIR_MOTION_FILES[0])
"""The clips' own staircase on every row, for imitation episodes."""

_TASK_STAIRS_CFG = MotionStairsTerrainCfg(
    size=(9.0, 9.0),
    proportion=0.6,
    motion_file=STAIR_MOTION_FILES[0],
    height_scale_range=TASK_STAIR_HEIGHT_SCALES,
    num_levels=TASK_STAIR_LEVELS,
)
"""The clips' staircase with its heights scaled by row, for task episodes."""


@configclass
class G1PerceptionSceneCfg(InteractiveSceneCfg):
    """Robot, terrain, and sensor settings owned by the perception task."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(9.0, 9.0),
            border_width=5.0,
            num_rows=TASK_STAIR_LEVELS,
            num_cols=10,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            # rows are task stair levels; the motion command places each episode on its tile
            curriculum=True,
            sub_terrains={"motion_stairs": _MOTION_STAIRS_CFG, "task_stairs": _TASK_STAIRS_CFG},
        ),
        max_init_terrain_level=None,
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
    robot: ArticulationCfg = UNITREE_G1_29DOF_BOX_FOOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            func=elevation_grid_pattern, resolution=0.2, size=(1.5, 1.5), ordering="yx"
        ),
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
        debug_vis=True,
    )
    mid360_lidar: RayCasterCfg | None = None
    d435_camera: CameraCfg | None = None
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class G1DepthSceneCfg(G1PerceptionSceneCfg):
    """Enable only the actor's depth camera and critic's height scanner."""

    mid360_lidar = None
    d435_camera = DEPTH_CAMERA_CFG.copy()
