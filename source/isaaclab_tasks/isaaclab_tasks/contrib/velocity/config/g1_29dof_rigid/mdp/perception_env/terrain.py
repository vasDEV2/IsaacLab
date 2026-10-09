# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR  # type: ignore

"""
terrain collections
"""

CurriculumSoftTerrain = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=terrain_gen.TerrainGeneratorCfg(
        size=(8.0, 8.0),  # size of sub-terrain
        border_width=0.0,
        num_rows=10,
        num_cols=10,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=True,
        sub_terrains={
            # "hard_ground": terrain_gen.MeshPlaneTerrainCfg(
            #     proportion=0.5,
            #     ground_height_range=(0.0, 0.0),
            # ),
            "soft_ground": terrain_gen.MeshPlaneTerrainCfg(
                proportion=0.5,
                ground_height_range=(0.0, -0.12),
            ),
        },
    ),
    collision_group=-1,
    # this wont be used in soft terrain
    physics_material=sim_utils.RigidBodyMaterialCfg(
        # friction_combine_mode="average",
        # restitution_combine_mode="average",
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
    ),
    # visual_material=sim_utils.MdlFileCfg(
    #     mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
    #     project_uvw=True,
    #     texture_scale=(0.25, 0.25),
    #     albedo_brightness=0.2,
    # ),
    max_init_terrain_level=0,
    # debug_vis=True,
)

"""
soft terrain
"""

SoftTerrain = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=terrain_gen.TerrainGeneratorCfg(
        size=(10.0, 10.0),  # size of sub-terrain
        border_width=0.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=False,
        sub_terrains={
            "plane": terrain_gen.MeshPlaneTerrainCfg(
                proportion=1.0,
                ground_height_range=(0.0, 0.0),
            ),
        },
    ),
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        # friction_combine_mode="average",
        # restitution_combine_mode="average",
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
    ),
    visual_material=sim_utils.MdlFileCfg(
        # mdl_path=f"{ISAACLAB_ASSETS_DATA_DIR}/texture/Ground_039/Ground039_4K.mdl", # black sand
        mdl_path=f"{ISAACLAB_ASSETS_DATA_DIR}/texture/Ground_080/Ground080_4K.mdl",  # beach
        project_uvw=True,
        texture_scale=(0.25, 0.25),
        albedo_brightness=0.2,
    ),
    disable_collider=True,
)

RigidPatch = TerrainImporterCfg(
    prim_path="/World/rigid_patch",
    terrain_type="generator",
    terrain_generator=terrain_gen.TerrainGeneratorCfg(
        size=(5, 5),  # size of sub-terrain
        border_width=0.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=False,
        sub_terrains={
            "plane": terrain_gen.MeshPlaneTerrainCfg(
                proportion=1.0,
                ground_height_range=(0.0005, 0.0005),
            ),
        },
    ),
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        # friction_combine_mode="average",
        # restitution_combine_mode="average",
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
    ),
    visual_material=sim_utils.MdlFileCfg(
        mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        project_uvw=True,
        texture_scale=(0.5, 0.5),
        albedo_brightness=0.2,
    ),
)


SoftTerrainVisual = TerrainImporterCfg(
    prim_path="/World/ground_visual",
    terrain_type="generator",
    terrain_generator=terrain_gen.TerrainGeneratorCfg(
        # size=(15.0, 15.0),  # size of sub-terrain
        size=(100.0, 100.0),  # size of sub-terrain
        border_width=0.0,
        num_rows=5,
        num_cols=5,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=False,
        sub_terrains={
            "plane": terrain_gen.MeshPlaneTerrainCfg(
                proportion=1.0,
                ground_height_range=(-0.02, -0.02),
            ),
        },
    ),
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        # friction_combine_mode="average",
        # restitution_combine_mode="average",
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
    ),
    visual_material=sim_utils.MdlFileCfg(
        # mdl_path=f"{ISAACLAB_ASSETS_DATA_DIR}/texture/Ground_039/Ground039_4K.mdl", # black sand
        mdl_path=f"{ISAACLAB_ASSETS_DATA_DIR}/texture/Ground_080/Ground080_4K.mdl",  # beach
        project_uvw=True,
        texture_scale=(0.25, 0.25),
        albedo_brightness=0.2,
    ),
    disable_collider=True,
)

"""
rough terrain
"""
