# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physics backend presets for the G1 29-DoF soft-terrain locomotion task."""

from isaaclab_newton.physics import (
    KaminoPADMMSolverCfg,
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
)
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.physics import PhysxAutoCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg


@configclass
class G1PhysicsCfg(PresetCfg):
    """Backend presets for the G1 29-DoF soft-terrain locomotion task.

    The soft contact model composes an external wrench on the feet every physics step, so the PhysX
    backends re-apply external forces on every solver iteration. The remaining settings match the
    rigid-terrain task.
    """

    isaacsim_physx = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15, enable_external_forces_every_iteration=True)
    ovphysx = OvPhysxCfg(gpu_max_rigid_patch_count=10 * 2**15, enable_external_forces_every_iteration=True)
    physx = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=1000,
            nconmax=300,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
        num_substeps=2,
        debug_mode=False,
        default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0),
    )
    newton_kamino = NewtonCfg(solver_cfg=KaminoPADMMSolverCfg(max_contacts_per_world=64))
    default = newton_mjwarp
