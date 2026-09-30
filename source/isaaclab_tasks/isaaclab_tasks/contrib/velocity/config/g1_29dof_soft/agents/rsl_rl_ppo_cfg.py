# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    # RslRlRNNModelCfg,
    # RslRlSymmetryCfg,
)

@configclass
class PointNetMLPModelCfg(RslRlMLPModelCfg):
    """RslRlMLPModelCfg + the extra PointNet-specific fields our model needs."""

    class_name: str = "isaaclab_tasks.contrib.velocity.config.g1_29dof_soft.agents.custom_model.PointNetMLPModel"
    num_point_obs: int = 300          # e.g. 100 points * 3 (xyz) -- must match your obs term
    point_dim: int = 3
    pointnet_hidden_dims: tuple = (64, 128)
    pointnet_feature_dim: int = 256

@configclass
class G1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 500
    obs_groups = {"actor": ["policy"], "critic": ["critic", "privileged"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        # entropy_coef=0.0025,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # symmetry_cfg=RslRlSymmetryCfg(
        #     use_data_augmentation=True,
        #     data_augmentation_func=g1.compute_symmetric_states
        #     ),
    )
    logger = "wandb"
    wandb_project = "g1_29dof_soft_rough"
    experiment_name = "g1_29dof_soft_rough"

@configclass
class G1RoughPerceptionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 500
    obs_groups = {"actor": ["policy"], "critic": ["critic", "privileged"]}
    actor = PointNetMLPModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        num_point_obs=43200,
        point_dim=3,
        pointnet_hidden_dims=(64, 128),
        pointnet_feature_dim=256,
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        # entropy_coef=0.0025,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # symmetry_cfg=RslRlSymmetryCfg(
        #     use_data_augmentation=True,
        #     data_augmentation_func=g1.compute_symmetric_states
        #     ),
    )
    logger = "wandb"
    wandb_project = "g1_29dof_soft_rough"
    experiment_name = "g1_29dof_soft_rough"

@configclass
class G1FlatPPORunnerCfg(G1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()  # type: ignore

        self.max_iterations = 30_000
        self.wandb_project = "g1_29dof_soft_vanilla_ppo"
        self.experiment_name = "g1_29dof_soft_vanilla_ppo"

@configclass
class G1FlatPerceptionPPORunnerCfg(G1RoughPerceptionPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()  # type: ignore

        self.max_iterations = 30_000
        self.wandb_project = "g1_29dof_soft_vanilla_ppo"
        self.experiment_name = "g1_29dof_soft_vanilla_ppo"
