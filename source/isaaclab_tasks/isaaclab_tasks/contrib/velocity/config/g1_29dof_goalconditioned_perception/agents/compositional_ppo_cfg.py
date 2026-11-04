# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AME height-map PPO for goal-conditioned stair climbing, using the compositional RSL-RL library."""

from compositional_rsl_rl.algorithm import CompositionalPPOAlgorithmCfg
from compositional_rsl_rl.loss import LossesCfg
from compositional_rsl_rl.mdp.network import AttentionMapNetworkCfg
from compositional_rsl_rl.network import ActorHeadCfg, CriticHeadCfg, NetworkCfg
from compositional_rsl_rl.runner import CompositionalRunnerCfg

from isaaclab.utils.configclass import configclass

from isaaclab_contrib.mdp import compute_mirrored_states

from isaaclab_rl.rsl_rl import RslRlSymmetryCfg


@configclass
class G1PerceptionNetworksCfg(NetworkCfg):
    """One actor-visible map encoder shared by the policy and value heads."""

    map_encoder: AttentionMapNetworkCfg | None = None


@configclass
class G1RoughGoalConditionedPerceptionPPORunnerCfg(CompositionalRunnerCfg):
    """Train the map encoder and both heads jointly through PPO; only the critic sees the motion reference."""

    num_steps_per_env = 24
    max_iterations = 30_000
    save_interval = 500
    networks = G1PerceptionNetworksCfg(
        actor=ActorHeadCfg(
            input=["policy", "command", "map_encoder"],
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=False,
            distribution_cfg=ActorHeadCfg.GaussianDistributionCfg(init_std=1.0),
        ),
        critic=CriticHeadCfg(
            input=["critic", "command", "map_encoder"],
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=False,
        ),
        map_encoder=AttentionMapNetworkCfg(
            input=["elevation_map", "policy", "command"],
            map_input="elevation_map",
            map_shape=None,  # Inferred from the structured (33, 21, 3) observation.
            output_dim=64,
            feature_dim=64,
            num_heads=16,
            cnn_channels=16,
            cnn_downsample=True,
            # BatchNorm statistics differ between rollouts and minibatch updates, which breaks the PPO ratio.
            cnn_normalization="batch",
            attach_global=False,
            hidden_dims=[],
            obs_normalization=True,
            frozen=False,
            pretrained_cfg=None,
        ),
    )
    loss = LossesCfg()
    algorithm = CompositionalPPOAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            data_augmentation_func=compute_mirrored_states,
            mirror_loss_coeff=0.1,
        ),
    )
    # logger = "tensorboard"
    logger = "wandb"
    experiment_name = "g1_29dof_goalconditioned_perception_ame_stairs"
    wandb_project = "g1_29dof_goalconditioned_perception_ame"
