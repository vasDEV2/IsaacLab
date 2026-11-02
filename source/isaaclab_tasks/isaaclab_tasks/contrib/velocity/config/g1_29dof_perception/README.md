# G1 perception locomotion with AME

The default flat and rough tasks train an attention-based map encoder (AME) through
the standalone [compositional RSL-RL library](https://github.com/jnskkmhr/compositional_rsl_rl).
The standard IsaacLab RSL-RL training, playback, checkpoint, and export entry points remain in use.

## Install and run

From the IsaacLab checkout, install the sibling library into the existing environment:

```bash
uv pip install --python .venv/bin/python -e ../compositional_rsl_rl

uv run --no-sync isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Rough-G1-29dof-Perception --num_envs 128
```

The library depends on `rsl-rl-lib==5.5.1`; it extends that package rather than replacing
its Python namespace. The task registry selects `agents/compositional_ppo_cfg.py`, whose
algorithm class is `compositional_rsl_rl:CompositionalPPO`. No script modifications are needed.
Use `--no-sync` to preserve the locally installed extension if you otherwise synchronize
the IsaacLab environment from its lockfile.

A short pipeline check (not enough to learn locomotion):

```bash
uv run --no-sync isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Rough-G1-29dof-Perception \
  --num_envs 2 --max_iterations 1 --logger tensorboard \
  agent.num_steps_per_env=4 agent.algorithm.num_mini_batches=1 \
  agent.algorithm.num_learning_epochs=1 \
  'env.sim.visualizer_cfgs=[]' 'env.video_recorders=[]'
```

Play a trained checkpoint:

```bash
uv run --no-sync isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Rough-G1-29dof-Perception-Play \
  --num_envs 4 --checkpoint /path/to/model.pt --viz newton_gl
```

Replace `Rough` with `Flat` for the plane variants. Existing `-Perception-Depth`
and `-Perception-Depth-Play` task IDs are compatibility aliases for the same AME
height-map environments. They no longer create a depth camera. Neither variant
creates a LiDAR sensor. Old PointNet/depth-CNN checkpoints cannot resume this model;
start a new run. The previous sensor/encoder helper implementations remain available
in the task directory for separate experiments, but are not selected by these tasks.

## Observation and network contract

| Group | Shape per environment | Use |
| --- | --- | --- |
| `policy` | Flat proprioceptive history, 10 steps | Actor head and attention query |
| `command` | Flat velocity/heading command | Both heads and attention query |
| `elevation_map` | `(33, 21, 3)` | Shared map encoder |
| `critic`, `privileged` | Task-local state histories | Critic head only |

The map samples ground over a 3.2 m forward-axis span and a 2.0 m lateral-axis
span, centered on the torso-link origin, at 0.1 m spacing. Grid dimension 0
increases along torso yaw-frame X (forward), and dimension 1 along yaw-frame Y
(left). Each cell contains `(x, y, z)` in meters relative to the torso-link origin;
Z follows world up. Torso roll and pitch do not rotate the map axes.

The scanner origin has zero offset from the torso. Its ray **starts** are 20 m
above the torso in the yaw frame, so pitching the torso does not displace the scan
footprint. Rays point vertically down and intersect the shared terrain only.
The scan updates each control step (50 Hz by default). Non-finite hits retain
their grid XY and receive Z = −2 m relative to the torso. This is a missing-data
convention, not a measured surface or a validity mask. The map has no added noise,
occlusion, or sensor delay: this first training setup assumes an ideal terrain map.

AME turns the XYZ grid into CNN tokens and attends to those tokens using the
actor-visible `policy` and `command` vectors. One 64-dimensional map feature feeds
both actor and critic; the critic additionally receives privileged observations.
PPO jointly trains the encoder and both heads. There is no auxiliary loss or
supervision target. The full declarative configuration is in
[`agents/compositional_ppo_cfg.py`](agents/compositional_ppo_cfg.py).

Symmetry augmentation is enabled through `isaaclab_contrib.mdp` and the existing
compositional PPO interface. Each sample is paired with an XZ-plane reflection:
left/right joint and foot channels swap; roll/yaw joint signs change; linear and
angular vectors use their respective reflection rules; and XYZW heading
quaternions are reflected. Every history frame is transformed independently.
The elevation map reverses its lateral grid axis and negates point Y coordinates,
retaining X, Z, and missing-height markers. Actor, critic, and map encoder inputs
are mirrored consistently. `use_mirror_loss=False`: the logged symmetry loss is
diagnostic; the mirrored samples participate in the PPO update.

Observation dimensions and parameter shapes are unchanged, so existing AME
weights remain loadable. Further training is needed to benefit from augmentation
and the new rewards. Augmentation increases minibatch memory and training compute.
The configuration still uses BatchNorm; `cnn_normalization="none"` remains an option.

This follows AME's map/state cross-attention architecture, but uses one shared,
actor-visible attention query. It does not reproduce AME's separately queried
actor/critic branches with partially tied weights or its two-stage training
curriculum. The existing task-owned actions, rewards, events, terminations,
terrain curriculum, and Newton MJWarp physics configuration are retained.

## Gait penalties inspired by AME_Locomotion

The [reference reward formulas](https://github.com/SII-FUSC/AME_Locomotion/blob/1d3519ee946c846c1c89dd660539a224f99972b5/source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/mdp/rewards.py)
motivate two additional terms:

| Term | Weight | Behavior |
| --- | ---: | --- |
| `feet_air_time_variance` | -0.7 | Penalizes unequal completed swing and stance durations, clipping each duration at 0.5 seconds |
| `feet_stumble` | -2.0 | Penalizes either foot having horizontal contact force greater than four times its absolute vertical contact force |

Timing variance reuses the core velocity term, which disables the penalty for
planar commands at or below 0.1 m/s. Stumble detection uses world-frame normal
contact forces and ignores horizontal forces below 1 N to suppress numerical
noise. It targets foot impacts against stair faces. The existing `feet_slide`
penalty already discourages foot motion during contact.

These are starting weights from the reference's gentler configuration, not a
validated optimum for this task. They add gait balance and stair-impact feedback
without prescribing a fixed gait phase. The existing clearance reward remains.

Terrain-aware rewards use only finite scanner hits within 0.3 m of the root for
base height, or within 0.3 m of each foot for clearance and stance-foot angle.
They do not average or fit terrain across the full scanner grid.

## Inspect attention

With a loaded runner and environment:

```python
runner.alg.eval_mode()
obs = env.get_observations().to(runner.device)
inputs = runner.alg.networks.features("map_encoder", obs)
weights = runner.alg.networks.terms["map_encoder"].attention_weights(inputs)
mean_weights = weights.mean(dim=1)  # (num_envs, 17, 11)
```

These are attention weights over CNN map-token locations. Each head sums to one
over its grid; these weights are not terrain heights or contact probabilities.
Actor export keeps the normal action-only output. Exported observations are
flattened and concatenated in `runner.alg.actor.obs_groups` order, preserving
the same map frame, indexing, and missing-height convention used during training.
