# CrowdNav repository relationship and implementation guide

> The new Waffle Pi-specific primary backend is `gazebo_sim/` in the parent repository. See [GAZEBO_TRAINING.md](GAZEBO_TRAINING.md). The pinned TurtleBot2i submodule and the original CrowdSim code remain historical reference assets, not the hardware used for the new Gazebo training.

This guide describes the two independent Git repositories currently arranged as a parent checkout with a nested checkout:

```text
RL_Demo/                                    # parent: CrowdNav++ simulation, learning, evaluation, GST inference
└── CrowdNav_Sim2Real_Turtlebot/             # pinned submodule: sim-to-real procedure and ROS support assets
```

The nested checkout is maintained as an independent Git repository and is pinned by the parent as a Git submodule. The parent records the nested repository URL and the exact submodule commit; each checkout keeps its own history and working-tree status. Commit file changes inside the repository that owns them, then update the parent submodule pointer when you want the parent project to use a different nested revision.

## Checkout identity at the time this guide was written

| Checkout | Remote | Branch and HEAD | Role |
|---|---|---|---|
| Parent local checkout `RL_Demo` | `origin` upstream: `https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph`; publish destination: `https://github.com/Anurag9000/RL_Demo` | `main`, based on `390773137be04ed14e27620dc5fd7c5e1a5b1f62` plus the local documentation/submodule commit | CrowdNav++ navigation policy, simulated crowd environment, PPO training/evaluation, and bundled GST prediction/inference code and artifacts. |
| Nested `CrowdNav_Sim2Real_Turtlebot` submodule | `https://github.com/Shuijing725/CrowdNav_Sim2Real_Turtlebot.git` | `master`, pinned at `c21575abeffea89dfc0abe5d4b3fb97ada45650c` | Reference deployment procedure and a small set of ROS host/robot assets for transferring a trained navigation policy to TurtleBot 2i. |

These are provenance snapshots, not permanent revision pins. Recheck `git status`, `git rev-parse HEAD`, and `git remote -v` before comparing behavior or reproducing results.

## What each repository is for

### Parent: CrowdNav_Prediction_AttnGraph

The parent implements the research and simulation side of CrowdNav++. Its main job is to learn or evaluate a robot navigation policy in crowd scenarios. The policy consumes robot and pedestrian states, represents interactions with recurrent and attention-based networks, and can optionally use future pedestrian trajectories from GST to alter the navigation observation/reward. It also contains the ROS environment adapter used when evaluating a learned checkpoint against a real robot interface.

The parent is the source of truth for:

- simulated robot and human dynamics, goals, observations, rewards, termination, and crowd scenarios;
- ORCA, social-force, and learned SRNN/self-attention policy construction;
- PPO rollout collection, optimization, checkpoint saving, and navigation evaluation;
- GST model implementation bundled under `gst_updated/`, the parent-side inference wrapper, and the predictor artifacts/configuration it loads;
- the Python ROS environment adapter that translates robot/pedestrian messages into policy observations and policy actions into velocity commands.

The root README identifies the research contribution as intention-aware navigation with an attention-based interaction graph and describes GST as an optional future-trajectory predictor. The README also links this separate sim-to-real tutorial. Its statement that the repository does not include trajectory-predictor training should be read narrowly: this checkout does contain GST experiment scripts under `gst_updated/scripts/experiments/`, but those scripts have historical data/environment assumptions and are not evidence of a currently reproducible training pipeline.

### Nested: CrowdNav_Sim2Real_Turtlebot

The nested repository is a deployment and integration reference. It documents the hardware/software assumptions, ROS package setup, mapping/localization and people-detection sequence, network configuration, and how to run a trained parent policy through the parent’s ROS environment. It is not a second navigation-policy implementation and it does not contain the parent PPO trainer, crowd simulator, GST predictor, or a self-contained ROS package for the whole system.

Its checked-in implementation assets have focused jobs:

- `README.md`: host-versus-robot setup, ROS dependencies, map creation and localization procedure, detector setup, policy handoff back to the parent checkout, and reference hardware/software requirements.
- `tb2.bash`: placeholders for `ROS_MASTER_URI` (robot ROS master) and `ROS_IP` (host address). Replace these with the site-specific network values before use; the committed values are placeholders.
- `t265.launch`: launches the RealSense T265 ROS camera/odometry path using `realsense2_camera` launch content and configurable device/TF arguments.
- `findloc_bgrm.py`: ROS LiDAR scan filter that uses a saved static map to remove scan returns near static obstacles, republishes filtered person points, and optionally publishes a debug map visualization. It depends on the surrounding ROS graph, map files, frames, and detector topics described in the README; it is not a general standalone CrowdNav runtime.
- `figures/3humans.gif`, `figures/4humans.gif`, and `figures/real.gif`: qualitative demonstrations, not measured safety or reproducibility evidence.

The nested checkout describes an integration procedure; the parent owns the Python adapter that actually connects observations/actions to the policy. The nested README is therefore procedural context, while parent `crowd_sim/envs/ros_turtlebot2i_env.py` is the implementation reference for the policy-facing ROS interface.

## How the two repositories fit together

The intended flow is:

1. **Configure the parent policy and environment.** Select the simulator and prediction mode in the parent configuration and matching environment ID/architecture in `arguments.py`.
2. **Train or select a parent checkpoint.** `train.py` creates the policy checkpoint and stores run-specific arguments/configuration under the selected output directory. Existing checkpoints and logs are artifacts of past runs; they do not establish that current code/configuration will reproduce those runs.
3. **Validate in simulation.** The nested README asks users to test the trained policy in simulation before attempting robot execution. The actual simulator and evaluation logic are in the parent checkout.
4. **Prepare the ROS environment following the nested procedure.** Bring up the robot base and sensors, localization/mapping, and pedestrian detector; set the ROS master and host addresses; configure the site-specific map and frame/topic graph.
5. **Run the parent adapter with a saved policy.** The parent environment factory selects `rosTurtlebot2iEnv-v0`; that class subscribes to robot/person state and publishes velocity commands. The nested launch and shell files help establish the external ROS side of this connection.

The boundary is important: ROS setup, device drivers, maps, detector weights, and network values are not supplied or controlled by the parent model code. Conversely, changing the nested README or launch file does not change PPO, policy architecture, simulated rewards, or the observation/action conversion implemented in the parent.

## Which repository and files to change

| Desired change or question | Repository | Primary files to inspect/change | Related files and cautions |
|---|---|---|---|
| Change simulator dimensions, human count, robot kinematics, prediction method, reward, time limits, sensor range, or sim-to-real flags | Parent | `crowd_nav/configs/config.py` | Then trace the selected class in `crowd_sim/envs/` and compare the saved run's `trained_models/<run>/configs/config.py`. `predict_method` and `env_name` must be compatible. |
| Add or alter crowd simulation dynamics, observations, reward, reset, done, or rendering | Parent | `crowd_sim/envs/crowd_sim.py` | This is the base. Follow inheritance into `crowd_sim_var_num.py`, `crowd_sim_pred.py`, or `crowd_sim_pred_real_gst.py`; verify which registered environment is active. Shared state/action entities live in `crowd_sim/envs/utils/`. |
| Add/alter variable-population or trajectory-prediction simulation behavior | Parent | `crowd_sim/envs/crowd_sim_var_num.py`, `crowd_sim/envs/crowd_sim_pred.py`, `crowd_sim/envs/crowd_sim_pred_real_gst.py` | `crowd_sim_var_num_collect.py` is a collection-oriented variant. Check environment registration in `crowd_sim/__init__.py` and wrapper/factory behavior in `rl/networks/envs.py`. |
| Change simulator-side robot/human state, motion, action types, or outcome labels | Parent | `crowd_sim/envs/utils/agent.py`, `human.py`, `robot.py`, `state.py`, `action.py`, `info.py` | `recorder.py` owns trajectory/action recording formats. Update callers that consume flattened observations or recorder CSV schemas. |
| Change policy choice or scripted ORCA/social-force actions | Parent | `crowd_nav/policy/policy_factory.py`, then `crowd_nav/policy/orca.py` or `social_force.py` | Shared policy interface is `crowd_nav/policy/policy.py`; robot kinematics action conversion is in `crowd_nav/policy/srnn.py`. |
| Change neural navigation architecture, attention, recurrent state, or observation feature encoding | Parent | `rl/networks/model.py`, `rl/networks/srnn_model.py`, `rl/networks/selfAttn_srnn_temp_node.py`, `rl/networks/network_utils.py` | Trace shapes and masks from simulator observation through `rl/networks/envs.py`; recurrent rollout storage is `rl/networks/storage.py`. Saved checkpoint compatibility depends on module names and tensor shapes. |
| Change PPO losses, clipping, minibatches, or optimizer update | Parent | `rl/ppo/ppo.py` | Training schedule and rollout loop are in `train.py`; storage/returns are in `rl/networks/storage.py`. `train.py` currently constructs PPO regardless of some parser algorithm choices; inspect the active path instead of assuming parser flags are wired. |
| Change environment IDs, worker construction, vector wrappers, tensor conversion, or auto-reset behavior | Parent | `crowd_sim/__init__.py`, `rl/networks/envs.py`, `rl/networks/dummy_vec_env.py`, `rl/networks/shmem_vec_env.py`, and `rl/vec_env/` | Evaluation depends on terminal-observation and `info` semantics; trace reset behavior before changing episode metrics. |
| Change command-line defaults, network sizes, rollout length, checkpoint load path, process count, or output directory | Parent | `arguments.py` and `train.py` | `test.py` uses its own top-of-file test defaults and loads saved run settings with fallbacks; inspect `trained_models/<run>/arguments.py` and `configs/config.py` too. Some declared arguments may not be consumed by the active training path. |
| Change testing metrics, checkpoint evaluation, episode aggregation, or plots | Parent | `test.py`, `rl/evaluation.py`, `plot.py` | Compare environment outcome `info` types and auto-reset observations. Historical logs under `trained_models/` are not proof that the current evaluator generated them. |
| Change GST architecture, attention, graph/edge selection, temporal encoder/decoder, or forecast loss | Parent | `gst_updated/src/gumbel_social_transformer/` (especially `st_model.py`, `gumbel_social_transformer.py`, edge selectors, attention, and temporal modules) | Dataset tensor utilities are under `gst_updated/src/mgnn/` and `pec_net/`. The seven-pass static index under `.codex/repository-audits/CrowdNav_Prediction_AttnGraph/` gives per-file notes; do not assume all historical branches are active. |
| Change how the navigation policy calls GST or consumes predicted trajectories | Parent | `gst_updated/scripts/wrapper/crowd_nav_interface_parallel.py` and `rl/vec_env/vec_pretext_normalize.py` | Follow construction in `rl/networks/envs.py`, prediction mode in `crowd_nav/configs/config.py`, and any rendering/reward shaping in `crowd_sim/envs/crowd_sim_pred_real_gst.py`. Keep history length, masks, coordinate frame, and forecast horizon consistent. |
| Change GST dataset generation, packed windows, ETH/UCY/SDD representations, or training/evaluation scripts | Parent | `gst_updated/scripts/experiments/`, `gst_updated/src/mgnn/`, `gst_updated/src/pec_net/`, `gst_updated/run/`, and `gst_updated/tuning/` | Some data-generation scripts call creator programs absent from this checkout; some scripts use historical absolute paths or import paths. Confirm the actual executable workflow and data provenance before treating a script as reproducible. |
| Change robot/host installation instructions, ROS versions, physical setup sequence, mapping/localization, detector package setup, or operational checklist | Nested | `CrowdNav_Sim2Real_Turtlebot/README.md` | External ROS packages, map files, detector code/checkpoints, and network addresses are not included in this checkout. Confirm instructions against the actual robot/host before execution. |
| Change ROS master/host network variables | Nested | `CrowdNav_Sim2Real_Turtlebot/tb2.bash` | Values are placeholders. The parent adapter's ROS topic names and message types are in `crowd_sim/envs/ros_turtlebot2i_env.py`; changes must match on both sides. |
| Change T265 camera launch arguments or its ROS launch wiring | Nested | `CrowdNav_Sim2Real_Turtlebot/t265.launch` | The parent adapter currently requests T265 frame transforms. The nested README also describes LiDAR localization as an alternative, so do not assume the adapter supports that alternative without inspecting/changing its transform lookup. |
| Change static-obstacle scan filtering or debug visualization | Nested | `CrowdNav_Sim2Real_Turtlebot/findloc_bgrm.py` | This script assumes specific map YAML/PGM conventions, TF frames, scan geometry, ROS topics, and detector wiring. Coordinate with the README and parent ROS adapter's detection input contract. |
| Change conversion of ROS pose/person detections to the policy observation, action-to-Twist mapping, episode reset, ROS timing, or shutdown behavior | Parent | `crowd_sim/envs/ros_turtlebot2i_env.py` | This is not implemented in the nested repo. Check `crowd_sim/envs/crowd_sim_pred_real_gst.py` inheritance and saved `sim2real` settings as well. |
| Change demo videos/images | Nested | `CrowdNav_Sim2Real_Turtlebot/figures/` | These are qualitative assets; changing them does not change system behavior or provide quantitative validation. |

## Parent-side execution map

For a source-level trace of the normal navigation loop, follow:

```text
train.py / test.py
  -> crowd_sim/__init__.py environment registration
  -> rl/networks/envs.py environment and vector-wrapper construction
  -> crowd_sim/envs/<selected simulator>.py observation, reward, done, reset
  -> rl/networks/model.py plus srnn/self-attention implementation
  -> rl/networks/storage.py and rl/ppo/ppo.py during training
  -> rl/evaluation.py during testing
```

For inferred human trajectories, additionally follow:

```text
crowd_nav/configs/config.py prediction settings
  -> rl/networks/envs.py / rl/vec_env/vec_pretext_normalize.py wrapper selection
  -> gst_updated/scripts/wrapper/crowd_nav_interface_parallel.py
  -> gst_updated/src/gumbel_social_transformer/ model and graph attention
  -> forecast tensors returned to the CrowdSimPredRealGST path
```

For robot execution, follow:

```text
Nested README.md (ROS graph/setup procedure)
  -> nested tb2.bash / t265.launch / findloc_bgrm.py (supporting ROS assets)
  -> parent crowd_sim/envs/ros_turtlebot2i_env.py (policy-facing environment)
  -> parent rl/networks/envs.py (environment selection)
  -> parent test.py and rl/evaluation.py (checkpoint inference/evaluation)
```

## Compatibility and evidence cautions

- Environment selection is split across `crowd_nav/configs/config.py` (`sim.predict_method`, wrappers and state shapes) and `arguments.py` (`env_name`, model/network sizes). Changing only one side can select an incompatible environment or architecture.
- Evaluation settings may come from a saved run directory while root code/config remains active. Compare root files, saved `trained_models/<run>/arguments.py`, saved `configs/config.py`, checkpoint tensor shapes, and the historical log as separate evidence.
- Prior static review found a concrete mismatch in the checked-in non-randomized saved run: `41200.pt` expects a 12-feature first spatial-attention layer while the saved environment/active network path exposes two spatial features for that environment ID. The randomized and `my_model` checkpoint metadata also showed the 12-feature first layer. Recheck current checkpoint and model code before attempting strict loading; this is a static incompatibility finding, not a runtime test result.
- The parent ROS adapter requests T265 transform frames, while the nested README discusses LiDAR localization as an option. That difference must be resolved in code/configuration for any LiDAR-only deployment; the README statement alone does not switch the adapter's transform source.
- Terminal observations can be replaced by reset observations in vector backends. Evaluation path length and GST history at episode boundaries need explicit review when changing done/reset logic.
- Static source mapping does not establish that ROS, Gazebo, the robot, a detector, a checkpoint, or a training recipe works end to end. Historical performance logs, plots, and videos are artifacts with their own provenance, not fresh validation.

## Repository-local versus external state

The nested checkout is a pinned submodule, recorded in the parent by `.gitmodules` and a gitlink. A fresh clone must initialize submodules (for example, `git clone --recurse-submodules ...` or `git submodule update --init --recursive`) to materialize its files. To change nested files, work and commit inside `CrowdNav_Sim2Real_Turtlebot`; to make the parent use that new revision, commit the updated submodule pointer in the parent. Parent `git status` reports nested worktree changes separately from the submodule pointer.

## Review basis and limits

This guide combines the current README/file layout and checkout state with a pre-existing seven-pass static review of the parent and nested sources. That review covered parent `390773137be04ed14e27620dc5fd7c5e1a5b1f62` and nested `c21575abeffea89dfc0abe5d4b3fb97ada45650c`, including 179 files and 620 named Python definitions. It performed syntax/static/artifact inspection; it did not establish live ROS or robot operation, training completion, performance reproduction, or full external dependency availability. Recheck any source details against the current files if either checkout advances.
