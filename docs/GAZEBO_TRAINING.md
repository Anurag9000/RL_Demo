# Gazebo-first Waffle Pi training

Status: qualification foundation only. The existing tensor simulator is not
Gazebo and remains a baseline. No Gazebo PPO integration or real sensor timing
validation is claimed by this commit.

## Selected stack

Use ROS 2 Jazzy / Gazebo Harmonic on a compatible Ubuntu workstation, official
ROBOTIS turtlebot3_simulations Jazzy model, and evaluate GzDRL as the native
parallel stepping layer. Upstream GzDRL API inspected at commit
3e5c5f9a62c1160b3b2889e6d34d794a5361e178 (MIT).
https://github.com/amaldevh/gz-drl
https://gz-drl.readthedocs.io/en/latest/

The original Waffle Pi SDF (inspected blob
b898477ad9c9d702ecf22853913266fe1d295b6f) retains joints, contact, inertias,
DiffDrive, GPU LiDAR, IMU, and camera. Nominal stock parameters are not
measured hardware fidelity; calibrate floor friction, wheel controller,
LDS revision, noise, timing and payload separately. The original friction
parameters are explicitly described as unreliable in the SDF.

## Qualification fixture

Install the official robot simulation packages and build GzDRL for Harmonic
using its version-specific instructions in an isolated Python environment.
Then generate a reference SDF:

```sh
python -m waffle_pi.gazebo_world --model-sdf "$(ros2 pkg prefix --share turtlebot3_gazebo)/models/turtlebot3_waffle_pi/model.sdf" --output gazebo_runs/reference.sdf
```

The optional scene JSON contains rectangles_m (axis-aligned obstacles) and
static_targets_m ([x,y,radius] stationary cylinders). This is not the dynamic
training scene. Use an empty controlled simulator namespace and separate
Gazebo partition; do not bridge to a physical robot.

```sh
GZ_PARTITION=waffle_reference gz sim -s --headless-rendering gazebo_runs/reference.sdf
```

Stop that server before the native pool probe, which owns its own servers.

```sh
python -m scripts.probe_gazebo_pool --world gazebo_runs/reference.sdf --envs 1 --output gazebo_runs/probe1.json
```

Only after successful execution repeat for 2, 4, 8, etc. in fresh processes.
The probe measures physics and sensor retrieval, not PPO steps or scan
freshness. EGL/OGRE2 support and native pool sensor compatibility must be
checked on the actual host. Do not increase world count beyond memory or
renderer capacity. A state-only probe is diagnostic, not a fidelity result.

## Full rollout integration requirements

- A native controller interface that writes the actual differential-drive
  command or calibrated wheel actuator targets and waits for command
  acceptance at a physics boundary. Do NOT use GzDRL base-link velocity
  setters for robot actuation: they bypass the desired wheel physics.
- Dynamic articulated or calibrated pedestrian models and surveyed layouts.
  ORCA generates desired pedestrian motion; collision and sensor geometry
  remain in Gazebo. Validate pedestrian trajectories and interaction response.
- Physics substeps, timestamped fresh scan/odometry, wheel feedback, controller
  delay, safety gate and sensor-to-policy processing with identical contracts
  to the ROS runtime. Ground truth for reward/termination only.
- Reset simulation, motor commands, sensor and detector memory, GST histories,
  recurrent masks, and scenario RNG without leaking across episodes. GzDRL
  warns against respawning models with rendered sensors enabled.
- Integrate a Gazebo VecEnv with original GST, HH/RH attention, GRU, native
  actor-critic and PPO. Preserve latent action log-probs, timeout bootstrap,
  episode boundaries and contiguous recurrent minibatches.
- Distribute native pools across CPU hosts, use batch inference/GAE/learner on
  GPUs. Synchronize behavior-policy versions per rollout; no uncorrected stale
  rollouts. Optimize data transfers and batching before reducing physics quality.
- Compare complete pipeline results with unoptimized reference and real robot
  logs, including contact, scan, detector, localization, commands and response.
  GPU/Triton/compile and mixed precision require parity checks. Curriculum,
  halving, PBT and checkpoint continuation can be reused after backend integration.

10,000 worlds is a scaling target, not an achieved or guaranteed capacity.
Gazebo rendering, physics and perception resource consumption must be
measured before selecting number of servers, worlds, hosts or GPUs.
