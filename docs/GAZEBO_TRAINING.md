# Gazebo-first TurtleBot3 Waffle Pi training

## Status and model contract

The Gazebo-specific trainer and experiment runners now select Gazebo, never an
implicit tensor fallback. The native worker implementation must be built and
qualified on a real Gazebo Harmonic installation. Python fixture tests are not
physics tests; do not confuse the earlier tensor CI or a stationary GzDRL
probe with a successful Gazebo PPO run. No calibrated policy, full training
run, 10,000-world benchmark, or physical deployment result is supplied.

Use Ubuntu 24.04, Gazebo Harmonic (gz-sim8, gz-transport13, gz-msgs10), and the
official ROBOTIS Jazzy TurtleBot3 Waffle Pi SDF and mesh resources:
https://github.com/ROBOTIS-GIT/turtlebot3_simulations/tree/jazzy
The qualification workflow pins upstream commit
45633014a14e8f438495b532a723e4ad45cbbd31. Robot name must be
turtlebot3_waffle_pi; the two wheel joints, base joint, LDS, IMU and camera
must be present. No Burger, generic differential-drive substitute, or
TurtleBot2i. The original paper's simulator was custom 2D CrowdSim/ORCA.

## Architecture

- gazebo_sim/world.py loads the installed official SDF, resolves meshes, retains
  chassis, inertial data, wheels, joints, camera, IMU and LDS, inserts the
  surveyed scene and adjusts calibrated wheel friction/slip and laser geometry.
  The stock DiffDrive plugin is removed to prevent competing wheel writers.
- gazebo_sim/worker.cc uses the Gazebo Server API and an in-process System.
  Each request latches velocity commands before synchronous physics stepping.
  Left/right wheel references use nominal firmware geometry, command-delay
  queue, per-side gain/deadband/time constant and acceleration/braking limits;
  bounded PI torque acts on the actual wheel joints. No base pose or velocity
  override is used for the robot. This is an actuator emulator, not the OpenCR
  or DYNAMIXEL firmware or a validated motor electrical model.
- Native ground-truth pose, physical contacts, wheel feedback, quantized
  encoder-derived odometry and timestamped GPU laser frames are separate.
  Odom is integrated at physics steps; scan-time odom is interpolated. Wheel
  encoder odom uses nominal .033 m / .287 m geometry; effective radii/track
  from external measurements are calibration targets, not an additional
  fictitious kinematic motion update. Confirm wheel contact and slip in logs.
- gazebo_sim/worker.py gives each native server a unique Gazebo transport
  partition. Sequence IDs, bounded response waits and explicit shutdown
  prevent silent substitution of failed transitions. No ROS bridge to real
  hardware is started. Workers persist within an episode; a full process
  restart on reset deliberately clears physics, rendering and controller
  state. This costs more than in-place reset and must be profiled.
- gazebo_sim/vector_env.py runs independent worlds concurrently using a
  thread pool. ORCA computes crowd commands in a batch; actual Gazebo bodies
  produce leg-like LiDAR returns and collision surfaces. The people are
  simplified two-leg/torso/head bodies with prescribed planar velocity, not
  animated gait or independently validated human dynamics. Static obstacle
  rejection and conservative clearance supplement contact detection; no
  general obstacle-aware crowd planner is claimed.
- Each world passes actual Gazebo GPU LiDAR frames through the shared
  DR-SPAAM DetectorEngine, ROS geometry conversion, Kalman/Hungarian Tracker
  and timestamped GST history. No ground-truth human state enters the
  actor observation. One detector instance per world; current inference is
  not batched across recurrent detector streams. Ground truth is used only
  for reward, collision/termination, metrics and simulated crowd behavior.
- The exact existing GST, HH/RH attention, GRU, actor/critic and latent native
  action contract are reused. PPO stores the sampled latent action and log
  probability, not the actuator-filtered command. Safety runs at a profile
  interval (example .05 s) using the same safe_command and scan_coverage
  functions as the Pi gate. Policy interval is .25 s. Perception faults block
  motion and reset recurrent masks separately from GAE episode masks.
- Rewards retain original terminal values, optional additive hardware-specific
  shaping, GST future-intrusion penalty, and clearance/path metrics. The
  configured time horizon is a task terminal; proper-time-limits bootstrap
  is not supported in this backend. All rollout worlds use one behavior
  policy until the synchronous PPO update finishes.
- train.py and waffle_pi/evaluate.py select Gazebo through --waffle-gazebo.
  scripts/train_gazebo.py sets the required native flags automatically.
  Adaptive/portfolio/benchmark runners default to Gazebo; explicit tensor
  or live/replay choices preserve prior experiments. DDP, masked GRU,
  GPU rollout storage, GAE scan/Triton, compiled actor/GST, BF16/TF32,
  fused Adam and CUDA Graph options remain available subject to their
  existing constraints. Measured-reference mode refuses reduced precision
  and graph/compile flags until separately qualified. Adaptive PPO preserves
  Adam state, does curriculum, halving, PBT, KL feedback and early stopping.
  DDP uses one local world pool per rank; multi-host execution is via
  torchrun, not an automatic remote provisioning service.

## Installation and native qualification

Install Gazebo Harmonic per https://gazebosim.org/docs/harmonic/install_ubuntu/
and native development packages libgz-sim8-dev, libgz-transport13-dev,
libgz-msgs10-dev, nlohmann-json3-dev, cmake and g++. Install the official
robot model using ROS 2 Jazzy packages or a pinned Jazzy source checkout.
ROS is not required inside the training worker itself.

~~~sh
cmake -S gazebo_sim -B build/gazebo -DCMAKE_BUILD_TYPE=Release
cmake --build build/gazebo -j
export WAFFLE_MODEL="$(ros2 pkg prefix --share turtlebot3_gazebo)/models/turtlebot3_waffle_pi/model.sdf"
python -m scripts.qualify_gazebo --model "$WAFFLE_MODEL" --envs 1 --output gazebo_runs/qualification_1
~~~

Install a compatible Python 3.10–3.12 and PyTorch build, then
pip install -r requirements-waffle.txt. Install upstream trusted DR-SPAAM
separately (https://github.com/RWTHVision/2D_lidar_person_detection).
Its checkpoint is not bundled. The detector interface is shared with the ROS
runtime. Set its actual SHA-256 in the profile detector.weights_sha256.
Use the existing released GST and navigation checkpoints from this repository.
Native qualification requires only Python/NumPy, not detector/PPO. It commands
the simulated wheels, checks physical floor contact and movement, scan time
ordering/ray counts, and records truth/odom/wheels. It is not physical
calibration. A render or transport failure aborts. Run EGL/OGRE2 with a
supported GPU/driver or qualify software rendering independently. Repeat with
2, 4, 8, etc. worlds, never assuming linear scaling.

## Measured profile

Copy configs/waffle_gazebo.example.json to a new file. The example is explicitly
illustrative and is not a measured profile. Its detector hash is intentionally
a placeholder; fill in a trusted checkpoint hash even for a pilot. The profile
has hardware identity/LDS/firmware/floor/recording references, motor fits,
laser sample count, angles, range, scan frequency, mounting extrinsics, noise,
dropouts, gazebo physics/gate intervals, torque PI and limit parameters, contact
friction/slip, pedestrian geometry and scene cases with train/validation/test
splits. Select the actual LDS revision from observed LaserScan metadata, not
from the example's nominal LDS-01-like settings. Measure actual wheel response,
external motion truth, stopping distance, scan errors and detector accuracy
using independent fit/holdout datasets (waffle_pi.identify). Calibrate mass,
contact solver and payload against the actual robot as necessary; these are
not automatically identified by the current scripts. The official SDF's
nominal masses/inertias remain unchanged unless a reviewed model revision
changes them.

Set gazebo.model_sha256 from source_digest (includes the SDF and mesh bytes),
gazebo.worker_sha256 from the compiled binary, and gazebo.safety_sha256 from the
independent supervised safety record. Supply physics_calibration_record and
hardware evidence, and status measured only after reviewing residuals. These
checks verify provenance, not that a self-declared measured profile is correct.
Measured mode requires surveyed cases for each crowd count and split, detector
weights hash, safety file hash and model/worker hashes. The file must not
change during continuation. Current SDF scene support is square outer boundary
plus axis-aligned rectangles and explicit unobstructed pedestrian waypoints,
not arbitrary map import. Add measured static geometry before final training.

## Actual PPO training

Use a fresh output directory. For a native integration smoke test with an
illustrative profile whose trusted detector hash is filled in:

~~~sh
export PROFILE=calibration/waffle_gazebo.json
export DETECTOR=calibration/dr_spaam.pth
python -m scripts.train_gazebo --waffle-gazebo-model "$WAFFLE_MODEL" \
  --waffle-fidelity-profile "$PROFILE" --waffle-detector-weights "$DETECTOR" \
  --waffle-init-cartesian --resume \
  --load-path trained_models/GST_predictor_rand/checkpoints/41665.pt \
  --no-cuda --num-processes 2 --num-mini-batch 1 \
  --num-steps 4 --seq_length 4 --ppo-epoch 1 --num-env-steps 8 \
  --waffle-humans 1 --save-training-state --output_dir experiments/gazebo_smoke
~~~

The smoke test proves only end-to-end execution. Use the same profile with
measured values, actual independent safety calibration and
--waffle-require-measured for final reference training. Training uses transfer
learning and a native reinitialized action head; GST stays frozen. For a
scratch control omit --resume and --waffle-init-cartesian. For continuation
use --continue-state instead of weights-only resume, with matching contracts
and new rollouts. A tensor-trained checkpoint is not a Gazebo learner state.

## Paper-regime scenario banks

The profiles
mission/profiles/waffle_gazebo_paper_randomized_500x3.json and
mission/profiles/waffle_gazebo_paper_nonrandomized_500x3.json contain separate,
seeded train/validation/test banks (500 cases per split, four humans per case).
Rebuild them from an illustrative base profile with:

~~~sh
python scripts/generate_paper_regime_scenarios.py \
  --base mission/profiles/waffle_gazebo_paper_base.json \
  --output-dir mission/profiles --per-split 500
~~~

Both banks use each scenario's fixed cyclic waypoint route, advancing to the
next waypoint when a human reaches its active destination. The randomized bank
uses human radii 0.3–0.5 m and preferred speeds 0.5–1.5 m/s; at each five-second
policy boundary, each human independently receives a sampled temporary goal
with probability 0.5. On arrival at that temporary goal, its cyclic scenario
route resumes at the next waypoint. The nonrandomized bank fixes radius at
0.3 m and speed at 1.0 m/s and never samples a random goal. Neither regime
resamples a goal merely because a route endpoint was reached. Human goals and
simulator truth remain inside Gazebo; the policy receives only detector/tracker
observations. The training process writes gazebo_environment_summary.json at
shutdown with goal-change/sample-failure counts, episodes started, worker
timing and detector statistics when available.

These banks are nominal and illustrative, not hardware-calibrated crowd data.
They cover four humans only, so they do not yet establish the paper's full
crowd-count coverage or satisfy measured-profile qualification. Train and tune
on train/validation; reserve each bank's test split for one final locked
evaluation. A short integration smoke is not a quality or convergence result.

### Concurrent curriculum and policy search

~~~sh
python -m scripts.train_adaptive --backend gazebo \
  --gazebo-model "$WAFFLE_MODEL" --fidelity-profile "$PROFILE" \
  --detector-weights "$DETECTOR" --safety-calibration calibration/safety.json \
  --require-measured --gpus 0 --envs 4 --minibatches 2 \
  --humans 4 --rung-updates 8 24 64 128 256 \
  --mode hybrid --output experiments/gazebo_adaptive
~~~

The world count shown is a starting experiment, not a measured optimum. The
scheduler uses scene split validation/test, confidence-aware ranking in
measured mode, and a separate final evaluation. All candidates train in
Gazebo. Different GPUs are independent learners; synchronous DDP is separate.
The number of world processes across all ranks and learners is additive.
Use CPU workers when no CUDA is available. The original paper's 20M steps
are a reference, not a required budget or convergence guarantee.

### Benchmarking and distributed rollout

~~~sh
python -m scripts.benchmark_sweep --backend gazebo --gazebo-model "$WAFFLE_MODEL" \
  --fidelity-profile "$PROFILE" --detector-weights "$DETECTOR" \
  --envs 2 4 8 --minibatches 1 2 --presets eager scan \
  --output benchmarks/gazebo_fp32
~~~

Use --device cpu if needed; benchmark a specific crowd size with --humans.
Benchmark logs/JSON report end-to-end rates, not physical performance.
Advanced presets require supported CUDA and numerical parity testing. Do not
reduce physics steps or skip required sensors just to increase reported FPS.
For DDP use torchrun --nproc_per_node=N train.py --waffle-pi
--waffle-native-action --waffle-gazebo plus the same model/profile/weight
flags and normal trainer parameters. For multiple hosts use torchrun's
documented rendezvous flags, identical source/model/weights and a shared or
rank-zero accessible output directory. Native worker logs are per process.
No 10,000-world deployment, remote cluster provisioning, GPU physics,
shared-memory binary sensor transport, batched detector, in-place reset,
or performance claim is included. GzDRL remains a separate qualification
experiment, not the current actuation path.

### Evaluation and physical validation

Use the Gazebo trainer with --waffle-eval-episodes N,
--waffle-scene-split test, --load-path PATH, --waffle-eval-json PATH and
identical hardware profile/worker/weights. Keep run_args.json with the
checkpoint; hashes prevent silent backend/model changes. Evaluate separately
on held-out scenarios, independent seeds, repeated runs and physical recordings.
Report success, collisions, timeouts, clearance quantiles, interventions,
travel time/path length, and straight-line SPL. Wilson intervals are descriptive
for correlated scenarios. Do not use simulator collision fractions as physical
safety certification. See [the Waffle Pi paper metric schema](WAFFLE_PAPER_METRICS.md)
for exact denominators, percentage fields, and null behavior for success-only NT/PL.

Gazebo evaluation writes one compressed, non-pickle NPZ paper-metric truth
trace per completed episode. By default, --waffle-eval-json OUT/metrics.json
places them in OUT/metrics_traces; --waffle-eval-trace-dir DIR sets an explicit
directory. Each trace stores robot decision poses, ground-truth human positions
including the four-frame post-terminal suffix, human/robot radii, stable Gazebo
crowd entity IDs, actual simulator timestamps, scenario/split/seed, outcome and
profile/checkpoint/detector/model-SDF/worker file hashes. The evaluator
independently recomputes ITR and SD, checks unique scenario coverage and
online/offline parity, and reports the float64 trace scores after allowing only
1e-6 aggregate tolerance for online float32 rounding. A missing future horizon
or repeated/missing scenario trace is an error.

Recompute or inspect them later with:

~~~sh
python -m waffle_pi.offline_metrics --trace-dir OUT/metrics_traces --output OUT/offline_metrics.json
~~~

These traces support the paper's future-position ITR and current-time SD audit;
they do not constitute a full action, wheel, scan or video recording.

The real ROS 2 path uses actual LDS, DR-SPAAM, timestamped tracker/GST,
native actor, and independent Pi safety gate. Pass --fidelity-profile to
the detector and policy; the detector verifies model parameters, weight hash
and actual LDS geometry. Gazebo checkpoint motor-request mode additionally
requires matching detector and safety files. Keep policy in shadow mode and
gate in dry run until supervised calibration, firmware watchdog, e-stop,
source exclusivity, ROS topic/TF timing and actual safe stopping tests pass.
This does not automatically arm the robot. Raspberry Pi 4 deployment,
firmware behavior, AMCL/SLAM, actual IMU fusion, command/network jitter,
detector compute deadlines, leg gait and reflectivity, battery/load effects
and multi-world native renderer behavior remain to be qualified or modeled.
Camera and IMU exist in the official model; their complete streams are not
currently fed to the navigation learner, matching the original architecture.

## Validation commands

~~~sh
bash scripts/validate_waffle.sh
python -m scripts.qualify_gazebo --model "$WAFFLE_MODEL" --envs 1 --output gazebo_runs/qualification_new
~~~

The native GitHub Actions job builds against Harmonic and attempts a real
wheel/scan qualification with the pinned official SDF. Check its actual
conclusion. It does not install DR-SPAAM weights or train the policy.
Python integration fixtures use synthetic frames only to exercise the API,
episode/reset semantics and separate recurrent masks. Native execution,
detector inference, ROS integration and convergence are separate evidence.

The stock Gazebo fixture remains available using waffle_pi.gazebo_world
and scripts.probe_gazebo_pool for comparing against upstream DiffDrive.

## Dual EPYC/H100 performance path

See [H100_EPYC_TRAINING.md](H100_EPYC_TRAINING.md) for the new native shared-memory scan transport, overlapping resets, NUMA affinity, optional shared detector batching, pinned transfers, actual NCCL probe, matched-global-batch single/DDP comparison and independent candidate scheduling. These options retain Gazebo and the official Waffle Pi. Earlier statements about JSON-only communication and separate detector weights describe the reference path; the new options are not yet hardware-benchmarked. Measured mode uses an unbatched FP32 detector until parity is qualified.

## Paper-regime training launcher

Run the committed script on the H100 with the qualified run-root and Python
environment exported. It locks each run ID, records launch arguments and exit
status, loads the official randomized actor with a fresh PPO optimizer, and
saves recoverable state every ten updates. These examples use 16 Gazebo worlds
per regime and the corresponding train split:

~~~sh
export WAFFLE_RUN_ROOT=/path/to/durable/RL_Demo-run
export WAFFLE_REPO_ROOT=$PWD
export WAFFLE_VENV=/path/to/qualified/venv
mkdir -p "$WAFFLE_RUN_ROOT/logs" "$WAFFLE_RUN_ROOT/mission"
nohup setsid bash scripts/run_paper_regime_training.sh randomized 0 439 16 4 250000 paper_randomized_gpu0_seed439_20261128 \
  > "$WAFFLE_RUN_ROOT/logs/paper_randomized_gpu0_seed439_20261128.stdout" 2>&1 </dev/null &
echo "$!" > "$WAFFLE_RUN_ROOT/mission/paper_randomized_gpu0_seed439_20261128.launcher.pid"
nohup setsid bash scripts/run_paper_regime_training.sh nonrandomized 1 440 16 4 250000 paper_nonrandomized_gpu1_seed440_20261128 \
  > "$WAFFLE_RUN_ROOT/logs/paper_nonrandomized_gpu1_seed440_20261128.stdout" 2>&1 </dev/null &
echo "$!" > "$WAFFLE_RUN_ROOT/mission/paper_nonrandomized_gpu1_seed440_20261128.launcher.pid"
~~~

These are training commands, not quality claims. The test splits stay locked.

The launcher runs the repository train.py directly with normal on-demand CUDA allocation; it does not apply the older 20% per-process allocator cap. Check free VRAM before admission and keep the active workload within the verified available capacity.
A handled TERM to the launcher forwards SIGINT to its Python trainer, which
runs Gazebo cleanup and writes gazebo_environment_summary.json. Resume only
from a compatible training_state.pt using the matching source, profile and
run arguments.
