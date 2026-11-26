# Waffle Pi experimental adaptation

Opt-in simulation path preserving the Cartesian checkpoint action semantics.
Use the original legacy dependencies in an isolated environment.

## Evaluate before retraining

From repository root:
```sh
python train.py --waffle-pi --waffle-eval-episodes 100 --load-path trained_models/GST_predictor_rand/checkpoints/41665.pt
```

## Fine-tune if needed

```sh
python train.py --waffle-pi --resume --freeze-human-attention --fast-training --load-path trained_models/GST_predictor_rand/checkpoints/41665.pt --output_dir trained_models/waffle_transfer --num-processes 32 --num-mini-batch 4 --num-env-steps 500000
```

Choose a new output directory. For full policy fine-tuning omit
--freeze-human-attention. GST weights stay frozen. Evaluate the resulting
checkpoint using the first command with the corresponding --load-path and
identical Waffle profile options. The original saved test.py configurations
are not the entry point for this profile.

The profile uses nominal wheel geometry, coupled saturation, a provisional
first-order motor model, 0.25 s policy steps, four pedestrians and a 90 s
timeout. It retains the existing circle-crossing human generator and original
pre-transition reward timing (including unicycle turn penalties); it is an
open arena, not a measured wall-aware lab model. Sensor range, footprint,
motor lag and acceleration must be calibrated. No physical motion is enabled.

Headless training omits rendering-only prediction IPC; recurrent minibatches
are gathered by complete environment sequences; attention masks stay on device;
unused attention scores are suppressed; optional compact legacy edge state
reduces storage; PPO metrics transfer once per update. Waffle GST histories
clear at episode boundaries and reset-observation penalties are not assigned
to terminal transitions. Shared changes also affect the original code path;
compare against the unchanged main branch for baseline reproduction.

Tests: python -m unittest discover -s tests

Still needed: measured perception and dynamics, wall geometry, physical ROS
adapter and independent watchdog/collision monitor, trajectory replay parity, compiled neural training and DDP. Do not
run the TurtleBot2i ROS adapter on the Waffle Pi. No speed or real-world
performance is claimed before benchmarks and supervised validation.

## Experimental GPU trajectory replay

Generate a pool using the existing native RVO2 solver on independent CPU
workers. Each trajectory retains its matching robot start/goal.

```sh
python train.py --waffle-pi --waffle-build-pool pools/train.npz --waffle-pool-count 1024 --num-processes 8
python train.py --waffle-pi --waffle-replay pools/train.npz --resume --fast-training --load-path trained_models/GST_predictor_rand/checkpoints/41665.pt --output_dir trained_models/waffle_replay --num-processes 256 --num-mini-batch 16 --num-env-steps 500000
```

The replay backend keeps the pool, dynamics, observations, GST and PPO
rollouts on the training device. No live ORCA solve or worker IPC occurs
during replay. Optional --waffle-compile-dynamics needs a separately validated
PyTorch 2.x environment. Test on small batches first; benchmark end-to-end
steps/sec and time to a held-out target, not just robot dynamics throughput.
Pool construction is CPU-parallel and memory scales with scenario count.
Use different seeds for held-out evaluation; evaluate against live ORCA too.

**Important:** replay is an approximate training backend, not exact original
ORCA equivalence: upstream human respawns and goal samples depend on robot
position, which is stationary during precomputation. It also introduces
square wall collisions and post-transition reward/termination. It does not
simulate sensor occlusion, tracking errors, actual floor geometry or
swept collision checks. No real hardware use is authorized by this model.
Keep the live Waffle CPU path as the reference. Disable
--use-proper-time-limits for replay. GST remains unchanged.

## Experimental ROS 2 shadow integration

These nodes are an initial implementation, not validated on hardware.
ROS 2 is assumed only for this adapter; a ROS 1 robot needs a separate
adapter. Use the actual Waffle Pi bringup, TF tree and detector package.
On a host with ROS 2 and compatible PyTorch/GST dependencies:

```sh
python -m waffle_pi.ros2_policy --gst-dir gst_updated/results/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000_rand/sj
```

Default publishes **only** /crowdnav/cmd_vel_shadow. Input /odom,
timestamped detector PoseArray /dr_spaam_detections, and PoseStamped
/crowdnav/goal in odom coordinates are required. The policy applies
timestamped TF2 for detections; it does not provide AMCL, DR-SPAAM,
map filtering or calibration. Greedy fixed-slot tracking is provisional;
evaluate identity switches, loss of detection and temporal spacing.
Use synchronized clocks. The model must finish inference inside 250 ms.

On the Pi, start the gate in dry-run mode:
```sh
python -m waffle_pi.ros2_safety_gate
```
It outputs to a diagnostic topic, not /cmd_vel. The gate rejects stale
commands/scans, invalid or close ranges, reverse commands and software
e-stop; a true Bool on /crowdnav/estop latches it until restart. Its
stopping model is conservative but not certified; the stock lidar has
blind spots. No hardware testing was performed. Verify measured stopping
distance and actual Twist/TwistStamped interface, manual emergency stop,
exclusive command publisher, network/driver failure behavior and safety
of controlled empty-room trials before arming. The explicit
--enable-requests and --arm options are NOT a deployment recommendation
until those gates are passed. The original TurtleBot2i adapter stays
unchanged; this is not a full ROS integration or a person-safe system.

## Opt-in accelerated PPO

Use a separately installed PyTorch 2.x environment and benchmark each option.
`--compile-policy --masked-gru --gae-backend scan --fused-adam --tf32` enables
fixed-shape recurrent processing, associative GAE and fused Adam. `--compile-gst`
compiles the frozen predictor independently. `--compile-mode reduce-overhead`
requests compiler CUDA graph optimizations where supported (capture is not guaranteed).
`--amp-bf16` enables autocast for actor-critic rollout and learning on compatible CUDA GPUs;
Gaussian parameters, PPO ratios and advantages remain FP32. GST stays FP32.
`--gae-backend triton` requires a compatible Triton CUDA install. Use the reference
GAE to verify parity first. GPU RNG replaces CPU RNG for GST edge sampling; this
preserves the distribution but not the original seed-by-seed stochastic sequence.

For DDP, use `torchrun --standalone --nproc_per_node=2 train.py` followed by Waffle
training options. `--num-processes` is environments PER rank; the transition budget
is GLOBAL. DDP uses NCCL on CUDA and synchronizes global advantage statistics;
only rank zero writes checkpoints and logs (episode reward statistics local to rank 0).
Do not combine DDP and policy compilation in this initial version. Use replay for
DDP to avoid forking CPU workers after CUDA/NCCL initialization. The first-order
model, replay reward, frozen GST, and action contract are unchanged.
These paths require runtime smoke tests and throughput benchmarks; no speedup or
CUDA graph capture is established merely by enabling the flags.

## Live tensor ORCA (experimental)

`--waffle-live-orca` generates a GPU-resident bank of collision-free initial
circle layouts and advances people online with dense RVO2-style ORCA
half-planes, speed-circle LP2 and infeasibility LP3. No trajectory pool or CPU
worker process is needed. The solver loops only over constraints, not worlds
or pedestrians. `--waffle-reactive-humans` adds the robot to human ORCA inputs
as a reciprocal agent. There is no obstacle-line ORCA solver; pedestrians
are bounded by the square arena. Goals switch to the opposite endpoint;
no teleportation, and scenarios are not equivalent to the original generator.
No claim of numerical RVO2 parity until the optional reference test passes.
`--waffle-compile-dynamics` also compiles the live ORCA solver; compilation
may be expensive for many humans. Start with four slots.

For both tensor backends optional synthetic perception flags:
`--waffle-occlusion --waffle-noise 0.03 --waffle-dropout 0.05 --waffle-detection-delay 1 --waffle-pose-drift 0.002`.
These numbers are examples, not measured LDS characteristics. Occlusion uses
center rays and disks, dropout is temporally correlated, delay is fixed in
policy ticks, and pose drift is a random walk. No false-positive or leg-scan
model is provided. The tensor backend uses a swept relative line segment
for pedestrian collision checks (an approximation to curved robot motion),
with discontinuity resets on replay respawns. It is still an experimental
reward/geometry variant; compare separately against the reference simulator.

## Calibration, evaluation and throughput

Capture supervised low-speed calibration logs as CSV (seconds, m/s, rad/s):
`t_s,cmd_v,cmd_w,meas_v,meas_w`. Include different velocity steps in both
training and holdout portions. Do not use these commands to move the robot;
record existing supervised measurements. Offline fit:

```sh
python -m waffle_pi.calibration motor measurements.csv calibration/motor.json
```

`--waffle-calibration calibration/motor.json` loads fitted tau, wheel acceleration
and rounded actuation delay. This overrides the corresponding simulation flags;
nonzero delay requires a tensor backend. The fit is not a braking-safety bound.
`--waffle-dynamics-randomization 0.2` independently varies tau and acceleration
per reset, within +/-20 percent, without exceeding wheel speed limits.
Detector calibration CSV: `distance_m,detected,truth_x,truth_y,det_x,det_y`;
use `python -m waffle_pi.calibration detector detections.csv calibration/detector.json`.
Report false positives separately; manually select sensor noise/dropout flags
from measured data. Do not confuse the example simulator values with measured ones.

Run held-out evaluation with a distinct seed, using the same profile and backend:

```sh
python train.py --waffle-pi --waffle-live-orca --seed 90210 --num-processes 32 --waffle-eval-episodes 100 --load-path trained_models/GST_predictor_rand/checkpoints/41665.pt --waffle-eval-json eval/live_zero_shot.json
```

For throughput, add `--benchmark-updates 5 --benchmark-warmup 2` to a training
command with a new output directory. It performs actual rollout, GST, GAE and
PPO updates and saves `benchmark.json`, separating rollout, bootstrap/GAE and
learner timings. This overrides the normal transition budget. It excludes log
and checkpoint time, includes explicit phase synchronization, and excludes
warmup updates from reported rates. Compare multiple environment/minibatch
counts with the same seed, simulator and training configuration. Keep a held-out
success/collision-rate target; maximizing steps/sec alone is insufficient.

## ROS 2 integration details

Install SciPy in the external host Python environment for the gated Kalman/Hungarian
tracker and map filter. Launch the stock TurtleBot3 bringup and map localization
for your installed ROS distribution; do not start a second navigation controller
publishing `/cmd_vel`. The host now accepts map- or odom-frame goal messages and
uses timestamped TF2 for detection coordinates. GST histories are resampled at
0.25 s only between valid observations; slot reuse clears prior identity history.
Overflow, stale observations and inference deadlines produce zero requests.

Optional static background filtering (raw `/scan` stays untouched):

```sh
python -m waffle_pi.map_filter --threshold 0.10
```

Configure a separately installed ROS 2 DR-SPAAM detector to subscribe to
`/crowdnav/scan_filtered` (or `/scan` for a baseline) and publish timestamped
`geometry_msgs/PoseArray` person centers to the topic passed as `--detection-topic`.
Inspect the actual detector message type and remap/bridge if different. The filter
uses occupancy-grid origin/orientation/resolution, scan angular metadata and TF2.
It substitutes +inf for background ranges; verify detector invalid-return handling
and false-negative rates near walls. Scan motion distortion is not deskewed.

For initial data collection, use ROS bag recording of `/scan`, `/odom`, `/imu`,
`/joint_states`, `/tf`, `/tf_static`, `/map`, detector output, shadow commands and
the actual command topic. Export timestamp-aligned CSVs for offline calibration.
Do not generate motor commands from the calibration script. Verify the LDS model,
scan frequency and message types on your installation.

The gate now also requires fresh odometry, adequate 360-degree scan coverage and
uses measured speed in stopping clearance. Dry-run shows the gated diagnostic
command and never publishes `/cmd_vel`. Arming requires an independently measured
`waffle_safety_commissioning_v1` JSON with `footprint`, `sensor_offset`, `margin`,
`latency`, `braking`, `supervised_validated` and `firmware_watchdog_tested` fields.
No such record is generated or shipped. The latter flags are acknowledgements of
actual tests, not evidence that the software itself is certified. Conservative
minimum latency is 0.5 s. No human trials until actual braking, sensor blind spots,
command-source exclusivity, Pi/process/network failure and OpenCR watchdog have
been tested in a controlled empty area, with an independent physical stop procedure.
ROS 1, exact LDS/DR-SPAAM installation, actual AMCL mapping, calibration and
physical testing remain dependent on the robot's configuration and measurements.

For a safe ROS shadow-only end-to-end gate check, remap the gate's input to
shadow commands, without `--arm`:

```sh
python -m waffle_pi.ros2_safety_gate --ros-args -r /crowdnav/cmd_vel_requested:=/crowdnav/cmd_vel_shadow
```

Use `bash scripts/validate_waffle.sh` to syntax-check the checked-out source
and run the unit tests. This is not a ROS integration or GPU training test.

## Explicit CUDA Graph inference

`--cuda-graph-actor --cuda-graph-gst` optionally captures fixed-shape actor
rollout inference and GST inference separately using PyTorch CUDA Graphs.
This is distinct from compiler-managed graph capture. Each graph uses
preallocated input buffers and reused outputs; actor weights are updated
in place by the ordinary PPO optimizer. Autocast weight caching is disabled
inside capture to avoid stale weights. The two options are initially single-GPU
only; do not combine each with its corresponding compile flag. Capture failures
are reported rather than silently falling back. The graph flags are experimental
and have not been run on CUDA hardware here. Optimizer-update graph capture is
not implemented. The GST navigation wrapper now suppresses unused attention
visualization CPU transfers, and GST edge-selector masks and temporary tensors
stay on device. Standalone GST training/visualization retains collect_info=True
by default. Equality-based attention projection branching is replaced by
object-identity branching to avoid GPU synchronization; mathematically equivalent
projection paths can differ slightly numerically.

## Modern isolated tensor-training environment

The tensor path no longer imports native RVO2, legacy Gym, OpenAI Baselines,
TensorFlow, or ROS. Use a separate Python 3.10/3.11 environment, install a
compatible PyTorch 2.x wheel from pytorch.org (CPU or your CUDA version),
then `python -m pip install -r requirements-waffle.txt`. The CPU reference
simulator and native trajectory generation still require the original stack.
Gymnasium is used only for space metadata; this is not a change to the
project's internal four-return vectorized step API.

Run from the repository root:

```sh
python -m scripts.smoke_waffle --device cpu --backend both --updates 2
```

This uses the released GST and randomized navigation checkpoints, checks finite
observations and losses, executes real recurrent PPO updates, checks actor
weights changed, and checks checkpoint round-trip. Replay uses a known fixture,
not a generated RVO2 pool. `--device cuda:0 --gae triton` tests the CUDA GAE
path separately. No physical robot commands are sent. The GST config loader
accepts the published argparse.Namespace format, and modern PyTorch loads
weights in restricted mode. Keep legacy pickle/checkpoints from trusted sources.

## DR-SPAAM ROS 2 detector adapter

The new `waffle_pi.ros2_detector` uses the upstream
[2D lidar person detector](https://github.com/RWTHVision/2D_lidar_person_detection)
Python `Detector` API (inspected tree `d12e90225d52ba6c85d20061f1e7286d6a64aade`).
Install upstream `dr_spaam` separately using its instructions and a trusted,
compatible checkpoint (CPU-readable if using CPU). The adapter deliberately
uses the newer API (`model`, `panoramic_scan`, `set_laser_fov`), not the older
DR-SPAAM-Detector API. No external source code or weights were vendored.

On the ROS 2 host, after starting the stock bringup and verifying `/scan`:

```sh
python -m waffle_pi.ros2_detector --weights /path/to/compatible_detector.pth --device cpu
```

It publishes `/dr_spaam_detections` as timestamped PoseArray. It validates scan
geometry, removes a duplicated panorama endpoint, reverses positive-increment
ROS scans for the DROW rightward-angle convention, and reflects/rotates centers
back into the actual LaserScan frame. Invalid returns use the upstream far-range
padding value. Failed/stale detection callbacks publish no false empty crowd.
The detector's recurrent memory resets after a long scan gap. Initial loading
or over-deadline inference is rejected; the policy stops on stale detections.
Use `--input-topic /crowdnav/scan_filtered` only after measuring its effect on
recall. All-invalid filtered scans are treated as unavailable, not empty.

Verify detector output on recorded scans with known person positions and both
sides of the robot. No ROS runtime, upstream detector weights or real LDS data
were available for validation. `--confidence`, `--stride` and range need actual
calibration; the default confidence is illustrative.

Optional `--gst-adjacent-mask` (in both training/evaluation and ROS policy)
fixes a released-wrapper inconsistency: a displacement is valid only when
both consecutive positions are visible, as in GST's training dataset code.
The default retains the released checkpoint's wrapper semantics for controlled
zero-shot comparisons. Evaluate both before choosing; use matching flags in
training and deployment, particularly for occlusion/dropout experiments.

## Captured PPO learner and benchmark sweep

`--cuda-graph-ppo` adds explicit single-GPU capture of the fixed-shape PPO
minibatch forward, backward, gradient clipping and capturable Adam step.
It enables masked GRU automatically, requires a fresh optimizer and fixed
learning rate, and is incompatible with policy compilation or DDP. Warmup
updates are undone in-place (parameters, Adam moments and counters), so only
real minibatches contribute learning updates. Capture failures abort explicitly.
The CUDA parity test checks parameter and Adam state evolution against eager
updates. Neither that test nor actual GPU graph capture has run here.

To compare supported configurations on the actual training workstation:

```sh
python -m scripts.benchmark_sweep --output benchmarks/waffle_gpu --device cuda --backend live --envs 64 128 256 512 --minibatches 4 16 --presets eager scan triton compile graph graph-bf16 --updates 5 --warmup 3
```

Start with `eager scan`; other presets require compatible CUDA/Triton/PyTorch
versions and sufficient VRAM. The sweep executes isolated trainer subprocesses
sequentially, records failures and timeout logs, and ranks measured successful
runs by end-to-end transition throughput. It saves raw JSON, CSV rankings and
individual checkpoints. Peak CUDA memory includes warmup/compilation. Compiler
setup costs are included in per-run wall time, excluded from steady-state
phase timings. Use `--repeat 3` to measure variability. `--device cpu` supports
eager and scan presets. Select by held-out success/collision rates and time to
reward after throughput tuning; different batch sizes change PPO learning.


## Native Waffle policy (primary retraining route)

The original Cartesian adapter is now a baseline only. Select
`--waffle-native-action --waffle-live-orca` to train a forward-speed/yaw-rate
policy under the physical commissioning gate's 0.10 m/s and 0.50 rad/s caps.
The two Gaussian latent coordinates map to v=vmax*max(0,tanh(z0)) and
w=wmax*tanh(z1). Store the latent sample and its original log probability
in PPO, not the transformed or gated command. The model retains all shapes.
`--waffle-init-cartesian --resume` loads the original randomized checkpoint and
resets its action mean and log standard deviation; GST is frozen, all policy
feature layers train unless --freeze-human-attention is explicitly used.

```sh
python train.py --waffle-pi --waffle-native-action --waffle-init-cartesian --waffle-live-orca --resume --fast-training --masked-gru --gae-backend scan --load-path trained_models/GST_predictor_rand/checkpoints/41665.pt --output_dir trained_models/native_waffle --num-processes 256 --num-mini-batch 16 --num-env-steps 1000000
```

The native backend models all-around stopping clearance from ideal disk and
square-wall ranges before the motor transition, with measured-speed clearance
and command delay. `--waffle-safety-calibration` accepts the same parameter
schema as the physical gate. Defaults are unmeasured conservative examples.
Actual scan coverage, stale messages, moving human stopping guarantees, and
arbitrary laboratory maps are not modeled; the real gate remains unchanged.
Use `--waffle-calibration` for measured actuator parameters separately.
For native evaluation use --waffle-native-action, the trained checkpoint,
and matching profile flags, WITHOUT --waffle-init-cartesian. The checkpoint's
adjacent run_args.json is required to prevent silent action reinterpretation.
For ROS shadow inference pass --waffle-native-action and matching --speed,
--angular-speed, --radius, --sensor-range, --humans and GST mask settings.
Keep the run_args.json with checkpoints. Do not arm the robot based on
simulation results. A native-policy smoke test is added to CI.


## Concurrent independent PPO portfolio

Use a single batched learner on one GPU initially. For several GPUs, run
independent candidates (not shared gradients), each with its own frozen GST,
native simulator, PPO optimizer and seed. The scheduler runs at most one
candidate per listed GPU, starts another as soon as a slot frees up, then
evaluates every successful candidate on a common held-out seed.

```sh
python -m scripts.train_portfolio --output experiments/native_portfolio --gpus 0 1 --envs 256 --minibatches 16 --seeds 11 29 47 --lrs 0.00004 0.0001 --entropies 0 0.01 --transitions 1000000 --eval-episodes 100
```

On CPU use --cpu-workers 2 --envs 16 --minibatches 4; no CUDA needed.
Set the actual number of GPUs explicitly; this command assumes two GPUs.
The portfolio saves per-candidate training and evaluation logs, checkpoints,
results.json, and ranking.json. It rejects candidates exceeding the selected
simulated collision fraction before ranking by success and other metrics.
These are independent full-budget trials, not population-based training,
early stopping, or successive halving. A common selection seed is not a
final untouched test set; evaluate the chosen policy with further independent
seeds and the calibrated physical profile. No policy is automatically deployed.
