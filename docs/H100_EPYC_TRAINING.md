# Hardware-aware Gazebo PPO on dual EPYC 9354 and H100 NVL

## Scope and evidence

The training path is **Gazebo Harmonic only**, using the official ROBOTIS
TurtleBot3 Waffle Pi model, its simulated wheel joints and LDS frames, actual
DR-SPAAM, tracker, frozen GST, original attention/GRU architecture and native
PPO. No tensor physics replacement is selected by the hardware sweep. The
previous tensor paths remain explicitly selectable historical baselines.
No speedup, convergence, physical calibration or maximum stable world count
is claimed before actual server testing.

The target server has 2 x 32 physical Zen 4 cores (SMT off), 2 NUMA nodes,
503 GiB RAM (477 GiB available at the supplied snapshot), and two 95,830 MiB
H100 NVL GPUs. GPU 0 and GPU 1 are socket-local; the observed SYS path is
not evidence of an active NVLink bridge. GPU 1 was busy at the snapshot.
Use only resources assigned to this job; do not terminate other workloads,
change clocks, firmware, governor, SMT, MIG, MPS or machine settings without
administrator approval. The reported driver CUDA version is not a toolkit
installation. Qualify actual H100 EGL/OGRE2 rendering separately.

## Build and hardware qualification

Follow GAZEBO_TRAINING.md for official model, dependencies and measured
hardware profile. A new native binary changes its hash: qualify it and
review/update the profile's worker_sha256 before measured continuation.
Zen 4 and LTO are opt-in (compiler must support them); no fast-math.

~~~sh
cmake -S gazebo_sim -B build/gazebo -DCMAKE_BUILD_TYPE=Release \
  -DWAFFLE_ZEN4=ON -DWAFFLE_LTO=ON
cmake --build build/gazebo -j 64
nvidia-smi topo -m
nvidia-smi nvlink --status
lscpu -e=CPU,CORE,SOCKET,NODE
numactl --hardware
df -h /dev/shm
quota -s
WAFFLE_SCAN_TRANSPORT=json python -m scripts.qualify_gazebo \
  --model "$WAFFLE_MODEL" --envs 2 --output gazebo_runs/json_reference
WAFFLE_SCAN_TRANSPORT=shm python -m scripts.qualify_gazebo \
  --model "$WAFFLE_MODEL" --envs 2 --output gazebo_runs/shm_reference
~~~

## Implemented optimizations in this round

- Native anonymous memfd/mmap scan transfer (WAFFLE_SCAN_TRANSPORT=shm):
  fixed bounded capacity 256 x 4096 float32 values = 4 MiB mapped per
  worker, physically allocated on touch. Metadata and commands still
  use JSON; sequence checks and one transaction in flight prevent
  overwrites. The consumer copies the scan before another step. Same
  float32/nonfinite conversion as the existing JSON path. This is not
  zero-copy GPU transfer or a double-buffered ring. JSON remains default.
- Completion-driven episode reset scheduling: terminal workers restart
  concurrently with slower worlds' remaining steps, with independent
  per-world scenario RNG. One batch barrier remains per policy step.
  There is no stale-policy collection. Reset still restarts Gazebo;
  do not reuse stale physics/renderer/controller state to save time.
- Linux process placement: explicit rank cpusets and worker groups from
  sysfs NUMA topology intersect inherited job CPU affinity. Worker
  taskset wraps native server launch; preferred memory via numactl is
  opt-in. No global settings modified. CPU group sizes and oversubscription
  are benchmark variables; they are not hard resource reservations.
- Shared DR-SPAAM microbatching: one upstream neural model per local
  learner, separate temporal memory per world, reset and fresh-frame
  grouping, unchanged upstream cutouts and NMS. Batch detector remains
  experimental pending real-checkpoint parity. DROW3 also supported.
  No detector weights or perception sequence semantics are changed.
- Pinned observation staging, nonblocking copies, event-protected reuse;
  separate GPU allocations remain, not GPU zero-copy.
- Rank-local detector placement (learner), optional DDP bucket views
  and bucket size, explicit TF32 backend flags, measured FP32 reference.
  Existing graph/compile/BF16/fused Adam/GAE options remain subject to
  their compatibility checks; no FP8 or TensorRT implementation added.
- Optional global approximate-KL epoch termination in PPO, with
  synchronized decision across DDP ranks; disabled by default.
  Existing adaptive learning rate, curriculum, population-based
  training, successive halving and full optimizer continuation retained.
- Confidence selection defaults to 128 validation episodes, rejects
  impossible Wilson thresholds. 64 zero-collision episodes cannot
  meet a 5% upper 95% Wilson bound. Repeated correlated scenarios do
  not provide independent statistical evidence.
- Native worker/world timing, model parameter and gradient byte counts,
  GPU memory peaks, CPU affinity, detector batch statistics and
  fixed-global-batch hardware comparison.

## Measure single GPU vs two GPUs instead of guessing

~~~sh
python -m scripts.hardware_sweep --model "$WAFFLE_MODEL" \
  --profile "$PROFILE" --detector "$DETECTOR" \
  --gpus 0 1 --worlds 8 16 32 48 64 --minibatches 4 \
  --presets eager scan --output benchmarks/h100_plan
~~~

This read-only plan queries actual GPU UUIDs, PCI/NUMA topology, memory and
allowed CPU set, saves plan.json, and does not launch training. Inspect
GPU utilization and ownership before executing. It assumes two GPU indices
from nvidia-smi, resolves UUIDs, and checks both NUMA nodes are accessible.
If a node is not allocated, use a single-GPU benchmark_sweep instead.

Run the real comparison with a fresh output directory and --execute:

~~~sh
python -m scripts.hardware_sweep --model "$WAFFLE_MODEL" \
  --profile "$PROFILE" --detector "$DETECTOR" \
  --gpus 0 1 --worlds 8 16 32 48 64 --minibatches 4 \
  --presets eager scan --transport shm --pinned \
  --output benchmarks/h100_reference --execute
~~~

For an illustrative profile, compare CPU and GPU detector placement via
--detector-device cpu/cuda, then --batch-detector, detector batch sizes
4/8/16/32, wait budgets 0/.5/1/2 ms, worker core widths 1/2/4, and
--numa-memory where permitted. Try --transport json as a controlled
baseline. Measured mode uses --require-measured --safety PATH with
unbatched detector and eager/scan only. Keep the same actual simulator
profile, physics/sensor frequency, global worlds, rollout length, global
minibatches, epochs, seed and training precision for a matched comparison.
Use additional seeds and longer runs for learning quality, not just FPS.

The sweep compares:
1. One H100 with all allocated CPU cores, including the other NUMA node.
2. One H100 with its local CPU node.
3. The second H100 with its local CPU node.
4. Two H100 DDP ranks, each with half the global worlds and local cores.
5. Two independent concurrent policies, each with half the global worlds.

For DDP the global batch and minibatch sizes equal the single policy's
global sizes. Independent learners have smaller batches, not one global
policy. Ranking separates per-policy and aggregate throughput; it also
reports wall time including startup and logging and estimates 20M transition
collection times, explicitly not convergence guarantees. Use repeated
measurements. Optional graph/compile presets are skipped for DDP due to
current restrictions; do not compare incompatible precision as if identical.

Measure NCCL independently:

~~~sh
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 -m scripts.nccl_probe \
  --bytes 4096 1048576 4194304 16777216 67108864
~~~

Prefer actual device UUIDs from plan.json to avoid index ambiguity.
Repeat with WAFFLE_RANK_CPUSETS from the plan. Probe the measured gradient
byte size from benchmark.json (rounded to a multiple of 4). This measures
all-reduce latency, not full DDP overhead or renderer contention.

Tradeoff model: let B be global transitions/update, S1 and S2 measured
single and DDP transitions/s. T1=B/S1, T2=B/S2. DDP speedup=S2/S1; DDP
wins only when T2<T1 for the same learning configuration and target quality.
For 2 ranks, ring equivalent wire payload per rank is approximately one
gradient tensor per all-reduce. E epochs x M minibatches => E*M gradient
synchronizations, plus smaller reductions. At 16 MiB gradient, 10 GB/s
effective bandwidth, 80 synchronizations, serialization-only cost is
~0.134 s/update, not including latency or overlap. Use the measured latency
at the actual gradient size, not the nominal PCIe or NVLink specification.
If DDP scaling is poor, use one GPU plus both CPU sockets for a single
policy, or both GPUs independently for concurrent candidates/validation.
Fitting model state on one GPU is necessary but not sufficient for fastest
training; rendering and experience throughput may dominate.

## Perception parity before enabling batching

Collect chronological actual Gazebo/physical scans in an NPZ with ranges
[streams,ticks,rays] and optional reset [streams,ticks] boolean. Include
distinct pedestrian motion and independent resets, occlusions, range errors,
empty and invalid scans. The script uses actual trusted weights and upstream
model; fixture tests alone are insufficient.

~~~sh
python -m scripts.check_detector_parity --scans calibration/scans.npz \
  --profile "$PROFILE" --weights "$DETECTOR" --device cuda:0 \
  --output benchmarks/detector_parity.json
~~~

Check matched centers, classification threshold/NMS changes, memory reset,
fault parity and downstream policy quality. No automated promotion based
only on a successful detection comparison. The GPU renderer can compete
with neural compute; verify real renderer selection separately.

## Run adaptive retraining with selected allocation

For two independent GPU candidates, use the actual UUID/cpuset mappings
from plan.json and set the matching resource controls:

~~~sh
export WAFFLE_CANDIDATE_CPUSETS="$(python -c 'import json; print(json.dumps(json.load(open("benchmarks/h100_reference/plan.json"))["candidate_cpusets"]))')"
export WAFFLE_CANDIDATE_GPU_UUIDS="$(python -c 'import json; print(json.dumps(json.load(open("benchmarks/h100_reference/plan.json"))["candidate_gpu_uuids"]))')"
export WAFFLE_WORKER_AFFINITY=1 WAFFLE_WORKER_CORES=2 WAFFLE_RESERVE_CORES=2
export WAFFLE_SCAN_TRANSPORT=shm WAFFLE_PINNED_OBS=1
python -m scripts.train_adaptive --backend gazebo --gazebo-model "$WAFFLE_MODEL" \
  --fidelity-profile "$PROFILE" --detector-weights "$DETECTOR" \
  --gazebo-detector-device learner --safety-calibration calibration/safety.json \
  --require-measured --gpus 0 1 --envs 16 --minibatches 4 --steps 60 \
  --eval-episodes 128 --test-episodes 200 --mode hybrid \
  --output experiments/h100_gazebo_adaptive
~~~

16 is an initial per-GPU world count, not an optimum. Substitute the
qualified result. A single GPU candidate can use all allocated cores by
omitting WAFFLE_CANDIDATE_CPUSETS or specifying the full allowed cpuset
for that slot. For direct DDP use torchrun with WAFFLE_RANK_CPUSETS and
normal Gazebo flags, and --ddp-bucket-view/--ddp-bucket-mb if qualified.
Do not change optimizer semantics mid-continuation without explicit
configuration and new rollouts. For KL epoch termination try --ppo-stop-kl
0.02 only as a quality-controlled experiment, not a universally optimal
setting. Use longer rollouts, discount and GAE tuning with held-out
success, collisions, interventions, clearance and time-to-target.

## Remaining engineering/qualification

Full in-place native reset, full completion-driven rollout collection,
multiworld native C++ scheduler, native shared-memory actions/rewards,
double-buffered GPU transfers, native/batched Kalman tracking, actual
Hopper FP8/TensorRT, custom attention kernels, CUDA Graph + DDP/Adam
continuation integration, PGO, adaptive hyperparameter search beyond the
existing scheduler, and remote resource provisioning remain work items.
No claim that maximizing allocated memory or utilization improves quality.
Keep memory headroom for rendering/temporary allocations; increasing
worlds past saturation can reduce throughput. The server's rotational
disk and user quota matter: use bounded scratch/logs, preserve recovery
checkpoints and do not fill RAM or VRAM merely to achieve high utilization.
Actual OpenCR/DYNAMIXEL/LDS firmware and hardware behavior, physical
calibration, realistic humans, localization and full real-world validation
remain as documented in GAZEBO_TRAINING.md.
