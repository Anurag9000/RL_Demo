#!/usr/bin/env bash
set -euo pipefail
usage() { echo "usage: $0 <randomized|nonrandomized> <gpu> <seed> <worlds> <mini-batches> <transitions> <run-id>" >&2; exit 2; }
[[ $# -eq 7 ]] || usage
REGIME=$1 GPU=$2 SEED=$3 WORLDS=$4 MINI=$5 STEPS=$6 RUN_ID=$7
[[ "$REGIME" == randomized || "$REGIME" == nonrandomized ]] || usage
[[ "$GPU" =~ ^[0-9]+$ && "$SEED" =~ ^[0-9]+$ && "$WORLDS" =~ ^[1-9][0-9]*$ && "$MINI" =~ ^[1-9][0-9]*$ && "$STEPS" =~ ^[1-9][0-9]*$ && "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || usage
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=${WAFFLE_REPO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
: "${WAFFLE_RUN_ROOT:?Set WAFFLE_RUN_ROOT}"
: "${WAFFLE_VENV:?Set WAFFLE_VENV}"
ROOT=$WAFFLE_RUN_ROOT
PROFILE=$REPO/mission/profiles/waffle_gazebo_paper_${REGIME}_500x3.json
MODEL=${WAFFLE_MODEL:-$ROOT/upstream/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_waffle_pi/model.sdf}
WORKER=${WAFFLE_WORKER:-$ROOT/build/waffle_gazebo_worker}
DETECTOR=${WAFFLE_DETECTOR:-$ROOT/models/dr_spaam/ckpt_jrdb_ann_dr_spaam_e20.pth}
ACTOR=${WAFFLE_ACTOR:-$REPO/trained_models/GST_predictor_rand/checkpoints/41665.pt}
OUT=$ROOT/experiments/$RUN_ID
LOGROOT=$ROOT/logs/${RUN_ID}-workers
LOCK=$ROOT/mission/${RUN_ID}.lock
STATUS=$ROOT/mission/${RUN_ID}.exit.json
LAUNCH=$ROOT/mission/${RUN_ID}.launch.json
for path in "$PROFILE" "$MODEL" "$WORKER" "$DETECTOR" "$ACTOR"; do
  [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 66; }
done
[[ ! -e "$OUT" && ! -e "$STATUS" && ! -e "$LAUNCH" ]] || { echo "run id already exists: $RUN_ID" >&2; exit 76; }
mkdir -p "$ROOT/experiments" "$ROOT/logs" "$ROOT/mission"
exec 9>"$LOCK"
flock -n 9 || { echo "duplicate run lock: $RUN_ID" >&2; exit 75; }
source "$WAFFLE_VENV/bin/activate"
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH="$ROOT/python-deps-np126:$REPO:$ROOT/python-deps:$ROOT/upstream/2D_lidar_person_detection/dr_spaam"
export WAFFLE_GAZEBO_PODMAN_IMAGE=${WAFFLE_GAZEBO_PODMAN_IMAGE:-localhost/waffle-gazebo-harmonic:mission-20261126}
export WAFFLE_GAZEBO_RESOURCE_PATH=${WAFFLE_GAZEBO_RESOURCE_PATH:-$ROOT/upstream/turtlebot3_simulations/turtlebot3_gazebo/models}
export WAFFLE_GAZEBO_CONTAINER_WORKER=${WAFFLE_GAZEBO_CONTAINER_WORKER:-/usr/local/bin/waffle_gazebo_worker}
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
started=$(date -u +%FT%TZ)
printf '%s\n' "$$" > "$ROOT/mission/$RUN_ID.pid"
python - "$LAUNCH" "$RUN_ID" "$REGIME" "$GPU" "$SEED" "$WORLDS" "$MINI" "$STEPS" "$PROFILE" "$ACTOR" "$started" <<'PY'
import json,sys
p,run,regime,gpu,seed,worlds,mini,steps,profile,actor,start=sys.argv[1:]
x=dict(run_id=run,regime=regime,gpu=int(gpu),seed=int(seed),worlds=int(worlds),mini_batches=int(mini),requested_transitions=int(steps),profile=profile,split="train",actor=actor,started_utc=start,initialization="weights-only resume; native action head reset; fresh PPO optimizer")
with open(p,"x") as f: json.dump(x,f,indent=2); f.write("\n")
PY
train_pid=
finish() { rc=$?; tmp="$STATUS.tmp.$$"; printf '{"run_id":"%s","exit_code":%d,"finished_utc":"%s"}\n' "$RUN_ID" "$rc" "$(date -u +%FT%TZ)" > "$tmp"; mv -f "$tmp" "$STATUS"; }
stop_child() { if [[ -n "$train_pid" ]] && kill -0 "$train_pid" 2>/dev/null; then kill -INT "$train_pid" 2>/dev/null || true; wait "$train_pid" || true; fi; exit 143; }
trap finish EXIT
trap stop_child TERM HUP
trap stop_child INT
( trap - INT; exec python -u "$REPO/train.py" \
  --output_dir "$OUT" --waffle-pi --waffle-gazebo --resume --load-path "$ACTOR" \
  --waffle-init-cartesian --waffle-native-action \
  --waffle-fidelity-profile "$PROFILE" --waffle-gazebo-model "$MODEL" \
  --waffle-gazebo-worker "$WORKER" --waffle-gazebo-logs "$LOGROOT" \
  --waffle-gazebo-detector-device learner --waffle-detector-weights "$DETECTOR" \
  --waffle-gazebo-timeout 120 --waffle-humans 4 --waffle-scene-split train \
  --num-processes "$WORLDS" --num-steps 30 --seq_length 30 \
  --num-mini-batch "$MINI" --ppo-epoch 5 --num-env-steps "$STEPS" \
  --save-interval 10 --log-interval 2 --num_threads 1 --seed "$SEED" \
  --save-training-state ) &
train_pid=$!
if wait "$train_pid"; then rc=0; else rc=$?; fi
train_pid=
exit "$rc"
