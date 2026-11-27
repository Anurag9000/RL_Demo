#!/usr/bin/env bash
set -euo pipefail
RUN_ROOT=/home/rl-group4/waffle_runs/RL_Demo-20261126T161914Z
REPO=/home/rl-group4/Projects/Untapped/RL_Demo
VENV=/home/rl-group4/.venvs/fedqcl-sync-py312
if [[ $# -ne 7 ]]; then
  echo "usage: $0 CHECKPOINT OUTPUT_DIR LOG_DIR EVAL_SEED EPISODES PROCESSES PHYSICAL_GPU" >&2
  exit 64
fi
CHECKPOINT=$(realpath "$1")
OUT=$(realpath -m "$2")
LOG_DIR=$(realpath -m "$3")
SEED=$4
EPISODES=$5
PROCESSES=$6
GPU=$7
TAG=$(basename "$OUT")
if [[ ! "$CHECKPOINT" == "$RUN_ROOT"/* || ! "$OUT" == "$RUN_ROOT"/evaluations/* || ! "$LOG_DIR" == "$RUN_ROOT"/logs/* ]]; then
  echo "checkpoint, output and logs must remain under the mission run root" >&2
  exit 64
fi
if [[ ! -f "$CHECKPOINT" || -e "$OUT" || -e "$LOG_DIR" ]]; then
  echo "checkpoint missing or diagnostic output/log directory already exists" >&2
  exit 2
fi
if [[ ! "$SEED" =~ ^[0-9]+$ || ! "$EPISODES" =~ ^[0-9]+$ || ! "$PROCESSES" =~ ^[0-9]+$ || ! "$GPU" =~ ^[0-9]+$ || "$PROCESSES" -lt 1 || "$EPISODES" -lt "$PROCESSES" ]]; then
  echo "seed, episode, process and GPU values must be valid; episodes must cover all lanes" >&2
  exit 64
fi
LOCK="$RUN_ROOT/mission/$TAG.lock"
PID_FILE="$RUN_ROOT/mission/$TAG.pid"
EXIT_FILE="$RUN_ROOT/mission/$TAG.exit.json"
if [[ -e "$LOCK" || -e "$PID_FILE" || -e "$EXIT_FILE" ]]; then
  echo "refusing a reused or active diagnostic tag: $TAG" >&2
  exit 2
fi
mkdir "$LOCK"
STARTED=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf '%s\n' "$$" > "$PID_FILE"
finish() {
  rc=$?
  finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  tmp="$EXIT_FILE.tmp"
  printf '{"tag":"%s","pid":%s,"started_utc":"%s","finished_utc":"%s","exit_status":%s}\n' "$TAG" "$$" "$STARTED" "$finished" "$rc" > "$tmp"
  mv "$tmp" "$EXIT_FILE"
  rmdir "$LOCK"
}
trap finish EXIT
mkdir -p "$(dirname "$LOG_DIR")"
source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES="$GPU" CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="$RUN_ROOT/python-deps-np126:$REPO:$RUN_ROOT/python-deps:$RUN_ROOT/upstream/2D_lidar_person_detection/dr_spaam"
export WAFFLE_GAZEBO_PODMAN_IMAGE=localhost/waffle-gazebo-harmonic:mission-20261126
export WAFFLE_GAZEBO_RESOURCE_PATH="$RUN_ROOT/upstream/turtlebot3_simulations/turtlebot3_gazebo/models"
export WAFFLE_GAZEBO_CONTAINER_WORKER=/usr/local/bin/waffle_gazebo_worker
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$REPO"
python "$REPO/mission/records/RL_Demo-20261126T161914Z/scripts/diagnose_v5_action_gate.py" \
  --run-root "$RUN_ROOT" --repo-root "$REPO" --checkpoint "$CHECKPOINT" \
  --output-dir "$OUT" --logs-dir "$LOG_DIR" --seed "$SEED" \
  --episodes "$EPISODES" --processes "$PROCESSES" --gpu "$GPU"
