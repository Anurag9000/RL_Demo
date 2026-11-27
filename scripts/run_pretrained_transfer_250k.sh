#!/usr/bin/env bash
set -Eeuo pipefail
: "${WAFFLE_RUN_ROOT:?set WAFFLE_RUN_ROOT}"
: "${WAFFLE_REPO_ROOT:?set WAFFLE_REPO_ROOT}"
: "${WAFFLE_VENV:?set WAFFLE_VENV}"
RUN_ID=waffle_pretrained_transfer_seed438_rand41665_v5_250k_20261127
CHECKPOINT="$WAFFLE_REPO_ROOT/trained_models/GST_predictor_rand/checkpoints/41665.pt"
EXPECTED_CHECKPOINT_SHA=a60bdd3b8ca84457497365133b9055c80d55b11c5a3bc76b28e38d4200cfd8ce
OUTPUT_DIR="$WAFFLE_RUN_ROOT/experiments/$RUN_ID"
LOG_DIR="$WAFFLE_RUN_ROOT/logs"
LOG="$LOG_DIR/$RUN_ID.log"
EXIT_STATUS="$LOG_DIR/$RUN_ID.exit"
PROFILE="$WAFFLE_RUN_ROOT/mission/profiles/waffle_gazebo_illustrative_safe_clearance_diverse_v5_500x3.json"
MODEL="$WAFFLE_RUN_ROOT/upstream/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_waffle_pi/model.sdf"
WORKER="$WAFFLE_RUN_ROOT/build/waffle_gazebo_worker"
DETECTOR="$WAFFLE_RUN_ROOT/models/dr_spaam/ckpt_jrdb_ann_dr_spaam_e20.pth"
mkdir -p "$LOG_DIR"
exec 9>"$LOG_DIR/$RUN_ID.lock"
flock -n 9 || { echo 'duplicate launch refused' >&2; exit 73; }
if [[ -e "$OUTPUT_DIR" || -e "$LOG" || -e "$EXIT_STATUS" ]]; then echo "artifacts already exist: $RUN_ID" >&2; exit 74; fi
for f in "$CHECKPOINT" "$PROFILE" "$MODEL" "$WORKER" "$DETECTOR" "$WAFFLE_RUN_ROOT/mission/scripts/train_vram_bounded_20pct.py"; do [[ -r "$f" ]] || { echo "required input missing: $f" >&2; exit 66; }; done
actual_sha=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
[[ "$actual_sha" == "$EXPECTED_CHECKPOINT_SHA" ]] || { echo 'checkpoint SHA mismatch' >&2; exit 65; }
trap 'rc=$?; printf "%s\n" "$rc" > "$EXIT_STATUS.tmp"; mv "$EXIT_STATUS.tmp" "$EXIT_STATUS"' EXIT
source "$WAFFLE_VENV/bin/activate"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$WAFFLE_RUN_ROOT/python-deps-np126:$WAFFLE_REPO_ROOT:$WAFFLE_RUN_ROOT/python-deps:$WAFFLE_RUN_ROOT/upstream/2D_lidar_person_detection/dr_spaam"
export WAFFLE_GAZEBO_PODMAN_IMAGE=localhost/waffle-gazebo-harmonic:mission-20261126
export WAFFLE_GAZEBO_RESOURCE_PATH="$WAFFLE_RUN_ROOT/upstream/turtlebot3_simulations/turtlebot3_gazebo/models"
export WAFFLE_GAZEBO_CONTAINER_WORKER=/usr/local/bin/waffle_gazebo_worker
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$WAFFLE_REPO_ROOT"
{
 printf 'run_id=%s\ncheckpoint_sha256=%s\ngit_sha=%s\nstarted_utc=%s\n' "$RUN_ID" "$actual_sha" "$(git rev-parse HEAD)" "$(date -u +%FT%TZ)"
 python -u "$WAFFLE_RUN_ROOT/mission/scripts/train_vram_bounded_20pct.py" \
  --output_dir "$OUTPUT_DIR" --waffle-pi --waffle-native-action --waffle-gazebo \
  --resume --load-path "$CHECKPOINT" --waffle-init-cartesian \
  --waffle-fidelity-profile "$PROFILE" --waffle-gazebo-model "$MODEL" \
  --waffle-gazebo-worker "$WORKER" --waffle-gazebo-logs "$LOG_DIR/${RUN_ID}-workers" \
  --waffle-gazebo-detector-device learner --waffle-detector-weights "$DETECTOR" \
  --waffle-gazebo-timeout 120 --waffle-humans 4 --waffle-scene-split train \
  --num-processes 32 --num-steps 30 --seq_length 30 --num-mini-batch 8 \
  --ppo-epoch 5 --num-env-steps 250000 --save-interval 10 --log-interval 2 \
  --num_threads 1 --seed 438 --save-training-state
 printf 'finished_utc=%s\n' "$(date -u +%FT%TZ)"
} 2>&1 | tee "$LOG"
