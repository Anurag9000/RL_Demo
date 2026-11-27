#!/usr/bin/env bash
set -euo pipefail
RUN_ROOT=/home/rl-group4/waffle_runs/RL_Demo-20261126T161914Z
REPO=/home/rl-group4/Projects/Untapped/RL_Demo
if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 CHECKPOINT OUTPUT_DIR validation|test TRAIN_SEED [VALIDATION_LOCK_JSON]" >&2
  exit 64
fi
CHECKPOINT=$(realpath "$1")
OUT=$(realpath -m "$2")
SPLIT=$3
TRAIN_SEED=$4
LOCK=${5:-}
case "$SPLIT" in validation|test) ;; *) echo "split must be validation or test" >&2; exit 64;; esac
if [[ ! -f "$CHECKPOINT" || -e "$OUT" ]]; then echo "checkpoint missing or output already exists" >&2; exit 2; fi
TAG=$(basename "$OUT")
LOGROOT="$RUN_ROOT/logs/eval-$TAG"
REQUEST="$RUN_ROOT/mission/evaluation_requests/$TAG.json"
if [[ -e "$LOGROOT" || -e "$REQUEST" ]]; then echo "refusing reused log/request path" >&2; exit 2; fi
mkdir -p "$RUN_ROOT/mission/evaluation_requests"
python3 - "$CHECKPOINT" "$OUT" "$SPLIT" "$TRAIN_SEED" "$LOCK" "$REQUEST" "$RUN_ROOT" <<'PY'
import hashlib,json,sys
from pathlib import Path
checkpoint,out,split,seed,lock,request,root=sys.argv[1:]
checkpoint,out,root=Path(checkpoint),Path(out),Path(root)
if root not in checkpoint.parents or root not in out.parents:
    raise SystemExit('checkpoint and output must be under the mission run root')
manifest=checkpoint.parent.parent/'run_args.json'
if not manifest.is_file(): raise SystemExit('checkpoint has no run_args.json provenance')
a=json.loads(manifest.read_text())
if int(a.get('seed',-1))!=int(seed) or not a.get('waffle_pi') or not a.get('waffle_gazebo') or not a.get('waffle_native_action'):
    raise SystemExit('checkpoint run manifest does not match requested Waffle native Gazebo seed')
expected_profile=root/'mission/profiles/waffle_gazebo_illustrative_safe_clearance_diverse_v5_500x3.json'
profile=Path(a.get('waffle_fidelity_profile',''))
if not profile.is_file() or profile.resolve()!=expected_profile.resolve(): raise SystemExit('training did not use the canonical v5 bank')
profile_bytes=profile.read_bytes()
if hashlib.sha256(profile_bytes).hexdigest()!=a.get('waffle_fidelity_sha'): raise SystemExit('training profile hash differs from current bank')
profile_data=json.loads(profile_bytes)
if profile_data.get('status')!='illustrative': raise SystemExit('unexpected training profile status')
all_ids=[s.get('id') for s in profile_data['scene']['scenarios']]
if len(all_ids)!=1500 or len(set(all_ids))!=1500: raise SystemExit('v5 train/validation/test bank is not exactly 1500 unique IDs')
sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
cases=[x for x in profile_data['scene']['scenarios'] if len(x['humans'])==int(a['waffle_humans']) and x.get('split','train')==split]
if len(cases)!=500: raise SystemExit(f'expected exactly 500 {split} cases for this crowd size, found {len(cases)}')
if split=='test':
    if not lock: raise SystemExit('held-out test requires a validation lock JSON')
    choice=json.loads(Path(lock).read_text())
    if (choice.get('selection_status')!='locked' or choice.get('selected_by_validation') is not True
            or choice.get('validation_split')!='validation' or choice.get('validation_episodes')!=500
            or choice.get('training_seed')!=int(seed) or choice.get('test_split_used') is not False
            or not isinstance(choice.get('selection_criteria'),str) or not choice.get('selection_criteria').strip()):
        raise SystemExit('validation lock is incomplete or test split was already used')
    if str(Path(choice.get('checkpoint','')).resolve())!=str(checkpoint) or choice.get('checkpoint_sha256')!=sha:
        raise SystemExit('validation lock does not match this checkpoint and hash')
    if choice.get('profile_sha256')!=hashlib.sha256(profile_bytes).hexdigest():
        raise SystemExit('validation lock profile hash differs from v5 bank')
    metrics_path=Path(choice.get('validation_metrics_path',''))
    if not metrics_path.is_absolute() or not metrics_path.is_file():
        raise SystemExit('absolute validation evidence path referenced by checkpoint lock is missing')
Path(request).write_text(json.dumps(dict(checkpoint=str(checkpoint),checkpoint_sha256=sha,
    training_manifest=str(manifest),training_seed=int(seed),split=split,episode_count=500,
    profile=str(profile),profile_status=profile_data['status'],validation_lock=lock or None),indent=2)+'\n')
print('evaluation admission passed:',split,'episodes=500','checkpoint_sha256='+sha)
PY
source /home/rl-group4/.venvs/fedqcl-sync-py312/bin/activate
export CUDA_VISIBLE_DEVICES="${EVAL_PHYSICAL_GPU:-1}"
export PYTHONPATH="$RUN_ROOT/python-deps-np126:$REPO:$RUN_ROOT/python-deps:$RUN_ROOT/upstream/2D_lidar_person_detection/dr_spaam"
export WAFFLE_GAZEBO_PODMAN_IMAGE=localhost/waffle-gazebo-harmonic:mission-20261126
export WAFFLE_GAZEBO_RESOURCE_PATH="$RUN_ROOT/upstream/turtlebot3_simulations/turtlebot3_gazebo/models"
export WAFFLE_GAZEBO_CONTAINER_WORKER=/usr/local/bin/waffle_gazebo_worker
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$REPO"
python - "$CHECKPOINT" "$OUT" "$SPLIT" "$TRAIN_SEED" "$LOGROOT" "$RUN_ROOT" <<'PY'
import sys,torch
checkpoint,out,split,seed,logroot,root=sys.argv[1:]
torch.cuda.set_device(0);torch.cuda.init();torch.cuda.set_per_process_memory_fraction(.20,0)
import train
sys.argv=['train.py','--output_dir',out,'--waffle-pi','--waffle-native-action','--waffle-gazebo',
 '--waffle-fidelity-profile',root+'/mission/profiles/waffle_gazebo_illustrative_safe_clearance_diverse_v5_500x3.json',
 '--waffle-gazebo-model',root+'/upstream/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_waffle_pi/model.sdf',
 '--waffle-gazebo-worker',root+'/build/waffle_gazebo_worker','--waffle-gazebo-logs',logroot,
 '--waffle-gazebo-detector-device','learner','--waffle-detector-weights',root+'/models/dr_spaam/ckpt_jrdb_ann_dr_spaam_e20.pth',
 '--waffle-humans','4','--waffle-scene-split',split,'--waffle-eval-episodes','500',
 '--waffle-eval-json',out+'/metrics.json','--load-path',checkpoint,'--num-processes','8',
 '--num-mini-batch','8','--num-steps','30','--seq_length','30','--seed',str(900000+int(seed))]
train.main()
PY
