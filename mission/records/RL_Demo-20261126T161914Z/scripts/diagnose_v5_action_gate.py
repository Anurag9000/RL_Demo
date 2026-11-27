#!/usr/bin/env python3
"""Run a small real-Gazebo validation diagnostic and record policy/gate decisions."""
import argparse
import json
import os
from pathlib import Path
import sys

p = argparse.ArgumentParser()
p.add_argument('--run-root', required=True)
p.add_argument('--repo-root', required=True)
p.add_argument('--checkpoint', required=True)
p.add_argument('--output-dir', required=True)
p.add_argument('--logs-dir', required=True)
p.add_argument('--seed', type=int, required=True)
p.add_argument('--episodes', type=int, default=1)
p.add_argument('--processes', type=int, default=1)
p.add_argument('--gpu', type=int, default=0)
a = p.parse_args()
if a.processes < 1 or a.episodes < a.processes:
    raise SystemExit('episodes must be at least the process count')
root, repo, out, logs = map(Path, (a.run_root, a.repo_root, a.output_dir, a.logs_dir))
checkpoint = Path(a.checkpoint)
if not checkpoint.is_file() or not root.resolve() in checkpoint.resolve().parents:
    raise SystemExit('checkpoint must exist under the mission run root')
if out.exists() and any(out.iterdir()):
    raise SystemExit('diagnostic output must be absent or empty')
out.mkdir(parents=True, exist_ok=True)
os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import numpy as np
import torch
torch.cuda.set_device(0)
torch.cuda.init()
torch.cuda.set_per_process_memory_fraction(.20, 0)
from gazebo_sim.vector_env import PerWorld
orig_guard, orig_step = PerWorld.guard, PerWorld.step

def json_clean(value):
    if isinstance(value, dict):
        return {k: json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value

def guard(self, requested):
    cmd, blocked = orig_guard(self, requested)
    raw, frame = self.scan
    finite = np.isfinite(raw) & (raw >= frame['min']) & (raw <= frame['max'])
    if finite.any():
        idx = int(np.argmin(np.where(finite, raw, np.inf)))
        nearest = dict(range_m=float(raw[idx]), bearing_rad=float(frame['angle_min'] + idx*frame['angle_step']))
    else:
        nearest = None
    self._diag_guards.append(dict(requested=list(map(float, requested)), command=list(map(float, cmd)),
        blocked=bool(blocked), sensor_fault=bool(self.fault),
        scan_age_s=float(self.state['time']-frame['stamp']), nearest_return=nearest,
        detector_points=int(0 if self.points is None else len(self.points)),
        tracked_humans=int(self.tracker.seen.sum()), actual_twist=list(map(float, self.state['twist']))))
    return cmd, blocked

def step(self, latent, human_velocity):
    self._diag_guards = []
    before = dict(sensor_fault=bool(self.fault), pose=list(map(float, self.state['truth'])),
                  goal=list(map(float, self.goal)), actual_twist=list(map(float, self.state['twist'])))
    obs, reward, done, metrics = orig_step(self, latent, human_velocity)
    item = dict(scenario_id=self.scenario.get('id'), episode=self.episode, step=self.steps,
        latent_action=np.asarray(latent).tolist(), before=before, guard_substeps=self._diag_guards,
        after_pose=list(map(float, self.state['truth'])), after_twist=list(map(float, self.state['twist'])),
        reward=float(reward), done=bool(done), metrics=metrics)
    with (out / f'lane-{self.index:03d}-action-gate.jsonl').open('a') as stream:
        stream.write(json.dumps(json_clean(item), allow_nan=False)+'\n')
    return obs, reward, done, metrics

PerWorld.guard, PerWorld.step = guard, step
import train
sys.argv = ['train.py', '--output_dir', str(out), '--waffle-pi', '--waffle-native-action', '--waffle-gazebo',
    '--waffle-fidelity-profile', str(root/'mission/profiles/waffle_gazebo_illustrative_safe_clearance_diverse_v5_500x3.json'),
    '--waffle-gazebo-model', str(root/'upstream/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_waffle_pi/model.sdf'),
    '--waffle-gazebo-worker', str(root/'build/waffle_gazebo_worker'), '--waffle-gazebo-logs', str(logs),
    '--waffle-gazebo-detector-device', 'learner', '--waffle-detector-weights', str(root/'models/dr_spaam/ckpt_jrdb_ann_dr_spaam_e20.pth'),
    '--waffle-humans', '4', '--waffle-scene-split', 'validation', '--waffle-eval-episodes', str(a.episodes),
    '--waffle-eval-json', str(out/'metrics.json'), '--load-path', str(checkpoint), '--num-processes', str(a.processes),
    '--num-mini-batch', str(a.processes), '--num-steps', '30', '--seq_length', '30', '--seed', str(a.seed)]
train.main()
