"""Measure the native GzDRL pool on a user-supplied Gazebo SDF.

No motion commands, policy updates or real ROS communication. A successful
probe demonstrates only stationary physics/sensor availability and throughput.
Use a separate OS process per pool size, because renderer resources are global.
"""
import argparse
import json
from pathlib import Path
import time
import uuid
import numpy as np


def run(world,envs,steps,rounds,warmup,sensors,model):
    import gzdrl
    pool=gzdrl.AsyncDRLServerPool(envs,'waffle-probe-'+uuid.uuid4().hex+'-',
                                 str(Path(world).resolve()),[model],sensors)
    ids=list(range(envs))
    try:
        pool.run_N(ids,[steps])
        names=pool.lidar_sensor_names(ids) if sensors else []
        if sensors and any(len(x)!=1 for x in names):
            raise RuntimeError('Expected exactly one LiDAR per Waffle Pi world: '+str(names))
        scan_names=[x[0] for x in names]
        def tick():
            pool.run_N(ids,[steps])
            state=pool.state_info(ids,[model])
            if len(state)!=envs or any(not x for x in state):
                raise RuntimeError('Missing model state')
            if sensors:
                frames=pool.get_sensor_gpu_lidar(ids,scan_names)
                if len(frames)!=envs or any(np.asarray(f).size==0 for f in frames):
                    raise RuntimeError('Missing sensor frame')
                # Copy while owned by this iteration. Do not claim that cached
                # frames are fresh: timestamps require separate qualification.
                frames=[np.array(f,copy=True) for f in frames]
                if any(not np.isfinite(f).any() for f in frames):
                    raise RuntimeError('No finite sensor values')
                return [list(f.shape) for f in frames]
            return []
        for _ in range(warmup):
            tick()
        start=time.perf_counter()
        for _ in range(rounds):
            shapes=tick()
        elapsed=time.perf_counter()-start
        return dict(envs=envs,physics_steps_per_round=steps,rounds=rounds,
                    elapsed_s=elapsed,world_rounds_per_s=envs*rounds/elapsed,
                    physics_iterations_per_s=envs*rounds*steps/elapsed,
                    lidar_shapes=shapes,sensor_frames_requested=sensors,
                    sensor_freshness_verified=False,ppo_trained=False)
    finally:
        pool.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--world',required=True)
    p.add_argument('--envs',type=int,default=1)
    p.add_argument('--steps',type=int,default=250)
    p.add_argument('--rounds',type=int,default=20)
    p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--no-sensors',action='store_true')
    p.add_argument('--model',default='turtlebot3_waffle_pi')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if not Path(a.world).is_file() or not all(v>0 for v in (a.envs,a.steps,a.rounds)) or a.warmup<0:
        p.error('Invalid world or benchmark counts')
    path=Path(a.output)
    if path.exists():
        p.error('Use a new output path')
    result=run(a.world,a.envs,a.steps,a.rounds,a.warmup,not a.no_sensors,a.model)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
