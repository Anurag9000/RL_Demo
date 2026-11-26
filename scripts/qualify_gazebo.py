
"""Qualification with real native Gazebo, before using detector/PPO.

Runs isolated simulation worlds, checks wheel response and scan chronology.
The result is not proof of calibrated hardware fidelity or policy quality.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import numpy as np
from gazebo_sim.world import validate_gazebo_profile,source_digest,write_case
from gazebo_sim.worker import Worker


def qualify(a,profile,case,root,i):
    world,conf=write_case(root/str(i),a.model,profile,case,a.seed+i)
    w=Worker(a.worker,world,conf,root/str(i)/'native.log',a.timeout)
    n=len(case['humans'])
    dt=profile['gazebo']['physics_dt']
    hz=profile['laser']['hz']
    step=round(.25/dt)
    result=[]
    start=time.perf_counter()
    try:
        state=w.step(round(max(.5,2/hz)/dt),0.,0.,[[0.,0.]]*n,True)
        if not state['floor_contact_observed'] or state['contact']:
            raise RuntimeError('Floor or initial collision qualification failed')
        initial=state['truth']
        last=-1.
        for v,omega,rounds in [(0.08,0.,a.rounds),(0.,0.,4),(0.,.3,4),(0.,0.,4)]:
            for _ in range(rounds):
                state=w.step(step,v,omega,[[0.,0.]]*n,False)
                if state['contact']:
                    raise RuntimeError('Unexpected contact during qualification')
                for f in state['scans']:
                    if not last<f['stamp']<=state['time']+dt:
                        raise RuntimeError('Laser frame order/time failed')
                    if len(f['ranges'])!=profile['laser']['rays']:
                        raise RuntimeError('Wrong laser ray count')
                    last=f['stamp']
                if state['time']-last>1/hz+3*dt:
                    raise RuntimeError('Stale laser')
                result.append(dict(command=[v,omega],truth=state['truth'],odom=state['odom'],
                    wheels=state['wheels'],twist=state['twist'],time=state['time'],scan_stamp=last))
        forward=result[:a.rounds]
        distance=float(np.linalg.norm(np.asarray(forward[-1]['truth'][:2])-np.asarray(initial[:2])))
        if distance<.005 or distance>.25*a.rounds*.2:
            raise RuntimeError('Forward motion response outside diagnostic bounds')
        return dict(index=i,forward_distance_m=distance,elapsed_s=time.perf_counter()-start,transitions=result)
    finally:
        w.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--worker',default='build/gazebo/waffle_gazebo_worker')
    p.add_argument('--profile',default='configs/waffle_gazebo.example.json')
    p.add_argument('--envs',type=int,default=1)
    p.add_argument('--rounds',type=int,default=8)
    p.add_argument('--seed',type=int,default=71)
    p.add_argument('--timeout',type=float,default=120.)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if min(a.envs,a.rounds)<1 or a.rounds>20:
        p.error('Use positive env count and <=20 diagnostic forward rounds')
    profile=json.loads(Path(a.profile).read_text())
    validate_gazebo_profile(profile)
    case=next(c for c in profile['scene']['scenarios'] if len(c['humans'])==1 and c.get('split','train')=='train')
    root=Path(a.output)
    root.mkdir(parents=True,exist_ok=False)
    source=source_digest(a.model)
    start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.envs) as pool:
        results=list(pool.map(lambda i:qualify(a,profile,case,root,i),range(a.envs)))
    elapsed=time.perf_counter()-start
    report=dict(backend='gazebo',model_sha256=source,envs=a.envs,elapsed_s=elapsed,
        world_rounds_per_sec=sum(len(r['transitions']) for r in results)/elapsed,
        results=results,calibrated=False,ppo_trained=False)
    (root/'qualification.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))


if __name__=='__main__':
    main()
