#!/usr/bin/env python3
"""Build separate illustrative Waffle banks for the paper's two human regimes.

Human positions, routes and traits are sampled before a run. In the randomized
regime, scheduled mid-episode goal changes are applied by GazeboVecEnv from the
hidden profile contract. Both regimes retain the configured end-of-route goal
resampling used for continuous crowd flow. The policy only receives
detector/tracker observations.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

SPLIT_SEEDS={
 'randomized':{'train':731025,'validation':842137,'test':953249},
 'nonrandomized':{'train':1731025,'validation':1842137,'test':1953249},
}


def human_contract(regime):
    if regime=='randomized':
        return dict(regime=regime,random_goal_changing=True,
            random_goal_change_interval_s=5.0,random_goal_change_chance=.5,
            end_goal_changing=True,end_goal_change_chance=1.0,
            goal_clearance_m=.2)
    if regime=='nonrandomized':
        return dict(regime=regime,random_goal_changing=False,
            random_goal_change_interval_s=0.0,random_goal_change_chance=0.0,
            end_goal_changing=True,end_goal_change_chance=1.0,
            goal_clearance_m=.2)
    raise ValueError(regime)


def sample_case(rng,regime,split,index,arena,robot_radius):
    for _ in range(10000):
        humans=[]; starts=[]
        for _h in range(4):
            center=rng.uniform(-.35,.35,size=2)
            angle=rng.uniform(0,math.pi)
            unit=np.array([math.cos(angle),math.sin(angle)])
            half=rng.uniform(.95,1.38)
            start=center-half*unit; goal=center+half*unit
            if max(np.max(np.abs(start)),np.max(np.abs(goal)))>arena-.55:
                break
            if regime=='randomized':
                radius=float(rng.uniform(.3,.5))
                speed=float(rng.uniform(.5,1.5))
            else:
                radius=.3; speed=1.0
            if any(np.linalg.norm(start-p)<radius+pr+.08 for p,pr in starts):
                break
            starts.append((start,radius))
            humans.append({'start':start.tolist(),'radius':radius,'speed':speed,
                           'waypoints':[goal.tolist(),start.tolist()]})
        if len(humans)!=4:
            continue
        robot_start=rng.uniform(-.85,.85,size=2)
        robot_goal=rng.uniform(-.85,.85,size=2)
        distance=np.linalg.norm(robot_goal-robot_start)
        if not .9<=distance<=2.1:
            continue
        if any(np.linalg.norm(robot_start-h['start'])<=robot_radius+h['radius']+.08 for h in humans):
            continue
        if any(np.linalg.norm(robot_goal-h['start'])<=robot_radius+h['radius']+.08 for h in humans):
            continue
        yaw=math.atan2(robot_goal[1]-robot_start[1],robot_goal[0]-robot_start[0])
        return {'id':f'{regime}-{split}-{index:04d}','split':split,
                'robot_start':[float(robot_start[0]),float(robot_start[1]),yaw],
                'robot_goal':robot_goal.tolist(),'humans':humans}
    raise RuntimeError(f'could not sample {regime}/{split}/{index} after 10000 attempts')


def signature(case):
    return json.dumps({k:case[k] for k in ('robot_start','robot_goal','humans')},
                      sort_keys=True,separators=(',',':'))


def build_profile(base_path,output_path,regime,per_split):
    profile=json.loads(base_path.read_text())
    if profile.get('schema')!='waffle_fidelity_v1' or profile.get('status')!='illustrative':
        raise ValueError('base must be an illustrative waffle_fidelity_v1 profile')
    scene=profile['scene']; arena=float(scene['arena_half_extent_m'])
    if scene.get('rectangles_m'):
        raise ValueError('scenario sampler currently requires an empty-rectangle arena')
    bank=[]; signatures=set(); split_info={}
    for split,seed in SPLIT_SEEDS[regime].items():
        rng=np.random.default_rng(seed)
        cases=[sample_case(rng,regime,split,i,arena,.25) for i in range(per_split)]
        for case in cases:
            sig=signature(case)
            if sig in signatures:
                raise RuntimeError('duplicate scenario across splits')
            signatures.add(sig); bank.append(case)
        split_info[split]={'seed':seed,'count':len(cases)}
    profile['scene']['scenarios']=bank
    profile['human_dynamics']=human_contract(regime)
    profile['mission_note']=(
      f'Illustrative, nominal four-human Waffle Pi profile for the paper {regime} regime. '
      'Crowd goals use bounded hidden simulator-side updates; arena and robot physics remain Waffle-specific. '
      'Not calibrated against physical hardware or real crowd measurements.')
    profile['scenario_bank']={
      'generator':'generate_paper_regime_scenarios.py','generator_version':1,
      'base_profile_sha256':hashlib.sha256(base_path.read_bytes()).hexdigest(),
      'regime':regime,'per_split':split_info,'total':len(bank),
      'independent_seeded_sampling':True,
      'provenance':'procedural illustrative scenarios; no measured human or hardware data',
      'paper_regime_contract':profile['human_dynamics']}
    output_path.parent.mkdir(parents=True,exist_ok=True)
    with output_path.open('x') as f:
        json.dump(profile,f,indent=2,allow_nan=False); f.write('\n')
    return {'path':str(output_path),'sha256':hashlib.sha256(output_path.read_bytes()).hexdigest(),
            'regime':regime,'counts':split_info,'total':len(bank)}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--per-split',type=int,default=500)
    parser.add_argument('--regime',choices=['randomized','nonrandomized','both'],default='both')
    args=parser.parse_args(argv)
    if args.per_split<1: parser.error('--per-split must be positive')
    base=Path(args.base); out=Path(args.output_dir)
    regimes=('randomized','nonrandomized') if args.regime=='both' else (args.regime,)
    results=[]
    for regime in regimes:
        name=f'waffle_gazebo_paper_{regime}_{args.per_split}x3.json'
        results.append(build_profile(base,out/name,regime,args.per_split))
    print(json.dumps(results,indent=2))

if __name__=='__main__':
    main()
