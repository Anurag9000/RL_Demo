"""Concurrent native PPO curriculum, successive halving and population training.

Each rung is a fresh process and fresh on-policy rollout boundary, restoring
the complete learner state. No gradient sharing between candidates. Stages
share fixed robot limits; only crowd difficulty and perception variation change.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time


def atomic_json(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)


def score(row,max_collision,max_gate):
    e=row['evaluation']
    feasible=e['collision_rate']<=max_collision and e['gate_intervention_fraction']<=max_gate
    return (int(feasible),-max(0.,e['collision_rate']-max_collision),
            -max(0.,e['gate_intervention_fraction']-max_gate),e['success_rate'],
            -e['collision_rate'],-e['gate_intervention_fraction'],e['mean_reward'])


def target(row,a):
    e=row['evaluation']
    return (e['success_rate']>=a.target_success and e['collision_rate']<=a.max_collision
            and e['gate_intervention_fraction']<=a.max_gate)


def rank(rows,a):
    return sorted((r for r in rows if r['status']=='ok'),
                  key=lambda r:(score(r,a.max_collision,a.max_gate),-r['id']),reverse=True)


def curriculum(a):
    counts=list(dict.fromkeys([1,max(1,a.humans//2),a.humans])) if a.curriculum else [a.humans]
    stages=[]
    for i,n in enumerate(counts):
        fraction=(i+1)/len(counts)
        final=i==len(counts)-1
        stages.append(dict(humans=n,noise=a.noise*fraction if not final else a.noise,
            dropout=a.dropout*fraction,randomization=a.randomization*fraction,
            occlusion=a.occlusion and final,reactive=a.reactive and final,
            detection_delay=a.detection_delay if final else 0,pose_drift=a.pose_drift*fraction))
    return stages


def mutate(h,rng,a):
    # Bounded multiplicative exploration; entropy can recover from zero.
    lr=min(a.lr_max,max(a.lr_min,h['lr']*rng.choice([.8,1.2])))
    entropy=min(.05,max(0.,max(h['entropy_coef'],.001)*rng.choice([.8,1.2])))
    clip=min(.3,max(.05,h['clip_param']*rng.choice([.8,1.2])))
    return dict(lr=lr,entropy_coef=entropy,clip_param=clip)


def next_population(rows,a,rng):
    ordered=rank(rows,a)
    if not ordered:
        return [],[]
    if a.mode in ('halving','hybrid'):
        n=max(a.min_population if a.mode=='hybrid' else 1,math.ceil(len(ordered)/a.eta))
        keep=ordered[:min(n,len(ordered))]
    else:
        keep=ordered
    retained_ids={r['id'] for r in keep}
    pruned=[r['id'] for r in rows if r['id'] not in retained_ids]
    elite_count=max(1,math.ceil(len(keep)*a.elite_fraction))
    elites=keep[:elite_count]
    result=[]
    for i,r in enumerate(keep):
        donor=r
        hypers=dict(r['hyperparameters'])
        mutated=False
        if a.mode in ('pbt','hybrid') and i>=elite_count:
            donor=rng.choice(elites)
            hypers=mutate(donor['hyperparameters'],rng,a)
            mutated=True
        result.append(dict(id=r['id'],seed=r['seed'],state=donor['state'],
            donor=donor['id'],hyperparameters=hypers,retune=mutated,
            stage=donor['stage'],streak=0 if mutated else r['streak'],
            own_transitions=r['own_transitions']))
    return result,pruned


def common(a,stage,slot):
    cmd=[sys.executable,'train.py','--waffle-pi','--waffle-native-action','--waffle-live-orca',
         '--fast-training','--masked-gru','--gae-backend',a.gae,
         '--num-processes',str(a.envs),'--num-mini-batch',str(a.minibatches),
         '--num-steps',str(a.steps),'--seq_length',str(a.steps),'--ppo-epoch',str(a.epochs),
         '--waffle-humans',str(stage['humans']),'--waffle-pool-count',str(a.pool_count),
         '--waffle-timeout',str(a.episode_timeout),'--waffle-arena',str(a.arena),
         '--waffle-speed',str(a.speed),'--waffle-angular-speed',str(a.angular_speed),
         '--waffle-radius',str(a.radius),'--waffle-sensor-range',str(a.sensor_range),
         '--waffle-noise',str(stage['noise']),'--waffle-dropout',str(stage['dropout']),
         '--waffle-dynamics-randomization',str(stage['randomization']),
         '--waffle-detection-delay',str(stage['detection_delay']),
         '--waffle-pose-drift',str(stage['pose_drift']),
         '--adaptive-kl',str(a.adaptive_kl),'--lr-min',str(a.lr_min),'--lr-max',str(a.lr_max)]
    if slot.startswith('cpu:'):
        cmd+=['--no-cuda']
    if stage['occlusion']:
        cmd+=['--waffle-occlusion']
    if stage['reactive']:
        cmd+=['--waffle-reactive-humans']
    if a.fidelity_profile:
        cmd+=['--waffle-fidelity-profile',a.fidelity_profile]
    if a.require_measured:
        cmd+=['--waffle-require-measured']
    if a.freeze_human_attention:
        cmd+=['--freeze-human-attention']
    if a.gst_adjacent_mask:
        cmd+=['--gst-adjacent-mask']
    for flag,path in (('--waffle-calibration',a.calibration),
                      ('--waffle-safety-calibration',a.safety_calibration)):
        if path:
            cmd+=[flag,path]
    return cmd


def device_env(a,slot):
    env=os.environ.copy()
    for key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):
        env.pop(key,None)
    env.update(CUDA_VISIBLE_DEVICES='' if slot.startswith('cpu:') else slot,
               OMP_NUM_THREADS=str(a.threads),MKL_NUM_THREADS=str(a.threads),MPLBACKEND='Agg')
    return env


def execute(cmd,env,log,timeout):
    with log.open('w') as handle:
        subprocess.run(cmd,env=env,stdout=handle,stderr=subprocess.STDOUT,
                       timeout=timeout,check=True)


def evaluate(a,stage,slot,checkpoint,seed,episodes,out,log):
    cmd=common(a,stage,slot)+['--seed',str(seed),'--waffle-eval-episodes',str(episodes),
                              '--load-path',str(checkpoint),'--waffle-eval-json',str(out)]
    execute(cmd,device_env(a,slot),log,a.timeout_s)
    result=json.loads(out.read_text())
    for k in ('success_rate','collision_rate','gate_intervention_fraction','mean_reward'):
        if not math.isfinite(result[k]):
            raise ValueError('Non-finite validation metric: '+k)
    return result


def segment(a,root,slot,candidate,rung,stage_index,stage,budget):
    out=root/f'candidate_{candidate["id"]:03d}'/f'rung_{rung:03d}'
    out.parent.mkdir(parents=True,exist_ok=True)
    row=dict(candidate,status='failed',stage=stage_index,rung=rung,device=slot)
    start=time.perf_counter()
    # Distinct training streams across rungs and recipients, shared validation stream.
    seed=(candidate['seed']+100003*(rung+1)+1009*candidate['id'])%(2**31-1)
    while seed in (a.validation_seed,a.test_seed):
        seed=(seed+1)%(2**31-1)
    cmd=common(a,stage,slot)+['--seed',str(seed),'--save-training-state',
            '--num-env-steps',str(budget*a.envs*a.steps),'--output_dir',str(out)]
    for flag,key in (('--lr','lr'),('--entropy-coef','entropy_coef'),('--clip-param','clip_param')):
        cmd+=[flag,str(candidate['hyperparameters'][key])]
    if candidate.get('state'):
        cmd+=['--continue-state',candidate['state'],'--fresh-continuation-rng']
        if candidate['stage']!=stage_index:
            cmd+=['--curriculum-transition']
        if candidate['retune']:
            cmd+=['--retune-optimizer']
    else:
        cmd+=['--resume','--waffle-init-cartesian','--load-path',a.checkpoint]
    if a.compile:
        cmd+=['--compile-policy','--compile-gst','--waffle-compile-dynamics']
    if a.bf16:
        cmd+=['--amp-bf16','--fused-adam','--tf32']
    row['train_command']=cmd
    try:
        execute(cmd,device_env(a,slot),out.parent/(out.name+'.train.log'),a.timeout_s)
        summary=json.loads((out/'training_summary.json').read_text())
        checkpoint=max((out/'checkpoints').glob('*.pt'),key=lambda p:int(p.stem),default=None)
        if checkpoint is None:
            raise RuntimeError('Missing inference checkpoint')
        state=out/'training_state.pt'
        if not state.is_file() or summary['segment_updates']!=budget:
            raise RuntimeError('Incomplete segment state')
        row.update(state=str(state),checkpoint=str(checkpoint),
            own_transitions=candidate['own_transitions']+summary['segment_transitions'],
            lineage_transitions=summary['transitions'],
            hyperparameters={k:summary[k] for k in ('lr','entropy_coef','clip_param')})
        row['evaluation']=evaluate(a,stage,slot,checkpoint,a.validation_seed,a.eval_episodes,
                                    out/'validation.json',out.parent/(out.name+'.eval.log'))
        row['status']='ok'
        row['streak']=(candidate['streak'] if candidate['stage']==stage_index else 0)+1 if target(row,a) else 0
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as exc:
        row['error']=str(exc)
    finally:
        row['wall_s']=time.perf_counter()-start
    return row


def run_round(a,root,slots,candidates,rung,stage_index,stage,budget,worker=segment):
    # At most one learner per assigned GPU. Work queues are independent.
    from queue import Queue,Empty
    queue=Queue()
    for c in candidates:
        queue.put(c)
    def consume(slot):
        result=[]
        while True:
            try:
                candidate=queue.get_nowait()
            except Empty:
                return result
            result.append(worker(a,root,slot,candidate,rung,stage_index,stage,budget))
    with ThreadPoolExecutor(max_workers=len(slots)) as executor:
        groups=list(executor.map(consume,slots))
    return sorted(itertools.chain.from_iterable(groups),key=lambda r:r['id'])


def optimize(a,root,slots,candidates,worker=segment,final_eval=evaluate):
    stages=curriculum(a)
    rng=random.Random(a.scheduler_seed)
    history=[]
    final_rows=[]
    stage=0
    best_score=None
    stale=0
    reason='budget_exhausted'
    for rung,budget in enumerate(a.rung_updates):
        rows=run_round(a,root,slots,candidates,rung,stage,stages[stage],budget,worker)
        ranked=rank(rows,a)
        entry=dict(rung=rung,stage=stage,profile=stages[stage],budget_updates=budget,rows=rows,pruned=[])
        history.append(entry)
        if not ranked:
            reason='all_candidates_failed'
            atomic_json(root/'history.json',history)
            break
        if stage==len(stages)-1:
            final_rows.extend(ranked)
            current=score(ranked[0],a.max_collision,a.max_gate)
            if best_score is None or current>best_score:
                best_score,stale=current,0
            else:
                stale+=1
            if any(r['streak']>=a.target_streak for r in ranked):
                reason='validation_target_reached'
            elif a.patience and stale>=a.patience:
                reason='validation_plateau'
        if reason!='budget_exhausted' or rung+1==len(a.rung_updates):
            atomic_json(root/'history.json',history)
            break
        candidates,entry['pruned']=next_population(rows,a,rng)
        if stage<len(stages)-1 and ranked[0]['evaluation']['success_rate']>=a.promotion_success:
            stage+=1
            entry['promoted_to']=stage
        atomic_json(root/'history.json',history)
    # Only final-difficulty results are eligible. No easy-stage winner.
    eligible=rank(final_rows,a)
    eligible=[r for r in eligible if r['evaluation']['collision_rate']<=a.max_collision
              and r['evaluation']['gate_intervention_fraction']<=a.max_gate]
    result=dict(reason=reason,stages=stages,completed_rungs=len(history),
                attempted_candidates=sum(len(h['rows']) for h in history),
                successful_training_transitions=0,winner=None,converged=False)
    # Count only executed segment budgets, not accumulated donor lineage.
    result['successful_training_transitions']=sum(h['budget_updates']*a.envs*a.steps
                     for h in history for r in h['rows'] if r['status']=='ok')
    if eligible:
        # Prefer a candidate that achieved the required repeated validation.
        confirmed=[r for r in eligible if r['streak']>=a.target_streak]
        winner=(confirmed or eligible)[0]
        result['winner']=winner
        try:
            result['test']=final_eval(a,stages[-1],slots[0],winner['checkpoint'],a.test_seed,
                        a.test_episodes,root/'test.json',root/'test.log')
            result['converged']=bool(confirmed) and target(dict(evaluation=result['test']),a)
        except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as exc:
            result['test_error']=str(exc)
    atomic_json(root/'result.json',result)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--gpus',type=int,nargs='+')
    p.add_argument('--cpu-workers',type=int,default=0)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--mode',choices=['halving','pbt','hybrid'],default='hybrid')
    p.add_argument('--curriculum',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--rung-updates',type=int,nargs='+',default=[8,24,64,128,256])
    p.add_argument('--eta',type=int,default=2)
    p.add_argument('--min-population',type=int,default=2)
    p.add_argument('--elite-fraction',type=float,default=.5)
    p.add_argument('--seeds',type=int,nargs='+',default=[11,29])
    p.add_argument('--lrs',type=float,nargs='+',default=[4e-5,1e-4])
    p.add_argument('--entropies',type=float,nargs='+',default=[.001,.01])
    p.add_argument('--clip',type=float,default=.2)
    p.add_argument('--adaptive-kl',type=float,default=.01)
    p.add_argument('--lr-min',type=float,default=1e-6)
    p.add_argument('--lr-max',type=float,default=3e-4)
    p.add_argument('--scheduler-seed',type=int,default=45)
    p.add_argument('--validation-seed',type=int,default=90210)
    p.add_argument('--test-seed',type=int,default=190210)
    p.add_argument('--eval-episodes',type=int,default=64)
    p.add_argument('--test-episodes',type=int,default=200)
    p.add_argument('--promotion-success',type=float,default=0.)
    p.add_argument('--target-success',type=float,default=.85)
    p.add_argument('--max-collision',type=float,default=.05)
    p.add_argument('--max-gate',type=float,default=.3)
    p.add_argument('--target-streak',type=int,default=2)
    p.add_argument('--patience',type=int,default=3)
    p.add_argument('--checkpoint',default='trained_models/GST_predictor_rand/checkpoints/41665.pt')
    p.add_argument('--envs',type=int,default=256)
    p.add_argument('--minibatches',type=int,default=16)
    p.add_argument('--steps',type=int,default=30)
    p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--pool-count',type=int,default=1024)
    p.add_argument('--humans',type=int,default=4)
    p.add_argument('--episode-timeout',type=float,default=90.)
    p.add_argument('--arena',type=float,default=2.5)
    p.add_argument('--speed',type=float,default=.1)
    p.add_argument('--angular-speed',type=float,default=.5)
    p.add_argument('--radius',type=float,default=.25)
    p.add_argument('--sensor-range',type=float,default=3.)
    p.add_argument('--noise',type=float,default=0.)
    p.add_argument('--dropout',type=float,default=0.)
    p.add_argument('--randomization',type=float,default=0.)
    p.add_argument('--pose-drift',type=float,default=0.)
    p.add_argument('--detection-delay',type=int,default=0)
    p.add_argument('--occlusion',action='store_true')
    p.add_argument('--reactive',action='store_true')
    p.add_argument('--fidelity-profile')
    p.add_argument('--require-measured',action='store_true')
    p.add_argument('--calibration')
    p.add_argument('--safety-calibration')
    p.add_argument('--freeze-human-attention',action='store_true')
    p.add_argument('--gst-adjacent-mask',action='store_true')
    p.add_argument('--gae',choices=['reference','scan','triton'],default='scan')
    p.add_argument('--compile',action='store_true')
    p.add_argument('--bf16',action='store_true')
    p.add_argument('--timeout-s',type=int,default=14400)
    a=p.parse_args()
    if a.gpus is None and not a.cpu_workers:
        a.gpus=[0]
    if (a.cpu_workers<0 or (a.cpu_workers and a.gpus) or
        (a.gpus and (min(a.gpus)<0 or len(set(a.gpus))!=len(a.gpus))) or
        (a.cpu_workers and (a.bf16 or a.gae=='triton')) or
        not all(x>0 for x in [a.threads,a.envs,a.minibatches,a.steps,a.epochs,a.pool_count,
                a.humans,a.eval_episodes,a.test_episodes,a.target_streak,a.timeout_s,
                a.min_population,*a.rung_updates]) or a.envs%a.minibatches or a.eta<2 or
        not 0<a.elite_fraction<1 or not 0<a.lr_min<=a.lr_max or a.adaptive_kl<0 or
        any(not a.lr_min<=x<=a.lr_max for x in a.lrs) or any(not 0<=x<=.05 for x in a.entropies) or
        not .05<=a.clip<=.3 or a.patience<0 or
        any(not 0<=x<=1 for x in (a.promotion_success,a.target_success,a.max_collision,a.max_gate)) or
        a.validation_seed==a.test_seed or any(x in a.seeds for x in (a.validation_seed,a.test_seed)) or
        len(a.rung_updates)<len(curriculum(a))):
        p.error('Invalid scheduler configuration or insufficient curriculum rungs')
    slots=[f'cpu:{i}' for i in range(a.cpu_workers)] if a.cpu_workers else list(map(str,a.gpus))
    candidates=[dict(id=i,seed=seed,state=None,donor=None,retune=False,stage=-1,streak=0,
                     own_transitions=0,hyperparameters=dict(lr=lr,entropy_coef=entropy,clip_param=a.clip))
                for i,(seed,lr,entropy) in enumerate(itertools.product(a.seeds,a.lrs,a.entropies))]
    root=Path(a.output).resolve()
    root.mkdir(parents=True,exist_ok=False)
    atomic_json(root/'args.json',vars(a))
    result=optimize(a,root,slots,candidates)
    print(json.dumps(result,indent=2))
    if result['winner'] is None:
        raise SystemExit('No eligible final-stage candidate; inspect history.json')


if __name__=='__main__':
    main()
