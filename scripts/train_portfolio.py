"""Run independent native-action PPO candidates concurrently across devices.

One process per GPU by default, or explicitly selected CPU workers. Each
candidate has its own policy, optimizer, GST instance and simulator. This
is not DDP and never averages policies or merges off-policy rollouts.
"""
import argparse
import concurrent.futures as futures
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def rank_candidates(rows,max_collision):
    # Safety constraints precede success and reward; this is a simulator
    # screening criterion only, not authorization for real-world deployment.
    valid=[r for r in rows if r.get('status')=='ok' and
           r['evaluation']['collision_rate']<=max_collision]
    return sorted(valid,key=lambda r:(-r['evaluation']['success_rate'],
        r['evaluation']['collision_rate'],r['evaluation']['gate_intervention_fraction'],
        -r['evaluation']['mean_reward'],r['wall_s']))


def run_candidate(a,root,slot,job):
    index,seed,lr,entropy,freeze=job
    label=f'candidate_{index:03d}'
    out=root/label
    checkpoint=None
    row=dict(index=index,seed=seed,lr=lr,entropy=entropy,
             freeze_human_attention=freeze,device=slot,status='failed')
    env=os.environ.copy()
    # Do not inherit a torchrun distributed context into independent trials.
    for key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):
        env.pop(key,None)
    env.pop('WAFFLE_RANK_CPUSETS',None)
    from gazebo_sim.hardware import candidate_cpuset,candidate_gpu
    spec=candidate_cpuset(slot)
    if spec:
        env['WAFFLE_RANK_CPUSETS']=spec
    env['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    env['CUDA_VISIBLE_DEVICES']=candidate_gpu(slot)
    env['OMP_NUM_THREADS']=str(a.threads)
    env['MKL_NUM_THREADS']=str(a.threads)
    env['MPLBACKEND']='Agg'
    common=[sys.executable,'train.py','--waffle-pi','--waffle-native-action',
            '--fast-training','--masked-gru',
            '--gae-backend',a.gae,'--num-steps',str(a.steps),
            '--seq_length',str(a.steps),'--num-processes',str(a.envs),
            '--num-mini-batch',str(a.minibatches),'--waffle-humans',str(a.humans),
            '--waffle-timeout',str(a.episode_timeout),'--waffle-arena',str(a.arena),
            '--waffle-speed',str(a.speed),'--waffle-angular-speed',str(a.angular_speed),
            '--waffle-radius',str(a.radius),'--waffle-sensor-range',str(a.sensor_range),
            '--waffle-dynamics-randomization',str(a.dynamics_randomization)]
    from gazebo_sim.cli import backend_flags
    common+=backend_flags(a)
    if a.fidelity_profile:
        common+=['--waffle-fidelity-profile',a.fidelity_profile]
    if a.detector_weights:
        common+=['--waffle-detector-weights',a.detector_weights]
    if a.require_measured:
        common+=['--waffle-require-measured']
    if a.calibration:
        common+=['--waffle-calibration',a.calibration]
    if a.safety_calibration:
        common+=['--waffle-safety-calibration',a.safety_calibration]
    if a.occlusion:
        common+=['--waffle-occlusion']
    common+=['--waffle-noise',str(a.noise),'--waffle-dropout',str(a.dropout)]
    if slot.startswith('cpu:'):
        common+=['--no-cuda']
    train=common+['--resume','--waffle-init-cartesian','--load-path',a.checkpoint,
                  '--seed',str(seed),'--lr',str(lr),'--entropy-coef',str(entropy),
                  '--ppo-epoch',str(a.epochs),'--num-env-steps',str(a.transitions),
                  '--output_dir',str(out)]
    if freeze:
        train+=['--freeze-human-attention']
    if a.compile:
        train+=['--compile-policy','--compile-gst']
        if a.backend!='gazebo':
            train+=['--waffle-compile-dynamics']
    if a.bf16:
        train+=['--amp-bf16','--fused-adam','--tf32']
    row['train_command']=train
    start=time.perf_counter()
    try:
        with (root/(label+'.train.log')).open('w') as log:
            result=subprocess.run(train,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout_s)
        if result.returncode:
            row['train_returncode']=result.returncode
            return row
        checkpoints=list((out/'checkpoints').glob('*.pt'))
        if not checkpoints:
            raise RuntimeError('Trainer produced no checkpoints')
        checkpoint=max(checkpoints,key=lambda p:int(p.stem))
        row['checkpoint']=str(checkpoint)
        evaluation=out/'heldout.json'
        evaluate=common+['--waffle-eval-episodes',str(a.eval_episodes),
                         '--seed',str(a.eval_seed),'--load-path',str(checkpoint),
                         '--waffle-eval-json',str(evaluation)]
        if a.require_measured:
            evaluate+=['--waffle-scene-split','validation']
        row['eval_command']=evaluate
        with (root/(label+'.eval.log')).open('w') as log:
            result=subprocess.run(evaluate,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout_s)
        if result.returncode:
            row['eval_returncode']=result.returncode
            return row
        row.update(status='ok',evaluation=json.loads(evaluation.read_text()))
    except subprocess.TimeoutExpired:
        row['status']='timeout'
    except (OSError,ValueError,RuntimeError) as exc:
        row['error']=str(exc)
    finally:
        row['wall_s']=time.perf_counter()-start
    return row


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--backend',choices=['gazebo','tensor'],default='gazebo')
    from gazebo_sim.cli import add_flags
    add_flags(p)
    p.add_argument('--fidelity-profile')
    p.add_argument('--detector-weights')
    p.add_argument('--require-measured',action='store_true')
    p.add_argument('--gpus',type=int,nargs='+',default=None)
    p.add_argument('--cpu-workers',type=int,default=0)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--checkpoint',default='trained_models/GST_predictor_rand/checkpoints/41665.pt')
    p.add_argument('--envs',type=int,default=256)
    p.add_argument('--minibatches',type=int,default=16)
    p.add_argument('--steps',type=int,default=30)
    p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--transitions',type=int,default=1000000)
    p.add_argument('--seeds',type=int,nargs='+',default=[11,29,47])
    p.add_argument('--lrs',type=float,nargs='+',default=[4e-5,1e-4])
    p.add_argument('--entropies',type=float,nargs='+',default=[0.,.01])
    p.add_argument('--freeze',choices=['none','human','both'],default='none')
    p.add_argument('--eval-seed',type=int,default=90210)
    p.add_argument('--eval-episodes',type=int,default=100)
    p.add_argument('--max-collision',type=float,default=.05)
    p.add_argument('--humans',type=int,default=4)
    p.add_argument('--episode-timeout',type=float,default=90.)
    p.add_argument('--arena',type=float,default=2.5)
    p.add_argument('--speed',type=float,default=.10)
    p.add_argument('--angular-speed',type=float,default=.50)
    p.add_argument('--radius',type=float,default=.25)
    p.add_argument('--sensor-range',type=float,default=3.)
    p.add_argument('--calibration')
    p.add_argument('--safety-calibration')
    p.add_argument('--dynamics-randomization',type=float,default=0.)
    p.add_argument('--occlusion',action='store_true')
    p.add_argument('--noise',type=float,default=0.)
    p.add_argument('--dropout',type=float,default=0.)
    p.add_argument('--gae',choices=['reference','scan','triton'],default='scan')
    p.add_argument('--compile',action='store_true')
    p.add_argument('--bf16',action='store_true')
    p.add_argument('--timeout-s',type=int,default=14400)
    a=p.parse_args()
    from gazebo_sim.cli import backend_flags
    try:
        backend_flags(a)
    except ValueError as exc:
        p.error(str(exc))
    if a.gpus is None and a.cpu_workers==0:
        a.gpus=[0]
    if (a.cpu_workers<0 or (a.cpu_workers and a.gpus) or
        not all(x>0 for x in (a.envs,a.minibatches,a.steps,a.epochs,a.transitions,
                              a.threads,a.eval_episodes,a.timeout_s)) or
        a.envs%a.minibatches or a.transitions<a.envs*a.steps or
        a.eval_seed in a.seeds or not 0<=a.max_collision<=1 or
        any(x<=0 for x in a.lrs) or any(x<0 for x in a.entropies) or
        (a.cpu_workers and (a.bf16 or a.gae=='triton')) or
        (a.gpus and (min(a.gpus)<0 or len(set(a.gpus))!=len(a.gpus)))):
        p.error('Invalid portfolio configuration')
    slots=[f'cpu:{i}' for i in range(a.cpu_workers)] if a.cpu_workers else list(map(str,a.gpus))
    frozen=[False,True] if a.freeze=='both' else [a.freeze=='human']
    jobs=[(i,*j) for i,j in enumerate(itertools.product(a.seeds,a.lrs,a.entropies,frozen))]
    root=Path(a.output).resolve()
    root.mkdir(parents=True,exist_ok=False)
    (root/'portfolio_args.json').write_text(json.dumps(vars(a),indent=2)+'\n')
    pending=iter(jobs)
    rows=[]
    with futures.ThreadPoolExecutor(max_workers=len(slots)) as executor:
        active={}
        def submit(slot):
            job=next(pending,None)
            if job is not None:
                active[executor.submit(run_candidate,a,root,slot,job)]=slot
        for slot in slots:
            submit(slot)
        while active:
            completed,_=futures.wait(active,return_when=futures.FIRST_COMPLETED)
            for future in completed:
                slot=active.pop(future)
                row=future.result()
                rows.append(row)
                ranking=rank_candidates(rows,a.max_collision)
                (root/'results.json').write_text(json.dumps(sorted(rows,key=lambda r:r['index']),indent=2)+'\n')
                (root/'ranking.json').write_text(json.dumps(ranking,indent=2)+'\n')
                print('Candidate',row['index'],row['status'],row.get('evaluation'),flush=True)
                submit(slot)
    if not rank_candidates(rows,a.max_collision):
        raise SystemExit('No candidate met the simulator collision criterion; inspect logs')
    print('Best held-out simulator candidate:',rank_candidates(rows,a.max_collision)[0]['checkpoint'])


if __name__=='__main__':
    main()
