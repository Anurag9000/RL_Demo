"""Measure Gazebo/PPO placement on allocated Linux CPU/GPU resources.

The default emits a plan; --execute runs native Gazebo benchmarks. Equal
GLOBAL rollout size and minibatch count for single vs DDP; independent
candidates have half the worlds each and are never treated as one policy.
No GPU reset, clock, BIOS, driver or system-wide settings are changed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from gazebo_sim.hardware import numa_nodes,format_cpus
from scripts.benchmark_sweep import PRESETS


def capture(cmd):
    try:
        return subprocess.check_output(cmd,text=True,stderr=subprocess.STDOUT,timeout=20).strip()
    except (OSError,subprocess.SubprocessError) as exc:
        return 'unavailable: '+str(exc)


def inventory():
    import csv
    query='index,uuid,pci.bus_id,name,memory.total,memory.free,utilization.gpu'
    raw=capture(['nvidia-smi','--query-gpu='+query,'--format=csv,noheader,nounits'])
    if raw.startswith('unavailable:'):
        raise RuntimeError(raw)
    nodes=numa_nodes()
    gpus={}
    for row in csv.reader(raw.splitlines(),skipinitialspace=True):
        i,uuid,bus,name,total,free,util=row
        domain,rest=bus.lower().split(':',1)
        address=f'{int(domain,16):04x}:'+rest
        node=int((Path('/sys/bus/pci/devices')/address/'numa_node').read_text())
        if node not in nodes:
            raise RuntimeError('GPU NUMA node unknown or outside allocated CPUs: '+address)
        gpus[int(i)]=dict(uuid=uuid,bus=address,name=name,numa=node,total_mib=int(total),
                         free_mib=int(free),utilization=int(util))
    return dict(nodes=nodes,gpus=gpus,affinity=sorted(os.sched_getaffinity(0)),
                topology=capture(['nvidia-smi','topo','-m']),
                nvlink=capture(['nvidia-smi','nvlink','--status']),
                p2p=capture(['nvidia-smi','topo','-p2p','p']),
                memory=Path('/proc/meminfo').read_text(),
                shm_bytes=os.statvfs('/dev/shm').f_bavail*os.statvfs('/dev/shm').f_frsize)


def allocate(info,gpus):
    # If multiple selected GPUs share a NUMA node, partition its CPUs.
    mapping={}
    for node,cpus in info['nodes'].items():
        selected=[g for g in gpus if info['gpus'][g]['numa']==node]
        for j,g in enumerate(selected):
            mapping[g]=cpus[len(cpus)*j//len(selected):len(cpus)*(j+1)//len(selected)]
    if any(not mapping.get(g) for g in gpus):
        raise ValueError('No disjoint local CPU allocation for selected GPUs')
    return mapping


def modes(info,gpus,worlds,minibatches):
    if len(gpus)!=2 or worlds%2 or worlds%minibatches or (worlds//2)%minibatches:
        raise ValueError('Two GPUs; global worlds divisible by twice the minibatch count')
    local=allocate(info,gpus)
    a,b=gpus
    return [
        dict(name='single-all-cpus',devices=[a],cpus=[info['affinity']],envs=worlds,minibatches=minibatches,ddp=False),
        dict(name='single-local',devices=[a],cpus=[local[a]],envs=worlds,minibatches=minibatches,ddp=False),
        dict(name='single-local-second',devices=[b],cpus=[local[b]],envs=worlds,minibatches=minibatches,ddp=False),
        dict(name='ddp',devices=gpus,cpus=[local[g] for g in gpus],envs=worlds//2,minibatches=minibatches,ddp=True),
        dict(name='independent',devices=gpus,cpus=[local[g] for g in gpus],envs=worlds//2,minibatches=minibatches,ddp=False)]


def environment(info,devices,cpus,a):
    env=os.environ.copy()
    for key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):
        env.pop(key,None)
    env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=','.join(info['gpus'][g]['uuid'] for g in devices),
        WAFFLE_RANK_CPUSETS=';'.join(format_cpus(c) for c in cpus),
        WAFFLE_WORKER_AFFINITY='1',WAFFLE_WORKER_CORES=str(a.worker_cores),WAFFLE_RESERVE_CORES=str(a.reserve_cores),
        WAFFLE_NUMA_MEMORY='1' if a.numa_memory else '0',WAFFLE_SCAN_TRANSPORT=a.transport,
        WAFFLE_BATCH_DETECTOR='1' if a.batch_detector else '0',WAFFLE_DETECTOR_BATCH=str(a.detector_batch),
        WAFFLE_DETECTOR_WAIT_MS=str(a.detector_wait_ms),WAFFLE_PINNED_OBS='1' if a.pinned else '0',
        OMP_NUM_THREADS=str(a.threads),MKL_NUM_THREADS=str(a.threads),OPENBLAS_NUM_THREADS=str(a.threads),
        MPLBACKEND='Agg')
    return env


def build(a,mode,preset,out):
    flags=[x for x in PRESETS[preset] if x!='--waffle-compile-dynamics']
    if mode['ddp'] and (preset.startswith('graph') or preset=='compile'):
        raise ValueError('Existing graph/compile learner is single-device only')
    cmd=[sys.executable,'train.py','--waffle-pi','--waffle-native-action','--waffle-gazebo',
        '--waffle-gazebo-model',a.model,'--waffle-gazebo-worker',a.worker,
        '--waffle-fidelity-profile',a.profile,'--waffle-detector-weights',a.detector,
        '--waffle-gazebo-detector-device','learner' if a.detector_device=='cuda' else 'cpu',
        '--waffle-init-cartesian','--resume','--load-path',a.checkpoint,'--fast-training',
        '--num-processes',str(mode['envs']),'--num-mini-batch',str(mode['minibatches']),
        '--num-steps',str(a.steps),'--seq_length',str(a.steps),'--ppo-epoch',str(a.epochs),
        '--num-threads',str(a.threads),'--waffle-humans',str(a.humans),
        '--benchmark-updates',str(a.updates),'--benchmark-warmup',str(a.warmup),
        '--seed',str(a.seed),'--output_dir',str(out),'--waffle-gazebo-logs',str(out.parent/'worker-logs')]
    if a.safety:
        cmd+=['--waffle-safety-calibration',a.safety]
    if a.require_measured:
        cmd+=['--waffle-require-measured']
    if mode['ddp']:
        cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2']+cmd[1:]
        if a.bucket_view:
            cmd+=['--ddp-bucket-view']
        cmd+=['--ddp-bucket-mb',str(a.bucket_mb)]
    return cmd+flags


def run(cmd,env,out,timeout):
    started=time.perf_counter()
    try:
        with out.with_suffix('.log').open('w') as log:
            subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
        result=json.loads((out/'benchmark.json').read_text())
        result['wall_s']=time.perf_counter()-started
        return dict(status='ok',**result)
    except (OSError,ValueError,subprocess.SubprocessError) as exc:
        return dict(status='failed',error=str(exc),wall_s=time.perf_counter()-started)


def summary(rows):
    groups={}
    for row in rows:
        if row['status']=='ok':
            groups.setdefault((row['mode'],row['preset'],row['global_envs']),[]).append(row)
    result=[]
    for (mode,preset,envs),values in groups.items():
        result.append(dict(mode=mode,preset=preset,global_envs=envs,repeats=len(values),
            policy_transitions_per_sec=statistics.median(v['policy_transitions_per_sec'] for v in values),
            aggregate_transitions_per_sec=statistics.median(v['aggregate_transitions_per_sec'] for v in values),
            wall_transitions_per_sec=statistics.median(v['wall_transitions_per_sec'] for v in values)))
    single=[r for r in result if r['mode'] in ('single-all-cpus','single-local','single-local-second','ddp')]
    best=max(single,key=lambda r:r['policy_transitions_per_sec'],default=None)
    independent=max((r for r in result if r['mode']=='independent'),key=lambda r:r['aggregate_transitions_per_sec'],default=None)
    tradeoffs=[]
    for preset,envs in sorted({(r['preset'],r['global_envs']) for r in result}):
        choices=[r for r in result if r['preset']==preset and r['global_envs']==envs]
        solo=max((r for r in choices if r['mode'].startswith('single')),key=lambda r:r['policy_transitions_per_sec'],default=None)
        ddp=next((r for r in choices if r['mode']=='ddp'),None)
        duo=next((r for r in choices if r['mode']=='independent'),None)
        if solo:
            s=solo['policy_transitions_per_sec']
            tradeoffs.append(dict(preset=preset,global_envs=envs,best_single=solo['mode'],
                single_hours_20m=20e6/s/3600,
                ddp_speedup=ddp['policy_transitions_per_sec']/s if ddp else None,
                ddp_hours_20m=20e6/ddp['policy_transitions_per_sec']/3600 if ddp else None,
                independent_search_throughput_ratio=duo['aggregate_transitions_per_sec']/s if duo else None,
                independent_slowest_policy_hours_20m=20e6/duo['policy_transitions_per_sec']/3600 if duo else None))
    return dict(configurations=result,matched_batch_tradeoffs=tradeoffs,fastest_single_policy=best,fastest_independent_search=independent,
                note='Throughput only. Select by independent validation time-to-target before final training; no assumed equal sample efficiency across batch sizes.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--gpus',type=int,nargs=2,default=[0,1])
    p.add_argument('--worlds',type=int,nargs='+',default=[8,16,32,48,64])
    p.add_argument('--minibatches',type=int,default=4)
    p.add_argument('--model',required=True)
    p.add_argument('--worker',default='build/gazebo/waffle_gazebo_worker')
    p.add_argument('--profile',required=True)
    p.add_argument('--detector',required=True)
    p.add_argument('--safety')
    p.add_argument('--checkpoint',default='trained_models/GST_predictor_rand/checkpoints/41665.pt')
    p.add_argument('--presets',choices=list(PRESETS),nargs='+',default=['eager','scan'])
    p.add_argument('--steps',type=int,default=30)
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--updates',type=int,default=5)
    p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--humans',type=int,default=4)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--worker-cores',type=int,default=2)
    p.add_argument('--reserve-cores',type=int,default=2)
    p.add_argument('--numa-memory',action='store_true')
    p.add_argument('--transport',choices=['json','shm'],default='shm')
    p.add_argument('--batch-detector',action='store_true')
    p.add_argument('--detector-batch',type=int,default=8)
    p.add_argument('--detector-wait-ms',type=float,default=.5)
    p.add_argument('--detector-device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--pinned',action='store_true')
    p.add_argument('--bucket-view',action='store_true')
    p.add_argument('--bucket-mb',type=float,default=25.)
    p.add_argument('--require-measured',action='store_true')
    p.add_argument('--seed',type=int,default=425)
    p.add_argument('--repeat',type=int,default=2)
    p.add_argument('--timeout',type=float,default=7200)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if len(set(a.gpus))!=2 or min(a.gpus)<0 or min(a.minibatches,a.steps,a.epochs,a.updates,a.threads,a.worker_cores,a.repeat,*a.worlds)<1:
        p.error('Invalid sweep configuration')
    if a.require_measured and (a.batch_detector or any(x not in ('eager','scan') for x in a.presets)):
        p.error('Measured reference uses unbatched detector and eager FP32; compare separately')
    info=inventory()
    if any(g not in info['gpus'] for g in a.gpus):
        p.error('Requested GPUs absent')
    root=Path(a.output).resolve()
    root.mkdir(parents=True,exist_ok=False)
    plans=[(w,m) for w in a.worlds for m in modes(info,a.gpus,w,a.minibatches)]
    local=allocate(info,a.gpus)
    plan=dict(inventory=info,settings=vars(a),plans=[dict(global_envs=w,**m) for w,m in plans],
        candidate_cpusets={str(g):format_cpus(c) for g,c in local.items()},
        candidate_gpu_uuids={str(g):info['gpus'][g]['uuid'] for g in a.gpus})
    (root/'plan.json').write_text(json.dumps(plan,indent=2))
    if not a.execute:
        print('Plan written. Use --execute only inside your allocated resources.')
        return
    rows=[]
    for w,mode in plans:
        for preset in a.presets:
            if mode['ddp'] and (preset.startswith('graph') or preset=='compile'):
                continue
            for rep in range(a.repeat):
                label=f"{mode['name']}_{w}_{preset}_{rep}"
                out=root/label
                start=time.perf_counter()
                if mode['name']=='independent':
                    jobs=[]
                    for g,c in zip(mode['devices'],mode['cpus']):
                        dest=root/(label+'_'+str(g))
                        jobs.append((build(a,mode,preset,dest),environment(info,[g],[c],a),dest,a.timeout))
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        results=list(pool.map(lambda x:run(*x),jobs))
                else:
                    results=[run(build(a,mode,preset,out),environment(info,mode['devices'],mode['cpus'],a),out,a.timeout)]
                row=dict(mode=mode['name'],preset=preset,global_envs=w,repeat=rep,results=results,
                         status='ok' if all(r['status']=='ok' for r in results) else 'failed')
                elapsed=time.perf_counter()-start
                if row['status']=='ok':
                    rates=[r['transitions_per_sec'] for r in results]
                    row.update(policy_transitions_per_sec=min(rates),aggregate_transitions_per_sec=sum(rates),
                        wall_transitions_per_sec=sum(r['global_transitions'] for r in results)/elapsed)
                rows.append(row)
                (root/'results.json').write_text(json.dumps(rows,indent=2))
                (root/'ranking.json').write_text(json.dumps(summary(rows),indent=2))
                print(label,row['status'],row.get('aggregate_transitions_per_sec'),flush=True)


if __name__=='__main__':
    main()
