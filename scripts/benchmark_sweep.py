"""Sequential isolated end-to-end PPO throughput sweep, not reward evaluation."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time


PRESETS={
    'eager':[],
    'scan':['--gae-backend','scan','--masked-gru'],
    'triton':['--gae-backend','triton','--masked-gru','--fused-adam'],
    'compile':['--compile-policy','--compile-gst','--waffle-compile-dynamics',
               '--compile-mode','reduce-overhead','--gae-backend','scan','--fused-adam'],
    'graph':['--cuda-graph-actor','--cuda-graph-gst','--cuda-graph-ppo',
             '--gae-backend','triton','--fused-adam'],
    'graph-bf16':['--cuda-graph-actor','--cuda-graph-gst','--cuda-graph-ppo',
                  '--gae-backend','triton','--fused-adam','--amp-bf16','--tf32'],
}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--backend',choices=['live','replay'],default='live')
    p.add_argument('--pool')
    p.add_argument('--checkpoint',default='trained_models/GST_predictor_rand/checkpoints/41665.pt')
    p.add_argument('--envs',type=int,nargs='+',default=[16,64,256])
    p.add_argument('--minibatches',type=int,nargs='+',default=[4,16])
    p.add_argument('--presets',choices=list(PRESETS),nargs='+',default=['eager','scan'])
    p.add_argument('--updates',type=int,default=5)
    p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--steps',type=int,default=30)
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--humans',type=int,default=4)
    p.add_argument('--seed',type=int,default=425)
    p.add_argument('--timeout-s',type=int,default=1800)
    p.add_argument('--repeat',type=int,default=1)
    a=p.parse_args()
    if (not all(x>0 for x in [*a.envs,*a.minibatches,a.updates,a.steps,a.epochs,a.humans,a.timeout_s,a.repeat])
            or a.warmup<0 or (a.backend=='replay' and not a.pool)
            or (a.device=='cpu' and any(s not in ('eager','scan') for s in a.presets))):
        p.error('Invalid sweep configuration or CUDA-only preset on CPU')
    root=Path(a.output).resolve()
    root.mkdir(parents=True,exist_ok=False)
    common=[sys.executable,'train.py','--waffle-pi','--resume','--fast-training',
            '--load-path',a.checkpoint,'--num-steps',str(a.steps),'--seq_length',str(a.steps),
            '--ppo-epoch',str(a.epochs),'--waffle-humans',str(a.humans),
            '--benchmark-updates',str(a.updates),'--benchmark-warmup',str(a.warmup)]
    if a.backend=='live':
        common+=['--waffle-live-orca']
    else:
        common+=['--waffle-replay',a.pool]
    if a.device=='cpu':
        common+=['--no-cuda']
    rows=[]
    for n in dict.fromkeys(a.envs):
        for m in dict.fromkeys(a.minibatches):
            if m>n or n%m:
                continue
            for preset in dict.fromkeys(a.presets):
                for rep in range(a.repeat):
                    label=f'{preset}_n{n}_m{m}_r{rep}'
                    out=root/label
                    cmd=common+PRESETS[preset]+['--seed',str(a.seed+rep),
                        '--num-processes',str(n),'--num-mini-batch',str(m),
                        '--output_dir',str(out)]
                    row=dict(preset=preset,envs=n,minibatches=m,repeat=rep,
                             seed=a.seed+rep,status='failed',transitions_per_sec=None)
                    start=time.perf_counter()
                    try:
                        with (root/(label+'.log')).open('w') as log:
                            run=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout_s)
                        row['returncode']=run.returncode
                        if run.returncode==0:
                            result=json.loads((out/'benchmark.json').read_text())
                            row.update(result,status='ok')
                    except subprocess.TimeoutExpired:
                        row['status']='timeout'
                    except (OSError,ValueError) as exc:
                        row['error']=str(exc)
                    row['wall_s']=time.perf_counter()-start
                    row['command']=cmd
                    rows.append(row)
                    (root/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
                    print(label,row['status'],row['transitions_per_sec'],flush=True)
    ok=sorted((r for r in rows if r['status']=='ok'),key=lambda r:r['transitions_per_sec'],reverse=True)
    fields=['preset','envs','minibatches','repeat','seed','status','transitions_per_sec',
            'global_transitions','rollout_s','gae_and_bootstrap_s','learner_s','wall_s',
            'peak_allocated_bytes','peak_reserved_bytes','gpu_name']
    with (root/'ranking.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(ok)
    if not ok:
        raise SystemExit('No successful configurations; inspect logs')
    print('Highest measured throughput (not quality):',ok[0]['preset'],ok[0]['envs'],
          ok[0]['minibatches'],ok[0]['transitions_per_sec'])


if __name__=='__main__':
    main()
