"""Measure actual local inter-GPU all-reduce latency under torchrun."""
import argparse
import json
import statistics
import os
import torch
import torch.distributed as dist
from gazebo_sim.hardware import bind_rank


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bytes',type=int,nargs='+',default=[4096,1048576,4194304,16777216,67108864])
    p.add_argument('--repeat',type=int,default=50)
    a=p.parse_args()
    if a.repeat<1 or any(n<4 or n%4 for n in a.bytes):
        p.error('FP32 payload sizes must be positive multiples of four')
    bind_rank()
    rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    try:
        world=dist.get_world_size()
        for n in a.bytes:
            x=torch.zeros(n//4,device='cuda')
            for _ in range(10):
                dist.all_reduce(x)
            torch.cuda.synchronize()
            times=[]
            for _ in range(a.repeat):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                dist.all_reduce(x)
                end.record()
                end.synchronize()
                ms=torch.tensor(start.elapsed_time(end),device='cuda')
                dist.all_reduce(ms,op=dist.ReduceOp.MAX)
                times.append(float(ms))
            if dist.get_rank()==0:
                t=statistics.median(times)
                print(json.dumps(dict(bytes=n,median_ms=t,algorithm_GB_s=n/(t*1e6),
                  ring_bus_equivalent_GB_s=n/(t*1e6)*2*(world-1)/world,
                  ranks=world)),flush=True)
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    main()
