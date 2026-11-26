"""torchrun process-per-device setup; no automatic process spawning."""
import os
import torch


def setup(args):
    world=int(os.environ.get('WORLD_SIZE','1'))
    rank=int(os.environ.get('RANK','0'))
    local=int(os.environ.get('LOCAL_RANK','0'))
    if args.cuda:
        torch.cuda.set_device(local)
        device=torch.device('cuda',local)
    else:
        device=torch.device('cpu')
    if world>1:
        import torch.distributed as dist
        dist.init_process_group(backend='nccl' if args.cuda else 'gloo')
    return rank,world,device
