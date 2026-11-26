"""Real released GST + navigation checkpoint integration test on CPU or CUDA.

No RVO2, Baselines, ROS or physical motor commands. Runs real PPO updates
through both the live and replay tensor backend. No synthetic neural model.
"""
import argparse
import json
from pathlib import Path
import tempfile
import time
import numpy as np
import torch

from arguments import get_args
from crowd_nav.configs.config import Config
from rl.networks.model import Policy
from rl.networks.storage import RolloutStorage
from rl.ppo.ppo import PPO
from waffle_pi.replay import make_replay
from waffle_pi.checkpoints import load_weights


def finite_tree(tree):
    return all(torch.isfinite(t).all().item() for t in tree.values())


def run(device,backend,updates,checkpoint,seed=425,gae='scan'):
    argv=['--waffle-pi','--num-processes','4','--num-mini-batch','2',
          '--num-steps','4','--seq_length','4','--ppo-epoch','1',
          '--waffle-humans','4','--waffle-timeout','20',
          '--waffle-pool-count','32','--fast-training','--masked-gru',
          '--seed',str(seed)]
    if backend=='live':
        argv+=['--waffle-live-orca']
    with tempfile.TemporaryDirectory() as tmp:
        if backend=='replay':
            # Known fixture tests replay integration without native RVO2.
            # Separate native pool tests are required for ORCA data parity.
            path=Path(tmp)/'fixture.npz'
            starts=np.array([[0.,0.,0.],[-1.,0.,0.]],np.float32)
            goals=np.array([[1.6,0.],[1.6,0.]],np.float32)
            positions=np.array([[1.,1.],[-1.,1.],[-1.,-1.],[1.,-1.]],np.float32)
            frame=np.concatenate((positions,np.full((4,1),.3,np.float32)),1)
            humans=np.broadcast_to(frame,(2,81,4,3)).copy()
            meta=dict(dt=.25,timeout=20.,arena=2.5,humans=4,radius=.25)
            np.savez(path,starts=starts,goals=goals,humans=humans,metadata=json.dumps(meta))
            argv+=['--waffle-replay',str(path)]
        if device.type=='cpu':
            argv+=['--no-cuda']
        args=get_args(argv)
        cfg=Config(args)
        cfg.training.device=str(device)
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        env=make_replay(cfg,args,device)
        try:
            model=Policy(env.observation_space.spaces,env.action_space,
                         base=cfg.robot.policy,base_kwargs=args).to(device)
            model.load_state_dict(load_weights(checkpoint,device),strict=True)
            initial=model.dist.fc_mean.weight.detach().clone()
            opt=PPO(model,args.clip_param,args.ppo_epoch,args.num_mini_batch,
                    args.value_loss_coef,args.entropy_coef,lr=args.lr,eps=args.eps,
                    max_grad_norm=args.max_grad_norm)
            buf=RolloutStorage(args.num_steps,args.num_processes,
                               env.observation_space.spaces,env.action_space,
                               args.human_node_rnn_size,args.human_human_edge_rnn_size,
                               compact_edges=True)
            buf.to(device)
            obs=env.reset()
            assert finite_tree(obs), 'Nonfinite reset observation'
            for k in obs:
                buf.obs[k][0].copy_(obs[k])
            buf.masks[0].zero_()
            start=time.perf_counter()
            losses=[]
            for update in range(updates):
                for step in range(args.num_steps):
                    o={k:v[step] for k,v in buf.obs.items()}
                    h={k:v[step] for k,v in buf.recurrent_hidden_states.items()}
                    with torch.no_grad():
                        value,action,logp,hidden=model.act(o,h,buf.masks[step])
                        assert torch.isfinite(action).all().item()
                        # PPO must evaluate the same raw sampled actions.
                        obs,reward,done,_=env.step(action)
                    assert finite_tree(obs) and torch.isfinite(reward).all().item()
                    mask=(~done).float().unsqueeze(-1)
                    buf.insert(obs,hidden,action,logp,value,reward,mask,torch.ones_like(mask))
                with torch.no_grad():
                    next_value=model.get_value({k:v[-1] for k,v in buf.obs.items()},
                        {k:v[-1] for k,v in buf.recurrent_hidden_states.items()},buf.masks[-1])
                    buf.compute_returns(next_value,True,args.gamma,args.gae_lambda,False,gae)
                # Test finite gradients/optimizer output via the real learner.
                losses.append(opt.update(buf))
                assert all(np.isfinite(x) for x in losses[-1])
                assert all(torch.isfinite(p).all().item() for p in model.parameters())
                buf.after_update()
            assert not torch.equal(initial,model.dist.fc_mean.weight), 'Actor weights did not update'
            # Exercise serialization and strict reload in the same architecture.
            saved=Path(tmp)/'policy.pt'
            torch.save(model.state_dict(),saved)
            model.load_state_dict(load_weights(saved,device),strict=True)
            if device.type=='cuda':
                torch.cuda.synchronize(device)
            result=dict(backend=backend,device=str(device),updates=updates,
                        transitions=updates*args.num_steps*args.num_processes,
                        elapsed_s=time.perf_counter()-start,losses=losses,
                        checkpoint_roundtrip=True)
            return result
        finally:
            env.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='cpu')
    p.add_argument('--backend',choices=['live','replay','both'],default='both')
    p.add_argument('--updates',type=int,default=2)
    p.add_argument('--gae',choices=['reference','scan','triton'],default='scan')
    p.add_argument('--checkpoint',default='trained_models/GST_predictor_rand/checkpoints/41665.pt')
    p.add_argument('--output',default=None)
    a=p.parse_args()
    if a.updates<1:
        p.error('updates must be positive')
    result=[run(torch.device(a.device),b,a.updates,a.checkpoint,gae=a.gae)
            for b in (['live','replay'] if a.backend=='both' else [a.backend])]
    print(json.dumps(result,indent=2))
    if a.output:
        Path(a.output).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
