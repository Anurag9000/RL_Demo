"""Versioned learner continuation at PPO update boundaries.

Restores model, all Adam slots, optimizer groups, counters and learner RNG.
Environments and recurrent rollout histories intentionally restart; no stale
on-policy samples are reused. This is not bitwise simulator continuation.
"""
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from .checkpoints import load_weights
from .native import profile_contract

SCHEMA='waffle_learner_v1'
HYPERS=('lr','entropy_coef','clip_param')
FIXED=('gamma','gae_lambda','value_loss_coef','max_grad_norm','eps','ppo_epoch',
       'use_gae','use_proper_time_limits','freeze_human_attention',
       'human_node_rnn_size','human_human_edge_rnn_size','adaptive_kl','lr_min','lr_max')
SCENARIO=('waffle_arena','waffle_timeout','waffle_noise','waffle_dropout',
          'waffle_occlusion','waffle_pose_drift','waffle_detection_delay',
          'waffle_dynamics_randomization','waffle_reactive_humans','waffle_actuation_delay',
          'waffle_motor_tau','waffle_wheel_accel','waffle_live_orca','waffle_replay',
          'waffle_calibration','waffle_safety_calibration')


def parameter_names(model):
    return [name for name,p in model.named_parameters() if p.requires_grad]


def rng_state(device):
    n=np.random.get_state()
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device) if device.type=='cuda' else None,
                python=random.getstate(),numpy=[n[0],n[1].tolist(),n[2],n[3],n[4]])


def restore_rng(state,device):
    torch.set_rng_state(state['torch'].cpu())
    if device.type=='cuda' and state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'].cpu(),device)
    random.setstate(state['python'])
    n=state['numpy']
    np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))


def atomic_save(obj,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    torch.save(obj,tmp)
    os.replace(tmp,path)


def save(path,model,agent,args,updates,transitions,device):
    local_rng=rng_state(device)
    distributed=torch.distributed.is_initialized()
    if distributed:
        rank=torch.distributed.get_rank()
        states=[None]*torch.distributed.get_world_size() if rank==0 else None
        torch.distributed.gather_object(local_rng,states,dst=0)
        if rank!=0:
            return
    else:
        states=[local_rng]
    atomic_save(dict(schema=SCHEMA,model=model.state_dict(),optimizer=agent.optimizer.state_dict(),
        parameter_names=parameter_names(model),args=dict(vars(args)),updates=int(updates),
        transitions=int(transitions),rng=local_rng,rank_rng=states,world_size=len(states),
        hyperparameters=dict(lr=agent.optimizer.param_groups[0]['lr'],
            entropy_coef=agent.entropy_coef,clip_param=agent.clip_param)),path)


def load(path,model,agent,args,device):
    state=load_weights(path,'cpu')
    if state.get('schema')!=SCHEMA:
        raise ValueError('Expected versioned learner state, not inference weights')
    old=state['args']
    if bool(old.get('waffle_native_action'))!=args.waffle_native_action:
        raise ValueError('Action semantics changed')
    for key,value in profile_contract(args).items():
        if key=='waffle_humans' and args.curriculum_transition:
            continue
        if old.get(key)!=value:
            raise ValueError('Continuation profile mismatch: '+key)
    for key in FIXED:
        if old.get(key)!=getattr(args,key):
            raise ValueError('Continuation learner mismatch: '+key)
    if not args.curriculum_transition:
        for key in SCENARIO:
            if old.get(key)!=getattr(args,key):
                raise ValueError('Continuation scenario mismatch: '+key)
    if state['parameter_names']!=parameter_names(model):
        raise ValueError('Trainable parameter ordering changed')
    model.load_state_dict(state['model'],strict=True)
    agent.optimizer.load_state_dict(state['optimizer'])
    # Preserve current device-specific optimizer implementation options.
    for group in agent.optimizer.param_groups:
        group['fused']=True if args.fused_adam else None
        group['capturable']=False
    values=state['hyperparameters']
    if args.retune_optimizer:
        values={k:getattr(args,k) for k in HYPERS}
    for group in agent.optimizer.param_groups:
        group['lr']=float(values['lr'])
    agent.entropy_coef=float(values['entropy_coef'])
    agent.clip_param=float(values['clip_param'])
    if not args.fresh_continuation_rng:
        rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world=torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        if state.get('world_size',1)!=world:
            raise ValueError('Changed world size requires --fresh-continuation-rng')
        restore_rng(state.get('rank_rng',[state['rng']])[rank],device)
    return int(state['updates']),int(state['transitions'])
