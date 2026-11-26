"""Experimental tensor-resident Waffle Pi crowd replay backend."""
import json
import math
import os
import gym
import numpy as np
import torch
from rl.vec_env import VecEnv
from rl.vec_env.vec_pretext_normalize import VecPretextNormalize
from .batched import step
from .perception import Perception


class ReplayVecEnv(VecEnv):
    def __init__(self, config, args, device):
        if not getattr(config, 'waffle_pi', False) or config.robot.visible or config.sim.render:
            raise ValueError("Replay requires Waffle mode, invisible robot and no rendering")
        if args.use_proper_time_limits:
            raise ValueError("Replay currently treats episode horizon as terminal; disable proper-time-limits")
        self.cfg, self.device, self.n = config, device, args.num_processes
        self.h = config.sim.human_num
        self.live = args.waffle_live_orca
        self.args = args
        self.rng = torch.Generator(device=device).manual_seed(args.seed+int(os.environ.get('RANK','0'))*100003)
        if self.live:
            self._make_live_bank()
        else:
            self._load_pool(args)
        self.horizon = int(round(config.env.time_limit/config.env.time_step))
        if not self.live and self.humans.shape[1]!=self.horizon+1:
            raise ValueError("Replay trajectory length does not match configured horizon")
        self.perception = Perception(args,self.n,self.h,device,self.rng)
        self.slot_reset = torch.zeros(self.n,self.h,device=device,dtype=torch.bool)
        self.outcome = torch.zeros(self.n,device=device,dtype=torch.long)
        self.orca = None
        if self.live:
            from .orca_tensor import orca_velocity
            self.orca = orca_velocity
            if args.waffle_compile_dynamics:
                self.orca = torch.compile(self.orca,fullgraph=True)
        self._setup_spaces_and_state(config,args,device)

    def _load_pool(self,args):
        config,device=self.cfg,self.device
        with np.load(args.waffle_replay, allow_pickle=False) as pool:
            meta = json.loads(str(pool['metadata']))
            expected = dict(dt=config.env.time_step, timeout=config.env.time_limit,
                            arena=config.sim.arena_size, humans=self.h, radius=config.robot.radius)
            if any(meta.get(k) != v for k,v in expected.items()):
                raise ValueError("Pool geometry/timing/footprint differs from training profile")
            self.starts = torch.as_tensor(pool['starts'], device=device, dtype=torch.float32)
            self.goals = torch.as_tensor(pool['goals'], device=device, dtype=torch.float32)
            self.humans = torch.as_tensor(pool['humans'], device=device, dtype=torch.float32)
        count = self.starts.shape[0]
        if (count == 0 or self.starts.shape != (count,3) or self.goals.shape != (count,2)
                or self.humans.ndim != 4 or self.humans.shape[0] != count
                or self.humans.shape[2:] != (self.h,3) or self.humans.shape[1] < 2
                or not all(torch.isfinite(t).all().item() for t in (self.starts,self.goals,self.humans))):
            raise ValueError("Invalid trajectory pool")
    def _make_live_bank(self):
        # Generate a finite bank of valid starts on GPU. Unlike replay, human
        # states are updated online, with stable identities and no respawns.
        # Deliberately different scenario distribution from upstream.
        cfg,dev=self.cfg,self.device
        k=max(32,self.args.waffle_pool_count)
        ring=cfg.sim.arena_size-.45
        robot_ring=min(1.65,ring-.35-cfg.robot.radius-.12)
        if robot_ring<.75 or (self.h>1 and 2*ring*math.sin(math.pi/self.h)*.85<.70):
            raise ValueError("Arena too small for nonoverlapping tensor ORCA starts")
        phase=torch.rand(k,1,device=dev,generator=self.rng)*2*math.pi
        slots=torch.arange(self.h,device=dev)[None]*2*math.pi/self.h
        jitter=(torch.rand(k,self.h,device=dev,generator=self.rng)-.5)*.15*2*math.pi/self.h
        angles=phase+slots+jitter
        hp=ring*torch.stack((angles.cos(),angles.sin()),-1)
        self.bank_radius=.25+.10*torch.rand(k,self.h,device=dev,generator=self.rng)
        self.bank_speed=.5+torch.rand(k,self.h,device=dev,generator=self.rng)
        self.bank_pos=hp
        theta=torch.rand(k,device=dev,generator=self.rng)*2*math.pi
        rp=robot_ring*torch.stack((theta.cos(),theta.sin()),-1)
        heading=torch.rand(k,1,device=dev,generator=self.rng)*2*math.pi
        self.starts=torch.cat((rp,heading),-1)
        self.goals=-rp
        self.human_pos=torch.zeros(self.n,self.h,2,device=dev)
        self.human_vel=torch.zeros_like(self.human_pos)
        self.human_goal=torch.zeros_like(self.human_pos)
        self.human_start=torch.zeros_like(self.human_pos)
        self.human_radius=torch.zeros(self.n,self.h,device=dev)
        self.human_speed=torch.zeros_like(self.human_radius)

    def _setup_spaces_and_state(self,config,args,device):
        def box(shape, dtype=np.float32):
            return gym.spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=dtype)
        space = gym.spaces.Dict({
            'robot_node': box((1,7)), 'temporal_edges': box((1,2)),
            'spatial_edges': box((self.h,12)),
            'visible_masks': gym.spaces.Box(0,1,shape=(self.h,),dtype=np.bool_),
            'detected_human_num': box((1,))})
        super().__init__(self.n, space, box((2,)))
        self.motion = step
        if args.waffle_compile_dynamics:
            if not hasattr(torch, 'compile'):
                raise RuntimeError("Compiled dynamics requires PyTorch 2.x")
            self.motion = torch.compile(step, fullgraph=True)
        self.ids = torch.zeros(self.n, device=device, dtype=torch.long)
        self.t = torch.zeros_like(self.ids)
        self.pose = torch.zeros(self.n,3,device=device)
        self.goal = torch.zeros(self.n,2,device=device)
        self.wheels = torch.zeros_like(self.goal)
        self.velocity = torch.zeros_like(self.goal)

    def _reset(self, done):
        ids = torch.randint(len(self.starts),(self.n,),device=self.device,generator=self.rng)
        self.ids = torch.where(done,ids,self.ids)
        self.t = torch.where(done,0,self.t)
        self.pose = torch.where(done[:,None],self.starts[self.ids],self.pose)
        self.goal = torch.where(done[:,None],self.goals[self.ids],self.goal)
        self.wheels = torch.where(done[:,None],0.,self.wheels)
        self.velocity = torch.where(done[:,None],0.,self.velocity)
        self.perception.reset(done)
        self.slot_reset = self.slot_reset | done[:,None]
        if self.live:
            mask=done[:,None,None]
            self.human_pos=torch.where(mask,self.bank_pos[self.ids],self.human_pos)
            self.human_start=torch.where(mask,self.bank_pos[self.ids],self.human_start)
            self.human_goal=torch.where(mask,-self.bank_pos[self.ids],self.human_goal)
            self.human_vel=torch.where(mask,0.,self.human_vel)
            self.human_radius=torch.where(done[:,None],self.bank_radius[self.ids],self.human_radius)
            self.human_speed=torch.where(done[:,None],self.bank_speed[self.ids],self.human_speed)

    def _frame(self):
        if self.live:
            return torch.cat((self.human_pos,self.human_radius[...,None]),-1)
        return self.humans[self.ids,self.t]

    def _advance_humans(self,old,old_velocity):
        if not self.live:
            return
        cfg=self.cfg
        pos,vel,rad=self.human_pos,self.human_vel,self.human_radius
        # Stable out-and-back goals instead of robot-dependent teleportation.
        reached=torch.linalg.vector_norm(self.human_goal-pos,dim=-1)<rad
        self.human_goal=torch.where(reached[...,None],-self.human_goal,self.human_goal)
        delta=self.human_goal-pos
        preferred=delta/torch.linalg.vector_norm(delta,dim=-1,keepdim=True).clamp_min(1.)
        speed=self.human_speed
        if self.args.waffle_reactive_humans:
            pos=torch.cat((pos,old[:,None,:2]),1)
            vel=torch.cat((vel,old_velocity[:,None,:]),1)
            rad=torch.cat((rad,torch.full_like(rad[:,:1],cfg.robot.radius)),1)
            preferred=torch.cat((preferred,old_velocity[:,None,:]),1)
            speed=torch.cat((speed,torch.full_like(speed[:,:1],cfg.robot.v_pref)),1)
        self.human_vel=self.orca(pos,vel,rad,preferred,speed,dt=cfg.env.time_step,
                        horizon=cfg.orca.time_horizon,neighbor_dist=cfg.orca.neighbor_dist,
                        margin=.01+cfg.orca.safety_space)[:,:self.h]
        new=self.human_pos+self.human_vel*cfg.env.time_step
        bound=cfg.sim.arena_size-self.human_radius[...,None]
        new=torch.maximum(-bound,torch.minimum(bound,new))
        self.human_vel=(new-self.human_pos)/cfg.env.time_step
        self.human_pos=new

    def _observe(self):
        humans = self._frame()
        estimated,rel,visible = self.perception.observe(
            self.pose,humans,self.cfg.waffle_limits,self.cfg.robot.radius,self.cfg.robot.sensor_range)
        radius = torch.full_like(self.pose[:,:1], self.cfg.robot.radius)
        speed = torch.full_like(radius,self.cfg.robot.v_pref)
        node = torch.cat((estimated,radius,self.goal,speed,self.pose[:,2:]),-1)
        return dict(robot_node=node[:,None,:], temporal_edges=self.velocity[:,None,:],
                    spatial_edges=rel.repeat(1,1,6), visible_masks=visible,
                    detected_human_num=visible.sum(-1,keepdim=True).clamp_min(1).float(),
                    slot_reset=self.slot_reset.clone())

    def reset(self):
        self._reset(torch.ones(self.n,device=self.device,dtype=torch.bool))
        return self._observe()

    def step_async(self,actions):
        self.actions = actions

    def step_wait(self):
        old = self.pose
        old_velocity = self.velocity
        prev_humans = self._frame()
        old_dist = torch.linalg.vector_norm(self.goal-old[:,:2],dim=-1)
        self.pose,self.wheels,self.velocity,dtheta = self.motion(
            self.pose,self.wheels,self.actions,self.cfg.waffle_limits,self.cfg.env.time_step)
        self.t = (self.t+1).clamp(max=self.horizon)
        self._advance_humans(old,old_velocity)
        human = self._frame()
        # A replay slot may be respawned at a different position.
        displacement=torch.linalg.vector_norm(human[...,:2]-prev_humans[...,:2],dim=-1)
        self.slot_reset=displacement>1.6*self.cfg.env.time_step+.1
        self.perception.history.masked_fill_(self.slot_reset[None,:,:,None],0.)
        self.perception.masks.masked_fill_(self.slot_reset[None],False)
        # Swept relative line segment for robot-human contact; conservative
        # endpoint radius for teleported agents, no fictitious teleport sweep.
        before=prev_humans[...,:2]-old[:,None,:2]
        after=human[...,:2]-self.pose[:,None,:2]
        change=after-before
        t=(-(before*change).sum(-1)/change.square().sum(-1).clamp_min(1e-8)).clamp(0.,1.)
        closest=torch.linalg.vector_norm(before+t[...,None]*change,dim=-1)
        closest=torch.where(self.slot_reset,torch.linalg.vector_norm(after,dim=-1),closest)
        clearance=(closest-torch.maximum(human[...,2],prev_humans[...,2])-self.cfg.robot.radius).amin(-1)
        dist = torch.linalg.vector_norm(self.goal-self.pose[:,:2],dim=-1)
        wall = self.pose[:,:2].abs().amax(-1)+self.cfg.robot.radius >= self.cfg.sim.arena_size
        collision = (clearance < 0.) | wall
        success = (dist < self.cfg.robot.radius) & ~collision
        timeout = (self.t>=self.horizon) & ~collision & ~success
        done = collision | success | timeout
        # Deliberate experimental reward variant: post-transition collision,
        # room boundaries and actual progress. Not upstream reward parity.
        r = 3.*(old_dist-dist)
        r = torch.where(clearance<self.cfg.reward.discomfort_dist,
              (clearance-self.cfg.reward.discomfort_dist)*self.cfg.reward.discomfort_penalty_factor*self.cfg.env.time_step,r)
        r = r-4.5*dtheta.square()
        r = torch.where(timeout,0.,r)
        r = torch.where(success,self.cfg.reward.success_reward,r)
        r = torch.where(collision,self.cfg.reward.collision_penalty,r)
        self.outcome=torch.where(collision,2,torch.where(success,1,torch.where(timeout,3,0)))
        self._reset(done)
        return self._observe(),r[:,None],done,None

    def talk2Env_async(self,data):
        pass

    def talk2Env_wait(self):
        return [True]*self.n


def make_replay(config,args,device):
    env = ReplayVecEnv(config,args,device)
    return VecPretextNormalize(env, config=config, ret=False, ob=False, gamma=args.gamma, test=False)
