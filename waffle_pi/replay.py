"""Experimental tensor-resident Waffle Pi crowd replay backend."""
import json
import gym
import numpy as np
import torch
from rl.vec_env import VecEnv
from rl.vec_env.vec_pretext_normalize import VecPretextNormalize
from .batched import step


class ReplayVecEnv(VecEnv):
    def __init__(self, config, args, device):
        if not getattr(config, 'waffle_pi', False) or config.robot.visible or config.sim.render:
            raise ValueError("Replay requires Waffle mode, invisible robot and no rendering")
        if args.use_proper_time_limits:
            raise ValueError("Replay currently treats episode horizon as terminal; disable proper-time-limits")
        self.cfg, self.device, self.n = config, device, args.num_processes
        self.h = config.sim.human_num
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
        self.horizon = self.humans.shape[1]-1
        self.rng = torch.Generator(device=device).manual_seed(args.seed+int(__import__('os').environ.get('RANK','0'))*100003)
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

    def _observe(self):
        humans = self.humans[self.ids,self.t]
        rel = humans[...,:2]-self.pose[:,None,:2]
        visible = (torch.linalg.vector_norm(rel,dim=-1)-humans[...,2]-self.cfg.robot.radius
                   <= self.cfg.robot.sensor_range)
        rel = torch.where(visible[...,None],rel,15.)
        radius = torch.full_like(self.pose[:,:1], self.cfg.robot.radius)
        speed = torch.full_like(radius,self.cfg.robot.v_pref)
        node = torch.cat((self.pose[:,:2],radius,self.goal,speed,self.pose[:,2:]),-1)
        return dict(robot_node=node[:,None,:], temporal_edges=self.velocity[:,None,:],
                    spatial_edges=rel.repeat(1,1,6), visible_masks=visible,
                    detected_human_num=visible.sum(-1,keepdim=True).clamp_min(1).float())

    def reset(self):
        self._reset(torch.ones(self.n,device=self.device,dtype=torch.bool))
        return self._observe()

    def step_async(self,actions):
        self.actions = actions

    def step_wait(self):
        old = self.pose
        old_dist = torch.linalg.vector_norm(self.goal-old[:,:2],dim=-1)
        self.pose,self.wheels,self.velocity,dtheta = self.motion(
            self.pose,self.wheels,self.actions,self.cfg.waffle_limits,self.cfg.env.time_step)
        self.t = (self.t+1).clamp(max=self.horizon)
        human = self.humans[self.ids,self.t]
        d = torch.linalg.vector_norm(human[...,:2]-self.pose[:,None,:2],dim=-1)
        clearance = (d-human[...,2]-self.cfg.robot.radius).amin(-1)
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
        self._reset(done)
        return self._observe(),r[:,None],done,None

    def talk2Env_async(self,data):
        pass

    def talk2Env_wait(self):
        return [True]*self.n


def make_replay(config,args,device):
    env = ReplayVecEnv(config,args,device)
    return VecPretextNormalize(env, config=config, ret=False, ob=False, gamma=args.gamma, test=False)
