
"""Gazebo-backed recurrent PPO environment, using actual simulated LDS scans.

Ground truth is restricted to reward, termination and the pedestrian model.
The policy receives encoder odometry, shared detector and tracker output.
One synchronous batch of on-policy decisions; failures abort, never fabricate
observations or rewards. Physics stepping occurs concurrently across worlds.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import math
import os
from pathlib import Path
import tempfile
import numpy as np
import torch
from rl.vec_env import VecEnv
from rl.vec_env.vec_pretext_normalize import VecPretextNormalize
from waffle_pi.spaces import spaces
from waffle_pi.native import native_twist
from waffle_pi.safety import safe_command,scan_coverage
from waffle_pi.detector_engine import DetectorEngine
from waffle_pi.scan_geometry import ScanGeometry
from waffle_pi.tracking import Tracker,TrajectoryHistory
from waffle_pi.orca_tensor import orca_velocity
from waffle_pi.fidelity import disk_clearance
from waffle_pi.scene import scene_tensors,constrain_human_step
from waffle_pi.quality import navigation_reward
from .world import validate_gazebo_profile,source_digest,write_case
from .worker import Worker


class PerWorld:
    def __init__(self,config,args,device,root,index,worker_factory=Worker,detector_factory=DetectorEngine):
        self.cfg,self.args,self.device=config,args,device
        self.profile=config.fidelity
        self.g=self.profile['gazebo']
        self.laser=self.profile['laser']
        self.root=Path(root)/str(index)
        self.root.mkdir(parents=True,exist_ok=True)
        self.worker_factory=worker_factory
        self.detector_factory=detector_factory
        self.worker=self.detector=None
        self.n=config.sim.human_num
        self.gate_dt=self.g['gate_dt']
        self.physics_iterations=round(self.gate_dt/self.g['physics_dt'])
        self.substeps=round(config.env.time_step/self.gate_dt)
        self.rng=np.random.default_rng(args.seed+100003*int(os.environ.get('RANK','0'))+index)
        self.episode=0
        self.geometry=None

    def close(self):
        if self.worker is not None:
            self.worker.close()
            self.worker=None

    def reset(self,scenario):
        self.close()
        self.episode+=1
        self.scenario=scenario
        self.goal=np.asarray(scenario['robot_goal'],np.float32)
        self.radius=np.asarray([h['radius'] for h in scenario['humans']],np.float32)
        self.speed=np.asarray([h['speed'] for h in scenario['humans']],np.float32)
        self.routes=np.asarray([h['waypoints'] for h in scenario['humans']],np.float32)
        self.route=np.zeros(self.n,dtype=np.int64)
        self.human_velocity=np.zeros((self.n,2),np.float32)
        self.tracker=Tracker(self.n)
        self.history=TrajectoryHistory(self.n,5)
        self.last_stamp=self.used_stamp=-math.inf
        self.scan=None
        self.points=None
        self.fault=True
        self.steps=0
        if self.detector is not None:
            self.detector.reset()
        world,conf=write_case(self.root,self.args.waffle_gazebo_model,self.profile,scenario,int(self.rng.integers(2**31-1)))
        self.worker=self.worker_factory(self.args.waffle_gazebo_worker,world,conf,
                        self.root/f'episode_{self.episode:06d}.log',self.args.waffle_gazebo_timeout)
        # Settle on the floor with zero commands. This is not an RL transition.
        warm=round(max(.5,2/self.laser['hz'])/self.g['physics_dt'])
        self.state=self.worker.step(warm,0.,0.,[[0.,0.]]*self.n,True)
        if self.state['contact'] or not self.state['floor_contact_observed']:
            raise RuntimeError('Initial Gazebo contact qualification failed')
        self.ingest(self.state)
        if self.scan is None:
            raise RuntimeError('No initial laser observation')
        return self.observe()

    def ingest(self,state):
        for key,shape in [('truth',(3,)),('odom',(3,)),('twist',(2,)),('wheels',(2,)),('humans',(self.n,2))]:
            a=np.asarray(state[key])
            if a.shape!=shape or not np.isfinite(a).all():
                raise RuntimeError('Invalid native state '+key)
        for frame in state['scans']:
            stamp=float(frame['stamp'])
            if stamp<=self.last_stamp:
                raise RuntimeError('Nonmonotonic simulator scan')
            self.last_stamp=stamp
            raw=np.array([float('inf') if x is None else x for x in frame['ranges']],np.float32)
            if len(raw)!=self.laser['rays'] or not math.isclose(frame['min'],self.laser['range_min'],abs_tol=1e-5) or not math.isclose(frame['max'],self.laser['range_max'],abs_tol=1e-5):
                raise RuntimeError('Gazebo scan geometry differs from profile')
            if self.laser['drop_probability']:
                raw[self.rng.random(len(raw))<self.laser['drop_probability']]=np.inf
            geometry=ScanGeometry.from_metadata(len(raw),frame['angle_min'],frame['angle_max'],frame['angle_step'])
            if self.geometry is not None and geometry!=self.geometry:
                raise RuntimeError('Unexpected scan geometry change')
            self.geometry=geometry
            if self.detector is None:
                d=self.profile['detector']
                self.detector=self.detector_factory(self.args.waffle_detector_weights,d['model'],
                   self.args.waffle_gazebo_detector_device,d['stride'],d['confidence'],d['max_range'],geometry)
            self.scan=(raw,frame)
            try:
                points=self.detector.detect(raw,frame['min'],frame['max'])
                x,y,heading=frame['pose']
                c,s=math.cos(heading),math.sin(heading)
                sx,sy=self.laser['offset_x'],self.laser['offset_y']
                origin=np.array([x+c*sx-s*sy,y+s*sx+c*sy])
                a=heading+self.laser['yaw']
                ca,sa=math.cos(a),math.sin(a)
                self.points=points@np.array([[ca,sa],[-sa,ca]])+origin
            except (ValueError,RuntimeError):
                self.points=None
                self.detector.reset()

    def observe(self):
        x,y,theta=self.state['odom']
        xy=np.array([x,y],np.float32)
        valid=np.zeros(self.n,bool)
        rel=np.full((self.n,2),15.,np.float32)
        self.fault=(self.points is None or self.last_stamp<=self.used_stamp
                    or self.state['time']-self.last_stamp>.5)
        if not self.fault:
            if self.last_stamp-self.used_stamp>.5:
                self.tracker=Tracker(self.n)
                self.history.clear()
            self.used_stamp=self.last_stamp
            points=self.points[np.linalg.norm(self.points-xy,axis=-1)<=self.cfg.robot.sensor_range]
            reused=self.tracker.update(points,self.last_stamp)
            self.fault=self.tracker.overflow
            if not self.fault:
                r=self.tracker.pos-xy
                valid=self.tracker.seen&(np.linalg.norm(r,axis=-1)<=self.cfg.robot.sensor_range)
                rel[valid]=r[valid]
                self.history.append(self.last_stamp,self.tracker.pos,valid,reused)
        if self.fault:
            self.tracker=Tracker(self.n)
            self.history.clear()
            valid[:]=False
            rel[:]=15.
        hist,mask=self.history.sample(self.last_stamp)
        v=self.state['twist'][0]
        return dict(robot_node=np.asarray([[x,y,self.cfg.robot.radius,*self.goal,self.cfg.robot.v_pref,theta%(2*math.pi)]],np.float32),
                    temporal_edges=np.asarray([[v*math.cos(theta),v*math.sin(theta)]],np.float32),
                    spatial_edges=np.tile(rel,(1,6)),visible_masks=valid,
                    detected_human_num=np.asarray([max(1,int(valid.sum()))],np.float32),
                    gst_history=hist[0],gst_history_mask=mask[0])

    def guard(self,requested):
        raw,f=self.scan
        v,w=requested
        fresh=(not self.fault and 0<=self.state['time']-f['stamp']<=.5
               and scan_coverage(raw,f['angle_step'],f['min'],f['max']))
        mv,mw=self.state['twist']
        cmd=safe_command(v,w,raw,f['min'],f['max'],fresh=fresh,armed=True,estop=False,
                         measured_speed=mv,measured_angular=mw,**self.cfg.waffle_safety)
        blocked=not fresh or (cmd==(0.,0.) and (v!=0. or w!=0.))
        return cmd,blocked

    def step(self,latent,human_velocity):
        requested=native_twist(latent,self.cfg.waffle_limits)
        prev=np.asarray(self.state['truth'])
        before=np.linalg.norm(prev[:2]-self.goal)
        clearance=wall=math.inf
        path=turn=0.
        collision=intervened=False
        last_h=np.asarray(self.state['humans'])
        rect=torch.as_tensor(self.profile['scene']['rectangles_m'],dtype=torch.float32).reshape(-1,4)
        for _ in range(self.substeps):
            cmd,blocked=self.guard(requested)
            intervened|=blocked
            # Human static obstacle rejection supplements contact physics.
            # This does not implement a general obstacle ORCA planner.
            old_h=torch.as_tensor(self.state['humans'],dtype=torch.float32)
            vel=torch.as_tensor(human_velocity,dtype=torch.float32)
            new_h=constrain_human_step(old_h,old_h+vel*self.gate_dt,
                        torch.as_tensor(self.radius),self.cfg.sim.arena_size,rect)
            hv=((new_h-old_h)/self.gate_dt).tolist()
            state=self.worker.step(self.physics_iterations,*cmd,hv,blocked)
            self.state=state
            self.ingest(state)
            p=np.asarray(state['truth'])
            h=np.asarray(state['humans'])
            a=last_h-prev[:2]
            b=h-p[:2]
            change=b-a
            t=np.clip(-(a*change).sum(-1)/np.maximum((change*change).sum(-1),1e-10),0.,1.)
            clearance=min(clearance,float(np.min(np.linalg.norm(a+t[:,None]*change,axis=-1)-self.radius-self.cfg.robot.radius)))
            wall=min(wall,float(disk_clearance(torch.as_tensor(p[:2],dtype=torch.float32),self.cfg.robot.radius,
                                self.cfg.sim.arena_size,rect)))
            collision|=bool(state['contact']) or wall<=0. or clearance<0.
            path+=state['path'];turn+=state['turn']
            prev,last_h=p,h
        self.steps+=1
        self.human_velocity=(last_h-np.asarray(self.previous_humans))/self.cfg.env.time_step
        dist=float(np.linalg.norm(prev[:2]-self.goal))
        success=dist<self.cfg.robot.radius and not collision
        timeout=self.steps>=round(self.cfg.env.time_limit/self.cfg.env.time_step) and not collision and not success
        outcome=2 if collision else 1 if success else 3 if timeout else 0
        reward=3*(before-dist)
        if clearance<self.cfg.reward.discomfort_dist:
            reward=(clearance-self.cfg.reward.discomfort_dist)*self.cfg.reward.discomfort_penalty_factor*self.cfg.env.time_step
        reward-=4.5*turn**2
        if self.profile.get('reward'):
            reward=float(navigation_reward(*[torch.tensor(x) for x in (before,dist,clearance,wall,turn,intervened)],
                                     self.cfg.env.time_step,self.profile['reward']))
        if outcome:
            reward={1:self.cfg.reward.success_reward,2:self.cfg.reward.collision_penalty,3:0.}[outcome]
        metrics=dict(outcome=outcome,intervened=intervened,last_clearance=clearance,last_static_clearance=wall,
                     last_goal_distance=before,last_goal_distance_after=dist,last_path=path,last_turn=turn)
        return self.observe(),reward,bool(outcome),metrics


class GazeboVecEnv(VecEnv):
    def __init__(self,config,args,device,worker_factory=Worker,detector_factory=DetectorEngine):
        self.cfg,self.args,self.device=config,args,torch.device(device)
        self.n=args.num_processes
        self.h=config.sim.human_num
        self.rng=np.random.default_rng(args.seed+100003*int(os.environ.get('RANK','0')))
        self.profile=config.fidelity
        validate_gazebo_profile(self.profile,args.waffle_require_measured)
        scene_tensors(self.profile,config.sim.arena_size,self.h,config.robot.radius,torch.device('cpu'),
                      args.waffle_scene_split,args.waffle_require_measured)
        self.cases=[s for s in self.profile['scene']['scenarios'] if len(s['humans'])==self.h
                    and s.get('split','train')==args.waffle_scene_split]
        if args.waffle_eval_episodes and args.waffle_require_measured and args.waffle_eval_episodes<len(self.cases):
            raise ValueError('Evaluation must cover all measured cases')
        self.order=self.rng.permutation(len(self.cases))
        self.cursor=np.arange(self.n)
        self.case_ids=np.zeros(self.n,dtype=np.int64)
        root=Path(args.waffle_gazebo_logs)
        root.mkdir(parents=True,exist_ok=True)
        self.root=Path(tempfile.mkdtemp(prefix='gazebo-',dir=root))
        self.pool=ThreadPoolExecutor(max_workers=self.n)
        self.worlds=[PerWorld(config,args,self.device,self.root,i,worker_factory,detector_factory) for i in range(self.n)]
        def box(shape,dtype=np.float32):
            return spaces.Box(-np.inf,np.inf,shape=shape,dtype=dtype)
        obs=spaces.Dict(dict(robot_node=box((1,7)),temporal_edges=box((1,2)),
                      spatial_edges=box((self.h,12)),visible_masks=box((self.h,)),
                      detected_human_num=box((1,))))
        super().__init__(self.n,obs,box((2,)))
        self.actions=None

    def _case(self,i):
        if self.args.waffle_eval_episodes:
            index=int(self.order[self.cursor[i]%len(self.order)])
            self.cursor[i]+=self.n
        else:
            index=int(self.rng.integers(len(self.cases)))
        self.case_ids[i]=index
        return self.cases[index]

    def _tensor(self,rows):
        return {k:torch.as_tensor(np.stack([r[k] for r in rows]),device=self.device) for k in rows[0]}

    def reset(self):
        cases=[self._case(i) for i in range(self.n)]
        rows=list(self.pool.map(lambda x:x[0].reset(x[1]),zip(self.worlds,cases)))
        return self._tensor(rows)

    def step_async(self,actions):
        if self.actions is not None:
            raise RuntimeError('Outstanding Gazebo step')
        self.actions=actions.detach().cpu().numpy().copy()

    def step_wait(self):
        if self.actions is None:
            raise RuntimeError('No pending Gazebo step')
        actions,self.actions=self.actions,None
        # ORCA is batched across worlds. Human positions are Gazebo truth.
        pos=np.stack([w.state['humans'] for w in self.worlds])
        vel=np.stack([w.human_velocity for w in self.worlds])
        radius=np.stack([w.radius for w in self.worlds])
        speed=np.stack([w.speed for w in self.worlds])
        goals=[]
        for w,p in zip(self.worlds,pos):
            w.previous_humans=p.copy()
            goal=w.routes[np.arange(self.h),w.route]
            reached=np.linalg.norm(goal-p,axis=-1)<.15
            w.route=(w.route+reached)%w.routes.shape[1]
            goals.append(w.routes[np.arange(self.h),w.route])
        p,v,r,s,g=[torch.as_tensor(x,device=self.device,dtype=torch.float32) for x in (pos,vel,radius,speed,np.stack(goals))]
        preferred=(g-p)/(g-p).norm(dim=-1,keepdim=True).clamp_min(.1)*s[...,None]
        if self.args.waffle_reactive_humans:
            robot=torch.as_tensor([w.state['truth'][:2] for w in self.worlds],device=self.device,dtype=torch.float32)
            rv=torch.as_tensor([[w.state['twist'][0]*math.cos(w.state['truth'][2]),w.state['twist'][0]*math.sin(w.state['truth'][2])] for w in self.worlds],device=self.device)
            human=orca_velocity(torch.cat((p,robot[:,None]),1),torch.cat((v,rv[:,None]),1),
                       torch.cat((r,torch.full((self.n,1),self.cfg.robot.radius,device=self.device)),1),
                       torch.cat((preferred,rv[:,None]),1),torch.cat((s,torch.full((self.n,1),.1,device=self.device)),1),
                       dt=self.cfg.env.time_step)[:,:self.h]
        else:
            human=orca_velocity(p,v,r,preferred,s,dt=self.cfg.env.time_step)
        rows=list(self.pool.map(lambda x:x[0].step(x[1],x[2]),zip(self.worlds,actions,human.cpu().numpy())))
        obs,rewards,done,metrics=map(list,zip(*rows))
        self.last_scenario=torch.as_tensor(self.case_ids.copy(),device=self.device)
        for key in metrics[0]:
            setattr(self,key,torch.as_tensor([m[key] for m in metrics],device=self.device))
        resets=[(i,self._case(i)) for i,d in enumerate(done) if d]
        for i,o in self.pool.map(lambda x:(x[0],self.worlds[x[0]].reset(x[1])),resets):
            obs[i]=o
        return self._tensor(obs),torch.as_tensor(rewards,device=self.device,dtype=torch.float32)[:,None],torch.as_tensor(done,device=self.device,dtype=torch.bool),None

    def close(self):
        if self.closed:
            return
        self.closed=True
        list(self.pool.map(lambda w:w.close(),self.worlds))
        self.pool.shutdown(wait=True)

    def talk2Env_async(self,data):
        pass

    def talk2Env_wait(self):
        return [True]*self.n


def make_gazebo(config,args,device):
    env=GazeboVecEnv(config,args,device)
    try:
        return VecPretextNormalize(env,config=config,ret=False,ob=False,gamma=args.gamma,test=False)
    except BaseException:
        env.close()
        raise
