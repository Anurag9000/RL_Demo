
"""Gazebo-backed recurrent PPO environment, using actual simulated LDS scans.

Ground truth is restricted to reward, termination and the pedestrian model.
The policy receives encoder odometry, shared detector and tracker output.
One synchronous batch of on-policy decisions; failures abort, never fabricate
observations or rewards. Physics stepping occurs concurrently across worlds.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
import time
import hashlib
import json
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
from waffle_pi.paper_metrics import intrusion_metrics
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
        self.index=index
        self.timing=dict(reset_s=0.,physics_ipc_s=0.,detector_s=0.,step_s=0.,steps=0,resets=0)
        self.static_rect=torch.as_tensor(self.profile['scene']['rectangles_m'],dtype=torch.float32).reshape(-1,4)
        self.geometry=None

    def close(self):
        if self.worker is not None:
            self.worker.close()
            self.worker=None

    def reset(self,scenario):
        start=time.perf_counter()
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
        self.last_sim_time=-math.inf
        self.scan=None
        self.points=None
        self.fault=True
        self.reset_recurrent=True
        self.steps=0
        if self.detector is not None:
            self.detector.reset()
        world,conf=write_case(self.root,self.args.waffle_gazebo_model,self.profile,scenario,int(self.rng.integers(2**31-1)))
        worker_args=(self.args.waffle_gazebo_worker,world,conf,
                        self.root/f'episode_{self.episode:06d}.log',self.args.waffle_gazebo_timeout)
        # Settle on the floor with zero commands. This is not an RL transition.
        warm=round(max(.5,2/self.laser['hz'])/self.g['physics_dt'])
        for attempt in range(2):
            self.worker=self.worker_factory(*worker_args,seed=self.index) if self.worker_factory is Worker else self.worker_factory(*worker_args)
            try:
                self.state=self.worker.step(warm,0.,0.,[[0.,0.]]*self.n,True)
                break
            except RuntimeError as exc:
                message=str(exc)
                retryable='Gazebo worker closed stdout' in message
                self.close()
                event=dict(episode=self.episode,scenario_id=scenario.get('id'),
                           startup_attempt=attempt+1,retryable=retryable,error=message)
                with (self.root/'worker_startup_retries.jsonl').open('a') as f:
                    f.write(json.dumps(event,allow_nan=False)+'\n')
                if not retryable or attempt==1:
                    raise
        if self.state['contact'] or not self.state['floor_contact_observed']:
            raise RuntimeError('Initial Gazebo contact qualification failed')
        self.ingest(self.state)
        self.truth_robot_trace=[np.asarray(self.state['truth'][:2],dtype=np.float64)]
        self.truth_human_trace=[np.asarray(self.state['humans'],dtype=np.float64).copy()]
        self.truth_time_trace=[float(self.state['time'])]
        self.truth_human_ids=np.asarray([f'person_{i}' for i in range(self.n)])
        if self.scan is None:
            raise RuntimeError('No initial laser observation')
        obs=self.observe()
        self.timing['reset_s']+=time.perf_counter()-start
        self.timing['resets']+=1
        return obs

    def ingest(self,state):
        for key,shape in [('truth',(3,)),('odom',(3,)),('twist',(2,)),('wheels',(2,)),('humans',(self.n,2))]:
            a=np.asarray(state[key])
            if a.shape!=shape or not np.isfinite(a).all():
                raise RuntimeError('Invalid native state '+key)
        sim_time=float(state['time'])
        if not math.isfinite(sim_time) or sim_time<=self.last_sim_time:
            raise RuntimeError('Nonmonotonic Gazebo simulation time')
        self.last_sim_time=sim_time
        for frame in state['scans']:
            stamp=float(frame['stamp'])
            if stamp<=self.last_stamp:
                raise RuntimeError('Nonmonotonic simulator scan')
            self.last_stamp=stamp
            values=frame['ranges']
            raw=values.copy() if isinstance(values,np.ndarray) else np.array([float('inf') if x is None else x for x in values],np.float32)
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
                detector_start=time.perf_counter()
                points=self.detector.detect(raw,frame['min'],frame['max'])
                self.timing['detector_s']+=time.perf_counter()-detector_start
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
        previous_fault=self.fault
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
        self.reset_recurrent=previous_fault or self.fault
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
        fresh=(not self.fault and self.points is not None and 0<=self.state['time']-f['stamp']<=.5
               and scan_coverage(raw,f['angle_step'],f['min'],f['max']))
        mv,mw=self.state['twist']
        cmd=safe_command(v,w,raw,f['min'],f['max'],fresh=fresh,armed=True,estop=False,
                         measured_speed=mv,measured_angular=mw,**self.cfg.waffle_safety)
        blocked=not fresh or (cmd==(0.,0.) and (v!=0. or w!=0.))
        return cmd,blocked

    def _advance_ground_truth_humans(self, future_policy_steps):
        """Continue the simulated crowd after termination to complete paper lookahead."""
        for _ in range(future_policy_steps):
            pos=np.asarray(self.state['humans'],np.float32)
            vel=np.asarray(self.human_velocity,np.float32)
            idx=np.arange(self.n)
            goal=self.routes[idx,self.route]
            reached=np.linalg.norm(goal-pos,axis=-1)<.15
            self.route=(self.route+reached)%self.routes.shape[1]
            goal=self.routes[idx,self.route]
            p,v,r,s,g=[torch.as_tensor(x,device=self.device,dtype=torch.float32) for x in
                       (pos[None],vel[None],self.radius[None],self.speed[None],goal[None])]
            preferred=(g-p)/(g-p).norm(dim=-1,keepdim=True).clamp_min(.1)*s[...,None]
            if self.args.waffle_reactive_humans:
                robot=torch.as_tensor([[self.state['truth'][:2]]],device=self.device,dtype=torch.float32)
                twist=self.state['twist'][0]
                theta=self.state['truth'][2]
                rv=torch.as_tensor([[[twist*math.cos(theta),twist*math.sin(theta)]]],device=self.device,dtype=torch.float32)
                human=orca_velocity(torch.cat((p,robot),1),torch.cat((v,rv),1),
                    torch.cat((r,torch.full((1,1),self.cfg.robot.radius,device=self.device)),1),
                    torch.cat((preferred,rv),1),torch.cat((s,torch.full((1,1),.1,device=self.device)),1),
                    dt=self.cfg.env.time_step)[:,:self.n]
            else:
                human=orca_velocity(p,v,r,preferred,s,dt=self.cfg.env.time_step)
            command=human[0].detach().cpu().numpy()
            before=pos.copy()
            for _ in range(self.substeps):
                old=torch.as_tensor(self.state['humans'],dtype=torch.float32)
                proposed=old+torch.as_tensor(command,dtype=torch.float32)*self.gate_dt
                new=constrain_human_step(old,proposed,torch.as_tensor(self.radius),self.cfg.sim.arena_size,self.static_rect)
                hv=((new-old)/self.gate_dt).tolist()
                self.state=self.worker.step(self.physics_iterations,0.,0.,hv,True)
            self.human_velocity=(np.asarray(self.state['humans'],np.float32)-before)/self.cfg.env.time_step
            self.truth_human_trace.append(np.asarray(self.state['humans'],dtype=np.float64).copy())
            self.truth_time_trace.append(float(self.state['time']))

    def _paper_social_metrics(self):
        # T episode transitions need only four extra frames: frame T already
        # follows the final decision, so it supplies that decision's first
        # future point and the earlier T-1 decisions have five-point horizons.
        self._advance_ground_truth_humans(4)
        return intrusion_metrics(self.truth_robot_trace,self.truth_human_trace,self.radius,
                                 self.cfg.robot.radius,future_steps=5)

    def step(self,latent,human_velocity):
        start=time.perf_counter()
        requested=native_twist(latent,self.cfg.waffle_limits)
        prev=np.asarray(self.state['truth'])
        before=np.linalg.norm(prev[:2]-self.goal)
        clearance=wall=math.inf
        path=turn=0.
        collision=intervened=False
        last_h=np.asarray(self.state['humans'])
        rect=self.static_rect
        for _ in range(self.substeps):
            cmd,blocked=self.guard(requested)
            if collision:
                cmd,blocked=(0.,0.),True
            intervened|=blocked
            # Human static obstacle rejection supplements contact physics.
            # This does not implement a general obstacle ORCA planner.
            old_h=torch.as_tensor(self.state['humans'],dtype=torch.float32)
            vel=torch.as_tensor(human_velocity,dtype=torch.float32)
            new_h=constrain_human_step(old_h,old_h+vel*self.gate_dt,
                        torch.as_tensor(self.radius),self.cfg.sim.arena_size,rect)
            hv=((new_h-old_h)/self.gate_dt).tolist()
            t0=time.perf_counter()
            state=self.worker.step(self.physics_iterations,*cmd,hv,blocked)
            self.timing['physics_ipc_s']+=time.perf_counter()-t0
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
        self.truth_robot_trace.append(np.asarray(prev[:2],dtype=np.float64).copy())
        self.truth_human_trace.append(np.asarray(last_h,dtype=np.float64).copy())
        self.truth_time_trace.append(float(self.state['time']))
        dist=float(np.linalg.norm(prev[:2]-self.goal))
        success=dist<self.cfg.robot.radius and not collision
        timeout=self.steps>=round(self.cfg.env.time_limit/self.cfg.env.time_step) and not collision and not success
        outcome=2 if collision else 1 if success else 3 if timeout else 0
        paper={'paper_itr':float('nan'),'paper_sd_m':float('nan'),'paper_intrusion_steps':0,'paper_episode_steps':0,'paper_sd_sum_m':0.0,'paper_sd_count':0}
        if outcome and self.args.waffle_eval_episodes:
            paper=self._paper_social_metrics()
            if paper['paper_sd_m'] is None:
                paper['paper_sd_m']=float('nan')
        reward=3*(before-dist)
        if clearance<self.cfg.reward.discomfort_dist:
            reward=(clearance-self.cfg.reward.discomfort_dist)*self.cfg.reward.discomfort_penalty_factor*self.cfg.env.time_step
        reward-=4.5*turn**2
        if self.profile.get('reward'):
            reward=float(navigation_reward(*[torch.tensor(x) for x in (before,dist,clearance,wall,turn,intervened)],
                                     self.cfg.env.time_step,self.profile['reward']))
        if outcome:
            reward={1:self.cfg.reward.success_reward,2:self.cfg.reward.collision_penalty,3:0.}[outcome]
        metrics=dict(outcome=outcome,**paper,intervened=intervened,last_clearance=clearance,last_static_clearance=wall,
                     last_goal_distance=before,last_goal_distance_after=dist,last_path=path,last_turn=turn)
        obs=self.observe()
        self.timing['step_s']+=time.perf_counter()-start
        self.timing['steps']+=1
        return obs,reward,bool(outcome),metrics


class GazeboVecEnv(VecEnv):
    def __init__(self,config,args,device,worker_factory=Worker,detector_factory=DetectorEngine):
        self.cfg,self.args,self.device=config,args,torch.device(device)
        if self.device.type=='cuda' and self.device.index is None:
            self.device=torch.device('cuda',torch.cuda.current_device())
        self.n=args.num_processes
        self.h=config.sim.human_num
        self.rng=np.random.default_rng(args.seed+100003*int(os.environ.get('RANK','0')))
        if args.waffle_gazebo_detector_device=='learner':
            args.waffle_gazebo_detector_device=str(self.device)
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
        self.eval_quotas=None
        self.eval_completed=np.zeros(self.n,dtype=np.int64)
        self.eval_finished=np.zeros(self.n,dtype=bool)
        self.case_rng=[np.random.default_rng(np.random.SeedSequence([args.seed,int(os.environ.get('RANK','0')),i,971])) for i in range(self.n)]
        root=Path(args.waffle_gazebo_logs)
        root.mkdir(parents=True,exist_ok=True)
        trace_dir=getattr(args,'waffle_eval_trace_dir',None)
        if trace_dir is None and args.waffle_eval_episodes and getattr(args,'waffle_eval_json',None):
            metrics_path=Path(args.waffle_eval_json)
            trace_dir=metrics_path.with_suffix('').with_name(metrics_path.stem+'_traces')
        if trace_dir is None and args.waffle_eval_episodes:
            trace_dir=root/'paper_traces'
        self.eval_trace_dir=Path(trace_dir) if trace_dir else None
        if self.eval_trace_dir is not None:
            if self.eval_trace_dir.exists() and any(self.eval_trace_dir.iterdir()):
                raise FileExistsError(f'evaluation trace directory is not empty: {self.eval_trace_dir}')
            self.eval_trace_dir.mkdir(parents=True,exist_ok=True)
        def file_sha(path):
            p=Path(path) if path else None
            return hashlib.sha256(p.read_bytes()).hexdigest() if p is not None and p.is_file() else ''
        profile_path=getattr(args,'waffle_fidelity_profile',None)
        self.eval_hashes={
            'profile_sha256':file_sha(profile_path),
            'checkpoint_sha256':file_sha(getattr(args,'load_path',None)),
            'detector_sha256':file_sha(getattr(args,'waffle_detector_weights',None)),
            'model_sdf_sha256':file_sha(getattr(args,'waffle_gazebo_model',None)),
            'worker_sha256':file_sha(getattr(args,'waffle_gazebo_worker',None)),
        }
        self.root=Path(tempfile.mkdtemp(prefix='gazebo-',dir=root))
        self.pool=ThreadPoolExecutor(max_workers=self.n)
        self.reset_pool=ThreadPoolExecutor(max_workers=min(self.n, max(1,int(os.environ.get('WAFFLE_RESET_THREADS',str(self.n))))))
        self.detector_pool=None
        if os.environ.get('WAFFLE_BATCH_DETECTOR','0')=='1' and detector_factory is DetectorEngine:
            from .batched_detector import DetectorPool
            self.detector_pool=DetectorPool(args.waffle_detector_weights,self.profile['detector'],
                args.waffle_gazebo_detector_device,int(os.environ.get('WAFFLE_DETECTOR_BATCH','8')),
                float(os.environ.get('WAFFLE_DETECTOR_WAIT_MS','.5')))
            detector_factory=lambda *a:self.detector_pool.stream(a[-1])
        self.worlds=[PerWorld(config,args,self.device,self.root,i,worker_factory,detector_factory) for i in range(self.n)]
        def box(shape,dtype=np.float32):
            return spaces.Box(-np.inf,np.inf,shape=shape,dtype=dtype)
        obs=spaces.Dict(dict(robot_node=box((1,7)),temporal_edges=box((1,2)),
                      spatial_edges=box((self.h,12)),visible_masks=box((self.h,)),
                      detected_human_num=box((1,))))
        super().__init__(self.n,obs,box((2,)))
        self.actions=None
        self.pinned={}
        self.pinned_enabled=self.device.type=='cuda' and os.environ.get('WAFFLE_PINNED_OBS','0')=='1'

    def _case(self,i):
        if self.args.waffle_eval_episodes:
            index=int(self.order[self.cursor[i]%len(self.order)])
            self.cursor[i]+=self.n
        else:
            index=int(self.case_rng[i].integers(len(self.cases)))
        self.case_ids[i]=index
        return self.cases[index]

    def _tensor(self,rows):
        if not self.pinned_enabled:
            return {k:torch.as_tensor(np.stack([r[k] for r in rows]),device=self.device) for k in rows[0]}
        # Reused pinned buffers: wait for prior DMA before overwriting.
        # All copies and consumers use the current CUDA stream.
        out={}
        for k in rows[0]:
            arr=np.stack([r[k] for r in rows])
            if k not in self.pinned:
                host=torch.empty(arr.shape,dtype=torch.from_numpy(arr).dtype,pin_memory=True)
                self.pinned[k]=(host,None)
            host,event=self.pinned[k]
            if event is not None:
                event.synchronize()
            np.copyto(host.numpy(),arr)
            out[k]=host.to(self.device,non_blocking=True)
            event=torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            self.pinned[k]=(host,event)
        return out

    def set_eval_quotas(self,quotas):
        quotas=np.asarray(quotas,dtype=np.int64)
        if quotas.shape!=(self.n,) or (quotas<1).any() or int(quotas.sum())!=int(self.args.waffle_eval_episodes):
            raise ValueError('Evaluation quotas must be positive per lane and sum to requested episodes')
        if self.actions is not None or self.eval_completed.any():
            raise RuntimeError('Evaluation quotas must be set before the first step')
        self.eval_quotas=quotas.copy()
        self.eval_finished[:]=False

    def _save_eval_trace(self,lane,scenario_index,scenario_id,episode_number,metrics):
        world=self.worlds[lane]
        robot=np.asarray(world.truth_robot_trace,dtype=np.float32)
        humans=np.asarray(world.truth_human_trace,dtype=np.float32)
        times=np.asarray(world.truth_time_trace,dtype=np.float64)
        if len(robot)!=world.steps+1 or len(humans)<len(robot)+4 or len(times)!=len(humans):
            raise RuntimeError('evaluation truth trace is incomplete or lacks the four-frame suffix')
        if not np.isfinite(times).all() or np.any(np.diff(times)<=0):
            raise RuntimeError('evaluation truth timestamps are invalid or nonmonotonic')
        path=self.eval_trace_dir/f'lane-{lane:03d}-episode-{episode_number:04d}-case-{scenario_index:06d}.npz'
        tmp=path.with_name(path.name+'.tmp')
        sd_value=metrics['paper_sd_m']
        sd_valid=sd_value is not None and np.isfinite(sd_value)
        with tmp.open('wb') as stream:
            np.savez_compressed(stream,
                robot_positions=robot,human_positions=humans,human_radii=np.asarray(world.radius,dtype=np.float32),
                robot_radius=np.asarray(self.cfg.robot.radius,dtype=np.float32),future_steps=np.asarray(5,dtype=np.int32),
                policy_dt=np.asarray(self.cfg.env.time_step,dtype=np.float64),
                robot_times=times[:len(robot)],human_times=times,
                human_ids=world.truth_human_ids,
                episode_steps=np.asarray(world.steps,dtype=np.int32),scenario_index=np.asarray(scenario_index,dtype=np.int32),
                scenario_id=np.asarray(scenario_id),lane=np.asarray(lane,dtype=np.int32),episode_number=np.asarray(episode_number,dtype=np.int32),
                split=np.asarray(self.args.waffle_scene_split),seed=np.asarray(self.args.seed,dtype=np.int64),
                profile_sha256=np.asarray(self.eval_hashes['profile_sha256']),
                checkpoint_sha256=np.asarray(self.eval_hashes['checkpoint_sha256']),
                detector_sha256=np.asarray(self.eval_hashes['detector_sha256']),
                model_sdf_sha256=np.asarray(self.eval_hashes['model_sdf_sha256']),
                worker_sha256=np.asarray(self.eval_hashes['worker_sha256']),
                outcome=np.asarray(metrics['outcome'],dtype=np.int32),
                online_itr=np.asarray(metrics['paper_itr'],dtype=np.float64),
                online_sd_valid=np.asarray(sd_valid,dtype=np.bool_),
                online_sd_m=np.asarray(float(sd_value) if sd_valid else 0.,dtype=np.float64))
        os.replace(tmp,path)

    def reset(self):
        cases=[self._case(i) for i in range(self.n)]
        rows=list(self.pool.map(lambda x:x[0].reset(x[1]),zip(self.worlds,cases)))
        self.eval_completed[:]=0
        self.eval_finished[:]=False
        self._last_obs=rows
        self._last_metrics=[None]*self.n
        self.policy_reset=torch.ones(self.n,device=self.device,dtype=torch.bool)
        return self._tensor(rows)

    def step_async(self,actions):
        if self.actions is not None:
            raise RuntimeError('Outstanding Gazebo step')
        self.actions=actions.detach().cpu().numpy().copy()

    def step_wait(self):
        if self.actions is None:
            raise RuntimeError('No pending Gazebo step')
        actions,self.actions=self.actions,None
        active=np.ones(self.n,dtype=bool) if self.eval_quotas is None else ~self.eval_finished
        pos=np.stack([w.state['humans'] for w in self.worlds])
        vel=np.stack([w.human_velocity for w in self.worlds])
        radius=np.stack([w.radius for w in self.worlds])
        speed=np.stack([w.speed for w in self.worlds])
        goals=[]
        for i,(w,p) in enumerate(zip(self.worlds,pos)):
            if active[i]:
                w.previous_humans=p.copy()
                reached=np.linalg.norm(w.routes[np.arange(self.h),w.route]-p,axis=-1)<.15
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
        old_ids=self.case_ids.copy()
        old_scenario_ids=[self.cases[int(i)].get('id',str(int(i))) for i in old_ids]
        velocities=human.cpu().numpy()
        pending={self.pool.submit(self.worlds[i].step,actions[i],velocities[i]):i for i in range(self.n) if active[i]}
        rows=[None]*self.n
        for i in range(self.n):
            if not active[i]:
                rows[i]=(self._last_obs[i],0.,True,self._last_metrics[i])
        resets={}
        for future in as_completed(pending):
            i=pending[future]
            rows[i]=future.result()
            if rows[i][2]:
                if self.eval_quotas is None:
                    resets[i]=self.reset_pool.submit(self.worlds[i].reset,self._case(i))
                else:
                    self.eval_completed[i]+=1
                    if self.eval_trace_dir is not None:
                        self._save_eval_trace(i,int(old_ids[i]),old_scenario_ids[i],int(self.eval_completed[i]),rows[i][3])
                    if self.eval_completed[i]>=self.eval_quotas[i]:
                        self.eval_finished[i]=True
                    else:
                        resets[i]=self.reset_pool.submit(self.worlds[i].reset,self._case(i))
        obs,rewards,done,metrics=map(list,zip(*rows))
        self.last_scenario=torch.as_tensor(old_ids,device=self.device)
        self.last_scenario_ids=old_scenario_ids
        for key in metrics[0]:
            setattr(self,key,torch.as_tensor([m[key] for m in metrics],device=self.device))
        for i,future in resets.items():
            obs[i]=future.result()
        self._last_obs=obs
        self._last_metrics=metrics
        self.policy_reset=torch.as_tensor([w.reset_recurrent for w in self.worlds],device=self.device,dtype=torch.bool)
        return self._tensor(obs),torch.as_tensor(rewards,device=self.device,dtype=torch.float32)[:,None],torch.as_tensor(done,device=self.device,dtype=torch.bool),None

    def close(self):
        if self.closed:
            return
        self.closed=True
        self.pool.shutdown(wait=True)
        self.reset_pool.shutdown(wait=True)
        for w in self.worlds:
            w.close()
        if self.detector_pool is not None:
            self.detector_pool.close()
        for _,event in self.pinned.values():
            if event is not None:
                event.synchronize()

    def performance_report(self):
        # Worker times overlap; totals are CPU/world-seconds, not wall time.
        result={k:sum(w.timing[k] for w in self.worlds) for k in self.worlds[0].timing}
        if self.detector_pool is not None:
            result['detector_pool']=dict(self.detector_pool.stats)
        return result

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
