
"""Pure Python integration fixtures. These do NOT run Gazebo physics."""
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from gazebo_sim.world import training_world,validate_gazebo_profile
from gazebo_sim.vector_env import GazeboVecEnv
from waffle_pi.dynamics import DriveLimits
from waffle_pi.native import safety_parameters
from rl.networks.storage import RolloutStorage
from waffle_pi.spaces import spaces


FIXTURE='<sdf version="1.8"><model name="turtlebot3_waffle_pi"><pose>0 0 0 0 0 0</pose>' + ''.join(
    '<link name="wheel_'+s+'_link"><collision name="wheel"><surface><friction><ode/></friction></surface></collision></link>' for s in ('left','right')) + '<link name="base_scan"><sensor name="scan" type="gpu_lidar"><pose>0 0 .121 0 0 0</pose><topic>scan</topic><ray><scan><horizontal/></scan><range/><noise/></ray></sensor></link><plugin name="gz::sim::systems::DiffDrive" filename="gz-sim-diff-drive-system"><topic>cmd_vel</topic></plugin></model></sdf>'


class FakeWorker:
    def __init__(self,binary,world,config,log,timeout):
        self.cfg=json.loads(Path(config).read_text())
        self.pose=np.asarray(self.cfg['start'],dtype=float)
        self.humans=np.array([[-1.6,.6]])
        self.time=0.
    def step(self,iterations,v,w,humans,stop=False):
        dt=iterations*self.cfg['dt']
        self.time+=dt
        self.pose[:2]+=v*dt*np.array([math.cos(self.pose[2]),math.sin(self.pose[2])])
        self.pose[2]+=w*dt
        self.humans+=dt*np.asarray(humans)
        inc=2*math.pi/360
        return dict(truth=self.pose.tolist(),odom=self.pose.tolist(),twist=[v,w],wheels=[v/.033,v/.033],
            time=self.time,humans=self.humans.tolist(),contact=False,floor_contact_observed=True,
            path=v*dt,turn=abs(w*dt),scans=[dict(stamp=self.time,pose=self.pose.tolist(),ranges=[2.]*360,
                min=.12,max=3.5,angle_min=0.,angle_max=359*inc,angle_step=inc)])
    def close(self):
        pass


class FakeDetector:
    def __init__(self,*args):
        pass
    def reset(self):
        pass
    def detect(self,*args):
        return np.empty((0,2),np.float32)


class Integration(unittest.TestCase):
    def test_world_and_parallel_observation(self):
        profile=json.loads(Path('configs/waffle_gazebo.example.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.sdf'
            path.write_text(FIXTURE)
            scenario=profile['scene']['scenarios'][0]
            root=training_world(path,profile,scenario)
            self.assertIsNone(root.find(".//plugin[@name='gz::sim::systems::DiffDrive']"))
            self.assertEqual(len(root.findall(".//model[@name='person_0']")),1)
            with self.assertRaises(ValueError):
                validate_gazebo_profile(profile,True)
            cfg=SimpleNamespace(fidelity=profile,sim=SimpleNamespace(human_num=1,arena_size=2.5),
                env=SimpleNamespace(time_step=.25,time_limit=90.),
                robot=SimpleNamespace(radius=.25,v_pref=.1,sensor_range=3.),
                reward=SimpleNamespace(discomfort_dist=.25,discomfort_penalty_factor=10.,success_reward=10.,collision_penalty=-20.),
                waffle_safety=safety_parameters(None,.25),waffle_limits=DriveLimits(max_linear=.1,max_angular=.5))
            args=SimpleNamespace(num_processes=2,seed=4,waffle_require_measured=False,waffle_scene_split='train',
                waffle_eval_episodes=0,waffle_gazebo_logs=tmp,waffle_gazebo_model=str(path),
                waffle_gazebo_worker='fixture',waffle_gazebo_timeout=2.,waffle_detector_weights='fixture',
                waffle_gazebo_detector_device='cpu',waffle_reactive_humans=False)
            env=GazeboVecEnv(cfg,args,torch.device('cpu'),FakeWorker,FakeDetector)
            try:
                obs=env.reset()
                self.assertEqual(tuple(obs['robot_node'].shape),(2,1,7))
                self.assertEqual(tuple(obs['gst_history'].shape),(2,1,5,2))
                self.assertFalse(bool(obs['visible_masks'].any()))
                obs,reward,done,_=env.step(torch.ones(2,2))
                self.assertTrue(bool(torch.isfinite(reward).all()))
                self.assertFalse(bool(done.any()))
                self.assertEqual(tuple(obs['spatial_edges'].shape),(2,1,12))
                self.assertTrue(bool((env.last_path>0).all()))
            finally:
                env.close()
    def test_recurrent_reset_does_not_cut_return(self):
        box=spaces.Box(-1.,1.,shape=(1,12),dtype=np.float32)
        storage=RolloutStorage(2,1,{'spatial_edges':box},
            spaces.Box(-1.,1.,shape=(2,),dtype=np.float32),3,3,separate_rnn_masks=True)
        storage.rewards.fill_(1.)
        storage.policy_masks[1].zero_()
        storage.compute_returns(torch.zeros(1,1),True,1.,1.,False,'reference')
        self.assertEqual(float(storage.returns[0]),2.)
        self.assertEqual(float(storage.masks[1]),1.)
        self.assertEqual(float(storage.policy_masks[1]),0.)
