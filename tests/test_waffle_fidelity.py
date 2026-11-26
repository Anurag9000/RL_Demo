import math
import unittest
import torch
from waffle_pi.fidelity import read_profile,raycast,disk_clearance,motor_step,clearance_gate
from waffle_pi.dynamics import DriveLimits
from waffle_pi.safety import safe_command
from waffle_pi.native import safety_parameters


class Fidelity(unittest.TestCase):
    def setUp(self):
        self.profile,_=read_profile('configs/waffle_fidelity.example.json')
    def test_profile_provenance(self):
        with self.assertRaises(ValueError):
            read_profile('configs/waffle_fidelity.example.json',True)
    def test_ray_and_occlusion(self):
        laser=dict(self.profile['laser'],rays=360,offset_x=0.,angle_min=0.)
        pose=torch.zeros(1,3)
        human=torch.tensor([[[1.,0.,.2],[2.,0.,.2]]])
        scan,idx=raycast(pose,human,2.5,laser)
        self.assertAlmostEqual(float(scan[0,0]),.8,places=5)
        self.assertEqual(int(idx[0,0]),0)
        self.assertAlmostEqual(float(scan[0,90]),2.5,places=5)
        rect=torch.tensor([[.4,-.1,.6,.1]])
        scan,idx=raycast(pose,human,2.5,laser,rect)
        self.assertAlmostEqual(float(scan[0,0]),.4,places=5)
        self.assertEqual(int(idx[0,0]),-1)
        self.assertAlmostEqual(float(disk_clearance(pose[:,:2],.25,2.5,rect)),.15,places=5)
    def test_asymmetric_actuator(self):
        wheels=torch.zeros(1,2)
        target=torch.ones(1,2)*3
        out=motor_step(wheels,target,.05,torch.tensor([.1,.2]),torch.ones(2)*100,
                       torch.ones(2)*100,torch.ones(2),torch.zeros(2))
        torch.testing.assert_close(out,3*(1-torch.exp(-.05/torch.tensor([[.1,.2]]))))
    def test_gate_scalar_parity(self):
        laser=self.profile['laser']
        params=safety_parameters(None,.25)
        limits=DriveLimits(max_linear=.1,max_angular=.5)
        scan=torch.full((3,360),2.)
        scan[0,0]=.3
        scan[1,:45]=float('inf')
        target=torch.full((3,2),.1/.033)
        measured=torch.zeros(3,2)
        actual,blocked=clearance_gate(scan,measured,target,limits,params,laser)
        self.assertEqual(blocked.tolist(),[True,True,False])
        for i in (0,2):
            v,w=safe_command(.1,0.,scan[i].tolist(),laser['range_min'],laser['range_max'],
                         fresh=True,armed=True,estop=False,**params)
            self.assertAlmostEqual(float(actual[i].sum()*limits.wheel_radius/2),v,places=6)
