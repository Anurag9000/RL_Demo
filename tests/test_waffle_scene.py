import unittest
import torch
from waffle_pi.scene import scene_tensors,constrain_human_step
from waffle_pi.fidelity import FidelityDrive,read_profile


class Scene(unittest.TestCase):
    def test_survey_and_wall(self):
        profile={'scene':{'arena_half_extent_m':2.5,'rectangles_m':[[0.,-.5,.2,.5]],
            'scenarios':[{'robot_start':[-1.,-1.,0.],'robot_goal':[1.,-1.],
            'humans':[{'start':[-1.,1.],'radius':.2,'speed':.7,'waypoints':[[1.,1.],[-1.,1.]]}]}]}}
        scene=scene_tensors(profile,2.5,1,.25,torch.device('cpu'))
        old=torch.tensor([[[-.4,0.]]])
        new=torch.tensor([[[.4,0.]]])
        torch.testing.assert_close(constrain_human_step(old,new,torch.tensor([[.2]]),2.5,scene['rectangles']),old)
        profile['scene']['scenarios'][0]['humans'][0]['waypoints']=[[1.,0.],[-1.,0.]]
        with self.assertRaises(ValueError):
            scene_tensors(profile,2.5,1,.25,torch.device('cpu'))
    def test_scan_snapshot(self):
        profile,_=read_profile('configs/waffle_fidelity.example.json')
        drive=FidelityDrive(profile,1,torch.device('cpu'),torch.Generator().manual_seed(2))
        pose=torch.zeros(1,3)
        human=torch.tensor([[[1.,0.,.3]]])
        drive.ensure_scan(pose,human,2.5)
        old=drive.scan.clone()
        drive.ensure_scan(pose,human+1,2.5)
        torch.testing.assert_close(drive.scan,old)
        torch.testing.assert_close(drive.scan_humans,human)
