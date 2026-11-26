import unittest
import torch
from waffle_pi.quality import wilson,navigation_reward,validate_reward
from waffle_pi.scene import scene_tensors


class Quality(unittest.TestCase):
    def test_confidence(self):
        self.assertGreater(wilson(0,10)[1],.05)
        self.assertLess(wilson(0,200)[1],.05)
        self.assertLess(wilson(90,100)[0],.9)
    def test_reward(self):
        r=dict(progress=3.,time=.01,turn=.1,intervention=.5,human_clearance=2.,
               wall_clearance=1.,human_margin=.25,wall_margin=.1)
        validate_reward(r)
        a=torch.tensor([1.,1.])
        out=navigation_reward(a,torch.tensor([.9,.9]),a,a,torch.zeros(2),
                              torch.tensor([False,True]),.25,r)
        self.assertAlmostEqual(float(out[0]-out[1]),.125,places=6)
    def test_split(self):
        cases=[]
        for i,split in enumerate(('train','validation','test')):
            cases.append(dict(id=split,split=split,robot_start=[-1.,-1.,i*.1],
                robot_goal=[1.,-1.],humans=[dict(start=[-1.,1.],radius=.2,speed=.7,
                waypoints=[[1.,1.],[-1.,1.]])]))
        profile={'scene':dict(arena_half_extent_m=2.5,rectangles_m=[],scenarios=cases)}
        s=scene_tensors(profile,2.5,1,.25,torch.device('cpu'),'test',True)
        self.assertAlmostEqual(float(s['starts'][0,2]),.2)
        cases[2]['robot_start']=cases[0]['robot_start']
        with self.assertRaises(ValueError):
            scene_tensors(profile,2.5,1,.25,torch.device('cpu'),'test',True)
