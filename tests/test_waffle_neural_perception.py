import unittest
from types import SimpleNamespace
import numpy as np
import torch
from waffle_pi.fidelity import read_profile
from waffle_pi.neural_perception import NeuralPerception


class Fake:
    def __init__(self,*a):
        pass
    def reset(self):
        pass
    def detect(self,*a):
        return np.array([[1.,0.]],np.float32)


class Neural(unittest.TestCase):
    def test_shared_track_and_fault(self):
        profile,_=read_profile('configs/waffle_fidelity.example.json')
        profile['detector']=dict(model='DR-SPAAM',stride=1,confidence=.5,max_range=3.)
        p=NeuralPerception(SimpleNamespace(waffle_detector_weights='fixture'),profile,1,2,torch.device('cpu'),Fake)
        pose=torch.zeros(1,3)
        p.on_scan(torch.ones(1,360),pose,torch.zeros(1),torch.ones(1,dtype=torch.bool))
        rel,seen,h,m=p.observe(pose[:,:2],pose[:,:2],3.)
        self.assertFalse(bool(p.fault[0]))
        self.assertTrue(bool(seen[0,0]))
        self.assertAlmostEqual(float(rel[0,0,0]),.936,places=5)
        self.assertEqual(float(m.sum()),1.)
        p.observe(pose[:,:2],pose[:,:2],3.)
        self.assertTrue(bool(p.fault[0]))
