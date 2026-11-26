import unittest
import numpy as np
from waffle_pi.tracking import Tracker,TrajectoryHistory


class Tracking(unittest.TestCase):
    def test_reordered_detections(self):
        tr=Tracker(2)
        tr.update([[0.,0.],[2.,0.]],0.)
        reused=tr.update([[1.9,0.],[.1,0.]],.25)
        self.assertFalse(reused.any())
        self.assertLess(tr.pos[0,0],.5)
        self.assertGreater(tr.pos[1,0],1.5)
    def test_overflow(self):
        tr=Tracker(1)
        tr.update([[0.,0.],[2.,0.]],0.)
        self.assertTrue(tr.overflow)
    def test_history(self):
        h=TrajectoryHistory(1,3)
        h.append(0.,[[0.,0.]],[True],[True])
        h.append(.2,[[.2,0.]],[True],[False])
        h.append(.5,[[.5,0.]],[True],[False])
        p,m=h.sample(.5)
        np.testing.assert_allclose(p[0,0,:,0],[0.,.25,.5],atol=1e-6)
        self.assertTrue(m.all())
        h.append(.75,[[3.,0.]],[True],[True])
        _,m=h.sample(.75)
        self.assertEqual(m.sum(),1.)
