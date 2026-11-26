"""Test upstream-compatible batch orchestration with a stateful fixture.

Real DR-SPAAM checkpoint parity is a separate script, not proven by this.
"""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
import threading
import numpy as np
import torch
from gazebo_sim.batched_detector import DetectorPool
from waffle_pi.scan_geometry import ScanGeometry


class Gate:
    _memory=None
    def reset(self):
        self._memory=None


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate=Gate()
    def forward(self,x,inference=True):
        v=x[:,0,0,0].reshape(-1,1,1,1)
        self.gate._memory=v if self.gate._memory is None else self.gate._memory+v
        reg=self.gate._memory.reshape(-1,1,1).expand(-1,1,2)
        return torch.ones_like(reg[:,:,:1])*10,reg,None


class Batch(unittest.TestCase):
    def test_distinct_streams_and_reset(self):
        # Construct the batching state machine without importing upstream.
        from collections import deque
        pool=DetectorPool.__new__(DetectorPool)
        geom=ScanGeometry.from_metadata(32,-1.,1.,2/31)
        pool.geometry=geom
        pool.model=Model()
        pool.device=torch.device('cpu')
        pool.settings=dict(model='DR-SPAAM',stride=1,confidence=.5,max_range=100.)
        pool.utils=SimpleNamespace(scans_to_cutout=lambda scan,*args,**kw:scan[:,:1,None],
            nms_predicted_center=lambda scan,phi,cls,reg:(reg,np.ones(len(reg)),None))
        pool.max_batch=4
        pool.wait_s=.05
        pool.queue=deque()
        pool.condition=threading.Condition()
        pool.closed=False
        pool.stats=dict(batches=0,frames=0,max_batch=0,inference_s=0.)
        pool.thread=threading.Thread(target=pool._run)
        pool.thread.start()
        a,b=pool.stream(geom),pool.stream(geom)
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                def pair():
                    return list(executor.map(lambda e:e[0].detect(np.full(32,e[1],np.float32),.1,10.),[(a,1.),(b,2.)]))
                x,y=pair()
                np.testing.assert_allclose(x,[[1.,-1.]])
                np.testing.assert_allclose(y,[[2.,-2.]])
                x,y=pair()
                np.testing.assert_allclose(x,[[2.,-2.]])
                np.testing.assert_allclose(y,[[4.,-4.]])
                a.reset()
                x,y=pair()
                np.testing.assert_allclose(x,[[1.,-1.]])
                np.testing.assert_allclose(y,[[6.,-6.]])
            self.assertGreaterEqual(pool.stats['max_batch'],2)
        finally:
            pool.close()
