"""Shared DR-SPAAM weights with per-stream recurrent memory.

Uses the upstream model, cutout and NMS functions unchanged. Fresh and
continuing streams form separate batches: the upstream gate has a special
first-frame path. Each stream has at most one outstanding request. This
backend must pass real-checkpoint numerical and policy-quality parity tests
before becoming the measured-reference default.
"""
from concurrent.futures import Future
from collections import deque
import os
import threading
import time
import numpy as np
import torch


class Stream:
    def __init__(self,pool,geometry):
        self.pool,self.geometry=pool,geometry
        self.memory=None
        self.lock=threading.Lock()
    def reset(self):
        with self.lock:
            self.memory=None
    def detect(self,ranges,range_min,range_max):
        with self.lock:
            scan=self.geometry.preprocess(ranges,range_min,range_max)
            return self.pool.submit(self,scan)


class DetectorPool:
    def __init__(self,weights,settings,device,max_batch=8,wait_ms=.5):
        from dr_spaam.detector import Detector
        from dr_spaam.utils import utils
        self.factory=Detector
        self.utils=utils
        self.weights,self.settings=weights,settings
        self.device=torch.device(device)
        self.max_batch,self.wait_s=max_batch,wait_ms/1000
        if max_batch<1 or not 0<=wait_ms<=20:
            raise ValueError('Invalid detector batching controls')
        self.queue=deque()
        self.condition=threading.Condition()
        self.closed=False
        self.model=None
        self.geometry=None
        self.stats=dict(batches=0,frames=0,max_batch=0,inference_s=0.)
        self.thread=threading.Thread(target=self._run,daemon=True)
        self.thread.start()

    def stream(self,geometry):
        return Stream(self,geometry)

    def submit(self,stream,scan):
        result=Future()
        with self.condition:
            if self.closed:
                raise RuntimeError('Detector pool is closed')
            self.queue.append((stream,scan,result))
            self.condition.notify()
        return result.result()

    def _init(self,geometry):
        if self.model is not None:
            if self.geometry!=geometry:
                raise ValueError('Shared detector requires identical laser geometry')
            return
        self.geometry=geometry
        d=self.settings
        if self.device.type=='cuda':
            torch.cuda.set_device(self.device)
        wrapper=self.factory(self.weights,model=d['model'],gpu=self.device.type=='cuda',
                       stride=d['stride'],panoramic_scan=geometry.panoramic)
        self.model=wrapper._model.eval()
        self.model.requires_grad_(False)
        if d['model']=='DR-SPAAM' and not hasattr(self.model.gate,'_memory'):
            raise RuntimeError('Unsupported upstream DR-SPAAM gate interface')

    @staticmethod
    def _key(request):
        stream,_,_=request
        return stream.geometry,stream.memory is None

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda:self.closed or self.queue)
                if not self.queue:
                    return
                first=self.queue.popleft()
                key=self._key(first)
                batch=[first]
                deadline=time.monotonic()+self.wait_s
                while len(batch)<self.max_batch:
                    match=next((i for i,x in enumerate(self.queue) if self._key(x)==key),None)
                    if match is not None:
                        batch.append(self.queue[match])
                        del self.queue[match]
                    elif self.closed or time.monotonic()>=deadline:
                        break
                    else:
                        self.condition.wait(timeout=max(0.,deadline-time.monotonic()))
            try:
                self._infer(batch)
            except Exception as exc:
                for stream,_,result in batch:
                    stream.memory=None
                    if not result.done():
                        result.set_exception(RuntimeError('Batched detector failed: '+str(exc)))

    def _infer(self,batch):
        start=time.perf_counter()
        geometry=batch[0][0].geometry
        self._init(geometry)
        stride=self.settings['stride']
        half=.5*np.deg2rad(geometry.fov_deg)
        phi=np.linspace(-half,half,geometry.count,dtype=np.float32)
        scans=[scan for _,scan,_ in batch]
        ct=np.stack([self.utils.scans_to_cutout(scan[None,...],phi,stride=stride,
                        centered=True,fixed=True,window_width=1.0,window_depth=.5,
                        num_cutout_pts=56,padding_val=29.99,area_mode=True) for scan in scans])
        with torch.no_grad():
            x=torch.from_numpy(ct).float().to(self.device)
            if self.settings['model']=='DR-SPAAM':
                gate=self.model.gate
                gate._memory=None if batch[0][0].memory is None else torch.cat([s.memory for s,_,_ in batch],0)
                try:
                    cls,reg,_=self.model(x,inference=True)
                    memory=gate._memory
                    for i,(stream,_,_) in enumerate(batch):
                        stream.memory=memory[i:i+1].clone()
                finally:
                    gate.reset()
            else:
                cls,reg=self.model(x)
            cls=cls.sigmoid().cpu().numpy()
            reg=reg.cpu().numpy()
        outputs=[]
        for i,(stream,scan,_) in enumerate(batch):
            xy,conf,_=self.utils.nms_predicted_center(scan[::stride],phi[::stride],cls[i,:,0],reg[i])
            xy=np.asarray(xy,dtype=np.float32).reshape(-1,2)
            conf=np.asarray(conf,dtype=np.float32).reshape(-1)
            if len(xy)!=len(conf) or not np.isfinite(xy).all() or not np.isfinite(conf).all():
                raise ValueError('Invalid detector output')
            xy=stream.geometry.detections_to_ros(xy[conf>=self.settings['confidence']])
            outputs.append(xy[np.linalg.norm(xy,axis=-1)<=self.settings['max_range']])
        self.stats['batches']+=1
        self.stats['frames']+=len(batch)
        self.stats['max_batch']=max(self.stats['max_batch'],len(batch))
        self.stats['inference_s']+=time.perf_counter()-start
        for (_,_,future),out in zip(batch,outputs):
            future.set_result(out)

    def close(self):
        with self.condition:
            self.closed=True
            self.condition.notify_all()
        self.thread.join()
