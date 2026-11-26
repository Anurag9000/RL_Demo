"""Slow sensor-in-loop evaluation and fine-tuning using deployed components.

Simulated returns are still geometric approximations. Detector compute/network
latency and policy hidden-state resets during faults are not fully modeled.
"""
import math
import numpy as np
import torch
from .scan_geometry import ScanGeometry
from .detector_engine import DetectorEngine
from .tracking import Tracker,TrajectoryHistory


class NeuralPerception:
    def __init__(self,args,profile,n,slots,device,engine_factory=DetectorEngine):
        self.n,self.slots,self.device=n,slots,device
        self.laser=profile['laser']
        s=profile['detector']
        if s['model'] not in ('DR-SPAAM','DROW3') or not isinstance(s['stride'],int) or s['stride']<1 or not 0<=s['confidence']<=1 or not 0<s['max_range']<=self.laser['range_max']:
            raise ValueError('Invalid neural detector settings')
        r=self.laser['rays']
        inc=2*math.pi/r
        start=self.laser['angle_min']
        geometry=ScanGeometry.from_metadata(r,start,start+(r-1)*inc,inc)
        self.engines=[engine_factory(args.waffle_detector_weights,s['model'],str(device),s['stride'],s['confidence'],s['max_range'],geometry) for _ in range(n)]
        self.trackers=[Tracker(slots) for _ in range(n)]
        self.histories=[TrajectoryHistory(slots,5) for _ in range(n)]
        self.latest=[None]*n
        self.last=np.full(n,-np.inf)
        self.fault=torch.ones(n,device=device,dtype=torch.bool)

    def reset(self,done):
        for i in done.nonzero().flatten().tolist():
            self.engines[i].reset()
            self.trackers[i]=Tracker(self.slots)
            self.histories[i].clear()
            self.latest[i]=None
            self.last[i]=-np.inf
        self.fault|=done

    def on_scan(self,scan,pose,stamp,sample):
        for i in sample.nonzero().flatten().tolist():
            try:
                t=float(stamp[i])
                points=self.engines[i].detect(scan[i].cpu().numpy(),self.laser['range_min'],self.laser['range_max'])
                p=pose[i].cpu().numpy()
                angle=float(p[2])+self.laser['yaw']
                c,s=math.cos(angle),math.sin(angle)
                x,y=self.laser['offset_x'],self.laser['offset_y']
                cp,sp=math.cos(float(p[2])),math.sin(float(p[2]))
                origin=p[:2]+np.array([cp*x-sp*y,sp*x+cp*y])
                points=points@np.array([[c,s],[-s,c]])+origin
                self.latest[i]=(t,points)
            except (ValueError,RuntimeError):
                self.latest[i]=None
                self.engines[i].reset()

    def observe(self,estimated,bias,sensor_range):
        relative=np.full((self.n,self.slots,2),15.,np.float32)
        seen=np.zeros((self.n,self.slots),bool)
        history=np.full((self.n,self.slots,5,2),-999.,np.float32)
        mask=np.zeros((self.n,self.slots,5,1),np.float32)
        fault=np.ones(self.n,bool)
        for i in range(self.n):
            item=self.latest[i]
            if item is None or item[0]<=self.last[i]:
                self.histories[i].clear()
                self.trackers[i]=Tracker(self.slots)
                continue
            stamp,points=item
            self.last[i]=stamp
            xy=estimated[i].cpu().numpy()
            points=points+bias[i].cpu().numpy()
            points=points[np.linalg.norm(points-xy,axis=-1)<=sensor_range]
            tr=self.trackers[i]
            reused=tr.update(points,stamp)
            if tr.overflow:
                self.histories[i].clear()
                self.trackers[i]=Tracker(self.slots)
                continue
            rel=tr.pos-xy
            valid=tr.seen&(np.linalg.norm(rel,axis=-1)<=sensor_range)
            self.histories[i].append(stamp,tr.pos,valid,reused)
            h,m=self.histories[i].sample(stamp)
            relative[i,valid]=rel[valid]
            seen[i]=valid
            history[i],mask[i]=h[0],m[0]
            fault[i]=False
        self.fault=torch.as_tensor(fault,device=self.device)
        return (torch.as_tensor(relative,device=self.device),torch.as_tensor(seen,device=self.device),
                torch.as_tensor(history,device=self.device),torch.as_tensor(mask,device=self.device))
