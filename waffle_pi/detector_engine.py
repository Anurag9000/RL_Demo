"""Shared DR-SPAAM inference used by ROS and sensor-in-loop simulation."""
import numpy as np
import threading
from contextlib import nullcontext
import torch

_init_lock=threading.Lock()


class DetectorEngine:
    def __init__(self,weights,model,device,stride,confidence,max_range,geometry):
        from dr_spaam.detector import Detector
        self.device=torch.device(device)
        if self.device.type=='cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('Requested CUDA detector but CUDA unavailable')
            if self.device.index is None:
                self.device=torch.device('cuda',torch.cuda.current_device())
        self.model=model
        self.confidence,self.max_range=confidence,max_range
        self.geometry=geometry
        with _init_lock,self.context():
            benchmark=torch.backends.cudnn.benchmark
            try:
                self.detector=Detector(weights,model=model,gpu=self.device.type=='cuda',
                                       stride=stride,panoramic_scan=geometry.panoramic)
            finally:
                torch.backends.cudnn.benchmark=benchmark
        self.detector.set_laser_fov(geometry.fov_deg)

    def context(self):
        return torch.cuda.device(self.device) if self.device.type=='cuda' else nullcontext()

    def reset(self):
        if self.model=='DR-SPAAM':
            self.detector._model.gate.reset()

    def detect(self,ranges,range_min,range_max):
        scan=self.geometry.preprocess(ranges,range_min,range_max)
        # A different thread-pool thread can own this world on each step.
        with self.context():
            xy,conf,_=self.detector(scan)
        xy=np.asarray(xy,dtype=np.float32).reshape(-1,2)
        conf=np.asarray(conf,dtype=np.float32).reshape(-1)
        if len(xy)!=len(conf) or not np.isfinite(xy).all() or not np.isfinite(conf).all():
            raise ValueError('Invalid detector output')
        xy=self.geometry.detections_to_ros(xy[conf>=self.confidence])
        return xy[np.linalg.norm(xy,axis=-1)<=self.max_range]
