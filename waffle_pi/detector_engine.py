"""Shared DR-SPAAM inference used by ROS and sensor-in-loop simulation."""
import numpy as np


class DetectorEngine:
    def __init__(self,weights,model,device,stride,confidence,max_range,geometry):
        from dr_spaam.detector import Detector
        if device.startswith('cuda'):
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('Requested CUDA detector but CUDA unavailable')
            index=torch.device(device).index
            torch.cuda.set_device(torch.cuda.current_device() if index is None else index)
        self.model=model
        self.confidence,self.max_range=confidence,max_range
        self.geometry=geometry
        import torch
        benchmark=torch.backends.cudnn.benchmark
        try:
            self.detector=Detector(weights,model=model,gpu=device.startswith('cuda'),
                                   stride=stride,panoramic_scan=geometry.panoramic)
        finally:
            torch.backends.cudnn.benchmark=benchmark
        self.detector.set_laser_fov(geometry.fov_deg)

    def reset(self):
        if self.model=='DR-SPAAM':
            self.detector._model.gate.reset()

    def detect(self,ranges,range_min,range_max):
        scan=self.geometry.preprocess(ranges,range_min,range_max)
        xy,conf,_=self.detector(scan)
        xy=np.asarray(xy,dtype=np.float32).reshape(-1,2)
        conf=np.asarray(conf,dtype=np.float32).reshape(-1)
        if len(xy)!=len(conf) or not np.isfinite(xy).all() or not np.isfinite(conf).all():
            raise ValueError('Invalid detector output')
        xy=self.geometry.detections_to_ros(xy[conf>=self.confidence])
        return xy[np.linalg.norm(xy,axis=-1)<=self.max_range]
