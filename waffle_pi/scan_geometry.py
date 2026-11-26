"""ROS LaserScan <-> DROW/DR-SPAAM scanner geometry, independent of ROS."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class ScanGeometry:
    count: int
    original_count: int
    angle_min: float
    increment: float
    center: float
    fov_deg: float
    panoramic: bool
    reverse: bool

    @classmethod
    def from_metadata(cls,count,angle_min,angle_max,increment):
        if count<32 or not all(map(math.isfinite,(angle_min,angle_max,increment))) or increment==0:
            raise ValueError("Invalid scan geometry")
        if abs(angle_min+(count-1)*increment-angle_max)>max(1e-4,abs(increment)*.25):
            raise ValueError("LaserScan angles disagree with sample count")
        original_count=count
        # Remove an inclusive duplicate 360-degree endpoint.
        if abs(abs(increment)*(count-1)-2*math.pi)<1e-4:
            count-=1
        width=abs(increment)*(count-1)
        if width>2*math.pi+1e-4:
            raise ValueError("Scan spans more than one revolution")
        panoramic=abs(abs(increment)*count-2*math.pi)<=max(.02,1.5*abs(increment))
        center=angle_min+(count-1)*increment/2.
        # Detector phi increases clockwise in its x-forward/y-right frame.
        return cls(count,original_count,float(angle_min),float(increment),center,
                   math.degrees(width),panoramic,increment>0.)

    def preprocess(self,ranges,range_min,range_max):
        if not all(map(math.isfinite,(range_min,range_max))) or range_min<0 or range_max<=range_min:
            raise ValueError("Invalid scan range")
        raw=np.asarray(ranges,dtype=np.float32)
        if raw.shape!=(self.original_count,):
            raise ValueError("LaserScan length changed")
        raw=raw[:self.count]
        valid=np.isfinite(raw)&(raw>=range_min)&(raw<=range_max)&(raw>0.)
        if not valid.any():
            raise ValueError("No finite valid LiDAR return; detection unavailable")
        # DR-SPAAM's far-range padding convention; never pass NaN/inf.
        scan=np.where(valid,raw,29.99).astype(np.float32)
        return np.ascontiguousarray(scan[::-1] if self.reverse else scan)

    def detections_to_ros(self,xy):
        xy=np.asarray(xy,dtype=np.float32).reshape(-1,2)
        if not np.isfinite(xy).all():
            raise ValueError("Non-finite detector centers")
        c,s=math.cos(self.center),math.sin(self.center)
        # Reflect y-right to y-left, then rotate the scan center into
        # the actual ROS laser frame (which need not start at -pi).
        return xy@np.array([[c,s],[s,-c]],dtype=np.float32)
