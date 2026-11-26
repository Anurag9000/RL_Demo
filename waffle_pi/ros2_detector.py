"""ROS 2 adapter for the upstream DR-SPAAM Detector Python API.

Compatible with the 2D_lidar_person_detection API at tree
d12e90225d52ba6c85d20061f1e7286d6a64aade. Install upstream separately.
No motor commands, no ROS 1 dependency, no model weights bundled.
"""
import argparse
import math
from pathlib import Path
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Pose,PoseArray
from sensor_msgs.msg import LaserScan
from .scan_geometry import ScanGeometry
from .detector_engine import DetectorEngine


class DetectorNode(Node):
    def __init__(self,args):
        super().__init__('crowdnav_dr_spaam')
        self.args=args
        self.geometry=self.detector=None
        self.last_stamp=None
        self.profile=None
        if args.fidelity_profile:
            import hashlib
            from .fidelity import read_profile
            self.profile,_=read_profile(args.fidelity_profile)
            d=self.profile['detector']
            if hashlib.sha256(Path(args.weights).read_bytes()).hexdigest()!=d['weights_sha256'] or any(getattr(args,k)!=d[k] for k in ('model','stride','confidence','max_range')):
                raise ValueError('Detector runtime differs from fidelity profile')
        self.pub=self.create_publisher(PoseArray,args.output_topic,qos_profile_sensor_data)
        self.create_subscription(LaserScan,args.input_topic,self.on_scan,qos_profile_sensor_data)
        self.get_logger().info('Detector is perception-only; waiting for LaserScan')

    def initialize(self,geometry):
        self.detector=DetectorEngine(self.args.weights,self.args.model,self.args.device,
                                     self.args.stride,self.args.confidence,self.args.max_range,geometry)
        self.geometry=geometry

    def on_scan(self,msg):
        try:
            start=time.monotonic()
            stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
            now=self.get_clock().now().nanoseconds/1e9
            if not msg.header.frame_id or stamp<=0 or not -.05<=now-stamp<=self.args.max_age:
                raise ValueError("Stale or invalid scan header")
            if self.last_stamp is not None and stamp<=self.last_stamp:
                raise ValueError("Repeated or out-of-order scan")
            if self.profile is not None:
                laser=self.profile['laser']
                if (len(msg.ranges)!=laser['rays'] or not math.isclose(msg.range_min,laser['range_min'],abs_tol=1e-5)
                        or not math.isclose(msg.range_max,laser['range_max'],abs_tol=1e-5)
                        or abs(math.remainder(msg.angle_min-laser['angle_min'],2*math.pi))>1e-4):
                    raise ValueError('Live LDS metadata differs from fidelity profile')
            geometry=ScanGeometry.from_metadata(len(msg.ranges),msg.angle_min,msg.angle_max,msg.angle_increment)
            if self.detector is None:
                self.initialize(geometry)
            elif geometry!=self.geometry:
                # No stale cached angle grid or neural memory.
                raise ValueError("Scan geometry changed; restart detector after verifying sensor")
            if (self.last_stamp is not None and stamp-self.last_stamp>self.args.memory_gap
                    and self.args.model=='DR-SPAAM'):
                # The pinned upstream model uses a spatial attention memory.
                self.detector.reset()
            self.last_stamp=stamp
            xy=self.detector.detect(msg.ranges,msg.range_min,msg.range_max)
            if (time.monotonic()-start>self.args.deadline
                    or self.get_clock().now().nanoseconds/1e9-stamp>self.args.max_age):
                raise RuntimeError("Detector inference exceeded freshness/deadline")
            out=PoseArray()
            out.header=msg.header
            for x,y in xy:
                p=Pose()
                p.position.x,p.position.y=float(x),float(y)
                p.orientation.w=1.
                out.poses.append(p)
            self.pub.publish(out)
        except Exception as exc:
            # Do not publish a false empty crowd on detector failure.
            self.get_logger().warning('Person detections unavailable: '+str(exc))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',required=True)
    p.add_argument('--fidelity-profile')
    p.add_argument('--model',choices=['DR-SPAAM','DROW3'],default='DR-SPAAM')
    p.add_argument('--device',default='cpu')
    p.add_argument('--input-topic',default='/scan')
    p.add_argument('--output-topic',default='/dr_spaam_detections')
    p.add_argument('--stride',type=int,default=1)
    p.add_argument('--confidence',type=float,default=.5)
    p.add_argument('--max-range',type=float,default=3.)
    p.add_argument('--max-age',type=float,default=.5)
    p.add_argument('--deadline',type=float,default=.25)
    p.add_argument('--memory-gap',type=float,default=.75)
    args,ros_args=p.parse_known_args()
    if (not Path(args.weights).is_file() or args.stride<1
            or not 0<=args.confidence<=1 or not 0<args.max_range<=5
            or not all(math.isfinite(x) and x>0 for x in (args.max_age,args.deadline,args.memory_gap))
            or not (args.device=='cpu' or args.device=='cuda' or args.device.startswith('cuda:'))):
        p.error('Invalid detector configuration or missing weights')
    rclpy.init(args=ros_args)
    node=DetectorNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__=='__main__':
    main()
