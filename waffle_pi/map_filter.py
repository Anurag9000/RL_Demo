"""Timestamped ROS 2 map-based LiDAR background filter.

Produces a separate detector scan; never changes raw /scan for safety.
Unknown cells and out-of-map returns are preserved.
"""
import argparse
import copy
import math
import numpy as np
from scipy.ndimage import distance_transform_edt
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy,qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from tf2_ros import Buffer,TransformListener


def rotation(q):
    a=np.array([q.x,q.y,q.z,q.w],dtype=float)
    norm=np.linalg.norm(a)
    if not np.isfinite(norm) or norm<1e-8:
        raise ValueError("Invalid quaternion")
    x,y,z,w=a/norm
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def static_mask(points,grid,distance,origin,rot,resolution,threshold):
    local=(points-origin)@rot
    cell=np.floor(local[:,:2]/resolution).astype(np.int64)
    x,y=cell.T
    inside=(x>=0)&(x<grid.shape[1])&(y>=0)&(y<grid.shape[0])
    result=np.zeros(len(points),bool)
    xi,yi=x[inside],y[inside]
    result[inside]=(grid[yi,xi]>=0)&(distance[yi,xi]<=threshold)
    return result


class Filter(Node):
    def __init__(self,threshold):
        super().__init__('crowdnav_map_filter')
        self.threshold=threshold
        self.grid=None
        self.tf=Buffer()
        self.listener=TransformListener(self.tf,self)
        qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid,'/map',self.on_map,qos)
        self.create_subscription(LaserScan,'/scan',self.on_scan,qos_profile_sensor_data)
        self.pub=self.create_publisher(LaserScan,'/crowdnav/scan_filtered',qos_profile_sensor_data)

    def on_map(self,msg):
        self.grid=None
        if not msg.header.frame_id or msg.info.resolution<=0 or msg.info.width<1 or msg.info.height<1:
            return
        grid=np.asarray(msg.data,dtype=np.int16).reshape(msg.info.height,msg.info.width)
        occupied=grid>=65
        distance=distance_transform_edt(~occupied)*msg.info.resolution if occupied.any() else np.full(grid.shape,np.inf)
        p=msg.info.origin.position
        self.map_frame=msg.header.frame_id
        self.origin=np.array([p.x,p.y,p.z])
        self.rot=rotation(msg.info.origin.orientation)
        self.resolution=msg.info.resolution
        self.grid,self.distance=grid,distance

    def on_scan(self,msg):
        if self.grid is None:
            return
        try:
            ranges=np.asarray(msg.ranges,dtype=float)
            valid=np.isfinite(ranges)&(ranges>=msg.range_min)&(ranges<=msg.range_max)
            if not np.isfinite(msg.angle_increment) or msg.angle_increment==0.:
                raise ValueError("Invalid scan angles")
            # A scan is acquired across a revolution: one transform is an
            # approximation. Log motion distortion and deskew if needed.
            tf=self.tf.lookup_transform(self.map_frame,msg.header.frame_id,
                       Time.from_msg(msg.header.stamp),timeout=Duration(seconds=0.)).transform
            angles=msg.angle_min+np.arange(len(ranges))*msg.angle_increment
            r,a=ranges[valid],angles[valid]
            points=np.stack((r*np.cos(a),r*np.sin(a),np.zeros_like(r)),-1)
            points=points@rotation(tf.rotation).T+np.array([tf.translation.x,tf.translation.y,tf.translation.z])
            mask=static_mask(points,self.grid,self.distance,self.origin,self.rot,self.resolution,self.threshold)
            idx=np.flatnonzero(valid)[mask]
            out=copy.deepcopy(msg)
            ranges[idx]=float('inf')
            out.ranges=ranges.tolist()
            if len(msg.intensities)==len(ranges):
                intensities=list(msg.intensities)
                for i in idx:
                    intensities[i]=0.
                out.intensities=intensities
            self.pub.publish(out)
        except Exception as exc:
            self.get_logger().warning('Filtered scan unavailable: '+str(exc))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--threshold',type=float,default=.10)
    args,ros_args=parser.parse_known_args()
    if not math.isfinite(args.threshold) or not 0<=args.threshold<=.25:
        raise ValueError("Invalid background threshold")
    rclpy.init(args=ros_args)
    node=Filter(args.threshold)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__=='__main__':
    main()
