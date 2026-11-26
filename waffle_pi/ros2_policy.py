"""Host-side ROS 2 shadow inference; never publishes directly to /cmd_vel.

Input detection PoseArray must be a timestamped DR-SPAAM (or equivalent)
person-center detection stream. TF2 converts it to odom. Goals must be
given in map or odom. Verify perception, tracking, localization, timing and
safety externally before enabling command requests.
"""
import argparse
import math
import pickle
import sys
import time
from pathlib import Path
import numpy as np
import torch
from .spaces import spaces
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseArray, PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from tf2_ros import Buffer, TransformListener
from .tracking import Tracker, TrajectoryHistory
from .dynamics import DriveLimits, command_twist
from .native import native_twist,check_checkpoint,COMMISSION_V,COMMISSION_W


def rotation(q):
    a=np.array([q.x,q.y,q.z,q.w],dtype=float)
    norm=np.linalg.norm(a)
    if not np.isfinite(norm) or norm<1e-8:
        raise ValueError("Invalid quaternion")
    x,y,z,w=a/norm
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


class Shadow(Node):
    def __init__(self,opts,policy_args):
        super().__init__('crowdnav_waffle_policy')
        from rl.networks.model import Policy
        from gst_updated.scripts.wrapper.crowd_nav_interface_parallel import CrowdNavPredInterfaceMultiEnv
        self.opts=opts
        self.device=torch.device(opts.device)
        if self.device.type=='cuda':
            torch.cuda.set_device(self.device.index or 0)
        policy_args.num_processes=1
        policy_args.no_cuda=self.device.type=='cpu'
        self.n=opts.humans
        def box(shape):
            return spaces.Box(-np.inf,np.inf,shape=shape,dtype=np.float32)
        obs_spaces={'robot_node':box((1,7)),'temporal_edges':box((1,2)),
                'spatial_edges':box((self.n,12)),'visible_masks':box((self.n,)),
                'detected_human_num':box((1,))}
        self.model=Policy(obs_spaces,box((2,)),base='selfAttn_merge_srnn',base_kwargs=policy_args).to(self.device)
        from .checkpoints import load_weights,load_gst_args
        check_checkpoint(opts.checkpoint,opts.waffle_native_action,
            dict(waffle_speed=opts.speed,waffle_angular_speed=opts.angular_speed,
                 waffle_radius=opts.radius,waffle_sensor_range=opts.sensor_range,
                 waffle_humans=opts.humans,gst_adjacent_mask=opts.gst_adjacent_mask)
            if opts.waffle_native_action else None)
        self.model.load_state_dict(load_weights(opts.checkpoint,self.device),strict=True)
        self.model.eval()
        gst_args=load_gst_args(Path(opts.gst_dir)/'checkpoint'/'args.pickle')
        if gst_args.pred_seq_len!=5:
            raise ValueError("Expected five-step GST horizon")
        self.gst=CrowdNavPredInterfaceMultiEnv(opts.gst_dir,self.device,gst_args,1,adjacent_mask=opts.gst_adjacent_mask)
        self.history=TrajectoryHistory(self.n,gst_args.obs_seq_len)
        self.hidden={'human_node_rnn':torch.zeros(1,1,policy_args.human_node_rnn_size,device=self.device),
                     'human_human_edge_rnn':torch.zeros(1,1,policy_args.human_human_edge_rnn_size,device=self.device)}
        self.masks=torch.zeros(1,1,device=self.device)
        self.tracker=Tracker(self.n)
        self.odom=self.goal=self.detections=None
        self.odom_at=self.det_at=-float('inf')
        self.last_stamp=-float('inf')
        self.tf=Buffer()
        self.listener=TransformListener(self.tf,self)
        self.create_subscription(Odometry,'/odom',self.on_odom,qos_profile_sensor_data)
        self.create_subscription(PoseArray,opts.detection_topic,self.on_detections,qos_profile_sensor_data)
        self.create_subscription(PoseStamped,'/crowdnav/goal',self.on_goal,1)
        topic='/crowdnav/cmd_vel_requested' if opts.enable_requests else '/crowdnav/cmd_vel_shadow'
        self.pub=self.create_publisher(TwistStamped,topic,1)
        self.create_timer(.25,self.tick)
        self.get_logger().warning('Request mode; independent Pi gate required' if opts.enable_requests else 'Shadow mode; no motor requests')

    def on_odom(self,msg):
        self.odom,self.odom_at=msg,time.monotonic()

    def on_detections(self,msg):
        self.detections,self.det_at=msg,time.monotonic()

    def on_goal(self,msg):
        if msg.header.frame_id not in ('map','odom'):
            self.get_logger().error('Goal must be in map or odom')
            self.goal=None
        else:
            self.goal=msg
        self.clear_history()

    def goal_xy(self):
        p=self.goal.pose.position
        goal=np.array([p.x,p.y,p.z],dtype=float)
        if self.goal.header.frame_id!='odom':
            tf=self.tf.lookup_transform('odom','map',Time(),timeout=Duration(seconds=0.)).transform
            goal=rotation(tf.rotation)@goal+np.array([tf.translation.x,tf.translation.y,tf.translation.z])
        return goal[:2].astype(np.float32)

    def clear_history(self):
        self.history.clear()
        self.masks.zero_()
        self.tracker=Tracker(self.n)

    def publish(self,v,w):
        msg=TwistStamped()
        msg.header.stamp=self.get_clock().now().to_msg()
        msg.header.frame_id='base_link'
        msg.twist.linear.x,msg.twist.angular.z=float(v),float(w)
        self.pub.publish(msg)

    def fresh(self,header):
        stamp=header.stamp.sec+header.stamp.nanosec/1e9
        age=self.get_clock().now().nanoseconds/1e9-stamp
        return stamp>0 and -0.05<=age<=0.5

    @torch.no_grad()
    def tick(self):
        now=time.monotonic()
        try:
            if (self.goal is None or self.odom is None or self.detections is None
                    or now-self.odom_at>0.5 or now-self.det_at>0.5
                    or self.odom.header.frame_id!='odom'
                    or not self.fresh(self.odom.header)
                    or not self.fresh(self.detections.header)):
                self.clear_history()
                self.publish(0.,0.)
                return
            stamp=self.detections.header.stamp.sec+self.detections.header.stamp.nanosec/1e9
            if stamp<=self.last_stamp:
                # Do not insert duplicate detections as new trajectory samples.
                self.publish(0.,0.)
                return
            if stamp-self.last_stamp>.5:
                self.clear_history()
            self.last_stamp=stamp
            pos=self.odom.pose.pose.position
            xy=np.array([pos.x,pos.y],np.float32)
            rot=rotation(self.odom.pose.pose.orientation)
            heading=math.atan2(rot[1,0],rot[0,0])%(2*math.pi)
            goal=self.goal_xy()
            if not np.isfinite(xy).all() or not np.isfinite(goal).all():
                raise ValueError("Non-finite pose/goal")
            if np.linalg.norm(goal-xy)<self.opts.radius:
                self.goal=None
                self.clear_history()
                self.publish(0.,0.)
                return
            points=np.array([[p.position.x,p.position.y,p.position.z] for p in self.detections.poses],dtype=float).reshape(-1,3)
            frame=self.detections.header.frame_id
            if not frame:
                raise ValueError("Missing detection frame")
            if frame!='odom':
                tf=self.tf.lookup_transform('odom',frame,Time.from_msg(self.detections.header.stamp),timeout=Duration(seconds=0.0)).transform
                points=points@rotation(tf.rotation).T+np.array([tf.translation.x,tf.translation.y,tf.translation.z])
            points=points[:,:2]
            if not np.isfinite(points).all():
                raise ValueError('Invalid person detection coordinates')
            points=points[np.linalg.norm(points-xy,axis=-1)<=self.opts.sensor_range]
            reused=self.tracker.update(points,stamp)
            if self.tracker.overflow:
                raise RuntimeError("Unassociated detections exceed available track slots")
            rel=self.tracker.pos-xy
            seen=self.tracker.seen & (np.linalg.norm(rel,axis=-1)<=self.opts.sensor_range)
            self.history.append(stamp,self.tracker.pos,seen,reused)
            history,history_mask=self.history.sample(stamp)
            mask=torch.as_tensor(seen,device=self.device)
            pred,valid=self.gst.forward(torch.as_tensor(history,device=self.device),
                                        torch.as_tensor(history_mask,device=self.device))
            relative=torch.as_tensor(rel,device=self.device)
            relative=torch.where(mask[:,None],relative,15.)
            features=relative.repeat(1,6)
            future=(pred[0,:,:,:2]-torch.as_tensor(xy,device=self.device)).reshape(self.n,10)
            features[:,2:]=torch.where(valid[0].bool(),future,features[:,2:])
            order=torch.argsort(torch.linalg.vector_norm(relative,dim=-1).masked_fill(~mask,float('inf')))
            speed=self.odom.twist.twist.linear.x
            if not math.isfinite(speed):
                raise ValueError("Non-finite odometry velocity")
            robot=torch.tensor([pos.x,pos.y,self.opts.radius,*goal,self.opts.speed,heading],device=self.device,dtype=torch.float32)
            obs={'robot_node':robot.reshape(1,1,7),
                 'temporal_edges':torch.tensor([speed*math.cos(heading),speed*math.sin(heading)],device=self.device,dtype=torch.float32).reshape(1,1,2),
                 'spatial_edges':features[order][None],
                 'visible_masks':mask[order][None],
                 'detected_human_num':mask.sum().clamp_min(1).float().reshape(1,1)}
            _,action,_,self.hidden=self.model.act(obs,self.hidden,self.masks,deterministic=True)
            self.masks.fill_(1.)
            limits=DriveLimits(max_linear=self.opts.speed,max_angular=self.opts.angular_speed,
                               max_wheel=.26/.033 if self.opts.waffle_native_action else self.opts.speed/.033)
            raw=action[0].cpu().tolist()
            v,w=native_twist(raw,limits) if self.opts.waffle_native_action else command_twist(raw,heading,limits)
            if now+0.25<time.monotonic():
                raise RuntimeError("Inference deadline exceeded")
            self.publish(v,w)
        except Exception as exc:
            self.get_logger().warning('Policy unavailable: '+str(exc))
            self.clear_history()
            self.publish(0.,0.)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',default='trained_models/GST_predictor_rand/checkpoints/41665.pt')
    parser.add_argument('--gst-dir',required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--humans',type=int,default=4)
    parser.add_argument('--radius',type=float,default=.25)
    parser.add_argument('--speed',type=float,default=.26)
    parser.add_argument('--sensor-range',type=float,default=3.)
    parser.add_argument('--detection-topic',default='/dr_spaam_detections')
    parser.add_argument('--gst-adjacent-mask',action='store_true')
    parser.add_argument('--waffle-native-action',action='store_true')
    parser.add_argument('--angular-speed',type=float,default=1.82)
    parser.add_argument('--enable-requests',action='store_true')
    opts,ros_args=parser.parse_known_args()
    if opts.waffle_native_action:
        opts.speed=min(opts.speed,COMMISSION_V)
        opts.angular_speed=min(opts.angular_speed,COMMISSION_W)
    if not 0<opts.angular_speed<=1.82:
        raise ValueError('Invalid angular speed')
    if not 1<=opts.humans<=20 or not 0<opts.speed<=.26 or not 0<opts.radius or not 0<opts.sensor_range<=5:
        raise ValueError("Invalid Waffle configuration")
    old=sys.argv
    try:
        sys.argv=[old[0]]
        from arguments import get_args
        policy_args=get_args()
    finally:
        sys.argv=old
    rclpy.init(args=ros_args)
    node=Shadow(opts,policy_args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.publish(0.,0.)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__=='__main__':
    main()
