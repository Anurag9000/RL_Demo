"""Host-side ROS 2 shadow inference; never publishes directly to /cmd_vel.

Input detection PoseArray must be a timestamped DR-SPAAM (or equivalent)
person-center detection stream. TF2 converts it to odom. Goals must be
given in odom. Verify perception, tracking, localization, timing and
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
import gym
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseArray, PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from tf2_ros import Buffer, TransformListener
from .tracking import Tracker
from .dynamics import DriveLimits, command_twist


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
        policy_args.num_processes=1
        policy_args.no_cuda=self.device.type=='cpu'
        self.n=opts.humans
        def box(shape):
            return gym.spaces.Box(-np.inf,np.inf,shape=shape,dtype=np.float32)
        spaces={'robot_node':box((1,7)),'temporal_edges':box((1,2)),
                'spatial_edges':box((self.n,12)),'visible_masks':box((self.n,)),
                'detected_human_num':box((1,))}
        self.model=Policy(spaces,box((2,)),base='selfAttn_merge_srnn',base_kwargs=policy_args).to(self.device)
        self.model.load_state_dict(torch.load(opts.checkpoint,map_location=self.device),strict=True)
        self.model.eval()
        with open(Path(opts.gst_dir)/'checkpoint'/'args.pickle','rb') as handle:
            gst_args=pickle.load(handle)
        if gst_args.pred_seq_len!=5:
            raise ValueError("Expected five-step GST horizon")
        self.gst=CrowdNavPredInterfaceMultiEnv(opts.gst_dir,self.device,gst_args,1)
        self.history=torch.full((1,self.n,gst_args.obs_seq_len,2),-999.,device=self.device)
        self.valid=torch.zeros((1,self.n,gst_args.obs_seq_len,1),device=self.device)
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
        if msg.header.frame_id!='odom':
            self.get_logger().error('Goal must be transformed into odom first')
            self.goal=None
        else:
            self.goal=np.array([msg.pose.position.x,msg.pose.position.y],np.float32)
        self.clear_history()

    def clear_history(self):
        self.history.fill_(-999.)
        self.valid.zero_()
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
            self.last_stamp=stamp
            pos=self.odom.pose.pose.position
            xy=np.array([pos.x,pos.y],np.float32)
            rot=rotation(self.odom.pose.pose.orientation)
            heading=math.atan2(rot[1,0],rot[0,0])
            if not np.isfinite(xy).all() or not np.isfinite(self.goal).all():
                raise ValueError("Non-finite pose/goal")
            if np.linalg.norm(self.goal-xy)<self.opts.radius:
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
            reused=self.tracker.update(points[:,:2],stamp)
            reuse=torch.as_tensor(reused,device=self.device)
            self.history.masked_fill_(reuse[None,:,None,None],-999.)
            self.valid.masked_fill_(reuse[None,:,None,None],0.)
            self.history=torch.roll(self.history,-1,2)
            self.valid=torch.roll(self.valid,-1,2)
            rel=self.tracker.pos-xy
            seen=self.tracker.seen & (np.linalg.norm(rel,axis=-1)<=self.opts.sensor_range)
            current=torch.as_tensor(self.tracker.pos,device=self.device)
            mask=torch.as_tensor(seen,device=self.device)
            self.history[0,:,-1]=torch.where(mask[:,None],current,-999.)
            self.valid[0,:,-1,0]=mask.float()
            pred,valid=self.gst.forward(self.history,self.valid)
            relative=torch.as_tensor(rel,device=self.device)
            relative=torch.where(mask[:,None],relative,15.)
            features=relative.repeat(1,6)
            future=(pred[0,:,:,:2]-torch.as_tensor(xy,device=self.device)).reshape(self.n,10)
            features[:,2:]=torch.where(valid[0].bool(),future,features[:,2:])
            order=torch.argsort(torch.linalg.vector_norm(relative,dim=-1))
            speed=self.odom.twist.twist.linear.x
            robot=torch.tensor([pos.x,pos.y,self.opts.radius,*self.goal,self.opts.speed,heading],device=self.device,dtype=torch.float32)
            obs={'robot_node':robot.reshape(1,1,7),
                 'temporal_edges':torch.tensor([speed*math.cos(heading),speed*math.sin(heading)],device=self.device,dtype=torch.float32).reshape(1,1,2),
                 'spatial_edges':features[order][None],
                 'visible_masks':mask[order][None],
                 'detected_human_num':mask.sum().clamp_min(1).float().reshape(1,1)}
            _,action,_,self.hidden=self.model.act(obs,self.hidden,self.masks,deterministic=True)
            self.masks.fill_(1.)
            if now+0.25<time.monotonic():
                raise RuntimeError("Inference deadline exceeded")
            limits=DriveLimits(max_linear=self.opts.speed,max_wheel=self.opts.speed/0.033)
            v,w=command_twist(action[0].cpu().tolist(),heading,limits)
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
    parser.add_argument('--enable-requests',action='store_true')
    opts,ros_args=parser.parse_known_args()
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
