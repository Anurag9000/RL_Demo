"""Independent onboard ROS 2 safety gate; default output is diagnostic only.

Run on the Pi. Do not use --arm until commissioning, measured stopping
clearances, source exclusivity and a physical emergency-stop procedure
have been established. This software cannot guarantee safety.
"""
import argparse
import json
import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from .safety import safe_command,scan_coverage


class Gate(Node):
    def __init__(self,args):
        super().__init__('crowdnav_safety_gate')
        self.args = args
        self.command = self.scan = self.odom = None
        self.command_at = self.scan_at = self.odom_at = -float('inf')
        self.params=args.safety_params
        self.estop = False
        self.create_subscription(TwistStamped,'/crowdnav/cmd_vel_requested',self.on_command,1)
        self.create_subscription(LaserScan,'/scan',self.on_scan,qos_profile_sensor_data)
        self.create_subscription(Odometry,'/odom',self.on_odom,qos_profile_sensor_data)
        self.create_subscription(Bool,'/crowdnav/estop',self.on_estop,1)
        msg_type = TwistStamped if args.stamped else Twist
        topic = '/cmd_vel' if args.arm else '/crowdnav/cmd_vel_gate_dry_run'
        self.pub = self.create_publisher(msg_type,topic,1)
        self.create_timer(0.05,self.tick)
        self.get_logger().warning('ARMED: supervised commissioning only' if args.arm else 'DRY RUN: no motor commands')

    def on_command(self,msg):
        self.command,self.command_at = msg,time.monotonic()

    def on_scan(self,msg):
        self.scan,self.scan_at = msg,time.monotonic()

    def on_odom(self,msg):
        self.odom,self.odom_at=msg,time.monotonic()

    def on_estop(self,msg):
        self.estop = self.estop or msg.data

    def stamp_fresh(self,header,limit):
        stamp = header.stamp.sec+header.stamp.nanosec/1e9
        age = self.get_clock().now().nanoseconds/1e9-stamp
        return stamp>0 and -0.05<=age<=limit

    def publish(self,v,w):
        msg = Twist()
        msg.linear.x,msg.angular.z = float(v),float(w)
        if self.args.stamped:
            out = TwistStamped()
            out.header.stamp = self.get_clock().now().to_msg()
            out.header.frame_id = 'base_link'
            out.twist = msg
        else:
            out = msg
        self.pub.publish(out)

    def tick(self):
        now = time.monotonic()
        if self.command is None or self.scan is None or self.odom is None:
            self.publish(0.,0.)
            return
        fresh = (now-self.command_at<0.5 and now-self.scan_at<0.5 and now-self.odom_at<0.5
                 and self.stamp_fresh(self.command.header,0.5)
                 and self.stamp_fresh(self.scan.header,0.5)
                 and self.stamp_fresh(self.odom.header,0.5)
                 and self.command.header.frame_id=='base_link'
                 and scan_coverage(self.scan.ranges,self.scan.angle_increment,
                      self.scan.range_min,self.scan.range_max))
        cmd=self.command.twist
        if any(not math.isfinite(x) or abs(x)>1e-6 for x in
               (cmd.linear.y,cmd.linear.z,cmd.angular.x,cmd.angular.y)):
            fresh=False
        # The diagnostic publisher can show the authorized command without
        # ever sending it to the hardware. Arming changes the output topic.
        v,w = safe_command(cmd.linear.x,cmd.angular.z,
                           self.scan.ranges,self.scan.range_min,self.scan.range_max,
                           fresh=fresh,armed=True,estop=self.estop,
                           measured_speed=self.odom.twist.twist.linear.x,
                           measured_angular=self.odom.twist.twist.angular.z,**self.params)
        self.publish(v,w)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm',action='store_true')
    parser.add_argument('--stamped',action='store_true',help='Use TwistStamped final command')
    parser.add_argument('--safety-calibration',default=None)
    args,ros_args = parser.parse_known_args()
    args.safety_params={}
    if args.safety_calibration:
        with open(args.safety_calibration) as handle:
            data=json.load(handle)
        if data.get('schema')!='waffle_safety_commissioning_v1':
            raise ValueError("Expected separate measured safety commissioning record")
        args.safety_params={k:float(data[k]) for k in ('footprint','sensor_offset','margin','latency','braking')}
        if (any(not math.isfinite(v) or v<=0 for v in args.safety_params.values())
                or args.safety_params['footprint']<.20 or args.safety_params['latency']<.5):
            raise ValueError("Invalid conservative stopping parameters")
    if args.arm and (not args.safety_calibration or data.get('supervised_validated') is not True
                      or data.get('firmware_watchdog_tested') is not True):
        raise RuntimeError("Motor output requires a separate supervised safety commissioning record")
    rclpy.init(args=ros_args)
    node = Gate(args)
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


if __name__ == '__main__':
    main()
