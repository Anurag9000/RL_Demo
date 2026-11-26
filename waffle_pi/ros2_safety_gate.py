"""Independent onboard ROS 2 safety gate; default output is diagnostic only.

Run on the Pi. Do not use --arm until commissioning, measured stopping
clearances, source exclusivity and a physical emergency-stop procedure
have been established. This software cannot guarantee safety.
"""
import argparse
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from .safety import safe_command


class Gate(Node):
    def __init__(self,args):
        super().__init__('crowdnav_safety_gate')
        self.args = args
        self.command = self.scan = None
        self.command_at = self.scan_at = -float('inf')
        self.estop = False
        self.create_subscription(TwistStamped,'/crowdnav/cmd_vel_requested',self.on_command,1)
        self.create_subscription(LaserScan,'/scan',self.on_scan,qos_profile_sensor_data)
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
        if self.command is None or self.scan is None:
            self.publish(0.,0.)
            return
        fresh = (now-self.command_at<0.5 and now-self.scan_at<0.5
                 and self.stamp_fresh(self.command.header,0.5)
                 and self.stamp_fresh(self.scan.header,0.5))
        v,w = safe_command(self.command.twist.linear.x,self.command.twist.angular.z,
                           self.scan.ranges,self.scan.range_min,self.scan.range_max,
                           fresh=fresh,armed=self.args.arm,estop=self.estop)
        self.publish(v,w)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm',action='store_true')
    parser.add_argument('--stamped',action='store_true',help='Use TwistStamped final command')
    args,ros_args = parser.parse_known_args()
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
