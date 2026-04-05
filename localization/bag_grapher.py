import numpy as np

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import (
    PoseWithCovarianceStamped,
    PoseArray,
    Pose,
    TransformStamped,
)
from ackermann_msgs.msg import AckermannDriveStamped
import tf_transformations
from tf2_ros import TransformBroadcaster

from rclpy.node import Node
import rclpy

class BagPublisher(Node):
    def __init__(self):
        super().__init__('bag_publisher')
        #self.publisher = self.create_publisher(
        #    AckermannDriveStamped,'/drive', 10)
        #self.subscription = self.create_subscription(
        #    AckermannDriveStamped,'/vesc/low_level/input/teleop',self.driveCall,10)
        self.laserpublisher = self.create_publisher(
            LaserScan,'/laserscan', 10)
        self.lasersubscription = self.create_subscription(
            LaserScan,'/scan',self.laserCall,10)

    #def driveCall(self,drive):
    #    drive.header.frame_id = "base_link"
    #    self.publisher.publish(drive)

    def laserCall(self,scan):
        scan.header.frame_id = "base_link_pf"
        self.laserpublisher.publish(scan)


def main(args=None):
    rclpy.init(args=args)
    bag_publisher = BagPublisher()
    rclpy.spin(bag_publisher)
    bag_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
