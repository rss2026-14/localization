import numpy as np

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import csv
import os

from rclpy.node import Node
import rclpy

class BagPublisher(Node):
    def __init__(self):
        super().__init__('bag_publisher')

        self.laserpublisher = self.create_publisher(
            LaserScan,'/laserscan', 10)
        self.lasersubscription = self.create_subscription(
            LaserScan,'/scan',self.laserCall,10)

        self.csv_file = "test_error_localize.csv"
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["timestamp","nano","x","y"])

        self.subscription = self.create_subscription(
            Odometry,
            "/pf/pose/odom",
            self.odom_callback,
            10
        )
        self.subscription

    def laserCall(self,scan):
        scan.header.frame_id = "base_link_pf"
        self.laserpublisher.publish(scan)

    def odom_callback(self,odom):
        time=odom.header.stamp.sec
        nano=odom.header.stamp.nanosec
        x=odom.pose.pose.position.x
        y=odom.pose.pose.position.y

        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([time, nano, x, y])


def main(args=None):
    rclpy.init(args=args)
    bag_publisher = BagPublisher()
    rclpy.spin(bag_publisher)
    bag_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
