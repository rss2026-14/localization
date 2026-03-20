import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from custom_msgs.msg import OpenSpace

class OpenSpacePublisher(Node):

    def __init__(self):
        super().__init__('open_space_publisher')
        # ---- Declare parameters (with defaults) ----
        self.declare_parameter('subscriber_topic', 'fake_scan')
        self.declare_parameter('publisher_topic', 'open_space')

        # ---- Read parameters ----
        self.sub_topic = str(self.get_parameter('subscriber_topic').value)
        self.pub_topic = str(self.get_parameter('publisher_topic').value)

        # Subscriber
        self.subscription = self.create_subscription(
            LaserScan,
            self.sub_topic,
            self.scan_callback,
            10
        )

        # Publishers
        #self.dist_pub = self.create_publisher(
        #    Float32,
        #    'open_space/distance',
        #    10
        #)
        #self.angle_pub = self.create_publisher(
        #    Float32,
        #    'open_space/angle',
        #    10
        #)
        self.open_space_pub = self.create_publisher(OpenSpace, self.pub_topic, 10)
        self.get_logger().info(
            f"Subscribing to '{self.sub_topic}', publishing OpenSpace to '{self.pub_topic}'"
        )

    def scan_callback(self, msg: LaserScan):
        if not msg.ranges:
            return

        # Find maximum range and its index
        max_range = max(msg.ranges)
        max_index = msg.ranges.index(max_range)

        # Compute corresponding angle
        angle = msg.angle_min + max_index * msg.angle_increment

        ## Publish distance
        #dist_msg = Float32()
        #dist_msg.data = float(max_range)
        #self.dist_pub.publish(dist_msg)

        ## Publish angle
        #angle_msg = Float32()
        #angle_msg.data = float(angle)
        #self.angle_pub.publish(angle_msg)
        
        # One message publish
        out = OpenSpace()
        out.distance = float(max_range)
        out.angle = float(angle)
        self.open_space_pub.publish(out)

        self.get_logger().info(
            #f"Open space: distance={max_range:.3f} m, angle={angle:.3f} rad"
            f"OpenSpace: distance={out.distance:.3f} m, angle={out.angle:.3f} rad"
        )


def main(args=None):
    rclpy.init(args=args)

    node = OpenSpacePublisher()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


