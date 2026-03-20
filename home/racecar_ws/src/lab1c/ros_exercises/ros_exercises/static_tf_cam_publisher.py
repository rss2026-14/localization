#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped
import tf2_ros


class StaticTFCamPublisher(Node):
    def __init__(self):
        super().__init__('static_tf_cam_publisher')

        # Parameters (optional, nice for testing)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('left_frame', 'left_cam')
        self.declare_parameter('right_frame', 'right_cam')
        self.declare_parameter('cam_offset_y', 0.05)  # meters (left is +y)

        base_frame = str(self.get_parameter('base_frame').value)
        left_frame = str(self.get_parameter('left_frame').value)
        right_frame = str(self.get_parameter('right_frame').value)
        cam_offset_y = float(self.get_parameter('cam_offset_y').value)

        self.broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        # base_link -> left_cam (left is +y)
        t_base_left = TransformStamped()
        t_base_left.header.stamp = self.get_clock().now().to_msg()
        t_base_left.header.frame_id = base_frame
        t_base_left.child_frame_id = left_frame
        t_base_left.transform.translation.x = 0.0
        t_base_left.transform.translation.y = +cam_offset_y
        t_base_left.transform.translation.z = 0.0
        # identity rotation
        t_base_left.transform.rotation.x = 0.0
        t_base_left.transform.rotation.y = 0.0
        t_base_left.transform.rotation.z = 0.0
        t_base_left.transform.rotation.w = 1.0

        # left_cam -> right_cam
        # right is 0.05 m to the RIGHT of base_link, i.e. -y relative to base_link.
        # Since left_cam is +0.05 m, the vector from left_cam to right_cam is:
        #   y = (-0.05) - (+0.05) = -0.10
        t_left_right = TransformStamped()
        t_left_right.header.stamp = self.get_clock().now().to_msg()
        t_left_right.header.frame_id = left_frame
        t_left_right.child_frame_id = right_frame
        t_left_right.transform.translation.x = 0.0
        t_left_right.transform.translation.y = -2.0 * cam_offset_y  # -0.10 when offset is 0.05
        t_left_right.transform.translation.z = 0.0
        # identity rotation
        t_left_right.transform.rotation.x = 0.0
        t_left_right.transform.rotation.y = 0.0
        t_left_right.transform.rotation.z = 0.0
        t_left_right.transform.rotation.w = 1.0

        # IMPORTANT: broadcast both in ONE call
        self.broadcaster.sendTransform([t_base_left, t_left_right])

        self.get_logger().info(
            f"Published static TFs once: {base_frame}->{left_frame} and {left_frame}->{right_frame}"
        )

        # Keep node alive (so the static broadcaster stays available)
        # No timers needed; static transforms are latched.
        # You can still shut down with Ctrl+C.
        

def main(args=None):
    rclpy.init(args=args)
    node = StaticTFCamPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

