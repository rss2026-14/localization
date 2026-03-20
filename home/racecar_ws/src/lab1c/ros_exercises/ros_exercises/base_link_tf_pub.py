#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped
import tf2_ros


def quat_to_rotmat(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n == 0.0:
        return np.eye(3)
    x, y, z, w = x/n, y/n, z/n, w/n

    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    return np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),   2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),       1 - 2*(xx + yy)]
    ], dtype=float)


def rotmat_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    tr = float(np.trace(R))
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
    return float(x), float(y), float(z), float(w)


def tf_to_matrix(tf_msg: TransformStamped) -> np.ndarray:
    t = tf_msg.transform.translation
    q = tf_msg.transform.rotation
    R = quat_to_rotmat(q.x, q.y, q.z, q.w)

    T = np.eye(4, dtype=float)
    T[0:3, 0:3] = R
    T[0:3, 3] = np.array([t.x, t.y, t.z], dtype=float)
    return T


class BaseLink2Publisher(Node):
    """
    Listens for odom -> left_cam from TF, composes with left_cam -> base_link,
    and publishes odom -> base_link_2 dynamically.
    """
    def __init__(self):
        super().__init__('base_link_tf_pub')

        # Params
        self.declare_parameter('rate_hz', 30.0)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('left_cam_frame', 'left_cam')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('base2_frame', 'base_link_2')
        self.declare_parameter('cam_offset_y', 0.05)  # must match static publisher

        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.odom_frame = str(self.get_parameter('odom_frame').value)
        self.left_cam_frame = str(self.get_parameter('left_cam_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.base2_frame = str(self.get_parameter('base2_frame').value)
        self.cam_offset_y = float(self.get_parameter('cam_offset_y').value)

        if self.rate_hz <= 0.0:
            self.rate_hz = 30.0

        # TF listener + broadcaster
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.broadcaster = tf2_ros.TransformBroadcaster(self)

        # Precompute base_link -> left_cam (static), then invert to get left_cam -> base_link
        T_base_left = np.eye(4, dtype=float)
        T_base_left[1, 3] = +self.cam_offset_y  # left is +y
        self.T_left_base = np.linalg.inv(T_base_left)  # <-- hint used here

        self.timer = self.create_timer(1.0 / self.rate_hz, self.tick)
        self.get_logger().info(
            f"Publishing dynamic TF {self.odom_frame}->{self.base2_frame} "
            f"from {self.odom_frame}->{self.left_cam_frame} and static {self.base_frame}->{self.left_cam_frame}"
        )

    def tick(self):
        # Listen for odom -> left_cam
        try:
            tf_odom_left = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.left_cam_frame,
                rclpy.time.Time()  # latest
            )
        except Exception:
            # Don’t spam warnings every tick
            return

        T_odom_left = tf_to_matrix(tf_odom_left)

        # Compose: odom->base_link_2 = (odom->left_cam) * (left_cam->base_link)
        T_odom_base2 = T_odom_left @ self.T_left_base

        # Broadcast odom -> base_link_2
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.odom_frame
        msg.child_frame_id = self.base2_frame

        msg.transform.translation.x = float(T_odom_base2[0, 3])
        msg.transform.translation.y = float(T_odom_base2[1, 3])
        msg.transform.translation.z = float(T_odom_base2[2, 3])

        qx, qy, qz, qw = rotmat_to_quat(T_odom_base2[0:3, 0:3])
        msg.transform.rotation.x = qx
        msg.transform.rotation.y = qy
        msg.transform.rotation.z = qz
        msg.transform.rotation.w = qw

        self.broadcaster.sendTransform(msg)


def main(args=None):
    rclpy.init(args=args)
    node = BaseLink2Publisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

