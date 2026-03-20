#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped
import tf2_ros


def quat_to_rotmat(x: float, y: float, z: float, w: float) -> np.ndarray:
    """
    Convert quaternion (x,y,z,w) to 3x3 rotation matrix.
    """
    # Normalize to be safe
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n == 0.0:
        return np.eye(3)
    x, y, z, w = x/n, y/n, z/n, w/n

    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    R = np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),   2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),       1 - 2*(xx + yy)]
    ], dtype=float)
    return R


def rotmat_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """
    Convert 3x3 rotation matrix to quaternion (x,y,z,w).
    Numerically stable enough for typical TF use.
    """
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
    """
    Convert TransformStamped (translation + quaternion) to 4x4 homogeneous matrix.
    """
    t = tf_msg.transform.translation
    q = tf_msg.transform.rotation
    R = quat_to_rotmat(q.x, q.y, q.z, q.w)

    T = np.eye(4, dtype=float)
    T[0:3, 0:3] = R
    T[0:3, 3] = np.array([t.x, t.y, t.z], dtype=float)
    return T


class DynamicTFCamPublisher(Node):
    def __init__(self):
        super().__init__('dynamic_tf_cam_publisher')

        # --- Parameters (optional) ---
        self.declare_parameter('rate_hz', 30.0)
        self.declare_parameter('parent_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('left_frame', 'left_cam')
        self.declare_parameter('right_frame', 'right_cam')
        self.declare_parameter('cam_offset_y', 0.05)  # meters

        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.parent_frame = str(self.get_parameter('parent_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.left_frame = str(self.get_parameter('left_frame').value)
        self.right_frame = str(self.get_parameter('right_frame').value)
        self.cam_offset_y = float(self.get_parameter('cam_offset_y').value)

        if self.rate_hz <= 0.0:
            self.rate_hz = 30.0

        # --- TF2 ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # --- Precompute base_link -> cameras as 4x4 matrices ---
        # Coordinate convention given:
        # forward = +x, left = +y
        self.T_base_left = np.eye(4, dtype=float)
        self.T_base_left[1, 3] = +self.cam_offset_y  # +y

        self.T_base_right = np.eye(4, dtype=float)
        self.T_base_right[1, 3] = -self.cam_offset_y  # -y

        # Timer
        self.timer = self.create_timer(1.0 / self.rate_hz, self.tick)
        self.get_logger().info(
            f"Publishing TF: {self.parent_frame}->{self.left_frame} and "
            f"{self.left_frame}->{self.right_frame} at ~{self.rate_hz:.1f} Hz"
        )

    def tick(self):
        # 1) Get current TF: odom -> base_link
        try:
            tf_odom_base = self.tf_buffer.lookup_transform(
                self.parent_frame,
                self.base_frame,
                rclpy.time.Time()  # latest
            )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed ({self.parent_frame}->{self.base_frame}): {e}")
            return

        # 2) Convert odom->base_link to 4x4
        T_odom_base = tf_to_matrix(tf_odom_base)

        # 3) Compute odom->left_cam = (odom->base) * (base->left)
        T_odom_left = T_odom_base @ self.T_base_left

        # 4) Compute odom->right_cam (intermediate), then left->right
        T_odom_right = T_odom_base @ self.T_base_right
        T_left_right = np.linalg.inv(T_odom_left) @ T_odom_right

        # 5) Broadcast:
        #    - parent_frame (odom) -> left_cam
        #    - left_cam -> right_cam
        now_msg = self.get_clock().now().to_msg()

        # odom -> left_cam
        msg_left = TransformStamped()
        msg_left.header.stamp = now_msg
        msg_left.header.frame_id = self.parent_frame
        msg_left.child_frame_id = self.left_frame

        msg_left.transform.translation.x = float(T_odom_left[0, 3])
        msg_left.transform.translation.y = float(T_odom_left[1, 3])
        msg_left.transform.translation.z = float(T_odom_left[2, 3])

        qx, qy, qz, qw = rotmat_to_quat(T_odom_left[0:3, 0:3])
        msg_left.transform.rotation.x = qx
        msg_left.transform.rotation.y = qy
        msg_left.transform.rotation.z = qz
        msg_left.transform.rotation.w = qw

        # left_cam -> right_cam
        msg_right = TransformStamped()
        msg_right.header.stamp = now_msg
        msg_right.header.frame_id = self.left_frame
        msg_right.child_frame_id = self.right_frame

        msg_right.transform.translation.x = float(T_left_right[0, 3])
        msg_right.transform.translation.y = float(T_left_right[1, 3])
        msg_right.transform.translation.z = float(T_left_right[2, 3])

        qx, qy, qz, qw = rotmat_to_quat(T_left_right[0:3, 0:3])
        msg_right.transform.rotation.x = qx
        msg_right.transform.rotation.y = qy
        msg_right.transform.rotation.z = qz
        msg_right.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform([msg_left, msg_right])


def main(args=None):
    rclpy.init(args=args)
    node = DynamicTFCamPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
