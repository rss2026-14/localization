import csv
import os
import time

import numpy as np
import rclpy

from geometry_msgs.msg import Pose, PoseArray, PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64
from tf2_ros import TransformBroadcaster

from localization.motion_model import MotionModel
from localization.sensor_model import SensorModel


class ParticleFilter(Node):
    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter("particle_filter_frame", "/base_link_pf")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("num_particles", 100)

        self.declare_parameter("init_x_std", 0.7)
        self.declare_parameter("init_y_std", 0.7)
        self.declare_parameter("init_theta_std", 0.5)

        #self.declare_parameter("resample_position_std", 0.01)
        #self.declare_parameter("resample_theta_std", 0.005)
        self.declare_parameter("resample_position_std", 0.2)
        self.declare_parameter("resample_theta_std", 0.1)

        self.declare_parameter("runtime_csv_path", "")
        self.declare_parameter("debug_runtime_logs", False)

        self.particle_filter_frame = (
            self.get_parameter("particle_filter_frame").get_parameter_value().string_value
        )
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        self.num_particles = self.get_parameter("num_particles").get_parameter_value().integer_value
        self.init_x_std = self.get_parameter("init_x_std").get_parameter_value().double_value
        self.init_y_std = self.get_parameter("init_y_std").get_parameter_value().double_value
        self.init_theta_std = self.get_parameter("init_theta_std").get_parameter_value().double_value
        self.resample_position_std = (
            self.get_parameter("resample_position_std").get_parameter_value().double_value
        )
        self.resample_theta_std = (
            self.get_parameter("resample_theta_std").get_parameter_value().double_value
        )
        self.runtime_csv_path = (
            self.get_parameter("runtime_csv_path").get_parameter_value().string_value
        )
        self.debug_runtime_logs = (
            self.get_parameter("debug_runtime_logs").get_parameter_value().bool_value
        )

        self.laser_sub = self.create_subscription(
            LaserScan, scan_topic, self.laser_callback, 1
        )
        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 1
        )
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, "/initialpose", self.pose_callback, 1
        )

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)
        self.particles_pub = self.create_publisher(PoseArray, "/pf/particles", 1)

        self.motion_runtime_pub = self.create_publisher(Float64, "/pf/motion_runtime_ms", 10)
        self.sensor_runtime_pub = self.create_publisher(Float64, "/pf/sensor_runtime_ms", 10)
        self.total_runtime_pub = self.create_publisher(Float64, "/pf/update_runtime_ms", 10)

        self.tf_broadcaster = TransformBroadcaster(self)

        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        self.particles = None
        self.initialized = False
        self.last_odom_time = None
        self.particle_weights = None
        self.latest_odom_pose = None

        self.motion_runtimes_ms = []
        self.sensor_runtimes_ms = []
        self.total_runtimes_ms = []

        self.runtime_csv_file = None
        self.runtime_csv_writer = None
        if self.runtime_csv_path:
            runtime_dir = os.path.dirname(self.runtime_csv_path)
            if runtime_dir:
                os.makedirs(runtime_dir, exist_ok=True)
            self.runtime_csv_file = open(self.runtime_csv_path, "w", newline="")
            self.runtime_csv_writer = csv.writer(self.runtime_csv_file)
            self.runtime_csv_writer.writerow(["callback_type", "runtime_ms"])

        self.get_logger().info("=============+READY+=============")

    def yaw_to_quaternion(self, yaw):
        qx = 0.0
        qy = 0.0
        qz = np.sin(yaw / 2.0)
        qw = np.cos(yaw / 2.0)
        return qx, qy, qz, qw

    def quaternion_to_yaw(self, q):
        x = q.x
        y = q.y
        z = q.z
        w = q.w
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def wrap_angle(self, theta):
        return (theta + np.pi) % (2.0 * np.pi) - np.pi

    def initialize_particles(self, x, y, theta):
        self.particles = np.zeros((self.num_particles, 3), dtype=np.float64)
        self.particles[:, 0] = np.random.normal(x, self.init_x_std, self.num_particles)
        self.particles[:, 1] = np.random.normal(y, self.init_y_std, self.num_particles)
        self.particles[:, 2] = np.random.normal(theta, self.init_theta_std, self.num_particles)
        self.particles[:, 2] = self.wrap_angle(self.particles[:, 2])

        self.initialized = True
        self.particle_weights = None
        self.last_odom_time = None

        self.get_logger().info(
            f"Initialized {self.num_particles} particles at "
            f"x={x:.3f}, y={y:.3f}, theta={theta:.3f}"
        )

        self.publish_particles()
        self.publish_estimate()

    def compute_pose_estimate(self):
        if self.particles is None or len(self.particles) == 0:
            return None

        if self.particle_weights is None:
            best_particles = self.particles
        else:
            sorted_indices = np.argsort(self.particle_weights)[::-1]
            top_10_percent_idx = int(self.num_particles * 0.10)
            best_indices = sorted_indices[: max(1, top_10_percent_idx)]
            best_particles = self.particles[best_indices]

        x_mean = np.mean(best_particles[:, 0])
        y_mean = np.mean(best_particles[:, 1])

        cos_mean = np.mean(np.cos(best_particles[:, 2]))
        sin_mean = np.mean(np.sin(best_particles[:, 2]))
        theta_mean = np.arctan2(sin_mean, cos_mean)

        return np.array([x_mean, y_mean, theta_mean], dtype=np.float64)

    def resample_particles(self, weights):
        weights = np.asarray(weights, dtype=np.float64)

        if np.any(~np.isfinite(weights)):
            self.get_logger().warn("Non-finite weights encountered, using uniform weights")
            weights = np.ones(self.num_particles, dtype=np.float64)

        total = np.sum(weights)
        if total <= 0.0:
            self.get_logger().warn("All particle weights are zero, using uniform weights")
            weights = np.ones(self.num_particles, dtype=np.float64) / self.num_particles
        else:
            weights = weights / total

        indices = np.random.choice(
            self.num_particles,
            size=self.num_particles,
            replace=True,
            p=weights,
        )

        self.particles = self.particles[indices].copy()

        self.particles[:, 0] += np.random.normal(
            0.0, self.resample_position_std, self.num_particles
        )
        self.particles[:, 1] += np.random.normal(
            0.0, self.resample_position_std, self.num_particles
        )
        self.particles[:, 2] += np.random.normal(
            0.0, self.resample_theta_std, self.num_particles
        )
        self.particles[:, 2] = self.wrap_angle(self.particles[:, 2])

    def publish_particles(self):
        if self.particles is None:
            return

        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        poses = []
        for p in self.particles:
            pose = Pose()
            pose.position.x = float(p[0])
            pose.position.y = float(p[1])
            pose.position.z = 0.0

            qx, qy, qz, qw = self.yaw_to_quaternion(p[2])
            pose.orientation.x = qx
            pose.orientation.y = qy
            pose.orientation.z = qz
            pose.orientation.w = qw
            poses.append(pose)

        msg.poses = poses
        self.particles_pub.publish(msg)

    def publish_estimate(self):
        est = self.compute_pose_estimate()
        if est is None:
            return

        x_map_base, y_map_base, theta_map_base = est
        qx_mb, qy_mb, qz_mb, qw_mb = self.yaw_to_quaternion(theta_map_base)
        now = self.get_clock().now().to_msg()

        odom_msg = Odometry()
        odom_msg.header.stamp = now
        odom_msg.header.frame_id = "map"
        odom_msg.child_frame_id = "base_link"

        odom_msg.pose.pose.position.x = float(x_map_base)
        odom_msg.pose.pose.position.y = float(y_map_base)
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation.x = qx_mb
        odom_msg.pose.pose.orientation.y = qy_mb
        odom_msg.pose.pose.orientation.z = qz_mb
        odom_msg.pose.pose.orientation.w = qw_mb

        self.odom_pub.publish(odom_msg)

        if self.latest_odom_pose is None:
            return

        x_odom_base, y_odom_base, theta_odom_base = self.latest_odom_pose

        theta_map_odom = self.wrap_angle(theta_map_base - theta_odom_base)

        c = np.cos(theta_map_odom)
        s = np.sin(theta_map_odom)

        x_map_odom = x_map_base - (c * x_odom_base - s * y_odom_base)
        y_map_odom = y_map_base - (s * x_odom_base + c * y_odom_base)

        qx_mo, qy_mo, qz_mo, qw_mo = self.yaw_to_quaternion(theta_map_odom)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = now
        tf_msg.header.frame_id = "map"
        tf_msg.child_frame_id = "odom"
        tf_msg.transform.translation.x = float(x_map_odom)
        tf_msg.transform.translation.y = float(y_map_odom)
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation.x = qx_mo
        tf_msg.transform.rotation.y = qy_mo
        tf_msg.transform.rotation.z = qz_mo
        tf_msg.transform.rotation.w = qw_mo

        self.tf_broadcaster.sendTransform(tf_msg)

    def publish_runtime_sample(self, publisher, value_ms, history_list):
        msg = Float64()
        msg.data = float(value_ms)
        publisher.publish(msg)
        history_list.append(float(value_ms))

    def record_runtime(self, callback_type, runtime_ms):
        runtime_ms = float(runtime_ms)

        if callback_type == "motion":
            self.publish_runtime_sample(
                self.motion_runtime_pub, runtime_ms, self.motion_runtimes_ms
            )
        elif callback_type == "sensor":
            self.publish_runtime_sample(
                self.sensor_runtime_pub, runtime_ms, self.sensor_runtimes_ms
            )
        else:
            self.get_logger().warn(f"Unknown runtime callback_type: {callback_type}")
            return

        # Treat update runtime as "runtime of the filter update that just happened".
        # This guarantees /pf/update_runtime_ms is populated for both motion and sensor updates.
        self.publish_runtime_sample(
            self.total_runtime_pub, runtime_ms, self.total_runtimes_ms
        )

        if self.runtime_csv_writer is not None:
            self.runtime_csv_writer.writerow([callback_type, f"{runtime_ms:.6f}"])
            self.runtime_csv_file.flush()

    def print_runtime_summary(self):
        def summarize(name, values):
            if len(values) == 0:
                self.get_logger().info(f"{name}: no samples")
                return

            arr = np.array(values, dtype=np.float64)
            self.get_logger().info(
                f"{name}: mean={np.mean(arr):.3f} ms, "
                f"max={np.max(arr):.3f} ms, "
                f"min={np.min(arr):.3f} ms"
            )

        summarize("Motion update runtime", self.motion_runtimes_ms)
        summarize("Sensor update runtime", self.sensor_runtimes_ms)
        summarize("Filter update runtime", self.total_runtimes_ms)

    def close_runtime_file(self):
        if self.runtime_csv_file is not None:
            self.runtime_csv_file.close()
            self.runtime_csv_file = None
            self.runtime_csv_writer = None

    def pose_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = self.quaternion_to_yaw(msg.pose.pose.orientation)
        self.initialize_particles(x, y, theta)

    def odom_callback(self, msg):
        odom_x = msg.pose.pose.position.x
        odom_y = msg.pose.pose.position.y
        odom_theta = self.quaternion_to_yaw(msg.pose.pose.orientation)
        self.latest_odom_pose = np.array([odom_x, odom_y, odom_theta], dtype=np.float64)

        if not self.initialized:
            return

        current_time = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)

        if self.last_odom_time is None:
            self.last_odom_time = current_time
            return

        dt = current_time - self.last_odom_time
        self.last_odom_time = current_time

        if dt <= 0.0 or dt > 1.0:
            return

        start = time.perf_counter()

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vtheta = msg.twist.twist.angular.z

        dx_local = vx * dt
        dy_local = vy * dt
        dtheta = vtheta * dt

        delta_pose = np.array([dx_local, dy_local, dtheta], dtype=np.float64)

        self.particles = self.motion_model.evaluate(self.particles, delta_pose)
        self.particles[:, 2] = self.wrap_angle(self.particles[:, 2])

        self.publish_particles()
        self.publish_estimate()

        runtime_ms = (time.perf_counter() - start) * 1000.0
        if self.debug_runtime_logs:
            self.get_logger().info(f"motion callback runtime: {runtime_ms:.3f} ms")
        self.record_runtime("motion", runtime_ms)

    def laser_callback(self, msg):
        if not self.initialized:
            if self.debug_runtime_logs:
                self.get_logger().info("laser_callback skipped: filter not initialized")
            return

        if not self.sensor_model.map_set:
            if self.debug_runtime_logs:
                self.get_logger().info("laser_callback skipped: sensor_model.map_set is False")
            return

        observation = np.array(msg.ranges, dtype=np.float64)
        invalid = ~np.isfinite(observation)
        observation[invalid] = msg.range_max
        observation = np.clip(observation, msg.range_min, msg.range_max)

        start = time.perf_counter()

        weights = self.sensor_model.evaluate(self.particles, observation)
        if weights is None:
            if self.debug_runtime_logs:
                self.get_logger().info("laser_callback skipped: sensor weights is None")
            return

        self.particle_weights = weights
        self.resample_particles(weights)

        self.publish_particles()
        self.publish_estimate()

        runtime_ms = (time.perf_counter() - start) * 1000.0
        if self.debug_runtime_logs:
            self.get_logger().info(f"sensor callback runtime: {runtime_ms:.3f} ms")
        self.record_runtime("sensor", runtime_ms)

    def destroy_node(self):
        self.print_runtime_summary()
        self.close_runtime_file()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()

    try:
        rclpy.spin(pf)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            pf.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
