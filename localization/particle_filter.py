import numpy as np

from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import (
    PoseWithCovarianceStamped,
    PoseArray,
    Pose,
    TransformStamped,
)

from tf2_ros import TransformBroadcaster

from rclpy.node import Node
import rclpy

class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter('particle_filter_frame', "/base_link_pf")
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value
        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.

        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('scan_topic', "/scan")
        self.declare_parameter('num_particles', 500)
        # self.declare_parameter('init_x_std', 0.5)
        # self.declare_parameter('init_y_std', 0.5)
        # self.declare_parameter('init_theta_std', 0.3)
        self.declare_parameter('init_x_std', 0.7)
        self.declare_parameter('init_y_std', 0.7)
        self.declare_parameter('init_theta_std', 0.5)

        # self.declare_parameter('resample_position_std', 0.02)
        # self.declare_parameter('resample_theta_std', 0.01)
        self.declare_parameter('resample_position_std', 0.1)
        self.declare_parameter('resample_theta_std', 0.05)


        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        self.num_particles = self.get_parameter("num_particles").get_parameter_value().integer_value
        self.init_x_std = self.get_parameter("init_x_std").get_parameter_value().double_value
        self.init_y_std = self.get_parameter("init_y_std").get_parameter_value().double_value
        self.init_theta_std = self.get_parameter("init_theta_std").get_parameter_value().double_value
        self.resample_position_std = self.get_parameter("resample_position_std").get_parameter_value().double_value
        self.resample_theta_std = self.get_parameter("resample_theta_std").get_parameter_value().double_value

        self.laser_sub = self.create_subscription(LaserScan, scan_topic,
                                                  self.laser_callback,
                                                  1)

        self.odom_sub = self.create_subscription(Odometry, odom_topic,
                                                 self.odom_callback,
                                                 1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                                 self.pose_callback,
                                                 1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)
        self.particles_pub = self.create_publisher(PoseArray, "/pf/particles", 1)
        self.tf_broadcaster = TransformBroadcaster(self)


        # Initialize the models
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        #storing particles
        self.particles = None
        self.initialized = False
        self.last_odom_time = None
        self.particle_weights = None

        # self.total_distance_moved = 0.0
        # self.settle_cycles = 0

        self.get_logger().info("=============+READY+=============")

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.

    def yaw_to_quaternion(self, yaw):
        """
        Convert yaw angle to quaternion [x, y, z, w].
        """
        qx = 0.0
        qy = 0.0
        qz = np.sin(yaw / 2.0)
        qw = np.cos(yaw / 2.0)
        return qx, qy, qz, qw

    def quaternion_to_yaw(self, q):
        """
        Convert quaternion message to yaw.
        """
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
        """
        Initialize particles around a pose guess from RViz.
        """
        self.particles = np.zeros((self.num_particles, 3), dtype=np.float64)
        self.particles[:, 0] = np.random.normal(x, self.init_x_std, self.num_particles)
        self.particles[:, 1] = np.random.normal(y, self.init_y_std, self.num_particles)
        self.particles[:, 2] = np.random.normal(theta, self.init_theta_std, self.num_particles)
        self.particles[:, 2] = self.wrap_angle(self.particles[:, 2])
        self.initialized = True

        # self.settle_cycles = 70

        self.particle_weights = None
        # self.total_distance_moved = 0.0

        self.get_logger().info(
            f"Initialized {self.num_particles} particles at "
            f"x={x:.3f}, y={y:.3f}, theta={theta:.3f}"
        )

        self.publish_particles()
        self.publish_estimate()

    def compute_pose_estimate(self):
        """
        Compute a mean pose from particles.
        Uses circular mean for heading.
        """
        # if self.particles is None or len(self.particles) == 0:
        #     return None

        # x_mean = np.mean(self.particles[:, 0])
        # y_mean = np.mean(self.particles[:, 1])

        # cos_mean = np.mean(np.cos(self.particles[:, 2]))
        # sin_mean = np.mean(np.sin(self.particles[:, 2]))
        # theta_mean = np.arctan2(sin_mean, cos_mean)

        # return np.array([x_mean, y_mean, theta_mean], dtype=np.float64)
        if self.particles is None or len(self.particles) == 0:
            return None

        # prevent wall-merging
        if self.particle_weights is None:
            best_particles = self.particles
        else:
            sorted_indices = np.argsort(self.particle_weights)[::-1]
            top_10_percent_idx = int(self.num_particles * 0.10)
            best_indices = sorted_indices[:max(1, top_10_percent_idx)]
            best_particles = self.particles[best_indices]

        x_mean = np.mean(best_particles[:, 0])
        y_mean = np.mean(best_particles[:, 1])

        cos_mean = np.mean(np.cos(best_particles[:, 2]))
        sin_mean = np.mean(np.sin(best_particles[:, 2]))
        theta_mean = np.arctan2(sin_mean, cos_mean)

        return np.array([x_mean, y_mean, theta_mean], dtype=np.float64)

    def resample_particles(self, weights):
        """
        Resample particles according to weights, then add a little blur.
        """
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
            p=weights
        )

        self.particles = self.particles[indices].copy()

        # Small Gaussian blur after resampling
        self.particles[:, 0] += np.random.normal(0.0, self.resample_position_std, self.num_particles)
        self.particles[:, 1] += np.random.normal(0.0, self.resample_position_std, self.num_particles)
        self.particles[:, 2] += np.random.normal(0.0, self.resample_theta_std, self.num_particles)
        self.particles[:, 2] = self.wrap_angle(self.particles[:, 2])

    def publish_particles(self):
        """
        Publish particle cloud for RViz debugging.
        """
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
        """
        Publish estimated pose as Odometry and TF.
        """
        est = self.compute_pose_estimate()
        if est is None:
            return

        x, y, theta = est
        qx, qy, qz, qw = self.yaw_to_quaternion(theta)
        now = self.get_clock().now().to_msg()

        # Publish Odometry estimate
        odom_msg = Odometry()
        odom_msg.header.stamp = now
        odom_msg.header.frame_id = "map"
        odom_msg.child_frame_id = self.particle_filter_frame

        odom_msg.pose.pose.position.x = float(x)
        odom_msg.pose.pose.position.y = float(y)
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation.x = qx
        odom_msg.pose.pose.orientation.y = qy
        odom_msg.pose.pose.orientation.z = qz
        odom_msg.pose.pose.orientation.w = qw

        self.odom_pub.publish(odom_msg)

        # Publish TF: map -> particle_filter_frame
        tf_msg = TransformStamped()
        tf_msg.header.stamp = now
        tf_msg.header.frame_id = "map"
        tf_msg.child_frame_id = self.particle_filter_frame
        tf_msg.transform.translation.x = float(x)
        tf_msg.transform.translation.y = float(y)
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation.x = qx
        tf_msg.transform.rotation.y = qy
        tf_msg.transform.rotation.z = qz
        tf_msg.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(tf_msg)

    # ---------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------

    def pose_callback(self, msg):
        """
        Initialize particles from RViz /initialpose.
        """
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = self.quaternion_to_yaw(msg.pose.pose.orientation)

        self.initialize_particles(x, y, theta)

    def odom_callback(self, msg):
        """
        Motion update:
        whenever odometry arrives, propagate particles.
        """
        if not self.initialized:
            return

        # 1. Extract current time in seconds
        current_time = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)

        # 2. Handle first message initialization
        if self.last_odom_time is None:
            self.last_odom_time = current_time
            return

        # 3. Calculate dt (time elapsed since last message)
        dt = current_time - self.last_odom_time
        self.last_odom_time = current_time

        # Safety check: if dt is negative or suspiciously large (e.g., sim reset)
        if dt <= 0.0 or dt > 1.0:
            return

        # 4. Extract velocities from the TWIST component
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vtheta = msg.twist.twist.angular.z

        # 5. Calculate local delta (Velocity * Time)
        dx_local = vx * dt
        dy_local = vy * dt
        dtheta = vtheta * dt

        delta_pose = np.array([dx_local, dy_local, dtheta])

        # 6. Apply to motion model if the robot actually moved
        # if np.any(np.abs(delta_pose) > 1e-5):
        self.particles = self.motion_model.evaluate(self.particles, delta_pose)
        self.particles[:, 2] = self.wrap_angle(self.particles[:, 2])

        # self.total_distance_moved += np.hypot(dx_local, dy_local)

        self.publish_particles()
        self.publish_estimate()

    def laser_callback(self, msg):
        """
        Sensor update:
        compute particle weights from scan, then resample.
        """
        if not self.initialized or not self.sensor_model.map_set:
            return

        observation = np.array(msg.ranges, dtype=np.float64)
        invalid = ~np.isfinite(observation)
        observation[invalid] = msg.range_max
        observation = np.clip(observation, msg.range_min, msg.range_max)

        # 1. ALWAYS calculate the new weights based on the current scan
        weights = self.sensor_model.evaluate(self.particles, observation)
        if weights is None:
            return

        # 2. ALWAYS save the weights so the pose estimate updates dynamically
        self.particle_weights = weights

        # # Resample if we moved, OR if we are currently settling
        # if self.total_distance_moved > 0.05 or self.settle_cycles > 0:

        self.resample_particles(weights)

            # # If we are settling, count down. Otherwise, reset the distance tracker.
            # if self.settle_cycles > 0:
            #     self.settle_cycles -= 1
            # else:
            #     self.total_distance_moved = 0.0

        
        # 4. ALWAYS publish the updated estimate
        self.publish_particles()
        self.publish_estimate()


def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
