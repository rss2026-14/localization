import numpy as np
import rclpy

from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose, Quaternion
from nav_msgs.msg import OccupancyGrid
from localization.sensor_model import SensorModel


def yaw_to_quaternion(yaw):
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = np.sin(yaw / 2.0)
    q.w = np.cos(yaw / 2.0)
    return q


def make_pose(x, y, theta):
    p = Pose()
    p.position.x = float(x)
    p.position.y = float(y)
    p.position.z = 0.0
    q = yaw_to_quaternion(theta)
    p.orientation = q
    return p


class SensorModelVisualizer(Node):
    def __init__(self):
        super().__init__("sensor_model_visualizer")

        self.before_pub = self.create_publisher(PoseArray, "/sensor_test/before", 1)
        self.top_pub = self.create_publisher(PoseArray, "/sensor_test/top", 1)
        self.after_pub = self.create_publisher(PoseArray, "/sensor_test/after", 1)
        self.map_pub = self.create_publisher(OccupancyGrid, "/map", 1)
        self.sensor_model = SensorModel(self)

        self.setup_map()
        self.timer = self.create_timer(1.0, self.run_test_once)
        self.did_run = False

    def setup_map(self):
        width = 200
        height = 200
        resolution = 0.05

        grid = OccupancyGrid()
        grid.info.width = width
        grid.info.height = height
        grid.info.resolution = resolution
        grid.info.origin.position.x = 0.0
        grid.info.origin.position.y = 0.0
        grid.info.origin.position.z = 0.0
        grid.info.origin.orientation = yaw_to_quaternion(0.0)
        self.map_pub.publish(grid)

        data = np.zeros((height, width), dtype=np.int8)
        data[0, :] = 100
        data[-1, :] = 100
        data[:, 0] = 100
        data[:, -1] = 100
        data[60:140, 100] = 100

        grid.data = data.flatten().tolist()
        self.sensor_model.map_callback(grid)

    def publish_pose_array(self, pub, particles):
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.poses = [make_pose(x, y, t) for x, y, t in particles]
        pub.publish(msg)

    def systematic_resample(self, particles, weights):
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / np.sum(weights)
        n = len(weights)
        positions = (np.arange(n) + np.random.uniform()) / n
        indexes = np.zeros(n, dtype=int)
        cumulative_sum = np.cumsum(weights)

        i, j = 0, 0
        while i < n:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return particles[indexes]

    def run_test_once(self):
        if self.did_run or not self.sensor_model.map_set:
            return
        self.did_run = True

        true_pose = np.array([2.0, 2.0, 0.0])
        obs = self.sensor_model.scan_sim.scan(true_pose.reshape(1, 3))[0]

        particles = np.column_stack([
            np.random.uniform(0.5, 8.0, size=500),
            np.random.uniform(0.5, 8.0, size=500),
            np.random.uniform(-np.pi, np.pi, size=500),
        ])

        weights = self.sensor_model.evaluate(particles, obs)

        # top 50 particles by weight
        top_idx = np.argsort(weights)[-50:]
        top_particles = particles[top_idx]

        # resampled particles
        resampled = self.systematic_resample(particles, weights)

        self.publish_pose_array(self.before_pub, particles)
        self.publish_pose_array(self.top_pub, top_particles)
        self.publish_pose_array(self.after_pub, resampled)

        self.get_logger().info("Published sensor model visualization topics.")
        
def main(args=None):
    rclpy.init(args=args)
    node = SensorModelVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
