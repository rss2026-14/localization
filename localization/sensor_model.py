import numpy as np
from scan_simulator_2d import PyScanSimulator2D
# Try to change to just `from scan_simulator_2d import PyScanSimulator2D`
# if any error re: scan_simulator_2d occurs

from scipy.spatial.transform import Rotation as R

from nav_msgs.msg import OccupancyGrid

import sys

np.set_printoptions(threshold=sys.maxsize)


class SensorModel():

    def __init__(self, node):
        node.declare_parameter('map_topic', "default")
        node.declare_parameter('num_beams_per_particle', 1)
        node.declare_parameter('scan_theta_discretization', 1.0)
        node.declare_parameter('scan_field_of_view', 1.0)
        node.declare_parameter('lidar_scale_to_map_scale', 1.0)

        self.map_topic = node.get_parameter('map_topic').get_parameter_value().string_value
        self.num_beams_per_particle = node.get_parameter('num_beams_per_particle').get_parameter_value().integer_value
        self.scan_theta_discretization = node.get_parameter(
            'scan_theta_discretization').get_parameter_value().double_value
        self.scan_field_of_view = node.get_parameter('scan_field_of_view').get_parameter_value().double_value
        self.lidar_scale_to_map_scale = node.get_parameter(
            'lidar_scale_to_map_scale').get_parameter_value().double_value

        ####################################
        # Adjust these parameters
        self.alpha_hit = 0.82
        self.alpha_short = 0.03
        self.alpha_max = 0.05
        self.alpha_rand = 0.1
        self.sigma_hit = 3.0
        # self.sigma_hit = 12.0
        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width = 201
        ####################################

        self.node  = node

        node.get_logger().info("%s" % self.map_topic)
        node.get_logger().info("%s" % self.num_beams_per_particle)
        node.get_logger().info("%s" % self.scan_theta_discretization)
        node.get_logger().info("%s" % self.scan_field_of_view)

        # Precompute the sensor model table
        self.sensor_model_table = np.empty((self.table_width, self.table_width))
        self.precompute_sensor_model()

        # Create a simulated laser scan
        self.scan_sim = PyScanSimulator2D(
            self.num_beams_per_particle,
            self.scan_field_of_view,
            0,  # This is not the simulator, don't add noise
            0.01,  # This is used as an epsilon
            self.scan_theta_discretization)

        # Subscribe to the map
        self.map = None
        self.map_set = False
        self.map_subscriber = node.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            1)

    def precompute_sensor_model(self):
        """
        Generate and store a table which represents the sensor model.

        For each discrete computed range value, this provides the probability of
        measuring any (discrete) range. This table is indexed by the sensor model
        at runtime by discretizing the measurements and computed ranges from
        RangeLibc.
        This table must be implemented as a numpy 2D array.

        Compute the table based on class parameters alpha_hit, alpha_short,
        alpha_max, alpha_rand, sigma_hit, and table_width.

        args:
            N/A

        returns:
            No return type. Directly modify `self.sensor_model_table`.
        """
        z_max = self.table_width - 1
        W = self.table_width

        # z = measured range, d = expected range
        z = np.arange(W, dtype=np.float64)[:, None]   # shape (W, 1)
        d = np.arange(W, dtype=np.float64)[None, :]   # shape (1, W)

        # Broadcast to full (W, W) grids
        z_grid = np.broadcast_to(z, (W, W))
        d_grid = np.broadcast_to(d, (W, W))

        # p_hit
        p_hit = np.exp(-((z_grid - d_grid) ** 2) / (2.0 * self.sigma_hit**2))
        p_hit = p_hit / np.sum(p_hit, axis=0, keepdims=True)
        # p_short
        p_short = np.zeros((W, W), dtype=np.float64)
        valid_short = (z_grid <= d_grid) & (d_grid > 0)
        p_short[valid_short] = (2.0 / d_grid[valid_short]) * (
            1.0 - z_grid[valid_short] / d_grid[valid_short]
        )
        # p_max
        p_max = np.zeros((W, W), dtype=np.float64)
        p_max[z_grid == z_max] = 1.0
        # if you want to mimic your old epsilon-style spike instead:
        # p_max[z_grid >= (z_max - self.epsilon)] = 1.0 / self.epsilon

        # p_rand
        p_rand = np.zeros((W, W), dtype=np.float64)
        p_rand[z_grid < z_max] = 1.0 / z_max

        # Weighted mixture
        table = (
            self.alpha_hit * p_hit +
            self.alpha_short * p_short +
            self.alpha_max * p_max +
            self.alpha_rand * p_rand
        )
        table = table / np.sum(table, axis=0, keepdims=True)
        self.sensor_model_table = table


    def evaluate(self, particles, observation):
        """
        Evaluate how likely each particle is given
        the observed scan.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            observation: A vector of lidar data measured
                from the actual lidar. THIS IS Z_K. Each range in Z_K is Z_K^i

        returns:
           probabilities: A vector of length N representing
               the probability of each particle existing
               given the observation and the map.
        """

        if not self.map_set:
            return

        ####################################
        # TODO
        # Evaluate the sensor model here!
        #
        # You will probably want to use this function
        # to perform ray tracing from all the particles.
        # This produces a matrix of size N x num_beams_per_particle

        scans = self.scan_sim.scan(particles)

        #downsample observation to the same number of beams
        obs = np.asarray(observation, dtype=np.float64)
        if len(obs) != self.num_beams_per_particle:
            beam_indices = np.linspace(
                0, len(obs) - 1, self.num_beams_per_particle
            ).astype(int)
            obs = obs[beam_indices]

        #convert real ranges to the lookup-table scale table indices must be integers in [0, table_width-1]
        # Convert to pixels by DIVIDING by (resolution * scale)
        obs_idx = np.clip(
            np.rint(obs / (self.resolution * self.lidar_scale_to_map_scale)).astype(int),
            0,
            self.table_width - 1
        )

        scan_idx = np.clip(
            np.rint(scans / (self.resolution * self.lidar_scale_to_map_scale)).astype(int),
            0,
            self.table_width - 1
        )

        num_particles = particles.shape[0]
        probabilities = np.ones(num_particles, dtype=np.float64)

        #multiply beam likelihoods for each particle
        # for i in range(self.num_beams_per_particle):
        #     z_meas = obs_idx[i]
        #     z_exp = scan_idx[:, i]
        #     beam_probs = self.sensor_model_table[z_meas, z_exp]
        #     probabilities *= beam_probs

        SQUASH_FACTOR = 1.0 / 3.0

        for i in range(self.num_beams_per_particle):
            z_meas = obs_idx[i]
            z_exp = scan_idx[:, i]
            beam_probs = self.sensor_model_table[z_meas, z_exp]
            probabilities *= np.power(beam_probs, SQUASH_FACTOR)

        self.node.get_logger().info(f"{probabilities[1:3]}")

        return probabilities


    def map_callback(self, map_msg):
        # Convert the map to a numpy array
        self.map = np.array(map_msg.data, np.double) / 100.
        self.map = np.clip(self.map, 0, 1)

        self.resolution = map_msg.info.resolution

        # Convert the origin to a tuple
        origin_p = map_msg.info.origin.position
        origin_o = map_msg.info.origin.orientation
        quat = [origin_o.x, origin_o.y, origin_o.z, origin_o.w]
        yaw = R.from_quat(quat).as_euler("xyz")[2]

        origin = (origin_p.x, origin_p.y, yaw)

        # Initialize a map with the laser scan
        self.scan_sim.set_map(
            self.map,
            map_msg.info.height,
            map_msg.info.width,
            map_msg.info.resolution,
            origin,
            0.5)  # Consider anything < 0.5 to be free

        # Make the map set
        self.map_set = True

        print("Map initialized")
