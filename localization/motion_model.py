
import numpy as np


class MotionModel:

    def __init__(self, node):
        ####################################
        # TODO
        # Do any precomputation for the motion model here.

        node.declare_parameter("deterministic", False)
        self.deterministic = node.get_parameter("deterministic").get_parameter_value().bool_value

        self.sigma_x = 0.05
        self.sigma_y = 0.05
        self.sigma_theta = 0.1

        ####################################

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size
        """

        ####################################
        dx, dy, dtheta = odometry

        thetas = particles[:, 2]

        cos_thetas = np.cos(thetas)
        sin_thetas = np.sin(thetas)

        dx_global = dx * cos_thetas - dy * sin_thetas
        dy_global = dx * sin_thetas + dy * cos_thetas

        particles[:, 0] += dx_global
        particles[:, 1] += dy_global
        particles[:, 2] += dtheta

        # Noise
        if not self.deterministic:
            n = particles.shape[0]
            particles[:, 0] += np.random.normal(0, self.sigma_x, n)
            particles[:, 1] += np.random.normal(0, self.sigma_y, n)
            particles[:, 2] += np.random.normal(0, self.sigma_theta, n)

        particles[:, 2] = (particles[:, 2] + np.pi) % (2 * np.pi) - np.pi

        return particles

        ####################################

# import numpy as np
# from rcl_interfaces.msg import SetParametersResult

# class MotionModel:

#     def __init__(self, node):
#         self.node = node
#         self.get_logger = node.get_logger

#         # Parameters
#         node.declare_parameter("alpha1", 0.05)
#         node.declare_parameter("alpha2", 0.05)
#         node.declare_parameter("alpha3", 0.10)
#         node.declare_parameter("alpha4", 0.05)
#         node.declare_parameter("deterministic", False)

#         self.alpha1 = node.get_parameter("alpha1").get_parameter_value().double_value
#         self.alpha2 = node.get_parameter("alpha2").get_parameter_value().double_value
#         self.alpha3 = node.get_parameter("alpha3").get_parameter_value().double_value
#         self.alpha4 = node.get_parameter("alpha4").get_parameter_value().double_value
#         self.deterministic = node.get_parameter("deterministic").get_parameter_value().bool_value

#         # 3. Register the callback
#         node.add_on_set_parameters_callback(self.parameters_callback)

#         self.get_logger().info("Motion Model Initialized with dynamic parameters")

#     def parameters_callback(self, params):
#         """
#         Callback to update alpha parameters and deterministic flag at runtime.
#         """
#         for param in params:
#             if param.name == "alpha1":
#                 self.alpha1 = param.value
#                 self.get_logger().info(f"Updated alpha1 to {self.alpha1}")
#             elif param.name == "alpha2":
#                 self.alpha2 = param.value
#                 self.get_logger().info(f"Updated alpha2 to {self.alpha2}")
#             elif param.name == "alpha3":
#                 self.alpha3 = param.value
#                 self.get_logger().info(f"Updated alpha3 to {self.alpha3}")
#             elif param.name == "alpha4":
#                 self.alpha4 = param.value
#                 self.get_logger().info(f"Updated alpha4 to {self.alpha4}")
#             elif param.name == "deterministic":
#                 self.deterministic = param.value
#                 self.get_logger().info(f"Deterministic mode set to {self.deterministic}")

#         return SetParametersResult(successful=True)

#     def evaluate(self, particles, odometry):
#         """
#         Update particles based on RTR (Rotate-Translate-Rotate) model.
#         """
#         (x0, y0, t0), (x1, y1, t1) = odometry

#         # Calculate relative motion
#         d_rot1 = np.arctan2(y1 - y0, x1 - x0) - t0
#         d_trans = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
#         d_rot2 = t1 - t0 - d_rot1

#         # Angle normalization
#         d_rot1 = (d_rot1 + np.pi) % (2 * np.pi) - np.pi
#         d_rot2 = (d_rot2 + np.pi) % (2 * np.pi) - np.pi

#         n = particles.shape[0]

#         if self.deterministic:
#             dh_rot1, dh_trans, dh_rot2 = d_rot1, d_trans, d_rot2
#         else:
#             # Noise scales based on current alpha parameters
#             std_rot1 = np.sqrt(self.alpha1 * d_rot1**2 + self.alpha2 * d_trans**2)
#             std_trans = np.sqrt(self.alpha3 * d_trans**2 + self.alpha4 * (d_rot1**2 + d_rot2**2))
#             std_rot2 = np.sqrt(self.alpha1 * d_rot2**2 + self.alpha2 * d_trans**2)

#             dh_rot1 = d_rot1 - np.random.normal(0, std_rot1, n)
#             dh_trans = d_trans - np.random.normal(0, std_trans, n)
#             dh_rot2 = d_rot2 - np.random.normal(0, std_rot2, n)

#         # Apply to particles
#         thetas = particles[:, 2]
#         particles[:, 0] += dh_trans * np.cos(thetas + dh_rot1)
#         particles[:, 1] += dh_trans * np.sin(thetas + dh_rot1)
#         particles[:, 2] += dh_rot1 + dh_rot2

#         # Final wrap-around
#         particles[:, 2] = (particles[:, 2] + np.pi) % (2 * np.pi) - np.pi

#         return particles
