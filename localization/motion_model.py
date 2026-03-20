import numpy as np


class MotionModel:

    def __init__(self, node):
        node.declare_parameter("deterministic", False)
        self.deterministic = node.get_parameter(
            "deterministic"
        ).get_parameter_value().bool_value

        self.alpha_trans = 0.05
        self.alpha_rot = 0.02
        self.alpha_slip = 0.01

    def evaluate(self, particles, odometry):
        particles = np.asarray(particles, dtype=np.float64)
        odometry = np.asarray(odometry, dtype=np.float64)

        if particles.ndim != 2 or particles.shape[1] != 3:
            raise ValueError("particles must be an Nx3 matrix")
        if odometry.shape != (3,):
            raise ValueError("odometry must be a 3-vector [dx, dy, dtheta]")

        dx, dy, dtheta = odometry
        n = particles.shape[0]

        dx_hat = np.full(n, dx)
        dy_hat = np.full(n, dy)
        dtheta_hat = np.full(n, dtheta)

        if not self.deterministic:
            delta_trans = np.hypot(dx, dy)

            trans_std = np.sqrt(self.alpha_trans * delta_trans**2)
            rot_std = np.sqrt(self.alpha_rot * dtheta**2 + self.alpha_rot * delta_trans**2)
            slip_std = np.sqrt(self.alpha_slip * delta_trans**2)

            # forward noise along x
            dx_hat += np.random.normal(0.0, trans_std, size=n)
            # lateral noise along y
            dy_hat += np.random.normal(0.0, slip_std, size=n)
            # heading noise
            dtheta_hat += np.random.normal(0.0, rot_std, size=n)

        theta = particles[:, 2]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        updated_particles = particles.copy()
        updated_particles[:, 0] += dx_hat * cos_theta - dy_hat * sin_theta
        updated_particles[:, 1] += dx_hat * sin_theta + dy_hat * cos_theta
        updated_particles[:, 2] += dtheta_hat

        updated_particles[:, 2] = (updated_particles[:, 2] + np.pi) % (2.0 * np.pi) - np.pi

        return updated_particles
