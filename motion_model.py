import numpy as np


class MotionModel:

    def __init__(self, node):
        ####################################
        self.node = node

        #tune these
        self.alpha_trans = 0.05 # translation noise from translation
        self.alpha_rot = 0.02 # rotation noise from translation/rotation
        self.alpha_slip = 0.01 # small lateral slip noise
        ####################################

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y1 theta1]
                [    ...      ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size
        """

        ####################################
        # check shapes
        particles = np.asarray(particles)
        odometry = np.asarray(odometry)
        if particles.ndim != 2 or particles.shape[1] != 3:
            raise ValueError("particles must be an Nx3 matrix")
        if odometry.shape != (3,):
            raise ValueError("odometry must be a 3-vector [dx, dy, dtheta]")

        dx, dy, dtheta = odometry
        n = particles.shape[0]

        #translational displacement
        delta_trans = np.hypot(dx, dy)

        #motion-dependent noise
        trans_std = np.sqrt(self.alpha_trans * delta_trans**2)
        rot_std = np.sqrt(
            self.alpha_rot * dtheta**2 +
            self.alpha_rot * delta_trans**2
        )
        slip_std = np.sqrt(self.alpha_slip * delta_trans**2)

        #sample noise independently for each particle
        trans_noise = np.random.normal(0.0, trans_std, size=n)
        rot_noise = np.random.normal(0.0, rot_std, size=n)
        slip_noise = np.random.normal(0.0, slip_std, size=n)

        delta_trans_hat = delta_trans + trans_noise
        dtheta_hat = dtheta + rot_noise

        #updating
        theta = particles[:, 2]
        updated_particles = particles.copy()
        updated_particles[:, 0] += (
            delta_trans_hat * np.cos(theta) -
            slip_noise * np.sin(theta)
        )
        updated_particles[:, 1] += (
            delta_trans_hat * np.sin(theta) +
            slip_noise * np.cos(theta)
        )
        updated_particles[:, 2] += dtheta_hat

        return updated_particles
        ####################################
