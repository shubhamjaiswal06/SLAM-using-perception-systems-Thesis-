# The particle filter, prediciton and correction.
#
# slam_08_b_particle_correction.
# Claus Brenner, 04.01.2013
from lego_robot import *
from slam_e_library import get_cylinders_from_scan, assign_cylinders
from math import sin, cos, pi, atan2, sqrt
import random
from scipy.stats import norm as normal_dist


class ParticleFilter:
    def __init__(self, initial_particles,
                 robot_width, scanner_displacement,
                 control_motion_factor, control_turn_factor,
                 measurement_distance_stddev, measurement_angle_stddev):
        # The particles.
        self.particles = initial_particles

        # Some constants.
        self.robot_width = robot_width
        self.scanner_displacement = scanner_displacement
        self.control_motion_factor = control_motion_factor
        self.control_turn_factor = control_turn_factor
        self.measurement_distance_stddev = measurement_distance_stddev
        self.measurement_angle_stddev = measurement_angle_stddev

    # State transition. This is exactly the same method as in the Kalman filter.
    @staticmethod
    def g(state, control, w):
        x, y, theta = state
        l, r = control
        if r != l:
            alpha = (r - l) / w
            rad = l/alpha
            g1 = x + (rad + w/2.)*(sin(theta+alpha) - sin(theta))
            g2 = y + (rad + w/2.)*(-cos(theta+alpha) + cos(theta))
            g3 = (theta + alpha + pi) % (2*pi) - pi
        else:
            g1 = x + l * cos(theta)
            g2 = y + l * sin(theta)
            g3 = theta

        return (g1, g2, g3)

    def predict(self, control):
        """The prediction step of the particle filter."""

        left, right = control

        left_var = (self.control_motion_factor * left)**2 +\
                   (self.control_turn_factor * (left-right))**2
        right_var = (self.control_motion_factor * right)**2 +\
                    (self.control_turn_factor * (left-right))**2
        
        #left = random.gauss(left, sqrt(left_var))
        #right = random.gauss(right, sqrt(right_var))
        #ctrl = (left, right)
        particle=[]
        for i in self.particles:
            left_ = random.gauss(left, sqrt(left_var))
            right_ = random.gauss(right, sqrt(right_var))
            ctrl = (left_, right_)
            particle.append(self.g(i, ctrl, self.robot_width))

        self.particles = particle 

    # Measurement. This is exactly the same method as in the Kalman filter.
    @staticmethod
    def h(state, landmark, scanner_displacement):
        """Takes a (x, y, theta) state and a (x, y) landmark, and returns the
           corresponding (range, bearing)."""
        #print(state)
        #print(landmark)
        
        dx = landmark[0] - (state[0] + scanner_displacement * cos(state[2]))
        #print(dx)
        dy = landmark[1] - (state[1] + scanner_displacement * sin(state[2]))
        r = sqrt(dx * dx + dy * dy)
        alpha = (atan2(dy, dx) - state[2] + pi) % (2*pi) - pi
        return (r, alpha)

    def probability_of_measurement(self, measurement, predicted_measurement):
        """Given a measurement and a predicted measurement, computes
           probability."""
        # Compute differences to real measurements.

        # --->>> Compute difference in distance and bearing angle.
        # Important: make sure the angle difference works correctly and does
        # not return values offset by 2 pi or - 2 pi.
        # You may use the following Gaussian PDF function:
        # scipy.stats.norm.pdf(x, mu, sigma). With the import in the header,
        # this is normal_dist.pdf(x, mu, sigma).
        # Note that the two parameters sigma_d and sigma_alpha discussed
        # in the lecture are self.measurement_distance_stddev and
        # self.measurement_angle_stddev.

        r, alpha = measurement
        r_, alpha_ = predicted_measurement

        if alpha < 0:
            alpha = 2*pi - alpha
        if alpha_ < 0:
            alpha_ = 2*pi - alpha_

        R_pdf = normal_dist.pdf((r - r_), 0, self.measurement_distance_stddev**2)
        Alpha_pdf = normal_dist.pdf((alpha - alpha_), 0, self.measurement_angle_stddev**2)

        return R_pdf*Alpha_pdf 

    def compute_weights(self, cylinders, landmarks):
        """Computes one weight for each particle, returns list of weights."""
        weights = []
        for p in self.particles:
            # Get list of tuples:
            # [ ((range_0, bearing_0), (landmark_x, landmark_y)), ... ]
            assignment = assign_cylinders(cylinders, p,
                self.scanner_displacement, landmarks)

            # --->>> Insert code to compute weight for particle p here.
            # This will require a loop over all (measurement, landmark((from h))
            # in assignment. Append weight to the list of weights.
            #pdf = 0
            pdf = 1
            for m,l in assignment:
                #print(m)
                #print(l)
                predicted_measurement = ParticleFilter.h(p, l, self.scanner_displacement)
                #pdf = pdf + ParticleFilter.probability_of_measurement(self, m, predicted_measurement)
                pdf = pdf*ParticleFilter.probability_of_measurement(self, m, predicted_measurement)
            weights.append(pdf)  
        #print(len(weights))
        #print(weights)
        return weights

    def resample(self, weights):
        """Return a list of particles which have been resampled, proportional
           to the given weights."""


        # You may implement the 'resampling wheel' algorithm
        # described in the lecture.
        new_particles = []
        last = sum(weights)
        #chose random number between 0 and sum of all weights
        for k in self.particles:

            temp = random.uniform(0, last)

            #find index of that chosen number
            j = 0
            idx = 0
            for i in weights:
                j = j + i
                if temp>j:
                    idx = idx + 1
                else:
                    break
            #chose that index from predicted particle list and add to new particle list
            new_particles.append(self.particles[idx])
            #print(temp)
            #print(idx)
            #print(j)


        #new_particles = self.particles  # Replace this.
        #print(len(new_particles))
        #print(new_particles)
        return new_particles

    def correct(self, cylinders, landmarks):
        """The correction step of the particle filter."""
        # First compute all weights.
        weights = self.compute_weights(cylinders, landmarks)
        # Then resample, based on the weight array.
        self.particles = self.resample(weights)

    def print_particles(self, file_desc):
        """Prints particles to given file_desc output."""
        if not self.particles:
            return
        #print >> file_desc, "PA",
        file_desc.write("PA ")
        for p in self.particles:
            #print >> file_desc, "%.0f %.0f %.3f" % p,
            file_desc.write("%.0f %.0f %.3f " % p)
        #print >> file_desc
        file_desc.write("\n")
        print(file_desc)


if __name__ == '__main__':
    # Robot constants.
    scanner_displacement = 30.0
    ticks_to_mm = 0.349
    robot_width = 155.0

    # Cylinder extraction and matching constants.
    minimum_valid_distance = 20.0
    depth_jump = 100.0
    cylinder_offset = 90.0

    # Filter constants.
    control_motion_factor = 0.35  # Error in motor control.
    control_turn_factor = 0.6  # Additional error due to slip when turning.
    measurement_distance_stddev = 200.0  # Distance measurement error of cylinders.
    measurement_angle_stddev = 15.0 / 180.0 * pi  # Angle measurement error.

    # Generate initial particles. Each particle is (x, y, theta).
    number_of_particles = 100
    measured_state = (1850.0, 1897.0, 213.0 / 180.0 * pi)
    #measured_state = (1150.0, 1097.0, 213.0 / 180.0 * pi)
    standard_deviations = (100.0, 100.0, 10.0 / 180.0 * pi)
    initial_particles = []
    for i in range(number_of_particles):
        initial_particles.append(tuple([
            random.gauss(measured_state[j], standard_deviations[j])
            for j in range(3)]))

    # Setup filter.
    pf = ParticleFilter(initial_particles,
                        robot_width, scanner_displacement,
                        control_motion_factor, control_turn_factor,
                        measurement_distance_stddev,
                        measurement_angle_stddev)

    # Read data.
    logfile = LegoLogfile()
    logfile.read("robot4_motors.txt")
    logfile.read("robot4_scan.txt")
    logfile.read("robot_arena_landmarks.txt")
    reference_cylinders = [l[1:3] for l in logfile.landmarks]

    # Loop over all motor tick records.
    # This is the particle filter loop, with prediction and correction.
    f = open("particle_filter_corrected.txt", "w")
    for i in range(len(logfile.motor_ticks)):
        # Prediction.
        control = map(lambda x: x * ticks_to_mm, logfile.motor_ticks[i])
        pf.predict(control)

        # Correction.
        cylinders = get_cylinders_from_scan(logfile.scan_data[i], depth_jump,
            minimum_valid_distance, cylinder_offset)
        pf.correct(cylinders, reference_cylinders)

        # Output particles.
        pf.print_particles(f)

    f.close()
