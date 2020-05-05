"""
Simulator for the Moab plate+ball balancing device.
Author: Mike Estee
Copyright 2020 Microsoft
"""

# pyright: strict

import math
import random
import numpy as np
from typing import Dict, Tuple
import logging as log

# NOTE: pyrr can create Integer vec/quats if you're not carefull!
# If you start seeing vectors zeroed out, make sure you're using float!
from pyrr import Quaternion, Vector3, matrix44, quaternion, vector, ray
from pyrr.geometric_tests import point_height_above_plane, ray_intersect_plane
from pyrr.plane import create_from_position

# Some type aliases for clarity
Plane = np.ndarray
Ray = np.ndarray

DEFAULT_SIMULATION_RATE = 0.020  # s, 20ms
DEFAULT_GRAVITY = 9.81  # m/s^2, Earth: there's no place like it.

DEFAULT_BALL_RADIUS = 0.02  # m, Ping-Pong ball: 20mm
DEFAULT_BALL_SHELL = 0.0002  # m, Ping-Pong ball: 0.2mm
DEFAULT_BALL_MASS = 0.0027  # kg, Ping-Pong ball: 2.7g
DEFAULT_BALL_COR = 0.89  # unitless, Ping-Pong Coefficient Of Restitution: 0.89
DEFAULT_FRICTION = 0.5  # unitless, Ping-Pong on Acrylic

DEFAULT_PLATE_RADIUS = 0.225 / 2.0  # m, Moab: 225mm dia
PLATE_ORIGIN_TO_SURFACE_OFFSET = 0.009  # 9mm offset from plate rot origin to plate surface

# plate limits
PLATE_HEIGHT_MAX = 0.040  # m, Moab: 40mm
DEFAULT_PLATE_HEIGHT = PLATE_HEIGHT_MAX / 2.0
DEFAULT_PLATE_ANGLE_LIMIT = math.radians(44.0 * 0.5)  # rad, 1/2 full range
DEFAULT_HEIGHT_Z_LIMIT = PLATE_HEIGHT_MAX / 2.0  # m, +/- limit from center Z pos

# default ball Z position
DEFAULT_BALL_Z_POSITION = DEFAULT_PLATE_HEIGHT + PLATE_ORIGIN_TO_SURFACE_OFFSET + DEFAULT_BALL_RADIUS

PLATE_MAX_Z_VELOCITY = 1.0  # m/s
PLATE_Z_ACCEL = 10.0  # m/s^2

# Moab measured velocity at 15deg in 3/60ths, or 300deg/s
DEFAULT_PLATE_MAX_ANGULAR_VELOCITY = (60.0 / 3.0) * math.radians(15)  # rad/s

# Set acceleration to get the plate up to velocity in 1/100th of a sec
DEFAULT_PLATE_ANGULAR_ACCEL = (100.0 / 1.0) * DEFAULT_PLATE_MAX_ANGULAR_VELOCITY  # rad/s^2

# useful constants
X_AXIS = np.array([1.0, 0.0, 0.0])
Y_AXIS = np.array([0.0, 1.0, 0.0])
Z_AXIS = np.array([0.0, 0.0, 1.0])

# Sensor Actuator Noises
DEFAULT_PLATE_NOISE = 0.0  # radians
DEFAULT_BALL_NOISE = 0.0005  # add a little noise on perceived ball location in meters
DEFAULT_JITTER = 0.000  # No simulation jitter (s)


def clamp(val: float, min_val: float, max_val: float):
    return min(max_val, max(min_val, val))


# returns a noise value in the range [-scalar .. scalar] with a gaussian distribution
def random_noise(scalar: float) -> float:
    return scalar * clamp(random.gauss(mu=0, sigma=0.333), -1, 1)  # mean zero gauss with sigma = ~sqrt(scalar)/3


class MoabModel:
    def __init__(self):
        self.pitch = 0.0  # type: float
        self.roll = 0.0  # type: float
        self.height_z = 0.0  # type: float
        self.time_delta = 0.0  # type: float
        self.jitter = 0.0  # type: float
        self.step_time = 0.0  # type: float
        self.elapsed_time = 0.0  # type: float
        self.gravity = 0.0  # type: float
        self.plate_radius = 0.0  # type: float
        self.friction = 0.0  # type: float
        self.tilt_max_vel = 0.0  # type: float
        self.tilt_acc = 0.0  # type: float
        self.ball_noise = 0.0  # type: float
        self.plate_noise = 0.0  # type: float
        self.ball_mass = 0.0  # type: float
        self.ball_radius = 0.0  # type: float
        self.ball_shell = 0.0  # type: float
        self.ball_COR = 0.0  # type: float
        self.plate_pos = Vector3()  # type: Vector3

        self.plate_theta_x = 0.0  # type: float
        self.plate_theta_y = 0.0  # type: float

        self.plate_theta_vel_x = 0.0  # type: float
        self.plate_theta_vel_y = 0.0  # type: float
        self.plate_pos_vel_z = 0.0  # type: float
        self.ball = Vector3()  # type: Vector3
        self.ball_vel = Vector3()  # type: Vector3
        self.ball_acc = Vector3()  # type: Vector3
        self.ball_qat = Quaternion()  # type: Quaternion
        self.estimated_x = 0.0  # type: float
        self.estimated_y = 0.0  # type: float
        self.estimated_radius = 0.0  # type: float
        self.estimated_vel_x = 0.0  # type: float
        self.estimated_vel_y = 0.0  # type: float
        self.prev_estimated_x = 0.0  # type: float
        self.prev_estimated_y = 0.0  # type: float
        self.iteration_count = 0  # type: int
        self.experimental_physics = 0  # type: int

        self.reset()

    # set up all model parameters with sensable defaults
    def reset(self):
        # control inputs are Z-up, X-right, Y-forward
        # control input (unitless) [-1..1]
        self.pitch = 0.0
        self.roll = 0.0
        self.height_z = 0.0

        # servo positions inputs, unitless [0..1]
        # servos are PWM phase controlled.
        # self.motors = [0, 0, 0]

        # model coordinate system is also Z-up
        # units are Metric
        # constants
        self.time_delta = DEFAULT_SIMULATION_RATE
        self.jitter = DEFAULT_JITTER
        self.step_time = self.time_delta
        self.elapsed_time = 0.0

        self.gravity = DEFAULT_GRAVITY
        self.plate_radius = DEFAULT_PLATE_RADIUS
        self.friction = DEFAULT_FRICTION
        self.tilt_max_vel = DEFAULT_PLATE_MAX_ANGULAR_VELOCITY
        self.tilt_acc = DEFAULT_PLATE_ANGULAR_ACCEL
        self.tilt_limit = DEFAULT_PLATE_ANGLE_LIMIT
        self.height_z_limit = DEFAULT_HEIGHT_Z_LIMIT
        self.ball_noise = DEFAULT_BALL_NOISE
        self.plate_noise = DEFAULT_PLATE_NOISE
        self.ball_mass = DEFAULT_BALL_MASS
        self.ball_radius = DEFAULT_BALL_RADIUS
        self.ball_shell = DEFAULT_BALL_SHELL
        self.ball_COR = DEFAULT_BALL_COR

        # Coordinate System:
        # - world coordinate space origin is the plate surface origin
        #   at it's lowest resting position.
        # - plate origin is the Z-up top surface coordinate at
        #   the center of the plate.
        # - ball position is world origin relative

        self.target_pos_x = 0
        self.target_pos_y = 0

        # plate position (m)
        # Y == plate height mid-point
        self.plate_pos = Vector3([0.0, 0.0, DEFAULT_PLATE_HEIGHT])

        # setting these sets the plate normal (rad)
        self.plate_theta_x = 0.0
        self.plate_theta_y = 0.0

        # control velocities for plate (rad/s)
        self.plate_theta_vel_x = 0
        self.plate_theta_vel_y = 0
        self.plate_pos_vel_z = 0  # (m/s)

        # Linear position of ball (m)
        self.ball = Vector3([
            0.0,
            0.0,
            # this should be equivalent to DEFAULT_BALL_Z_POSITION
            self.plate_pos.z + PLATE_ORIGIN_TO_SURFACE_OFFSET + DEFAULT_BALL_RADIUS
        ])

        # Linear velocity of ball (m/sec)
        self.ball_vel = Vector3([0.0, 0.0, 0.0])

        # Linear accel of ball (m/sec^2)
        self.ball_acc = Vector3([0.0, 0.0, 0.0])

        # Orientation of ball (unitless)
        self.ball_qat = Quaternion([0.0, 0.0, 0.0, 1.0])  # identity

        # Modelled camera observations
        self.estimated_x = 0.0
        self.estimated_y = 0.0
        self.estimated_radius = self.ball_radius

        self.estimated_vel_x = 0.0
        self.estimated_vel_y = 0.0

        # These are used to calculate estimated_vel_x/y from previous position
        self.prev_estimated_x = 0.0
        self.prev_estimated_y = 0.0

        # Meta variables
        self.iteration_count = 0  # int [0..N]
        self.experimental_physics = 0

    def halted(self) -> bool:
        # ball is still on the plate?
        return self.get_ball_distance_to_center() > self.plate_radius

    def step(self):
        # update the step time for this step
        self.step_time = self.time_delta + random_noise(self.jitter)
        self.elapsed_time += self.step_time

        # update plate metrics from inputs
        self.update_plate()

        # update ball metrics
        self.update_ball()

        # update meta
        self.iteration_count += 1

    # single axis acceleration of a pos towards a destination
    # with a hard stop at the destination
    def _accel_param(self,
                     q: float, dest: float, vel: float, acc: float, max_vel: float):

        # direction of accel
        dir = 0.0
        if q < dest:
            dir = 1.0
        if q > dest:
            dir = -1.0

        # calculate the change in velocity and position
        acc = acc * dir * self.step_time
        vel_end = clamp(vel + acc * self.step_time, -max_vel, max_vel)
        vel_avg = (vel + vel_end) * 0.5
        delta = vel_avg * self.step_time
        vel = vel_end

        # moving towards the dest?
        if (dir > 0 and q < dest and q + delta < dest) or \
                (dir < 0 and q > dest and q + delta > dest):
            q = q + delta

        # stop at dest
        else:
            q = dest
            vel = 0

        return (q, vel)

    # convert X/Y theta components into a Z-Up RH plane normal
    def _nor_from_xy_theta(self, x_theta: float, y_theta: float) -> np.ndarray:
        x_rot = matrix44.create_from_axis_rotation(axis=X_AXIS, theta=x_theta)
        y_rot = matrix44.create_from_axis_rotation(axis=Y_AXIS, theta=y_theta)

        # pitch then roll
        nor = matrix44.apply_to_vector(mat=x_rot, vec=Z_AXIS)
        nor = matrix44.apply_to_vector(mat=y_rot, vec=nor)

        nor = vector.normalize(nor)
        return nor

    def _plate_nor(self) -> Vector3:
        return Vector3(self._nor_from_xy_theta(self.plate_theta_x, self.plate_theta_y))

    def update_plate(self, plate_reset: bool = False):
        # Find the target xth,yth & zpos
        # convert xy[-1..1] to zx[-self.tilt_limit .. self.tilt_limit]
        # convert z[-1..1] to [PLATE_HEIGHT_MAX/2 - self.height_z_limit .. PLATE_HEIGHT_MAX/2 + self.height_z_limit]
        theta_x_target = self.tilt_limit * self.pitch  # pitch around X axis
        theta_y_target = self.tilt_limit * self.roll  # roll around Y axis
        z_target = (self.height_z * self.height_z_limit) + PLATE_HEIGHT_MAX / 2.0

        # quantize target positions to whole degree increments
        # the Moab hardware can only command by whole degrees
        theta_y_target = math.radians(round(math.degrees(theta_y_target)))
        theta_x_target = math.radians(round(math.degrees(theta_x_target)))

        # get the current xth,yth & zpos
        theta_x, theta_y = self.plate_theta_x, self.plate_theta_y
        z_pos = self.plate_pos.z

        # on reset, bypass the motion equations
        if plate_reset:
            theta_x = theta_x_target
            theta_y = theta_y_target
            z_pos = z_target

        # smooth transition to target based on accel and velocity limits
        else:
            theta_x, self.plate_theta_vel_x = \
                self._accel_param(theta_x, theta_x_target, self.plate_theta_vel_x, self.tilt_acc, self.tilt_max_vel)
            theta_y, self.plate_theta_vel_y = \
                self._accel_param(theta_y, theta_y_target, self.plate_theta_vel_y, self.tilt_acc, self.tilt_max_vel)
            z_pos, self.plate_pos_vel_z = \
                self._accel_param(z_pos, z_target, self.plate_pos_vel_z, PLATE_Z_ACCEL, PLATE_MAX_Z_VELOCITY)

            # add noise to the plate positions
            theta_x += random_noise(self.plate_noise)
            theta_y += random_noise(self.plate_noise)

        # clamp to range limits
        theta_x = clamp(theta_x, -self.tilt_limit, self.tilt_limit)
        theta_y = clamp(theta_y, -self.tilt_limit, self.tilt_limit)
        z_pos = clamp(z_pos,
                      PLATE_HEIGHT_MAX / 2.0 - self.height_z_limit,
                      PLATE_HEIGHT_MAX / 2.0 + self.height_z_limit)

        # Now convert back to plane parameters
        self.plate_theta_x = theta_x
        self.plate_theta_y = theta_y
        self.plate_pos.z = z_pos

    # ball intertia with radius and hollow radius
    # I = 2/5 * m * ((r^5 - h^5) / (r^3 - h^3))
    def _ball_inertia(self):
        hollow_radius = self.ball_radius - self.ball_shell
        return 2.0 / 5.0 * self.ball_mass * (
                (math.pow(self.ball_radius, 5.0) - math.pow(hollow_radius, 5.0)) /
                (math.pow(self.ball_radius, 3.0) - math.pow(hollow_radius, 3.0)))

    def _camera_pos(self) -> Vector3:
        """ camera origin (lens center) in world space """
        return Vector3([0., 0., -0.052])

    def _estimated_ball(self, ball: Vector3) -> Tuple[float, float, float]:
        """
        ray trace the ball position and an edge of the ball back to the camera
        origin and use the collision points with the tilted plate to estimate
        what a camera might perceive the ball position and size to be.
        """
        # contact ray from camera to plate
        camera = self._camera_pos()
        displacement = camera - self.ball
        displacement_radius = camera - (self.ball + Vector3([self.ball_radius, 0, 0]))

        ball_ray = ray.create(camera, displacement)
        ball_radius_ray = ray.create(camera, displacement_radius)

        surface_plane = self._surface_plane()

        contact = Vector3(ray_intersect_plane(ball_ray, surface_plane, False))
        radius_contact = Vector3(ray_intersect_plane(ball_radius_ray, surface_plane, False))

        x, y = contact.x, contact.y
        r = math.fabs(contact.x - radius_contact.x)

        # add the noise in
        estimated_x = x + random_noise(self.ball_noise)
        estimated_y = y + random_noise(self.ball_noise)
        estimated_radius = r + random_noise(self.ball_noise)

        return estimated_x, estimated_y, estimated_radius

    def _estimated_speed(self) -> float:
        return vector.length([self.ball_vel.x, self.ball_vel.y, self.ball_vel.z])

    def _estimated_direction(self) -> float:
        # get the vector to the target
        dx = self.target_pos_x - self.estimated_x
        dy = self.target_pos_y - self.estimated_y

        # vectors and lengths
        u = vector.normalize([dx, dy, 0.0])
        v = vector.normalize([self.estimated_vel_x, self.estimated_vel_y, 0.0])
        ul = vector.length(u)
        vl = vector.length(v)

        # no velocity? already on the target?
        if ul == 0.0 or vl == 0.0:
            return 0.0
        else:
            # angle between vectors
            uv_dot = vector.dot(u, v)

            # signed angle
            x = u[0]
            y = u[1]
            angle = math.atan2(vector.dot([-y, x, 0.0], v), uv_dot)
            if math.isnan(angle):
                return 0.0
            else:
                return angle

    def _surface_plane(self) -> Plane:
        """
        Return the surface plane of the plate
        """
        plate_surface = np.array([
            self.plate_pos.x,
            self.plate_pos.y,
            self.plate_pos.z + PLATE_ORIGIN_TO_SURFACE_OFFSET
        ])
        return create_from_position(plate_surface, self._plate_nor())

    def _ball_rest_plane(self) -> Plane:
        """
        Return the plane which represents all the valid positions
        for the ball origin at rest.
        """
        plate_surface = np.array([
            self.plate_pos.x,
            self.plate_pos.y,
            self.plate_pos.z + PLATE_ORIGIN_TO_SURFACE_OFFSET + self.ball_radius
        ])
        return create_from_position(plate_surface, self._plate_nor())

    def _motion_for_time(self, u: Vector3, a: Vector3, t: float) -> Tuple[Vector3, Vector3]:
        """
        Equations of motion for displacement and final velocity
        u: initial velocity
        a: acceleration
        d: displacement
        v: final velocity

        d = ut + 1/2at^2
        v = u + at

        returns (d, v)
        """
        d = (u * t) + (0.5 * a * (t ** 2))
        v = u + a * t
        return d, v

    def _time_for_motion(self, u: Vector3, a: Vector3, d: Vector3) -> float:
        """
        Compute the time delta given:
        u: initial velocity
        a: acceleration
        d: displacement

        t = iff a>0 and 2*d*a+u**2>=0; (-u + sqrt(2*d*a + u**2))/a
        t = iff a<0 and 2*d*a+u**2>=0; -(u + sqrt(2*d*a + u**2))/a
        t = iff a==0; 0
        """

        # here's the t = (v - u) / a form:
        # if a.x != 0.0:
        #     t =  (v.x - u.x) / a.x
        # elif a.y != 0.0:
        #     t = (v.y - u.y) / a.y
        # elif a.z != 0.0:
        #     t = (v.z - u.z) / a.z

        t = 0.0
        k = 2 * d.x * a.x + (u.x ** 2)
        if a.x > 0.0 and k > 0.0:
            t = (-u.x + math.sqrt(k)) / a.x
        elif a.x < 0.0 and k > 0.0:
            t = -(u.x + math.sqrt(k)) / a.x

        k = 2 * d.y * a.y + (u.y ** 2)
        if a.y > 0.0 and k > 0.0:
            t = (-u.y + math.sqrt(k)) / a.y
        elif a.y < 0.0 and k > 0.0:
            t = -(u.y + math.sqrt(k)) / a.y

        k = 2 * d.z * a.z + (u.z ** 2)
        if a.z > 0.0 and k > 0.0:
            t = (-u.z + math.sqrt(k)) / a.z
        elif a.z < 0.0 and k > 0.0:
            t = -(u.z + math.sqrt(k)) / a.z

        return t

    def _ball_airbone(self, step_t: float) -> float:
        """
        Calculate the final position of the ball while airborne.

        if a collision happens:
            - updates the ball position and velocity at point of contact
            - returns the remaining time in the simulation
        """

        # calculate the acceleration
        self.ball_acc = Vector3([0.0, 0.0, -self.gravity])
        self.ball_rot_acc = Vector3([0.0, 0.0, 0.0])  # no change in a vacuum

        # calculate the displacement, new velocity, and remaining time for this simulation step
        dispplacement, vel = self._motion_for_time(self.ball_vel, self.ball_acc, step_t)
        remaining_t = 0.0

        # get the collision plane, which is the contact plane + offset by the ball radius
        ball_rest = self._ball_rest_plane()
        collision = point_height_above_plane(self.ball + dispplacement, ball_rest) < 0
        if collision:
            # a ray intersecting a plane
            motion_ray = ray.create(self.ball, dispplacement)
            ball_rest = self._ball_rest_plane()
            contact = Vector3(ray_intersect_plane(motion_ray, ball_rest, False))

            # invert equations of motion, solve for time at contact pt
            contact_disp = contact - self.ball
            contact_t = self._time_for_motion(self.ball_vel, self.ball_acc, contact_disp)
            remaining_t = step_t - contact_t

            # re-calc motion at time
            dispplacement, vel = self._motion_for_time(self.ball_vel, self.ball_acc, contact_t)

        # update ball position and velocities
        self.ball.x += dispplacement.x
        self.ball.y += dispplacement.y
        self.ball.z += dispplacement.z
        self.ball_vel = vel

        # return the remaining time in the equation, if any
        # this implicitly means that the ball has collided with the plate
        # and we will transition to the collision state
        return remaining_t

    def _ball_plate_collision(self, remaining_t: float) -> float:
        # Using the velocity as a incident ray, reflect it off the
        # ball plate for the new velocity at the contact point.
        # Where:
        #  R: reflected vector
        #  I: incident vector
        #  N: normal of plate
        #  R = I - 2N(I dot N)
        vel = self.ball_vel
        nor = self._plate_nor()
        vel_refl = vel - 2.0 * nor * (vel | nor)

        # Apply KE loss using Coefficient Of Restitution.
        #  e: Coefficient Of Restitution
        #  KE: kinetic energy
        #  m: mass
        #  u: pre-collision velocity
        #  v: post-collision velocity
        #  KE = 0.5 * m * v^2
        #  e = sqrt(0.5 * m * v^2 / 0.5 * m * u^2)
        #  v = u*e
        vel_len = vector.length(vel_refl) * self.ball_COR
        vel_final = Vector3(vector.set_length(vel_refl, vel_len))

        # set the reflected velocity
        self.ball_vel = vel_final
        return remaining_t

    def _update_ball_z(self):
        self.ball.z = (
                self.ball.x * math.sin(-self.plate_theta_y) +
                self.ball.y * math.sin(self.plate_theta_x) +
                self.ball_radius +
                self.plate_pos.z +
                PLATE_ORIGIN_TO_SURFACE_OFFSET
        )

    def _ball_plate_contact(self, step_t: float) -> float:
        # NOTE: the x_theta axis creates motion in the Y-axis, and vice versa
        # x_theta, y_theta = self._xy_theta_from_nor(self.plate_nor.xyz)
        x_theta = self.plate_theta_x
        y_theta = self.plate_theta_y

        # Equations for acceleration on a plate at rest
        # accel = (mass * g * theta) / (mass + inertia / radius^2)
        # (y_theta,x are intentional swapped here.)
        theta = Vector3([y_theta, -x_theta, 0])
        self.ball_acc = (
                theta / (self.ball_mass + self._ball_inertia() / (self.ball_radius ** 2)) *
                self.ball_mass * self.gravity
        )

        # get contact displacement
        disp, vel = self._motion_for_time(self.ball_vel, self.ball_acc, step_t)

        # simplified ball mechanics against a plane
        self.ball.x += disp.x
        self.ball.y += disp.y
        self._update_ball_z()
        self.ball_vel = vel

        # For rotation on plate motion we use infinite friction and
        # perfect ball / plate coupling.
        # Calculate the distance we traveled across the plate during
        # this time slice.
        rot_distance = math.hypot(disp.x, disp.y)
        if rot_distance > 0:
            # Calculate the fraction of the circumference that we traveled
            # (in radians).
            rot_angle = rot_distance / self.ball_radius

            # Create a quaternion that represents the delta rotation for this time period.
            # Note that we translate the (x, y) direction into (y, -x) because we're
            # creating a vector that represents the axis of rotation which is normal
            # to the direction the ball traveled in the x/y plane.
            rot_q = quaternion.normalize(
                np.array([
                    disp.y / rot_distance * math.sin(rot_angle / 2.0),
                    -disp.x / rot_distance * math.sin(rot_angle / 2.0),
                    0.0,
                    math.cos(rot_angle / 2.0)
                ])
            )

            old_rot = self.ball_qat.xyzw
            new_rot = quaternion.cross(quat1=old_rot, quat2=rot_q)
            self.ball_qat.xyzw = quaternion.normalize(new_rot)
        return 0.0

    def _airborne(self) -> bool:
        """
        returns: True when the ball is not in contact with the plate
        """
        contact_plane = self._ball_rest_plane()
        dist_to_contact = point_height_above_plane(self.ball, contact_plane)

        if dist_to_contact <= 0.0:
            return False
        else:
            return True

    def set_initial_ball(self, x: float, y: float, z: float):
        self.ball.xyz = [x, y, z]
        self._update_ball_z()

        # Set initial observations
        self.estimated_x, self.estimated_y, self.estimated_radius = self._estimated_ball(self.ball)

        self.prev_estimated_x = self.estimated_x
        self.prev_estimated_y = self.estimated_y
        pass

    def update_ball(self):
        """
        Update the ball position with the physics model. We have three phases:
        - airborn: ball is not contacting the plate
        - collision: the instant the ball contacts the plate
        - contact: the ball is contacting the plate
        """
        remaining_t = self.step_time

        # experimental bounce physics
        if self.experimental_physics != 0:
            if self._airborne():
                log.debug("airborne")
                remaining_t = self._ball_airbone(remaining_t)
                if remaining_t > 0.0:
                    log.debug("collision")
                    remaining_t = self._ball_plate_collision(remaining_t)
                    remaining_t = self._ball_airbone(remaining_t)

                    # ESTEE: swallow small bounces that would re-collide with the plate...
            else:
                log.debug("contact")
                remaining_t = self._ball_plate_contact(self.step_time)

        # contact only...
        else:
            remaining_t = self._ball_plate_contact(self.step_time)

        # Finally, lets make some approximations for observations
        self.estimated_x, self.estimated_y, self.estimated_radius = self._estimated_ball(self.ball)

        # Use n-1 states to calculate an estimated velocity.
        self.estimated_vel_x = (self.estimated_x - self.prev_estimated_x) / self.step_time
        self.estimated_vel_y = (self.estimated_y - self.prev_estimated_y) / self.step_time

        # update for next time
        self.prev_estimated_x = self.estimated_x
        self.prev_estimated_y = self.estimated_y
        pass

    def get_ball_distance_to_center(self) -> float:
        # ball.z relative to plate
        zpos = self.ball.z - (self.plate_pos.z + self.ball_radius +
                              PLATE_ORIGIN_TO_SURFACE_OFFSET)

        # ball distance from ball position on plate at origin
        return math.sqrt(
            math.pow(self.ball.x, 2.0) +
            math.pow(self.ball.y, 2.0) +
            math.pow(zpos, 2.0))

    def get_ball_velocity(self) -> float:
        return vector.length(self.ball_vel.xyz)

    def state(self) -> Dict[str, float]:
        # x_theta, y_theta = self._xy_theta_from_nor(self.plate_nor)
        plate_nor = self._plate_nor()

        return dict(

            # reflected input controls
            tilt_x=self.roll,
            tilt_y=self.pitch,

            roll=self.roll,
            pitch=self.pitch,

            height_z=self.height_z,

            # reflected constants
            time_delta=self.time_delta,
            jitter=self.jitter,
            step_time=self.step_time,
            elapsed_time=self.elapsed_time,

            gravity=self.gravity,
            plate_radius=self.plate_radius,
            friction=self.friction,
            tilt_max_vel=self.tilt_max_vel,
            tilt_acc=self.tilt_acc,
            tilt_limit=self.tilt_limit,
            height_z_limit=self.height_z_limit,

            ball_mass=self.ball_mass,
            ball_radius=self.ball_radius,
            ball_shell=self.ball_shell,
            ball_COR=self.ball_COR,

            target_pos_x=self.target_pos_x,
            target_pos_y=self.target_pos_y,

            # modelled plate metrics
            plate_pos_x=self.plate_pos.x,
            plate_pos_y=self.plate_pos.y,
            plate_pos_z=self.plate_pos.z,

            plate_nor_x=plate_nor.x,
            plate_nor_y=plate_nor.y,
            plate_nor_z=plate_nor.z,

            plate_theta_x=self.plate_theta_x,
            plate_theta_y=self.plate_theta_y,

            plate_theta_vel_x=self.plate_theta_vel_x,
            plate_theta_vel_y=self.plate_theta_vel_y,
            plate_pos_vel_z=self.plate_pos_vel_z,

            # modelled ball metrics
            ball_x=self.ball.x,
            ball_y=self.ball.y,
            ball_z=self.ball.z,

            ball_vel_x=self.ball_vel.x,
            ball_vel_y=self.ball_vel.y,
            ball_vel_z=self.ball_vel.z,

            ball_acc_x=self.ball_acc.x,
            ball_acc_y=self.ball_acc.y,
            ball_acc_z=self.ball_acc.z,

            ball_qat_x=self.ball_qat.x,
            ball_qat_y=self.ball_qat.y,
            ball_qat_z=self.ball_qat.z,
            ball_qat_w=self.ball_qat.w,

            # modelled camera observations
            estimated_x=self.estimated_x,
            estimated_y=self.estimated_y,
            estimated_radius=self.estimated_radius,

            estimated_vel_x=self.estimated_vel_x,
            estimated_vel_y=self.estimated_vel_y,

            estimated_speed=self._estimated_speed(),
            estimated_direction=self._estimated_direction(),

            ball_noise=self.ball_noise,
            plate_noise=self.plate_noise,

            # meta vars
            ball_fell_off=1 if self.halted() else 0,
            iteration_count=self.iteration_count,
            experimental_physics=self.experimental_physics
        )
