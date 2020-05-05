import sys
from typing import Optional, Tuple

import gym
import numpy as np
from gym.spaces import Box
from pyrr import vector, matrix33

from moab_env.moab_model import MoabModel, clamp


_CLOSE_ENOUGH = 0.01
_DEFAULT_BALL_NOISE = 0.0005
_DEFAULT_FRICTION = 0.6
_DEFAULT_GRAVITY = 9.81
_DEFAULT_HEIGHT_LIMIT = 0.0
_DEFAULT_PLATE_NOISE = (np.pi / 180.0) * 1
_DEFAULT_TILT_LIMIT = (np.pi / 180.0) * 15
_DEFAULT_TIME_DELTA = 0.02
_MAX_ITER_COUNT = 250
_MAX_FRICTION = 1.0
_MAX_GRAVITY = _DEFAULT_GRAVITY * 2
_MAX_TILT_VELOCITY = (np.pi / 180.0) * 300
_MAX_VELOCITY = 1.0
_MIN_FRICTION = 0.2
_MIN_GRAVITY = _DEFAULT_GRAVITY / 2
_PING_PONG_COR = 0.89
_PING_PONG_MASS = 0.0027
_PING_PONG_RADIUS = 0.020
_PING_PONG_SHELL = 0.0002
_PLATE_RADIUS = 0.225 / 2.0
_PLATE_Z_OFFSET = 0.009
_TILT_ACC = (60.0 / 3.0) * _MAX_TILT_VELOCITY


class MoabSim(gym.Env):

    """
    Arguments:
    config: needs to incorporate the following parameters
        seed: The random seed for the random number generator.
        use_dr: The flag to turn on the domain randomization.
        use_normalize_action: The flag to enable the action normalization.
                              Normally, we use it for PPO. For SAC and other
                              algorithms, RLLib may take care of it.
        use_normalize_state: The flag to enable the range state normalization.


    Gym-like Moab Simulator - Example:

    >> env = MoabSim(config=dict(seed=None, use_normalize_action=False))
    >> env.reset()
    >> episode_reward = 0
    >> episode_count = 0
    >> for _ in range(1000):
    >>     env.render()
    >>     _, reward, done, _ = env.step(env.action_space.sample())
    >>     episode_reward += reward
    >>     if done:
    >>         episode_count += 1
    >>         env.reset()
    >>         print(f"Episode {episode_count} reward: {episode_reward}")
    >>         episode_reward = 0
    >> env.close()
    """

    def __init__(
            self,
            config: dict
    ):
        self.model = MoabModel()
        self.model.reset()
        self._random_state = np.random.RandomState(config.get("seed"))
        self._use_dr = config.get("use_dr", False)
        self._use_normalize_action = config.get("use_normalize_action", False)
        self._use_normalize_state = config.get("use_normalize_state", True)
        print(self._use_dr, self._use_normalize_action, self._use_normalize_state)

        obs_high = np.array(
            [sys.maxsize, sys.maxsize, sys.maxsize, sys.maxsize]
        )
        obs_low = np.array(
            [- sys.maxsize - 1, - sys.maxsize - 1,
             - sys.maxsize - 1, - sys.maxsize - 1]
        )
        action_high = np.array([1.0, 1.0])

        self.action_space = Box(
            low=-action_high,
            high=action_high,
            dtype=np.float32
        )
        self.observation_space = Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )

    def seed(self, seed: Optional[int] = None) -> None:
        """
        Create the random number generator with seed.
        """
        self._random_state = np.random.RandomState(seed)

    def step(
            self,
            action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Perform the one simulation step for Moab simulator.
        """
        if self._use_normalize_action:
            self.model.roll = clamp(
                self._normalize_action(action[0]), -1.0, 1.0)
            self.model.pitch = clamp(
                self._normalize_action(action[1]), -1.0, 1.0)
        else:
            self.model.roll = clamp(action[0], -1.0, 1.0)
            self.model.pitch = clamp(action[1], -1.0, 1.0)
        self.model.step()
        return (
            self._get_obs_from_state(),
            self._get_reward(),
            self._is_penalty_terminal(),
            {}
        )

    def reset(self) -> np.ndarray:
        """
        Reset the Moab simulator.
        """
        return self._do_reset()

    def render(
            self,
            mode: str = 'human'
    ) -> None:
        # TODO: Create the visual simulator.
        pass

    @staticmethod
    def _normalize_action(action: float) -> float:
        """
        Normalize the action following the PDP2 output preprocessor.
        """
        # NOTE: This follows the action normalizer in PDP2 output
        # preprocessor. We suggest using it only when RLLib doesn't have
        # the action normalizer for the algorithm.
        return 0.5 * np.clip(action, -2.0, 2.0)

    def _do_reset(self) -> np.ndarray:
        """
        Reset the simulator with all initial configs depending on the
        domain randomization.
        """
        self.model.reset()
        if self._use_dr:
            self.model.roll = self._random_state.uniform(low=-0.2, high=0.2)
            self.model.pitch = self._random_state.uniform(low=-0.2, high=0.2)
        else:
            self.model.roll = -0.1
            self.model.pitch = -0.1
        self.model.height_z = 0.0
        self.model.time_delta = _DEFAULT_TIME_DELTA
        self.model.gravity = _DEFAULT_GRAVITY
        self.model.friction = _DEFAULT_FRICTION
        self.model.plate_radius = _PLATE_RADIUS
        self.model.tilt_max_vel = _MAX_TILT_VELOCITY
        self.model.tilt_acc = _TILT_ACC
        self.model.tilt_limit = _DEFAULT_TILT_LIMIT
        self.model.height_z_limit = _DEFAULT_HEIGHT_LIMIT
        self.model.ball_mass = _PING_PONG_MASS
        self.model.ball_radius = _PING_PONG_RADIUS
        self.model.ball_shell = _PING_PONG_SHELL
        self.model.ball_COR = _PING_PONG_COR
        self.model.target_pos_x = 0.0
        self.model.target_pos_y = 0.0
        self.model.ball_noise = _DEFAULT_BALL_NOISE
        self.model.plate_noise = _DEFAULT_PLATE_NOISE
        self.model.update_plate(plate_reset=True)
        if self._use_dr:
            self.model.set_initial_ball(
                self._random_state.uniform(
                    low=-_PLATE_RADIUS * 0.75,
                    high=_PLATE_RADIUS * 0.75
                ),
                self._random_state.uniform(
                    low=-_PLATE_RADIUS * 0.75,
                    high=_PLATE_RADIUS * 0.75
                ),
                0.0
            )
            self.model.ball_vel.x = self._random_state.uniform(
                low=-0.02, high=0.02)
            self.model.ball_vel.y = self._random_state.uniform(
                low=-0.02, high=0.02)
        else:
            self.model.set_initial_ball(
                -_PLATE_RADIUS * 0.5,
                -_PLATE_RADIUS * 0.5,
                0.0
            )
            self.model.ball_vel.x = 0.0
            self.model.ball_vel.y = 0.0
        self.model.ball_vel.z = 0.0
        # self._set_velocity_for_speed_and_direction(0.0, 0.0)
        self.model.iteration_count = 0
        return self._get_obs_from_state()

    def _get_obs_from_state(self) -> np.ndarray:
        """
        Get the observation data from the simulator state.
        """
        full_state = self.model.state()
        # NOTE: The observation can be:
        # estimated_x, estimated_y,
        # estimated_velocity_x, estimated_velocity_y
        if self._use_normalize_state:
            c_ball_x = float(np.clip(
                full_state["ball_x"],
                -1 * _PLATE_RADIUS,
                _PLATE_RADIUS
            ))
            c_ball_y = float(np.clip(
                full_state["ball_y"],
                -1 * _PLATE_RADIUS,
                _PLATE_RADIUS
            ))
            c_ball_vel_x = float(np.clip(
                full_state["ball_vel_x"],
                -1 * _MAX_VELOCITY,
                _MAX_VELOCITY
            ))
            c_ball_vel_y = float(np.clip(
                full_state["ball_vel_y"],
                -1 * _MAX_VELOCITY,
                _MAX_VELOCITY
            ))
            return np.array(
                [c_ball_x / (2 * _PLATE_RADIUS),
                 c_ball_y / (2 * _PLATE_RADIUS),
                 c_ball_vel_x / (2 * _MAX_VELOCITY),
                 c_ball_vel_y / (2 * _MAX_VELOCITY)
                 ]
            )
        else:
            return np.array(
                [full_state["ball_x"], full_state["ball_y"],
                 full_state["ball_vel_x"], full_state["ball_vel_y"]
                 ]
            )

    def _get_reward(self) -> float:
        """
        Get the reward value on a simulation step.
        """
        # NOTE: The reward definition follows our inkling file
        # moab.ink in brain/src/sdk2/samples/moabsim/moab.ink
        full_state = self.model.state()
        if self._is_penalty_terminal():
            return -10
        dx = full_state["ball_x"] - full_state["target_pos_x"]
        dy = full_state["ball_y"] - full_state["target_pos_y"]
        ball_origin_at_center = full_state["ball_radius"] + full_state[
            "plate_pos_z"] + _PLATE_Z_OFFSET
        dz = full_state["ball_z"] - ball_origin_at_center

        distance_to_target = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
        if distance_to_target < _CLOSE_ENOUGH:
            return 10
        return 10 * _CLOSE_ENOUGH / distance_to_target

    def _is_penalty_terminal(self) -> bool:
        """
        Check if it is the valid terminal for getting penalty.
        """
        full_state = self.model.state()
        return full_state["ball_fell_off"] > 0 or full_state[
            "iteration_count"] >= _MAX_ITER_COUNT

    def _set_velocity_for_speed_and_direction(
            self,
            speed: float,
            direction: float
    ) -> None:
        """
        Set velocity for the ball speed and direction.
        """
        # get the heading
        dx = self.model.target_pos_x - self.model.ball.x
        dy = self.model.target_pos_y - self.model.ball.y

        # direction is meaningless if we're already at the target
        if (dx != 0) or (dy != 0):

            # set the magnitude
            vel = vector.set_length([dx, dy, 0.0], speed)

            # rotate by direction around Z-axis at ball position
            rot = matrix33.create_from_axis_rotation(
                [0.0, 0.0, 1.0],
                direction
            )
            vel = matrix33.apply_to_vector(rot, vel)

            # unpack into ball velocity
            self.model.ball_vel.x = vel[0]
            self.model.ball_vel.y = vel[1]
            self.model.ball_vel.z = vel[2]
