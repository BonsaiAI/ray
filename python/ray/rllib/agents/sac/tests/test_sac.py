from typing import Tuple

import gym
import pytest
import ray
import numpy as np
import tensorflow as tf

from sac import __version__, DEFAULT_CONFIG, SACTrainer
from sac.custom_model import CustomModel, build_model, build_fcn


class GymEnv:
    def __init__(self, config: dict):
        import pybullet_envs  # Needed to register PyBullet envs with gym
        assert pybullet_envs  # Just to ensure that the above import is not removed
        self.config = config
        self.env_id = config["env_id"]
        self._env = gym.make(self.env_id)

    def __getattr__(self, item):
        return getattr(self._env, item)


@pytest.fixture
def ray_env():
    if not ray.is_initialized():
        ray.init()
    yield
    ray.shutdown()


@pytest.fixture
def config(env_id):
    config = DEFAULT_CONFIG.copy()
    config["env_config"] = dict(env_id=env_id)
    if env_id == "Pendulum-v0":
        config["timesteps_per_iteration"] = 100
    config["num_workers"] = 8
    config["num_envs_per_worker"] = 8
    return config


@pytest.fixture
def trainer(config):
    return SACTrainer(config=config, env=GymEnv)


@pytest.mark.parametrize(
    "env_id, threshold", [
        ("Pendulum-v0", -750),
    ]
)
@pytest.mark.usefixtures("ray_env")
def test_convergence(trainer, env_id, threshold):
    mean_episode_reward = -float("inf")
    for i in range(350):
        result = trainer.train()
        mean_episode_reward = result["episode_reward_mean"]
        print(f"{i}: {mean_episode_reward}")
    assert mean_episode_reward >= threshold


class TestCustomModel:
    @pytest.fixture
    def env(self, env_id: str) -> gym.Env:
        return gym.make(env_id)

    @pytest.fixture
    def action_space(self, env: gym.Env) -> gym.spaces.Space:
        return env.action_space

    @pytest.fixture
    def observation_space(self, env: gym.Env) -> gym.spaces.Space:
        return env.observation_space

    @pytest.fixture
    def num_outputs(self, observation_space: gym.Space) -> int:
        return int(np.product(observation_space.shape))

    @pytest.fixture
    def action_dim(self, action_space: gym.Space) -> int:
        return np.product(action_space.shape)

    @pytest.fixture
    def model_config(self, hiddens: Tuple[int, ...], hidden_activation: str) -> dict:
        return dict(fcnet_hiddens=hiddens, fcnet_activation=hidden_activation)

    @pytest.fixture
    def oracle_shift_and_log_scale_diag_model(
        self,
        hiddens: Tuple[int, ...],
        hidden_activation: str,
        action_dim: int,
        num_outputs: int
    ) -> tf.keras.Model:
        result = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(num_outputs,))
            ] + [
                tf.keras.layers.Dense(
                    units=hidden,
                    activation=getattr(tf.nn, hidden_activation),
                    name="action_hidden_{}".format(i),
                )
                for i, hidden in enumerate(hiddens)
            ] + [
                tf.keras.layers.Dense(
                    units=2 * action_dim, activation=None, name="action_out"
                )
            ]
        )
        result.build(input_shape=(num_outputs,))
        return result

    @pytest.fixture
    def shift_and_log_scale_diag_model(
        self,
        hiddens: Tuple[int, ...],
        hidden_activation: str,
        action_dim: int,
        num_outputs: int,
    ) -> tf.keras.Model:
        return build_fcn(
            input_shapes=dict(model_out=(num_outputs,)),
            num_outputs=2*action_dim,
            hidden_layer_sizes=hiddens,
            hidden_activations=hidden_activation,
        )

    @pytest.fixture
    def oracle_q_net(
        self,
        hiddens: Tuple[int, ...],
        hidden_activation: str,
        action_dim: int,
        num_outputs: int,
    ) -> tf.keras.Model:
        observations = tf.keras.layers.Input(shape=(num_outputs,), name="observations")
        actions = tf.keras.layers.Input(shape=(action_dim,), name="actions")
        q_net = tf.keras.Sequential(
            [tf.keras.layers.Concatenate(axis=1), ]
            + [
                tf.keras.layers.Dense(
                    units=units,
                    activation=getattr(tf.nn, hidden_activation),
                    name=f"oracle_hidden_{i}"
                )
                for i, units in enumerate(hiddens)
            ]
            + [
                tf.keras.layers.Dense(
                    units=1, activation=None, name="oracle_out"
                )
            ]
        )

        return tf.keras.Model(
            [observations, actions], q_net([observations, actions])
        )

    @pytest.fixture
    def actual_q_net(
        self,
        hiddens: Tuple[int, ...],
        hidden_activation: str,
        action_dim: int,
        num_outputs: int,
    ) -> tf.keras.Model:
        return build_fcn(
            input_shapes=dict(observations=(num_outputs,), actions=(action_dim,)),
            num_outputs=1,
            hidden_layer_sizes=hiddens,
            hidden_activations=hidden_activation,
        )

    @pytest.mark.parametrize("env_id,", ["Pendulum-v0"])
    @pytest.mark.parametrize(
        "hiddens, hidden_activation", [((256, 256), "relu")]
    )
    def test_shift_and_log_scale_models_equal(
        self,
        oracle_shift_and_log_scale_diag_model: tf.keras.Model,
        shift_and_log_scale_diag_model: tf.keras.Model,
        hiddens: Tuple[int, ...],
        hidden_activation: str,
        env_id: str,
    ):
        self.assert_models_equal(
            oracle_shift_and_log_scale_diag_model,
            shift_and_log_scale_diag_model
        )

    @pytest.mark.parametrize("env_id,", ["Pendulum-v0"])
    @pytest.mark.parametrize(
        "hiddens, hidden_activation", [((256, 256), "relu")]
    )
    def test_q_net_models_equal(
        self,
        oracle_q_net: tf.keras.Model,
        actual_q_net: tf.keras.Model,
        hiddens: Tuple[int, ...],
        hidden_activation: str,
        env_id: str,
    ):
        self.assert_models_equal(oracle_q_net, actual_q_net)

    @staticmethod
    def assert_models_equal(expected: tf.keras.Model, actual: tf.keras.Model):
        # TODO(Adi): Ensure activations are also equal.
        for exp_weights, act_weights in zip(
            expected.get_weights(), actual.get_weights()
        ):
            if exp_weights.shape != act_weights.shape:
                pytest.fail(
                    f"Models not equal:"
                    f"\nExpected:"
                    f"\n{expected.summary()}"
                    f"\nActual"
                    f"\n{actual.summary()}"
                )


def test_version():
    assert __version__ == '0.1.0'
