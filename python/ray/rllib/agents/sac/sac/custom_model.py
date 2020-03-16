from collections import OrderedDict
from typing import Tuple, Dict, Iterable, Union, Optional, Sequence, Callable

import gym
import numpy as np
import tensorflow as tf

from sac import config
from sac.rllib_proxy import TFModelV2, normc_initializer, get_activation_fn

ArrayShape = Tuple[int, ...]
KernelInitializer = Callable[[], tf.keras.initializers.Initializer]


def build_fcn(
    input_shapes: Dict[str, Tuple[int, ...]],
    num_outputs: int,
    hidden_layer_sizes: Sequence[int],
    hidden_activations: Union[str, Sequence[str]],
    output_activation: Optional[str] = None,
    kernel_initializers: Optional[
        Union[KernelInitializer, Sequence[KernelInitializer]]
    ] = None,
    name=None,
) -> tf.keras.Model:

    n_hidden_layers = len(hidden_layer_sizes)
    if isinstance(hidden_activations, str):
        hidden_activations = [hidden_activations] * n_hidden_layers
    if kernel_initializers is None:
        kernel_initializers = lambda: normc_initializer(1.)
    if not isinstance(kernel_initializers, Iterable):
        kernel_initializers = [kernel_initializers for _ in range(n_hidden_layers + 1)]

    assert len(kernel_initializers) == n_hidden_layers + 1
    assert len(hidden_activations) == n_hidden_layers

    inputs = [
        tf.keras.layers.Input(shape=shape, name=name)
        for name, shape in input_shapes.items()
    ]
    if len(inputs) > 1:
        last_layer = tf.keras.layers.Concatenate(axis=1)(inputs)
    else:
        last_layer = inputs[0]
    for i, layer_size in enumerate(hidden_layer_sizes):
        last_layer = tf.keras.layers.Dense(
            layer_size,
            name=f"fc_{i}",
            activation=hidden_activations[i],
            kernel_initializer=kernel_initializers[i]()
        )(last_layer)
    output = tf.keras.layers.Dense(
        num_outputs,
        name="fc_out",
        activation=output_activation,
        kernel_initializer=kernel_initializers[-1]()
    )(last_layer)
    return tf.keras.Model(inputs, [output], name=name)


def build_model(
    obs_space: gym.Space,
    action_space: gym.Space,
    num_outputs: int,
    model_config: dict,
    name: str,
) -> tf.keras.Model:
    """

    :param obs_space: observation space of the target gym env. This may have an
        `original_space` attribute that specifies how to unflatten the tensor into a
        ragged tensor.
    :param action_space: action space of the target gym env
    :param num_outputs: number of output units of the model
    :param model_config: config for the model, documented in ModelCatalog
    :param name: name (scope) for the model
    """
    activation = get_activation_fn(model_config.get("fcnet_activation"))
    hiddens = model_config.get("fcnet_hiddens")
    no_final_linear = model_config.get("no_final_linear")
    vf_share_layers = model_config.get("vf_share_layers")

    # Bonsai SDK returns the observation dictionary so we can perform transformations in
    # the tensorflow graph. So we should look at the original_space instead of the
    # flattened obs_space
    if hasattr(obs_space, "original_space"):
        original_space = obs_space.original_space
        inputs = [
            tf.keras.layers.Input(shape=space.shape, name=name)
            for name, space in original_space.spaces.items()
        ]
        inputs = tf.keras.layers.Concatenate(axis=1)(inputs)
    else:
        inputs = tf.keras.layers.Input(
            shape=(np.product(obs_space.shape),), name="observations"
        )
    last_layer = inputs
    i = 1

    if no_final_linear:
        # the last layer is adjusted to be of size num_outputs
        for size in hiddens[:-1]:
            last_layer = tf.keras.layers.Dense(
                size,
                name="fc_{}".format(i),
                activation=activation,
                kernel_initializer=normc_initializer(1.0),
            )(last_layer)
            i += 1
        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="fc_out",
            activation=activation,
            kernel_initializer=normc_initializer(1.0),
        )(last_layer)
    else:
        # the last layer is a linear to size num_outputs
        for size in hiddens:
            last_layer = tf.keras.layers.Dense(
                size,
                name="fc_{}".format(i),
                activation=activation,
                kernel_initializer=normc_initializer(1.0),
            )(last_layer)
            i += 1
        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="fc_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01),
        )(last_layer)

    if not vf_share_layers:
        # build a parallel set of hidden layers for the value net
        last_layer = inputs
        i = 1
        for size in hiddens:
            last_layer = tf.keras.layers.Dense(
                size,
                name="fc_value_{}".format(i),
                activation=activation,
                kernel_initializer=normc_initializer(1.0),
            )(last_layer)
            i += 1

    value_out = tf.keras.layers.Dense(
        1,
        name="value_out",
        activation=None,
        kernel_initializer=normc_initializer(0.01),
    )(last_layer)

    reshaped_value_out = tf.keras.layers.Reshape([-1])(value_out)

    return tf.keras.Model(inputs, [layer_out, reshaped_value_out], name=name)


class CustomModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.custom_model_config: dict = model_config["custom_options"]
        if self.custom_model_config[config.CustomKeys.UseRLLibModel]:
            self.base_model = build_model(
                obs_space, action_space, num_outputs, model_config, name
            )
        else:
            raise NotImplementedError(
                "This functionality will be implemented in SAC v1"
            )

        self.observation_transform = lambda x: x
        self._value_out = None

        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        observations: "OrderedDict[str, tf.Tensor]" = input_dict["obs"]
        transformed_observations = self.observation_transform(observations)
        model_out, self._value_out = self.base_model(transformed_observations)
        return model_out, state

    def value_function(self):
        return self._value_out
