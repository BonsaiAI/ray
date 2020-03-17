from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

import tensorflow as tf

from sac.rllib_proxy import normc_initializer

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


