import tensorflow as tf


def add_mixins(base, mixins):
    """Returns a new class with mixins applied in priority order."""

    mixins = list(mixins or [])

    while mixins:

        class new_base(mixins.pop(), base):
            pass

        base = new_base

    return base


def make_tf_callable(session_or_none, dynamic_shape=False):
    """Returns a function that can be executed in either graph or eager mode.

    The function must take only positional args.

    If eager is enabled, this will act as just a function. Otherwise, it
    will build a function that executes a session run with placeholders
    internally.

    Arguments:
        session_or_none (tf.Session): tf.Session if in graph mode, else None.
        dynamic_shape (bool): True if the placeholders should have a dynamic
            batch dimension. Otherwise they will be fixed shape.

    Returns:
        a Python function that can be called in either mode.
    """

    assert session_or_none is not None

    def make_wrapper(fn):
        if session_or_none:
            placeholders = []
            symbolic_out = [None]

            def call(*args):
                args_flat = []
                for a in args:
                    if type(a) is list:
                        args_flat.extend(a)
                    else:
                        args_flat.append(a)
                args = args_flat
                if symbolic_out[0] is None:
                    with session_or_none.graph.as_default():
                        for i, v in enumerate(args):
                            if dynamic_shape:
                                if len(v.shape) > 0:
                                    shape = (None,) + v.shape[1:]
                                else:
                                    shape = ()
                            else:
                                shape = v.shape
                            placeholders.append(
                                tf.placeholder(
                                    dtype=v.dtype, shape=shape, name="arg_{}".format(i)
                                )
                            )
                        symbolic_out[0] = fn(*placeholders)
                feed_dict = dict(zip(placeholders, args))
                return session_or_none.run(symbolic_out[0], feed_dict)

            return call
        else:
            return fn

    return make_wrapper
