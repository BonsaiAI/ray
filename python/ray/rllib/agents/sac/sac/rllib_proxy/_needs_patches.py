import logging

import tensorflow as tf
from ray.rllib import models
from ray.rllib.agents.sac.sac.rllib_proxy._modelv1_compat import make_v1_wrapper
from ray.rllib.agents.sac.sac.rllib_proxy._tf_model_v2 import ModelV2
from ray.rllib.agents.sac.proxy.utils.tf_ops import executing_eagerly
from ray.rllib.utils.annotations import DeveloperAPI
from ray.tune.registry import RLLIB_MODEL, _global_registry

logger = logging.getLogger(__name__)

__all__ = ["ModelCatalog"]


class ModelCatalog(models.ModelCatalog):
    @staticmethod
    def _wrap_if_needed(model_cls, model_interface):

        if not model_interface or issubclass(model_cls, model_interface):
            return model_cls

        class wrapper(model_interface, model_cls):
            pass

        name = "{}_as_{}".format(model_cls.__name__, model_interface.__name__)
        wrapper.__name__ = name
        wrapper.__qualname__ = name

        return wrapper

    @staticmethod
    @DeveloperAPI
    def get_model_v2(
        obs_space,
        action_space,
        num_outputs,
        model_config,
        framework,
        name="default_model",
        model_interface=None,
        default_model=None,
        **model_kwargs
    ):
        """Returns a suitable model compatible with given spaces and output.

        Args:
            obs_space (Space): Observation space of the target gym env. This
                may have an `original_space` attribute that specifies how to
                unflatten the tensor into a ragged tensor.
            action_space (Space): Action space of the target gym env.
            num_outputs (int): The size of the output vector of the model.
            framework (str): Either "tf" or "torch".
            name (str): Name (scope) for the model.
            model_interface (cls): Interface required for the model
            default_model (cls): Override the default class for the model. This
                only has an effect when not using a custom model
            model_kwargs (dict): args to pass to the ModelV2 constructor

        Returns:
            model (ModelV2): Model to use for the policy.
        """

        if model_config.get("custom_model"):
            model_cls = _global_registry.get(RLLIB_MODEL, model_config["custom_model"])
            if issubclass(model_cls, ModelV2):
                if framework == "tf":
                    logger.info("Wrapping {} as {}".format(model_cls, model_interface))
                    model_cls = ModelCatalog._wrap_if_needed(model_cls, model_interface)
                    created = set()

                    # Track and warn if vars were created but not registered
                    def track_var_creation(next_creator, **kw):
                        v = next_creator(**kw)
                        created.add(v)
                        return v

                    with tf.variable_creator_scope(track_var_creation):
                        instance = model_cls(
                            obs_space,
                            action_space,
                            num_outputs,
                            model_config,
                            name,
                            **model_kwargs
                        )
                    registered = set(instance.variables())
                    not_registered = set()
                    for var in created:
                        if var not in registered:
                            not_registered.add(var)
                    if not_registered:
                        raise ValueError(
                            "It looks like variables {} were created as part "
                            "of {} but does not appear in model.variables() "
                            "({}). Did you forget to call "
                            "model.register_variables() on the variables in "
                            "question?".format(not_registered, instance, registered)
                        )
                else:
                    # no variable tracking
                    instance = model_cls(
                        obs_space,
                        action_space,
                        num_outputs,
                        model_config,
                        name,
                        **model_kwargs
                    )
                return instance
            elif executing_eagerly():
                raise ValueError(
                    "Eager execution requires a TFModelV2 model to be "
                    "used, however you specified a custom model {}".format(model_cls)
                )

        if framework == "tf":
            v2_class = None
            # try to get a default v2 model
            if not model_config.get("custom_model"):
                v2_class = default_model or ModelCatalog._get_v2_model(
                    obs_space, model_config
                )
            # fallback to a default v1 model
            if v2_class is None:
                if executing_eagerly():
                    raise ValueError(
                        "Eager execution requires a TFModelV2 model to be "
                        "used, however there is no default V2 model for this "
                        "observation space: {}, use_lstm={}".format(
                            obs_space, model_config.get("use_lstm")
                        )
                    )
                v2_class = make_v1_wrapper(ModelCatalog.get_model)
            # wrap in the requested interface
            wrapper = ModelCatalog._wrap_if_needed(v2_class, model_interface)
            return wrapper(
                obs_space, action_space, num_outputs, model_config, name, **model_kwargs
            )
        elif framework == "torch":
            if default_model:
                return default_model(
                    obs_space, action_space, num_outputs, model_config, name
                )
            return ModelCatalog._get_default_torch_model_v2(
                obs_space, action_space, num_outputs, model_config, name
            )
        else:
            raise NotImplementedError(
                "Framework must be 'tf' or 'torch': {}".format(framework)
            )
