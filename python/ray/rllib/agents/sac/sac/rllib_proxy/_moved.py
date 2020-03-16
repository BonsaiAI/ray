""" Classes and functions that have been moved between to different places between
ray 0.8.1 and ray 0.6.6
"""
from sac.dev_utils import using_ray_8

if using_ray_8():
    from ray.tune.resources import Resources
    from ray.rllib.models.tf.misc import get_activation_fn
else:
    from ray.tune.trial import Resources
    from ray.rllib.models.misc import get_activation_fn

__all__ = [
    "Resources",
    "get_activation_fn",
]
