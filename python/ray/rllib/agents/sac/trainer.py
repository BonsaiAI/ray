from ray.rllib.agents.sac.config import DEFAULT_CONFIG
from ray.rllib.agents.sac.sac.sac_policy import SACTFPolicy
from ray.rllib.agents.sac.proxy.agents.dqn.trainer import GenericOffPolicyTrainer

SACTrainer = GenericOffPolicyTrainer.with_updates(
    name="SAC", default_config=DEFAULT_CONFIG, default_policy=SACTFPolicy)
