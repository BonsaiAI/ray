import logging

from ray.rllib.agents import Trainer as UnpatchedTrainer
from ray.rllib.utils.annotations import DeveloperAPI, PublicAPI

logger = logging.getLogger(__name__)


@PublicAPI
class Trainer(UnpatchedTrainer):
    """A trainer coordinates the optimization of one or more RL policies.

    All RLlib trainers extend this base class, e.g., the A3CTrainer implements
    the A3C algorithm for single and multi-agent training.

    Trainer objects retain internal model state between calls to train(), so
    you should create a new trainer instance for each training session.

    Attributes:
        env_creator (func): Function that creates a new training env.
        config (obj): Algorithm-specific configuration data.
        logdir (str): Directory in which training outputs should be placed.
    """

    @DeveloperAPI
    def _make_workers(self, env_creator, policy, config, num_workers):
        config.setdefault(
            "local_evaluator_tf_session_args",
            config.get("local_tf_session_args")
        )
        self.config["multiagent"].setdefault(
            # PolicyGraph has been renamed to Policy in v 0.8.1
            "policy_graphs",
            self.config["multiagent"].get("policies")
        )
        self.local_evaluator = self.make_local_evaluator(env_creator, policy,
                                                         config)
        self.remote_evaluators = self.make_remote_evaluators(env_creator, policy,
                                                             num_workers)
        return self.local_evaluator, self.remote_evaluators
