"""ApexSAC Trainer and default config."""

from ray.rllib.agents.dqn.apex import APEX_TRAINER_PROPERTIES
from ray.rllib.agents.sac.sac import SACTrainer, DEFAULT_CONFIG as SAC_CONFIG

# yapf: disable
# __sphinx_doc_begin__

APEX_SAC_DEFAULT_CONFIG = SACTrainer.merge_trainer_configs(
    SAC_CONFIG,  # see also the options in sac.py, which are also supported
    {
        "optimizer": {
            "max_weight_sync_delay": 400,
            "num_replay_buffer_shards": 4,
            "debug": False,
        },
        "n_step": 1,
        "num_gpus": 0,
        "num_workers": 32,
        "buffer_size": 2000000,
        "learning_starts": 50000,
        "train_batch_size": 512,
        "rollout_fragment_length": 50,
        "target_network_update_freq": 0,
        "timesteps_per_iteration": 25000,
        "exploration_config": {"type": "PerWorkerEpsilonGreedy"},
        "worker_side_prioritization": True,
        "min_iter_time_s": 30,
        "prioritized_replay": True,
    },
)

# __sphinx_doc_end__
# yapf: enable

ApexSACTrainer = SACTrainer.with_updates(
    name="APEX_SAC", default_config=APEX_SAC_DEFAULT_CONFIG, **APEX_TRAINER_PROPERTIES
)