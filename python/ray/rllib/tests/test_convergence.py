import gym
import pytest
import ray

# from ray.rllib.agents.ppo import DEFAULT_CONFIG, PPOTrainer as Trainer
from ray.rllib.agents.sac import DEFAULT_CONFIG, SACTrainer as Trainer
from moab_env.moab_env import MoabSim


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
        ray.init(local_mode=True)
    yield
    ray.shutdown()


"""
@pytest.fixture
def config(env_id):
    config = DEFAULT_CONFIG.copy()
    config["env_config"] = dict(env_id=env_id)
    if env_id == "Pendulum-v0":
        config["timesteps_per_iteration"] = 100
    config["num_workers"] = 8
    config["num_envs_per_worker"] = 8
    return config
"""


@pytest.fixture
def trainer(config):
    return Trainer(config=config, env=MoabSim)


#@pytest.mark.parametrize(
#    "env_id, threshold", [
#        ("Pendulum-v0", -750),
#    ]
#)
#@pytest.mark.usefixtures("ray_env")
#def test_convergence(trainer, env_id, threshold):
#    mean_episode_reward = -float("inf")
#    for i in range(500):
#        result = trainer.train()
#        mean_episode_reward = result["episode_reward_mean"]
#        print(f"{i}: {mean_episode_reward}")
#    assert mean_episode_reward >= threshold


@pytest.fixture
def config(env_id):
    config = DEFAULT_CONFIG.copy()
    config["env_config"] = {
        "use_normalize_action": False,
        "use_dr": False,
        "use_normalize_state": True
    }
    config["evaluation_config"] = {
        "exploration_enabled": False
    }
    config["evaluation_interval"] = 1
    config["evaluation_num_episodes"] = 10
    if env_id == "Moab-v0":
        config["timesteps_per_iteration"] = 2000
    config["num_workers"] = 8
    config["num_envs_per_worker"] = 8
    config["learning_starts"] = 6000
    config["normalize_actions"] = True
    return config


@pytest.mark.parametrize(
    "env_id, threshold", [
        ("Moab-v0", 1000),
    ]
)
@pytest.mark.usefixtures("ray_env")
def test_convergence(trainer, env_id, threshold):
    mean_episode_reward = -float("inf")
    for i in range(10000):
        result = trainer.train()
        mean_episode_reward = result["episode_reward_mean"]
        total_samples = result["timesteps_total"]
        print(f"SAC Iteration {i}: Train Episode Reward Mean: "
              f"{mean_episode_reward}, "
              f"Number of samples {total_samples}")
