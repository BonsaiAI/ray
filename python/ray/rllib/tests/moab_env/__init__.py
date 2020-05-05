from gym.envs.registration import register

# Register a custom environment for frozen lake with deterministic states
register(
    id='Moab-v0',
    entry_point='moab_env.moab_env:MoabSim',
    kwargs={
        "config":{
            "use_normalize_action": True,
            "use_normalize_state": True,
            "use_dr": False
        }
    },
    reward_threshold=1000,
)
