from gym.envs.registration import register

register(
    id='HVAC-Tair-v0',
    entry_point='tair_env.envs:HVACTairEnv',
)
