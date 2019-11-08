from gym.envs.registration import register

register(
    id='HVACTair-v0',
    entry_point='tair_env.envs:HVACTairEnv',
)
