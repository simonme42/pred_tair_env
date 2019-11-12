from gym.envs.registration import register

register(
    id='HVACTair-v0',
    entry_point='tair_env.envs:HVACTairEnv_1846',
)

register(
    id='HVACTair-v1',
    entry_point='tair_env.envs:HVACTairEnv_7390',
)
