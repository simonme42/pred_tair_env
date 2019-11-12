from gym.envs.registration import register

register(
    id='HVACTair-v0_1846',
    entry_point='tair_env.envs:HVACTairEnv_1846',
)

register(
    id='HVACTair-v1_7390',
    entry_point='tair_env.envs:HVACTairEnv_7390',
)
