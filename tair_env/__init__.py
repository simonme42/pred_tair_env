from gym.envs.registration import register

register(
    id='HVAC-tair-v0',
    entry_point='tair-env.envs:TairEnv',
)
