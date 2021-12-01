import gym

from gym.envs.registration import register

register(id='CrossroadEnd2endMixPiFix-v0', entry_point='environment.env_mix_single.endtoend:CrossroadEnd2endMixPiFix')

register(id='HorizonCrossing-v0', entry_point='environment.env_horizon.crossing:HorizonCrossingEnv')

register(id='HorizonMultiLane-v0', entry_point='environment.env_horizon.multi_lane:HorizonMultiLaneEnv')


register(id='HorizonCrossing-v1', entry_point='environment.env_horizon.crossing_test:HorizonCrossingEnv')