from gym.envs.registration import register
from env.CarRacing_env.environment import CarRacingEnv

try:
    register(
        id='CarIntersect-v5',
        entry_point='CarRacing_env:CarRacingHackatonContinuousFixed',
    )
    # print('successfully register gym env \'CarIntersect-v5\'')
except:
    print('fail to register gym env \'CarIntersect-v5\'')
