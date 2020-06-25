from gym.envs.registration import register
from env.CarIntersect.environment import CarIntersect

try:
    register(
        id='CarIntersect-v0',
        entry_point='env.CarIntersect.environment:CarIntersect',
    )
except:
    print('fail to register gym env \'CarIntersect\'')
