from gym.envs.registration import register
from env.CarIntersect.environment import CarIntersect

try:
    register(
        id='CarIntersect',
        entry_point='environment:CarIntersect',
    )
except:
    print('fail to register gym env \'CarIntersect\'')
