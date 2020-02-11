from gym.envs.registration import register
from envs.gym_car_intersect_fixed.environment import CarRacingHackatonContinuousFixed

try:
    register(
        id='CarIntersect-v5',
        entry_point='envs.gym_car_intersect_fixed:CarRacingHackatonContinuousFixed',
    )
    # print('successfully register gym env \'CarIntersect-v5\'')
except:
    print('fail to register gym env \'CarIntersect-v5\'')
