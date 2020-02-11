from gym.envs.registration import register
from envs.gym_car_intersect.hack_env__latest import CarRacingHackatonContinuous2

try:
    register(
        id='CarIntersect-v3',
        entry_point='envs.gym_car_intersect:CarRacingHackatonContinuous2',
    )
    # print('successfully register gym env \'CarIntersect-v3\'')
except:
    print('fail to register gym env \'CarIntersect-v3\'')
