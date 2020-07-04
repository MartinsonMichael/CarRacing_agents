from gym.envs.registration import register
from env.CarIntersect.environment import CarIntersect
from env.CarIntersect.car import DummyCar
from env.CarIntersect.utils import DataSupporter
from env.CarIntersect.cvat_loader import CvatDataset
from env.CarIntersect.rewards import Rewarder
from env.CarIntersect.contact_listner import RefactoredContactListener

try:
    register(
        id='CarIntersect-v0',
        entry_point='env.CarIntersect.environment:CarIntersect',
    )
except:
    print('fail to register gym env \'CarIntersect\'')
