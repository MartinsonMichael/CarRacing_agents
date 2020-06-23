import numpy as np
import gym


class DiscreteWrapper(gym.ActionWrapper):
    def reverse_action(self, action):
        raise NotImplemented

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.discrete.Discrete(5)

    def action(self, action):
        if action == 0:
            return [0.0, 0.0, 0.0]
        if action == 1:
            return [-0.6, 0.0, 0.0]
        if action == 2:
            return [+0.6, 0.0, 0.0]
        if action == 3:
            return [0.0, 0.8, 0.0]
        if action == 4:
            return [0.0, 0.0, 1.0]
        raise KeyError


class ExtendedDiscreteWrapper(gym.ActionWrapper):
    def reverse_action(self, action):
        raise NotImplemented

    def __init__(self, env):
        super().__init__(env)
        self.action_space.n = 7
        self.action_space = gym.spaces.discrete.Discrete(self.action_space.n)

    def action(self, action):
        if action == 0:
            return [0, 0, 0]
        if action == 1:
            return [-0.6, 0.1, 0]
        if action == 2:
            return [-0.3, 0.2, 0]

        if action == 3:
            return [+0.6, 0.1, 0]
        if action == 4:
            return [+0.3, 0.2, 0]

        if action == 5:
            return [0.0, 0.3, 0]
        if action == 6:
            return [0.0, 0.0, 1.0]
        raise KeyError


class DiscreteOnlyLRWrapper(gym.ActionWrapper):

    def reverse_action(self, action):
        raise NotImplemented

    def __init__(self, env):
        super().__init__(env)
        self.action_space.n = 3
        self.action_space = gym.spaces.discrete.Discrete(self.action_space.n)

    def action(self, action):
        steer = 0.6
        speed = 0.2
        if action == 0:
            return [0, 0, 0]
        if action == 1:
            return [-steer, speed, 0]
        if action == 2:
            return [+steer, speed, 0]
        raise KeyError


class ContinuesCartPolyWrapper(gym.ActionWrapper):

    def reverse_action(self, action):
        print('HERE!!!!!')
        raise NotImplemented

    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        print(action)
        return np.array([0 if x < 0 else 1 for x in np.array(action).ravel()], dtype=np.uint8)


class ContinueOnlyLRWrapper(gym.ActionWrapper):

    def reverse_action(self, action):
        raise NotImplemented

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.box.Box(low=-1.0, high=1.0, shape=(1, ), dtype=np.float32)

    def action(self, action):
        # action shape is (1, ) and it is steer
        # we should return (3, )
        steer = action[0]
        speed = 0.2
        return [steer, speed, 0]
