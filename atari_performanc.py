import time

import gym


def measure() -> float:

    env = gym.make('BreakoutDeterministic-v4')
    ENV_STEP_NUM = 100000

    start_time = time.time()
    env.reset()
    for _ in range(ENV_STEP_NUM):
        s, r, d, info = env.step(0)
        if d or info.get('need_reset', False):
            env.reset()
    end_time = time.time()

    return ENV_STEP_NUM / (end_time - start_time)


if __name__ == '__main__':
    print(measure())
