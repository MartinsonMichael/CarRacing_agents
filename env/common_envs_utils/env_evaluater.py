import os
from multiprocessing import Process
from typing import Callable, Optional
import time

from common_agents_utils.logger import Logger
from env import CarIntersectEvalWrapper, CarIntersect
from env.common_envs_utils.visualizer import save_as_mp4


def create_eval_env(env):
    return CarIntersectEvalWrapper(env)


def evaluate_and_log(
        eval_env: CarIntersect,
        action_get_method: Callable,
        logger: Logger,
        log_animation: bool = True,
        exp_class: Optional[str] = None,
        exp_name: Optional[str] = None,
        max_episode_len: int = 500,
        debug: bool = False,
) -> None:
    state = eval_env.reset()

    if debug:
        print('DEBUG MODE')
        print(f"state type : {type(state)}")
        print(f"state len {len(state)}")
        if hasattr(state, 'shape'):
            print(f"state shape : {state.shape}")

        print('Try to get action...')
        action = action_get_method(state)
        print(f"action type : {type(action)}")
        if hasattr(action, 'shape'):
            print(f"action shape : {action.shape}")

    total_reward = 0
    total_steps = 0
    last_info = {}

    img = []

    state = eval_env.reset()

    while True:
        action = action_get_method(state)
        state, reward, done, info = eval_env.step(action)

        total_reward += reward
        total_steps += 1
        last_info = info

        if log_animation:
            img.append(eval_env.render(full_image=True))

        if done:
            break

        if max_episode_len is not None and total_steps > max_episode_len:
            break

    logger.log_and_publish({
        'EVAL reward': total_reward,
        'EVAL track_progress': last_info.get('track_progress', 0),
        'EVAL env_steps': total_steps,
    })

    if log_animation:
        assert exp_name is not None
        assert exp_class is not None
        Process(
            target=save_as_mp4,
            args=(
                img,
                os.path.join(
                    'animation',
                    exp_class,
                    exp_name,
                    f"EVAL_R:_{total_reward}_Time:_{total_steps}_{time.time()}.mp4",
                ),
                True
            ),
        ).start()
