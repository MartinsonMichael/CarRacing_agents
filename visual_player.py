import os
import argparse

import time
import gym
from gym.envs.classic_control.rendering import SimpleImageViewer
from pyglet.window import key

from envs.common_envs_utils import *
from envs.gym_car_intersect import *
from envs.gym_car_intersect_fixed import *

from envs.gym_car_intersect_fixed.environment import CarRacingHackatonContinuousFixed

from envs.common_envs_utils.env_wrappers import DiscreteWrapper

action = 0
restart = False
KEYS = {key.LEFT, key.RIGHT, key.UP, key.DOWN}
KEY_MAP = {
    key.LEFT: 1,
    key.RIGHT: 2,
    key.UP: 3,
    key.DOWN: 4,
}


def key_press(k, modifier):
    global restart, action
    if k == key.ESCAPE:
        restart = True
    if k in KEYS:
        action = KEY_MAP[k]


def key_release(k, modifier):
    global action
    if k in KEYS:
        action = 0


def main():
    global restart, action
    parser = argparse.ArgumentParser()
    parser.add_argument("--bot", type=int, default=0,
                        help="Number of bot cars_full in environment.")
    parser.add_argument("--track", type=int, default=0,
                        help="Track for agents cars_full in environment.")
    parser.add_argument("--discrete", type=int, default=1, help="Apply discrete wrapper?")
    parser.add_argument("--sleep", type=float, default=None, help="time in s between actions")
    parser.add_argument("--debug", action='store_true', default=False, help="debug mode")
    parser.add_argument(
        "--env-settings",
        type=str,
        default='envs/gym_car_intersect_fixed/settings_sets/env_settings__basic_small_rotation.json',
        help="debug mode"
    )

    args = parser.parse_args()


    env = CarRacingHackatonContinuousFixed(args.env_settings)
    env = DiscreteWrapper(env)


    env.reset()
    time.sleep(3.0)

    viewer = SimpleImageViewer()
    viewer.imshow(env.get_true_picture())
    viewer.window.on_key_press = key_press
    viewer.window.on_key_release = key_release
    # while True:
    env.reset()
    total_reward = 0.0
    steps = 0
    restart = False
    while True:
        s = None
        done = None
        info = {}
        for _ in range(1):
            s, r, done, info = env.step(action)
            total_reward += r
        print("\naction " + str(action))
        print("step {} total_reward {}".format(steps, total_reward))
        print(info)

        steps += 1
        viewer.imshow(env.get_true_picture())

        if done or restart or 'need_restart' in info.keys():
            print('restart')
            break


if __name__ == "__main__":
    main()
