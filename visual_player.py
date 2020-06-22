import argparse
import time
from typing import Dict, Any

import yaml
from gym.envs.classic_control.rendering import SimpleImageViewer
from pyglet.window import key

if __name__ == '__main__':
    try:
        import os
        import sys
        sys.path.insert(0, os.path.abspath(os.path.pardir))
    except:
        print("If you launch this from env folder, you probably will have some import problems.")

from env.CarIntersect.environment import CarIntersect
from env.common_envs_utils.env_wrappers import DiscreteWrapper

action = 0
restart = False
start_to_use_pause = False
KEYS = {key.LEFT, key.RIGHT, key.UP, key.DOWN}
KEY_MAP = {
    key.LEFT: 1,
    key.RIGHT: 2,
    key.UP: 3,
    key.DOWN: 4,
}
print('''
To control car use Arrows.
Up to accelerate, Down to break,
Left/Right to rotate steer,
note that car wont rotate immediately.
''')


def key_press(k, modifier):
    global restart, action, start_to_use_pause
    if k == key.ESCAPE:
        restart = True
    if k in KEYS:
        action = KEY_MAP[k]
    if k == key.P:
        start_to_use_pause = not start_to_use_pause


def key_release(k, modifier):
    global action
    if k in KEYS:
        action = 0


def make_common_env_config(exp_env_config: Dict[str, str]) -> Dict[str, Any]:
    assert isinstance(exp_env_config, dict)
    for config_part_name in ['path_config', 'reward_config', 'state_config']:
        assert config_part_name in exp_env_config.keys()
        assert isinstance(exp_env_config[config_part_name], str)

    env_config = dict()
    for config_part_name in ['path_config', 'reward_config', 'state_config']:
        config_part = yaml.load(open(exp_env_config[config_part_name], 'r'))
        env_config.update(config_part)

    print(env_config)

    return env_config


def main():
    global restart, action
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, default=None, help="time in s between actions")
    parser.add_argument("--debug", action='store_true', default=False, help="debug mode")
    parser.add_argument("--stop-on-finish", action='store_true', default=False,
                        help="do no restart simulation after finish")
    parser.add_argument("--stop-on-fail", action='store_true', default=False,
                        help="do not restart simulation after fail")
    parser.add_argument("--exp-settings", type=str)
    parser.add_argument("--env-settings", type=str)
    parser.add_argument("--show-agent-track", default=False, action='store_true', help="debug mode")
    args = parser.parse_args()

    if args.exp_settings is not None:
        print('make env by exp-settings')
        env = CarIntersect(make_common_env_config(yaml.load(open(args.exp_settings, 'r'))['env']))
    elif args.env_settings is not None:
        print('make env by json settings')
        env = CarIntersect(args.env_settings)
    else:
        raise ValueError('provide settings to use')

    env = DiscreteWrapper(env)

    env.reset()
    time.sleep(3.0)

    # create window with image
    viewer = SimpleImageViewer()
    viewer.imshow(env.env.render(full_image=True))
    viewer.window.on_key_press = key_press
    viewer.window.on_key_release = key_release
    while True:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            done = None
            info = {}
            for _ in range(1):
                s, r, done, info = env.step(action)
                total_reward += r
            print(f"action {action}")
            print("step {} total_reward {}".format(steps, total_reward))

            print('Car info:')
            print(info, end='\n\n')

            steps += 1
            viewer.imshow(env.render_with_settings(full_image=True, draw_agent_track=args.show_agent_track))

            if args.delay is not None:
                time.sleep(args.delay)

            if start_to_use_pause:
                time.sleep(1.0)

            if done and args.stop_on_finish and info.get('is_finish', False):
                print('Exit on finish')
                exit(0)

            if done and args.stop_on_fail:
                print('Exit on fail')
                exit(0)

            if done or restart or info.get('need_reset', False):
                print('restart')
                break


if __name__ == "__main__":
    main()
