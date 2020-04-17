import argparse
import time
from gym.envs.classic_control.rendering import SimpleImageViewer
from pyglet.window import key

if __name__ == '__main__':
    try:
        import os
        import sys
        sys.path.insert(0, os.path.abspath(os.path.pardir))
    except:
        print("If you launch this from env folder, you probably will have some import problems.")

from env.CarRacing_env.environment import CarRacingEnv
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


def main():
    global restart, action
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, default=None, help="time in s between actions")
    parser.add_argument("--debug", action='store_true', default=False, help="debug mode")
    parser.add_argument("--stop-on-finish", action='store_true', default=False,
                        help="do no restart simulation after finish")
    parser.add_argument("--stop-on-fail", action='store_true', default=False,
                        help="do not restart simulation after fail")
    parser.add_argument(
        "--env-settings",
        type=str,
        default='settings_sets/env_settings__TEST.json',
        help="debug mode"
    )
    args = parser.parse_args()

    # create env by settings
    env = CarRacingEnv(args.env_settings)
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
            viewer.imshow(env.render(full_image=True))

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
