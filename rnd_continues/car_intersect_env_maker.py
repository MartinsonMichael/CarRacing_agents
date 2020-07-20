try:
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.pardir))

    from env.common_envs_utils.env_wrappers import OnlyImageTaker, DictToTupleWrapper, ImageToGreyScale
    from env.common_envs_utils.action_wrappers import DiscreteWrapper
    from env.common_envs_utils.env_evaluater import evaluate_and_log, create_eval_env

    from common_agents_utils.logger import Logger

    from env import CarIntersect
except:
    print("If you launch this from . folder, you probably will have some import problems.")


def makeCarIntersect(settings):
    env = CarIntersect(settings_file_path_or_settings=settings)
    env = DictToTupleWrapper(env)
    # env = ChannelSwapper(env)
    env = ImageToGreyScale(env)
    # env = DiscreteWrapper(env)
    env = OnlyImageTaker(env)

    return env
