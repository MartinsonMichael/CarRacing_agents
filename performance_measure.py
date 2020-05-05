import argparse
import collections
import os
import time

import numpy as np
from typing import Any, Dict, List, Tuple

import wandb
import yaml
from multiprocessing import Process

from common_agents_utils import Config
from env import DiscreteWrapper
from env.common_envs_utils.env_makers import get_state_type_from_settings, get_EnvCreator_with_memory_safe_combiner
from ppo.PPO_ICM_continuous import PPO_ICM
from rainbow.rainbow import Rainbow
# from sac.SAC import SAC
from td3.TD3 import TD3


AGENT_TYPE_MAP = {
    "ppo": PPO_ICM,
    "rainbow": Rainbow,
    "td3": TD3,
    # "sac": SAC,
}


def make_common_env_config(exp_env_config: Dict[str, str]) -> Dict[str, Any]:
    assert isinstance(exp_env_config, dict)
    for config_part_name in ['path_config', 'reward_config', 'state_config']:
        assert config_part_name in exp_env_config.keys()
        assert isinstance(exp_env_config[config_part_name], str)

    env_config = dict()
    for config_part_name in ['path_config', 'reward_config', 'state_config']:
        config_part = yaml.load(open(exp_env_config[config_part_name], 'r'))
        env_config.update(config_part)

    return env_config


def make_single_config(agent_config_dict) -> Config:
    config = Config()
    for name, value in agent_config_dict.items():
        if hasattr(config, name):
            setattr(config, name, value)
            continue
        config.hyperparameters.update({name: value})
    return config


def make_general_agents_config(exp_agents_config: Dict[str, str], common_config_path: str) -> List[Dict[str, Any]]:
    # assert isinstance(exp_agents_config, dict)
    # assert len(exp_agents_config.keys()) >= 1
    # assert len(set(exp_agents_config.keys()) - {'ppo', 'rainbow', 'td3', 'sac'}) == 0

    general_agents_config_list = []
    for agent_type, path_to_agent_config in exp_agents_config.items():
        if agent_type == 'general_config':
            continue
        assert agent_type in AGENT_TYPE_MAP.keys()

        agent_hyperparameters = yaml.load(open(path_to_agent_config, 'r'))
        agent_common_config = yaml.load(open(common_config_path, 'r'))

        general_agents_config_list.append({
            'agent_class_name': agent_type,
            'common_config': agent_common_config,
            'agent_class': AGENT_TYPE_MAP[agent_type],
            'hyperparameters': agent_hyperparameters,
        })
    return general_agents_config_list


def deep_dict_update(d: Dict, u: Dict) -> Dict:
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def create_single_launch_name(env_config: Dict, agent_class_name: str) -> str:
    state_config = env_config['state']
    state_record = 'image' if state_config['picture'] is True else ""
    if len(state_config['vector_car_features']) != 0:
        if len(state_record) != 0:
            state_record += "__"
        state_record += "__".join(sorted(state_config['vector_car_features']))

    return agent_class_name + '__' + state_record + '__' + str(np.random.randint(0, 10**6))


WANDB_ENTITY = 'michaelmd'


def launch(exp_config: Dict[str, Any]) -> Tuple[float, str, str]:

    # for key, value in exp_config.items():
    #     print(key)
    #     print(value)
    #     print()

    changed_env_config = deep_dict_update(exp_config['env_config'], exp_config['env_change'])
    env = get_EnvCreator_with_memory_safe_combiner(changed_env_config)[0]()
    ENV_STEP_NUM = 100000

    start_time = time.time()
    env.reset()
    for _ in range(ENV_STEP_NUM):
        s, r, d, info = env.step([0, 0, 0])
        if d or info.get('need_reset', False):
            env.reset()
    end_time = time.time()

    name = []
    if len([x for x in changed_env_config['state']['vector_car_features'] if 'radar' not in x]) != 0:
        name.append('Vec')
    if changed_env_config['state']['picture']:
        name.append('Img')
    if len([x for x in changed_env_config['state']['vector_car_features'] if 'radar' in x]) != 0:
        name.append('Rad')

    return ENV_STEP_NUM / (end_time - start_time), str(" + ".join(name)), changed_env_config['bot_number'] != 0


def make_launch_list_from_config(exp_series_config) -> List[Any]:
    general_env_config = make_common_env_config(exp_series_config['env'])
    # general_agents_config = make_general_agents_config(
    #     exp_series_config['agents'],
    #     exp_series_config['general_config'],
    # )

    launch_list = []
    for env_change in exp_series_config.get('env_changes', [{}]):
        launch_list.append({
            'env_config': general_env_config,
            'env_change': env_change,
        })

    return launch_list


def main(_args):
    exp_series_config = yaml.load(open(_args.exp_settings, 'r'))
    launch_list = make_launch_list_from_config(exp_series_config)

    for launch_config in launch_list:
        value, name, use_bot = launch(launch_config)
        print(use_bot, name, value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-settings', type=str, help='path to experiment config')
    _args = parser.parse_args()

    main(_args)
