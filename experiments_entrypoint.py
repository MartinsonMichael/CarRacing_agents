import argparse
import collections
from typing import Callable, Any, Dict, List, Optional
import yaml

from common_agents_utils import Config
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


def make_general_env_config(exp_env_config: Dict[str, str]) -> Dict[str, Any]:
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
    assert isinstance(exp_agents_config, dict)
    assert len(exp_agents_config.keys()) >= 1
    assert len(set(exp_agents_config.keys()) - {'ppo', 'rainbow', 'td3', 'sac'}) == 0

    general_agents_config_list = []
    for agent_type, path_to_agent_config in exp_agents_config.items():
        if agent_type == 'general_config':
            continue
        assert agent_type in AGENT_TYPE_MAP.keys()

        agent_hyperparameters = yaml.load(open(path_to_agent_config, 'r'))
        agent_common_config = yaml.load(open(common_config_path, 'r'))

        general_agents_config_list.append({
            'agent_name': agent_type,
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


def update_agent_config():
    pass


def launch(exp_config: Dict[str, Any]) -> None:

    for key, value in exp_config.items():
        print(key)
        print(value)
        print()

    final_agent_config = make_single_config(exp_config['common_config'])
    changed_env_config = deep_dict_update(exp_config['env_config'], exp_config['env_change'])

    final_agent_config.record_animation = True
    final_agent_config.device = exp_config['device']

    final_agent_config.env_config = changed_env_config
    final_agent_config.mode = get_state_type_from_settings(changed_env_config)
    final_agent_config.hyperparameters = exp_config['hyperparameters']
    final_agent_config.environment_make_function, final_agent_config.phi = \
        get_EnvCreator_with_memory_safe_combiner(changed_env_config)
    final_agent_config.test_environment_make_function = final_agent_config.environment_make_function

    print('final config:')
    print(final_agent_config)

    print('start to create agent..')
    agent = exp_config['agent_class'](final_agent_config)

    print('start to train agent...')
    agent.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-record-animation', default=False, action='store_true', help='use icm')
    parser.add_argument('--name', type=str, help='name for experiment')
    parser.add_argument('--exp-settings', type=str, help='path to experiment config')
    parser.add_argument('--device', type=str, help='name for experiment')
    _args = parser.parse_args()
    _args.record_animation = not _args.no_record_animation

    devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']

    exp_series_config = yaml.load(open(_args.exp_settings, 'r'))

    print(exp_series_config)

    general_env_config = make_general_env_config(exp_series_config['env'])
    general_agents_config = make_general_agents_config(
        exp_series_config['agents'],
        exp_series_config['general_config'],
    )

    for env_change in exp_series_config.get('env_changes', [{}]):
        for agent_for_exp in general_agents_config:
            # for agent_change in exp_series_config.get('agent_changes', [{}]):
            agent_change = {}

            launch({
                'agent_class': agent_for_exp['agent_class'],
                'hyperparameters': agent_for_exp['hyperparameters'],
                'env_config': general_env_config,
                'env_change': env_change,
                'agent_change': agent_change,
                'device': _args.device,
                'common_config': agent_for_exp['common_config'],
            })
