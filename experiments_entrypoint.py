import argparse
import collections
import os
import numpy as np
from typing import Any, Dict, List

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


def launch(exp_config: Dict[str, Any]) -> None:

    # for key, value in exp_config.items():
    #     print(key)
    #     print(value)
    #     print()

    final_agent_config = make_single_config(exp_config['common_config'])
    changed_env_config = deep_dict_update(exp_config['env_config'], exp_config['env_change'])
    launch_note = create_single_launch_name(changed_env_config, exp_config['agent_class_name'])
    launch_id = exp_config['exp_series_name'] + '__' + exp_config['agent_class_name'] + '__' + str(np.random.randint(0, 10**6))
    final_agent_config.name = launch_id

    final_agent_config.record_animation = True
    final_agent_config.seed = np.random.randint(0, 2**16)
    final_agent_config.device = exp_config['device']
    # setup gpu params for rainbow, -1 for cpu, and cuda card number for gpu
    final_agent_config.rainbow_gpu = -1 if exp_config['device'] == 'cpu' else int(exp_config['device'].split(':')[1])

    final_agent_config.table_path = os.path.join('exp_tables', exp_config['exp_series_name'])
    if not os.path.exists(final_agent_config.table_path):
        os.makedirs(final_agent_config.table_path)

    final_agent_config.agent_class = exp_config['agent_class_name']
    final_agent_config.env_config = changed_env_config
    final_agent_config.mode = get_state_type_from_settings(changed_env_config)
    final_agent_config.hyperparameters = \
        deep_dict_update(exp_config['hyperparameters'], exp_config['agent_change'])
    if exp_config['agent_class_name'] == 'rainbow':
        final_agent_config.environment_make_function, final_agent_config.phi = \
            get_EnvCreator_with_memory_safe_combiner(changed_env_config, DiscreteWrapper)
    else:
        final_agent_config.environment_make_function, final_agent_config.phi = \
            get_EnvCreator_with_memory_safe_combiner(changed_env_config)
    final_agent_config.test_environment_make_function = final_agent_config.environment_make_function

    print('final config:')
    print(final_agent_config)

    wandb.init(
        project='EXP',
        reinit=True,
        name=launch_id,
        notes=launch_note,
        config={
            'exp_name': exp_config['exp_series_name'],
            'agent_class': final_agent_config.agent_class,
            'mode': final_agent_config.mode,
            'env_config': final_agent_config.env_config,
            'hyperparameters': final_agent_config.hyperparameters,
        },
    )

    print('start to create agent..')
    agent = exp_config['agent_class'](final_agent_config)

    print('start to train agent...')
    agent.train()


def _exp_worker_function(exp_list: List[Dict], device: str, save_launch: bool) -> None:
    for exp_config in exp_list:
        exp_config['device'] = device

        if save_launch:
            try:
                launch(exp_config)
            except:
                print(f"EXPERIMENT WORKER {device} : FAIL DURING EXPERIMENT")
        else:
            launch(exp_config)

    print(f"EXPERIMENT WORKER {device} : DONE.")


def multi_process_launch(device_list, launch_list) -> None:
    num_exp_per_device = (len(launch_list) + len(device_list) - 1) // len(device_list)
    print(f"num_exp_per_device : {num_exp_per_device}")

    process_list = []
    for index, device in enumerate(device_list):
        print(f"device : {device} will deal with indexes "
              f"{index * num_exp_per_device} - {(index + 1) * num_exp_per_device}")
        if index * num_exp_per_device + 1 >= len(launch_list):
            break
        process_list.append(Process(
            target=_exp_worker_function,
            args=(
                launch_list[index * num_exp_per_device: (index + 1) * num_exp_per_device],
                device,
                not _args.no_save_launch,
            )
        ))
        process_list[-1].start()

    print(f"Wait for {process_list} process...")
    for pr in process_list:
        try:
            pr.join()
        except:
            pass


def main(_args):
    exp_series_config = yaml.load(open(_args.exp_settings, 'r'))

    general_env_config = make_common_env_config(exp_series_config['env'])
    general_agents_config = make_general_agents_config(
        exp_series_config['agents'],
        exp_series_config['general_config'],
    )

    launch_list = []
    for _ in range(exp_series_config.get('repeat_num', 1)):
        for env_change in exp_series_config.get('env_changes', [{}]):
            for agent_for_exp in general_agents_config:
                for agent_change in exp_series_config.get('agent_changes', {}).\
                        get(agent_for_exp['agent_class_name'], [{}]):

                    launch_list.append({
                        'exp_series_name': _args.name,
                        'agent_class_name': agent_for_exp['agent_class_name'],
                        'agent_class': agent_for_exp['agent_class'],
                        'hyperparameters': agent_for_exp['hyperparameters'],
                        'env_config': general_env_config,
                        'env_change': env_change,
                        'agent_change': agent_change,
                        'device': _args.device,
                        'common_config': agent_for_exp['common_config'],
                    })

    if _args.multi_launch:
        device_list = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
        multi_process_launch(device_list, launch_list)
        return
    else:
        _exp_worker_function(launch_list, _args.device, not _args.no_save_launch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-record-animation', default=False, action='store_true', help='use icm')
    parser.add_argument('--name', type=str, default=None, help='name for experiment')
    parser.add_argument('--exp-settings', type=str, help='path to experiment config')
    parser.add_argument('--device', type=str, default=None, help='name for experiment')
    parser.add_argument('--no-save-launch', default=False, action='store_true', help='use or not save launch')
    parser.add_argument('--multi-launch', default=False, action='store_true')
    _args = parser.parse_args()
    _args.record_animation = not _args.no_record_animation

    if _args.name is None:
        raise ValueError('set name')

    main(_args)
