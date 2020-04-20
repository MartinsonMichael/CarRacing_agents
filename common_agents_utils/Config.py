from typing import Dict, Union, Any, Callable


class Config(object):
    """Object to hold the config requirements for an agent/game"""

    def __init__(self):
        self.environment_make_function: Callable = None
        self.test_environment_make_function: Callable = None
        self.phi: Callable = None
        self.env_config: Dict[str, Any] = {}

        self.hyperparameters: Dict[str, Any] = {}

        self.mode: str = None
        self.seed: int = None
        self.device: str = None
        self.rainbow_gpu: int = None
        self.agent_class: str = None

        self.table_path: str = None
        self.name: str = None
        self.log_interval: int = None
        self.animation_record_frequency: int = None
        self.record_animation: bool = None

    def __repr__(self) -> str:
        s = ""
        for attr in ['mode', 'seed', 'device', 'agent_class', 'table_path', 'name', 'log_interval',
                     'animation_record_frequency', 'record_animation']:
            value = self.__getattribute__(attr)
            if value is None:
                value = 'None'
            s += attr + ' -> ' + str(value) + '\n'

        return s


def config_to_key_value(config: Config) -> Dict[str, Union[str, int, float]]:
    ans = {'experiment_name': config.name}

    def get_sub_items(dct: Dict[str, Any], prefix) -> Dict[str, Union[str, int, float]]:
        cur_ans = {}
        for name, value in dct.items():
            cur_name = prefix + ' -> ' + name if len(prefix) > 0 else name
            if isinstance(value, (int, float, str)):
                cur_ans[cur_name] = value
            elif isinstance(value, list):
                cur_ans[cur_name] = '[' + ', '.join(map(str, value)) + ' ]'
            elif isinstance(value, dict):
                cur_ans.update(get_sub_items(value, cur_name))
            else:
                raise ValueError(f'unknown config hyperparameters of type {type(value)} : {value}')
        return cur_ans

    ans.update(get_sub_items(config.hyperparameters, ""))
    return ans
