from typing import Dict, Union, Any


class Config(object):
    """Object to hold the config requirements for an agent/game"""

    def __init__(self):
        self.environment_make_function = None
        self.test_environment_make_function = None
        self.name: str = ""
        self.hyperparameters: Dict[str, Any] = {}
        self.agent_class: str = ""
        self.tf_writer = None
        self.debug: bool = False


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
