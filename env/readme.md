# CarIntersect environment

This package contains the CarIntersect environment.
For requirements see [requirements.txt](https://github.com/MartinsonMichael/CarRacing_agents/blob/master/requirements.txt) from main part of project.

Folder structure:
 - CarRacing_env
    - car.py - car class
    - contact_listner.py - class with contact listener for pybox2d
    - cvat_loader.py - class with utility functionality for loading xml cvat markup
    - rewards.py - class with reward policy, coefficient are setted via settings file
    - utils.py - class with data loader, this class contain images and provide instruments for work with external data sources
    - environment.py - main class of this package, provide environment with gym-like interfaces
    
 - env_data - folder with background images, its cvat markup, car images ect
 - configs - folder with basic configs
 - common_envs_utils - set of files with some code relevant to env.
    - env_makers.py - some functions for creating env with specific config file
    - *_wrappers.py - files with env wrappers
    - visualizer.py - some dirty functions for work with animations
    
 - visual_player.py - runnable .py file, that can be used for running env and controlling car from keyboard
 
 
Config structure:
Config for env is .yaml files and consist of three parts:
 - reward part - coefficients for reward policy described in rewarder.py. [link to example](https://github.com/MartinsonMichael/CarRacing_agents/blob/master/env/configs/basic_REWARD_config.yaml)
 - state part - like the behavior of env: list of vector state to use for an agent, use of not picture as a state, number of bots, agents and bots tracks. [link to example](https://github.com/MartinsonMichael/CarRacing_agents/blob/master/env/configs/basic_STATE_config.yaml) 
 - path part - paths to car images, the path to the back image, and cvat file. [link to example](https://github.com/MartinsonMichael/CarRacing_agents/blob/master/env/configs/basic_PATH_config.yaml)
 
The final config is just the concatenation of these parts. (dict.update in python) 

CVAT file is used for creating a track line, track polygons, and some other.



  
This environment was made during the internship in CDS Lab [link](https://mipt.ru/english/research/labs/cds).
The project was started as part of existed lab env - [github link](https://github.com/cds-mipt/raai-summer-school-2019).