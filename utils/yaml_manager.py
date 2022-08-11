# type: ignore

import os
from typing import Dict

import yaml


def load_yaml_config(config_name: str):
    with open(os.path.join(os.getcwd(), config_name)) as file:
        config = yaml.safe_load(file)
    return config


def save_yaml_config(config_dict: Dict, saving_path: str, yaml_filename: str):
    with open(os.path.join(saving_path, yaml_filename), "w") as file:
        yaml.dump(config_dict, file, sort_keys=False)
