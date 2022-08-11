import glob
import json
import os

import torch
import yaml


def auto_version_tag(config):
    folder_path = config["logger"]["folder_path"]
    experiment_name = config["logger"]["experiment_name"]
    dir = os.path.join(folder_path, experiment_name)

    if not os.path.exists(dir):
        os.makedirs(dir)

    auto_version_tag_name = str(config["logger"]["auto_version_tag_name"])

    versions = [i for i in os.listdir(dir) if i.startswith(auto_version_tag_name)]
    versions.sort(key=lambda x: int(x.split("_")[-1]))

    if len(versions) == 0:
        version = f"{auto_version_tag_name}_0"
    else:
        new_version = int(versions[-1].split("_")[-1]) + 1
        version = f"{auto_version_tag_name}_{new_version}"
    return version


def fix_dataset_naming(path):
    """
    ImageNetV2 dataset dirs naming  : 0, 1, 2, 3, ...
    After fixing, dataset dirs will be named: 0000, 0001, 0002, 0003, ...

    """
    for path in glob.glob(f"{path}*"):
        if os.path.isdir(path):
            for subpath in glob.glob(f"{path}/*"):
                dirname = subpath.split("/")[-1]
                os.rename(
                    subpath, "/".join(subpath.split("/")[:-1]) + "/" + dirname.zfill(4)
                )


def modify_state_dict(PATH):
    state_dict = torch.load(PATH)["state_dict"]
    # create new OrderedDict that does not contain `model.`
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[6:]  # remove `model.`
        new_state_dict[name] = v

    return new_state_dict


def json_to_yaml(PATH):

    assert PATH.endswith(".json")

    json_file = open(PATH)
    json_data = json.load(json_file)
    json_file.close()

    with open(PATH.replace(".json", ".yaml"), "w") as f:
        yaml.dump(json_data, f)
