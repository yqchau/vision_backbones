import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch2trt import torch2trt
from torchvision import datasets

from models.timm_models import create_timm_models
from utils.checker import testing_sanity_check
from utils.inference import calc_acc, calc_speed
from utils.support import modify_state_dict


def test(config):
    batch_size = config["tester"]["batch_size"]
    fp16_mode = config["tester"]["fp16_mode"]

    transform = instantiate(config["tester"]["test_transform"], _convert_="partial")
    dataset = datasets.ImageFolder(
        root=config["tester"]["dataset_path"],
        transform=transform,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    arch = (
        config["models"]["architecture"]
        if config["tester"]["architecture"] == "auto"
        else config["tester"]["architecture"]
    )
    transfer_learning = True if config["tester"]["checkpoint"] == "auto" else False
    classes = (
        config["datamodule"]["classes"]
        if config["tester"]["classes"] == "auto"
        else config["tester"]["classes"]
    )
    model = create_timm_models(
        arch=arch,
        transfer_learning=transfer_learning,
        classes=classes,
        in_chans=1,
    )
    if config["tester"]["checkpoint"] != "auto":
        state_dict = modify_state_dict(config["tester"]["checkpoint"])
        model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    img_shape = list(dataset.__getitem__(0)[0].shape)
    inputs = torch.zeros(batch_size, *img_shape).cuda()
    if config["tester"]["trt_opt"]:
        model = torch2trt(
            model,
            [inputs],
            max_batch_size=batch_size,
            fp16_mode=fp16_mode,
        )

    print(f"Measuring accuracy of {arch}...")
    accuracy = calc_acc(model, dataloader)
    print(f"Accuracy: {accuracy}\n")

    speed = 0
    if config["tester"]["measure_speed"]:
        print(f"Measuring speed of {arch}...")
        speed = calc_speed(model, inputs, config["tester"]["num_samples"])
        print(f"Speed: {speed}ms")
    return accuracy, speed


def get_metrics(config):
    return test(config)


@hydra.main(version_base=None, config_path="configuration/", config_name="configs.yaml")
def main(config: DictConfig) -> None:
    # Sanity Check for Configurations
    testing_sanity_check(config)
    get_metrics(config)


if __name__ == "__main__":
    main()
