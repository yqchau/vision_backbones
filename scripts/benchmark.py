import argparse

import pandas as pd
import timm
import torch
from torch2trt import torch2trt
from tqdm import tqdm

from models.timm_models import create_timm_models
from utils.inference import calc_speed


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--number-trials",
        type=int,
        default=100,
        help="Number of repeated model inferences to calculate speed (Mean taken at the end)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch Size for Model"
    )
    parser.add_argument(
        "--classes",
        type=int,
        default=1000,
        help="Number of classes (final layer of model)",
    )
    parser.add_argument("--img-width", type=int, default=224, help="Image Width")
    parser.add_argument("--img-height", type=int, default=224, help="Image Width")
    parser.add_argument("--img-channels", type=int, default=3, help="Image Channels")

    parser.add_argument(
        "--get-all-models",
        action="store_true",
        default=False,
        help="Benchmark all models including models without pre-trained weights.",
    )
    parser.add_argument(
        "--check-fp16",
        action="store_true",
        default=False,
        help="Get inference for FP16 Optimised Model too",
    )
    parser.add_argument(
        "--file-location",
        type=str,
        default="./benchmark_list.csv",
        help="Benchmark File Location",
    )

    parser.add_argument(
        "--find-specific-models",
        action="store_true",
        default=False,
        help="Enable if you are looking to find specific models.",
    )

    parser.add_argument(
        "--specific-model-family",
        type=str,
        default="resnet",
        help="Give the family of model name, it will find any model with that string in its name",
    )

    parser.add_argument(
        "--off-tensorrt",
        action="store_true",
        default=False,
        help="To off TensorRT optimisation and only benchmark PyTorch Models.",
    )
    parser.add_argument(
        "--exclude-list",
        type=list,
        default=["resnest50d_1s4x24d"],
        help="list of model to exclude from benchmarking",
    )
    return parser.parse_args()


def gpu_main(opt):

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise SystemExit(
            "CUDA is not available. This benchmark script is specifically for NVIDIA GPU."
        )

    batch_size = opt.batch_size
    classes = opt.classes
    img_height = opt.img_height
    img_width = opt.img_width
    channels = opt.img_channels
    number_speed_trials = opt.number_trials
    on_fp16_flag = opt.check_fp16
    transfer_learning = (
        False if opt.get_all_models else True
    )  # if not all models then just take models with pretrained weights

    # Below this error threshold, we assume tensorrt model is correctly optimised.
    threshold = 0.1
    available_models = timm.list_models(pretrained=transfer_learning)
    all_data = []

    if opt.find_specific_models:  # look for specific models
        model_df = pd.DataFrame(available_models, columns=["models"])
        models_of_interest = model_df[
            model_df.models.str.contains(opt.specific_model_family)
        ]
        available_models = models_of_interest["models"].values.tolist()

    for model_name in tqdm(available_models):
        # check excluded model
        is_excluded = False
        for excluded in opt.exclude_list:
            if model_name.startswith(excluded):
                is_excluded = True
                break
        if is_excluded:
            continue
        (
            speed,
            trt_speed_32,
            trt_speed_16,
            model,
            model_trt,
            torch_outputs,
            trt_outputs,
            dummy_inputs,
        ) = [None] * 8

        # If model was created, ensure we destroy (memory accumulation issue)
        try:
            del dummy_inputs
            del model
            del torch_outputs
            del model_trt
            del trt_outputs
        except Exception:
            pass

        dummy_inputs = torch.ones(batch_size, channels, img_height, img_width).to(
            device
        )
        try:
            # PyTorch Model
            model = (
                create_timm_models(
                    arch=model_name,
                    classes=classes,
                    transfer_learning=False,
                )
                .eval()
                .to(device)
            )
            torch_outputs = model(dummy_inputs)
            speed = calc_speed(model, dummy_inputs, number_speed_trials)

            # TensorRT Model
            if not opt.off_tensorrt:
                try:
                    # trt optimized model
                    model_trt = torch2trt(
                        model,
                        [dummy_inputs],
                        max_batch_size=batch_size,
                        fp16_mode=False,
                    )
                    trt_outputs = model_trt(dummy_inputs)

                    assert trt_outputs.shape == torch_outputs.shape

                    # output difference < threshold, benchmark trt model
                    if torch.max(torch.abs(torch_outputs - trt_outputs)) < threshold:
                        trt_speed_32 = calc_speed(
                            model_trt, dummy_inputs, number_speed_trials
                        )

                        # get fp16 speed too.
                        if on_fp16_flag:
                            del model_trt  # just to ensure that we delete previous model
                            model_trt = torch2trt(
                                model,
                                [dummy_inputs],
                                max_batch_size=batch_size,
                                fp16_mode=True,
                            )
                            trt_outputs = model_trt(dummy_inputs)
                            assert trt_outputs.shape == torch_outputs.shape
                            trt_speed_16 = calc_speed(
                                model_trt, dummy_inputs, number_speed_trials
                            )
                except Exception:
                    print(f"Creation of TensorRT model: {model_name} is problematic.")
        except Exception:
            print(f"Creation of PyTorch model: {model_name} is problematic.")

        all_data.append([model_name, speed, trt_speed_32, trt_speed_16])
        torch.cuda.empty_cache()  # got some memory accumulation issue.

    df = pd.DataFrame(
        all_data,
        columns=["model", "torch_speed_fp32", "trt_speed_fp32", "trt_speed_fp16"],
    )
    df.to_csv(opt.file_location, index=False)


if __name__ == "__main__":

    # Future reference: Customise TensorRT optimisation (https://github.com/NVIDIA-AI-IOT/torch2trt/issues/568)
    # Future works: Can do for CPU (OpenVINO)

    opt = parse_opt()
    gpu_main(opt)

    # Put results into our generated benchmark (note that this merge only works for pre-trained models)
    # result_csv_location = (
    #     "./results/results-imagenet.csv"  # Take csv with model results on ImageNet
    # )
    # results_df = pd.read_csv(result_csv_location)

    # benchmark_csv_location = (
    #     "./results/full_speed_benchmark.csv"  # Take csv with speed benchmark
    # )
    # speed_df = pd.read_csv(benchmark_csv_location)

    # df_final = pd.merge(results_df, speed_df, on="model")

    # # Extract only necessary columns
    # cols = ["model", "torch_speed_fp32", "trt_speed_fp32", "top1", "top5"]
    # df = df_final[cols]
    # # df = df.dropna()  # No need data that has any missing values
    # df.to_csv("./results/full.csv", index=False)

    # Final Goal: Filtering according to constraint and selecting top k most promising models
