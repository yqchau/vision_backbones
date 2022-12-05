import os

import timm
import torch
from omegaconf import DictConfig


def sanity_check(cfg: DictConfig):
    """Use this function to notify of user that the configuration does not make
    sense or it is not feasible.

    Can add reminders too.
    """

    # Check Trainer:
    # If you are resuming training, you need to have the checkpoint (model weights):
    resume_training = cfg["resume_full_training"] or cfg["resume_weights_training"]
    checkpoint_path = cfg["checkpoint_path"]
    if resume_training and not os.path.exists(checkpoint_path):
        raise SystemExit(
            f"Resume training set to True but checkpoint path: {checkpoint_path} was not found"
        )

    if cfg["resume_full_training"] and cfg["resume_weights_training"]:
        raise SystemExit(
            f"""
            Please select only one option to resume either from full or weight only training.
            Current selection:\n
            (1) Resume full training: {cfg['resume_full_training']}\n
            (2) Resume with Weights only: {cfg['resume_weights_training']}
            """
        )

    # Check callback
    callback = cfg["callback"]
    allowed_metric = ["val_acc1", "val_acc5", "train_acc1", "train_acc5", "train_loss"]
    allowed_mode = ["min", "max"]
    monitor_metric = callback["monitor_metric"]
    mode = callback["mode"]

    if monitor_metric not in allowed_metric:
        raise SystemExit(
            f"[callback/monitor_metric]: {monitor_metric}. Monitor metric must be one of {allowed_metric}"
        )
    if mode not in allowed_mode:
        raise SystemExit(
            f"[callback/mode]: {mode}. Metric mode must be one of {allowed_mode}"
        )

    # Check datamodule
    datamodule = cfg["datamodule"]
    path = datamodule["path"]
    num_workers = datamodule["num_workers"]
    classes = datamodule["classes"]

    if not os.path.exists(path):
        raise SystemExit(f"[datamodule/path]: {path}. Dataset path do not exist")
    if not os.path.exists(os.path.join(path, "train")):
        raise SystemExit(
            f"[datamodule/path]: {path}. Train set does not exist in the dataset directory"
        )
    if not os.path.exists(os.path.join(path, "val")):
        raise SystemExit(
            f"[datamodule/path]: {path}. Validation set does not exist in the dataset directory"
        )
    if num_workers > os.cpu_count():
        raise SystemExit(
            f"[datamodule/num_workers]: {num_workers}. Number of cpu workers set is greater than available"
        )
    if not classes == len(os.listdir(os.path.join(path, "train"))):
        raise SystemExit(
            f"[datamodule/classes]: {classes}. Number of classes set is not the same as train dataset provided"
        )
    if not classes == len(os.listdir(os.path.join(path, "val"))):
        raise SystemExit(
            f"[datamodule/classes]: {classes}. Number of classes set is not the same as validation dataset provided"
        )

    # Check logger
    logger = cfg["logger"]
    experiment_version_tag = logger["experiment_version_tag"]
    auto_version_tag_name = logger["auto_version_tag_name"]

    if experiment_version_tag is None and (auto_version_tag_name is None):
        raise SystemExit(
            f"[logger/auto_version_tag_name]: {auto_version_tag_name}, [logger/experiment_version_tag]: {experiment_version_tag}. Set either an experiment_version_tag or an auto_version_tag_name"
        )

    # Check models
    models = cfg["models"]
    architecture = models["architecture"]

    if architecture not in timm.list_models() and architecture not in ["fcnn", "cnn"]:
        raise SystemExit(
            f"[models/architecture]: {architecture}. Model must be one of timm.list_models()"
        )

    # optimizers
    optimizers = cfg["optimizers"]
    allowed_optimizers = [
        "auto",
        "SGD",
        "Adam",
        "AdamW",
        "Nadam",
        "Radam",
        "AdamP",
        "SGDP",
        "Adadelta",
        "Adafactor",
        "RMSprop",
        "NovoGrad",
    ]
    option = optimizers["option"]
    devices = cfg["trainer"]["devices"]
    tune = cfg["trainer"]["tune"]
    max_step_count = optimizers["max_step_count"]

    if option not in allowed_optimizers:
        raise SystemExit(
            f"[optimizers/option]: {option}. Optimizer option must be one of {allowed_optimizers}"
        )
    if option == "auto" and devices > 1 and not tune:
        raise SystemExit(
            f"[optimizers/option]: {option}, [trainer/devices]: {devices}, [trainer/tune]: {tune}. Optimizer must not be auto when devices>1"
        )
    if option == "auto" and optimizers["max_step_count"] is None:
        raise SystemExit(
            f"[optimizers/max_step_count]: {max_step_count}, [optimizers/option]: {option}. max_step_count must be greater than 0 during auto optimizer search."
        )

    # schedulers
    sched = cfg["schedulers"]["sched"]
    schedulers_available = [
        "cosine",
        "tanh",
        "step",
        "multistep",
        "plateau",
        "poly",
        None,
    ]
    if sched not in schedulers_available:
        raise SystemExit(
            f"[scheduler/sched]: {sched}. Scheduler that was specified does not exist in this library.\
            Selections of schedulers are as follows: {schedulers_available}"
        )

    # trainer
    trainer = cfg["trainer"]
    precision = trainer["precision"]
    devices = trainer["devices"]
    ddp = trainer["ddp"]
    batch_size = datamodule["batch_size"]
    sync_batchnorm = trainer["sync_batchnorm"]
    n_trial_in_parallel = cfg["trainer"]["n_trial_in_parallel"]

    if precision != 32 and precision != 16:
        raise SystemExit(
            f"[trainer/precision]: {precision}. Precision of training has to be set to either 32 or 16"
        )
    if devices > torch.cuda.device_count():
        raise SystemExit(
            f"[trainer/devices]: {devices}. Number of gpu devices set is greater than available"
        )
    if ddp and devices <= 1:
        raise SystemExit(
            f"[trainer/devices]: {devices}, [trainer/ddp]: {ddp}. Devices must be >1 for ddp strategy"
        )
    if ddp and num_workers % devices != 0:
        raise SystemExit(
            f"[datamodule/num_workers]: {num_workers}, [trainer/ddp]: {ddp}, [trainer/devices]: {devices}. Number of workers must integer multiples of gpus during ddp."
        )
    if ddp and batch_size % devices != 0:
        raise SystemExit(
            f"[datamodule/batch_size]: {batch_size}, [trainer/ddp]: {ddp}, [trainer/devices]: {devices}. Batch size must integer multiples of gpus during ddp."
        )
    if not ddp and sync_batchnorm:
        raise SystemExit(
            f"[trainer/sync_batchnorm]: {sync_batchnorm}, [trainer/ddp]: {ddp}. Sync Batchnorm can only be turned on when ddp is set to True"
        )
    if not ddp and devices > 1 and not tune:
        raise SystemExit(
            f"[trainer/devices]: {devices}, [trainer/ddp]: {ddp}, [trainer/tune]: {tune}. Devices must be 1 if ddp strategy is not used"
        )

    # tuning
    if tune and ddp:
        raise SystemExit(
            f"[trainer/ddp]: {ddp}, [trainer/tune]: {tune}. ddp must be turned off during tuning"
        )

    if tune and num_workers % trainer["n_trial_in_parallel"] != 0:
        raise SystemExit(
            f"[datamodule/num_worker]: {num_workers}, [trainer/tune]: {tune}, [trainer/n_trial_in_parallel]: {n_trial_in_parallel}. Number of workers must be integer multiples of n_trial_in_parallel"
        )


def testing_sanity_check(config: DictConfig):
    # tester
    tester = config["tester"]
    dataset_path = tester["dataset_path"]
    classes = tester["classes"]

    if dataset_path is None or not os.path.exists(dataset_path):
        raise SystemExit(
            f"[tester/dataset_path]: {dataset_path}. Dataset path do not exist"
        )
    if not classes == len(os.listdir(dataset_path)):
        raise SystemExit(
            f"[tester/classes]: {classes}. Number of classes set is not the same as dataset provided"
        )
