import os

import hydra
import pytorch_lightning as pl
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.strategies import DDPStrategy
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch
from timm.optim.optim_factory import create_optimizer_v2

from datamodule.dataset import AlbuDataModule
from models.timm_lightning import TimmLightning
from models.timm_models import create_timm_models
from utils.checker import sanity_check
from utils.support import auto_version_tag
from utils.timm_sched_help import create_scheduler_v2


def train(config, search=False, checkpoint_dir=None):

    # Decide whether to seed run
    if config["seed"] is not None:
        pl.utilities.seed.seed_everything(seed=config["seed"], workers=True)
    else:
        print("Run is not seeded")

    # Set up the Data:
    num_workers = config["datamodule"]["num_workers"]
    if config["trainer"]["tune"]:
        num_workers = int(
            config["datamodule"]["num_workers"]
            / config["trainer"]["n_trial_in_parallel"]
        )
    if not config["trainer"]["tune"] and config["trainer"]["ddp"]:
        num_workers = int(
            config["datamodule"]["num_workers"] / config["trainer"]["devices"]
        )

    batch_size = config["datamodule"]["batch_size"]
    if config["trainer"]["ddp"]:
        batch_size = int(
            config["datamodule"]["batch_size"] / config["trainer"]["devices"]
        )
    data = AlbuDataModule(
        data_path=config["datamodule"]["path"],
        batch_size=batch_size,
        workers=num_workers,
        num_classes=config["datamodule"]["classes"],
        prob_cutmix=config["datamodule"]["probability_cutmix"],
        alpha_cutmix=config["datamodule"]["alpha_cutmix"],
        image_width=config["datamodule"]["image_width"],
        image_height=config["datamodule"]["image_height"],
        train_transform=config["datamodule"]["train_transform"],
        test_transform=config["datamodule"]["test_transform"],
        weighted_sampling=config["datamodule"]["weighted_sampling"],
        in_chans=config["models"]["in_chans"],
    )

    # Set up model
    timm_model = create_timm_models(
        arch=config["models"]["architecture"],
        classes=config["datamodule"]["classes"],
        drop_rate=config["models"]["dropout"],
        transfer_learning=config["models"]["transfer_learning"],
        in_chans=config["models"]["in_chans"],
    )

    # Set up Optimizer
    timm_optimizer = create_optimizer_v2(
        timm_model,
        opt=config["optimizers"]["option"],
        lr=config["optimizers"]["learning_rate"],
        weight_decay=config["optimizers"]["weight_decay"],
        momentum=config["optimizers"]["momentum"],
    )

    # Set up Scheduler
    timm_scheduler, config["trainer"]["num_epochs"] = create_scheduler_v2(
        timm_optimizer, epochs=config["trainer"]["num_epochs"], **config["schedulers"]
    )  # Scheduling can reconfigure number of epochs depending on your configuration.

    # "Lightning-ise" model
    model_kwargs = {
        "model": timm_model,
        "optimizer": timm_optimizer,
        "scheduler": timm_scheduler,
        "label_smoothing": config["models"]["label_smoothing"],
        "visualise_callback": config["callback"]["visualise_amount_failed_images"],
        "stop_visualise_callback_rounds": config["callback"]["do_not_visualise_rounds"],
        "datamodule_info": data,
        "output_dir": config["callback"]["output_dir"],
    }
    model = TimmLightning(**model_kwargs)

    # Set up the callbacks:
    early_stop_callback = EarlyStopping(
        monitor=config["callback"]["monitor_metric"],
        min_delta=0.0,
        patience=config["callback"]["e_stop_wait_n_epoch"],
        verbose=False,
        mode=config["callback"]["mode"],
    )

    checkpoint_callback = ModelCheckpoint(
        monitor=config["callback"]["monitor_metric"],
        filename="best-results-{epoch}-{val_acc1:.2f}",
        save_last=config["callback"]["save_last_model"],
        mode=config["callback"]["mode"],
        save_weights_only=config["callback"]["save_weights_only"],
        save_top_k=1,
    )

    lr_callback = LearningRateMonitor(logging_interval="epoch")

    tune_callback = TuneReportCallback(
        {
            "train_loss": "train_loss",
            "train_acc1": "train_acc1",
            "val_acc1": "val_acc1",
        },
        on="validation_end",
    )

    callbacks = []
    if config["callback"]["add_checkpoint_callback"]:
        callbacks.append(checkpoint_callback)
    if config["callback"]["add_early_stop_callback"]:
        callbacks.append(early_stop_callback)
    if config["trainer"]["tune"]:
        callbacks.append(tune_callback)
    if config["schedulers"]["sched"] is not None:
        callbacks.append(lr_callback)

    # Set up the loggers:
    version = (
        config["logger"]["experiment_version_tag"]
        if config["logger"]["experiment_version_tag"] is not None
        else auto_version_tag(config)
    )
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir="./" if config["trainer"]["tune"] else config["logger"]["folder_path"],
        name="./" if config["trainer"]["tune"] else config["logger"]["experiment_name"],
        version="./" if config["trainer"]["tune"] else version,
    )
    loggers = [
        tb_logger,
    ]
    if config["logger"]["add_cloud_logger"]:
        loggers.append(
            pl_loggers.WandbLogger(
                save_dir="./"
                if config["trainer"]["tune"]
                else config["logger"]["folder_path"],
                name="./"
                if config["trainer"]["tune"]
                else config["logger"]["experiment_name"],
                version="./" if config["trainer"]["tune"] else version,
            )
        )

    # Set up the trainer:
    devices = config["trainer"]["devices"] if not config["trainer"]["tune"] else 1
    strategy = (
        DDPStrategy(find_unused_parameters=False) if config["trainer"]["ddp"] else None
    )
    sync_batchnorm = True if config["trainer"]["sync_batchnorm"] is True else None
    enable_progress_bar = False if config["trainer"]["tune"] else True

    # Optimiser Search Overrides
    check_val_every_n_epoch = (
        config["trainer"]["check_val_every_n_epoch"] if not search else int(1e9)
    )

    trainer = pl.Trainer(
        devices=devices,
        strategy=strategy,
        check_val_every_n_epoch=check_val_every_n_epoch,
        logger=loggers,
        callbacks=callbacks,
        enable_progress_bar=enable_progress_bar,
        sync_batchnorm=sync_batchnorm,
        num_sanity_val_steps=0,  # if user wants to do sanity check, use fast_dev_run
        accelerator=config["trainer"]["accelerator"],
        log_every_n_steps=config["logger"]["log_every_n_steps"],
        precision=config["trainer"]["precision"],
        gradient_clip_algorithm=config["trainer"]["grad_clip_algo"],
        gradient_clip_val=config["trainer"]["grad_clip_value"],
        max_steps=config["trainer"]["max_steps"],
        max_epochs=config["trainer"]["num_epochs"],
        max_time=config["trainer"]["max_training_time"],
        fast_dev_run=config["fast_dev_run"],
        deterministic=True,
    )

    # Save full configurations
    if not config["fast_dev_run"]:
        log_directory = os.path.join(
            os.getcwd(),
            config["logger"]["folder_path"],
            config["logger"]["experiment_name"],
            version,
        )
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)

        full_log_path = os.path.join(
            log_directory,
            config["logger"]["config_filename"],
        )  # path to where we are logging everything to.

        if not isinstance(config["optimizers"]["learning_rate"], float):
            config["optimizers"]["learning_rate"] = config["optimizers"][
                "learning_rate"
            ].item()
        if not isinstance(config["optimizers"]["weight_decay"], float):
            config["optimizers"]["weight_decay"] = config["optimizers"][
                "weight_decay"
            ].item()
        if not isinstance(config["optimizers"]["momentum"], float):
            config["optimizers"]["momentum"] = config["optimizers"]["momentum"].item()
        with open(full_log_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    # Run Trainer:
    if config["resume_weights_training"]:
        model = TimmLightning.load_from_checkpoint(
            checkpoint_path=config["checkpoint_path"],
            **model_kwargs,
        )
        trainer.fit(
            model,
            data,
        )
    else:
        trainer.fit(
            model,
            data,
            ckpt_path=config["checkpoint_path"]
            if config["resume_full_training"]
            else None,
        )

    return trainer.callback_metrics


def train_with_tuning(config, checkpoint_dir=None):
    search_space = config["trainer"]["search_space"]

    # overwrite optimizers config for auto tuning
    for (param, value) in search_space.items():
        config["optimizers"][param] = instantiate(value, _convert_="partial")

    reporter = CLIReporter(
        parameter_columns=[
            "optimizers/learning_rate",
            "optimizers/weight_decay",
            "optimizers/momentum",
        ],
        metric_columns=[
            "train_loss",
            "train_acc1",
            "val_acc1",
            "training_iteration",
        ],
    )
    resources_per_trial = {
        "cpu": int(
            config["datamodule"]["num_workers"]
            / config["trainer"]["n_trial_in_parallel"]
        ),
        "gpu": config["trainer"]["devices"] / config["trainer"]["n_trial_in_parallel"],
    }
    scheduler = ASHAScheduler(
        max_t=config["trainer"]["num_epochs"],
        grace_period=config["trainer"]["grace_period"],
        reduction_factor=3,
    )
    search_alg = BayesOptSearch(
        metric=config["callback"]["monitor_metric"], mode=config["callback"]["mode"]
    )

    analysis = tune.run(
        train,
        resources_per_trial=resources_per_trial,
        metric=config["callback"]["monitor_metric"],
        mode=config["callback"]["mode"],
        config=config,
        num_samples=config["trainer"]["num_samples"],
        progress_reporter=reporter,
        local_dir=os.path.join(
            os.getcwd(),
            config["logger"]["folder_path"],
            config["logger"]["experiment_name"],
        ),
        name=config["logger"]["experiment_version_tag"]
        if config["logger"]["experiment_version_tag"] is not None
        else auto_version_tag(config),
        scheduler=scheduler if config["trainer"]["scheduler"] else None,
        search_alg=search_alg if config["trainer"]["search_alg"] else None,
    )

    print("Best hyperparameters found were: ", analysis.best_config)


def auto_optimizer_search(config):

    # Temporary variables to allow overwriting
    experiment_version_tag = config["logger"]["experiment_version_tag"]
    tune = config["trainer"]["tune"]
    devices = config["trainer"]["devices"]
    num_steps = config["trainer"]["max_steps"]
    num_epochs = config["trainer"]["num_epochs"]
    train_time = config["trainer"]["max_training_time"]
    scheduler = config["schedulers"]["sched"]
    prob_cutmix = config["datamodule"]["probability_cutmix"]
    save_weights = config["callback"]["add_checkpoint_callback"]

    # overwrite config for auto optimizer search to work (currently only works in non-ddp mode, single GPU)
    config["trainer"]["tune"] = False
    config["trainer"]["devices"] = 1
    config["trainer"]["ddp"] = False
    config["trainer"]["sync_batchnorm"] = False
    config["trainer"]["max_steps"] = config["optimizers"]["max_step_count"]
    config["trainer"]["num_epochs"] = config["optimizers"]["num_epochs"]
    config["trainer"]["max_training_time"] = config["optimizers"]["max_training_time"]
    config["schedulers"]["sched"] = None
    config["datamodule"]["probability_cutmix"] = 0.0
    config["callback"]["add_checkpoint_callback"] = False
    print("Auto Optimizer Search Mode...")
    allowed_optimizers = [
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
    print(f"Available optimizers: {allowed_optimizers}")
    results = {}
    for opt in allowed_optimizers:
        config["optimizers"]["option"] = opt
        config["logger"]["experiment_version_tag"] = opt
        metrics = train(config=config, search=True)
        results[opt] = metrics["train_loss"].item()
    print(results)
    best_opt = min(results, key=results.get)
    best_loss = results[best_opt]
    print(f"best optimizer: {best_opt}, loss: {best_loss}")

    # restore actual config
    config["logger"]["experiment_version_tag"] = experiment_version_tag
    config["trainer"]["tune"] = tune
    config["trainer"]["devices"] = devices
    config["trainer"]["max_steps"] = num_steps
    config["trainer"]["num_epochs"] = num_epochs
    config["trainer"]["max_training_time"] = train_time
    config["schedulers"]["sched"] = scheduler
    config["datamodule"]["probability_cutmix"] = prob_cutmix
    config["callback"]["add_checkpoint_callback"] = save_weights

    return best_opt


@hydra.main(
    version_base=None,
    config_path="configuration/",
    config_name="configs.yaml",
)
def main(config: DictConfig) -> None:

    # Convert config type to python dictionary
    config = OmegaConf.to_container(config, resolve=True)

    # Configuration Sanity Checks
    sanity_check(config)

    # Auto Optimizer Search
    if config["optimizers"]["option"] == "auto":
        best_opt = auto_optimizer_search(config)
        config["optimizers"]["option"] = best_opt

    # Training
    if config["trainer"]["tune"]:
        train_with_tuning(config=config)
    else:
        train(config=config)


if __name__ == "__main__":
    main()
