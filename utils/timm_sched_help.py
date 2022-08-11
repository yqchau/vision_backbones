"""
timm schedulers does not have a factory function to deal with arguments that are not from argument parsers.
Issue has been raised here: https://github.com/rwightman/pytorch-image-models/issues/1168

This script provides a workaround for us to create timm schedulers with our configuration style training script.
official code reference: https://github.com/rwightman/pytorch-image-models/blob/master/timm/scheduler/scheduler_factory.py
Workaround is to call `create_scheduler_v2`

If new timm version has factory function method, we should migrate to it.
"""

from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.multistep_lr import MultiStepLRScheduler
from timm.scheduler.plateau_lr import PlateauLRScheduler
from timm.scheduler.poly_lr import PolyLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.tanh_lr import TanhLRScheduler


def scheduler_kwargs(cfg):
    """cfg/argparse to kwargs helper
    Convert scheduler args in argparse args or cfg like object to keyword args for updated create fn.
    """
    # Get all the arguments (just so that we are compatible with version 1 of create_scheduler function and train.py)
    kwargs = vars(cfg)
    return kwargs


def create_scheduler(args, optimizer):
    """To still allow argument parser style of passing in argument.

    Args:
        args (dict): argument parser variables
        optimizer (timm.optimizer): optimizers

    Returns:
        func: factory function to create timm scheduler
    """
    return create_scheduler_v2(optimizer, **scheduler_kwargs(cfg=args))


def create_scheduler_v2(
    optimizer,
    sched="cosine",
    epochs=300,
    min_lr=1e-6,
    warmup_lr=10,
    warmup_epochs=10,
    lr_k_decay=1.0,
    decay_epochs=100,
    decay_rate=0.1,
    patience_epochs=10,
    cooldown_epochs=10,
    lr_noise=None,
    lr_noise_pct=0.67,
    lr_noise_std=1,
    seed=42,
    lr_cycle_mul=1.0,
    lr_cycle_decay=0.1,
    lr_cycle_limit=1,
    **kwargs,
):

    num_epochs = epochs
    if sched is None:
        return None, num_epochs

    if lr_noise is not None:
        if isinstance(lr_noise, (list, tuple)):
            noise_range = [n * num_epochs for n in lr_noise]
            if len(noise_range) == 1:
                noise_range = noise_range[0]
        else:
            noise_range = lr_noise * num_epochs
    else:
        noise_range = None

    # On-off warm up learning rate:
    if kwargs["on_lr_warm_up"] is False:
        warmup_epochs = 0

    kwargs["noise_args"] = dict(
        noise_range_t=noise_range,
        noise_pct=lr_noise_pct,
        noise_std=lr_noise_std,
        noise_seed=seed,
    )
    kwargs["cycle_args"] = dict(
        cycle_mul=lr_cycle_mul,
        cycle_decay=lr_cycle_decay,
        cycle_limit=lr_cycle_limit,
    )

    if sched == "cosine":
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=min_lr,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_epochs,
            k_decay=lr_k_decay,
            **kwargs["cycle_args"],
            **kwargs["noise_args"],
        )
        num_epochs = lr_scheduler.get_cycle_length() + cooldown_epochs

    elif sched == "tanh":
        lr_scheduler = TanhLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=min_lr,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_epochs,
            t_in_epochs=True,
            **kwargs["cycle_args"],
            **kwargs["noise_args"],
        )
        num_epochs = lr_scheduler.get_cycle_length() + cooldown_epochs

    elif sched == "step":
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_epochs,
            decay_rate=decay_rate,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_epochs,
            **kwargs["noise_args"],
        )

    elif sched == "multistep":
        lr_scheduler = MultiStepLRScheduler(
            optimizer,
            decay_t=decay_epochs,
            decay_rate=decay_rate,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_epochs,
            **kwargs["noise_args"],
        )

    elif sched == "plateau":
        mode = "min" if "loss" in kwargs["lr_eval_metric"] else "max"
        lr_scheduler = PlateauLRScheduler(
            optimizer,
            decay_rate=decay_rate,
            patience_t=patience_epochs,
            lr_min=min_lr,
            mode=mode,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_epochs,
            cooldown_t=0,
            **kwargs["noise_args"],
        )
    elif sched == "poly":
        lr_scheduler = PolyLRScheduler(
            optimizer,
            power=decay_rate,  # overloading 'decay_rate' as polynomial power
            t_initial=num_epochs,
            lr_min=min_lr,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_epochs,
            k_decay=lr_k_decay,
            **kwargs["cycle_args"],
            **kwargs["noise_args"],
        )
        num_epochs = lr_scheduler.get_cycle_length() + cooldown_epochs

    return lr_scheduler, num_epochs
