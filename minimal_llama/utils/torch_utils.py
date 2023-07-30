import os
import torch
from functools import partial
import torch.optim.lr_scheduler as lr_scheduler


def get_requires_grad(model):
    requires_grad_params = [n for n, p in model.named_parameters() if p.requires_grad]
    state_dict = model.state_dict()
    return {k: state_dict[k] for k in requires_grad_params}


def _get_linear_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


if os.environ.get("CHECK_NAN"):
    def check_nan(x):
        if torch.isnan(x).any():
            import pdb
            pdb.set_trace()
else:
    # noinspection PyUnusedLocal
    def check_nan(x):
        pass
