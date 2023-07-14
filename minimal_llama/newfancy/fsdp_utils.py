import os
import functools
from pkg_resources import packaging


import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,  # general model non-sharded, non-flattened params
    LocalStateDictConfig,  # flattened params, usable only by FSDP
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
import torch.distributed.checkpoint as dist_cp
import time
from torch.distributed.checkpoint.default_planner import (
    DefaultSavePlanner,
    DefaultLoadPlanner,
)

import minimal_llama.newfancy.fsdp_policies as policies


def setup(rank, world_size):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12356'
    # print(f"rank {rank} world_size {world_size}")

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def bfloat_support():
    return (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and packaging.version.parse(torch.version.cuda).release >= (11, 0)
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    )


def get_policies(cfg, rank, layer_class):

    """establish current policies for mixed precision and fsdp wrapping"""

    mixed_precision_policy = None

    # mixed precision -----
    if cfg.mixed_precision:
        bfloat_available = bfloat_support()
        if bfloat_available and not cfg.use_fp16:
            mixed_precision_policy = policies.bfSixteen
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = policies.fpSixteen
            if rank == 0:
                print(f"FP16 enabled. ")
        else:
            # mixed_precision_policy = policies.fpSixteen
            print(
                f"bFloat16 support not present. Will use FP32, and not mixed precision"
            )

    wrapping_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={layer_class},
    )

    return mixed_precision_policy, wrapping_policy


def apply_fsdp_checkpointing(model, layer_class):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    print(f"--> applying fdsp activation checkpointing...")
    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        offload_to_cpu=False,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    check_fn = lambda submodule: isinstance(submodule, layer_class)  # noqa: E731
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )


def save_model_and_optimizer_sharded(model, rank, save_using_num_threads: int, save_dir, optim=None):
    """save model and optimizer via sharded_state_dict to save_dir"""
    if rank == 0:
        print(f"Saving model to {save_dir}")
        os.makedirs(save_dir, exist_ok=True)

    distributed_writer = dist_cp.FileSystemWriter(
        save_dir,
    )
    t0 = time.perf_counter()

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):

        state_dict = {"model": model.state_dict()}
        if optim is not None:
            state_dict["optim"] = FSDP.optim_state_dict(model, optim)

        dist_cp.save_state_dict(
            state_dict=state_dict,
            storage_writer=distributed_writer,
            planner=DefaultSavePlanner(),

        )
    dist.barrier()
    t1 = time.perf_counter()
    if rank == 0:
        print(f"Sharded state checkpoint saved to {save_dir}")
        print(
            f"Checkpoint Time = {t1 - t0:.4f}\n using {save_using_num_threads=} total threads"
        )


def save_optimizer_checkpoint(model, optimizer, rank, optimizer_save_path):
    """save optimizer state via full state dict"""
    # pull all sharded optimizer states to rank0 cpu...
    optim_state = FSDP.full_optim_state_dict(model, optimizer)
    if rank == 0:
        torch.save(optim_state, optimizer_save_path)


def load_optimizer_checkpoint(model, optimizer, rank, optimizer_load_path):
    """load an fdsp optimizer full_state checkpoint using scatter method
    this ensures only rank 0 loads the optimizer state dict and scatters to other ranks
    """
    full_osd = None
    if rank == 0:
        full_osd = torch.load(optimizer_load_path)
    # called from all ranks, though only rank0 has a valid param for full_osd
    sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, model)
    optimizer.load_state_dict(sharded_osd)
