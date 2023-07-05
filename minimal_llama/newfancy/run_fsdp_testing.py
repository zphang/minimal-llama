import argparse
import os
import torch
import functools
from pkg_resources import packaging
import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    LocalStateDictConfig,
    ShardedStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.distributed.tensor.parallel.fsdp import enable_2d_with_fsdp
import minimal_llama.pref.llama_simple3 as llama_simple3
import minimal_llama.newfancy.fsdp_policies as policies
from accelerate import init_empty_weights

FSDP_IS_AVAILABLE = enable_2d_with_fsdp()


def setup(rank, world_size):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12356'
    print(f"rank {rank} world_size {world_size}")

    # initialize the process group
    print(-1)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(0)


def bfloat_support():
    return (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and packaging.version.parse(torch.version.cuda).release >= (11, 0)
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    )


def get_policies(cfg, rank):

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
        transformer_layer_cls={
            llama_simple3.LLaMALayer,
        },
    )

    return mixed_precision_policy, wrapping_policy


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--model_size", type=str, default="7b")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--mixed_precision", action="store_true")
    # parser.add_argument("--save_path", type=str)
    args = parser.parse_args()

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    setup(rank=rank, world_size=world_size)
    mixed_precision_policy, auto_wrap_policy = get_policies(args, rank)
    model_config = llama_simple3.LLAMA_CONFIG_DICT[args.model_size]
    print(1)
    with init_empty_weights():
        model = llama_simple3.LLaMAModel(config=model_config)
    print(2)
    model = FSDP(model, auto_wrap_policy=auto_wrap_policy)
    for state_dict_type, state_dict_config in [
        (StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)),
        # (StateDictType.LOCAL_STATE_DICT, LocalStateDictConfig(offload_to_cpu=True)),
        (StateDictType.SHARDED_STATE_DICT, ShardedStateDictConfig(offload_to_cpu=True)),
    ]:
        with FSDP.state_dict_type(
            model, state_dict_type, state_dict_config,
        ):
            state_dict = model.state_dict()
            if rank == 0:
                # print(rank, state_dict_type, list(state_dict.keys())[:10])
                for k in list(state_dict.keys())[:10]:
                    tensor = state_dict[k]
                    if hasattr(tensor, "local_tensor"):
                        print(state_dict_type, k, f"F={tensor.shape}, L={tensor.local_tensor().shape}" , type(tensor))
                    else:
                        print(state_dict_type, k, tensor.shape, type(tensor))


    print("hi2")


if __name__ == "__main__":
    run()
