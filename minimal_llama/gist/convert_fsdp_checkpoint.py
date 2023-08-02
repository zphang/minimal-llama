import argparse
import os
import numpy as np
import tqdm.auto as tqdm
import math
import torch
import torch.optim as optim
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
import datasets
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.tensor.parallel.fsdp import enable_2d_with_fsdp
import minimal_llama.gist.llama_simple3 as llama_simple3

import minimal_llama.utils.io_utils as io_utils
import minimal_llama.utils.torch_utils as torch_utils
from accelerate import init_empty_weights
import minimal_llama.newfancy.fsdp_utils as fsdp_utils

FSDP_IS_AVAILABLE = enable_2d_with_fsdp()


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="7b")
    parser.add_argument("--load_dir", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--expand_embedding", type=int, default=256)
    parser.add_argument("--save_optimizer", action="store_true", default=False)
    args = parser.parse_args()

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    fsdp_utils.setup(rank=rank, world_size=world_size)
    mixed_precision_policy, auto_wrap_policy = fsdp_utils.get_policies(
        args, rank, layer_class=llama_simple3.LLaMALayer)
    model_config = llama_simple3.LLAMA_CONFIG_DICT[args.model_size]
    model_config.num_gist_tokens = args.expand_embedding

    model_config.dtype = torch.bfloat16
    model = llama_simple3.LLaMAModel(config=model_config)

    model = FSDP(
        model,
        # process_group=None,  # see: tp
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
    )
    optimizer = optim.AdamW(model.parameters(), lr=1)
    fsdp_utils.load_model_and_optimizer_sharded(
        model=model,
        rank=rank,
        load_dir=args.load_dir,
        optim=optimizer,
    )
    fsdp_utils.save_model_checkpoint(
        model=model,
        rank=rank,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    run()
