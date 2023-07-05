import argparse
import os
import tqdm.auto as tqdm
import math
import torch
import functools
from pkg_resources import packaging
import torch
import torch.optim as optim
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
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.tensor.parallel.fsdp import enable_2d_with_fsdp
import minimal_llama.pref.llama_simple3 as llama_simple3

import minimal_llama.utils.io_utils as io_utils
from accelerate import init_empty_weights
import minimal_llama.newfancy.fsdp_utils as fsdp_utils

FSDP_IS_AVAILABLE = enable_2d_with_fsdp()


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--model_size", type=str, default="7b")
    parser.add_argument("--hf_path", type=str)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--total_steps", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=8)
    # parser.add_argument("--save_path", type=str)
    args = parser.parse_args()

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    fsdp_utils.setup(rank=rank, world_size=world_size)
    mixed_precision_policy, auto_wrap_policy = fsdp_utils.get_policies(args, rank)
    model_config = llama_simple3.LLAMA_CONFIG_DICT[args.model_size]
    if not args.use_fp16:
        model_config.dtype = torch.bfloat16
    with init_empty_weights():
        model = llama_simple3.LLaMAModel(config=model_config)

    # See: https://github.com/HamidShojanazeri/examples/blob/FSDP_example/distributed/FSDP/T5_training.py
    model = FSDP(
        model,
        # process_group=None,  # see: tp
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
    )
    if args.activation_checkpointing:
        fsdp_utils.apply_fsdp_checkpointing(model)

    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        weight_map = io_utils.read_json(os.path.join(args.hf_path, "pytorch_model.bin.index.json"))["weight_map"]
        filename_list = sorted(list(set(weight_map.values())))
        for filename in tqdm.tqdm(filename_list):
            loaded = torch.load(os.path.join(args.hf_path, filename), map_location="cpu")
            model.load_state_dict(loaded, strict=False)
        # if rank == 0:
        #     for k, v in model.state_dict().items():
        #         print(k, v.dtype)

    device = torch.device(f"cuda:{local_rank}")
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    train_iterator = get_train_iterator(
        torch.zeros([1000, 1024]),
        rank=rank, world_size=world_size,
        batch_size=args.batch_size, num_workers=args.num_workers,
        total_steps=args.total_steps, start_step=0, grad_accum_steps=args.grad_accum_steps,
    )

    loss = None
    for _ in train_iterator:
        optimizer.zero_grad()
        for elem in range(args.grad_accum_steps):
            output = model(input_ids=torch.zeros([4, 1024]).long().to(device))
            loss = output.mean()
            loss.backward()
        if local_rank == 0:
            print("Mem:", torch.cuda.max_memory_allocated(device), loss.item())
        optimizer.step()

    if local_rank == 0:
        print("Mem:", torch.cuda.max_memory_allocated(device))
        print("Done", loss.item())


def get_train_iterator(dataset,
                       rank: int, world_size: int,
                       batch_size: int, num_workers: int,
                       total_steps: int,
                       start_step: int = 0, grad_accum_steps: int = 1):
    total_micro_steps = total_steps * grad_accum_steps
    start_micro_step = start_step * grad_accum_steps
    sampler = DistributedSampler(
        dataset,
        rank=rank,
        num_replicas=world_size,
        shuffle=True,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        # shuffle=False,
    )
    num_batches_per_epoch = len(dataset) // batch_size
    curr_epoch = start_micro_step // num_batches_per_epoch
    curr_micro_step = curr_epoch * num_batches_per_epoch
    epoch_ceil = math.ceil(total_micro_steps / num_batches_per_epoch)
    if rank == 0:
        print("Dataset size: {}, num_batches_per_epoch: {}".format(len(dataset), num_batches_per_epoch))
        print("total_micro_steps: {}, start_micro_step: {}".format(total_micro_steps + 1, start_micro_step + 1))
        print("curr_epoch: {}, curr_micro_step: {}, epoch_ceil: {}".format(
            curr_epoch + 1, curr_micro_step + 1, epoch_ceil + 1))
        if curr_micro_step < start_micro_step:
            print("Skipping from macro step {} to {}".format(
                curr_micro_step // grad_accum_steps + 1, start_micro_step // grad_accum_steps + 1))
    for epoch in range(curr_epoch, epoch_ceil):
        if rank == 0:
            print("Macro Step={}, Epoch={}".format(curr_micro_step // grad_accum_steps + 1, epoch + 1))
        sampler.set_epoch(epoch)
        for micro_batch in data_loader:
            metadata = {
                "curr_micro_step": curr_micro_step,
                "curr_step": curr_micro_step // grad_accum_steps,
                "grad_accum_index": curr_micro_step % grad_accum_steps,
                # Use completed_steps for checkpoint naming
                "completed_steps": curr_micro_step // grad_accum_steps + 1,
            }
            if curr_micro_step >= start_micro_step:
                yield metadata, micro_batch
            curr_micro_step += 1
            if curr_micro_step == total_micro_steps:
                return


if __name__ == "__main__":
    run()
