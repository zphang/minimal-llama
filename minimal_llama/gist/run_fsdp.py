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
import wandb

FSDP_IS_AVAILABLE = enable_2d_with_fsdp()


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--model_size", type=str, default="7b")
    parser.add_argument("--hf_path", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--total_steps", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--expand_embedding", type=int, default=256)
    parser.add_argument("--num_gist_tokens", type=int, default=8)
    parser.add_argument("--no_wandb", action="store_true", default=False)
    parser.add_argument("--add_hf_shift", action="store_true")
    parser.add_argument("--save_freq", type=int, default=None)
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
    # model_config.num_gist_tokens = 0
    if not args.use_fp16:
        model_config.dtype = torch.bfloat16
    with init_empty_weights():
        model = llama_simple3.LLaMAModel(config=model_config)

    use_wandb = not args.no_wandb and rank == 0
    if use_wandb:
        wandb.init(
            name=args.run_name,
            project="hyper2",
            config={
                "total_steps": args.total_steps,
                "lr": args.lr,
                "full_batch_size": args.batch_size * args.grad_accum_steps * world_size,
            },
        )

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
        fsdp_utils.apply_fsdp_checkpointing(model, layer_class=llama_simple3.LLaMALayer)

    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        weight_map = io_utils.read_json(os.path.join(args.hf_path, "pytorch_model.bin.index.json"))["weight_map"]
        filename_list = sorted(list(set(weight_map.values())))
        for filename in tqdm.tqdm(filename_list):
            loaded = torch.load(os.path.join(args.hf_path, filename), map_location="cpu")
            if 'model.embed_tokens.weight' in loaded:
                embed_weight = loaded['model.embed_tokens.weight']
                indices = torch.randint(model_config.vocab_size, (args.expand_embedding,))
                new_embed_weight = torch.cat([
                    embed_weight,
                    embed_weight[indices],
                ], dim=0).contiguous()
                print(new_embed_weight.shape)
                loaded['model.embed_tokens.weight'] = new_embed_weight
            model.load_state_dict(loaded, strict=False)
        # if rank == 0:
        #     for k, v in model.state_dict().items():
        #         print(k, v.dtype)

    device = torch.device(f"cuda:{local_rank}")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch_utils.get_linear_schedule_with_warmup(
        optimizer,
        # num_warmup_steps=(args.total_steps * args.grad_accum_steps // 10),
        # num_training_steps=args.total_steps * args.grad_accum_steps,
        num_warmup_steps=(args.total_steps // 10),
        num_training_steps=args.total_steps,
    )

    ds = datasets.load_from_disk(args.dataset_path)
    ds = GistDatasetWrapper(ds, num_gist_tokens=args.num_gist_tokens, add_hf_shift=args.add_hf_shift)
    train_iterator = get_train_iterator(
        ds,
        rank=rank, world_size=world_size,
        batch_size=args.batch_size, num_workers=args.num_workers,
        total_steps=args.total_steps, start_step=0, grad_accum_steps=args.grad_accum_steps,
    )

    if local_rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)

    optimizer.zero_grad()
    loss = None
    for batch_metadata, batch in train_iterator:
        output = model(
            input_ids=batch["input_ids"].to(device),
            # input_ids=batch["input_ids"].clamp(0, 31999).to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        # loss = output.mean()
        # print(output.view(-1, output.size(-1)).shape)
        # print(batch["labels"].view(-1).to(device).shape)
        loss = F.cross_entropy(
            output.view(-1, output.size(-1)),
            batch["labels"].view(-1).to(device),
        )
        loss.backward()
        if batch_metadata["grad_accum_index"] == args.grad_accum_steps - 1:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            if use_wandb:
                wandb.log({
                    "loss": loss.item(), "step": batch_metadata["curr_step"], "lr": optimizer.param_groups[0]["lr"]
                })
            if local_rank == 0:
                print(batch_metadata["curr_step"], "Mem:", torch.cuda.max_memory_allocated(device), loss.item())

        if args.save_freq is not None and batch_metadata["completed_steps"] % args.save_freq == 0:
            fsdp_utils.save_model_checkpoint(
                model=model,
                rank=rank,
                save_path=os.path.join(args.save_dir, f"model_{batch_metadata['completed_steps']}.p"),
            )

    # fsdp_utils.save_model_and_optimizer_sharded(
    #     model, rank, save_using_num_threads=6,
    #     save_dir=args.save_dir,
    #     optim=optimizer,
    # )
    fsdp_utils.save_model_checkpoint(
        model=model,
        rank=rank,
        save_path=os.path.join(args.save_dir, f"model_{args.total_steps}.p"),
    )

    if local_rank == 0:
        print("Mem:", torch.cuda.max_memory_allocated(device))
        print("Done", loss.item())


class GistDatasetWrapper(torch.utils.data.Dataset):

    def __init__(self, ds, num_gist_tokens: int = 16, add_hf_shift: bool = False):
        self.ds = ds
        self.num_gist_tokens = num_gist_tokens
        self.add_hf_shift = add_hf_shift

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        ex = self.ds[index]
        if self.add_hf_shift:
            input_ids = ex["input_ids"][:-1]
            labels = [
                x if 0 < x < 32_000 else -100
                for x in ex["input_ids"][1:]
            ]
        else:
            input_ids = ex["input_ids"]
            labels = ex["labels"]
        return {
            "input_ids": input_ids,
            "attention_mask": create_multigist_attention_mask(
                input_ids,
                num_gist_tokens=self.num_gist_tokens,
            ),
            "labels": labels,
        }


def convert_mask_to_soft_mask(mask, dtype):
    """Convert binary mask to mask that can be added to logits.

    (i.e. 0 for attention, large negative for masked)
    """
    mask = mask.to(dtype=dtype)
    mask = (1.0 - mask) * torch.finfo(dtype).min
    return mask


def create_gist_attention_mask(gist_token_type):
    mask_list = []
    batch_size, max_seq_len = gist_token_type.shape
    for i in range(batch_size):
        ex_gist_token_type = gist_token_type[i]
        num_pre_gist_tokens = (ex_gist_token_type == 0).long().sum()
        num_gist_tokens = (ex_gist_token_type == 1).long().sum()
        ex_attention_mask = torch.ones([max_seq_len, max_seq_len]).tril()
        ex_attention_mask[num_pre_gist_tokens + num_gist_tokens:, :num_pre_gist_tokens] = 0
        mask_list.append(ex_attention_mask)
    return torch.stack(mask_list)[:, None]


def create_multigist_attention_mask(input_ids, num_gist_tokens):
    if isinstance(input_ids, list):
        input_ids = np.array(input_ids)
    first_gist_token_id = 32_000
    input_len = len(input_ids)
    mask = torch.ones([input_len, input_len]).tril().bool()
    plain_indices = np.arange(input_len)
    for first_token_idx in plain_indices[input_ids == first_gist_token_id]:
        x0 = first_token_idx + num_gist_tokens
        y1 = first_token_idx
        mask[x0:, :y1] = False
    return mask[None]


def data_collator(features: list) -> dict:
    batch = {
        "input_ids": torch.stack([torch.LongTensor(f["input_ids"]) for f in features]),
        "labels": torch.stack([torch.LongTensor(f["labels"]) for f in features]),
        "attention_mask": torch.stack([torch.BoolTensor(f["attention_mask"]) for f in features]),
    }
    return batch


def get_train_iterator(dataset,
                       rank: int, world_size: int,
                       batch_size: int, num_workers: int,
                       total_steps: int,
                       start_step: int = 0, grad_accum_steps: int = 1,
                       seed: int = 0):
    total_micro_steps = total_steps * grad_accum_steps
    start_micro_step = start_step * grad_accum_steps
    sampler = DistributedSampler(
        dataset,
        rank=rank,
        num_replicas=world_size,
        shuffle=True,
        seed=seed,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=data_collator,
        sampler=sampler,
        # shuffle=False,
    )
    num_batches_per_epoch = len(dataset) // batch_size // world_size
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
