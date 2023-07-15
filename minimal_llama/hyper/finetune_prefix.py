import argparse
import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
import minimal_llama.utils.io_utils as io_utils
import datasets
import bitsandbytes.optim

import minimal_llama.hyper.prefix_llama as prefix_llama
import minimal_llama.hyper.prefix_makers as prefix_makers


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_path", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_prefix_tokens", type=int, default=16)
    parser.add_argument("--prefix_type", type=str, default="mlp")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--total_steps", type=int, default=3000)
    parser.add_argument("--save_freq", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    config = prefix_llama.LLAMA_7B_CONFIG
    config.dtype = torch.bfloat16
    config.gradient_checkpointing = True
    model = prefix_llama.create_model(
        "7b",
        config=config,
        hf_path=args.hf_path,
        device=device,
        use_4bit=True,
    )
    prefix_maker = prefix_makers.create_prefix_maker(
        num_tokens=args.num_prefix_tokens,
        config=config,
        prefix_type=args.prefix_type,
    ).to(device)
    # optimizer = optim.AdamW(prefix_maker.parameters(), lr=args.lr)
    optimizer = bitsandbytes.optim.AdamW(prefix_maker.parameters(), lr=args.lr, is_paged=True, optim_bits=32)

    ds = datasets.load_from_disk(args.dataset_path)
    train_iterator = get_train_iterator(
        ds,
        rank=rank, world_size=world_size,
        batch_size=args.batch_size, num_workers=args.num_workers,
        total_steps=args.total_steps, start_step=0, grad_accum_steps=args.grad_accum_steps,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    loss_list = []
    loss = None
    optimizer.zero_grad()
    for batch_metadata, batch in train_iterator:
        prefixes = prefix_maker(batch_size=batch["input_ids"].shape[0])
        output = model(
            input_ids=batch["input_ids"][:, :-1].to(device),
            prefixes=prefixes,
        )
        loss = F.cross_entropy(
            output.view(-1, output.size(-1)),
            batch["labels"][:, 1:].reshape(-1).to(device),
        )
        loss.backward()
        if batch_metadata["grad_accum_index"] == args.grad_accum_steps - 1:
            optimizer.step()
            optimizer.zero_grad()
            if local_rank == 0:
                print(batch_metadata["curr_step"], "Mem:", torch.cuda.max_memory_allocated(device), loss.item())
                loss_list.append(loss.item())
        completed_steps = batch_metadata["curr_step"] + 1
        if completed_steps % args.save_freq == 0:
            torch.save({
                "model": prefix_maker.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, os.path.join(args.save_dir, f"checkpoint_{completed_steps:05d}.pt"))
            io_utils.write_json(loss_list, os.path.join(args.save_dir, f"loss_{completed_steps:05d}.json"))

    torch.save({
        "model": prefix_maker.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, os.path.join(args.save_dir, "checkpoint_final.pt"))
    io_utils.write_json(loss_list, os.path.join(args.save_dir, "loss_final.json"))

    if local_rank == 0:
        print("Mem:", torch.cuda.max_memory_allocated(device))
        print("Done", loss.item())


def data_collator(features: list) -> dict:
    keys = features[0].keys()
    batch = {
        k: torch.stack([
            torch.LongTensor(f[k])
            for f in features
        ])
        for k in keys
    }
    return batch


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
        collate_fn=data_collator,
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
