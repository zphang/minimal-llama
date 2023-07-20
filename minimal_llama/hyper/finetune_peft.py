import argparse
import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
import minimal_llama.utils.io_utils as io_utils
import datasets
import bitsandbytes.optim
import wandb

import minimal_llama.hyper.prefix_llama as prefix_llama
import minimal_llama.hyper.prefix_makers as prefix_makers
import minimal_llama.hyper.lora_llama as lora_llama
import minimal_llama.utils.torch_utils as torch_utils


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_path", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--peft_type", type=str, default="prefix")
    parser.add_argument("--num_prefix_tokens", type=int, default=16)
    parser.add_argument("--prefix_mode", type=str, default=prefix_llama.PREFIX_MODE_PREFIX)
    parser.add_argument("--prefix_maker_mode", type=str, default="mlp")
    parser.add_argument("--prefix_include_gates", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--total_steps", type=int, default=3000)
    parser.add_argument("--save_freq", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--model_size", type=str, default="7b")
    args = parser.parse_args()
    assert args.prefix_maker_mode in ["plain", "mlp", "hidden_states"]

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if rank == 0:
        wandb.init(
            name=args.run_name,
            project="hyper2",
            config={
                "total_steps": args.total_steps,
                "lr": args.lr,
                "full_batch_size": args.batch_size * args.grad_accum_steps,
                "peft_type": args.peft_type,
                "num_prefix_tokens": args.num_prefix_tokens,
                "prefix_mode": args.prefix_mode,
                "prefix_maker_mode": args.prefix_maker_mode,
                "prefix_include_gates": args.prefix_include_gates,
                "lora_rank": args.lora_rank,
            },
        )

    if args.peft_type == "prefix":
        config = prefix_llama.LLAMA_CONFIG_DICT[args.model_size]
        config.dtype = torch.bfloat16
        config.gradient_checkpointing = True
        model = prefix_llama.create_model(
            args.model_size,
            config=config,
            hf_path=args.hf_path,
            device=device,
            use_4bit=True,
            prefix_config=prefix_llama.PrefixConfig(prefix_mode=args.prefix_mode),
        )
        prefix_maker = prefix_makers.create_prefix_maker(
            num_tokens=args.num_prefix_tokens,
            config=config,
            prefix_type=args.prefix_maker_mode,
            include_gates=args.prefix_include_gates,
        ).to(device)
        optimizer = bitsandbytes.optim.AdamW(prefix_maker.parameters(), lr=args.lr, is_paged=True, optim_bits=32)
    elif args.peft_type == "lora":
        config = lora_llama.LLAMA_CONFIG_DICT[args.model_size]
        config.dtype = torch.bfloat16
        config.gradient_checkpointing = True
        config.lora_rank = args.lora_rank
        model = lora_llama.create_model(
            args.model_size,
            config=config,
            hf_path=args.hf_path,
            use_4bit=True,
        )
        optimizer = bitsandbytes.optim.AdamW(model.parameters(), lr=args.lr, is_paged=True, optim_bits=32)
    else:
        raise KeyError(args.peft_type)

    # Loading
    if os.path.exists(os.path.join(args.save_dir, "train_state.json")):
        train_state = io_utils.read_json(os.path.join(args.save_dir, "train_state.json"))
        completed_steps = train_state["completed_steps"]
        load_path = os.path.join(args.save_dir, f"checkpoint_{completed_steps:05d}.pt")
        print("Resuming from", load_path)
        loaded = torch.load(load_path)
        if args.peft_type == "prefix":
            prefix_maker.load_state_dict(loaded["model"])
        elif args.peft_type == "lora":
            model.load_state_dict(loaded["model"], strict=False)
        else:
            raise KeyError(args.peft_type)
        optimizer.load_state_dict(loaded["optimizer"])
        loss_list = io_utils.read_json(os.path.join(args.save_dir, f"loss_{completed_steps:05d}.json"))
    else:
        train_state = {"completed_steps": 0}
        os.makedirs(args.save_dir, exist_ok=True)
        loss_list = []

    ds = datasets.load_from_disk(args.dataset_path)
    train_iterator = get_train_iterator(
        ds,
        rank=rank, world_size=world_size,
        batch_size=args.batch_size, num_workers=args.num_workers,
        total_steps=args.total_steps,
        start_step=train_state["completed_steps"],
        seed=train_state["completed_steps"],  # use steps as stand-in for seed
        grad_accum_steps=args.grad_accum_steps,
    )
    loss = None
    optimizer.zero_grad()
    for batch_metadata, batch in train_iterator:
        if args.peft_type == "prefix":
            prefixes = prefix_maker(batch_size=batch["input_ids"].shape[0])
            output = model(
                input_ids=batch["input_ids"][:, :-1].to(device),
                prefixes=prefixes,
            )
        elif args.peft_type == "lora":
            output = model(
                input_ids=batch["input_ids"][:, :-1].to(device),
                use_pefts=True,
            )
        else:
            raise KeyError(args.peft_type)
        loss = F.cross_entropy(
            output.view(-1, output.size(-1)),
            batch["labels"][:, 1:].reshape(-1).to(device),
        )
        if rank == 0:
            wandb.log({"loss": loss.item(), "step": batch_metadata["curr_step"]})
        loss.backward()
        if batch_metadata["grad_accum_index"] == args.grad_accum_steps - 1:
            optimizer.step()
            optimizer.zero_grad()
            if local_rank == 0:
                print(batch_metadata["curr_step"], "Mem:", torch.cuda.max_memory_allocated(device), loss.item())
                loss_list.append(loss.item())
        completed_steps = batch_metadata["curr_step"] + 1
        if completed_steps % args.save_freq == 0:
            if args.peft_type == "prefix":
                model_state_dict = prefix_maker.state_dict()
            elif args.peft_type == "lora":
                model_state_dict = torch_utils.get_requires_grad(model)
                if not model_state_dict:
                    raise RuntimeError()
            else:
                raise KeyError(args.peft_type)
            torch.save({
                "model": model_state_dict,
                "optimizer": optimizer.state_dict(),
            }, os.path.join(args.save_dir, f"checkpoint_{completed_steps:05d}.pt"))
            io_utils.write_json(loss_list, os.path.join(args.save_dir, f"loss_{completed_steps:05d}.json"))
            io_utils.write_json({"completed_steps": completed_steps}, os.path.join(args.save_dir, f"train_state.json"))

    if args.peft_type == "prefix":
        model_state_dict = prefix_maker.state_dict()
    elif args.peft_type == "lora":
        model_state_dict = torch_utils.get_requires_grad(model)
    else:
        raise KeyError(args.peft_type)

    torch.save({
        "model": model_state_dict,
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
                       start_step: int = 0, grad_accum_steps: int = 1,
                       seed: int  = 0):
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
