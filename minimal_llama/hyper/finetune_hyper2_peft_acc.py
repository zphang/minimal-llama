import argparse
import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
import minimal_llama.utils.io_utils as io_utils
import wandb
import accelerate
import datasets
import datasets.distributed
import bitsandbytes
from torch.distributed.optim import ZeroRedundancyOptimizer
import minimal_llama.hyper.hyper2 as hyper2
import minimal_llama.hyper.data.hyper_dataset2 as hyper_dataset2
import minimal_llama.utils.torch_utils as torch_utils


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_path", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--dataset_type", type=str, default="pre")
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--num_gist_tokens", type=int, default=8)
    parser.add_argument("--only_reset_extra_tokens", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--total_steps", type=int, default=3000)
    parser.add_argument("--save_freq", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--model_size", type=str, default="7b")
    parser.add_argument("--lr_scheduler", action="store_true")
    parser.add_argument("--data_mode", type=str, default="v1")
    parser.add_argument("--no_wandb", action="store_true", default=False)
    args = parser.parse_args()
    torch.manual_seed(1)

    # Initialize Accelerate
    # Needed because technically some LoRA weights aren't actually used (final attn-O and MLP)
    ddp_kwargs = accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = accelerate.Accelerator(
        mixed_precision="bf16",
        # mixed_precision="no",
        dispatch_batches=False,
        gradient_accumulation_steps=args.grad_accum_steps,
        kwargs_handlers=[ddp_kwargs],
    )
    device = accelerator.device
    print0 = accelerator.on_local_main_process(print)
    use_wandb = not args.no_wandb and accelerator.is_main_process

    if use_wandb:
        wandb.init(
            name=args.run_name,
            project="hyper2",
            config={
                "total_steps": args.total_steps,
                "lr": args.lr,
                "full_batch_size": args.batch_size * args.grad_accum_steps * accelerator.num_processes,
                "lora_rank": args.lora_rank,
            },
        )

    config = hyper2.LLAMA_CONFIG_DICT[args.model_size]
    config.dtype = torch.bfloat16
    config.gradient_checkpointing = True
    config.lora_rank = args.lora_rank
    config.actual_num_gist_tokens = args.num_gist_tokens
    model = hyper2.create_model(
        "7b",
        config=config,
        hf_path=args.hf_path,
        use_4bit=True,
        device=device,
        only_reset_extra_tokens=args.only_reset_extra_tokens,
    )
    optimizer = bitsandbytes.optim.AdamW(model.parameters(), lr=args.lr, is_paged=True, optim_bits=32)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    # trainable_params = [p for p in model.parameters() if p.requires_grad]
    # for k, v in model.named_parameters():
    #     if v.requires_grad:
    #         print0(k, v.dtype)
    #         if v.dtype != torch.bfloat16:
    #             raise RuntimeError(f"WARNING: {k} is not bf16")
    # accelerator.wait_for_everyone()
    # optimizer = ZeroRedundancyOptimizer(
    #     trainable_params, lr=args.lr,
    #     optimizer_class=torch.optim.AdamW,
    # )
    save_model = model

    if args.lr_scheduler:
        scheduler = torch_utils.get_linear_schedule_with_warmup(
            optimizer,
            # num_warmup_steps=(args.total_steps * args.grad_accum_steps // 10),
            # num_training_steps=args.total_steps * args.grad_accum_steps,
            num_warmup_steps=(args.total_steps * accelerator.num_processes // 10),
            num_training_steps=args.total_steps * accelerator.num_processes,
        )
    else:
        scheduler = None

    # print(f"Rank: {accelerator.process_index}", model.model.layers[0].self_attn.q_proj.lora_a.mean())

    # Loading
    if os.path.exists(os.path.join(args.save_dir, "train_state.json")):
        train_state = io_utils.read_json(os.path.join(args.save_dir, "train_state.json"))
        completed_steps = train_state["completed_steps"]
        load_path = os.path.join(args.save_dir, f"checkpoint_{completed_steps:05d}.pt")
        print0("Resuming from", load_path)
        loaded = torch.load(load_path)
        model.load_state_dict(loaded["model"], strict=False)
        optimizer.load_state_dict(loaded["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(loaded["scheduler"])
    else:
        train_state = {"completed_steps": 0}
        os.makedirs(args.save_dir, exist_ok=True)

    if scheduler:
        # noinspection PyTypeChecker
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    else:
        # noinspection PyTypeChecker
        model, optimizer = accelerator.prepare(model, optimizer)

    if args.dataset_type == "pre":
        ds = datasets.load_from_disk(args.dataset_path)
        train_iterator = get_train_iterator(
            ds,
            rank=accelerator.process_index, world_size=accelerator.num_processes,
            batch_size=args.batch_size, num_workers=args.num_workers,
            total_steps=args.total_steps,
            start_step=train_state["completed_steps"],
            seed=train_state["completed_steps"],  # use steps as stand-in for seed
            grad_accum_steps=args.grad_accum_steps,
        )
    elif args.dataset_type == "hyper":
        ds = hyper_dataset2.FewshotHyperTrainDataset(
            args.dataset_path, seed_offset=accelerator.process_index * 1000,
            max_input_length=args.max_input_length,
            mode=args.data_mode,
        )
        train_iterator = get_hyper_train_iterator(
            ds,
            rank=accelerator.process_index,
            batch_size=args.batch_size, num_workers=args.num_workers,
            total_steps=args.total_steps,
            start_step=train_state["completed_steps"],
            grad_accum_steps=args.grad_accum_steps,
        )
    else:
        raise KeyError(args.dataset_type)
    torch.cuda.empty_cache()
    accelerator.wait_for_everyone()

    loss = None
    optimizer.zero_grad()
    for batch_metadata, batch in train_iterator:
        # print(f"Rank: {accelerator.process_index}", batch["input_ids"][0, 0:8])
        with accelerator.accumulate(model):
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )["logits"]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch["labels"].reshape(-1).to(device),
            )
            # print(f"Rank: {accelerator.process_index}, Loss: {loss}, Step: {batch_metadata['curr_step']}")
            if torch.isnan(loss):
                unwrapped_model = accelerator.unwrap_model(save_model)
                model_state_dict = torch_utils.get_requires_grad(unwrapped_model)
                torch.save(model_state_dict, "/fsx/zphang/working/2307/14_hyper/testing/checkpoint.p")
                torch.save(batch, "/fsx/zphang/working/2307/14_hyper/testing/bad_batch.p")
                raise RuntimeError(f"NaN on rank {accelerator.process_index}")
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()
        if batch_metadata["grad_accum_index"] == args.grad_accum_steps - 1:
            print0(batch_metadata["curr_step"], "Mem:", torch.cuda.max_memory_allocated(device), loss.item())
            if use_wandb:
                # global_loss = accelerator.reduce(loss, "mean")
                wandb.log({
                    "loss": loss.item(), "step": batch_metadata["curr_step"], "lr": optimizer.param_groups[0]["lr"]
                })
        completed_steps = batch_metadata["curr_step"] + 1
        if completed_steps % args.save_freq == 0:
            save_checkpoint(
                accelerator=accelerator,
                save_model=save_model,
                optimizer=optimizer,
                scheduler=scheduler,
                save_dir=args.save_dir,
                completed_steps=completed_steps,
            )

    save_checkpoint(
        accelerator=accelerator,
        save_model=save_model,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=args.save_dir,
        completed_steps=args.total_steps,
    )
    print0("Mem:", torch.cuda.max_memory_allocated(device))
    print0("Done", loss.item())


def data_collator(features: list) -> dict:
    batch = {
        "input_ids": torch.stack([
            torch.LongTensor(f["input_ids"])
            for f in features
        ]),
        "labels": torch.stack([
            torch.LongTensor(f["labels"])
            for f in features
        ]),
        "attention_mask": torch.stack([
            torch.BoolTensor(f["attention_mask"])
            for f in features
        ]),
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


def get_hyper_train_iterator(dataset,
                             rank: int,
                             batch_size: int, num_workers: int,
                             total_steps: int,
                             start_step: int = 0, grad_accum_steps: int = 1):
    total_micro_steps = total_steps * grad_accum_steps
    start_micro_step = start_step * grad_accum_steps
    curr_micro_step = start_micro_step
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=data_collator,
    )

    if rank == 0:
        print("total_micro_steps: {}, start_micro_step: {}".format(total_micro_steps + 1, start_micro_step + 1))
        print("curr_micro_step: {}".format(curr_micro_step + 1))
        print("Macro Step={}".format(curr_micro_step // grad_accum_steps + 1))
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


def save_checkpoint(accelerator, save_model, optimizer, scheduler, save_dir, completed_steps):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(save_model)
        model_state_dict = torch_utils.get_requires_grad(unwrapped_model)
        torch.save({
            "model": model_state_dict,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
        }, os.path.join(save_dir, f"checkpoint_{completed_steps:05d}.pt"))
        io_utils.write_json({
            "completed_steps": completed_steps,
        }, os.path.join(save_dir, f"train_state.json"))
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    run()
