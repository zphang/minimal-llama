import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
import minimal_llama.utils.io_utils as io_utils
import wandb
import accelerate

import minimal_llama.loralayer.lora_llama2 as lora_llama
import minimal_llama.utils.torch_utils as torch_utils
import minimal_llama.neox_data.data_utils as data_utils


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str)
    parser.add_argument("--data_prefix", type=str)
    parser.add_argument("--index_base_path", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--torch_save_dir", type=str, default=None)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--raw_lora_layers", type=str, default="attn,ffn")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--layer_mapping", type=str, default="single")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--total_steps", type=int, default=143000)
    parser.add_argument("--save_freq", type=int, default=5000)
    parser.add_argument("--checkpoint_freq", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=230826)
    parser.add_argument("--no_wandb", action="store_true", default=False)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.torch_save_dir is None:
        args.torch_save_dir = args.save_dir

    # Initialize Accelerate
    ddp_kwargs = accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = accelerate.Accelerator(
        mixed_precision="bf16",
        # mixed_precision="no",
        dispatch_batches=False,
        gradient_accumulation_steps=args.grad_accum_steps,
        kwargs_handlers=[ddp_kwargs],
        step_scheduler_with_optimizer=False,
    )
    device = accelerator.device
    print0 = accelerator.on_local_main_process(print)
    use_wandb = not args.no_wandb and accelerator.is_main_process

    # local_rank = int(os.environ.get('LOCAL_RANK', 0))
    # rank = int(os.environ.get('RANK', 0))
    # world_size = int(os.environ.get('WORLD_SIZE', 1))
    # torch.cuda.set_device(local_rank)
    # device = torch.device("cuda", local_rank)

    if use_wandb:
        wandb.init(
            name=args.run_name,
            project="loralayer",
            config={
                "total_steps": args.total_steps,
                "lr": args.lr,
                "full_batch_size": args.batch_size * args.grad_accum_steps,
                "model_size": args.model_size,
                "raw_lora_layers": args.raw_lora_layers,
                "lora_rank": args.lora_rank,
                "layer_mapping": args.layer_mapping,
            },
        )

    config = lora_llama.LLAMA_CONFIG_DICT[args.model_size]
    config.dtype = torch.bfloat16
    # config.gradient_checkpointing = True
    # config.raw_lora_layers = args.raw_lora_layers
    # config.lora_rank = args.lora_rank
    # config.raw_layer_mapping = args.layer_mapping
    # model = lora_llama.create_model(
    #     args.model_size,
    #     config=config,
    # )

    config.raw_lora_layers = args.raw_lora_layers
    config.lora_rank = args.lora_rank
    config.raw_layer_mapping = args.layer_mapping
    model = lora_llama.create_model(
        args.model_size,
        config=config,
    )

    # state_dict = model.state_dict()
    # assert state_dict["model.layers.0.self_attn.q_proj.weight"].data_ptr() == \
    #     state_dict["model.layers.10.self_attn.q_proj.weight"].data_ptr()
    lora_llama.initialize_model(model=model, device=device)
    # model.load_state_dict(torch.load("/fsx/zphang/working/2308/24_lora_layer/runs1/global_step0/test.p"))
    # print("Yes I'm loading a state dict")
    # optimizer = bitsandbytes.optim.AdamW(model.parameters(), lr=args.lr, is_paged=True, optim_bits=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    scheduler = torch_utils.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps * args.grad_accum_steps,
        num_training_steps=args.total_steps * args.grad_accum_steps,
    )

    # print0(f"Rank: {accelerator.process_index}",
    #       "lm weight mean", model.lm_head.weight.mean(), model.lm_head.weight.dtype)

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    # state_dict = accelerator.unwrap_model(model).state_dict()
    # assert state_dict["model.layers.0.self_attn.q_proj.weight"].data_ptr() == \
    #     state_dict["model.layers.10.self_attn.q_proj.weight"].data_ptr()

    # print0(f"Converted Rank: {accelerator.process_index}",
    #       "lm weight mean", model.module.lm_head.weight.mean(), model.module.lm_head.weight.dtype)

    # Loading
    if os.path.exists(os.path.join(args.save_dir, "train_state.json")):
        train_state = io_utils.read_json(os.path.join(args.save_dir, "train_state.json"))
        completed_steps = train_state["completed_steps"]
        load_path = os.path.join(args.save_dir, f"full_checkpoint")
        print0("Resuming from", completed_steps, )
        accelerator.load_state(load_path)
    else:
        train_state = {"completed_steps": 0}
        os.makedirs(args.save_dir, exist_ok=True)

    ds = data_utils.build_the_dataset(
        data_prefix=args.data_prefix,
        name="train_0",
        num_samples=args.total_steps * args.batch_size * args.grad_accum_steps * accelerator.num_processes,
        seq_length=2048,
        seed=args.seed,
        skip_warmup=True,
        index_base_path=args.index_base_path,
    )
    train_iterator = get_train_iterator(
        ds,
        rank=accelerator.process_index, world_size=accelerator.num_processes,
        batch_size=args.batch_size, num_workers=args.num_workers,
        total_steps=args.total_steps,
        start_step=train_state["completed_steps"],
        grad_accum_steps=args.grad_accum_steps,
    )
    torch.cuda.empty_cache()
    accelerator.wait_for_everyone()

    loss = None
    optimizer.zero_grad()
    for batch_metadata, batch in train_iterator:
        # print(f"Rank: {accelerator.process_index}", batch["input_ids"][0, 0:8])
        with accelerator.accumulate(model):
            output = model(
                input_ids=batch["input_ids"][:, :-1].to(device),
            )
            loss = F.cross_entropy(
                output.view(-1, output.size(-1)),
                batch["input_ids"][:, 1:].reshape(-1).to(device),
            )
            # print(f"Rank: {accelerator.process_index}, Loss: {loss}, Step: {batch_metadata['curr_step']}")
            if torch.isnan(loss):
                raise RuntimeError(f"NaN on rank {accelerator.process_index}")
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        if batch_metadata["grad_accum_index"] == args.grad_accum_steps - 1:
            print0(
                batch_metadata["curr_step"],
                "Mem:", "{:.2f}".format(torch.cuda.max_memory_allocated(device) / 10 ** 9),
                "Loss", loss.item(), loss.dtype,
                "output", output.dtype,
                "micro", batch_metadata["curr_micro_step"],
                "step", batch_metadata["curr_step"],
                "lr", optimizer.param_groups[0]["lr"],
                "shape", tuple(batch["input_ids"].shape),
            )
            if use_wandb:
                # global_loss = accelerator.reduce(loss, "mean")
                wandb.log({
                    "loss": loss.item(), "step": batch_metadata["curr_step"], "lr": optimizer.param_groups[0]["lr"]})
        completed_steps = batch_metadata["curr_step"] + 1
        if completed_steps % args.save_freq == 0:
            save_model(
                accelerator=accelerator,
                model=model,
                torch_save_dir=args.torch_save_dir,
                completed_steps=completed_steps,
            )
        if completed_steps % args.checkpoint_freq == 0:
            save_checkpoint(
                accelerator=accelerator,
                save_dir=args.save_dir,
                completed_steps=completed_steps,
            )

    save_checkpoint(
        accelerator=accelerator,
        save_dir=args.save_dir,
        completed_steps=args.total_steps,
    )
    save_model(
        accelerator=accelerator,
        model=model,
        torch_save_dir=args.torch_save_dir,
        completed_steps=args.total_steps,
    )
    print0("Mem:", torch.cuda.max_memory_allocated(device))
    print0("Done", loss.item())


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
    batch_sampler = data_utils.DistributedBatchSampler(
        sampler=torch.utils.data.SequentialSampler(dataset),
        batch_size=batch_size * world_size,
        drop_last=True,
        rank=rank,
        world_size=world_size,
    )
    batch_sampler.start_iter = start_micro_step
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    curr_micro_step = start_micro_step
    for micro_batch in data_loader:
        metadata = {
            "curr_micro_step": curr_micro_step,
            "curr_step": curr_micro_step // grad_accum_steps,
            "grad_accum_index": curr_micro_step % grad_accum_steps,
            # Use completed_steps for checkpoint naming
            "completed_steps": curr_micro_step // grad_accum_steps + 1,
        }
        yield metadata, micro_batch
        curr_micro_step += 1
        if curr_micro_step == total_micro_steps:
            return


# def save_checkpoint(accelerator, save_model, optimizer, scheduler, save_dir, torch_save_dir, completed_steps):
#     accelerator.wait_for_everyone()
#     checkpoint_path = os.path.join(save_dir, f"checkpoint_{completed_steps:05d}")
#     accelerator.save_state(checkpoint_path)
#     if save_dir != torch_save_dir:
#         io_utils.fsspec_move(
#             checkpoint_path,
#             os.path.join(torch_save_dir, f"checkpoint_{completed_steps:05d}"),
#         )
#     if accelerator.is_main_process:
#         io_utils.write_json({
#             "completed_steps": completed_steps,
#         }, os.path.join(save_dir, f"train_state.json"))
#     accelerator.wait_for_everyone()


def save_model(accelerator, model, torch_save_dir, completed_steps):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        model_state_dict = torch_utils.get_requires_grad(unwrapped_model)
        io_utils.fsspec_torch_save(
            model_state_dict,
            os.path.join(torch_save_dir, f"model_{completed_steps:05d}.pt")
        )
    accelerator.wait_for_everyone()


def save_checkpoint(accelerator, save_dir, completed_steps):
    accelerator.save_state(os.path.join(save_dir, "full_checkpoint"))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        io_utils.write_json({
            "completed_steps": completed_steps,
        }, os.path.join(save_dir, f"train_state.json"))
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    run()
