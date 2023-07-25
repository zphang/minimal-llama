import argparse
import os
import tqdm.auto as tqdm
import torch
from torch.utils.data.distributed import DistributedSampler
import minimal_llama.utils.io_utils as io_utils
import datasets
import pandas as pd
import glob

import minimal_llama.hyper.hyper1 as hyper1
import torch.nn.functional as F
torch.set_grad_enabled(False)


def run():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hf_path", type=str)
    parser.add_argument("--load_path_ls", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--suffix", type=str, default="eval")
    parser.add_argument("--model_size", type=str, default="7b")
    args = parser.parse_args()

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    print(f"Local rank: {local_rank}, rank: {rank}, world size: {world_size}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    config = hyper1.LLAMA_CONFIG_DICT[args.model_size]
    config.dtype = torch.bfloat16
    config.lora_rank = args.lora_rank
    model = hyper1.create_model(
        "7b",
        config=config,
        hf_path=args.hf_path,
        use_4bit=True,
        device=device,
    )

    ds = datasets.load_from_disk(args.dataset_path)
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        pin_memory=True,
        collate_fn=data_collator,
    )

    load_path_ls = sorted(glob.glob(args.load_path_ls))
    for i, load_path in enumerate(load_path_ls):
        print("Evaluating {} ({}/{})".format(load_path, i+1, len(load_path_ls)))
        save_path = load_path.replace(".pt", "") + "__" + args.suffix + ".json"

        loaded = torch.load(load_path, map_location="cpu")
        load_out = model.load_state_dict(loaded["model"], strict=False)
        assert not load_out.unexpected_keys

        loss_ls = []
        for batch in tqdm.tqdm(dataloader):
            with torch.inference_mode():
                logits = model(
                    hyper_input_ids=batch["hyper_input_ids"].to(device),
                    input_ids=batch["input_ids"][:, :-1].to(device),
                )
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    batch["labels"][:, 1:].reshape(-1).to(device),
                ).item()
                for _ in range(batch["input_ids"].shape[0]):
                    loss_ls.append(loss)
        io_utils.write_json(
            {"loss": float(pd.Series(loss_ls).mean())},
            save_path,
        )


def data_collator(features: list) -> dict:
    keys = features[0].keys()
    batch = {
        k: torch.stack([
            torch.LongTensor(f[k])
            for f in features
        ])
        for k in keys
        if k != "task_name"
    }
    batch["task_name"] = [x["task_name"] for x in features]
    return batch


def convert_batch_for_generation(batch):
    batch_size = batch["input_ids"].shape[0]
    valid_row_list = []
    for i in range(batch["input_ids"].shape[0]):
        valid_row = batch["input_ids"][i][
            (batch["labels"][i] == -100)
            & (batch["input_ids"][i] != 0)
        ]
        valid_row_list.append(valid_row)
    max_length = max(len(x) for x in valid_row_list)
    new_input_ids = torch.zeros(batch_size, max_length).long()
    for i, valid_row in enumerate(valid_row_list):
        new_input_ids[i, :len(valid_row)] = valid_row
    return {
        "input_ids": new_input_ids,
    }


if __name__ == "__main__":
    run()
