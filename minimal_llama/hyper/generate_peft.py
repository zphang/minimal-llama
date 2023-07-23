import argparse
import os
import tqdm.auto as tqdm
import torch
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
import minimal_llama.utils.io_utils as io_utils
import datasets
import transformers

import minimal_llama.hyper.prefix_llama as prefix_llama
import minimal_llama.hyper.prefix_makers as prefix_makers
import minimal_llama.hyper.lora_llama as lora_llama
import minimal_llama.utils.torch_utils as torch_utils


def run():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hf_path", type=str)
    parser.add_argument("--load_path", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--peft_type", type=str, default="prefix")
    parser.add_argument("--num_prefix_tokens", type=int, default=16)
    parser.add_argument("--prefix_mode", type=str, default=prefix_llama.PREFIX_MODE_PREFIX)
    parser.add_argument("--prefix_maker_mode", type=str, default="mlp")
    parser.add_argument("--prefix_include_gates", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--generation_length", type=int, default=128)
    parser.add_argument("--eval_nat_inst", action="store_true", default=False)
    parser.add_argument("--filename", type=str, default="gen_data")
    parser.add_argument("--model_size", type=str, default="7b")
    args = parser.parse_args()
    assert args.prefix_maker_mode in ["plain", "mlp", "hidden_states"]

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    print(f"Local rank: {local_rank}, rank: {rank}, world size: {world_size}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    loaded = torch.load(args.load_path, map_location="cpu")
    tokenizer = transformers.LlamaTokenizer.from_pretrained(args.hf_path)

    # Blah
    prefixes = None
    prefix_maker = None

    if args.peft_type == "prefix":
        config = prefix_llama.LLAMA_CONFIG_DICT[args.model_size]
        config.dtype = torch.bfloat16
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
        prefix_maker.load_state_dict(loaded["model"])
        prefixes = prefix_maker(args.batch_size)
    elif args.peft_type == "lora":
        config = lora_llama.LLAMA_CONFIG_DICT[args.model_size]
        config.dtype = torch.bfloat16
        config.lora_rank = args.lora_rank
        model = lora_llama.create_model(
            args.model_size,
            config=config,
            hf_path=args.hf_path,
            use_4bit=True,
            device=device,
        )
        load_outcome = model.load_state_dict(loaded["model"], strict=False)
        assert not load_outcome.unexpected_keys

    else:
        raise KeyError(args.peft_type)

    ds = datasets.load_from_disk(args.dataset_path)
    if world_size > 1:
        indices = list(range(rank, len(ds), world_size))
        print(f"[Rank {rank}] evaluating on", indices[:3], "...", indices[-3:])
        ds = ds.select(indices)
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        pin_memory=True,
        collate_fn=data_collator,
    )
    os.makedirs(args.save_dir, exist_ok=True)

    eval_data = {}
    iterator = tqdm.tqdm(dataloader) if rank == 0 else dataloader
    for batch in iterator:
        converted_batch = convert_batch_for_generation(batch)
        if args.peft_type == "prefix":
            if prefixes is None or prefixes[0]["key"].shape[0] != converted_batch["input_ids"].shape[0]:
                prefixes = prefix_maker(converted_batch["input_ids"].shape[0])
            with torch.inference_mode():
                out = model.generate(
                    input_ids=converted_batch["input_ids"].cuda(),
                    generation_length=args.generation_length,
                    stop_on_eos=True,
                    prefixes=prefixes,
                )
        elif args.peft_type == "lora":
            with torch.inference_mode():
                out = model.generate(
                    input_ids=converted_batch["input_ids"].cuda(),
                    generation_length=args.generation_length,
                    stop_on_eos=True,
                    use_pefts=True,
                )
        else:
            raise KeyError(args.peft_type)

        predictions = clean_predictions(out, tokenizer=tokenizer)
        references = clean_labels(batch["labels"], tokenizer=tokenizer)
        if args.eval_nat_inst:
            for i, row in enumerate(out):
                task_name = batch["task_name"][i]
                if task_name not in eval_data:
                    eval_data[task_name] = {"predictions": [], "references": []}
                eval_data[task_name]["predictions"] += predictions
                eval_data[task_name]["references"] += references
        else:
            if "predictions" not in eval_data:
                eval_data["predictions"] = []
            eval_data["predictions"] += predictions

    if world_size > 1:
        io_utils.write_json(
            eval_data,
            os.path.join(args.save_dir, f"{args.filename}_shard_{rank:03d}.json"),
        )
    else:
        io_utils.write_json(
            eval_data,
            os.path.join(args.save_dir, f"{args.filename}.json"),
        )
        if args.eval_nat_inst:
            metric = datasets.load_metric("rouge")
            all_results = []
            for task_name, task_data in eval_data.items():
                raw_scores = metric.compute(
                    predictions=task_data["predictions"],
                    references=task_data["references"],
                )
                granular_scores = {
                    "rouge1-precision": float(raw_scores["rouge1"].mid.precision),
                    "rouge1-recall": float(raw_scores["rouge1"].mid.recall),
                    "rouge1-fmeasure": float(raw_scores["rouge1"].mid.fmeasure),
                    "rouge2-precision": float(raw_scores["rouge2"].mid.precision),
                    "rouge2-recall": float(raw_scores["rouge2"].mid.recall),
                    "rouge2-fmeasure": float(raw_scores["rouge2"].mid.fmeasure),
                    "rougeL-precision": float(raw_scores["rougeL"].mid.precision),
                    "rougeL-recall": float(raw_scores["rougeL"].mid.recall),
                    "rougeL-fmeasure": float(raw_scores["rougeL"].mid.fmeasure),
                    "rougeLsum-precision": float(raw_scores["rougeLsum"].mid.precision),
                    "rougeLsum-recall": float(raw_scores["rougeLsum"].mid.recall),
                    "rougeLsum-fmeasure": float(raw_scores["rougeLsum"].mid.fmeasure),
                }
                scalar_score = granular_scores["rougeL-fmeasure"]
                all_results.append({
                    "task_name": task_name,
                    "score": scalar_score,
                    "metrics": granular_scores,
                })
            io_utils.write_json(
                all_results,
                os.path.join(args.save_dir, "nat_inst_eval_results.json"),
            )


def clean_predictions(model_out, tokenizer):
    rows = []
    for i, row in enumerate(model_out):
        row = row.tolist()
        if tokenizer.eos_token_id in row:
            row = row[:row.index(tokenizer.eos_token_id)]
        rows.append(tokenizer.decode(row))
    return rows


def clean_labels(labels, tokenizer):
    rows = []
    for i in range(labels.shape[0]):
        label = [x for x in labels[i].tolist() if x != -100]
        if tokenizer.eos_token_id in label:
            label = label[:label.index(tokenizer.eos_token_id)]
        rows.append(tokenizer.decode(label))
    return rows


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
