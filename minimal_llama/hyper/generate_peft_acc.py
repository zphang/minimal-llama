import argparse
import os
import tqdm.auto as tqdm
import torch
from torch.utils.data.distributed import DistributedSampler
import minimal_llama.utils.io_utils as io_utils
import datasets
import transformers

import minimal_llama.hyper.lora_llama as lora_llama
import accelerate
from accelerate.utils.operations import gather_object


def run():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hf_path", type=str)
    parser.add_argument("--load_path", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--peft_type", type=str, default="prefix")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--generation_length", type=int, default=128)
    parser.add_argument("--eval_nat_inst", action="store_true", default=False)
    parser.add_argument("--filename", type=str, default="gen_data")
    parser.add_argument("--model_size", type=str, default="7b")
    parser.add_argument("--skip_sub_model", action="store_true", default=False)
    args = parser.parse_args()
    torch.manual_seed(1)

    ddp_kwargs = accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = accelerate.Accelerator(
        mixed_precision="bf16",
        # mixed_precision="no",
        dispatch_batches=False,
        kwargs_handlers=[ddp_kwargs],
    )
    device = accelerator.device
    print0 = accelerator.on_local_main_process(print)
    torch.cuda.set_device(device)
    loaded = torch.load(args.load_path, map_location="cpu")
    tokenizer = transformers.LlamaTokenizer.from_pretrained(args.hf_path)

    if args.peft_type == "lora":
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
    else:
        raise KeyError(args.peft_type)
    if not args.skip_sub_model:
        loaded = loaded["model"]
    load_out = model.load_state_dict(loaded, strict=False)
    assert not load_out.unexpected_keys

    ds = datasets.load_from_disk(args.dataset_path)
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        pin_memory=True,
        collate_fn=data_collator,
    )
    dataloader = accelerator.prepare_data_loader(dataloader)
    os.makedirs(args.save_dir, exist_ok=True)

    eval_data = {}
    iterator = tqdm.tqdm(dataloader) if accelerator.process_index == 0 else dataloader
    for batch in iterator:
        converted_batch = convert_batch_for_generation(batch)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=converted_batch["input_ids"].cuda(),
                generation_length=args.generation_length,
                use_pefts=True,
            )

        predictions = clean_predictions(generated, tokenizer=tokenizer)
        references = clean_labels(batch["labels"], tokenizer=tokenizer)
        rows_to_gather = [
            {"task_name": batch["task_name"][i], "prediction": predictions[i], "reference": references[i]}
            for i, row in enumerate(generated)
        ]
        gathered_rows = gather_object(rows_to_gather)
        if accelerator.process_index == 0:
            for row in gathered_rows:
                task_name = row["task_name"]
                if task_name not in eval_data:
                    eval_data[task_name] = {"predictions": [], "references": []}
                eval_data[task_name]["predictions"].append(row["prediction"])
                eval_data[task_name]["references"].append(row["reference"])
        accelerator.wait_for_everyone()

    if accelerator.process_index == 0:
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
