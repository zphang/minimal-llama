import argparse
import json
import numpy as np
import random
import tqdm.auto as tqdm

import os
import datasets
import transformers
import pyutils.io as io


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--task_index", type=int, default=None)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--phase", type=str)
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()

    train_tasks = io.read_json("/home/zp489/code/msft/hypertuning/assets/subsets/p3_t0_short_tasks.json")
    if args.task_name is None:
        task_name = train_tasks[args.task_index]
    else:
        task_name = args.task_name
    print(f"Tokenizing task {task_name}")
    llama_tokenizer = transformers.LlamaTokenizer.from_pretrained(args.tokenizer_path)
    t5_tokenizer = transformers.T5TokenizerFast.from_pretrained("t5-base")

    def map_fn(example):
        out = {
            "t5_inputs": t5_tokenizer(example["inputs_pretokenized"], add_special_tokens=False)["input_ids"],
            "t5_targets": llama_tokenizer(example["targets_pretokenized"], add_special_tokens=False)["input_ids"],
            "llama_inputs": llama_tokenizer(example["inputs_pretokenized"], add_special_tokens=False)["input_ids"],
            "llama_targets": llama_tokenizer(example["targets_pretokenized"], add_special_tokens=False)["input_ids"],
        }
        if "is_correct" in example:
            out["is_correct"] = example["is_correct"]
        return out
    for phase in args.phase.split(","):
        print(f"Tokenizing phase {phase}...")
        ds = datasets.load_dataset(
            "bigscience/P3", task_name,
            cache_dir="/home/zp489/scratch/working/2206/08_msft/p3/cache/",
            split=[phase],
        )[0]
        remove = [feat for feat in ds.features if feat not in ["inputs", "targets", "is_correct"]]
        out_ds = ds.map(map_fn, remove_columns=remove)
        out_ds.save_to_disk(os.path.join(args.save_path, phase, task_name))
        print(f"Tokenizing phase {phase} DONE: {os.path.join(args.save_path, phase, task_name)}")


if __name__ == "__main__":
    main()
