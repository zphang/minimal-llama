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
    # parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--phase", type=str)
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()

    train_tasks = io.read_json("/home/zp489/code/msft/hypertuning/assets/subsets/p3_t0_short_tasks.json")
    task_name = train_tasks[args.task_index]
    print(f"Tokenizing task {task_name}")
    tokenizer = transformers.LlamaTokenizer.from_pretrained(args.tokenizer_path)
    ds = datasets.load_dataset(
        "bigscience/P3", task_name,
        cache_dir="/home/zp489/scratch/working/2206/08_msft/p3/cache/",
        split=[args.phase],
    )[0]

    def map_fn(example):
        return {
            "inputs": tokenizer(example["inputs_pretokenized"], add_special_tokens=False)["input_ids"],
            "targets": tokenizer(example["targets_pretokenized"], add_special_tokens=False)["input_ids"],
        }

    remove = [feat for feat in ds.features if feat not in ["inputs", "targets"]]
    out_ds = ds.map(map_fn, remove_columns=remove)
    out_ds.save_to_disk(os.path.join(args.save_path, args.phase, task_name))


if __name__ == "__main__":
    main()
