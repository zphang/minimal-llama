from dataclasses import dataclass, field
import transformers
import datasets
import tqdm.auto as tqdm
import os
import math


@dataclass
class TokenizeArguments:
    dataset_path: str = field()
    save_fol: str = field()
    rank: int = field()
    world_size: int = field()
    tokenizer_path: str = field(default="meta-llama/Llama-2-7b-hf")


def main():
    args, = transformers.HfArgumentParser((
        TokenizeArguments,
    )).parse_args_into_dataclasses()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_path)
    ds = datasets.load_from_disk(args.dataset_path)

    data_dict = {"inputs": [], "targets": [], "task_source": [], "task_name": [], "template_type": []}
    chunk_size = math.ceil(len(ds) / args.world_size)
    indices = list(range(args.rank * chunk_size, min((args.rank + 1) * chunk_size, len(ds))))
    print("Rank:", args.rank)
    print("Indices:", indices[:3], "...", indices[-3:])

    for idx in indices:
        if idx % 1000 == 0:
            print(idx)
        example = ds[idx]
        data_dict["inputs"].append(tokenizer(example["inputs"], add_special_tokens=False).input_ids)
        data_dict["targets"].append(tokenizer(example["targets"], add_special_tokens=False).input_ids)
        data_dict["task_source"].append(example["task_source"])
        data_dict["task_name"].append(example["task_name"])
        data_dict["template_type"].append(example["template_type"])

    ds = datasets.Dataset.from_dict(data_dict)
    os.makedirs(args.save_fol, exist_ok=True)
    ds.save_to_disk(os.path.join(args.save_fol, f"part-{args.rank:04d}-of-{args.world_size:04d}"))


if __name__ == "__main__":
    main()
