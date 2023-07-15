from dataclasses import dataclass, field
import transformers
import datasets
import tqdm.auto as tqdm
import os
import json

"""
Note, this version follows the HF format and doesn't offset input_ids and labels.
"""


@dataclass
class TokenizeArguments:
    dataset_path: str = field()
    save_path: str = field()
    tokenizer_path: str = field(default="openlm-research/open_llama_7b_v2")
    max_seq_len: int = field(default=512)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def main():
    args, = transformers.HfArgumentParser((
        TokenizeArguments,
    )).parse_args_into_dataclasses()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_path)
    data = read_jsonl(args.dataset_path)
    data_dict = {"input_ids": [], "labels": []}
    for example in tqdm.tqdm(data):
        x_input = tokenizer(example["input"], add_special_tokens=False).input_ids
        y_output = tokenizer(example["output"], add_special_tokens=False).input_ids

        input_ids = [tokenizer.bos_token_id] + x_input + y_output + [tokenizer.eos_token_id]
        labels = [-100] * (1 + len(x_input)) + y_output + [tokenizer.eos_token_id]
        diff = len(input_ids) - args.max_seq_len

        if diff < 0:
            input_ids += [0] * (-diff)
            labels += [-100] * (-diff)
        elif diff > 0:
            input_ids = input_ids[:args.max_seq_len]
            labels = labels[:args.max_seq_len]
        assert len(input_ids) == len(labels)
        data_dict["input_ids"].append(input_ids)
        data_dict["labels"].append(labels)
    ds = datasets.Dataset.from_dict(data_dict)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    ds.save_to_disk(args.save_path)


if __name__ == "__main__":
    main()