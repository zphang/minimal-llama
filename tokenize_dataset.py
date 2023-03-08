import argparse
import json
import numpy as np
import random
import tqdm.auto as tqdm

import datasets
import transformers


def read_jsonl(path):
    # Manually open because .splitlines is different from iterating over lines
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--jsonl_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    args = parser.parse_args()

    tokenizer = transformers.LLaMATokenizer.from_pretrained(args.tokenizer_path)

    all_tokenized = []
    for elem in tqdm.tqdm(read_jsonl(args.jsonl_path)):
        all_tokenized.append(tokenizer.encode(elem["text"]))
    random.shuffle(all_tokenized)

    all_tokens = [1] + [
        tok
        for row in all_tokenized
        for tok in row + [tokenizer.eos_token_id, tokenizer.bos_token_id]
    ]

    truncated_tokens = all_tokens[:(len(all_tokens) // args.max_seq_length) * args.max_seq_length]
    arr = np.array(truncated_tokens)
    ds = datasets.Dataset.from_dict({"input_ids": arr.reshape(-1, args.max_seq_length)})
    ds.save_to_disk(args.save_path)


if __name__ == "__main__":
    main()
