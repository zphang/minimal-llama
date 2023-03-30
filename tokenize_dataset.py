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


def read_lm_dataformat(path):
    import lm_dataformat
    reader = lm_dataformat.Reader(path)
    yield from reader.stream_data()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--data_format", type=str, default="jsonl")
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    args = parser.parse_args()

    tokenizer = transformers.LlamaTokenizer.from_pretrained(args.tokenizer_path)

    all_tokenized = []
    if args.data_format == "jsonl":
        reader = read_jsonl(args.data_path)
    elif args.data_format == "lm_dataformat":
        reader = read_lm_dataformat(args.data_path)
    else:
        raise KeyError(args.data_format)

    for elem in tqdm.tqdm(reader):
        text = elem["text"] if args.data_format == "jsonl" else elem
        all_tokenized.append(tokenizer.encode(text))
    random.shuffle(all_tokenized)

    all_tokens = [tokenizer.bos_token_id] + [
        tok
        for row in all_tokenized
        for tok in row + [tokenizer.eos_token_id, tokenizer.bos_token_id]
    ]

    truncated_tokens = all_tokens[:(len(all_tokens) // args.max_seq_length) * args.max_seq_length]
    arr = np.array(truncated_tokens).reshape(-1, args.max_seq_length)
    ds = datasets.Dataset.from_dict({"input_ids": arr})
    ds.save_to_disk(args.save_path)
    print(f"Generated {arr.shape[0]} samples.")


if __name__ == "__main__":
    main()
