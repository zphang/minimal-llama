import math
import datasets
import os
import numpy as np
import torch
from torch.utils.data import IterableDataset
import minimal_llama.utils.io_utils as io_utils
import pandas as pd

NON_LABEL_TOKEN_ID = -100
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
PAD_TOKEN_ID = 0
LEFT = "left"
RIGHT = "right"
VOCAB_SIZE = 32_000

DEFAULT_START_TOKEN = 32_255
DEFAULT_SEP_TOKEN = 32_254


def pad(input_ids: list, max_length: int, side: str,
        check: bool = True, pad_token_id=PAD_TOKEN_ID):
    curr_length = len(input_ids)
    if curr_length < max_length:
        padding = [pad_token_id] * (max_length - curr_length)
        if side == LEFT:
            return padding + input_ids
        elif side == RIGHT:
            return input_ids + padding
        else:
            raise KeyError(side)
    if check and curr_length > max_length:
        raise RuntimeError(max_length, len(input_ids))
    return input_ids


def truncate(input_ids: list, max_length: int, side: str):
    curr_length = len(input_ids)
    if curr_length > max_length:
        if side == LEFT:
            return input_ids[-max_length:]
        elif side == RIGHT:
            return input_ids[:max_length]
        else:
            raise KeyError(side)
    else:
        return input_ids


def format_input_ids(example, start_tokens, sep_tokens):
    return (
        start_tokens
        + example["inputs"]
        + sep_tokens
        + example["targets"]
        + [EOS_TOKEN_ID]
    )


def format_labels(example, start_tokens, sep_tokens):
    return (
        [NON_LABEL_TOKEN_ID] * len(start_tokens)
        + [NON_LABEL_TOKEN_ID] * len(example["inputs"])
        + [NON_LABEL_TOKEN_ID] * len(sep_tokens)
        # + sep_tokens
        + example["targets"]
        + [EOS_TOKEN_ID]
    )


class FewshotHyperTrainIterator:
    def __init__(self,
                 rng_seed,
                 base_path,
                 num_gist_tokens: int = 16,
                 max_num_examples: int = 64,
                 max_input_length: int = 2048 + 1,
                 max_consecutive_examples: int = 8,
                 min_consecutive_examples: int = 1,
                 ones_proportion: float = 0.5,
                 mode: str = "v1"):
        self.base_path = base_path
        self.num_gist_tokens = num_gist_tokens
        self.rng = np.random.default_rng(rng_seed)
        self.max_num_examples = max_num_examples
        self.max_input_length = max_input_length
        self.min_consecutive_examples = min_consecutive_examples
        self.max_consecutive_examples = max_consecutive_examples
        self.ones_proportion = ones_proportion

        self.gist_tokens = list(range(VOCAB_SIZE, VOCAB_SIZE + self.num_gist_tokens))
        if mode == "v1":
            self.start_tokens = [DEFAULT_START_TOKEN]
            self.sep_tokens = [DEFAULT_SEP_TOKEN]
        elif mode == "v2":
            self.start_tokens = [BOS_TOKEN_ID]
            self.sep_tokens = []
        else:
            raise KeyError(mode)

        if os.path.isfile(os.path.join(base_path)):
            # Multi Data
            self.nested_metadata, self.ds = load_multi_dataset(multi_metadata_path=base_path)
            self.task_list = []
            self.task_indices = {}
            self.task_weights = {}
            for ds_name, metadata in self.nested_metadata.items():
                task_weights_within_ds = {}
                for task, task_metadata in metadata.items():
                    full_task_name = f"{ds_name}:{task}"
                    self.task_list.append(full_task_name)
                    self.task_indices[full_task_name] = np.arange(
                        task_metadata["from"],
                        task_metadata["to_ex"]
                    )
                    clipped_example_count = min(len(self.task_indices[full_task_name]), 30_000)
                    task_weights_within_ds[full_task_name] = clipped_example_count
                task_weights_within_ds = pd.Series(task_weights_within_ds)
                task_weights_within_ds = (
                    task_weights_within_ds / task_weights_within_ds.sum() / len(self.nested_metadata)
                )
                for k, v in task_weights_within_ds.items():
                    self.task_weights[k] = v
            self.task_weights = pd.Series(self.task_weights)
            self.task_weights = self.task_weights / self.task_weights.sum()

        else:
            # Single Data
            self.metadata, self.ds = load_single_dataset(base_path=base_path)
            self.task_list = list(self.metadata.keys())
            self.task_indices = {
                task: np.arange(
                    self.metadata[task]["from"],
                    self.metadata[task]["to_ex"]
                )
                for task in self.task_list
            }
            self.examples_per_task_srs = pd.Series({
                task: min(len(self.task_indices[task]), 30_000)
                for task in self.task_list
            })
            self.task_weights = self.examples_per_task_srs / self.examples_per_task_srs.sum()
            self.task_weights = self.task_weights / self.task_weights.sum()

    def __iter__(self):
        return self

    def __next__(self):
        task = self.rng.choice(self.task_list, p=self.task_weights)
        size = min(len(self.task_indices[task]), self.max_num_examples)
        selected_indices = self.rng.choice(
            a=self.task_indices[task],
            size=size,
            replace=False,
        )
        # Very crude way of constructing partitions
        # noinspection PyTypeChecker
        partitions: list = self.rng.integers(
            self.min_consecutive_examples, self.max_consecutive_examples + 1,
            size=self.max_num_examples,
        ).tolist()
        partitions += [1] * math.floor(self.ones_proportion * self.max_consecutive_examples)
        self.rng.shuffle(partitions)

        examples = [self.ds[int(idx)] for idx in selected_indices]
        partition_idx = 0
        partition_counter = 0
        curr_partition_size = partitions[partition_idx]
        input_ids = self.gist_tokens[:]
        labels = [NON_LABEL_TOKEN_ID] * self.num_gist_tokens
        for example in examples:
            input_ids += format_input_ids(example, start_tokens=self.start_tokens, sep_tokens=self.sep_tokens)
            labels += format_labels(example, start_tokens=self.start_tokens, sep_tokens=self.sep_tokens)

            partition_counter += 1
            if partition_counter == curr_partition_size:
                partition_idx += 1
                curr_partition_size = partitions[partition_idx]
                partition_counter = 0
                input_ids += self.gist_tokens
                labels += [NON_LABEL_TOKEN_ID] * len(self.gist_tokens)

            if len(input_ids) > self.max_input_length:
                break

        if len(input_ids) < self.max_input_length:
            input_ids = pad(input_ids, max_length=self.max_input_length, side=RIGHT)
            labels = pad(labels, max_length=self.max_input_length, side=RIGHT, pad_token_id=NON_LABEL_TOKEN_ID)
        elif len(input_ids) > self.max_input_length:
            input_ids = truncate(input_ids, max_length=self.max_input_length, side=RIGHT)
            labels = truncate(labels, max_length=self.max_input_length, side=RIGHT)

        input_ids = input_ids[:-1]
        labels = labels[1:]
        attention_mask = create_multigist_attention_mask(input_ids, self.num_gist_tokens)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "task": task,
        }


class FewshotHyperTrainDataset(IterableDataset):

    def __init__(self, base_path,
                 num_gist_tokens: int = 16,
                 max_num_examples: int = 64,
                 max_input_length: int = 2048 + 1,
                 max_consecutive_examples: int = 8,
                 min_consecutive_examples: int = 1,
                 ones_proportion: float = 0.5,
                 seed_offset: int = 0,
                 mode: str = "v1"):
        self.base_path = base_path
        self.num_gist_tokens = num_gist_tokens
        self.max_num_examples = max_num_examples
        self.max_input_length = max_input_length
        self.max_consecutive_examples = max_consecutive_examples
        self.min_consecutive_examples = min_consecutive_examples
        self.ones_proportion = ones_proportion
        self.seed_offset = seed_offset
        self.mode = mode

    def __getitem__(self, index):
        raise NotImplementedError

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        rng_seed = worker_info.seed if worker_info is not None else np.random.randint(1_000_000)
        rng_seed += self.seed_offset
        print(f"Using seed {rng_seed}")
        return FewshotHyperTrainIterator(
            rng_seed=rng_seed,
            base_path=self.base_path,
            num_gist_tokens=self.num_gist_tokens,
            max_num_examples=self.max_num_examples,
            max_input_length=self.max_input_length,
            max_consecutive_examples=self.max_consecutive_examples,
            min_consecutive_examples=self.min_consecutive_examples,
            ones_proportion=self.ones_proportion,
            mode=self.mode,
        )


def create_multigist_attention_mask(input_ids, num_gist_tokens):
    if isinstance(input_ids, list):
        input_ids = np.array(input_ids)
    first_gist_token_id = VOCAB_SIZE
    input_len = len(input_ids)
    mask = torch.ones([input_len, input_len]).tril().bool()
    plain_indices = np.arange(input_len)
    for first_token_idx in plain_indices[input_ids == first_gist_token_id]:
        x0 = first_token_idx + num_gist_tokens
        y1 = first_token_idx
        mask[x0:, :y1] = False
    return mask[None]


def load_single_dataset(base_path):
    metadata = io_utils.read_json(os.path.join(base_path, "ds_metadata.json"))
    ds = datasets.load_from_disk(base_path, keep_in_memory=True)
    return metadata, ds


def load_multi_dataset(multi_metadata_path):
    multi_metadata = io_utils.read_json(multi_metadata_path)
    ds_list = []
    curr = 0
    nested_metadata = {}
    for ds_name, path in multi_metadata.items():
        metadata, ds = load_single_dataset(path)
        ds_list.append(ds)
        for k, v in metadata.items():
            v["from"] += curr
            v["to_ex"] += curr
        nested_metadata[ds_name] = metadata
        curr += len(ds)
    combined_ds = datasets.concatenate_datasets(ds_list)
    return nested_metadata, combined_ds
