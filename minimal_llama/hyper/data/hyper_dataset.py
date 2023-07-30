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

HYPER_INPUT_INDICATOR = [10567, 29901, 13]
HYPER_OUTPUT_INDICATOR = [13, 10604, 29901, 13]


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


def format_hyper_inputs(example):
    return (
        HYPER_INPUT_INDICATOR
        + example["inputs"]
        + HYPER_OUTPUT_INDICATOR
        + example["targets"]
    )


def format_input_ids(example, max_downstream_length=None):
    input_ids = (
        HYPER_INPUT_INDICATOR
        + example["inputs"]
        + HYPER_OUTPUT_INDICATOR
        + example["targets"]
        + [EOS_TOKEN_ID]
    )
    if max_downstream_length is not None:
        input_ids = truncate(input_ids, side=LEFT, max_length=max_downstream_length - 1)
        input_ids = pad(input_ids, side=RIGHT, max_length=max_downstream_length - 1)
    return [BOS_TOKEN_ID] + input_ids


def format_labels(example, max_downstream_length=None):
    labels = (
        [NON_LABEL_TOKEN_ID] * (
            len(HYPER_INPUT_INDICATOR) + len(example["inputs"])
        )
        + [NON_LABEL_TOKEN_ID] * len(HYPER_OUTPUT_INDICATOR)
        + example["targets"]
        + [EOS_TOKEN_ID]
    )
    if max_downstream_length is not None:
        labels = truncate(labels, side=LEFT, max_length=max_downstream_length - 1)
        labels = pad(labels, side=RIGHT, max_length=max_downstream_length - 1,
                     pad_token_id=NON_LABEL_TOKEN_ID)
    return [NON_LABEL_TOKEN_ID] + labels


class FewshotHyperTrainIterator:
    def __init__(self,
                 rng_seed,
                 base_path,
                 num_gist_tokens: int = 16,
                 max_num_hyper_examples: int = 32,
                 max_hyper_length: int = 1024,
                 max_downstream_length: int = 384
                 ):
        self.base_path = base_path
        self.num_gist_tokens = num_gist_tokens
        self.rng = np.random.default_rng(rng_seed)
        self.max_num_hyper_examples = max_num_hyper_examples
        self.max_hyper_length = max_hyper_length
        self.max_downstream_length = max_downstream_length

        self.gist_tokens = list(range(VOCAB_SIZE, VOCAB_SIZE + self.num_gist_tokens))
        self.metadata = io_utils.read_json(os.path.join(base_path, "ds_metadata.json"))
        self.ds = datasets.load_from_disk(base_path, keep_in_memory=True)
        self.task_list = list(self.metadata.keys())
        self.task_indices = {
            task: np.arange(
                self.metadata[task]["from"],
                self.metadata[task]["to_ex"]
            )
            for task in self.task_list
        }
        self.examples_per_task_srs = pd.Series({
            task: len(self.task_indices[task])
            for task in self.task_list
        })
        self.task_weights = self.examples_per_task_srs / self.examples_per_task_srs.sum()
        self.task_weights = self.task_weights / self.task_weights.sum()
        self.task_start_index_dict = self.examples_per_task_srs.cumsum().shift().fillna(0).to_dict()

    def __iter__(self):
        return self

    def __next__(self):
        task = self.rng.choice(self.task_list, p=self.task_weights)
        size = min(len(self.task_indices[task]), self.max_num_hyper_examples + 1)
        selected_indices = self.rng.choice(
            a=self.task_indices[task],
            size=size,
            replace=False,
        )
        examples = [self.ds[int(idx)] for idx in selected_indices]
        hyper_examples, actual_example = examples[:-1], examples[-1]
        candidate_hyper_input_ids = [
            format_hyper_inputs(ex)
            for ex in hyper_examples
        ]
        ex0 = candidate_hyper_input_ids[0]
        if len(ex0) > self.max_hyper_length:
            # noinspection PyTypeChecker
            hyper_input_ids = truncate(
                format_hyper_inputs(hyper_examples[0]),
                max_length=self.max_hyper_length - self.num_gist_tokens - 1,
                side=LEFT,
            ) + self.gist_tokens
        else:
            hyper_input_ids = []
            for ex in candidate_hyper_input_ids:
                if len(hyper_input_ids) + len(ex) > self.max_hyper_length - self.num_gist_tokens - 1:
                    break
                hyper_input_ids += ex
            hyper_input_ids += self.gist_tokens
            hyper_input_ids = pad(hyper_input_ids, side=RIGHT, max_length=self.max_hyper_length - 1)
        hyper_input_ids = [BOS_TOKEN_ID] + hyper_input_ids
        assert len(hyper_input_ids) == self.max_hyper_length

        input_ids = format_input_ids(actual_example, max_downstream_length=self.max_downstream_length)
        labels = format_labels(actual_example, max_downstream_length=self.max_downstream_length)
        assert len(input_ids) == self.max_downstream_length
        assert len(labels) == self.max_downstream_length

        return {
            "hyper_input_ids": hyper_input_ids,
            "input_ids": input_ids,
            "labels": labels,
        }


class FewshotHyperTrainDataset(IterableDataset):

    def __init__(self, base_path,
                 max_num_hyper_examples: int = 32,
                 max_hyper_length: int = 1024,
                 max_downstream_length: int = 384,
                 seed_offset: int = 0):
        self.base_path = base_path
        self.max_num_hyper_examples = max_num_hyper_examples
        self.max_hyper_length = max_hyper_length
        self.max_downstream_length = max_downstream_length
        self.seed_offset = seed_offset

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
            max_num_hyper_examples=self.max_num_hyper_examples,
            max_hyper_length=self.max_hyper_length,
            max_downstream_length=self.max_downstream_length,
        )


def split_dataset(ds_list, base_path):
    curr_key = ""
    curr_data_dict = {
        "inputs": [], "targets": [],
        "task_source": [], "task_name": [], "template_type": []
    }
    for ds in ds_list:
        for row in ds:
            new_key = row["task_name"]  # TODO: figure out the key
            if curr_key and new_key != curr_key:
                save_path = os.path.join(base_path, curr_key)
                assert not os.path.exists(save_path)
                new_ds = datasets.Dataset.from_dict(curr_data_dict)
                new_ds.save_to_disk(save_path)
                curr_data_dict = {
                    "inputs": [], "targets": [],
                    "task_source": [], "task_name": [], "template_type": []
                }
            for k, v in row.items():
                curr_data_dict[k].append(v)
            curr_key = new_key

    if curr_data_dict["inputs"]:
        save_path = os.path.join(base_path, curr_key)
        new_ds = datasets.Dataset.from_dict(curr_data_dict)
        new_ds.save_to_disk(save_path)


def construct_metadata(ds_list, base_path):
    new_ds = datasets.concatenate_datasets([
        ds_list
    ])
    metadata = {}
    curr_key = ""
    start_idx = 0
    for i, row in new_ds:
        new_key = row["task_name"]  # TODO: figure out the key
        if curr_key and new_key != curr_key:
            assert not curr_key in metadata
            metadata[curr_key] = {
                "task_source": row["task_source"],
                "task_name": row["task_name"],
                "template_type": row["template_type"],
                "from": start_idx,
                "to_ex": i + 1
            }
            start_idx = i + 1
        curr_key = new_key

    if curr_key not in metadata:
        # noinspection PyUnboundLocalVariable
        metadata[curr_key] = {
            "task_source": row["task_source"],
            "task_name": row["task_name"],
            "template_type": row["template_type"],
            "from": start_idx,
            "to_ex": i + 1
        }
    new_ds.save_to_disk(base_path)
    io_utils.write_json(metadata, "ds_metadata.json")