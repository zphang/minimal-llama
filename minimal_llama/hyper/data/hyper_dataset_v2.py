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

# hard-coded for now
# HYPER_INPUT_INDICATOR = [32255]
# HYPER_OUTPUT_INDICATOR = [32254]
NEWLINE = [13]
POST_DEFINITION_INDICATOR = NEWLINE
INPUT_INDICATOR = [10567, 29901, 13]
OUTPUT_INDICATOR = [13, 10604, 29901, 13]
EXAMPLE_SEPARATOR = NEWLINE * 2


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
        INPUT_INDICATOR
        # + example["inputs"]
        + example["input"]
        + OUTPUT_INDICATOR
        # + example["targets"]
        + example["output"]
    )


def format_input_ids(example, max_downstream_length=None):
    input_ids = (
        INPUT_INDICATOR
        # + example["inputs"]
        + example["input"]
        + OUTPUT_INDICATOR
        + example["output"]
        + [EOS_TOKEN_ID]
    )
    if max_downstream_length is not None:
        input_ids = truncate(input_ids, side=LEFT, max_length=max_downstream_length - 1)
        input_ids = pad(input_ids, side=RIGHT, max_length=max_downstream_length - 1)
    return [BOS_TOKEN_ID] + input_ids


def format_labels(example, max_downstream_length=None):
    labels = (
        [NON_LABEL_TOKEN_ID] * len(INPUT_INDICATOR)
        # + [NON_LABEL_TOKEN_ID] * len(example["inputs"])
        + [NON_LABEL_TOKEN_ID] * len(example["input"])
        + [NON_LABEL_TOKEN_ID] * len(OUTPUT_INDICATOR)
        # + example["targets"]
        + example["output"]
        + [EOS_TOKEN_ID]
    )
    if max_downstream_length is not None:
        labels = truncate(labels, side=LEFT, max_length=max_downstream_length - 1)
        labels = pad(labels, side=RIGHT, max_length=max_downstream_length - 1,
                     pad_token_id=NON_LABEL_TOKEN_ID)
    return [NON_LABEL_TOKEN_ID] + labels


class NatInstHyperTrainIterator:
    def __init__(self,
                 rng_seed,
                 base_path,
                 num_gist_tokens: int = 16,
                 max_num_hyper_examples: int = 32,
                 num_downstream_examples: int = 1,
                 max_hyper_length: int = 1024,
                 max_downstream_length: int = 384,
                 add_definition: bool = True,
                 hyper_example_mode: str = "sample",
                 hyper_example_copy_factor: float = 1.0,
                 ):
        self.base_path = base_path
        self.num_gist_tokens = num_gist_tokens
        self.rng = np.random.default_rng(rng_seed)
        self.max_num_hyper_examples = max_num_hyper_examples
        self.num_downstream_examples = num_downstream_examples
        self.max_hyper_length = max_hyper_length
        self.max_downstream_length = max_downstream_length
        self.add_definition = add_definition
        self.hyper_example_mode = hyper_example_mode
        self.hyper_example_copy_factor = hyper_example_copy_factor

        self.gist_tokens = list(range(VOCAB_SIZE, VOCAB_SIZE + self.num_gist_tokens))
        self.metadata = torch.load(os.path.join(base_path, "metadata.p"))
        self.ds = datasets.load_from_disk(os.path.join(base_path, "examples.ds"))
        self.task_list = list(self.metadata.keys())
        self.task_indices = {
            task: np.arange(
                self.metadata[task]["start_idx"],
                self.metadata[task]["end_postidx"]
            )
            for task in self.task_list
        }
        self.examples_per_task_srs = pd.Series({
            task: len(self.task_indices[task])
            for task in self.task_list
        })
        self.task_weights = self.examples_per_task_srs / self.examples_per_task_srs.sum()
        self.task_weights = self.task_weights / self.task_weights.sum()

    def __iter__(self):
        return self

    def __next__(self):
        task = self.rng.choice(self.task_list, p=self.task_weights)
        if self.hyper_example_mode == "sample":
            size = self.max_num_hyper_examples + self.num_downstream_examples
            replace = len(self.task_indices[task]) < size
            selected_indices = self.rng.choice(
                a=self.task_indices[task],
                size=size,
                replace=replace,
            )
            examples = [self.ds[int(idx)] for idx in selected_indices]
            hyper_examples = examples[:self.max_num_hyper_examples]
            actual_examples = examples[self.max_num_hyper_examples:]
        elif self.hyper_example_mode == "pos":
            pos_examples = self.metadata[task]["pos_ex"]
            num_hyper_examples = min(len(pos_examples), self.max_num_hyper_examples)
            hyper_indices = self.rng.choice(
                a=len(pos_examples),
                size=num_hyper_examples,
                replace=False,
            )
            hyper_examples = [pos_examples[idx] for idx in hyper_indices]
            size = self.num_downstream_examples
            replace = len(self.task_indices[task]) < size
            selected_indices = self.rng.choice(
                a=self.task_indices[task],
                size=size,
                replace=replace,
            )
            actual_examples = [self.ds[int(idx)] for idx in selected_indices]
        elif self.hyper_example_mode == "copy":
            # assert self.num_downstream_examples == self.max_num_hyper_examples
            # size = self.max_num_hyper_examples + self.num_downstream_examples
            # replace = len(self.task_indices[task]) < size
            # selected_indices = self.rng.choice(
            #     a=self.task_indices[task],
            #     size=size,
            #     replace=replace,
            # )
            # hyper_indices = selected_indices[:self.max_num_hyper_examples]
            # actual_indices = selected_indices[self.max_num_hyper_examples:]
            # do_copy = self.rng.binomial(n=self.max_num_hyper_examples, p=self.hyper_example_copy_factor)
            # hyper_indices = [
            #     hyper_indices[idx] if do_copy else actual_indices[idx]
            #     for idx in range(self.max_num_hyper_examples)
            # ]
            size = self.max_num_hyper_examples + self.num_downstream_examples
            replace = len(self.task_indices[task]) < size
            selected_indices = self.rng.choice(
                a=self.task_indices[task],
                size=size,
                replace=replace,
            )
            hyper_indices = selected_indices[:self.max_num_hyper_examples]
            actual_indices = selected_indices[self.max_num_hyper_examples:]
            num_copy = self.rng.binomial(n=self.num_downstream_examples, p=self.hyper_example_copy_factor)
            hyper_indices[:num_copy] = actual_indices[:num_copy]
            self.rng.shuffle(hyper_indices)

            hyper_examples = [self.ds[int(idx)] for idx in hyper_indices]
            actual_examples = [self.ds[int(idx)] for idx in actual_indices]
        else:
            raise KeyError(self.hyper_example_mode)

        candidate_hyper_input_ids = [
            format_hyper_inputs(ex)
            for ex in hyper_examples
        ]
        if len(candidate_hyper_input_ids[0]) > self.max_hyper_length:
            # Special handling if first hyper-example is already too long
            # noinspection PyTypeChecker
            raw_hyper_input_ids = []
            if self.add_definition:
                raw_hyper_input_ids += self.metadata[task]["def"] + POST_DEFINITION_INDICATOR
            raw_hyper_input_ids += format_hyper_inputs(hyper_examples[0])
            hyper_input_ids = truncate(
                raw_hyper_input_ids,
                max_length=self.max_hyper_length - self.num_gist_tokens - 1,
                side=LEFT,
            ) + self.gist_tokens
        else:
            hyper_input_ids = []
            if self.add_definition:
                hyper_input_ids += self.metadata[task]["def"] + POST_DEFINITION_INDICATOR
            for ex in candidate_hyper_input_ids:
                if len(hyper_input_ids) + len(ex) > \
                        self.max_hyper_length - self.num_gist_tokens - 1 - len(EXAMPLE_SEPARATOR):
                    break
                hyper_input_ids += ex + EXAMPLE_SEPARATOR
            hyper_input_ids += self.gist_tokens
            hyper_input_ids = pad(hyper_input_ids, side=RIGHT, max_length=self.max_hyper_length - 1)
        hyper_input_ids = [BOS_TOKEN_ID] + hyper_input_ids
        assert len(hyper_input_ids) == self.max_hyper_length

        actual_input_ids = []
        actual_labels = []
        for actual_example in actual_examples:
            input_ids = format_input_ids(actual_example, max_downstream_length=self.max_downstream_length)
            labels = format_labels(actual_example, max_downstream_length=self.max_downstream_length)
            assert len(input_ids) == self.max_downstream_length
            assert len(labels) == self.max_downstream_length
            actual_input_ids.append(input_ids)
            actual_labels.append(labels)

        return {
            "hyper_input_ids": hyper_input_ids,
            "input_ids": actual_input_ids,
            "labels": actual_labels,
        }


class NatInstHyperTrainDataset(IterableDataset):

    def __init__(self, base_path,
                 max_num_hyper_examples: int = 32,
                 num_downstream_examples: int = 1,
                 max_hyper_length: int = 1024,
                 max_downstream_length: int = 384,
                 add_definition: bool = True,
                 hyper_example_mode: str = "sample",
                 hyper_example_copy_factor: float = 1.0,
                 seed_offset: int = 0):
        self.base_path = base_path
        self.max_num_hyper_examples = max_num_hyper_examples
        self.num_downstream_examples = num_downstream_examples
        self.max_hyper_length = max_hyper_length
        self.max_downstream_length = max_downstream_length
        self.add_definition = add_definition
        self.hyper_example_mode = hyper_example_mode
        self.hyper_example_copy_factor = hyper_example_copy_factor
        self.seed_offset = seed_offset

    def __getitem__(self, index):
        raise NotImplementedError

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        rng_seed = worker_info.seed if worker_info is not None else np.random.randint(1_000_000)
        rng_seed += self.seed_offset
        print(f"Using seed {rng_seed}")
        return NatInstHyperTrainIterator(
            rng_seed=rng_seed,
            base_path=self.base_path,
            max_num_hyper_examples=self.max_num_hyper_examples,
            num_downstream_examples=self.num_downstream_examples,
            max_hyper_length=self.max_hyper_length,
            max_downstream_length=self.max_downstream_length,
            add_definition=self.add_definition,
            hyper_example_mode=self.hyper_example_mode,
            hyper_example_copy_factor=self.hyper_example_copy_factor,
        )
