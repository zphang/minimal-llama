import os
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

import datasets
import pandas as pd
import proj_shared.io_utils as io_utils


NON_LABEL_TOKEN_ID = -100
T5_EXAMPLE_INPUT_TOKEN_ID = 32000
T5_EXAMPLE_OUTPUT_TOKEN_ID = 32001
T5_EOS_TOKEN_ID = 1
T5_PAD_TOKEN_ID = 0


def pad_tokens(x, max_length, pad_token_id,
               truncate_from="left",
               pad_from="right"):
    """Pad tokens, either with pad token for inputs or -100 for labels."""
    assert truncate_from in ("left", "right")
    assert pad_from in ("left", "right")
    if len(x) > max_length:
        if truncate_from == "left":
            return x[-max_length:]
        else:
            return x[:max_length]
    elif len(x) < max_length:
        padding = [pad_token_id] * (max_length - len(x))
        if pad_from == "left":
            return padding + x
        else:
            return x + padding
    else:
        return x


def pad_inputs(input_ids, tokenizer, max_input_length,
               truncate_from="left",
               pad_from="right"):
    assert pad_from in ("left", "right")
    num_valid_input_tokens = min(max_input_length, len(input_ids))
    input_ids = pad_tokens(
        input_ids,
        max_length=max_input_length,
        pad_token_id=tokenizer.pad_token_id,
        truncate_from=truncate_from,
        pad_from=pad_from,
    )
    valid_mask = [1] * num_valid_input_tokens
    padding_mask = [T5_PAD_TOKEN_ID] * (max_input_length - num_valid_input_tokens)
    if pad_from == "left":
        input_mask = padding_mask + valid_mask
    else:
        input_mask = valid_mask + padding_mask
    return {
        "input_ids": torch.LongTensor(input_ids),
        "attention_mask": torch.FloatTensor(input_mask),
    }


def format_fewshot_hyper_input(example) -> list:
    tokens = [T5_EXAMPLE_INPUT_TOKEN_ID] + example["input"]
    if "output" in example:
        target = example["output"][0]
        if len(target) and target[-1] == T5_EOS_TOKEN_ID:
            target = target[:-1]
        tokens += [T5_EXAMPLE_OUTPUT_TOKEN_ID] + target
    # Note: the target already ends with </s>
    # noinspection PyTypeChecker
    return tokens


def prep_joint_hyper_examples(instruction, hyper_examples, tokenizer, max_hyper_input_length):
    if instruction:
        instruction_and_hyper_examples = [{"input": instruction}] + hyper_examples
    else:
        instruction_and_hyper_examples = hyper_examples
    padded_hyper_examples = []
    for hyper_example in instruction_and_hyper_examples:
        hyper_input_ids = format_fewshot_hyper_input(hyper_example)
        padded_hyper_example = pad_inputs(
            input_ids=hyper_input_ids,
            tokenizer=tokenizer,
            max_input_length=max_hyper_input_length,
            truncate_from="left",
            pad_from="left",
        )
        padded_hyper_examples.append(padded_hyper_example)
    return padded_hyper_examples


def prep_split_hyper_examples(instruction, hyper_examples, tokenizer, max_hyper_input_length):
    if instruction:
        instruction_and_hyper_examples = [{"input": instruction}] + hyper_examples
    else:
        instruction_and_hyper_examples = hyper_examples
    padded_hyper_examples = []
    for hyper_example in instruction_and_hyper_examples:
        padded_hyper_inputs = pad_inputs(
            input_ids=hyper_example["input"],
            tokenizer=tokenizer,
            max_input_length=max_hyper_input_length,
            truncate_from="left",
            pad_from="right",
        )
        padded_hyper_targets = pad_inputs(
            input_ids=hyper_example["output"][0],
            tokenizer=tokenizer,
            max_input_length=max_hyper_input_length,
            truncate_from="left",
            pad_from="right",
        )
        padded_hyper_examples.append({
            "input_input_ids": padded_hyper_inputs["input_ids"],
            "input_attention_mask": padded_hyper_inputs["attention_mask"],
            "target_input_ids": padded_hyper_targets["input_ids"],
            "target_attention_mask": padded_hyper_targets["attention_mask"],
        })
    return padded_hyper_examples


def prep_concatenated_hyper_examples(instruction, hyper_examples, tokenizer, max_hyper_input_length):
    hyper_input_ids = []
    if instruction:
        hyper_input_ids = instruction.copy()
        # Delimiter is prepended to following examples, so we don't need to add it here
    for hyper_example in hyper_examples:
        hyper_input_ids += format_fewshot_hyper_input(hyper_example)
    return pad_inputs(
        input_ids=hyper_input_ids,
        tokenizer=tokenizer,
        max_input_length=max_hyper_input_length,
        truncate_from="right",
        pad_from="right",
    )


def get_hyper_examples_format_func(format_str):
    return {
        "joint": prep_joint_hyper_examples,
        "split": prep_split_hyper_examples,
        "concatenated": prep_concatenated_hyper_examples,
    }[format_str]


class NatInstFewshotHyperTrainIterator:
    def __init__(self,
                 rng_seed,
                 tokenizer,
                 base_path,
                 add_instructions=True,
                 add_2pos=False,
                 phase="train",
                 num_hyper_examples=4,
                 num_actual_examples=1,
                 max_hyper_input_length=512,
                 max_downstream_input_length=512,
                 max_downstream_target_length=128,
                 hyper_input_format="joint",
                 actual_examples_mode="single",
                 task_exclude_list=None,
                 add_eos_to_input=False,
                 ):
        self.tokenizer = tokenizer
        self.base_path = base_path
        self.phase = phase
        self.add_instructions = add_instructions
        self.add_2pos = add_2pos
        self.num_hyper_examples = num_hyper_examples
        self.num_actual_examples = num_actual_examples
        self.rng = np.random.default_rng(rng_seed)
        self.max_hyper_input_length = max_hyper_input_length
        self.max_downstream_input_length = max_downstream_input_length
        self.max_downstream_target_length = max_downstream_target_length
        self.hyper_input_format = hyper_input_format
        self.actual_examples_mode = actual_examples_mode
        self.hyper_examples_format_func = get_hyper_examples_format_func(self.hyper_input_format)
        self.task_exclude_list = task_exclude_list
        self.add_eos_to_input = add_eos_to_input

        if self.actual_examples_mode == "single":
            assert self.num_actual_examples == 1

        self.metadata = io_utils.read_json(os.path.join(base_path, "metadata.json"))
        self.task_list = io_utils.read_json(os.path.join(base_path, "splits.json"))[self.phase]
        self.ds = datasets.load_from_disk(os.path.join(base_path, f"{phase}_dataset"), keep_in_memory=True)
        self.examples_per_task_srs = pd.DataFrame(self.metadata).T.loc[self.task_list]["num_instances"]
        self.task_weights = self.examples_per_task_srs / self.examples_per_task_srs.sum()

        # Don't modify task_list/examples_per_task_srs directly. They are tied to the dataset
        if task_exclude_list:
            for task_name in task_exclude_list:
                if task_name in self.task_weights:
                    self.task_weights.loc[task_name] = 0.0
            self.task_weights = self.task_weights / self.task_weights.sum()

        self.task_start_index_dict = self.examples_per_task_srs.cumsum().shift().fillna(0).to_dict()

    def __iter__(self):
        return self

    def __next__(self):
        task_name = self.rng.choice(self.task_list, p=self.task_weights)
        num_examples = self.examples_per_task_srs.loc[task_name]
        within_task_indices = self.rng.choice(
            a=np.arange(num_examples),
            size=self.num_hyper_examples + self.num_actual_examples,
            replace=False,
        )

        # apply offset
        indices = within_task_indices + self.task_start_index_dict[task_name]
        fewshot_indices = indices[:self.num_hyper_examples]
        actual_indices = indices[self.num_hyper_examples:]
        raw_hyper_examples = [self.ds[int(idx)] for idx in fewshot_indices]
        if self.add_2pos:
            raw_hyper_examples = process_positive_examples(
                metadata=self.metadata,
                task_name=task_name,
                num=2,
            ) + raw_hyper_examples
        if self.add_instructions:
            instruction = self.metadata[task_name]["definition"][0]
        else:
            instruction = None
        hyper_examples = self.hyper_examples_format_func(
            instruction=instruction,
            hyper_examples=raw_hyper_examples,
            tokenizer=self.tokenizer,
            max_hyper_input_length=self.max_hyper_input_length,
        )
        actual_examples = [self.ds[int(idx)] for idx in actual_indices]
        actual_inputs_list = []
        for actual_example in actual_examples:
            actual_example_input_ids = actual_example["input"]
            if self.add_eos_to_input:
                actual_example_input_ids = actual_example_input_ids + [T5_EOS_TOKEN_ID]
            actual_inputs_list.append(pad_inputs(
                input_ids=actual_example_input_ids,
                tokenizer=self.tokenizer,
                max_input_length=self.max_downstream_input_length,
            ))
        actual_target_ids_list = [
            torch.LongTensor(pad_tokens(
                actual_example["output"][0] + [T5_EOS_TOKEN_ID],
                max_length=self.max_downstream_target_length,
                pad_token_id=NON_LABEL_TOKEN_ID,
            ))
            for actual_example in actual_examples
        ]
        if self.actual_examples_mode == "single":
            assert len(actual_inputs_list) == 1
            out_example = {
                "input_ids": actual_inputs_list[0]["input_ids"],
                "attention_mask": actual_inputs_list[0]["attention_mask"],
                "labels": actual_target_ids_list[0],
                "hyper_examples": hyper_examples,
            }
        elif self.actual_examples_mode == "multiple":
            out_example = {
                "input_ids": [example["input_ids"] for example in actual_inputs_list],
                "attention_mask": [example["attention_mask"] for example in actual_inputs_list],
                "labels": actual_target_ids_list,
                "hyper_examples": hyper_examples,
            }
        else:
            raise KeyError(self.actual_examples_mode)
        return out_example


# noinspection PyAbstractClass
class NatInstFewshotHyperTrainDataset(IterableDataset):

    def __init__(self, tokenizer, base_path,
                 add_instructions=True,
                 add_2pos=False,
                 num_hyper_examples=4,
                 num_actual_examples=1,
                 max_hyper_input_length=512,
                 max_downstream_input_length=512,
                 max_downstream_target_length=128,
                 hyper_input_format="joint",
                 actual_examples_mode="single",
                 add_eos_to_input=False,
                 task_exclude_list=None):
        self.tokenizer = tokenizer
        self.base_path = base_path
        self.add_instructions = add_instructions
        self.add_2pos = add_2pos
        self.num_hyper_examples = num_hyper_examples
        self.num_actual_examples = num_actual_examples
        self.max_hyper_input_length = max_hyper_input_length
        self.max_downstream_input_length = max_downstream_input_length
        self.max_downstream_target_length = max_downstream_target_length
        self.hyper_input_format = hyper_input_format
        self.actual_examples_mode = actual_examples_mode
        self.add_eos_to_input = add_eos_to_input
        self.task_exclude_list = task_exclude_list

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        rng_seed = worker_info.seed if worker_info is not None else np.random.randint(1_000_000)
        return NatInstFewshotHyperTrainIterator(
            rng_seed=rng_seed,
            tokenizer=self.tokenizer,
            base_path=self.base_path,
            phase="train",
            add_instructions=self.add_instructions,
            add_2pos=self.add_2pos,
            num_hyper_examples=self.num_hyper_examples,
            num_actual_examples=self.num_actual_examples,
            max_hyper_input_length=self.max_hyper_input_length,
            max_downstream_input_length=self.max_downstream_input_length,
            max_downstream_target_length=self.max_downstream_target_length,
            hyper_input_format=self.hyper_input_format,
            actual_examples_mode=self.actual_examples_mode,
            add_eos_to_input=self.add_eos_to_input,
            task_exclude_list=self.task_exclude_list,
        )


class NatInstFewshotHyperValidationDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 base_path,
                 rng_seed,
                 add_instructions=True,
                 add_2pos=False,
                 num_hyper_examples=4,
                 num_actual_examples=1,
                 max_hyper_input_length=512,
                 max_downstream_input_length=512,
                 max_downstream_target_length=128,
                 hyper_input_format="joint",
                 actual_examples_mode="single",
                 subset=None,
                 do_test=False):
        self.tokenizer = tokenizer
        self.base_path = base_path
        self.add_instructions = add_instructions
        self.add_2pos = add_2pos
        self.num_hyper_examples = num_hyper_examples
        self.num_actual_examples = num_actual_examples
        self.rng = np.random.default_rng(rng_seed)
        self.max_hyper_input_length = max_hyper_input_length
        self.max_downstream_input_length = max_downstream_input_length
        self.max_downstream_target_length = max_downstream_target_length
        self.actual_examples_mode = actual_examples_mode
        self.hyper_input_format = hyper_input_format
        self.hyper_examples_format_func = get_hyper_examples_format_func(self.hyper_input_format)
        self.subset = subset
        self.do_test = do_test

        eval_phase = "test" if do_test else "val"
        self.metadata = io_utils.read_json(os.path.join(base_path, "metadata.json"))
        self.task_list = io_utils.read_json(os.path.join(base_path, "splits.json"))[eval_phase]
        self.ds = datasets.load_from_disk(os.path.join(base_path, f"{eval_phase}_dataset"))
        if subset is not None:
            assert set(subset) <= set(self.task_list)
            self.task_list = [task for task in self.task_list if task in subset]
            if subset:
                self.ds = self.ds.filter(lambda example: example["task_name"] in subset)
        self.examples_per_task_srs = pd.DataFrame(self.metadata).T.loc[self.task_list]["num_instances"]
        self.task_start_index_srs = self.examples_per_task_srs.cumsum().shift().fillna(0)
        self.train_indices = self.create_train_indices()
        # Eval always uses 1 actual example for now
        assert self.num_actual_examples == 1

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        task_name = self.ds[idx]["task_name"]
        raw_hyper_examples = [self.ds[int(train_idx)] for train_idx in self.train_indices[idx]]
        if self.add_2pos:
            raw_hyper_examples = process_positive_examples(
                metadata=self.metadata,
                task_name=task_name,
                num=2,
            ) + raw_hyper_examples
        if self.add_instructions:
            instruction = self.metadata[task_name]["definition"][0]
        else:
            instruction = None
        hyper_examples = self.hyper_examples_format_func(
            instruction=instruction,
            hyper_examples=raw_hyper_examples,
            tokenizer=self.tokenizer,
            max_hyper_input_length=self.max_hyper_input_length,
        )

        actual_examples = [self.ds[int(idx)]]
        actual_inputs_list = [
            pad_inputs(
                input_ids=actual_example["input"],
                tokenizer=self.tokenizer,
                max_input_length=self.max_downstream_input_length,
            )
            for actual_example in actual_examples
        ]
        actual_target_ids_list = [
            torch.LongTensor(pad_tokens(
                actual_example["output"][0] + [T5_EOS_TOKEN_ID],
                max_length=self.max_downstream_target_length,
                pad_token_id=NON_LABEL_TOKEN_ID,
            ))
            for actual_example in actual_examples
        ]
        if self.actual_examples_mode == "single":
            assert len(actual_inputs_list) == 1
            out_example = {
                "input_ids": actual_inputs_list[0]["input_ids"],
                "attention_mask": actual_inputs_list[0]["attention_mask"],
                "labels": actual_target_ids_list[0],
                "hyper_examples": hyper_examples,
            }
        elif self.actual_examples_mode == "multiple":
            out_example = {
                "input_ids": [example["input_ids"] for example in actual_inputs_list],
                "attention_mask": [example["attention_mask"] for example in actual_inputs_list],
                "labels": actual_target_ids_list,
                "hyper_examples": hyper_examples,
            }
        else:
            raise KeyError(self.actual_examples_mode)
        return out_example

    def create_train_indices(self):
        fewshot_train_indices = []
        for task_name in self.task_list:
            num_examples = self.examples_per_task_srs.loc[task_name]
            for i in range(num_examples):
                within_task_indices = self.rng.choice(
                    a=arr_ex_one(num_examples, i),  # exclude this example
                    size=self.num_hyper_examples,
                )
                global_index = self.task_start_index_srs.loc[task_name] + within_task_indices
                fewshot_train_indices.append(global_index)
        return np.array(fewshot_train_indices)


def arr_ex_one(n, i):
    x = np.arange(n)
    s = np.ones(n).astype(bool)
    s[i] = False
    return x[s]


def process_positive_examples(metadata, task_name, num=2):
    raw_examples = metadata[task_name]["pos_examples"][:num]
    return [
        {
            "input": example["input"],
            "output": [example["output"]],
        }
        for example in raw_examples
    ]

