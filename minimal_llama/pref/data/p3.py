import os
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from tqdm.auto import tqdm
import datasets
import proj2_p3_pretraining.data.p3_metadata as p3_metadata


LLAMA_PAD_TOKEN_ID = 0
LLAMA_BOS_TOKEN_ID = 1
LLAMA_EOS_TOKEN_ID = 2

LLAMA_PAD_TYPE_ID = 0
LLAMA_INPUT_TYPE_ID = 1
LLAMA_TARGET_TYPE_ID = 2

ANSWER_INDICATOR_TOKENS = [673, 29901]


def pad_tokens(x, max_length, pad_token_id,
               truncate_from="left",
               pad_from="left"):
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


class P3FewshotHyperTrainIterator:
    def __init__(self, rng_seed, ds_list, ds_weights,
                 block_size=64,
                 full_sequence_length=512,
                 add_special_tokens=True,
                 add_answer_indicator=False,
                 predict_input=False,
                 ):
        assert len(ds_weights) == len(ds_list)
        assert full_sequence_length % block_size == 0
        self.rng = np.random.default_rng(rng_seed)
        self.ds_list = ds_list
        self.ds_weights = ds_weights / np.sum(ds_weights)
        self.block_size = block_size
        self.full_sequence_length = full_sequence_length
        self.add_special_tokens = add_special_tokens
        self.predict_input = predict_input
        self.add_answer_indicator = add_answer_indicator

        self.num_examples = full_sequence_length // block_size

    def __iter__(self):
        return self

    def __next__(self):
        ds_choice = self.rng.choice(len(self.ds_list), p=self.ds_weights)
        ds = self.ds_list[ds_choice]
        indices = self.rng.integers(len(ds), size=self.num_examples)

        all_input_ids = []
        all_labels = []
        all_type_masks = []
        for index in indices:
            example = ds[int(index)]
            if self.add_answer_indicator:
                raw_input_ids = example["inputs"] + ANSWER_INDICATOR_TOKENS + example["targets"]
                raw_type_mask = (
                    [LLAMA_INPUT_TYPE_ID] * len(example["inputs"])
                    + [LLAMA_INPUT_TYPE_ID] * len(ANSWER_INDICATOR_TOKENS)
                    + [LLAMA_TARGET_TYPE_ID] * len(example["targets"])
                )
            else:
                raw_input_ids = example["inputs"] + example["targets"]
                raw_type_mask = (
                    [LLAMA_INPUT_TYPE_ID] * len(example["inputs"])
                    + [LLAMA_TARGET_TYPE_ID] * len(example["targets"])
                )
            if self.add_special_tokens:
                raw_input_ids = [LLAMA_BOS_TOKEN_ID] + raw_input_ids + [LLAMA_EOS_TOKEN_ID]
                raw_type_mask = [LLAMA_INPUT_TYPE_ID] + raw_type_mask + [LLAMA_TARGET_TYPE_ID]
            input_ids = pad_tokens(raw_input_ids, max_length=self.block_size, pad_token_id=LLAMA_PAD_TOKEN_ID)
            type_mask = pad_tokens(raw_type_mask, max_length=self.block_size, pad_token_id=LLAMA_PAD_TYPE_ID)
            if self.predict_input:
                # Don't predict BOS
                labels = [
                    input_ids[i] if (type_mask[i] != LLAMA_PAD_TYPE_ID and input_ids[i] != LLAMA_BOS_TOKEN_ID) else -100
                    for i in range(1, self.block_size)
                ] + [-100]
            else:
                labels = [
                    input_ids[i] if type_mask[i] == LLAMA_TARGET_TYPE_ID else -100
                    for i in range(1, self.block_size)
                ] + [-100]

            label_type_mask = type_mask[1:] + [LLAMA_PAD_TYPE_ID]
            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_type_masks.append(label_type_mask)

        return {
            "input_ids": torch.LongTensor(all_input_ids).view(-1),
            "labels": torch.LongTensor(all_labels).view(-1),
            "type_mask": torch.LongTensor(all_type_masks).view(-1),
        }


# noinspection PyAbstractClass
class P3FewshotHyperTrainDataset(IterableDataset):

    def __init__(self, base_path,
                 block_size=64,
                 full_sequence_length=512,
                 add_special_tokens=True,
                 predict_input=False,
                 add_answer_indicator=False,
                 subset=None,
                 explicit_seed=None):
        assert full_sequence_length % block_size == 0
        self.base_path = base_path
        self.block_size = block_size
        self.full_sequence_length = full_sequence_length
        self.add_special_tokens = add_special_tokens
        self.add_answer_indicator = add_answer_indicator
        self.predict_input = predict_input
        self.subset = subset
        self.explicit_seed = explicit_seed

        train_and_caps = p3_metadata.get_full_t0_train_and_caps()
        # Should take about 3 seconds to initialize
        self.ds_list = []
        self.ds_weights = []
        for name, weight in zip(tqdm(train_and_caps["names"]), train_and_caps["caps"]):
            if subset and name not in subset:
                continue
            self.ds_list.append(datasets.load_from_disk(
                os.path.join(base_path, "train", name)))
            self.ds_weights.append(weight)
        assert self.ds_list, "No valid datasets."
        self.ds_weights = np.array(self.ds_weights)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if self.explicit_seed is None:
            rng_seed = worker_info.seed if worker_info is not None else np.random.randint(1_000_000)
        else:
            rng_seed = self.explicit_seed
        return P3FewshotHyperTrainIterator(
            rng_seed=rng_seed,
            ds_list=self.ds_list,
            ds_weights=self.ds_weights,
            predict_input=self.predict_input,
            add_answer_indicator=self.add_answer_indicator,
        )
