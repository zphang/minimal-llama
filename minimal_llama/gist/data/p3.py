import os
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from tqdm.auto import tqdm
import datasets
import proj2_p3_pretraining.data.p3_metadata as p3_metadata

NON_LABEL_TOKEN_ID = -100

LLAMA_PAD_TOKEN_ID = 0
LLAMA_BOS_TOKEN_ID = 1
LLAMA_EOS_TOKEN_ID = 2

LLAMA_PAD_TYPE_ID = 0
LLAMA_INPUT_TYPE_ID = 1
LLAMA_TARGET_TYPE_ID = 2
LLAMA_GIST_TYPE_ID = 3
LLAMA_HYPER_INPUT_TYPE_ID = 4

LLAMA_VOCAB_SIZE = 32000

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
                 num_gist_tokens,
                 mode="multigist",
                 max_num_examples=32,
                 full_sequence_length=512,
                 add_special_tokens=True,
                 add_answer_indicator=False,
                 predict_input=False,
                 ):
        assert len(ds_weights) == len(ds_list)
        self.rng = np.random.default_rng(rng_seed)
        self.ds_list = ds_list
        self.ds_weights = ds_weights / np.sum(ds_weights)
        self.num_gist_tokens = num_gist_tokens
        self.mode = mode
        self.max_num_examples = max_num_examples
        self.full_sequence_length = full_sequence_length
        self.add_special_tokens = add_special_tokens
        self.predict_input = predict_input
        self.add_answer_indicator = add_answer_indicator

    def __iter__(self):
        return self

    def __next__(self):
        ds_choice = self.rng.choice(len(self.ds_list), p=self.ds_weights)
        ds = self.ds_list[ds_choice]
        if self.mode == "multigist":
            return self.process_multigist(ds=ds)
        elif self.mode == "varied_multigist":
            return self.process_varied_multigist(ds=ds)
        elif self.mode == "zero_shot":
            return self.process_zero_shot(ds=ds)
        else:
            raise KeyError(f"Invalid mode {self.mode}")

    def get_gist_tokens(self):
        gist_tokens = list(range(LLAMA_VOCAB_SIZE, LLAMA_VOCAB_SIZE + self.num_gist_tokens))
        return gist_tokens

    def process_zero_shot(self, ds):
        # assert self.max_num_examples == 1
        index = self.rng.integers(len(ds), size=1)
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
        input_ids = pad_tokens(
            raw_input_ids, max_length=self.full_sequence_length, pad_token_id=LLAMA_PAD_TOKEN_ID)
        type_masks = pad_tokens(
            raw_type_mask, max_length=self.full_sequence_length, pad_token_id=LLAMA_PAD_TYPE_ID)
        if self.predict_input:
            # Don't predict BOS
            labels = [
                input_ids[i] if (
                    type_masks[i] not in (LLAMA_PAD_TYPE_ID, LLAMA_GIST_TYPE_ID)
                    and input_ids[i] != LLAMA_BOS_TOKEN_ID
                ) else -100
                for i in range(1, self.full_sequence_length)
             ] + [-100]
        else:
            labels = [
                input_ids[i] if type_masks[i] == LLAMA_TARGET_TYPE_ID else -100
                for i in range(1, self.full_sequence_length)
            ] + [-100]

        attention_mask = generate_padded_causal_attention_mask(type_mask=type_masks)

        return {
            "input_ids": torch.LongTensor(input_ids),
            "labels": torch.LongTensor(labels),
            "type_mask": torch.LongTensor(type_masks),
            "attention_mask": attention_mask,
        }

    def process_multigist(self, ds):
        indices = self.rng.integers(len(ds), size=self.max_num_examples)
        gist_tokens = self.get_gist_tokens()
        all_input_ids = []
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

            raw_input_ids += gist_tokens
            raw_type_mask += [LLAMA_GIST_TYPE_ID] * self.num_gist_tokens

            all_input_ids.append(raw_input_ids)
            all_type_masks.append(raw_type_mask)

        all_input_ids = [x for input_ids in all_input_ids for x in input_ids]
        all_type_masks = [x for type_mask in all_type_masks for x in type_mask]
        all_input_ids = pad_tokens(
            all_input_ids, max_length=self.full_sequence_length, pad_token_id=LLAMA_PAD_TOKEN_ID)
        all_type_masks = pad_tokens(
            all_type_masks, max_length=self.full_sequence_length, pad_token_id=LLAMA_PAD_TYPE_ID)

        if self.predict_input:
            # Don't predict BOS
            all_labels = [
                all_input_ids[i] if (
                    all_type_masks[i] not in (LLAMA_PAD_TYPE_ID, LLAMA_GIST_TYPE_ID)
                    and all_input_ids[i] != LLAMA_BOS_TOKEN_ID
                ) else -100
                for i in range(1, self.full_sequence_length)
             ] + [-100]
        else:
            all_labels = [
                all_input_ids[i] if all_type_masks[i] == LLAMA_TARGET_TYPE_ID else -100
                for i in range(1, self.full_sequence_length)
            ] + [-100]

        attention_mask = compute_gist_attention_mask(
            all_input_ids,
            num_gist_tokens=self.num_gist_tokens,
            type_mask=all_type_masks,
        )

        return {
            "input_ids": torch.LongTensor(all_input_ids),
            "labels": torch.LongTensor(all_labels),
            "type_mask": torch.LongTensor(all_type_masks),
            "attention_mask": attention_mask,
        }

    def process_varied_multigist(self, ds):
        gist_tokens = self.get_gist_tokens()
        num_fewshot_examples = self.rng.integers(2, self.max_num_examples, size=1)
        indices = self.rng.integers(len(ds), size=num_fewshot_examples)

        all_input_ids = []
        all_type_masks = []
        # Don't include last example
        for index in indices[:-1]:
            example = ds[int(index)]
            if self.add_answer_indicator:
                raw_input_ids = example["inputs"] + ANSWER_INDICATOR_TOKENS + example["targets"]
                raw_type_mask = [LLAMA_HYPER_INPUT_TYPE_ID] * (
                    len(example["inputs"])
                    + len(ANSWER_INDICATOR_TOKENS)
                    + len(example["targets"])
                )
            else:
                raw_input_ids = example["inputs"] + example["targets"]
                raw_type_mask = [LLAMA_HYPER_INPUT_TYPE_ID] * (len(example["inputs"]) + len(example["targets"]))
            if self.add_special_tokens:
                raw_input_ids = [LLAMA_BOS_TOKEN_ID] + raw_input_ids + [LLAMA_EOS_TOKEN_ID]
                raw_type_mask = [LLAMA_HYPER_INPUT_TYPE_ID] + raw_type_mask + [LLAMA_HYPER_INPUT_TYPE_ID]

            all_input_ids.append(raw_input_ids)
            all_type_masks.append(raw_type_mask)

        # Final example
        example = ds[int(indices[-1])]
        last_input_ids = gist_tokens[:]
        last_type_masks = [LLAMA_GIST_TYPE_ID] * self.num_gist_tokens
        if self.add_answer_indicator:
            last_input_ids += example["inputs"] + ANSWER_INDICATOR_TOKENS + example["targets"]
            last_type_masks += (
                [LLAMA_INPUT_TYPE_ID] * len(example["inputs"])
                + [LLAMA_INPUT_TYPE_ID] * len(ANSWER_INDICATOR_TOKENS)
                + [LLAMA_TARGET_TYPE_ID] * len(example["targets"])
            )
        else:
            last_input_ids += example["inputs"] + example["targets"]
            last_type_masks += (
                [LLAMA_INPUT_TYPE_ID] * len(example["inputs"])
                + [LLAMA_TARGET_TYPE_ID] * len(example["targets"])
            )
        if self.add_special_tokens:
            last_input_ids = [LLAMA_BOS_TOKEN_ID] + last_input_ids + [LLAMA_EOS_TOKEN_ID]
            last_type_masks = [LLAMA_INPUT_TYPE_ID] + last_type_masks + [LLAMA_TARGET_TYPE_ID]
        last_input_ids += gist_tokens
        last_type_masks += [LLAMA_GIST_TYPE_ID] * self.num_gist_tokens
        all_input_ids.append(last_input_ids)
        all_type_masks.append(last_type_masks)

        # Collate
        all_input_ids = [x for input_ids in all_input_ids for x in input_ids]
        all_type_masks = [x for type_mask in all_type_masks for x in type_mask]
        all_input_ids = pad_tokens(
            all_input_ids, max_length=self.full_sequence_length, pad_token_id=LLAMA_PAD_TOKEN_ID)
        all_type_masks = pad_tokens(
            all_type_masks, max_length=self.full_sequence_length, pad_token_id=LLAMA_PAD_TYPE_ID)

        if self.predict_input:
            # Don't predict BOS
            all_labels = [
                all_input_ids[i] if (
                    all_type_masks[i] not in (LLAMA_PAD_TYPE_ID, LLAMA_GIST_TYPE_ID)
                    and all_input_ids[i] != LLAMA_BOS_TOKEN_ID
                ) else -100
                for i in range(1, self.full_sequence_length)
             ] + [-100]
        else:
            all_labels = [
                all_input_ids[i] if all_type_masks[i] == LLAMA_TARGET_TYPE_ID else -100
                for i in range(1, self.full_sequence_length)
            ] + [-100]

        attention_mask = compute_gist_attention_mask(
            all_input_ids,
            num_gist_tokens=self.num_gist_tokens,
            type_mask=all_type_masks,
        )

        return {
            "input_ids": torch.LongTensor(all_input_ids),
            "labels": torch.LongTensor(all_labels),
            "type_mask": torch.LongTensor(all_type_masks),
            "attention_mask": attention_mask,
        }


# noinspection PyAbstractClass
class P3FewshotHyperTrainDataset(IterableDataset):

    def __init__(self,
                 base_path,
                 num_gist_tokens,
                 mode="multigist",
                 max_num_examples=32,
                 full_sequence_length=512,
                 add_special_tokens=True,
                 predict_input=False,
                 add_answer_indicator=False,
                 subset=None,
                 explicit_seed=None,
                 single_task_mode=False):
        self.base_path = base_path
        self.num_gist_tokens = num_gist_tokens
        self.mode = mode
        self.max_num_examples = max_num_examples
        self.full_sequence_length = full_sequence_length
        self.add_special_tokens = add_special_tokens
        self.add_answer_indicator = add_answer_indicator
        self.predict_input = predict_input
        self.subset = subset
        self.explicit_seed = explicit_seed
        self.single_task_mode = single_task_mode

        train_and_caps = p3_metadata.get_full_t0_train_and_caps()
        # Should take about 3 seconds to initialize
        if self.single_task_mode:
            assert len(subset) == 1
            self.ds_list = [
                datasets.load_from_disk(os.path.join(base_path, "train", subset[0]))
            ]
            self.ds_weights = np.array([1.])
        else:
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
            num_gist_tokens=self.num_gist_tokens,
            mode=self.mode,
            max_num_examples=self.max_num_examples,
            full_sequence_length=self.full_sequence_length,
            predict_input=self.predict_input,
            add_answer_indicator=self.add_answer_indicator,
        )


class P3FewshotHyperValidationDataset(Dataset):
    def __init__(self,
                 dataset_name,
                 base_path,
                 rng_seed,
                 num_gist_tokens,
                 mode="multigist",
                 max_num_examples=32,
                 full_sequence_length=512,
                 add_special_tokens=True,
                 add_answer_indicator=False,
                 predict_input=False,
                 do_test=False):
        self.dataset_name = dataset_name
        self.base_path = base_path
        self.rng = np.random.default_rng(rng_seed)
        self.num_gist_tokens = num_gist_tokens
        self.mode = mode
        self.max_num_examples = max_num_examples
        self.full_sequence_length = full_sequence_length
        self.add_special_tokens = add_special_tokens
        self.predict_input = predict_input
        self.add_answer_indicator = add_answer_indicator

        eval_phase = "test" if do_test else "validation"

        self.train_ds = datasets.load_from_disk(
            os.path.join(base_path, "train", self.dataset_name))
        self.val_ds = datasets.load_from_disk(
            os.path.join(base_path, eval_phase, self.dataset_name))

        self.train_indices = self.rng.choice(
            len(self.train_ds),
            size=(len(self.val_ds), (self.max_num_examples-1)),
        )

    def __len__(self):
        return len(self.val_ds)

    def __getitem__(self, idx):
        if self.mode in ("multigist", "varied_multigist"):
            return self.process_multigist(idx)
        elif self.mode == "zero_shot":
            return self.process_zero_shot(idx)
        else:
            raise KeyError(f"Invalid mode {self.mode}")

    def get_gist_tokens(self):
        gist_tokens = list(range(LLAMA_VOCAB_SIZE, LLAMA_VOCAB_SIZE + self.num_gist_tokens))
        return gist_tokens

    def process_zero_shot(self, idx):
        example = self.val_ds[idx]
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
        input_ids = pad_tokens(
            raw_input_ids, max_length=self.full_sequence_length, pad_token_id=LLAMA_PAD_TOKEN_ID)
        type_masks = pad_tokens(
            raw_type_mask, max_length=self.full_sequence_length, pad_token_id=LLAMA_PAD_TYPE_ID)
        labels = [
            input_ids[i] if type_masks[i] == LLAMA_TARGET_TYPE_ID else -100
            for i in range(1, self.full_sequence_length)
        ] + [-100]

        attention_mask = generate_padded_causal_attention_mask(type_mask=type_masks)

        out = {
            "input_ids": torch.LongTensor(input_ids),
            "labels": torch.LongTensor(labels),
            "type_mask": torch.LongTensor(type_masks),
            "attention_mask": attention_mask,
        }
        if "is_correct" in example:
            out["is_correct"] = example["is_correct"]
            out["key"] = tuple(example["inputs"])
        return out

    def process_multigist(self, idx):
        gist_tokens = self.get_gist_tokens()

        all_input_ids = []
        all_type_masks = []
        # Don't include last example
        for index in self.train_indices[idx]:
            example = self.train_ds[int(index)]
            if self.add_answer_indicator:
                raw_input_ids = example["inputs"] + ANSWER_INDICATOR_TOKENS + example["targets"]
                raw_type_mask = [LLAMA_HYPER_INPUT_TYPE_ID] * (
                    len(example["inputs"])
                    + len(ANSWER_INDICATOR_TOKENS)
                    + len(example["targets"])
                )
            else:
                raw_input_ids = example["inputs"] + example["targets"]
                raw_type_mask = [LLAMA_HYPER_INPUT_TYPE_ID] * (len(example["inputs"]) + len(example["targets"]))
            if self.add_special_tokens:
                raw_input_ids = [LLAMA_BOS_TOKEN_ID] + raw_input_ids + [LLAMA_EOS_TOKEN_ID]
                raw_type_mask = [LLAMA_HYPER_INPUT_TYPE_ID] + raw_type_mask + [LLAMA_HYPER_INPUT_TYPE_ID]

            all_input_ids.append(raw_input_ids)
            all_type_masks.append(raw_type_mask)

        # Final example
        example = self.val_ds[idx]
        last_input_ids = gist_tokens[:]
        last_type_masks = [LLAMA_GIST_TYPE_ID] * self.num_gist_tokens
        if self.add_answer_indicator:
            last_input_ids += example["inputs"] + ANSWER_INDICATOR_TOKENS + example["targets"]
            last_type_masks += (
                [LLAMA_INPUT_TYPE_ID] * len(example["inputs"])
                + [LLAMA_INPUT_TYPE_ID] * len(ANSWER_INDICATOR_TOKENS)
                + [LLAMA_TARGET_TYPE_ID] * len(example["targets"])
            )
        else:
            last_input_ids += example["inputs"] + example["targets"]
            last_type_masks += (
                [LLAMA_INPUT_TYPE_ID] * len(example["inputs"])
                + [LLAMA_TARGET_TYPE_ID] * len(example["targets"])
            )
        if self.add_special_tokens:
            last_input_ids = [LLAMA_BOS_TOKEN_ID] + last_input_ids + [LLAMA_EOS_TOKEN_ID]
            last_type_masks = [LLAMA_INPUT_TYPE_ID] + last_type_masks + [LLAMA_TARGET_TYPE_ID]
        last_input_ids += gist_tokens
        last_type_masks += [LLAMA_GIST_TYPE_ID] * self.num_gist_tokens
        all_input_ids.append(last_input_ids)
        all_type_masks.append(last_type_masks)

        # Collate
        all_input_ids = [x for input_ids in all_input_ids for x in input_ids]
        all_type_masks = [x for type_mask in all_type_masks for x in type_mask]
        all_input_ids = pad_tokens(
            all_input_ids, max_length=self.full_sequence_length, pad_token_id=LLAMA_PAD_TOKEN_ID)
        all_type_masks = pad_tokens(
            all_type_masks, max_length=self.full_sequence_length, pad_token_id=LLAMA_PAD_TYPE_ID)

        if self.predict_input:
            # Don't predict BOS
            all_labels = [
                all_input_ids[i] if (
                    all_type_masks[i] not in (LLAMA_PAD_TYPE_ID, LLAMA_GIST_TYPE_ID)
                    and all_input_ids[i] != LLAMA_BOS_TOKEN_ID
                ) else -100
                for i in range(1, self.full_sequence_length)
             ] + [-100]
        else:
            all_labels = [
                all_input_ids[i] if all_type_masks[i] == LLAMA_TARGET_TYPE_ID else -100
                for i in range(1, self.full_sequence_length)
            ] + [-100]

        attention_mask = compute_gist_attention_mask(
            all_input_ids,
            num_gist_tokens=self.num_gist_tokens,
            type_mask=all_type_masks,
        )

        out = {
            "input_ids": torch.LongTensor(all_input_ids),
            "labels": torch.LongTensor(all_labels),
            "type_mask": torch.LongTensor(all_type_masks),
            "attention_mask": attention_mask,
        }
        if "is_correct" in example:
            out["is_correct"] = example["is_correct"]
            out["key"] = tuple(example["inputs"])
        return out


def compute_gist_attention_mask(input_ids, num_gist_tokens, type_mask):
    # Note, only works if there are gist tokens after.
    # TODO: write tests for this.
    input_ids = torch.LongTensor(input_ids)
    max_seq_len = len(input_ids)
    attention_mask = torch.ones([max_seq_len, max_seq_len]).tril()
    last_indices = torch.arange(len(input_ids))[input_ids == LLAMA_VOCAB_SIZE + num_gist_tokens - 1].tolist()
    for last_index, prev_last_index in zip(last_indices[1:], last_indices[:-1]):
        a = prev_last_index + 1
        b = last_index - num_gist_tokens + 1
        c = 0
        d = prev_last_index - num_gist_tokens + 1
        attention_mask[a:b, c:d] = 0
    attention_mask = mask_padded_tokens(attention_mask, type_mask)
    return attention_mask.long().cpu()


def data_collator(features: list) -> dict:
    keys = features[0].keys()
    batch = {
        k: torch.stack([torch.LongTensor(f[k]) for f in features])
        for k in keys
        if k not in ("key", "is_correct")
    }
    if "is_correct" in keys:
        for key in ["key", "is_correct"]:
            batch[key] = [f[key] for f in features]
    return batch


def clean_input(tokens):
    return [x for x in tokens if x != LLAMA_PAD_TOKEN_ID and x < LLAMA_VOCAB_SIZE]


def clean_label(tokens):
    return [x for x in tokens if x != NON_LABEL_TOKEN_ID]


def clean_pred(pred):
    pred = pred.tolist()
    if pred[0] == LLAMA_BOS_TOKEN_ID:
        pred = pred[1:]
    if LLAMA_EOS_TOKEN_ID in pred:
        pred = pred[:pred.index(LLAMA_EOS_TOKEN_ID)+1]
    return pred


def mask_padded_tokens(attention_mask, type_mask):
    num_padding_tokens = sum([1 for x in type_mask if x != LLAMA_PAD_TYPE_ID])
    attention_mask[:, :num_padding_tokens] = 0
    return attention_mask


def generate_padded_causal_attention_mask(type_mask):
    max_seq_len = len(type_mask)
    # padding tokens are always on the left
    attention_mask = torch.ones([max_seq_len, max_seq_len]).tril().long().cpu()
    return mask_padded_tokens(attention_mask, type_mask)
