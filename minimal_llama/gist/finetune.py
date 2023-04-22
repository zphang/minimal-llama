import os
import sys

import datasets
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

from transformers import (
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)
import transformers
from transformers.trainer_utils import get_last_checkpoint
from torch.utils.data import Dataset
import minimal_llama.gist.data.p3 as p3_datasets
import minimal_llama.gist.llama_gist as llama_gist
import proj_shared.assets_utils as assets_utils
import proj_shared.io_utils as io_utils


@dataclass
class FinetuneArguments:
    dataset_path: str = field()
    hf_path: str = field()
    model_name: str = field(default="7b")
    input_loss_weight: float = field(default=None)

    # p3_specific
    p3_subset_name: str = field(default="t0_short")
    add_answer_indicator: bool = field(default=False)


@dataclass
class GistArguments:
    max_num_examples: int = field(default=32)
    num_gist_tokens: int = field(default=8)
    max_sequence_length: int = field(default=512)


class ModifiedTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):

        labels = inputs["labels"]
        input_ids = inputs["input_ids"]

        # logits will be 1 block shorter than input_ids, since we're dropping off the first block
        logits = model(input_ids=input_ids, attention_mask=inputs["attention_mask"])

        if model.finetune_args.input_loss_weight is not None:
            assert "type_mask" in inputs
            type_mask = inputs["type_mask"]
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
            all_loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            loss_weights_by_type = torch.Tensor([0.0, model.finetune_args.input_loss_weight, 1.0, 0.0]).float().cuda()
            loss_weights = loss_weights_by_type[type_mask.reshape(-1)]
            loss = (all_loss * loss_weights).sum() / loss_weights.sum()
        else:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.reshape(
                -1, logits.size(-1)), labels.reshape(-1)
            )
        if return_outputs:
            return loss, logits
        else:
            return loss

    def _final_ops_before_train(self):
        pass

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        torch.save(
            self.model.state_dict(),
            os.path.join(output_dir, f"checkpoint.p"),
        )

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def data_collator(features: list) -> dict:
    keys = features[0].keys()
    batch = {
        k: torch.stack([torch.LongTensor(f[k]) for f in features])
        for k in keys
    }
    return batch


def main():
    finetune_args, gist_args, training_args = HfArgumentParser((
        FinetuneArguments,
        GistArguments,
        TrainingArguments,
    )).parse_args_into_dataclasses()
    training_args.remove_unused_columns = False
    if finetune_args.p3_subset_name == "t0_short":
        subset_filename = "p3_t0_short_tasks.json"
        subset = io_utils.read_json(assets_utils.get_assets_path("subsets", subset_filename))
    elif finetune_args.p3_subset_name.startswith("single_task:"):
        subset = [finetune_args.p3_subset_name.split("single_task:")[1]]
    else:
        raise KeyError(finetune_args.p3_subset_name)

    train_ds = p3_datasets.P3FewshotHyperTrainDataset(
        base_path=finetune_args.dataset_path,
        num_gist_tokens=gist_args.num_gist_tokens,
        full_sequence_length=gist_args.max_sequence_length,
        add_special_tokens=True,
        add_answer_indicator=finetune_args.add_answer_indicator,
        subset=subset,
        predict_input=finetune_args.input_loss_weight is not None,
    )
    print(training_args.bf16)
    model = llama_gist.create_model(
        model_name=finetune_args.model_name,
        hf_path=finetune_args.hf_path,
        num_gist_tokens=gist_args.num_gist_tokens,
        dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
    )
    model.finetune_args = finetune_args
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    trainer = ModifiedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
    )
    trainer.train()


if __name__ == "__main__":
    main()

