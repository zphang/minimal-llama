import os
import sys

import datasets
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
import transformers
from transformers.trainer_utils import get_last_checkpoint
from torch.utils.data import Dataset


@dataclass
class FinetuneArguments:
    dataset_path: str = field()
    tokenizer_path: str = field()
    model_path: str = field()


class ModifiedTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            attention_mask=torch.ones_like(inputs["input_ids"]),
            labels=inputs["input_ids"],
        ).loss


def data_collator(features: list) -> dict:
    batch = {
        "input_ids": torch.stack([
            torch.LongTensor(f["input_ids"][:7])
            for f in features
        ])
    }
    return batch


def main():
    finetune_args, training_args = HfArgumentParser((
        FinetuneArguments,
        TrainingArguments,
    )).parse_args_into_dataclasses()
    train_ds = datasets.load_from_disk(finetune_args.dataset_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(finetune_args.model_path)

    trainer = ModifiedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
    )
    trainer.train()


if __name__ == "__main__":
    main()

