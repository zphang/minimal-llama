import argparse
import os
import math
from dataclasses import dataclass, field
import tqdm.auto as tqdm
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import datasets
import transformers
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
import minimal_llama.pref.llama_compress as llama_compress


@dataclass
class FinetuneArguments:
    dataset_path: str = field()
    hf_path: str = field()
    model_name: str = field(default="7b")
    use_8bit: bool = field(default=False)


@dataclass
class CompressArguments:
    peft_mode: str = field(default="prefix")
    num_prefix_tokens: int = field(default=16)
    block_size: int = field(default=64)
    factorized_compressor: bool = field(default=True)
    adapter_gate_mode: str = field(default="fixed")


class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)


def only_tunable_params(model):
    requires_grad = {k: v.requires_grad for k, v in model.named_parameters()}
    return {
        k: v
        for k, v in model.state_dict().items()
        if k in requires_grad and requires_grad[k]
    }


class ModifiedTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        batch_size = inputs["input_ids"].shape[0]
        labels = inputs["input_ids"]
        input_ids = torch.cat([
            torch.ones(batch_size, 1).long().to(labels.device),
            inputs["input_ids"][:, :-1],
        ], dim=1)
        logits = model(input_ids=input_ids)
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.reshape(
            -1, logits.size(-1)), labels.view(-1)
        )
        if return_outputs:
            return loss, logits
        else:
            return loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        torch.save(
            only_tunable_params(self.model),
            os.path.join(output_dir, f"checkpoint.p"),
        )

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _final_ops_before_train(self):
        pass


def data_collator(features: list) -> dict:
    return {
        "input_ids": torch.stack([
            torch.LongTensor(f["input_ids"])
            for f in features
        ])
    }


def save_tunable_parameters(model, path):
    saved_params = {
        k: v.to("cpu")
        for k, v in model.named_parameters()
        if v.requires_grad
    }
    torch.save(saved_params, path)


def main():
    finetune_args, compress_args, training_args = HfArgumentParser((
        FinetuneArguments,
        CompressArguments,
        TrainingArguments,
    )).parse_args_into_dataclasses()

    print("Setup Data")
    dataset = datasets.load_from_disk(finetune_args.dataset_path)

    print("Setup Model")
    train_config = llama_compress.TrainConfig(
        peft_mode=compress_args.peft_mode,
        num_prefix_tokens=compress_args.num_prefix_tokens,
        block_size=compress_args.block_size,
        factorized_compressor=compress_args.factorized_compressor,
        adapter_gate_mode=compress_args.adapter_gate_mode,
    )
    model = llama_compress.create_model(
        model_name=finetune_args.model_name,
        train_config=train_config,
        hf_path=finetune_args.hf_path,
        use_8bit=finetune_args.use_8bit,
    )
    model.lm_head = CastOutputToFloat(model.lm_head)
    model.gradient_checkpointing_enable()
    # model.enable_input_require_grads()

    print("Train")
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=data_collator,
    )
    trainer.train()
    save_tunable_parameters(model, os.path.join(training_args.output_dir, "params.p"))


if __name__ == "__main__":
    main()
