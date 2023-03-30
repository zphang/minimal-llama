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
from peft import (
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    TaskType,
)


@dataclass
class FinetuneArguments:
    dataset_path: str = field()
    model_path: str = field()


@dataclass
class PEFTArguments:
    peft_mode: str = field(default="lora")
    lora_rank: int = field(default=8)
    num_virtual_tokens: int = field(default=32)  # Used for prompt tuning, prefix tuning and p-tuning
    mapping_hidden_dim: int = field(default=1024)


def get_peft_config(peft_args: PEFTArguments):
    if peft_args.peft_mode == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=peft_args.lora_rank,
            lora_alpha=32, lora_dropout=0.1
        )
    elif peft_args.peft_mode == "prefix":
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=peft_args.num_virtual_tokens,
            encoder_hidden_size=peft_args.mapping_hidden_dim,
            prefix_projection=True,
        )
    elif peft_args.peft_mode == "ptuning":
        peft_config = PromptEncoderConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=peft_args.num_virtual_tokens,
            encoder_hidden_size=peft_args.mapping_hidden_dim,
        )
    elif peft_args.peft_mode == "prompt":
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=peft_args.num_virtual_tokens,
        )
    else:
        raise KeyError(peft_args.peft_mode)
    return peft_config


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
        return model(
            input_ids=inputs["input_ids"],
            attention_mask=torch.ones_like(inputs["input_ids"]),
            labels=inputs["input_ids"],  # HF model does the slicing for us
        ).loss

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
    finetune_args, peft_args, training_args = HfArgumentParser((
        FinetuneArguments,
        PEFTArguments,
        TrainingArguments,
    )).parse_args_into_dataclasses()

    print("Setup Data")
    dataset = datasets.load_from_disk(finetune_args.dataset_path)

    print("Setup Model")

    import json
    def read_json(path):
        with open(path, "r") as f:
            return json.load(f)

    device_id = int(os.environ["LOCAL_RANK"])
    num_layers = read_json(os.path.join(finetune_args.model_path, "config.json"))["num_hidden_layers"]
    device_map = {
        "model.embed_tokens": device_id,
        "model.norm.weight": device_id,
        "lm_head": device_id
    }
    for layer_i in range(num_layers):
        device_map[f"model.layers.{layer_i}.self_attn.q_proj.weight"] = device_id
        device_map[f"model.layers.{layer_i}.self_attn.k_proj.weight"] = device_id
        device_map[f"model.layers.{layer_i}.self_attn.v_proj.weight"] = device_id
        device_map[f"model.layers.{layer_i}.self_attn.o_proj.weight"] = device_id
        device_map[f"model.layers.{layer_i}.mlp.gate_proj.weight"] = device_id
        device_map[f"model.layers.{layer_i}.mlp.down_proj.weight"] = device_id
        device_map[f"model.layers.{layer_i}.mlp.up_proj.weight"] = device_id
        device_map[f"model.layers.{layer_i}.input_layernorm.weight"] = device_id
        device_map[f"model.layers.{layer_i}.post_attention_layernorm.weight"] = device_id
        device_map[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = device_id


    model = transformers.LLaMAForCausalLM.from_pretrained(
        finetune_args.model_path,
        load_in_8bit=True,
        device_map=device_map,
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.lm_head = CastOutputToFloat(model.lm_head)
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    print("Setup PEFT")
    peft_config = get_peft_config(peft_args=peft_args)
    model = get_peft_model(model, peft_config)
    model.to(f"cuda:{device_id}")

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
