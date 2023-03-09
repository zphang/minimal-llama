import argparse
import os
import json
import math
import tqdm.auto as tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import datasets
import transformers
from finetune_pp import RepeatingLoader, DatasetDataset
from finetune_peft import get_peft_config, CastOutputToFloat, save_tunable_parameters
from peft import (
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    TaskType,
)


def read_json(path):
    # Manually open because .splitlines is different from iterating over lines
    with open(path, "r") as f:
        return json.load(f)


def model_forward(model, inputs):
    h = inputs
    h = h.to(model.base_model.model.model.embed_tokens.weight.device)
    h = model.base_model.model.model.embed_tokens(h)
    for layer in model.base_model.model.model.layers:
        h = h.to(layer.input_layernorm.weight.device)
        h = layer(h)[0]
    h = h.to(model.base_model.model.model.norm.weight.device)
    h = model.base_model.model.model.norm(h)
    h = model.base_model.model.lm_head(h)
    return h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_steps", type=int)
    parser.add_argument("--save_interval", type=int)

    parser.add_argument("--peft_mode", type=str, default="lora")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--num_virtual_tokens", type=int, default=32)
    parser.add_argument("--mapping_hidden_dim", type=int, default=1024)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print("Setup Data")
    dataset = datasets.load_from_disk(args.dataset_path)
    dataloader = RepeatingLoader(torch.utils.data.DataLoader(
        DatasetDataset(dataset),
        batch_size=args.batch_size,
        shuffle=True
    ))

    print("Setup Model")
    # The auto/balance balancing strategy doesn't seem to work correctly,
    # so we manually compute the mappings.
    num_layers = read_json(os.path.join(args.model_path, "config.json"))["num_hidden_layers"]
    device_ids = list(range(torch.cuda.device_count()))
    device_map = {
        "model.embed_tokens": device_ids[0],
        "model.norm.weight": device_ids[-1],
        "lm_head": device_ids[-1],
    }
    allocations = [
        device_ids[i] for i in
        sorted(list(range(len(device_ids))) * math.ceil(num_layers / len(device_ids)))
    ]
    for layer_i, device_id in enumerate(allocations):
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
        args.model_path,
        load_in_8bit=True,
        device_map=device_map,
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.lm_head = CastOutputToFloat(model.lm_head)
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    print("Setup PEFT")
    peft_config = get_peft_config(peft_args=args)
    model = get_peft_model(model, peft_config)

    print("Setup optimizer")
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Train (maybe can replace with Trainer? I think Trainer might mess up the device mappings though.)
    print("Start training")
    generator = iter(dataloader)
    for step in tqdm.trange(args.num_train_steps):
        input_ids, labels = next(generator)
        logits = model_forward(model, input_ids)
        loss = F.cross_entropy(
            logits.view(-1, model.config.vocab_size),
            labels.view(-1).to(logits.device),
        )
        loss.backward()
        opt.step()

        actual_step = step + 1
        if actual_step % args.gradient_accumulation_steps == 0:
            opt.zero_grad()

        if actual_step % args.save_interval == 0 and actual_step != args.num_train_steps:
            save_tunable_parameters(model, os.path.join(args.save_dir, f"params-{actual_step:06d}.p"))
    save_tunable_parameters(model, os.path.join(args.save_dir, "params-last.p"))


if __name__ == "__main__":
    main()
