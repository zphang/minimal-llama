import dataclasses
import tqdm.auto as tqdm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import proj_shared.io_utils as io_utils
from minimal_llama.pref.llama_simple2 import (
    LLaMAConfig,
    LLAMA_CONFIG_DICT,
    LLaMAInnerModel,
    NoInitLinear,
    create_rope_embed_ids,
    convert_mask_to_soft_mask,
    create_attention_mask,
)


@dataclasses.dataclass
class GistLLaMAConfig:
    num_gist_tokens: int = 20
    mask_format: str = "prefix_sees_all"


class LLaMAModel(nn.Module):
    def __init__(self, config: LLaMAConfig, gist_config: GistLLaMAConfig):
        super().__init__()
        self.config = config
        self.gist_config = gist_config
        self.model = LLaMAInnerModel(
            dataclasses.replace(config, vocab_size=config.vocab_size + gist_config.num_gist_tokens))
        self.lm_head = NoInitLinear(config.dim, config.vocab_size, bias=False, dtype=config.dtype)

    def forward(self,
                input_ids,
                attention_mask):
        """Forward pass (with full decode sequence, intended for training or loss-scoring)

        :param input_ids: [batch_size, seq_len]
        :param attention_mask: [batch_size, seq_len, seq_len]
        :return: logits [batch_size, seq_len]
        """
        # 1) Create masks
        # decoder mask
        attention_mask = convert_mask_to_soft_mask(attention_mask[:, None, :, :], dtype=self.config.dtype)
        # attention_mask = create_attention_mask(input_ids=input_ids, dtype=self.config.dtype)
        # [batch_size, num_heads=1, q_len=seq_len, kv_len=seq_len]
        rope_embed_ids = create_rope_embed_ids(input_ids=input_ids)
        cos, sin = self.get_cos_sin(rope_embed_ids)
        cos, sin = cos[:, None, :, :], sin[:, None, :, :]

        # 2) Forward pass
        # [batch_size, seq_len, hidden_dim]
        model_out = self.model(
            input_ids,
            attention_mask=attention_mask,
            cos=cos, sin=sin,
        )
        # [batch_size, seq_len, vocab_size]
        logits = self.lm_head(model_out["hidden_states"])
        return logits

    def get_cos_sin(self, rope_embed_ids):
        cos = F.embedding(
            rope_embed_ids,
            self.model.layers[0].self_attn.rotary_emb.cos_cached[0, 0].to(rope_embed_ids.device)
        ).to(self.config.dtype)
        sin = F.embedding(
            rope_embed_ids,
            self.model.layers[0].self_attn.rotary_emb.sin_cached[0, 0].to(rope_embed_ids.device)
        ).to(self.config.dtype)
        return cos, sin

    def gradient_checkpointing_enable(self):
        self.config.gradient_checkpointing = True

    def enable_input_require_grads(self):
        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)
        self.model.embed_tokens.register_forward_hook(make_inputs_require_grads)


def create_model(model_name, hf_path, num_gist_tokens, device=None, dtype=torch.float16):
    config = LLAMA_CONFIG_DICT[model_name]
    config = dataclasses.replace(config, dtype=dtype)
    weight_map = io_utils.read_json(os.path.join(hf_path, "pytorch_model.bin.index.json"))["weight_map"]
    filename_list = sorted(list(set(weight_map.values())))
    if device is None:
        device = torch.device("cuda:{}".format(os.environ.get("LOCAL_RANK", 0)))
    # noinspection PyUnresolvedReferences
    if dtype == torch.float16:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = LLaMAModel(config=config, gist_config=GistLLaMAConfig(num_gist_tokens=num_gist_tokens)).to(device)
        torch.set_default_tensor_type(torch.FloatTensor)
    else:
        model = LLaMAModel(config=config, gist_config=GistLLaMAConfig(num_gist_tokens=num_gist_tokens)).to(device)
    if model_name == "debug":
        return model
    state_keys = set(model.state_dict())
    for filename in tqdm.tqdm(filename_list):
        loaded = torch.load(os.path.join(hf_path, filename), map_location="cpu")
        if "model.embed_tokens.weight" in loaded:
            loaded["model.embed_tokens.weight"] = torch.cat([
                loaded["model.embed_tokens.weight"],
                loaded["model.embed_tokens.weight"][:num_gist_tokens].clone(),
            ], dim=0)
        model.load_state_dict(loaded, strict=False)
        for k in loaded:
            state_keys.remove(k)
    assert not state_keys, "Some keys were not loaded: {}".format(state_keys)

    # # DEBUG
    # model.model.embed_tokens = model.model.embed_tokens.to(torch.float32)
    # for n, p in model.named_parameters():
    #     if "model.embed_tokens" in n:
    #         p.requires_grad = True
    #     else:
    #         p.requires_grad = False
    return model
