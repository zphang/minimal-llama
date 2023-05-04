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
    create_generation_attention_mask,
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

    def init_kv_cache(self, input_ids):
        # noinspection GrazieInspection
        """Initialize KV cache for decoding.

        A KV cache consists of a list of dicts (one per layer):
            dict(
              key = [batch_size, num_heads, kv_seq_len=0, head_dim]
              value = [batch_size, num_heads, kv_seq_len=0, head_dim]
            )

        :param input_ids: [batch_size, dec_seq_len]
        :return: 0-length kv_cache
        """
        kv_cache = []
        batch_size = input_ids.shape[0]
        num_heads = self.config.n_heads
        head_dim = self.config.head_dim
        for layer in self.model.layers:
            device = layer.input_layernorm.weight.device
            kv_cache.append({
                "key": torch.zeros([batch_size, num_heads, 0, head_dim]).to(
                    device=device, dtype=self.config.dtype),
                "value": torch.zeros([batch_size, num_heads, 0, head_dim]).to(
                    device=device, dtype=self.config.dtype),
            })
        return kv_cache

    def generate(self,
                 input_ids,
                 initial_attention_mask,
                 generation_length: int = 20,
                 return_output_only=True):
        """Generate tokens with efficient caching of KV.

        TODO: Add stopping conditions
        TODO: Add sampling capabilities

        :param input_ids: [batch_size, enc_seq_len]
        :param initial_attention_mask:
        :param generation_length: int
        :return: [batch_size, generation_length]
        """
        original_input_ids = input_ids
        batch_size, seq_len = input_ids.shape
        # noinspection PyUnresolvedReferences
        num_valid_tokens = (input_ids != self.config.pad_token_id).long().sum(dim=1)

        # 1) Setup
        if input_ids is None:
            # [batch_size, dec_seq_len=1]
            input_ids = torch.LongTensor(
                [[self.config.pad_token_id]] * batch_size
            ).to(self.lm_head.weights.device)
        # See: init_kv_cache. list[dict]
        kv_cache = self.init_kv_cache(input_ids)
        generated_token_ids_list = [original_input_ids]
        total_seq_len = seq_len

        # 2) First encoding
        # [batch_size=1, num_heads=1, q_len=1, kv_len=1]
        attention_mask = convert_mask_to_soft_mask(
            initial_attention_mask[:, None, :, :], dtype=self.config.dtype)
        # dict(
        #   hidden_states = [batch_size, dec_seq_len=decode_step+1, hidden_dim]
        #   kv_cache = list[dict(
        #     key = [batch_size, num_heads, kv_seq_len=decode_step+1, head_dim]
        #     value = [batch_size, num_heads, kv_seq_len=decode_step+1, head_dim]
        #   )]
        # )
        rope_embed_ids = create_rope_embed_ids(input_ids=input_ids)
        cos, sin = self.get_cos_sin(rope_embed_ids)
        cos, sin = cos[:, None, :, :], sin[:, None, :, :]
        model_out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cos=cos, sin=sin,
            kv_cache=kv_cache,
        )
        logits = self.lm_head(model_out["hidden_states"])
        kv_cache = model_out["kv_cache"]
        generated_token_ids = logits.argmax(-1)[
            torch.arange(batch_size, dtype=torch.long, device=input_ids.device),
            num_valid_tokens-1,
        ][:, None]
        generated_token_ids_list.append(generated_token_ids)
        input_ids = generated_token_ids

        # 3) Subsequent steps
        for decode_step in range(generation_length-1):
            num_valid_tokens += 1
            total_seq_len += 1
            # [batch_size=1, num_heads=1, q_len=1, kv_len=1]
            attention_mask = convert_mask_to_soft_mask(create_generation_attention_mask(
                batch_size=batch_size,
                seq_len=total_seq_len,
                num_valid_tokens=num_valid_tokens,
                device=input_ids.device,
            ), dtype=self.config.dtype)
            # dict(
            #   hidden_states = [batch_size, dec_seq_len=decode_step+1, hidden_dim]
            #   kv_cache = list[dict(
            #     key = [batch_size, num_heads, kv_seq_len=decode_step+1, head_dim]
            #     value = [batch_size, num_heads, kv_seq_len=decode_step+1, head_dim]
            #   )]
            # )
            rope_embed_ids = create_rope_embed_ids(input_ids=input_ids) + num_valid_tokens[:, None]
            cos, sin = self.get_cos_sin(rope_embed_ids)
            cos, sin = cos[:, None, :, :], sin[:, None, :, :]
            model_out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
                cos=cos, sin=sin,
            )
            # [batch_size, dec_seq_len=1, vocab_size]
            logits = self.lm_head(model_out["hidden_states"])
            kv_cache = model_out["kv_cache"]
            # [batch_size, dec_seq_len=1]
            generated_token_ids = logits.argmax(-1)[:, -1:]
            generated_token_ids_list.append(generated_token_ids)
            input_ids = generated_token_ids
        output = torch.cat(generated_token_ids_list, dim=1)
        if return_output_only:
            output = output[:, seq_len:]
        return output

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

    return model
