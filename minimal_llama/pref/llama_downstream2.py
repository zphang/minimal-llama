import dataclasses

import math
import tqdm.auto as tqdm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb
from accelerate import init_empty_weights
from typing import Union

import proj_shared.io_utils as io_utils
from transformers.utils.bitsandbytes import set_module_8bit_tensor_to_device
import minimal_llama.pref.peft as peft
from minimal_llama.pref.llama_compress import (
    RMSNorm,
    Compressor,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    create_attention_mask,
    create_rope_embed_ids,
    zeros_like,
    NoInitLinear,
    NoInit8bitLinear,
    convert_mask_to_soft_mask,
    create_generation_attention_mask,
    shift_kv_cache_right,
    check_nan,
)


@dataclasses.dataclass
class LLaMAConfig:
    dim: int
    n_layers: int
    n_heads: int
    vocab_size: int = 32000
    max_seq_length: int = 2048
    dtype = torch.float16
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    use_8bit: bool = False

    @property
    def head_dim(self):
        return self.dim // self.n_heads


LLAMA_7B_CONFIG = LLaMAConfig(
    dim=4096,
    n_layers=32,
    n_heads=32,
)

LLAMA_CONFIG_DICT = {
    "7b": LLAMA_7B_CONFIG,
}


FORWARD_PEFT = "forward_peft"
FORWARD_COMPRESS = "forward_compress"

PEFT_PREFIX = "prefix"
PEFT_PREFIX_ADAPTER = "prefix_adapter"
PEFT_NO = "nothing"


@dataclasses.dataclass
class DownstreamConfig:
    peft_mode: str
    num_prefix_tokens: int = None
    block_size: int = 64
    factorized_compressor: bool = True
    adapter_gate_mode: str = "fixed"
    downstream_only: bool = False

    def check(self):
        assert self.peft_mode in (
            PEFT_PREFIX, PEFT_PREFIX_ADAPTER,
        )


class LLaMAModel(nn.Module):
    def __init__(self, config: LLaMAConfig, downstream_config: DownstreamConfig):
        super().__init__()
        self.config = config
        self.downstream_config = downstream_config
        self.model = LLaMAInnerModel(config, downstream_config=downstream_config)
        self.lm_head = NoInitLinear(config.dim, config.vocab_size, bias=False, dtype=config.dtype)

    def forward(self,
                input_ids,
                peft_params=None,
                forward_mode=FORWARD_PEFT):
        if forward_mode == FORWARD_PEFT:
            return self.peft_forward(input_ids, peft_params=peft_params)
        elif forward_mode == FORWARD_COMPRESS:
            return self.compress_forward(input_ids)

    def compress_forward(self, input_ids):
        attention_mask = create_attention_mask(input_ids=input_ids, dtype=self.config.dtype)
        rope_embed_ids = create_rope_embed_ids(input_ids=input_ids)
        cos, sin = self.get_cos_sin(rope_embed_ids)
        cos, sin = cos[:, None, :, :], sin[:, None, :, :]

        compress_mask = convert_mask_to_soft_mask(input_ids != self.config.pad_token_id, dtype=self.config.dtype)

        # 2) Forward pass
        # [batch_size, seq_len, hidden_dim]
        model_out = self.model(
            input_ids,
            attention_mask=attention_mask,
            cos=cos, sin=sin,
            forward_mode=FORWARD_COMPRESS,
            compress_mask=compress_mask,
        )
        return model_out

    def peft_forward(self,
                     input_ids,
                     peft_params=None):
        """Forward pass (with full decode sequence, intended for training or loss-scoring)

        :param input_ids: [batch_size, seq_len]
        :param peft_params
        :return: logits [batch_size, seq_len]
        """
        # 1) Create masks
        # decoder mask
        # [batch_size, num_heads=1, q_len=seq_len, kv_len=seq_len]
        attention_mask = create_attention_mask(input_ids=input_ids, dtype=self.config.dtype)
        if self.downstream_config.peft_mode == peft.PEFT_PREFIX:
            attention_mask = torch.cat([
                zeros_like([1, 1, input_ids.shape[1], self.downstream_config.num_prefix_tokens], tensor=attention_mask),
                attention_mask,
            ], dim=3)
        rope_embed_ids = create_rope_embed_ids(input_ids=input_ids)
        cos, sin = self.get_cos_sin(rope_embed_ids)
        cos, sin = cos[:, None, :, :], sin[:, None, :, :]

        # 1.5) prep
        if self.downstream_config.peft_mode == peft.PEFT_PREFIX:
            kv_cache = self.create_prefix_kv_cache(peft_params)
        else:
            kv_cache = None

        # 2) Forward pass
        # [batch_size, seq_len, hidden_dim]
        model_out = self.model(
            input_ids,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            cos=cos, sin=sin,
            forward_mode=FORWARD_PEFT,
        )
        # [batch_size, seq_len, vocab_size]
        logits = self.lm_head(model_out["hidden_states"])
        return logits

    @classmethod
    def create_prefix_kv_cache(cls, peft_params):
        # noinspection GrazieInspection
        """Initialize KV cache from prefixes.

        Used for decoder in both forward pass (train) and decoding

        A KV cache consists of a list of dicts (one per layer):
            dict(
              key = [batch_size, num_heads, kv_seq_len=num_prefix_tokens, head_dim]
              value = [batch_size, num_heads, kv_seq_len=num_prefix_tokens, head_dim]
            )

        :param peft_params:
        :return: kv_cache
        """
        return peft_params

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
                "key": torch.zeros([batch_size, num_heads, 0, head_dim]).to(device),
                "value": torch.zeros([batch_size, num_heads, 0, head_dim]).to(device),
            })
        return kv_cache

    def generate(self, input_ids, generation_length: int = 20, peft_params=None):
        """Generate tokens with efficient caching of KV.

        TODO: Add stopping conditions
        TODO: Add sampling capabilities

        :param input_ids: [batch_size, enc_seq_len]
        :param generation_length: int
        :param peft_params:
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
        if self.downstream_config.peft_mode == peft.PEFT_PREFIX:
            kv_cache = self.create_prefix_kv_cache(peft_params)
            num_valid_kv_cache = num_valid_tokens + self.downstream_config.num_prefix_tokens
        else:
            kv_cache = self.init_kv_cache(input_ids)
            num_valid_kv_cache = num_valid_tokens
        generated_token_ids_list = [original_input_ids]
        total_seq_len = seq_len

        # 2) First encoding
        # [batch_size=1, num_heads=1, q_len=1, kv_len=1]
        attention_mask = create_attention_mask(input_ids=input_ids, dtype=self.config.dtype)
        # dict(
        #   hidden_states = [batch_size, dec_seq_len=decode_step+1, hidden_dim]
        #   kv_cache = list[dict(
        #     key = [batch_size, num_heads, kv_seq_len=decode_step+1, head_dim]
        #     value = [batch_size, num_heads, kv_seq_len=decode_step+1, head_dim]
        #   )]
        # )
        if self.downstream_config.peft_mode == peft.PEFT_PREFIX:
            num_prefix_tokens = self.downstream_config.num_prefix_tokens
            total_seq_len += num_prefix_tokens
            # [batch_size, num_heads=1, q_len=seq_len, kv_len=num_prefix_tokens + dec_seq_len]
            attention_mask = torch.cat([
                zeros_like([1, 1, input_ids.shape[1], num_prefix_tokens], tensor=attention_mask),
                attention_mask,
            ], dim=3)

        rope_embed_ids = create_rope_embed_ids(input_ids=input_ids)
        cos, sin = self.get_cos_sin(rope_embed_ids)
        cos, sin = cos[:, None, :, :], sin[:, None, :, :]
        model_out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cos=cos, sin=sin,
            kv_cache=kv_cache,
            peft_params=peft_params,
        )
        logits = self.lm_head(model_out["hidden_states"])
        kv_cache = model_out["kv_cache"]
        generated_token_ids = logits.argmax(-1)[
            torch.arange(batch_size, dtype=torch.long, device=input_ids.device),
            num_valid_tokens-1,
        ][:, None]
        generated_token_ids_list.append(generated_token_ids)
        input_ids = generated_token_ids

        # 2.1 shift KV cache
        print(num_valid_kv_cache)
        for layer_kv_cache in kv_cache:
            layer_kv_cache["key"] = shift_kv_cache_right(
                layer_kv_cache["key"], num_valid_tokens=num_valid_kv_cache)
            layer_kv_cache["value"] = shift_kv_cache_right(
                layer_kv_cache["value"], num_valid_tokens=num_valid_kv_cache)

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
                peft_params=peft_params,
            )
            # [batch_size, dec_seq_len=1, vocab_size]
            logits = self.lm_head(model_out["hidden_states"])
            kv_cache = model_out["kv_cache"]
            # [batch_size, dec_seq_len=1]
            generated_token_ids = logits.argmax(-1)[:, -1:]
            generated_token_ids_list.append(generated_token_ids)
            input_ids = generated_token_ids
        return torch.cat(generated_token_ids_list, dim=1)

    def get_cos_sin(self, rope_embed_ids):
        cos = F.embedding(
            rope_embed_ids,
            self.model.layers[0].self_attn.rotary_emb.cos_cached[0, 0]
        ).to(self.config.dtype)
        sin = F.embedding(
            rope_embed_ids,
            self.model.layers[0].self_attn.rotary_emb.sin_cached[0, 0]
        ).to(self.config.dtype)
        return cos, sin


class LLaMAInnerModel(nn.Module):
    def __init__(self, config: LLaMAConfig, downstream_config: DownstreamConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim, dtype=config.dtype)
        self.layers = nn.ModuleList([
            LLaMALayer(config=config, downstream_config=downstream_config)
            for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(dim=config.dim)

    def forward(self,
                input_ids,
                attention_mask,
                cos, sin,
                kv_cache=None,
                peft_params=None,
                forward_mode=FORWARD_PEFT,
                compress_mask=None):
        """
        :param input_ids: [batch_size, seq_len]
        :param attention_mask: [batch_size=1, num_heads=1, seq_len, seq_len]
        :param cos:
        :param sin:
        :param kv_cache: See init_kv_cache.
            We use the presence of kv_cache to determine if we're generating
        :param peft_params
        :param forward_mode
        :param compress_mask
        """
        hidden_states = self.embed_tokens(input_ids)

        new_kv_cache = []
        peft_cache = []
        for layer_i, layer in enumerate(self.layers):
            if kv_cache:
                # dict(
                #   key = [batch_size, num_heads, kv_seq_len=decode_step+1, head_dim]
                #   value = [batch_size, num_heads, kv_seq_len=decode_step+1, head_dim]
                # )
                layer_kv_cache = kv_cache[layer_i]
            else:
                layer_kv_cache = None

            layer_out = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                kv_cache=layer_kv_cache,
                cos=cos, sin=sin,
                forward_mode=forward_mode,
                peft_params=peft_params[layer_i] if peft_params else None,
                compress_mask=compress_mask,
            )
            hidden_states = layer_out["hidden_states"]
            if kv_cache:
                new_kv_cache.append(layer_out["kv_cache"])
            if forward_mode == FORWARD_COMPRESS:
                peft_cache.append(layer_out["peft"])

        if forward_mode == FORWARD_COMPRESS:
            return peft_cache
        hidden_states = self.norm(hidden_states)
        output = {"hidden_states": hidden_states}
        if kv_cache:
            output["kv_cache"] = new_kv_cache
        return output


class LLaMALayer(nn.Module):
    def __init__(self, config: LLaMAConfig, downstream_config: DownstreamConfig):
        super().__init__()
        self.config = config
        self.self_attn = Attention(config=config, downstream_config=downstream_config)
        self.mlp = MLP(config=config)
        self.input_layernorm = RMSNorm(dim=config.dim, dtype=config.dtype)
        self.post_attention_layernorm = RMSNorm(dim=config.dim, dtype=config.dtype)

    def forward(
        self,
        hidden_states,
        attention_mask,
        cos, sin,
        kv_cache=None,
        forward_mode=FORWARD_PEFT,
        peft_params=None,
        compress_mask=None,
    ):
        # 1) Self-attention
        # [batch_size, seq_len, hidden_dim]
        normed_hidden_states = self.input_layernorm(hidden_states)
        # dict(
        #   attn_output = [batch_size, seq_len, hidden_dim]
        #   kv_cache = dict(
        #     key = [batch_size, num_heads, kv_seq_len, head_dim]
        #     value = [batch_size, num_heads, kv_seq_len, head_dim]
        #   )
        # )
        check_nan(normed_hidden_states)
        raw_self_attn_output = self.self_attn(
            hidden_states=normed_hidden_states,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            cos=cos, sin=sin,
            forward_mode=forward_mode,
            peft_params=peft_params,
            compress_mask=compress_mask,
        )
        # [batch_size, seq_len, hidden_dim]
        hidden_states = hidden_states + raw_self_attn_output["attn_output"]
        check_nan(hidden_states)
        # 2) FFN
        # [batch_size, seq_len, hidden_dim]
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        check_nan(hidden_states)
        output = {"hidden_states": hidden_states}
        if kv_cache:
            output["kv_cache"] = raw_self_attn_output["kv_cache"]
        if forward_mode == FORWARD_COMPRESS:
            output["peft"] = raw_self_attn_output["peft"]
        return output


class MLP(nn.Module):
    def __init__(
        self,
        config: LLaMAConfig,
        multiple_of: int = 256,
    ):
        super().__init__()
        dim = config.dim
        hidden_dim = 4 * dim
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        if config.use_8bit:
            self.gate_proj = NoInit8bitLinear(dim, hidden_dim, bias=False, threshold=6.0, has_fp16_weights=False)
            self.up_proj = NoInit8bitLinear(dim, hidden_dim, bias=False, threshold=6.0, has_fp16_weights=False)
            self.down_proj = NoInit8bitLinear(hidden_dim, dim, bias=False, threshold=6.0, has_fp16_weights=False)
        else:
            self.gate_proj = NoInitLinear(dim, hidden_dim, bias=False, dtype=config.dtype)
            self.up_proj = NoInitLinear(dim, hidden_dim, bias=False, dtype=config.dtype)
            self.down_proj = NoInitLinear(hidden_dim, dim, bias=False, dtype=config.dtype)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    def __init__(self, config: LLaMAConfig, downstream_config: DownstreamConfig):
        super().__init__()
        self.config = config
        self.downstream_config = downstream_config
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads

        if config.use_8bit:
            self.q_proj = NoInit8bitLinear(config.dim, config.dim, bias=False, threshold=6.0, has_fp16_weights=False)
            self.k_proj = NoInit8bitLinear(config.dim, config.dim, bias=False, threshold=6.0, has_fp16_weights=False)
            self.v_proj = NoInit8bitLinear(config.dim, config.dim, bias=False, threshold=6.0, has_fp16_weights=False)
            self.o_proj = NoInit8bitLinear(config.dim, config.dim, bias=False, threshold=6.0, has_fp16_weights=False)
        else:
            self.q_proj = NoInitLinear(config.dim, config.dim, bias=False, dtype=config.dtype)
            self.k_proj = NoInitLinear(config.dim, config.dim, bias=False, dtype=config.dtype)
            self.v_proj = NoInitLinear(config.dim, config.dim, bias=False, dtype=config.dtype)
            self.o_proj = NoInitLinear(config.dim, config.dim, bias=False, dtype=config.dtype)
        self.rotary_emb = RotaryEmbedding(dim=self.head_dim)

        if not self.downstream_config.downstream_only:
            if self.downstream_config.peft_mode in (PEFT_PREFIX, PEFT_PREFIX_ADAPTER):
                # noinspection PyTypeChecker
                self.k_compressor = Compressor(config=config, train_config=downstream_config)
                # noinspection PyTypeChecker
                self.v_compressor = Compressor(config=config, train_config=downstream_config)
            elif self.downstream_config.peft_mode == PEFT_NO:
                pass
            else:
                raise KeyError(self.downstream_config.peft_mode)
            if self.downstream_config.peft_mode == PEFT_PREFIX_ADAPTER:
                if self.downstream_config.adapter_gate_mode == "fixed":
                    self.adapter_gates = nn.Parameter(torch.zeros(self.n_heads))
                else:
                    raise KeyError(self.downstream_config.adapter_gate_mode)

    def forward(self, hidden_states, attention_mask, cos, sin, kv_cache=None,
                forward_mode=FORWARD_PEFT, peft_params=None, compress_mask=None):
        """
        precomputed_kv_hidden_states is for init (pre-compute KV activations, e.g. for added prefixes)
        kv_cache is for generation (cached past KV)
        """
        batch_size, q_seq_len, hidden_dim = hidden_states.size()

        # (batch_size, num_heads, q_seq_len, head_dim)
        query_states = self.q_proj(hidden_states).view(
            batch_size, q_seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(
            batch_size, q_seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            batch_size, q_seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos=cos, sin=sin)
        if kv_cache:
            # Generate mode
            key_states = torch.cat([kv_cache["key"], key_states], dim=2)
            value_states = torch.cat([kv_cache["value"], value_states], dim=2)
        elif forward_mode == FORWARD_PEFT and self.downstream_config.peft_mode == PEFT_PREFIX:
            # Forward Mode (no cache=not generation), PEFT mode and PEFT prefix
            key_states = torch.cat([peft_params["key"], key_states], dim=2)
            value_states = torch.cat([peft_params["value"], value_states], dim=2)

        scores = torch.matmul(
            query_states, key_states.transpose(3, 2).type_as(query_states) / math.sqrt(self.head_dim)
        )
        scores += attention_mask

        # (batch_size, num_heads, q_seq_len, kv_seq_len)
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        # (batch_size, num_heads, q_seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, value_states.type_as(query_states))

        if forward_mode == FORWARD_PEFT and self.downstream_config.peft_mode == PEFT_PREFIX_ADAPTER:
            prefix_k, prefix_v = peft_params["key"], peft_params["value"]
            gate = peft_params["gate"].view(1, -1, 1, 1)
            # [batch_size, num_blocks, num_heads, block_size, num_prefix_tokens]
            adapter_scores = torch.matmul(
                query_states,
                prefix_k.transpose(-1, -2).type_as(query_states) / math.sqrt(self.head_dim)
            )
            # [batch_size, num_blocks, num_heads, block_size, num_prefix_tokens]
            adapter_attn_weights = F.softmax(adapter_scores.float(), dim=-1).type_as(adapter_scores)
            # [batch_size=1, num_blocks=1, num_heads, block_size=1, head_dim=1]
            # [batch_size, num_blocks, num_heads, block_size, head_dim]
            adapter_attn_output = gate * torch.matmul(
                adapter_attn_weights,
                prefix_v.type_as(query_states),
            )
            # [batch_size, num_blocks, num_heads, block_size, head_dim]
            attn_output += adapter_attn_output
        # (batch_size, q_seq_len, hidden_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, q_seq_len, hidden_dim,
        )
        attn_output = self.o_proj(attn_output)
        check_nan(attn_output)
        if forward_mode == FORWARD_PEFT:
            if kv_cache:
                new_kv_cache = {"key": key_states, "value": value_states}
                return {"attn_output": attn_output, "kv_cache": new_kv_cache}
            else:
                return {"attn_output": attn_output}
        elif forward_mode == FORWARD_COMPRESS:
            # This is a hack
            self.k_compressor.train_config.block_size = q_seq_len
            self.v_compressor.train_config.block_size = q_seq_len

            # [batch_size, num_heads, num_prefix_tokens, head_dim]
            if self.downstream_config.peft_mode in (PEFT_PREFIX, PEFT_PREFIX_ADAPTER):
                # [:, 0] indexes out the num_blocks dimension
                peft_dict = {
                    "key": self.k_compressor(
                        hidden_states=hidden_states,
                        past_kvs=key_states,
                        compress_mask=compress_mask,
                    )[:, 0],
                    "value": self.v_compressor(
                        hidden_states=hidden_states,
                        past_kvs=value_states,
                        compress_mask=compress_mask,
                    )[:, 0],
                }
                if self.downstream_config.peft_mode == PEFT_PREFIX_ADAPTER:
                    peft_dict["gate"] = F.tanh(self.adapter_gates.to(hidden_states.dtype))

                return {
                    "attn_output": attn_output,
                    "peft": peft_dict
                }
            elif self.downstream_config.peft_mode == PEFT_NO:
                raise NotImplementedError()
            else:
                raise KeyError(self.downstream_config.peft_mode)
        else:
            raise KeyError(forward_mode)


def create_model(model_name, hf_path, downstream_config: DownstreamConfig, use_8bit=False, device=None):
    config = LLAMA_CONFIG_DICT[model_name]
    weight_map = io_utils.read_json(os.path.join(hf_path, "pytorch_model.bin.index.json"))["weight_map"]
    filename_list = sorted(list(set(weight_map.values())))
    if device is None:
        # TODO: Local rank
        device = torch.device("cuda:0")
    if use_8bit:
        config = dataclasses.replace(config, use_8bit=True)
        with init_empty_weights():
            model = LLaMAModel(config=config, downstream_config=downstream_config)

        for layer_i in range(config.n_layers):
            model.model.layers[layer_i].self_attn.rotary_emb.cos_cached = \
                model.model.layers[layer_i].self_attn.rotary_emb.cos_cached.to(device)
            model.model.layers[layer_i].self_attn.rotary_emb.sin_cached = \
                model.model.layers[layer_i].self_attn.rotary_emb.sin_cached.to(device)
            model.model.layers[layer_i].self_attn.k_compressor.to_empty(device=device)
            model.model.layers[layer_i].self_attn.k_compressor.init_weights()
            model.model.layers[layer_i].self_attn.v_compressor.to_empty(device=device)
            model.model.layers[layer_i].self_attn.v_compressor.init_weights()
            if downstream_config.peft_mode == PEFT_PREFIX_ADAPTER:
                model.model.layers[layer_i].self_attn.adapter_gates.to_empty(device=device)
                model.model.layers[layer_i].self_attn.adapter_gates.weight.zeros_()

        state_keys = set(model.state_dict())
        filename_list = sorted(list(set(weight_map.values())))
        for filename in tqdm.tqdm(filename_list):
            loaded = torch.load(os.path.join(hf_path, filename), map_location="cpu")
            for k, v in loaded.items():
                set_module_8bit_tensor_to_device(model, tensor_name=k, device=device, value=v)
                state_keys.remove(k)
                if downstream_config.factorized_compressor:
                    if "self_attn.k_proj" in k:
                        layer_i = int(k.split(".")[2])
                        # noinspection PyTypeChecker
                        model.load_state_dict({
                            f"model.layers.{layer_i}.self_attn.k_compressor.sub_k_proj.weight": v.clone(),
                        }, strict=False)
                        state_keys.remove(f"model.layers.{layer_i}.self_attn.k_compressor.sub_k_proj.weight")
                    if "self_attn.v_proj" in k:
                        layer_i = int(k.split(".")[2])
                        # noinspection PyTypeChecker
                        model.load_state_dict({
                            f"model.layers.{layer_i}.self_attn.v_compressor.sub_k_proj.weight": v.clone(),
                        }, strict=False)
                        state_keys.remove(f"model.layers.{layer_i}.self_attn.v_compressor.sub_k_proj.weight")
                    if "self_attn.q_proj" in k:
                        layer_i = int(k.split(".")[2])
                        # noinspection PyTypeChecker
                        model.load_state_dict({
                            f"model.layers.{layer_i}.self_attn.k_compressor.sub_q_proj.weight": v.clone(),
                            f"model.layers.{layer_i}.self_attn.v_compressor.sub_q_proj.weight": v.clone(),
                        }, strict=False)
                        state_keys.remove(f"model.layers.{layer_i}.self_attn.k_compressor.sub_q_proj.weight")
                        state_keys.remove(f"model.layers.{layer_i}.self_attn.v_compressor.sub_q_proj.weight")

        # assert not state_keys
    else:
        # noinspection PyUnresolvedReferences
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = LLaMAModel(config=config, downstream_config=downstream_config).cuda()
        torch.set_default_tensor_type(torch.FloatTensor)
        state_keys = set(model.state_dict())
        for filename in tqdm.tqdm(filename_list):
            loaded = torch.load(os.path.join(hf_path, filename), map_location="cpu")
            if downstream_config.factorized_compressor:
                for k, v in list(loaded.items()):
                    if "self_attn.k_proj" in k:
                        layer_i = int(k.split(".")[2])
                        new_k = f"model.layers.{layer_i}.self_attn.k_compressor.sub_k_proj.weight"
                        loaded[new_k] = v.clone()
                    if "self_attn.v_proj" in k:
                        layer_i = int(k.split(".")[2])
                        new_k = f"model.layers.{layer_i}.self_attn.v_compressor.sub_k_proj.weight"
                        loaded[new_k] = v.clone()
                    if "self_attn.q_proj" in k:
                        layer_i = int(k.split(".")[2])
                        new_k = f"model.layers.{layer_i}.self_attn.k_compressor.sub_q_proj.weight"
                        loaded[new_k] = v.clone()
                        new_k = f"model.layers.{layer_i}.self_attn.v_compressor.sub_q_proj.weight"
                        loaded[new_k] = v.clone()
            model.load_state_dict(loaded, strict=False)
            for k in loaded:
                state_keys.remove(k)

    print(f"Not loaded: {state_keys}")

    for n, p in model.named_parameters():
        if "_compressor" in n:
            p.requires_grad = True
        elif ".adapter_gates" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

    for layer_i in range(config.n_layers):
        model.model.layers[layer_i].self_attn.k_compressor.float()
        model.model.layers[layer_i].self_attn.v_compressor.float()
        if downstream_config.peft_mode == PEFT_PREFIX_ADAPTER:
            model.model.layers[layer_i].self_attn.adapter_gates = nn.Parameter(
                model.model.layers[layer_i].self_attn.adapter_gates.float())

    return model
