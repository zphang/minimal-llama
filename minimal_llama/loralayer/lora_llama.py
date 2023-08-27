import dataclasses

import tqdm.auto as tqdm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Union

import minimal_llama.utils.io_utils as io_utils
import minimal_llama.utils.torch_utils as torch_utils


@dataclasses.dataclass
class LLaMAConfig:
    dim: int
    n_layers: int
    n_heads: int
    vocab_size: int = 32000  # ?
    max_seq_length: int = 2048
    dtype: torch.Type = torch.bfloat16
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    gradient_checkpointing: bool = False

    use_lora: bool = True
    lora_rank: int = 16
    raw_lora_layers: str = "attn,ffn"
    raw_layer_mapping: Union[dict, str] = "single"

    @property
    def head_dim(self):
        return self.dim // self.n_heads

    @property
    def lora_layers(self):
        return self.raw_lora_layers.split(",")

    @property
    def hidden_size(self):
        return self.dim

    def to_dict(self):
        return dataclasses.asdict(self)

    @property
    def layer_mapping(self):
        if self.raw_layer_mapping == "single":
            return {
                i: 0
                for i in range(self.n_layers)
            }
        elif self.raw_layer_mapping == "all":
            return {
                i: i
                for i in range(self.n_layers)
            }
        else:
            assert isinstance(self.raw_layer_mapping, dict)
            return self.raw_layer_mapping


LLAMA_160M_CONFIG = LLaMAConfig(
    dim=768,
    n_layers=12,
    n_heads=12,
    vocab_size=50512,
)
LLAMA_7B_CONFIG = LLaMAConfig(
    dim=4096,
    n_layers=32,
    n_heads=32,
)
DEBUG_CONFIG = LLaMAConfig(
    dim=64,
    n_layers=3,
    n_heads=4,
)

LLAMA_CONFIG_DICT = {
    "160m": LLAMA_160M_CONFIG,
    "7b": LLAMA_7B_CONFIG,
    "debug": DEBUG_CONFIG,
}


class LLaMAModel(nn.Module):
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config
        self.model = LLaMAInnerModel(config)
        self.lm_head = NoInitLinear(config.dim, config.vocab_size, bias=False, dtype=config.dtype)

    def forward(self,
                input_ids,
                attention_mask=None):
        """Forward pass (with full decode sequence, intended for training or loss-scoring)

        :param input_ids: [batch_size, seq_len]
            - Always right-padded. Masks are generated based on padding tokens
        :param attention_mask
        :return: logits [batch_size, seq_len]
        """
        # 1) Create masks
        # decoder mask
        # [batch_size, num_heads=1, q_len=seq_len, kv_len=seq_len]
        rope_embed_ids = create_rope_embed_ids(input_ids=input_ids)
        cos, sin = self.get_cos_sin(rope_embed_ids)

        # 2) Forward pass
        # [batch_size, seq_len, hidden_dim]
        model_out = self.model(
            input_ids,
            cos=cos, sin=sin,
            use_kv_cache=False,
            attention_mask=attention_mask,
        )
        # [batch_size, seq_len, vocab_size]
        logits = self.lm_head(model_out["hidden_states"])
        return logits

    def init_kv_cache(self, batch_size):
        # noinspection GrazieInspection
        """Initialize an empty KV cache for decoding.

        A KV cache consists of a list of dicts (one per layer):
            dict(
              key = [batch_size, num_heads, kv_seq_len=0, head_dim]
              value = [batch_size, num_heads, kv_seq_len=0, head_dim]
            )

        :param batch_size
        :return: 0-length kv_cache
        """
        kv_cache = []
        num_heads = self.config.n_heads
        head_dim = self.config.head_dim
        for layer in self.model.layers:
            device = layer.input_layernorm.weight.device
            kv_cache.append({
                "key": torch.zeros([batch_size, num_heads, 0, head_dim]).to(device=device, dtype=self.config.dtype),
                "value": torch.zeros([batch_size, num_heads, 0, head_dim]).to(device=device, dtype=self.config.dtype),
            })
        return kv_cache

    def generate(self, input_ids, generation_length: int = 20,
                 return_output_only=True):
        """Generate tokens with efficient caching of KV.

        TODO: Add stopping conditions
        TODO: Add sampling capabilities

        :param input_ids: [batch_size, input_seq_len]
            - Always right-padded. Masks are generated based on padding tokens
        :param generation_length: int
        :param return_output_only: True = return continuation only. False = return whole sequence
        :return: [batch_size, generation_length]
        """
        original_input_ids = input_ids
        batch_size, input_seq_len = input_ids.shape
        # noinspection PyUnresolvedReferences
        num_valid_tokens = (input_ids != self.config.pad_token_id).long().sum(dim=1)
        orig_num_valid_tokens = num_valid_tokens.clone()

        # 1) Setup
        if input_ids is None:
            # [batch_size, dec_seq_len=1]
            input_ids = torch.LongTensor(
                [[self.config.pad_token_id]] * batch_size
            ).to(self.lm_head.weights.device)
        # See: init_kv_cache. list[dict]
        kv_cache = self.init_kv_cache(batch_size)
        generated_token_ids_list = [original_input_ids]
        total_seq_len = input_seq_len

        # 2) First encoding
        # [batch_size=1, num_heads=1, q_len=1, kv_len=1]
        # dict(
        #   hidden_states = [batch_size, dec_seq_len=decode_step+1, hidden_dim]
        #   kv_cache = list[dict(
        #     key = [batch_size, num_heads, kv_seq_len=decode_step+1, head_dim]
        #     value = [batch_size, num_heads, kv_seq_len=decode_step+1, head_dim]
        #   )]
        # )
        rope_embed_ids = create_rope_embed_ids(input_ids=input_ids, pad_token_id=self.config.pad_token_id)
        cos, sin = self.get_cos_sin(rope_embed_ids)
        model_out = self.model(
            input_ids=input_ids,
            cos=cos, sin=sin,
            use_kv_cache=True,
            kv_cache=kv_cache,
            num_valid_tokens=num_valid_tokens,
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

            # dict(
            #   hidden_states = [batch_size, dec_seq_len=decode_step+1, hidden_dim]
            #   kv_cache = list[dict(
            #     key = [batch_size, num_heads, kv_seq_len=decode_step+1, head_dim]
            #     value = [batch_size, num_heads, kv_seq_len=decode_step+1, head_dim]
            #   )]
            # )
            rope_embed_ids = create_rope_embed_ids(input_ids=input_ids, pad_token_id=self.config.pad_token_id)
            decoding_attention_mask = create_decoding_mask(
                orig_num_valid_tokens=orig_num_valid_tokens,
                max_seq_len=total_seq_len,
                initial_max_len=original_input_ids.shape[1]
            ).to(input_ids.device)
            rope_embed_ids += num_valid_tokens[:, None]
            cos, sin = self.get_cos_sin(rope_embed_ids)
            model_out = self.model(
                input_ids=input_ids,
                cos=cos, sin=sin,
                use_kv_cache=True,
                kv_cache=kv_cache,
                num_valid_tokens=num_valid_tokens,
                attention_mask=decoding_attention_mask,
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
            output = output[:, input_seq_len:]
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
        cos, sin = cos[:, None, :, :], sin[:, None, :, :]
        return cos, sin

    def gradient_checkpointing_enable(self):
        self.config.gradient_checkpointing = True

    def enable_input_require_grads(self):
        # noinspection PyUnusedLocal
        def make_inputs_require_grads(module, _, output):
            output.requires_grad_(True)
        self.model.embed_tokens.register_forward_hook(make_inputs_require_grads)


class LLaMAInnerModel(nn.Module):
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = NoInitEmbedding(config.vocab_size, config.dim, dtype=config.dtype)
        self.layers = nn.ModuleList([
            LLaMALayer(config=config)
            for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(dim=config.dim, dtype=config.dtype)

    def forward(
        self,
        input_ids,
        cos, sin,
        use_kv_cache=False,
        kv_cache=None,
        num_valid_tokens=None,
        attention_mask=None,
    ):
        """
        :param input_ids: [batch_size, seq_len]
        :param kv_cache: See init_kv_cache.
            We use the presence of kv_cache to determine if we're generating
        :param cos:
        :param sin:
        :param use_kv_cache: If True, we are going to maintain a kv_cache
            if kv_cache is None (i.e. the first encoding step in decoding), the kv_cache is just the
            based on the kv states.
        :param kv_cache: {"key"/"value": [batch_size, num_heads, cache_seq_len, head_dim]}
            Only used for decoding
        :param num_valid_tokens: [batch_size]
            Only used for decoding
        :param attention_mask: [batch_size, num_heads, q_len, kv_len]
        """
        assert input_ids.max() < self.config.vocab_size, input_ids.max()
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states.to(self.config.dtype)

        new_kv_cache = []
        for layer_i, layer in enumerate(self.layers):
            if kv_cache:
                # dict(
                #   key = [batch_size, num_heads, kv_seq_len=decode_step+1, head_dim]
                #   value = [batch_size, num_heads, kv_seq_len=decode_step+1, head_dim]
                # )
                layer_kv_cache = kv_cache[layer_i]
            else:
                layer_kv_cache = None

            if self.config.gradient_checkpointing:
                # noinspection PyUnresolvedReferences
                layer_out = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    cos, sin,
                    use_kv_cache,
                    layer_kv_cache,
                    num_valid_tokens,
                    attention_mask,
                    # use_reentrant=False,
                )
            else:
                layer_out = layer(
                    hidden_states=hidden_states,
                    cos=cos, sin=sin,
                    use_kv_cache=use_kv_cache,
                    kv_cache=layer_kv_cache,
                    num_valid_tokens=num_valid_tokens,
                    attention_mask=attention_mask,
                )

            hidden_states = layer_out["hidden_states"]
            if kv_cache:
                new_kv_cache.append(layer_out["kv_cache"])
        hidden_states = self.norm(hidden_states)
        output = {
            "hidden_states": hidden_states
        }
        if kv_cache:
            output["kv_cache"] = new_kv_cache
        return output


class LLaMALayer(nn.Module):
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config
        self.self_attn = Attention(config=config)
        self.mlp = MLP(config=config)
        self.input_layernorm = RMSNorm(dim=config.dim, dtype=config.dtype)
        self.post_attention_layernorm = RMSNorm(dim=config.dim, dtype=config.dtype)

    # noinspection PyUnusedLocal
    def forward(
        self,
        hidden_states,
        cos, sin,
        use_kv_cache=False,
        kv_cache=None,
        num_valid_tokens=None,
        attention_mask=None,
        offload_to_cpu=False,  # Needed for activation checkpointing? idk
    ):
        # 1) Self-attention
        # [batch_size, seq_len, hidden_dim]
        normed_hidden_states = self.input_layernorm(hidden_states).to(self.config.dtype)
        # dict(
        #   attn_output = [batch_size, seq_len, hidden_dim]
        #   kv_cache = dict(
        #     key = [batch_size, num_heads, kv_seq_len, head_dim]
        #     value = [batch_size, num_heads, kv_seq_len, head_dim]
        #   )
        # )
        torch_utils.check_nan(normed_hidden_states)
        raw_self_attn_output = self.self_attn(
            hidden_states=normed_hidden_states,
            cos=cos, sin=sin,
            use_kv_cache=use_kv_cache,
            kv_cache=kv_cache,
            attention_mask=attention_mask,
        )
        # [batch_size, seq_len, hidden_dim]
        hidden_states = hidden_states + raw_self_attn_output["attn_output"]
        torch_utils.check_nan(hidden_states)
        # 2) FFN
        # [batch_size, seq_len, hidden_dim]
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        torch_utils.check_nan(hidden_states)
        if kv_cache:
            return {
                "hidden_states": hidden_states,
                "kv_cache": raw_self_attn_output["kv_cache"],
            }
        else:
            return {
                "hidden_states": hidden_states
            }


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

        self.use_lora_gate = "ffn" in config.lora_layers
        self.use_lora_up = "ffn" in config.lora_layers
        self.use_lora_down = "ffn" in config.lora_layers

        self.gate_proj = create_linear(dim, hidden_dim, bias=False, dtype=config.dtype,
                                       use_lora=self.use_lora_gate, lora_rank=config.lora_rank)
        self.up_proj = create_linear(dim, hidden_dim, bias=False, dtype=config.dtype,
                                     use_lora=self.use_lora_up, lora_rank=config.lora_rank)
        self.down_proj = create_linear(hidden_dim, dim, bias=False, dtype=config.dtype,
                                       use_lora=self.use_lora_down, lora_rank=config.lora_rank)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, dtype=torch.float16):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def reset_parameters(self):
        pass

    def reset_actual_parameters(self, dtype: torch.dtype, device):
        self.weight = nn.Parameter(torch.ones_like(self.weight, dtype=dtype, device=device))


class Attention(nn.Module):
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads

        self.use_lora_q = "attn" in config.lora_layers
        self.use_lora_k = "attn" in config.lora_layers
        self.use_lora_v = "attn" in config.lora_layers
        self.use_lora_o = "attn" in config.lora_layers

        self.q_proj = create_linear(config.dim, config.dim, bias=False, dtype=config.dtype,
                                    use_lora=self.use_lora_q, lora_rank=config.lora_rank)
        self.k_proj = create_linear(config.dim, config.dim, bias=False, dtype=config.dtype,
                                    use_lora=self.use_lora_k, lora_rank=config.lora_rank)
        self.v_proj = create_linear(config.dim, config.dim, bias=False, dtype=config.dtype,
                                    use_lora=self.use_lora_v, lora_rank=config.lora_rank)
        self.o_proj = create_linear(config.dim, config.dim, bias=False, dtype=config.dtype,
                                    use_lora=self.use_lora_o, lora_rank=config.lora_rank)
        self.rotary_emb = RotaryEmbedding(dim=self.head_dim, max_position_embeddings=config.max_seq_length)

    def forward(self, hidden_states, cos, sin,
                use_kv_cache=False,
                kv_cache=None,
                attention_mask=None,):
        """
        :param hidden_states: [batch_size, seq_len, hidden_dim]
        :param cos:
        :param sin:
        :param use_kv_cache: If True, we are going to maintain a kv_cache
            if kv_cache is None (i.e. the first encoding step in decoding), the kv_cache is just the
            based on the kv states.
        :param kv_cache: {"key"/"value": [batch_size, num_heads, cache_seq_len, head_dim]}
            Only used for decoding
        :param attention_mask: [batch_size, num_heads, q_len, kv_len]
        :return:
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
        if use_kv_cache:
            key_states, value_states = self.append_to_kv_cache(
                kv_cache=kv_cache,
                new_key_state=key_states,
                new_value_state=value_states,
            )
        if q_seq_len == key_states.shape[2]:

            if attention_mask is None:
                # noinspection PyUnresolvedReferences
                with torch.backends.cuda.sdp_kernel(
                    enable_math=False, enable_flash=True, enable_mem_efficient=False,
                ):
                    attn_output = torch.nn.functional.scaled_dot_product_attention(
                        query=query_states,
                        key=key_states,
                        value=value_states,
                        is_causal=True,
                    )
            else:
                # noinspection PyUnresolvedReferences
                with torch.backends.cuda.sdp_kernel(
                    enable_math=True, enable_flash=True, enable_mem_efficient=True,
                ):
                    attn_output = torch.nn.functional.scaled_dot_product_attention(
                        query=query_states,
                        key=key_states,
                        value=value_states,
                        attn_mask=attention_mask,
                    )
        else:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query=query_states,
                key=key_states,
                value=value_states,
                attn_mask=attention_mask,
            )
        # (batch_size, q_seq_len, hidden_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, q_seq_len, hidden_dim,
        )
        attn_output = self.o_proj(attn_output)
        torch_utils.check_nan(attn_output)
        if use_kv_cache:
            new_kv_cache = {"key": key_states, "value": value_states}
            return {"attn_output": attn_output, "kv_cache": new_kv_cache}
        else:
            return {"attn_output": attn_output}

    @classmethod
    def append_to_kv_cache(cls, kv_cache, new_key_state, new_value_state):
        """

        :param kv_cache: {"key"/"value": [batch_size, num_heads, cache_seq_len, head_dim]}
        :param new_key_state: [batch_size, num_heads, seq_len=1, head_dim]
        :param new_value_state: [batch_size, num_heads, seq_len=1, head_dim]
        :return:
        """
        # We need to do some fancy indexing, because we are appending to a right-padded cache
        key_cache, value_cache = kv_cache["key"], kv_cache["value"]
        key_cache = torch.cat([key_cache, new_key_state], dim=2)
        value_cache = torch.cat([value_cache, new_value_state], dim=2)
        return key_cache, value_cache


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device=device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device).to(self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device).to(self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :].to(dtype=x.dtype)
            self.sin_cached = emb.sin()[None, None, :, :].to(dtype=x.dtype)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype, device=x.device),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype, device=x.device),
        )

    def reset_parameters(self):
        pass

    def reset_inv_freq(self, device):
        # noinspection PyAttributeOutsideInit
        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device=device) / self.dim))


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class NoInitLinear(nn.Linear):
    def reset_parameters(self) -> None:
        pass

    def reset_linear_parameters(self,
                                init_method: str,
                                dtype: torch.Type,
                                device,
                                n_layers: Optional[int] = None) -> None:
        self.weight = nn.Parameter(torch.empty(
            self.weight.shape,
            dtype=dtype,
            device=device,
        ))
        if init_method == "small":
            small_init(self.weight, dim=self.weight.shape[0])
        elif init_method == "wang":
            assert n_layers is not None
            wang_init(self.weight, dim=self.weight.shape[0], n_layers=n_layers)
        else:
            raise KeyError(init_method)

    def set_linear_parameters(self, weight):
        self.weight = nn.Parameter(weight)


def create_linear(in_features: int, out_features: int, bias: bool, dtype: torch.Type,
                  use_lora: bool = False, lora_rank: int = 16):
    if use_lora:
        return NoInitLoraLinear(
            in_features=in_features,
            out_features=out_features,
            rank=lora_rank,
            dtype=dtype,
        )
    else:
        return NoInitLinear(in_features, out_features, bias=bias, dtype=dtype)


class NoInitLoraLinear(NoInitLinear):

    def __init__(self,
                 in_features: int, out_features: int,
                 rank: int, alpha: int = 16,
                 device=None, dtype=None) -> None:
        super().__init__(
            in_features=in_features, out_features=out_features, bias=False,
            device=device, dtype=dtype,
        )
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.rank = rank
        self.alpha = alpha
        self.lora_a = nn.Parameter(torch.empty((rank, self.in_features), **factory_kwargs))
        self.lora_b = nn.Parameter(torch.empty((self.out_features, rank), **factory_kwargs))
        self.scaling = self.alpha / self.rank

    def forward(self, x: torch.Tensor, use_lora: bool = True) -> torch.Tensor:
        out = super().forward(x)
        if use_lora:
            return out + self.scaling * (
                F.linear(F.linear(x, self.lora_a), self.lora_b)
            )
        else:
            return out

    def reset_lora_parameters(self):
        if self.lora_b.device == torch.device("meta"):
            # meta devices are weirdly broken; no better way to move off meta device
            self.lora_b = nn.Parameter(torch.empty_like(self.lora_b.data, device=self.weight.device))
            self.lora_a = nn.Parameter(torch.empty_like(self.lora_a.data, device=self.weight.device))
        nn.init.zeros_(self.lora_b)
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))


class NoInitEmbedding(nn.Embedding):
    def reset_parameters(self) -> None:
        pass

    def reset_actual_parameters(self,
                                dtype: torch.Type,
                                device) -> None:
        self.weight = nn.Parameter(torch.empty(
            self.weight.shape,
            dtype=dtype,
            device=device,
        ))
        small_init(self.weight, dim=self.weight.shape[1])


def create_model(model_name, config=None):
    if config is None:
        config = LLAMA_CONFIG_DICT[model_name]
    # noinspection PyUnresolvedReferences
    model = LLaMAModel(config=config).to(torch.device("meta"))
    return model


def initialize_model(model: LLaMAModel, device):
    config = model.config
    model.model.embed_tokens.reset_actual_parameters(dtype=config.dtype, device=device)
    layer_mapping = config.layer_mapping
    for layer_i, layer in enumerate(model.model.layers):
        # Linear params
        layer_mapped_i = layer_mapping[layer_i]
        if layer_i == layer_mapped_i:
            # create layer
            layer.self_attn.q_proj.reset_linear_parameters(
                init_method="small", dtype=config.dtype, device=device)
            layer.self_attn.k_proj.reset_linear_parameters(
                init_method="small", dtype=config.dtype, device=device)
            layer.self_attn.v_proj.reset_linear_parameters(
                init_method="small", dtype=config.dtype, device=device)
            layer.self_attn.o_proj.reset_linear_parameters(
                init_method="wang", dtype=config.dtype, device=device, n_layers=config.n_layers)

            layer.mlp.gate_proj.reset_linear_parameters(
                init_method="small", dtype=config.dtype, device=device)
            layer.mlp.up_proj.reset_linear_parameters(
                init_method="small", dtype=config.dtype, device=device)
            layer.mlp.down_proj.reset_linear_parameters(
                init_method="wang", dtype=config.dtype, device=device, n_layers=config.n_layers)
        else:
            mapped_layer = model.model.layers[layer_mapped_i]
            layer.self_attn.q_proj.set_linear_parameters(mapped_layer.self_attn.q_proj.weight)
            layer.self_attn.k_proj.set_linear_parameters(mapped_layer.self_attn.k_proj.weight)
            layer.self_attn.v_proj.set_linear_parameters(mapped_layer.self_attn.v_proj.weight)
            layer.self_attn.o_proj.set_linear_parameters(mapped_layer.self_attn.o_proj.weight)

            layer.mlp.gate_proj.set_linear_parameters(mapped_layer.mlp.gate_proj.weight)
            layer.mlp.up_proj.set_linear_parameters(mapped_layer.mlp.up_proj.weight)
            layer.mlp.down_proj.set_linear_parameters(mapped_layer.mlp.down_proj.weight)

        # LoRA params
        layer.input_layernorm.reset_actual_parameters(dtype=config.dtype, device=device)
        layer.post_attention_layernorm.reset_actual_parameters(dtype=config.dtype, device=device)
        layer.self_attn.rotary_emb.reset_inv_freq(device)
        if layer.self_attn.use_lora_q:
            layer.self_attn.q_proj.reset_lora_parameters()
        if layer.self_attn.use_lora_k:
            layer.self_attn.k_proj.reset_lora_parameters()
        if layer.self_attn.use_lora_v:
            layer.self_attn.v_proj.reset_lora_parameters()
        if layer.self_attn.use_lora_o:
            layer.self_attn.o_proj.reset_lora_parameters()

        if layer.mlp.use_lora_gate:
            layer.mlp.gate_proj.reset_lora_parameters()
        if layer.mlp.use_lora_up:
            layer.mlp.up_proj.reset_lora_parameters()
        if layer.mlp.use_lora_down:
            layer.mlp.down_proj.reset_lora_parameters()

    model.model.norm.reset_actual_parameters(dtype=config.dtype, device=device)
    model.lm_head.reset_linear_parameters(
        init_method="small", dtype=config.dtype, device=device)


def create_rope_embed_ids(input_ids, pad_token_id=0):
    # Note: this is a dummy value. The embedding for this position is not used practically as
    # the token will be masked out by the attention mask. This primarily serves for readability
    # of the inputs.
    dummy_position = 2047
    rope_embed_ids = (input_ids != pad_token_id).cumsum(-1) - 1
    rope_embed_ids[input_ids == pad_token_id] = dummy_position
    return rope_embed_ids


def create_decoding_mask(orig_num_valid_tokens, max_seq_len, initial_max_len):
    """Generate mask for decoding steps

    :param orig_num_valid_tokens: array of number of valid tokens in initial encoding
    :param max_seq_len: max seq len for this step
    :param initial_max_len: max seq len in initial encoding
    :return:
    """
    batch_size = len(orig_num_valid_tokens)
    mask = torch.ones([batch_size, 1, max_seq_len])
    for i, nvt in enumerate(orig_num_valid_tokens):
        mask[i, :, nvt:initial_max_len] = 0
    return mask[:, None, -1:, ].bool()


def small_init(tensor, dim):
    std = math.sqrt(2 / (5 * dim))
    return torch.nn.init.normal_(tensor, mean=0.0, std=std)


def wang_init(tensor, dim, n_layers):
    std = 2 / n_layers / math.sqrt(dim)
    return torch.nn.init.normal_(tensor, mean=0.0, std=std)
