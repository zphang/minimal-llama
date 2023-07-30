import math
import dataclasses

import tqdm.auto as tqdm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb
from accelerate import init_empty_weights

from typing import Optional
import minimal_llama.utils.io_utils as io_utils
from transformers.utils.bitsandbytes import set_module_quantized_tensor_to_device

if os.environ.get("CHECK_NAN"):
    def check_nan(x):
        if torch.isnan(x).any():
            import pdb
            pdb.set_trace()
else:
    # noinspection PyUnusedLocal
    def check_nan(x):
        pass


@dataclasses.dataclass
class LLaMAConfig:
    dim: int
    n_layers: int
    n_heads: int
    vocab_size: int = 32000
    max_seq_length: int = 2048
    dtype: torch.Type = torch.float16
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    use_4bit: bool = False
    gradient_checkpointing: bool = False
    num_gist_tokens: int = 256
    lora_rank: int = 8
    device: Optional[torch.device] = None

    @property
    def head_dim(self):
        return self.dim // self.n_heads

    @property
    def hidden_size(self):
        return self.dim

    def to_dict(self):
        return dataclasses.asdict(self)


LLAMA_7B_CONFIG = LLaMAConfig(
    dim=4096,
    n_layers=32,
    n_heads=32,
)
LLAMA_13B_CONFIG = LLaMAConfig(
    dim=5120,
    n_layers=40,
    n_heads=40,
)
DEBUG_CONFIG = LLaMAConfig(
    dim=64,
    n_layers=3,
    n_heads=4,
)

LLAMA_CONFIG_DICT = {
    "7b": LLAMA_7B_CONFIG,
    "13b": LLAMA_13B_CONFIG,
    "debug": DEBUG_CONFIG,
}

MODE_HYPER = "hyper"
MODE_DOWNSTREAM_TRAIN = "downstream_train"
MODE_DOWNSTREAM_ENCODE = "downstream_decode_encode"
MODE_DOWNSTREAM_DECODE = "downstream_decode_decode"


class LLaMAModel(nn.Module):
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config
        self.model = LLaMAInnerModel(config)
        self.lm_head = create_linear(config.dim, config.vocab_size, dtype=config.dtype,
                                     use_4bit=config.use_4bit, use_lora=False, device=config.device)

    def forward(self,
                hyper_input_ids,
                input_ids):
        """Forward pass (with full decode sequence, intended for training or loss-scoring)
        :param hyper_input_ids: [batch_size, seq_len]
        :param input_ids: [batch_size, seq_len]
            - Always right-padded. Masks are generated based on padding tokens
        :return: logits [batch_size, seq_len]
        """
        hyper_model_out = self.hyper_forward_pass(
            hyper_input_ids=hyper_input_ids,
        )
        logits = self.downstream_forward_pass(
            input_ids=input_ids,
            gist_cache=hyper_model_out["gist_cache"],
            t_offset=hyper_model_out["t_offset"],
        )
        return logits

    def hyper_forward_pass(self, hyper_input_ids):
        rope_embed_ids = create_rope_embed_ids(input_ids=hyper_input_ids)
        cos, sin = self.get_cos_sin(rope_embed_ids)
        hyper_model_out = self.model(
            hyper_input_ids,
            cos=cos, sin=sin,
            # use_kv_cache=False,
            mode=MODE_HYPER,
        )
        t_offset = 1 + hyper_input_ids.argmax(-1)
        hyper_model_out["t_offset"] = t_offset
        return hyper_model_out

    def downstream_forward_pass(self, input_ids, gist_cache, t_offset):
        # 2) Downstream Forward pass
        t_offset = t_offset[:, None]
        prefix_length = gist_cache[0]["key"].shape[2]
        # TODO: We don't actually want to clamp, we want to expend to ensure it can fit both hyper
        #       and regular IDs, and then throw an error if it still doesn't fit
        rope_embed_ids = (create_rope_embed_ids(input_ids=input_ids) + prefix_length + t_offset).clamp(
            max=self.config.max_seq_length - 1,
        )
        cos, sin = self.get_cos_sin(rope_embed_ids)
        attention_mask = create_prefix_train_attention_mask(input_ids, prefix_length)
        # [batch_size, seq_len, hidden_dim]
        model_out = self.model(
            input_ids,
            cos=cos, sin=sin,
            # use_kv_cache=True,
            kv_cache=gist_cache,
            attention_mask=attention_mask,
            mode=MODE_DOWNSTREAM_TRAIN,
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

    def downstream_generate(self,
                            input_ids, gist_cache, t_offset,
                            generation_length: int = 20,
                            return_output_only=True, stop_on_eos=True):
        """Generate tokens with efficient caching of KV.

        TODO: Add stopping conditions
        TODO: Add sampling capabilities

        :param input_ids: [batch_size, input_seq_len]
            - Always right-padded. Masks are generated based on padding tokens
        :param gist_cache:
        :param t_offset: [batch_size]
        :param generation_length: int
        :param return_output_only: True = return continuation only. False = return whole sequence
        :param stop_on_eos
        :return: [batch_size, generation_length]
        """
        original_input_ids = input_ids
        batch_size, input_seq_len = input_ids.shape
        # noinspection PyUnresolvedReferences
        num_valid_tokens = (input_ids != self.config.pad_token_id).long().sum(dim=1)
        orig_num_valid_tokens = num_valid_tokens.clone()
        seen_eos = torch.zeros([batch_size], dtype=torch.bool).to(device=input_ids.device)
        t_offset = t_offset[:, None]
        prefix_length = gist_cache[0]["key"].shape[2]

        # 1) Setup
        if input_ids is None:
            # [batch_size, dec_seq_len=1]
            input_ids = torch.LongTensor(
                [[self.config.pad_token_id]] * batch_size
            ).to(self.lm_head.weights.device)
        # See: init_kv_cache. list[dict]
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
        rope_embed_ids = (create_rope_embed_ids(input_ids=input_ids) + prefix_length + t_offset).clamp(
            max=self.config.max_seq_length - 1,
        )
        cos, sin = self.get_cos_sin(rope_embed_ids)
        attention_mask = create_prefix_train_attention_mask(input_ids, prefix_length)
        model_out = self.model(
            input_ids=input_ids,
            cos=cos, sin=sin,
            mode=MODE_DOWNSTREAM_ENCODE,
            attention_mask=attention_mask,
            kv_cache=gist_cache,
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
        seen_eos = seen_eos | (generated_token_ids == self.config.eos_token_id)

        # 3) Subsequent steps
        for decode_step in range(generation_length-1):
            if stop_on_eos and seen_eos.all():
                break
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
                orig_num_valid_tokens=orig_num_valid_tokens + prefix_length,
                max_seq_len=total_seq_len + prefix_length,
                initial_max_len=original_input_ids.shape[1] + prefix_length,
            ).to(input_ids.device)
            rope_embed_ids += num_valid_tokens[:, None] + prefix_length + t_offset
            cos, sin = self.get_cos_sin(rope_embed_ids)
            model_out = self.model(
                input_ids=input_ids,
                cos=cos, sin=sin,
                mode=MODE_DOWNSTREAM_DECODE,
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
            seen_eos = seen_eos | (generated_token_ids == self.config.eos_token_id)
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
        def make_inputs_require_grads(module, inputs, output):
            output.requires_grad_(True)
        self.model.embed_tokens.register_forward_hook(make_inputs_require_grads)


class LLaMAInnerModel(nn.Module):
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = NoInitExtendedEmbedding(
            config.vocab_size, config.dim,
            additional_tokens=config.num_gist_tokens,
            dtype=config.dtype,
        )
        self.layers = nn.ModuleList([
            LLaMALayer(config=config)
            for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(dim=config.dim, dtype=config.dtype)

    def forward(
        self,
        input_ids,
        cos, sin,
        # use_kv_cache=False,
        kv_cache=None,
        num_valid_tokens=None,
        attention_mask=None,
        mode=None,
    ):
        """
        :param input_ids: [batch_size, seq_len]
        :param kv_cache: See init_kv_cache.
            We use the presence of kv_cache to determine if we're generating
        :param cos:
        :param sin:
        # :param use_kv_cache: If True, we are going to maintain a kv_cache
        #     if kv_cache is None (i.e. the first encoding step in decoding), the kv_cache is just the
        #     based on the kv states.
        :param kv_cache: {"key"/"value": [batch_size, num_heads, cache_seq_len, head_dim]}
            Only used for decoding
        :param num_valid_tokens: [batch_size]
            Only used for decoding
        :param mode:
        :param attention_mask: [batch_size, num_heads, q_len, kv_len]
        """
        hidden_states = self.embed_tokens(input_ids, use_extended=mode == MODE_HYPER)
        hidden_states = hidden_states.to(self.config.dtype)

        if mode == MODE_HYPER:
            is_gist_token = input_ids >= self.config.vocab_size
        else:
            is_gist_token = None

        new_kv_cache = []
        gist_cache = []
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
                    # use_kv_cache,
                    layer_kv_cache,
                    num_valid_tokens,
                    attention_mask,
                    mode,
                    use_reentrant=False,
                )
            else:
                layer_out = layer(
                    hidden_states=hidden_states,
                    cos=cos, sin=sin,
                    # use_kv_cache=use_kv_cache,
                    kv_cache=layer_kv_cache,
                    num_valid_tokens=num_valid_tokens,
                    attention_mask=attention_mask,
                    mode=mode,
                )

            hidden_states = layer_out["hidden_states"]
            if mode == MODE_HYPER:
                gist_cache.append({
                    "key": extract_gist_kv(layer_out["kv_cache"]["key"], is_gist_token),
                    "value": extract_gist_kv(layer_out["kv_cache"]["value"], is_gist_token),
                })
            if mode in (MODE_DOWNSTREAM_ENCODE, MODE_DOWNSTREAM_DECODE):
                new_kv_cache.append(layer_out["kv_cache"])

        if mode == MODE_HYPER:
            return {"gist_cache": gist_cache}

        hidden_states = self.norm(hidden_states)
        output = {
            "hidden_states": hidden_states
        }
        if mode in (MODE_DOWNSTREAM_ENCODE, MODE_DOWNSTREAM_DECODE):
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

    def forward(
        self,
        hidden_states,
        cos, sin,
        # use_kv_cache=False,
        kv_cache=None,
        num_valid_tokens=None,
        attention_mask=None,
        mode=None,
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
        check_nan(normed_hidden_states)
        raw_self_attn_output = self.self_attn(
            hidden_states=normed_hidden_states,
            cos=cos, sin=sin,
            # use_kv_cache=use_kv_cache,
            kv_cache=kv_cache,
            num_valid_tokens=num_valid_tokens,
            attention_mask=attention_mask,
            mode=mode,
        )
        # [batch_size, seq_len, hidden_dim]
        hidden_states = hidden_states + raw_self_attn_output["attn_output"]
        check_nan(hidden_states)
        # 2) FFN
        # [batch_size, seq_len, hidden_dim]
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states),
            use_pefts=mode == MODE_HYPER,
        )
        check_nan(hidden_states)
        if mode in (MODE_HYPER, MODE_DOWNSTREAM_ENCODE, MODE_DOWNSTREAM_DECODE):
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

        self.gate_proj = create_linear(dim, hidden_dim, dtype=config.dtype, use_4bit=config.use_4bit,
                                       rank=config.lora_rank)
        self.up_proj = create_linear(dim, hidden_dim, dtype=config.dtype, use_4bit=config.use_4bit,
                                     rank=config.lora_rank)
        self.down_proj = create_linear(hidden_dim, dim, dtype=config.dtype, use_4bit=config.use_4bit,
                                       rank=config.lora_rank)

    def forward(self, x, use_pefts=False):
        return self.down_proj(
            F.silu(self.gate_proj(x, use_lora=use_pefts))
            * self.up_proj(x, use_lora=use_pefts),
            use_lora=use_pefts,
        )


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


class Attention(nn.Module):
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads

        self.q_proj = create_linear(config.dim, config.dim, dtype=config.dtype, use_4bit=config.use_4bit,
                                    rank=config.lora_rank)
        self.k_proj = create_linear(config.dim, config.dim, dtype=config.dtype, use_4bit=config.use_4bit,
                                    rank=config.lora_rank)
        self.v_proj = create_linear(config.dim, config.dim, dtype=config.dtype, use_4bit=config.use_4bit,
                                    rank=config.lora_rank)
        self.o_proj = create_linear(config.dim, config.dim, dtype=config.dtype, use_4bit=config.use_4bit,
                                    rank=config.lora_rank)
        self.rotary_emb = RotaryEmbedding(dim=self.head_dim, max_position_embeddings=config.max_seq_length)

    def forward(self, hidden_states, cos, sin,
                # use_kv_cache=False,
                kv_cache=None,
                num_valid_tokens=None,
                attention_mask=None,
                mode=None):
        """
        :param hidden_states: [batch_size, seq_len, hidden_dim]
        :param cos:
        :param sin:
        # :param use_kv_cache: If True, we are going to maintain a kv_cache
        #     if kv_cache is None (i.e. the first encoding step in decoding), the kv_cache is just the
        #     based on the kv states.
        :param kv_cache: {"key"/"value": [batch_size, num_heads, cache_seq_len, head_dim]}
            Only used for decoding
        :param num_valid_tokens: [batch_size]
            Only used for decoding
        :param attention_mask: [batch_size, num_heads, q_len, kv_len]
        :param mode
        :return:
        """
        batch_size, q_seq_len, hidden_dim = hidden_states.size()
        use_lora = mode == MODE_HYPER

        # (batch_size, num_heads, q_seq_len, head_dim)
        query_states = self.q_proj(hidden_states, use_lora=use_lora).view(
            batch_size, q_seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states, use_lora=use_lora).view(
            batch_size, q_seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states, use_lora=use_lora).view(
            batch_size, q_seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos=cos, sin=sin)
        if mode in (MODE_DOWNSTREAM_TRAIN, MODE_DOWNSTREAM_ENCODE, MODE_DOWNSTREAM_DECODE):
            key_states, value_states = self.append_to_kv_cache(
                kv_cache=kv_cache,
                new_key_state=key_states,
                new_value_state=value_states,
            )
        if q_seq_len == key_states.shape[2]:
            # We shouldn't encounter this case because of prefixes

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
        # (batch_size, q_seq_len, hidden_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, q_seq_len, hidden_dim,
        )
        attn_output = self.o_proj(attn_output, use_lora=use_lora)
        check_nan(attn_output)
        if mode in (MODE_HYPER, MODE_DOWNSTREAM_ENCODE, MODE_DOWNSTREAM_DECODE):
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


class NoInit4bitLinear(bnb.nn.Linear4bit):
    source_cls = nn.Linear

    def reset_parameters(self) -> None:
        pass


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


class NoInitLora4bitLinear(NoInit4bitLinear):
    source_cls = nn.Linear

    def __init__(self,
                 input_features: int, output_features: int,
                 rank: int, alpha: int = 16,
                 compute_dtype=None,
                 compress_statistics=True,
                 quant_type="fp4",
                 lora_device=None) -> None:
        super().__init__(
            input_features=input_features, output_features=output_features,
            bias=False,
            compute_dtype=compute_dtype,
            compress_statistics=compress_statistics,
            quant_type=quant_type,
        )
        factory_kwargs = {'device': lora_device, 'dtype': compute_dtype}
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


def create_linear(in_features: int, out_features: int, dtype: torch.Type,
                  use_4bit: bool = False, double_quant: bool = True,
                  use_lora=True,
                  rank: int = 8, device: Optional[torch.device] = None) -> nn.Module:
    if use_4bit:
        if use_lora:
            return NoInitLora4bitLinear(
                in_features, out_features,
                compute_dtype=dtype,
                compress_statistics=double_quant,
                quant_type="nf4",
                rank=rank,
                lora_device=device,
            )
        else:
            return NoInit4bitLinear(
                in_features, out_features, bias=False,
                compute_dtype=dtype,
                compress_statistics=double_quant,
                quant_type="nf4",
            )
    else:
        if use_lora:
            return NoInitLoraLinear(in_features, out_features, dtype=dtype, rank=rank)
        else:
            return NoInitLinear(in_features, out_features, bias=False, dtype=dtype)


class NoInitExtendedEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, additional_tokens: int, dtype=None,
                 device: Optional[torch.device] = None) -> None:
        super().__init__(num_embeddings, embedding_dim, dtype=dtype, device=device)
        self.additional_tokens = additional_tokens
        self.weight.requires_grad_(False)
        self.extended_weight = nn.Parameter(torch.empty([
            self.additional_tokens, embedding_dim
        ], device=device), requires_grad=True)

    def forward(self, input_ids: torch.Tensor, use_extended: bool = False) -> torch.Tensor:
        if not use_extended:
            return super().forward(input_ids)

        regular_input_ids = input_ids.clamp(max=self.num_embeddings - 1)
        is_regular = (input_ids < self.num_embeddings)
        regular_embedded = super().forward(regular_input_ids)
        regular_embedded = regular_embedded * is_regular[:, :, None]

        extended_input_ids = (input_ids - self.num_embeddings).clamp(min=0)
        extended_embedded = F.embedding(extended_input_ids, self.extended_weight)
        extended_embedded = extended_embedded * (~is_regular)[:, :, None]
        return regular_embedded + extended_embedded

    def reset_parameters(self) -> None:
        pass

    def reset_extended_embeddings(self, ):
        indices = torch.randint(self.num_embeddings, (self.additional_tokens,))
        extended_embeddings = self.weight[indices]
        self.extended_weight = nn.Parameter(extended_embeddings)


def create_model(model_name, hf_path, use_4bit=False, device=None, config=None):
    if config is None:
        config = LLAMA_CONFIG_DICT[model_name]
    weight_map = io_utils.read_json(os.path.join(hf_path, "pytorch_model.bin.index.json"))["weight_map"]
    filename_list = sorted(list(set(weight_map.values())))
    if device is None:
        # TODO: Local rank
        device = torch.device("cuda:0")
    config.device = device
    if use_4bit:
        config = dataclasses.replace(config, use_4bit=True)
        with init_empty_weights():
            model = LLaMAModel(config=config)
        state_keys = set(model.state_dict())
        filename_list = sorted(list(set(weight_map.values())))
        for filename in tqdm.tqdm(filename_list):
            loaded = torch.load(os.path.join(hf_path, filename), map_location="cpu")
            for k, v in loaded.items():
                if "lm_head" in k or "layernorm" in k or ".norm" in k:
                    v = v.to(config.dtype)
                set_module_quantized_tensor_to_device(model, tensor_name=k, device=device, value=v)
                state_keys.remove(k)
        for k in list(state_keys):
            if "lora" in k or "extended" in k:
                state_keys.remove(k)
        assert not state_keys
    else:
        # noinspection PyUnresolvedReferences
        # torch.set_default_tensor_type(torch.cuda.HalfTensor)
        # model = LLaMAModel(config=config).cuda()
        # torch.set_default_tensor_type(torch.FloatTensor)
        model = LLaMAModel(config=config).to(device)
        if model_name == "debug":
            return model
        state_keys = set(model.state_dict())
        for filename in tqdm.tqdm(filename_list):
            loaded = torch.load(os.path.join(hf_path, filename), map_location="cpu")
            model.load_state_dict(loaded, strict=False)
            for k in loaded:
                state_keys.remove(k)

    initialize_pefts(model)

    return model


def initialize_pefts(model):
    model.model.embed_tokens.reset_extended_embeddings()
    for layer in model.model.layers:
        layer.self_attn.q_proj.reset_lora_parameters()
        layer.self_attn.k_proj.reset_lora_parameters()
        layer.self_attn.v_proj.reset_lora_parameters()
        layer.self_attn.o_proj.reset_lora_parameters()
        layer.mlp.gate_proj.reset_lora_parameters()
        layer.mlp.up_proj.reset_lora_parameters()
        layer.mlp.down_proj.reset_lora_parameters()
    # model.lm_head.reset_lora_parameters()
    for k, v in model.named_parameters():
        if "lora" in k or "extended" in k:
            v.requires_grad_(True)
        else:
            v.requires_grad_(False)


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


def create_prefix_train_attention_mask(input_ids, prefix_length):
    """Create attention mask for prefix training

    :param input_ids: input ids
    :param prefix_length:
    :return:
    """
    batch_size, seq_len = input_ids.shape
    input_mask = torch.ones([seq_len, seq_len], dtype=torch.bool)
    input_mask.tril_()
    prefix_mask = torch.ones([seq_len, prefix_length], dtype=torch.bool)
    full_mask = torch.cat([prefix_mask, input_mask], dim=1)
    return full_mask[None, None, :, :].to(input_ids.device)


def extract_gist_kv(kv, is_gist_token):
    num_gist_tokens = is_gist_token.long()[0].sum()
    bs, num_heads, seq_len, head_dim = kv.shape
    kv = kv.transpose(1, 2).reshape(bs * seq_len, num_heads, head_dim)
    gist_kv = kv[is_gist_token.flatten()]
    gist_kv = gist_kv.reshape(bs, num_gist_tokens, num_heads, head_dim).transpose(1, 2)
    return gist_kv
