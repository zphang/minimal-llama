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


class LLaMAModel(nn.Module):
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config
        self.model = LLaMAInnerModel(config)
        self.lm_head = NoInitLinear(config.dim, config.vocab_size, bias=False, dtype=config.dtype)

    def forward(self,
                input_ids):
        """Forward pass (with full decode sequence, intended for training or loss-scoring)

        :param input_ids: [batch_size, seq_len]
        :return: logits [batch_size, seq_len]
        """
        # 1) Create masks
        # decoder mask
        # [batch_size, num_heads=1, q_len=seq_len, kv_len=seq_len]
        attention_mask = create_attention_mask(input_ids=input_ids, dtype=self.config.dtype)
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
                "key": torch.zeros([batch_size, num_heads, 0, head_dim]).to(device),
                "value": torch.zeros([batch_size, num_heads, 0, head_dim]).to(device),
            })
        return kv_cache

    def generate(self, input_ids, generation_length: 20):
        """Generate tokens with efficient caching of KV.

        TODO: Add stopping conditions
        TODO: Add sampling capabilities

        :param input_ids: [batch_size, enc_seq_len]
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
        attention_mask = create_attention_mask(input_ids=input_ids, dtype=self.config.dtype)
        # dict(
        #   hidden_states = [batch_size, dec_seq_len=decode_step+1, hidden_dim]
        #   kv_cache = list[dict(
        #     key = [batch_size, num_heads, kv_seq_len=decode_step+1, head_dim]
        #     value = [batch_size, num_heads, kv_seq_len=decode_step+1, head_dim]
        #   )]
        # )
        rope_embed_ids = create_rope_embed_ids(input_ids=input_ids)
        cos, sin = self.get_cos_sin(rope_embed_ids)
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

        # 2.1 shift KV cache
        for layer_kv_cache in kv_cache:
            for i in range(batch_size):
                layer_kv_cache["key"] = shift_kv_cache_right(
                    layer_kv_cache["key"], num_valid_tokens=num_valid_tokens)
                layer_kv_cache["value"] = shift_kv_cache_right(
                    layer_kv_cache["value"], num_valid_tokens=num_valid_tokens)

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
            rope_embed_ids = create_rope_embed_ids(input_ids=input_ids) + num_valid_tokens
            cos, sin = self.get_cos_sin(rope_embed_ids)
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
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim, dtype=config.dtype)
        self.layers = nn.ModuleList([
            LLaMALayer(config=config)
            for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(dim=config.dim)

    def forward(self,
                input_ids,
                attention_mask,
                cos, sin,
                kv_cache=None):
        """
        :param input_ids: [batch_size, seq_len]
        :param attention_mask: [batch_size=1, num_heads=1, seq_len, seq_len]
        :param kv_cache: See init_kv_cache.
            We use the presence of kv_cache to determine if we're generating
        :param cos:
        :param sin:
        """
        hidden_states = self.embed_tokens(input_ids)

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

            layer_out = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                kv_cache=layer_kv_cache,
                cos=cos, sin=sin,
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

    def forward(
        self,
        hidden_states,
        attention_mask,
        cos, sin,
        kv_cache=None,
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
        )
        # [batch_size, seq_len, hidden_dim]
        hidden_states = hidden_states + raw_self_attn_output["attn_output"]
        check_nan(hidden_states)
        # 2) FFN
        # [batch_size, seq_len, hidden_dim]
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        check_nan(hidden_states)
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


class Attention(nn.Module):
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config
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

    def forward(self, hidden_states, attention_mask, cos, sin, kv_cache=None):
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
            key_states = torch.cat([kv_cache["key"], key_states], dim=2)
            value_states = torch.cat([kv_cache["value"], value_states], dim=2)

        scores = torch.matmul(
            query_states, key_states.transpose(3, 2).type_as(query_states) / math.sqrt(self.head_dim)
        )
        scores += attention_mask

        # (batch_size, num_heads, q_seq_len, kv_seq_len)
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        # (batch_size, num_heads, q_seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, value_states.type_as(query_states))
        # (batch_size, q_seq_len, hidden_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, q_seq_len, hidden_dim,
        )
        attn_output = self.o_proj(attn_output)
        check_nan(attn_output)
        if kv_cache:
            new_kv_cache = {"key": key_states, "value": value_states}
            return {"attn_output": attn_output, "kv_cache": new_kv_cache}
        else:
            return {"attn_output": attn_output}


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


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def create_attention_mask(input_ids,
                          dtype=torch.float32,
                          return_soft_mask=True):
    """Create mask for decoder attention.

    Decoder masks have two use-cases:

    1) Training, where we see the full decoder sequence. In that case,
       we want a causal mask.

    2) Generation, where we only see one token at once. In that case,
       it doesn't really matter what we give, we can just give a 1.
       (i.e. seq_len = 1)

    Note that in both cases we do not care about which decoder_input_ids
    are valid, and also we can always simply broadcast over the batch size
    and heads.

    :param input_ids: [batch_size, seq_len]
    :param dtype: dtype
    :param return_soft_mask: whether to return mask or logits-mask
    :return: float [batch_size=1, num_heads=1, q_len=seq_len, kv_len=seq_len]
    """
    batch_size, seq_length = input_ids.shape
    # [seq_len]
    seq_ids = torch.arange(seq_length, device=input_ids.device)
    # [seq_len, seq_len]
    causal_mask = seq_ids[None, :].repeat(seq_length, 1) <= seq_ids[:, None]
    # [batch_size=1, num_heads=1, seq_len, seq_len]
    causal_mask = causal_mask[None, None, :, :]
    if return_soft_mask:
        return convert_mask_to_soft_mask(causal_mask, dtype=dtype)
    else:
        return causal_mask


def convert_mask_to_soft_mask(mask, dtype):
    """Convert binary mask to mask that can be added to logits.

    (i.e. 0 for attention, large negative for masked)
    """
    mask = mask.to(dtype=dtype)
    mask = (1.0 - mask) * torch.finfo(dtype).min
    return mask


class NoInitLinear(nn.Linear):
    def reset_parameters(self) -> None:
        pass


class NoInit8bitLinear(bnb.nn.Linear8bitLt):
    def reset_parameters(self) -> None:
        pass


def get_linear_class(use_8bit=False):
    if use_8bit:
        return NoInit8bitLinear
    else:
        return NoInitLinear


class NoInitEmbedding(nn.Embedding):
    def reset_parameters(self) -> None:
        pass


def check_nan(x):
    if torch.isnan(x).any():
        import pdb
        pdb.set_trace()


def create_model(model_name, hf_path, use_8bit=False, device=None):
    config = LLAMA_CONFIG_DICT[model_name]
    weight_map = io_utils.read_json(os.path.join(hf_path, "pytorch_model.bin.index.json"))["weight_map"]
    filename_list = sorted(list(set(weight_map.values())))
    if device is None:
        # TODO: Local rank
        device = torch.device("cuda:0")
    if use_8bit:
        config = dataclasses.replace(config, use_8bit=True)
        with init_empty_weights():
            model = LLaMAModel(config=config)
        state_keys = set(model.state_dict())
        filename_list = sorted(list(set(weight_map.values())))
        for filename in tqdm.tqdm(filename_list):
            loaded = torch.load(os.path.join(hf_path, filename), map_location="cpu")
            for k, v in loaded.items():
                set_module_8bit_tensor_to_device(model, tensor_name=k, device=device, value=v)
                state_keys.remove(k)
        assert not state_keys
    else:
        # noinspection PyUnresolvedReferences
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = LLaMAModel(config=config).cuda()
        torch.set_default_tensor_type(torch.FloatTensor)
        state_keys = set(model.state_dict())
        for filename in tqdm.tqdm(filename_list):
            loaded = torch.load(os.path.join(hf_path, filename), map_location="cpu")
            model.load_state_dict(loaded, strict=False)
            for k in loaded:
                state_keys.remove(k)
    return model


def shift_kv_cache_right(layer_cache, num_valid_tokens):
    """
    :param layer_cache: left-aligned kv cache element, [batch_size, num_heads, seq_len, dim]
    :param num_valid_tokens: [batch_size]
    :return:
    """
    batch_size = layer_cache.shape[0]
    # noinspection PyUnresolvedReferences
    return torch.stack([
        torch.cat([
            layer_cache[i, :, num_valid_tokens[i]:, :],
            layer_cache[i, :, :num_valid_tokens[i], :],
        ], dim=1)
        for i in range(batch_size)
    ], dim=0)


def create_generation_attention_mask(batch_size, seq_len, num_valid_tokens, device):
    """
    :param batch_size: int
    :param seq_len: int
    :param num_valid_tokens: [batch_size]
    :param device:
    :return:
    """
    # For right-aligned, based on num_valid_tokens
    # noinspection PyTypeChecker
    attn_mask = torch.zeros([batch_size, 1, 1, seq_len], dtype=bool)
    for i in range(batch_size):
        valid = num_valid_tokens[i]
        # noinspection PyTypeChecker
        # attn_mask[i, 0, -valid:, -valid:] = torch.tril(torch.ones([valid, valid], dtype=bool))
        attn_mask[i, 0, 0, -valid:] = True
    return attn_mask.to(device=device)


def create_casual_attention_mask(seq_len, device):
    # noinspection PyTypeChecker
    attn_mask = torch.tril(torch.ones([seq_len, seq_len], dtype=bool))[None, None, :, :]
    return attn_mask.to(device=device)


def create_rope_embed_ids(input_ids):
    pad_token_id = 0
    max_position = 2047
    x = (input_ids != pad_token_id).cumsum(-1) - 1
    x[input_ids == pad_token_id] = max_position
    return x
