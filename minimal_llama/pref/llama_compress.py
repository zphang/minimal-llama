import dataclasses

import math
import tqdm.auto as tqdm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import bitsandbytes as bnb
from accelerate import init_empty_weights

import proj_shared.io_utils as io_utils
from transformers.utils.bitsandbytes import set_module_8bit_tensor_to_device

PEFT_PREFIX = "prefix"
PEFT_PREFIX_ADAPTER = "prefix_adapter"
PEFT_NO = "nothing"


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
    gradient_checkpointing: bool = False

    @property
    def head_dim(self):
        return self.dim // self.n_heads

    def to_dict(self):
        return dataclasses.asdict(self)


LLAMA_7B_CONFIG = LLaMAConfig(
    dim=4096,
    n_layers=32,
    n_heads=32,
)

LLAMA_CONFIG_DICT = {
    "7b": LLAMA_7B_CONFIG,
}


@dataclasses.dataclass
class TrainConfig:
    peft_mode: str
    num_prefix_tokens: int = None
    block_size: int = 64
    factorized_compressor: bool = True
    adapter_gate_mode: str = "fixed"

    def check(self):
        assert self.peft_mode in (
            PEFT_PREFIX, PEFT_PREFIX_ADAPTER,
        )


class LLaMAModel(nn.Module):
    def __init__(self, config: LLaMAConfig, train_config: TrainConfig):
        super().__init__()
        self.config = config
        self.train_config = train_config
        self.model = LLaMAInnerModel(config, train_config=train_config)
        self.lm_head = NoInitLinear(config.dim, config.vocab_size, bias=False, dtype=config.dtype)

    def forward(self,
                input_ids,
                output_full=False):
        """Forward pass (with full decode sequence, intended for training or loss-scoring)

        :param input_ids: [batch_size, seq_len]
        :param output_full:
        :return: logits [batch_size, seq_len]
        """
        block_size = self.train_config.block_size
        num_blocks = input_ids.shape[1] // block_size
        device = input_ids.device
        # 1.1) Create full masks and rope embeds
        # decoder mask
        # [batch_size, num_heads=1, q_len=seq_len, kv_len=seq_len]
        full_attention_mask = create_attention_mask(input_ids=input_ids, dtype=self.config.dtype)
        full_rope_embed_ids = create_rope_embed_ids(input_ids=input_ids)
        full_cos, full_sin = self.get_cos_sin(full_rope_embed_ids)
        full_cos, full_sin = full_cos[:, None, :, :], full_sin[:, None, :, :]

        # 1.2) Create conditional masks and rope embeds
        conditional_attention_mask = create_attention_mask(input_ids=input_ids[:, :block_size], dtype=self.config.dtype)
        if self.train_config.peft_mode == PEFT_PREFIX:
            num_prefix_tokens = self.train_config.num_prefix_tokens
            conditional_attention_mask = torch.cat([
                zeros_like([1, 1, block_size, num_prefix_tokens], tensor=conditional_attention_mask),
                conditional_attention_mask,
            ], dim=3)[None]
        # Assume fully packed
        # [1, block_size*num_blocks=seq_len]
        conditional_rope_embed_ids = torch.arange(block_size)[None, :].expand(num_blocks, block_size)
        conditional_rope_embed_ids = conditional_rope_embed_ids.long().to(device)
        conditional_cos, conditional_sin = self.get_cos_sin(conditional_rope_embed_ids)
        conditional_cos, conditional_sin = (
            conditional_cos[None, :, None, :, :],
            conditional_sin[None, :, None, :, :],
        )

        # 2) Forward pass
        # [batch_size, seq_len, hidden_dim]
        model_out = self.model(
            input_ids=input_ids,
            full_attention_mask=full_attention_mask,
            conditional_attention_mask=conditional_attention_mask,
            full_cos=full_cos, full_sin=full_sin,
            conditional_cos=conditional_cos, conditional_sin=conditional_sin,
        )
        # [batch_size, seq_len, vocab_size]
        logits = self.lm_head(model_out["conditional_hidden_states"])
        if output_full:
            full_logits = self.lm_head(model_out["full_hidden_states"])
            return {
                "logits": logits,
                "full_logits": full_logits,
            }
        else:
            return logits

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

    def gradient_checkpointing_enable(self):
        self.config.gradient_checkpointing = True

    def enable_input_require_grads(self):
        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)
        self.model.embed_tokens.register_forward_hook(make_inputs_require_grads)


class LLaMAInnerModel(nn.Module):
    def __init__(self, config: LLaMAConfig, train_config: TrainConfig):
        super().__init__()
        self.config = config
        self.train_config = train_config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim, dtype=config.dtype)
        self.layers = nn.ModuleList([
            LLaMALayer(config=config, train_config=train_config)
            for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(dim=config.dim)

    def forward(self,
                input_ids,
                full_attention_mask,
                conditional_attention_mask,
                full_cos, full_sin,
                conditional_cos, conditional_sin):
        """
        :param input_ids: [batch_size, seq_len]
        :param full_attention_mask: [batch_size=1, num_heads=1, seq_len, seq_len]
        :param conditional_attention_mask: [batch_size=1, num_heads=1, num_blocks, num_prefix_tokens=1, seq_len]
        :param full_cos:
        :param full_sin:
        :param conditional_cos:
        :param conditional_sin:
        """
        full_hidden_states = self.embed_tokens(input_ids)
        conditional_hidden_states = full_hidden_states.detach().clone()
        if self.config.gradient_checkpointing:
            conditional_hidden_states.requires_grad_(True)
        for layer_i, layer in enumerate(self.layers):
            if self.config.gradient_checkpointing:
                layer_out = torch.utils.checkpoint.checkpoint(
                    layer,
                    full_hidden_states,
                    full_attention_mask,
                    conditional_hidden_states,
                    conditional_attention_mask,
                    full_cos, full_sin,
                    conditional_cos, conditional_sin,
                )
            else:
                layer_out = layer(
                    full_hidden_states=full_hidden_states,
                    full_attention_mask=full_attention_mask,
                    conditional_hidden_states=conditional_hidden_states,
                    conditional_attention_mask=conditional_attention_mask,
                    full_cos=full_cos, full_sin=full_sin,
                    conditional_cos=conditional_cos, conditional_sin=conditional_sin,
                )
            # full_hidden_states = layer_out["full_hidden_states"]
            # conditional_hidden_states = layer_out["conditional_hidden_states"]
            full_hidden_states, conditional_hidden_states = layer_out
        full_hidden_states = self.norm(full_hidden_states)
        conditional_hidden_states = self.norm(conditional_hidden_states)
        output = {
            "full_hidden_states": full_hidden_states,
            "conditional_hidden_states": conditional_hidden_states,
        }
        return output


class LLaMALayer(nn.Module):
    def __init__(self, config: LLaMAConfig, train_config: TrainConfig):
        super().__init__()
        self.config = config
        self.train_config = train_config
        self.self_attn = Attention(config=config, train_config=train_config)
        self.mlp = MLP(config=config)
        self.input_layernorm = RMSNorm(dim=config.dim, dtype=config.dtype)
        self.post_attention_layernorm = RMSNorm(dim=config.dim, dtype=config.dtype)

    def forward(
        self,
        full_hidden_states, full_attention_mask,
        conditional_hidden_states, conditional_attention_mask,
        full_cos, full_sin,
        conditional_cos, conditional_sin
    ):
        # 1) Self-attention
        # [batch_size, seq_len, hidden_dim]
        normed_full_hidden_states = self.input_layernorm(full_hidden_states)
        normed_conditional_hidden_states = self.input_layernorm(conditional_hidden_states)
        # dict(
        #   attn_output = [batch_size, seq_len, hidden_dim]
        #   kv_cache = dict(
        #     key = [batch_size, num_heads, kv_seq_len, head_dim]
        #     value = [batch_size, num_heads, kv_seq_len, head_dim]
        #   )
        # )
        check_nan(normed_full_hidden_states)
        check_nan(normed_conditional_hidden_states)
        raw_self_attn_output = self.self_attn(
            full_hidden_states=normed_full_hidden_states,
            full_attention_mask=full_attention_mask,
            conditional_hidden_states=normed_conditional_hidden_states,
            conditional_attention_mask=conditional_attention_mask,
            full_cos=full_cos, full_sin=full_sin,
            conditional_cos=conditional_cos, conditional_sin=conditional_sin,
        )
        # [batch_size, seq_len, hidden_dim]
        full_hidden_states = full_hidden_states + raw_self_attn_output["full_attn_output"]
        conditional_hidden_states = conditional_hidden_states + raw_self_attn_output["conditional_attn_output"]
        check_nan(full_hidden_states)
        check_nan(conditional_hidden_states)
        # 2) FFN
        # [batch_size, seq_len, hidden_dim]
        full_hidden_states = full_hidden_states + self.mlp(
            self.post_attention_layernorm(full_hidden_states))
        conditional_hidden_states = conditional_hidden_states + self.mlp(
            self.post_attention_layernorm(conditional_hidden_states))
        check_nan(full_hidden_states)
        check_nan(conditional_hidden_states)
        # return {
        #     "full_hidden_states": full_hidden_states,
        #     "conditional_hidden_states": conditional_hidden_states,
        # }
        return full_hidden_states, conditional_hidden_states


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
    def __init__(self, config: LLaMAConfig, train_config: TrainConfig):
        super().__init__()
        self.config = config
        self.train_config = train_config
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.block_size = self.train_config.block_size

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

        if self.train_config.peft_mode in (PEFT_PREFIX, PEFT_PREFIX_ADAPTER):
            self.k_compressor = Compressor(config=config, train_config=train_config)
            self.v_compressor = Compressor(config=config, train_config=train_config)
        elif self.train_config.peft_mode == PEFT_NO:
            pass
        else:
            raise KeyError(self.train_config.peft_mode)

        if self.train_config.peft_mode == PEFT_PREFIX_ADAPTER:
            if self.train_config.adapter_gate_mode == "fixed":
                self.adapter_gates = nn.Parameter(torch.zeros(self.n_heads))
            else:
                raise KeyError(self.train_config.adapter_gate_mode)

    def forward(self,
                full_hidden_states, full_attention_mask,
                conditional_hidden_states, conditional_attention_mask,
                full_cos, full_sin,
                conditional_cos, conditional_sin):
        """
        """
        batch_size, full_seq_len, hidden_dim = full_hidden_states.size()
        num_blocks = full_seq_len // self.train_config.block_size
        # 1) Compute history
        # (batch_size, num_heads, q_seq_len, head_dim)
        full_query_states = self.q_proj(full_hidden_states).view(
            batch_size, full_seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        full_key_states = self.k_proj(full_hidden_states).view(
            batch_size, full_seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        full_value_states = self.v_proj(full_hidden_states).view(
            batch_size, full_seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        full_query_states, full_key_states = apply_rotary_pos_emb(
            full_query_states, full_key_states,
            cos=full_cos, sin=full_sin)
        full_scores = torch.matmul(
            full_query_states,
            full_key_states.transpose(-1, -2).type_as(full_query_states) / math.sqrt(self.head_dim)
        )
        full_scores += full_attention_mask

        # (batch_size, num_heads, block_size, full_seq_len, kv_seq_len)
        full_attn_weights = F.softmax(full_scores.float(), dim=-1).type_as(full_scores)
        # (batch_size, num_heads, full_seq_len, head_dim)
        full_attn_output = torch.matmul(full_attn_weights, full_value_states.type_as(full_query_states))
        # (batch_size, q_seq_len, hidden_dim)
        full_attn_output = full_attn_output.transpose(1, 2).contiguous().view(
            batch_size, full_seq_len, hidden_dim,
        )
        full_attn_output = self.o_proj(full_attn_output)
        check_nan(full_attn_output)
        # ====

        # 2) Compute prefix
        # [batch_size, num_heads, num_blocks, num_prefix_tokens, head_dim]
        if self.train_config.peft_mode in (PEFT_PREFIX, PEFT_PREFIX_ADAPTER):
            prefix_k = self.k_compressor(hidden_states=full_hidden_states, past_kvs=full_key_states)
            prefix_v = self.v_compressor(hidden_states=full_hidden_states, past_kvs=full_value_states)
        elif self.train_config.peft_mode == PEFT_NO:
            pass
        else:
            raise KeyError(self.train_config.peft_mode)
        # ====

        # 3) Compute conditional LM
        # [batch_size, num_blocks, num_heads, block_size, head_dim]
        conditional_query_states = self.q_proj(conditional_hidden_states).view(
            batch_size, num_blocks, self.block_size, self.n_heads, self.head_dim).transpose(-2, -3)
        conditional_key_states = self.k_proj(conditional_hidden_states).view(
            batch_size, num_blocks, self.block_size, self.n_heads, self.head_dim).transpose(-2, -3)
        conditional_value_states = self.v_proj(conditional_hidden_states).view(
            batch_size, num_blocks, self.block_size, self.n_heads, self.head_dim).transpose(-2, -3)
        conditional_query_states, conditional_key_states = apply_rotary_pos_emb(
            conditional_query_states, conditional_key_states,
            cos=conditional_cos, sin=conditional_sin,
        )
        if self.train_config.peft_mode == PEFT_PREFIX:
            # [batch_size, num_blocks, num_heads, num_prefix_tokens+block_size, head_dim]
            conditional_key_states = torch.cat([prefix_k, conditional_key_states], dim=-2)
            conditional_value_states = torch.cat([prefix_v, conditional_value_states], dim=-2)
            # [batch_size, num_blocks, num_heads, block_size, num_prefix_tokens+block_size]
            conditional_scores = torch.matmul(
                conditional_query_states,
                conditional_key_states.transpose(-1, -2).type_as(conditional_query_states) / math.sqrt(self.head_dim)
            )
            conditional_scores += conditional_attention_mask
            # [batch_size, num_blocks, num_heads, block_size, num_prefix_tokens+block_size]
            conditional_attn_weights = F.softmax(conditional_scores.float(), dim=-1).type_as(conditional_scores)
            # [batch_size, num_blocks, num_heads, block_size, head_dim]
            conditional_attn_output = torch.matmul(
                conditional_attn_weights,
                conditional_value_states.type_as(conditional_query_states),
            )
            # [batch_size, num_heads, full_seq_len, head_dim]
            conditional_attn_output = conditional_attn_output.transpose(2, 3).contiguous().view(
                batch_size, full_seq_len, hidden_dim,
            )
            conditional_attn_output = self.o_proj(conditional_attn_output)
            check_nan(conditional_attn_output)
        elif self.train_config.peft_mode == PEFT_PREFIX_ADAPTER:
            conditional_scores = torch.matmul(
                conditional_query_states,
                conditional_key_states.transpose(-1, -2).type_as(conditional_query_states) / math.sqrt(self.head_dim)
            )
            conditional_scores += conditional_attention_mask
            # [batch_size, num_blocks, num_heads, block_size, num_prefix_tokens+block_size]
            conditional_attn_weights = F.softmax(conditional_scores.float(), dim=-1).type_as(conditional_scores)
            # [batch_size, num_blocks, num_heads, block_size, head_dim]
            conditional_attn_output = torch.matmul(
                conditional_attn_weights,
                conditional_value_states.type_as(conditional_query_states),
            )
            # [batch_size, num_blocks, num_heads, block_size, num_prefix_tokens]
            adapter_scores = torch.matmul(
                conditional_query_states,
                prefix_k.transpose(-1, -2).type_as(conditional_query_states) / math.sqrt(self.head_dim)
            )
            # [batch_size, num_blocks, num_heads, block_size, num_prefix_tokens]
            adapter_attn_weights = F.softmax(adapter_scores.float(), dim=-1).type_as(adapter_scores)
            # [batch_size=1, num_blocks=1, num_heads, block_size=1, head_dim=1]
            gate = F.tanh(self.adapter_gates.view(1, 1, -1, 1, 1).to(full_hidden_states.dtype))
            # [batch_size, num_blocks, num_heads, block_size, head_dim]
            adapter_attn_output = gate * torch.matmul(
                adapter_attn_weights,
                prefix_v.type_as(conditional_query_states),
            )
            # [batch_size, num_blocks, num_heads, block_size, head_dim]
            conditional_attn_output += adapter_attn_output
            # [batch_size, num_heads, full_seq_len, head_dim]
            conditional_attn_output = conditional_attn_output.transpose(2, 3).contiguous().view(
                batch_size, full_seq_len, hidden_dim,
            )
            conditional_attn_output = self.o_proj(conditional_attn_output)
            check_nan(conditional_attn_output)
        elif self.train_config.peft_mode == PEFT_NO:
            conditional_attn_output = full_attn_output
        else:
            raise KeyError(self.train_config.peft_mode)
        return {
            "full_attn_output": full_attn_output,
            "conditional_attn_output": conditional_attn_output,
        }


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
        raise NotImplementedError()


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


def create_model(model_name, hf_path, train_config: TrainConfig, use_8bit=False, device=None):
    config = LLAMA_CONFIG_DICT[model_name]
    weight_map = io_utils.read_json(os.path.join(hf_path, "pytorch_model.bin.index.json"))["weight_map"]
    filename_list = sorted(list(set(weight_map.values())))
    if device is None:
        # TODO: Local rank
        device = torch.device("cuda:0")
    if use_8bit:
        config = dataclasses.replace(config, use_8bit=True)
        with init_empty_weights():
            model = LLaMAModel(config=config, train_config=train_config)

        for layer_i in range(config.n_layers):
            model.model.layers[layer_i].self_attn.rotary_emb.cos_cached = \
                model.model.layers[layer_i].self_attn.rotary_emb.cos_cached.to(device)
            model.model.layers[layer_i].self_attn.rotary_emb.sin_cached = \
                model.model.layers[layer_i].self_attn.rotary_emb.sin_cached.to(device)
            model.model.layers[layer_i].self_attn.k_compressor.to_empty(device=device)
            model.model.layers[layer_i].self_attn.k_compressor.init_weights()
            model.model.layers[layer_i].self_attn.v_compressor.to_empty(device=device)
            model.model.layers[layer_i].self_attn.v_compressor.init_weights()
            if train_config.peft_mode == PEFT_PREFIX_ADAPTER:
                model.model.layers[layer_i].self_attn.adapter_gates.to_empty(device=device)
                model.model.layers[layer_i].self_attn.adapter_gates.weight.zeros_()

        state_keys = set(model.state_dict())
        filename_list = sorted(list(set(weight_map.values())))
        for filename in tqdm.tqdm(filename_list):
            loaded = torch.load(os.path.join(hf_path, filename), map_location="cpu")
            for k, v in loaded.items():
                set_module_8bit_tensor_to_device(model, tensor_name=k, device=device, value=v)
                state_keys.remove(k)
                if train_config.factorized_compressor:
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
        model = LLaMAModel(config=config, train_config=train_config).cuda()
        torch.set_default_tensor_type(torch.FloatTensor)
        state_keys = set(model.state_dict())
        for filename in tqdm.tqdm(filename_list):
            loaded = torch.load(os.path.join(hf_path, filename), map_location="cpu")
            if train_config.factorized_compressor:
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
        if train_config.peft_mode == PEFT_PREFIX_ADAPTER:
            model.model.layers[layer_i].self_attn.adapter_gates = nn.Parameter(
                model.model.layers[layer_i].self_attn.adapter_gates.float())

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


def create_rope_embed_ids(input_ids):
    pad_token_id = 0
    max_position = 2047
    x = (input_ids != pad_token_id).cumsum(-1) - 1
    x[input_ids == pad_token_id] = max_position
    return x


class Compressor(nn.Module):
    def __init__(self, config: LLaMAConfig, train_config: TrainConfig):
        super().__init__()
        self.config = config
        self.train_config = train_config
        if self.train_config.factorized_compressor:
            self.embed = nn.Embedding(train_config.num_prefix_tokens, config.dim)
            self.sub_q_proj = nn.Linear(config.dim, config.dim, bias=False)
            self.sub_k_proj = nn.Linear(config.dim, config.dim, bias=False)
            self.sub_q_proj.weight.data.normal_(mean=0.0, std=0.02)
            self.sub_k_proj.weight.data.normal_(mean=0.0, std=0.02)
        else:
            self.a_proj = nn.Linear(config.dim, config.n_heads * train_config.num_prefix_tokens, bias=False)
            self.a_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.head_dim = config.dim // config.n_heads

    def init_weights(self):
        if self.train_config.factorized_compressor:
            self.sub_q_proj.weight.data.normal_(mean=0.0, std=0.02)
            self.sub_k_proj.weight.data.normal_(mean=0.0, std=0.02)
        else:
            self.a_proj.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, hidden_states, past_kvs):
        """
        :param hidden_states: [batch_size, seq_len, hidden_dim]
        :param past_kvs: [batch_size, n_heads, seq_len, head_dim]
        :return:
        """
        batch_size, seq_len, _ = hidden_states.shape
        num_blocks = seq_len // self.train_config.block_size
        device = hidden_states.device
        # [batch_size = 1, num_heads = 1, num_blocks, num_prefix_tokens = 1, seq_len]
        block_attn_mask = create_black_attention_mask(
            num_blocks=num_blocks,
            block_size=self.train_config.block_size,
            seq_len=seq_len,
            dtype=torch.float16,
        ).to(device)
        hidden_states = hidden_states.float()
        past_kvs = past_kvs.float()

        if self.train_config.factorized_compressor:
            num_prefix_tokens = self.train_config.num_prefix_tokens
            prefix_embed = self.embed(torch.arange(self.train_config.num_prefix_tokens).long().to(device))
            sub_q = self.sub_q_proj(prefix_embed) \
                .expand(batch_size, num_prefix_tokens, self.config.dim) \
                .view(batch_size, num_prefix_tokens, self.config.n_heads, self.head_dim) \
                .transpose(1, 2)
            sub_k = self.sub_k_proj(hidden_states) \
                .view(batch_size, seq_len, self.config.n_heads, self.head_dim) \
                .transpose(1, 2)
            a_projector = torch.matmul(sub_q, sub_k.transpose(-2, -1) / math.sqrt(self.head_dim))
        else:
            # [batch_size, seq_len, num_heads * num_prefix_tokens]
            a_projector = self.a_proj(hidden_states)
            # [batch_size, num_heads, num_prefix_tokens, seq_len]
            a_projector = a_projector.view(
                batch_size, seq_len, self.config.n_heads, self.train_config.num_prefix_tokens,
            ).permute(0, 2, 3, 1)
        # [batch_size, num_blocks, num_heads, num_prefix_tokens, seq_len]
        blocked_a_projector = a_projector[:, None, :, :, :].expand(
            batch_size, num_blocks, self.config.n_heads, self.train_config.num_prefix_tokens, seq_len,
        )
        # [batch_size, num_blocks, num_heads, num_prefix_tokens, seq_len]
        blocked_scores = blocked_a_projector + block_attn_mask
        # [batch_size, num_blocks, num_heads, num_prefix_tokens, seq_len]
        blocked_attn_weights = softmax(blocked_scores)
        # [batch_size, num_blocks, num_heads, seq_len, head_dim]
        blocked_past_kvs = past_kvs[:, None, :, :, :].expand(
            batch_size, num_blocks, self.config.n_heads, seq_len, self.head_dim,
        )
        # [batch_size, num_blocks, num_heads, num_prefix_tokens, head_dim]
        attn_output = torch.matmul(blocked_attn_weights, blocked_past_kvs)
        check_nan(attn_output)
        return attn_output.to(hidden_states.dtype)


def apply_attn(q, k, v, causal_attention_mask=None):
    """
    :param q: [..., q_seq_len, attn_dim]
    :param k: [..., kv_seq_len, attn_dim]
    :param v: [..., kv_seq_len, out_dim]
    :param causal_attention_mask: [..., q_seq_len, kv_seq_len]
    :return: [..., q_seq_len, out_dim]
    """
    # [..., q_seq_len, kv_seq_len]
    scores = torch.matmul(q, k.transpose(-2, -1) / math.sqrt(q.shape[-1]))
    if causal_attention_mask is not None:
        scores = scores + causal_attention_mask
    scores += causal_attention_mask
    # [..., q_seq_len, kv_seq_len]
    attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
    # [..., q_seq_len, out_dim]
    attn_output = torch.matmul(attn_weights, v)
    return attn_output


def apply_partial_attn(scores, v, causal_attention_mask=None):
    """
    :param scores: [..., q_seq_len, kv_seq_len]
    :param v: [..., kv_seq_len, out_dim]
    :param causal_attention_mask: [..., q_seq_len, kv_seq_len]
    :return: [..., q_seq_len, out_dim]
    """
    if causal_attention_mask is not None:
        scores = scores + causal_attention_mask
    # [..., q_seq_len, kv_seq_len]
    attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
    # [..., q_seq_len, out_dim]
    attn_output = torch.matmul(attn_weights, v)
    return attn_output


def softmax(scores, dim=-1):
    return F.softmax(scores.float(), dim=dim).type_as(scores)


def create_black_attention_mask(num_blocks: int,
                                block_size: int,
                                seq_len: int,
                                dtype=torch.float16):
    """
    :param num_blocks:
    :param block_size:
    :param seq_len:
    :param dtype:
    :return: [batch_size=1, num_blocks, num_heads=1, num_prefix_tokens=1, seq_len]
    """
    assert seq_len == num_blocks * block_size
    # noinspection PyTypeChecker
    base_mask = torch.tril(torch.ones([num_blocks, num_blocks], dtype=bool))
    mask = base_mask[:, :, None].expand(num_blocks, num_blocks, block_size).reshape(num_blocks, seq_len)
    mask = convert_mask_to_soft_mask(mask, dtype)[None, :, None, None, :]
    return mask


def zeros_like(shape, tensor):
    return torch.zeros(shape).type_as(tensor).to(tensor.device)
