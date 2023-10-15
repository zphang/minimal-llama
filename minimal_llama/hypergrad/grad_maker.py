import torch
import torch.nn as nn
import torch.nn.functional as F
from minimal_llama.hypergrad.llama_simple_jvp_peft import (
    LLaMAConfig, RMSNorm, MLP, check_nan,
    NoInitLinear, RotaryEmbedding, rotate_half,
    create_rope_embed_ids,
)


class GradMakerLayer(nn.Module):
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config
        self.cross_attn = Attention(config=config)
        self.mlp = MLP(config=config)
        self.peft_input_layernorm = RMSNorm(dim=config.dim, dtype=config.dtype)
        self.model_input_layernorm = RMSNorm(dim=config.dim, dtype=config.dtype)
        self.post_attention_layernorm = RMSNorm(dim=config.dim, dtype=config.dtype)

    def forward(
        self,
        peft_hidden_states,
        model_hidden_states,
        cos, sin,
        attention_mask,
    ):
        normed_peft_hidden_states = self.peft_input_layernorm(peft_hidden_states).to(self.config.dtype)
        normed_model_hidden_states = self.model_input_layernorm(model_hidden_states).to(self.config.dtype)
        check_nan(normed_model_hidden_states)
        raw_self_attn_output = self.cross_attn(
            peft_hidden_states=normed_peft_hidden_states,
            model_hidden_states=normed_model_hidden_states,
            cos=cos, sin=sin,
            attention_mask=attention_mask,
        )
        # [batch_size, seq_len, hidden_dim]
        peft_hidden_states = peft_hidden_states + raw_self_attn_output["attn_output"]
        check_nan(peft_hidden_states)
        # 2) FFN
        # [batch_size, seq_len, hidden_dim]
        peft_hidden_states = peft_hidden_states + self.mlp(
            self.post_attention_layernorm(peft_hidden_states),
        )
        check_nan(peft_hidden_states)
        return peft_hidden_states


def apply_rotary_pos_emb(k, cos, sin):
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return k_embed


class Attention(nn.Module):
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads

        self.q_proj = NoInitLinear(config.dim, config.dim, bias=False, dtype=config.dtype)
        self.k_proj = NoInitLinear(config.dim, config.dim, bias=False, dtype=config.dtype)
        self.v_proj = NoInitLinear(config.dim, config.dim, bias=False, dtype=config.dtype)
        self.o_proj = NoInitLinear(config.dim, config.dim, bias=False, dtype=config.dtype)
        self.rotary_emb = RotaryEmbedding(dim=self.head_dim, max_position_embeddings=config.max_seq_length)

    def forward(
        self,
        peft_hidden_states,
        model_hidden_states,
        cos, sin,
        attention_mask=None,
    ):
        _, p_seq_len, _ = peft_hidden_states.size()
        batch_size, m_seq_len, hidden_dim = model_hidden_states.size()

        # (batch_size, num_heads, q_seq_len, head_dim)
        query_states = self.q_proj(peft_hidden_states).view(
            batch_size, p_seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(model_hidden_states).view(
            batch_size, m_seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(model_hidden_states).view(
            batch_size, m_seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        key_states = apply_rotary_pos_emb(key_states, cos=cos, sin=sin)
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
            batch_size, p_seq_len, hidden_dim,
        )
        attn_output = self.o_proj(attn_output)
        check_nan(attn_output)
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


class SimpleGradMaker(nn.Module):
    def __init__(self, config: LLaMAConfig, num_peft_layers: int = 1, return_diff: bool = True):
        super().__init__()
        self.config = config
        self.num_peft_layers = num_peft_layers
        self.num_scalers = config.n_layers
        self.scalers_proj = nn.Linear(config.n_layers, config.dim, dtype=self.config.dtype)
        self.layers = nn.ModuleList([
            GradMakerLayer(config)
            for _ in range(num_peft_layers)
        ])
        self.scalers_up_proj = nn.Linear(config.dim, config.n_layers, dtype=self.config.dtype)
        self.return_diff = return_diff

    def forward(self, input_ids, peft_params, model_hidden_states: list):
        batch_size, peft_len, _ = peft_params[0]["hidden_states"].shape
        scalers = torch.stack([
            layer_peft_params["scaler"]
            for layer_peft_params in peft_params
        ], dim=-1)
        scalers_token = self.scalers_proj(scalers)[:, None, :]
        params_tokens = torch.stack([
            layer_peft_params["hidden_states"]
            for layer_peft_params in peft_params
        ], dim=1).transpose(0, 1).reshape(batch_size, self.config.n_layers * peft_len, self.config.dim)
        peft_hidden_states = torch.cat([
            scalers_token,
            params_tokens,
        ], dim=1)
        attention_mask = create_cross_mask(input_ids)
        cos, sin = self.get_cos_sin(create_rope_embed_ids(input_ids=input_ids))
        for i, layer in enumerate(self.layers):
            peft_hidden_states = layer(
                peft_hidden_states=peft_hidden_states,
                model_hidden_states=model_hidden_states[i],
                cos=cos, sin=sin, attention_mask=attention_mask,
            )
        scalers = self.scalers_up_proj(peft_hidden_states[:, 0, :])
        peft_hidden_states = peft_hidden_states[:, 1:, :].view(
            batch_size, self.config.n_layers, peft_len, self.config.dim,
        )
        new_peft_params = []
        for i in range(self.config.n_layers):
            layer_peft = {
                "scaler": scalers[:, i],
                "hidden_states": peft_hidden_states[:, i, :, :],
            }
            if self.return_diff:
                layer_peft["scaler"] = layer_peft["scaler"] - peft_params[i]["scaler"]
                layer_peft["hidden_states"] = layer_peft["hidden_states"] - peft_params[i]["hidden_states"]
            new_peft_params.append(layer_peft)
        return new_peft_params

    def get_cos_sin(self, rope_embed_ids):
        cos = F.embedding(
            rope_embed_ids,
            self.layers[0].cross_attn.rotary_emb.cos_cached[0, 0].to(rope_embed_ids.device)
        ).to(self.config.dtype)
        sin = F.embedding(
            rope_embed_ids,
            self.layers[0].cross_attn.rotary_emb.sin_cached[0, 0].to(rope_embed_ids.device)
        ).to(self.config.dtype)
        cos, sin = cos[:, None, :, :], sin[:, None, :, :]
        return cos, sin


def create_cross_mask(input_ids, pad_token_id=0):
    is_valid = (input_ids != pad_token_id)
    return is_valid[:, None, None, :]
