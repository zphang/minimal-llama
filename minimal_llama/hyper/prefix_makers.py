import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
import minimal_llama.hyper.prefix_llama as prefix_llama


class PrefixWrapper(nn.Module):
    def __init__(self, num_tokens, num_layers, num_heads, head_dim,
                 include_gates: bool = False,
                 dtype=torch.bfloat16):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.include_gates = include_gates
        self.params = nn.Parameter(torch.empty([num_layers, 2, num_heads, num_tokens, head_dim], dtype=dtype))
        torch.nn.init.normal_(self.params)

        if self.include_gates:
            self.gates = nn.Parameter(torch.empty([num_layers, num_heads], dtype=dtype))
            torch.nn.init.zeros_(self.gates)

    def forward(self, batch_size):
        prefixes = []
        for layer_i in range(self.num_layers):
            layer_prefix = {
                "key": self.params[layer_i, 0][None].expand(batch_size, -1, -1, -1),
                "value": self.params[layer_i, 1][None].expand(batch_size, -1, -1, -1),
            }
            if self.include_gates:
                layer_prefix["gate"] = self.gates[layer_i]
            prefixes.append(layer_prefix)
        return prefixes


class PrefixMLPWrapper(nn.Module):
    def __init__(self, num_tokens, num_layers, num_heads, head_dim,
                 include_gates: bool = False,
                 intermediate_size=1024,
                 dtype=torch.bfloat16):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.include_gates = include_gates
        hidden_dim = num_heads * head_dim
        self.params = nn.Parameter(torch.empty([num_tokens, hidden_dim], dtype=dtype))
        self.f1 = nn.Linear(hidden_dim, intermediate_size, dtype=dtype)
        self.f2 = nn.Linear(intermediate_size, num_layers * 2 * hidden_dim, dtype=dtype)
        torch.nn.init.normal_(self.params)

        if self.include_gates:
            self.gates = nn.Parameter(torch.empty([num_layers, num_heads], dtype=dtype))
            torch.nn.init.zeros_(self.gates)

    def forward(self, batch_size):
        prefixes = []
        # num_tokens, num_layers, 2, hidden_dim
        params = self.f2(F.tanh(self.f1(self.params)))
        # num_layers, 2, num_heads, num_tokens, head_dim
        params = params.view(
            self.num_tokens, self.num_layers, 2, self.num_heads, self.head_dim,
        ).permute(1, 2, 3, 0, 4)
        for layer_i in range(self.num_layers):
            layer_prefix = {
                "key": params[layer_i, 0][None].expand(batch_size, -1, -1, -1),
                "value": params[layer_i, 1][None].expand(batch_size, -1, -1, -1),
            }
            if self.include_gates:
                layer_prefix["gate"] = self.gates[layer_i]
            prefixes.append(layer_prefix)
        return prefixes


class HiddenStateMLPWrapper(nn.Module):
    def __init__(self, num_tokens, num_layers, num_heads, head_dim,
                 include_gates: bool = False,
                 apply_internal_gates: bool = False,
                 intermediate_size=1024,
                 dtype=torch.bfloat16):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = num_heads * head_dim
        self.include_gates = include_gates
        self.apply_internal_gates = apply_internal_gates
        hidden_dim = num_heads * head_dim
        self.params = nn.Parameter(torch.empty([num_tokens, hidden_dim], dtype=dtype))
        self.f1 = nn.Linear(hidden_dim, intermediate_size, dtype=dtype)
        self.f2 = nn.Linear(intermediate_size, num_layers * hidden_dim, dtype=dtype)
        torch.nn.init.normal_(self.params)

        if self.include_gates:
            self.gates = nn.Parameter(torch.zeros([num_layers, num_heads], dtype=dtype))
            torch.nn.init.zeros_(self.gates)

    def forward(self, batch_size):
        prefixes = []
        # num_tokens, num_layers, hidden_dim
        params = self.f2(F.tanh(self.f1(self.params)))
        # num_layers, num_heads, num_tokens, head_dim
        params = params.view(
            self.num_tokens, self.num_layers, self.hidden_dim
        ).permute(1, 0, 2)
        for layer_i in range(self.num_layers):
            layer_prefix = {
                "hidden_states": params[layer_i][None].expand(batch_size, -1, -1),
            }
            if self.include_gates:
                if self.apply_internal_gates:
                    layer_prefix = layer_prefix["hidden_states"] * self.gates[layer_i]
                else:
                    layer_prefix["gate"] = self.gates[layer_i]
            prefixes.append(layer_prefix)
        return prefixes


class GistMLPWrapper(nn.Module):
    def __init__(self, num_tokens, num_layers, num_heads, head_dim,
                 include_gates: bool = False,
                 intermediate_size=1024,
                 dtype=torch.bfloat16,):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_layers = num_layers - 1  # Skip first layer
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.include_gates = include_gates
        hidden_dim = num_heads * head_dim
        self.params = nn.Parameter(torch.empty([num_tokens, hidden_dim], dtype=dtype))
        self.f1 = nn.Linear(hidden_dim, intermediate_size, dtype=dtype)
        self.f2 = nn.Linear(intermediate_size, self.num_layers * 2 * hidden_dim, dtype=dtype)
        torch.nn.init.normal_(self.params)

        if self.include_gates:
            self.gates = nn.Parameter(torch.empty([self.num_layers, num_heads], dtype=dtype))
            torch.nn.init.zeros_(self.gates)

        self.cos = None
        self.sin = None
        self.layer0 = None

    def forward(self, batch_size):
        prefixes = [{
            "key": self.layer0["key"].detach().clone().expand(batch_size, -1, -1, -1),
            "value": self.layer0["value"].detach().clone().expand(batch_size, -1, -1, -1),
        }]
        # num_tokens, num_layers, 2, hidden_dim
        params = self.f2(F.tanh(self.f1(self.params)))
        # num_layers, 2, num_heads, num_tokens, head_dim
        params = params.view(
            self.num_tokens, self.num_layers, 2, self.num_heads, self.head_dim,
        ).permute(1, 2, 3, 0, 4)
        for layer_i in range(self.num_layers):
            key_states = params[layer_i, 0][None].expand(batch_size, -1, -1, -1)
            value_states = params[layer_i, 1][None].expand(batch_size, -1, -1, -1)
            key_states = prefix_llama.apply_single_rotary_pos_emb(key_states, self.cos, self.sin)
            layer_prefix = {
                "key": key_states,
                "value": value_states,
            }
            if self.include_gates:
                layer_prefix["gate"] = self.gates[layer_i]
            prefixes.append(layer_prefix)
        assert len(prefixes) == (self.num_layers + 1)
        return prefixes

    def setup_layer0_kv(self, model):
        with torch.no_grad():
            self.cos, self.sin, self.layer0 = self.create_layer0_kv(model)
        self.layer0["key"].requires_grad = False
        self.layer0["value"].requires_grad = False
        self.cos.requires_grad = False
        self.sin.requires_grad = False

    def create_layer0_kv(self, model):
        num_gist = self.num_tokens

        device = model.lm_head.weight.device
        gist_embed_ids = torch.LongTensor(torch.arange(
            32_000,
            32_000 + num_gist,
        )).to(device)[None]
        rope_embed_ids = prefix_llama.create_rope_embed_ids(input_ids=gist_embed_ids)
        cos, sin = model.get_cos_sin(rope_embed_ids)
        h = model.model.embed_tokens(gist_embed_ids)
        layer0 = model.model.layers[0]
        h = layer0.input_layernorm(h)
        key_states = layer0.self_attn.k_proj(h)
        value_states = layer0.self_attn.v_proj(h)
        key_states = key_states.view(
            1, num_gist, model.config.n_heads, model.config.head_dim,
        ).transpose(1, 2).contiguous()
        value_states = value_states.view(
            1, num_gist, model.config.n_heads, model.config.head_dim,
        ).transpose(1, 2).contiguous()
        key_states = prefix_llama.apply_single_rotary_pos_emb(key_states, cos=cos, sin=sin)
        return cos, sin, {
            "key": key_states,
            "value": value_states,
        }


class HiddenStateWrapper(nn.Module):
    def __init__(self, num_tokens, num_layers, num_heads, head_dim,
                 include_gates: bool = False,
                 dtype=torch.bfloat16):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.include_gates = include_gates
        self.params = nn.Parameter(torch.empty([num_layers, num_tokens, num_heads * head_dim], dtype=dtype))
        torch.nn.init.normal_(self.params)

        if self.include_gates:
            self.gates = nn.Parameter(torch.empty([num_layers, num_heads], dtype=dtype))
            torch.nn.init.zeros_(self.gates)

    def forward(self, batch_size):
        prefixes = []
        for layer_i in range(self.num_layers):
            layer_prefix = {
                "hidden_states": self.params[layer_i][None].expand(batch_size, -1, -1),
            }
            if self.include_gates:
                layer_prefix["gate"] = self.gates[layer_i]
            prefixes.append(layer_prefix)
        return prefixes


def create_prefix_maker(num_tokens: int, config: prefix_llama.LLaMAConfig, prefix_type: str = "mlp",
                        include_gates: bool = False, internal_gates: bool = False, model=None):
    if prefix_type == "plain":
        return PrefixWrapper(
            num_tokens=num_tokens,
            num_layers=config.n_layers,
            num_heads=config.n_heads,
            head_dim=config.head_dim,
            include_gates=include_gates,
        )
    elif prefix_type == "mlp":
        return PrefixMLPWrapper(
            num_tokens=num_tokens,
            num_layers=config.n_layers,
            num_heads=config.n_heads,
            head_dim=config.head_dim,
            include_gates=include_gates,
        )
    elif prefix_type == "hidden_states":
        return HiddenStateWrapper(
            num_tokens=num_tokens,
            num_layers=config.n_layers,
            num_heads=config.n_heads,
            head_dim=config.head_dim,
            include_gates=include_gates,
        )
    elif prefix_type == "hidden_states_mlp":
        return HiddenStateMLPWrapper(
            num_tokens=num_tokens,
            num_layers=config.n_layers,
            num_heads=config.n_heads,
            head_dim=config.head_dim,
            include_gates=include_gates,
            apply_internal_gates=internal_gates,
        )
    elif prefix_type == "gist":
        return GistMLPWrapper(
            num_tokens=num_tokens,
            num_layers=config.n_layers,
            num_heads=config.n_heads,
            head_dim=config.head_dim,
            include_gates=include_gates,
        )
    else:
        raise KeyError(prefix_type)
