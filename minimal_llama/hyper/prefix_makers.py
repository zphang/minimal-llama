import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
import minimal_llama.hyper.prefix_llama as prefix_llama


class PrefixWrapper(nn.Module):
    def __init__(self, num_tokens, num_layers, num_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.params = nn.Parameter(torch.empty([num_layers, 2, num_heads, num_tokens, head_dim], dtype=dtype))
        torch.nn.init.normal_(self.params)

    def forward(self, batch_size):
        prefixes = []
        for layer_i in range(self.num_layers):
            prefixes.append({
                "key": self.params[layer_i, 0][None].expand(batch_size, -1, -1, -1),
                "value": self.params[layer_i, 1][None].expand(batch_size, -1, -1, -1),
            })
        return prefixes


class PrefixMLPWrapper(nn.Module):
    def __init__(self, num_tokens, num_layers, num_heads, head_dim,
                 intermediate_size=1024,
                 dtype=torch.bfloat16):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        hidden_dim = num_heads * head_dim
        self.params = nn.Parameter(torch.empty([num_tokens, hidden_dim], dtype=dtype))
        self.f1 = nn.Linear(hidden_dim, intermediate_size, dtype=dtype)
        self.f2 = nn.Linear(intermediate_size, num_layers * 2 * hidden_dim, dtype=dtype)
        torch.nn.init.normal_(self.params)

    def forward(self, batch_size):
        prefixes = []
        # num_tokens, num_layers, 2, hidden_dim
        params = self.f2(F.tanh(self.f1(self.params)))
        # num_layers, 2, num_heads, num_tokens, head_dim
        params = params.view(
            self.num_tokens, self.num_layers, 2, self.num_heads, self.head_dim,
        ).permute(1, 2, 3, 0, 4)
        for layer_i in range(self.num_layers):
            prefixes.append({
                "key": params[layer_i, 0][None].expand(batch_size, -1, -1, -1),
                "value": params[layer_i, 1][None].expand(batch_size, -1, -1, -1),
            })
        return prefixes


def create_prefix_maker(num_tokens: int, config: prefix_llama.LLaMAConfig, prefix_type: str = "mlp"):
    if prefix_type == "plain":
        return PrefixWrapper(
            num_tokens=num_tokens,
            num_layers=config.n_layers,
            num_heads=config.n_heads,
            head_dim=config.head_dim,
        )
    elif prefix_type == "mlp":
        return PrefixMLPWrapper(
            num_tokens=num_tokens,
            num_layers=config.n_layers,
            num_heads=config.n_heads,
            head_dim=config.head_dim,
        )
    else:
        raise KeyError(prefix_type)