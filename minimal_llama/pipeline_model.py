import torch
import torch.nn as nn
import torch.nn.functional as F
from deepspeed.pipe import PipelineModule, LayerSpec
from minimal_llama.model import (
    ModelArgs,
    TransformerBlock,
    precompute_cos_sin,
    RMSNorm,
)


def loss_fn(logits, labels):
    return F.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        labels.view(-1),
    )


class PipelineLLaMA(PipelineModule):
    def __init__(self, model_args: ModelArgs, **kwargs):
        specs = [
            LayerSpec(InitialLayer, model_args=model_args),
        ]
        for layer_i in range(model_args.n_layers):
            specs.append(LayerSpec(PipelineTransformerBlock, model_args=model_args))
        specs.append(LayerSpec(FinalLayer, model_args=model_args))
        super().__init__(layers=specs, loss_fn=loss_fn, **kwargs)


class InitialLayer(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.model_args = model_args
        self.tok_embeddings = nn.Embedding(
            model_args.vocab_size, model_args.dim
        )

    def forward(self, tokens: torch.Tensor):
        hidden_state = self.tok_embeddings(tokens)
        return hidden_state


class PipelineTransformerBlock(TransformerBlock):
    def __init__(self, model_args: ModelArgs):
        super().__init__(args=model_args)
        self.cos_cached, self.sin_cached = precompute_cos_sin(
            model_args.max_seq_length, model_args.dim // model_args.n_heads,
            dtype=torch.float16,
            device="cpu",
        )
        self.mask = torch.full(
            (1, 1, model_args.max_seq_length, model_args.max_seq_length),
            float("-inf"),
        )
        self.mask = torch.triu(self.mask, diagonal=1)

    def forward(self, x):
        _, seq_len, _ = x.shape
        self.cos_cached = self.cos_cached.to(device=x.device, dtype=x.dtype)[:, :seq_len]
        self.sin_cached = self.sin_cached.to(device=x.device, dtype=x.dtype)[:, :seq_len]
        self.mask = self.mask.to(device=x.device, dtype=x.dtype)[:, :, :seq_len, :seq_len]
        return super().forward(x, self.cos_cached, self.sin_cached, self.mask)


class FinalLayer(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.vocab_size = model_args.vocab_size
        self.norm = RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

    def forward(self, hidden_state):
        hidden_state = self.norm(hidden_state)
        output = self.output(hidden_state)
        return output.view(-1, self.vocab_size)
