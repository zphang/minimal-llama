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
            specs.append(LayerSpec(PipelineTransformerBlock, args=model_args))
        specs.append(LayerSpec(FinalLayer, model_args=model_args))
        super().__init__(layers=specs, loss_fn=loss_fn, **kwargs)


class InitialLayer(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.model_args = model_args
        self.tok_embeddings = nn.Embedding(
            model_args.vocab_size, model_args.dim
        )
        self.cos_cached, self.sin_cached = precompute_cos_sin(
            model_args.max_seq_length, model_args.dim // model_args.n_heads,
            dtype=self.tok_embeddings.weight.dtype,
            device=self.tok_embeddings.weight.device,
        )

    def forward(self, tokens: torch.Tensor):
        _bsz, seq_len = tokens.shape
        hidden_state = self.tok_embeddings(tokens)
        cos = self.cos_cached[:, :seq_len].to(hidden_state.dtype)
        sin = self.sin_cached[:, :seq_len].to(hidden_state.dtype)
        mask = torch.full(
            (1, 1, seq_len, seq_len),
            float("-inf"), device=tokens.device,
        )
        mask = torch.triu(mask, diagonal=1).type_as(hidden_state)
        return hidden_state, cos, sin, mask


class PipelineTransformerBlock(TransformerBlock):
    def forward(self, inputs):
        x, cos, sin, mask = inputs
        cos, sin = cos.to(x.device), sin.to(x.device)
        return super().forward(x, cos, sin, mask), cos, sin, mask


class FinalLayer(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.vocab_size = model_args.vocab_size
        self.norm = RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

    def forward(self, inputs):
        hidden_state, _, _, _ = inputs
        hidden_state = self.norm(hidden_state)
        output = self.output(hidden_state)
        return output.view(-1, self.vocab_size)
