from functools import partial
import os
import json
from pathlib import Path
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass
from typing import Optional, Tuple
from socket import gethostname
import pickle

from einops import rearrange

import torch
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
from random import shuffle

from apex.transformer import parallel_state
from apex.transformer import tensor_parallel
from apex.transformer.pipeline_parallel import get_forward_backward_func, build_model
from apex.transformer.pipeline_parallel.utils import (
    average_losses_across_data_parallel_group,
    setup_microbatch_calculator,
    _reconfigure_microbatch_calculator,
)

from apex.contrib.optimizers.distributed_fused_adam import DistributedFusedAdam
from apex.optimizers.fused_adam import FusedAdam

from datasets import load_dataset
import wandb
from tqdm import tqdm

import torch._dynamo
import warnings
warnings.filterwarnings("ignore")

torch._dynamo.allow_in_graph(rearrange)


def identity(x):
    return x


torch.compile = identity
torch._dynamo.config.cache_size_limit = 100

SP = True


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: Optional[int] = None
    vocab_size: int = 1024  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, dtype=torch.float32):
        super().__init__()
        self.eps = torch.tensor(eps)
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    @torch.compile
    def _norm(x, eps, weight):
        out = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).type_as(x)
        return out * weight

    def forward(self, x):
        return RMSNorm._norm(x, self.eps, self.weight)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def precompute_freqs(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return torch.view_as_real(freqs_cis)


def reshape_for_broadcast(freqs, x_shape):
    ndim = len(x_shape)
    assert 0 <= 1 < ndim
    assert freqs.shape == (
        x_shape[1],
        x_shape[-2],
        x_shape[-1],
    ), f"{freqs.shape=} not compatible with {x_shape=}"
    shape = [d if i == 1 or i >= ndim - 2 else 1 for i, d in enumerate(x_shape)]
    return freqs.view(*shape)


def cmul(x, y):
    return torch.stack(
        [
            x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1],
            x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0],
        ],
        dim=-1,
    )


@torch.compile
def apply_rotary_emb(
        x: torch.Tensor,
        freqs: torch.Tensor,
) -> torch.Tensor:
    x_ = x.float().reshape(*x.shape[:-1], -1, 2)
    x_out = cmul(x_, freqs).flatten(3)
    return x_out.type_as(x)


def add_bias(x: Tuple[torch.tensor, Optional[torch.Tensor]]):
    x, bias = x
    if bias is not None:
        x = x + bias
    return x


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, dtype: torch.dtype = torch.float32):
        super().__init__()
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        assert args.n_heads % tp_size == 0
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads // tp_size
        self.n_local_kv_heads = self.n_kv_heads // tp_size
        self.head_dim = args.dim // args.n_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.wq = tensor_parallel.ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            params_dtype=dtype,
            sequence_parallel_enabled=SP,
            no_async_tensor_model_parallel_allreduce=SP,
        )
        self.wk = tensor_parallel.ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            params_dtype=dtype,
            sequence_parallel_enabled=SP,
            no_async_tensor_model_parallel_allreduce=SP,
        )
        self.wv = tensor_parallel.ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            params_dtype=dtype,
            sequence_parallel_enabled=SP,
            no_async_tensor_model_parallel_allreduce=SP,
        )
        self.wo = tensor_parallel.RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            params_dtype=dtype,
            sequence_parallel_enabled=SP,
        )

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            kv_freqs: torch.Tensor,
            q_freqs: torch.Tensor,
            mask: Optional[torch.Tensor],
    ):
        seqlen, bsz, _ = x.shape
        x = x.contiguous()

        xq, xk, xv = add_bias(self.wq(x)), add_bias(self.wk(x)), add_bias(self.wv(x))

        xq = rearrange(xq, "s b (nh hd) -> b s nh hd", nh=self.n_local_heads)
        xk = rearrange(xk, "s b (nh hd) -> b s nh hd", nh=self.n_local_kv_heads)
        xv = rearrange(xv, "s b (nh hd) -> b s nh hd", nh=self.n_local_kv_heads)
        xk = apply_rotary_emb(xk, freqs=kv_freqs)
        xq = apply_rotary_emb(xq, freqs=q_freqs)

        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xk = rearrange(xk, "b s nh hd -> b nh s hd")
        xv = rearrange(xv, "b s nh hd -> b nh s hd")
        xq = rearrange(xq, "b s nh hd -> b nh s hd")

        with torch.backends.cuda.sdp_kernel(
                enable_math=False, enable_flash=True, enable_mem_efficient=False
        ):
            output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
            output = rearrange(output, "b nh s hd -> s b (nh hd)").contiguous()
            return add_bias(self.wo(output))


@torch.compile
def gated_silu(x, gate):
    return F.silu(x) * gate


class FeedForward(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            multiple_of: int,
            ffn_dim_multiplier: Optional[float],
            dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = tensor_parallel.ColumnParallelLinear(
            dim,
            hidden_dim,
            bias=False,
            gather_output=False,
            params_dtype=dtype,
            sequence_parallel_enabled=SP,
            no_async_tensor_model_parallel_allreduce=SP,
        )
        self.w2 = tensor_parallel.RowParallelLinear(
            hidden_dim,
            dim,
            bias=False,
            input_is_parallel=True,
            params_dtype=dtype,
            sequence_parallel_enabled=SP,
        )
        self.w3 = tensor_parallel.ColumnParallelLinear(
            dim,
            hidden_dim,
            bias=False,
            gather_output=False,
            params_dtype=dtype,
            sequence_parallel_enabled=SP,
            no_async_tensor_model_parallel_allreduce=SP,
        )

    def forward(self, x):
        return add_bias(self.w2(gated_silu(add_bias(self.w1(x)), add_bias(self.w3(x)))))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs, dtype: torch.dtype):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args, dtype=dtype)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            dtype=dtype,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps, dtype=dtype)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps, dtype=dtype)

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            kv_freqs: torch.Tensor,
            q_freqs: torch.Tensor,
            mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, kv_freqs, q_freqs, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class SplitLlama(nn.Module):
    def __init__(self, args: ModelArgs, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        self.pp_world = parallel_state.get_pipeline_model_parallel_world_size()
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self.tp_world = parallel_state.get_tensor_model_parallel_world_size()

        curr_rank_layers = args.n_layers // self.pp_world
        start_layer = self.pp_rank * curr_rank_layers

        self.layers = nn.ModuleList(
            [TransformerBlock(i + start_layer, args, dtype) for i in range(curr_rank_layers)]
        )
        self.freqs = precompute_freqs(args.dim // args.n_heads, args.max_seq_len * 2)

        if self.pp_rank == 0:
            print("tok embeds of size", args.vocab_size, args.dim)
            self.tok_embeddings = tensor_parallel.VocabParallelEmbedding(
                args.vocab_size, args.dim, params_dtype=dtype
            )

        if self.pp_rank == self.pp_world - 1:
            self.output = tensor_parallel.ColumnParallelLinear(
                args.dim,
                args.vocab_size,
                bias=False,
                params_dtype=dtype,
                gather_output=False,
                sequence_parallel_enabled=SP,
                no_async_tensor_model_parallel_allreduce=SP,
            )
            self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.args = args

    # factored out for torch.compile
    @torch.compile
    def transformer_block(self, x, start_pos, kv_freqs, q_freqs, mask):
        for layer in self.layers:
            x = layer(x, start_pos, kv_freqs, q_freqs, mask)
        return x

    def forward(self, tokens_or_hidden_state: torch.Tensor, start_pos: int):
        if self.pp_rank == 0:
            x = self.tok_embeddings(tokens_or_hidden_state)
            x = rearrange(x, "b s d -> s b d")
            if SP:
                x = tensor_parallel.mappings.scatter_to_sequence_parallel_region(x)
        else:
            x = tokens_or_hidden_state

        seq_len, batch_size, _ = x.shape
        total_seq_len = seq_len * self.tp_world if SP else seq_len

        mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(x)

        kv_freqs = self.freqs[start_pos: start_pos + total_seq_len].to(x.device)
        sp_n_queries = seq_len // self.tp_world
        q_freqs = kv_freqs
        # q_freqs = self.freqs[start_pos + sp_n_queries * self.tp_rank : start_pos + sp_n_queries * (self.tp_rank + 1)].to(x.device)
        head_dim = self.args.dim // self.args.n_heads
        kv_shape = (batch_size, total_seq_len, self.args.n_heads, head_dim // 2, 2)
        q_shape = (batch_size, total_seq_len, self.args.n_heads, head_dim // 2, 2)
        kv_freqs = reshape_for_broadcast(kv_freqs, kv_shape).to(x.device)
        q_freqs = reshape_for_broadcast(q_freqs, q_shape).to(x.device)
        x = self.transformer_block(x, start_pos, kv_freqs, q_freqs, mask=None)

        if self.pp_rank == self.pp_world - 1:
            x = self.norm(x)
            x = add_bias(self.output(x))
            return x
        else:
            return x


class PipelineStage(nn.Module):
    input_tensors: Optional[List[torch.Tensor]] = None

    def __init__(self, module):
        super().__init__()
        self.input_tensors = None
        self.wrapped = module

    def set_input_tensor(self, tensor: List[torch.Tensor]):
        self.input_tensors = tensor

    def forward(self, *x, **kwargs):
        if parallel_state.is_pipeline_first_stage():
            inputs = x
        else:
            inputs = self.input_tensors
        return self.wrapped(*inputs, **kwargs)


def model_provider_func(llama_args, *args, **kwargs):
    return PipelineStage(SplitLlama(llama_args, dtype=torch.bfloat16))


def loss_func(pred, label):
    label = rearrange(label, "b s -> s b").contiguous()
    # logits = rearrange(pred, "s b n -> b n s").contiguous()
    logits = tensor_parallel.gather_from_tensor_model_parallel_region(
        pred
    )
    # loss = tensor_parallel.vocab_parallel_cross_entropy(pred, label).mean()
    # print(label.shape, logits.shape)
    loss = F.cross_entropy(logits.view(-1, logits.shape[-1]).contiguous(), label.view(-1).contiguous())
    averaged_loss = average_losses_across_data_parallel_group([loss])
    return loss, {"nice_loss": averaged_loss}


def train_forward_step_func(batch, model):
    input, label = batch
    out = model(input, start_pos=0)
    return out.contiguous(), lambda pred: loss_func(pred.float(), label)


def inference_forward_step_func(batch, model):
    (input,) = batch
    out = model(input, start_pos=0)
    return out.contiguous(), lambda pred: (pred, {"logits": pred})


# from apex
def set_random_seed(seed: int):
    """Set random seed for reproducability."""
    # Ensure that different pipeline MP stages get different seeds.
    # TP seeds are automatically offset by the TP rank by apex.

    seed = seed + (100 * parallel_state.get_pipeline_model_parallel_rank())
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tensor_parallel.model_parallel_cuda_manual_seed(seed)


def convert_llama_state_dict(
        args: ModelArgs,
        state_dict,
        tp_rank: int,
        tp_world: int,
        pp_rank: int,
        pp_world: int,
        add_new_tokens=0,
        map_dtype=torch.bfloat16,
):
    state_dict = state_dict.copy()
    state_dict.pop("rope.freqs")
    # in original code, token embeddings are sharded across latent dim, but apex shards them along vocab dim
    if pp_rank == 0:
        tok_embeds = state_dict["tok_embeddings.weight"].cuda()
        full_embeds = tensor_parallel.gather_from_tensor_model_parallel_region(tok_embeds)
        if add_new_tokens > 0:
            full_embeds = torch.cat([full_embeds, torch.randn(add_new_tokens, args.dim).cuda()])
        local_vocab_size = args.vocab_size // tp_world
        tok_embeds = full_embeds[tp_rank * local_vocab_size: (tp_rank + 1) * local_vocab_size]
        state_dict["tok_embeddings.weight"] = tok_embeds.cpu()
    else:
        state_dict.pop("tok_embeddings.weight")

    if pp_rank != (pp_world - 1):
        state_dict.pop("norm.weight")
        state_dict.pop("output.weight")
    else:
        if add_new_tokens > 0:
            output_weight = rearrange(state_dict["output.weight"].cuda(), "hidden vocab -> vocab hidden").contiguous()
            full_embeds = tensor_parallel.gather_from_tensor_model_parallel_region(output_weight)
            full_embeds = torch.cat(
                [full_embeds, torch.zeros((args.dim, add_new_tokens)).cuda()], dim=1
            )
            local_vocab_size = args.vocab_size // tp_world
            local_embeds = full_embeds[:, tp_rank * local_vocab_size: (tp_rank + 1) * local_vocab_size]
            state_dict["output.weight"] = rearrange(local_embeds.cpu(), "vocab hidden -> hidden vocab")

    def offset_layer_idx(name):
        stage_layers = args.n_layers // pp_world
        if name.startswith("layers."):
            layer_idx = int(name.split(".")[1])
            if pp_rank * stage_layers <= layer_idx < (pp_rank + 1) * stage_layers:
                new_layer_idx = layer_idx - pp_rank * stage_layers
                return name.replace(f"layers.{layer_idx}", f"layers.{new_layer_idx}")
            else:
                return None
        else:
            return name

    state_dict = {
        offset_layer_idx(k): v for k, v in state_dict.items() if offset_layer_idx(k) is not None
    }

    state_dict = {("module.wrapped." + k): v for k, v in state_dict.items()}
    state_dict = {k: v.to(dtype=map_dtype).cpu() for k, v in state_dict.items()}
    return state_dict


from sentencepiece import SentencePieceProcessor
from logging import getLogger
from typing import List
import os
import numpy as np

logger = getLogger()


def preprocess(instruction: str, input: str, output: str):
    """Build Alpaca prompt and output from instruction and input/output examples"""
    if input:
        prefix = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request."
        )
        prompt = f"### System:\n{instruction}\n\n### User:\n{input}\n\n### Assistant:\n"
        return [prompt, output]
    else:
        prefix = (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request."
        )
        prompt = f"### User:\n{instruction}\n\n### Assistant:\n"
        return [prompt, output]


def packed_dataset(tokenizer, dataset: str, num_passes: int = 4):
    cache = Path(f"{dataset.replace('/', '--')}_packed.pkl")
    if cache.exists():
        with open(cache, 'rb') as f:
            return pickle.load(f)
    if torch.distributed.get_rank() == 0:
        ds = load_dataset(dataset, split="train")
        turns = list()
        for iteration in range(num_passes):
            for item in tqdm(ds):
                prompt, output = preprocess(item["system"], item["prompt"], item["response"])
                prompt = [1] + tokenizer.encode(prompt)  # add bos first...
                output = tokenizer.encode(output)
                input = prompt + output + [2]
                mask = [-100 for _ in range(len(prompt) - 1)] + output + [2]
                if len(prompt) < 2000:
                    # turns.append([input, mask])
                    # if len(turns[-1][0]) < 2048:
                    #     turns[-1][0].extend([0 for _ in range(ModelArgs.max_seq_len - len(turns[-1][0]))])
                    #     turns[-1][1].extend([-100 for _ in range(ModelArgs.max_seq_len - len(turns[-1][1]))])
                    # elif len(turns[-1][1]) < 2048:
                    #     # Since it's one less, could and has hit 2047 exactly here, ask me how I know -_-
                    #     turns[-1][1].extend([-100 for _ in range(ModelArgs.max_seq_len - len(turns[-1][1]))])
                    # elif len(turns[-1][0]) > 2048:
                    #     turns[-1][0] = turns[-1][0][:2048]
                    #     turns[-1][1] = turns[-1][1][:2048]
                    if len(turns) == 0:
                        turns.append([input, mask])
                    else:
                        if len(turns[-1][0]) + len(input) <= ModelArgs.max_seq_len:
                            turns[-1][0].extend(input)
                            turns[-1][1].extend([-100] + mask)
                        else:
                            if len(turns[-1][1]) == ModelArgs.max_seq_len:
                                turns.append([input, mask])
                            elif len(turns[-1][0]) == ModelArgs.max_seq_len:
                                turns[-1][1].extend([-100 for _ in range(ModelArgs.max_seq_len - len(turns[-1][1]))])
                                turns.append([input, mask])
                            else:
                                turns[-1][0].extend([0 for _ in range(ModelArgs.max_seq_len - len(turns[-1][0]))])
                                turns[-1][1].extend([-100 for _ in range(ModelArgs.max_seq_len - len(turns[-1][1]))])
                                turns.append([input, mask])
                    turns[-1][0] = turns[-1][0][:2048]
                    turns[-1][1] = turns[-1][1][:2048]
        if len(turns[-1][0]) < 2048:
            turns[-1][0].extend([0 for _ in range(ModelArgs.max_seq_len - len(turns[-1][0]))])
            turns[-1][1].extend([-100 for _ in range(ModelArgs.max_seq_len - len(turns[-1][1]))])
        elif len(turns[-1][1]) < 2048:
            turns[-1][1].extend([-100 for _ in range(ModelArgs.max_seq_len - len(turns[-1][1]))])
        print(f"Saving {dataset} to {cache}")
        shuffle(turns)
        with open(cache, "wb") as f:
            pickle.dump(turns, f)
    torch.distributed.barrier()
    with open(cache, 'rb') as f:
        return pickle.load(f)

def sample_random_chunks(data, chunk_size, batch_size, global_batch_size, dp_rank):
    # generator = torch.Generator()
    # generator.manual_seed(seed)

    data_len = len(data)
    i = batch_size * dp_rank
    while True:
        chunks = (
            torch.stack([torch.from_numpy(np.array(data[i % data_len][0]).copy().astype(np.int64)) for i in range(i, i + batch_size)]),
            torch.stack([torch.from_numpy(np.array(data[i % data_len][1]).copy().astype(np.int64)) for i in range(i, i + batch_size)]),
        )
        yield chunks
        i += global_batch_size
        if i > data_len:
            i = i - data_len


def inference(models, tok, texts: list[str], llama_args: ModelArgs, micro_batch_size: int, rank: int,
              forward_backward_func, n: int, global_batch_size: int, data_parallel_size: int, stream=False):
    prompts = [tok.encode(text, add_bos=True, add_eos=False) for text in texts]
    prompt_lengths = [len(p) for p in prompts]
    inputs = [p + [tok.eos_id()] * (llama_args.max_seq_len - len(p)) for p in prompts]

    inputs = torch.tensor(inputs).long().cuda()

    _reconfigure_microbatch_calculator(
        rank=rank,
        rampup_batch_size=None,
        global_batch_size=len(prompts) * data_parallel_size,
        micro_batch_size=len(prompts),
        data_parallel_size=data_parallel_size,
    )

    with torch.no_grad():
        for i in range(n):
            output = forward_backward_func(
                inference_forward_step_func,
                [inputs],
                models,
                forward_only=True,
                tensor_shape=(inputs.shape[1], micro_batch_size, llama_args.dim),
                dtype=torch.bfloat16,
            )
            if parallel_state.is_pipeline_last_stage():
                logits = torch.cat([o["logits"] for o in output], dim=1).float()

                logits = rearrange(logits, "s b n -> b s n").contiguous()
                logits = tensor_parallel.gather_from_tensor_model_parallel_region(
                    logits
                )

                # vocab is padded to maximize performance
                logits = logits[:, :, :tok.vocab_size()]
                logprobs = torch.nn.functional.softmax(logits / 0.7, dim=-1)
                for i, l in enumerate(prompt_lengths):
                    # new_tok = torch.multinomial(logprobs[i, l - 1], 1)
                    new_tok = torch.argmax(logprobs[i, l - 1])
                    print(f"{new_tok=}")
                    inputs[i, l] = new_tok
                src = parallel_state.get_pipeline_model_parallel_last_rank()
                group = parallel_state.get_embedding_group()
                torch.distributed.broadcast(inputs, src, group)
            elif parallel_state.is_pipeline_first_stage():
                src = parallel_state.get_pipeline_model_parallel_last_rank()
                group = parallel_state.get_embedding_group()
                torch.distributed.broadcast(inputs, src, group)

            prompt_lengths = [l + 1 for l in prompt_lengths]

            if stream:
                print([tok.decode(p[:l].cpu().numpy().tolist()) for p, l in zip(inputs, prompt_lengths)])
                yield [tok.decode(p[:l].cpu().numpy().tolist()) for p, l in zip(inputs, prompt_lengths)]

    _reconfigure_microbatch_calculator(
        rank=rank,
        rampup_batch_size=None,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        data_parallel_size=data_parallel_size,
    )

    return [tok.decode(p.cpu().numpy().tolist()) for p in inputs]


import gradio as gr


def inference_server(
        llama: Path,
        tokenizer: Path,
        tp_world: int,
        pp_world: int,
):
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["WORLD_SIZE"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    print(f"hi from {rank}/{world_size} on {gethostname()}", flush=True)

    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)

    parallel_state.initialize_model_parallel(
        tp_world,
        pp_world,
        None
    )

    tok = SentencePieceProcessor(str(tokenizer))

    world_size = torch.distributed.get_world_size()
    data_parallel_size = world_size // (tp_world * pp_world)

    with open(llama / "params.json") as f:
        params = json.load(f)
        params["vocab_size"] = tok.vocab_size()
    llama_args = ModelArgs(**params)

    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()

    # state_dict = torch.load(llama / f"tp-consolidated.{tp_rank:02d}.pth")

    # saved with DDP
    def trim_module(name):
        return name[len("module."):]

    # state_dict = {trim_module(k): v for k, v in state_dict.items()}

    state_dict = torch.load(llama / f"consolidated.{tp_rank:02d}.pth")

    state_dict = convert_llama_state_dict(
        llama_args, state_dict, tp_rank=tp_rank, tp_world=tp_world, pp_rank=pp_rank, pp_world=pp_world,
        add_new_tokens=0)

    state_dict = {trim_module(k): v for k, v in state_dict.items()}

    global_batch_size = 1
    micro_batch_size = 1

    setup_microbatch_calculator(
        rank=rank,
        rampup_batch_size=None,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        data_parallel_size=data_parallel_size,
    )

    set_random_seed(2023)

    forward_backward_func = get_forward_backward_func(
        None, pp_world
    )
    print(f"{forward_backward_func=}")

    model_kwargs = dict(llama_args=llama_args)
    wrap_with_ddp = False

    print(model_kwargs, flush=True)
    models = build_model(
        model_provider_func,
        wrap_with_ddp,
        virtual_pipeline_model_parallel_size=None,
        **model_kwargs,
    )

    models[0].load_state_dict(state_dict)
    del state_dict

    # First rank runs the python server, awaiting requests.
    # Other ranks wait on broadcast of tokens from first rank.

    def infer_text(text: Optional[str]):
        if text is not None:
            toks = tok.encode(text)
            padded = toks + [tok.eos_id()] * (llama_args.max_seq_len - len(toks))
            padded = padded[:llama_args.max_seq_len]
            toks = torch.tensor(padded).long().cuda()
        else:
            toks = torch.empty((llama_args.max_seq_len,)).long().cuda()
        torch.distributed.broadcast(toks, 0)
        recv_text = tok.decode(toks.cpu().numpy().tolist())
        print(f"{rank=}: {recv_text} {len(recv_text)=}", flush=True)
        inferred = inference(models, tok, [recv_text], llama_args, 1, rank, forward_backward_func, 32, 1, 1,
                             stream=True)
        yield from (result[0] for result in inferred)

    system_prompt = """<|SYSTEM|># StableAssistant - StableAssistant is A helpful and harmless Open Source AI Language Model developed by Stability and CarperAI. - StableAssistant is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user. - StableAssistant is more than just an information source, StableAssistant is also able to write poetry, short stories, and make jokes. - StableAssistant will refuse to participate in anything that could harm a human."""

    system_prompt = """The following is a description of an AI assistant followed by a transcript of a conversation with this AI assistant.\n\n# StableAssistant - StableAssistant is A helpful and harmless Open Source AI Language Model developed by Stability and CarperAI. - StableAssistant is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user. - StableAssistant is more than just an information source, StableAssistant is also able to write poetry, short stories, and make jokes. - StableAssistant will refuse to participate in anything that could harm a human.\n\n"""

    # test_inference = inference(models, tok, [system_prompt], llama_args, 1, rank, forward_backward_func, 32, 1, 1, stream=False)
    # print(f"{rank=}: {test_inference}", flush=True)
    # for i in test_inference:
    #    print("char", i, flush=True)

    result = "".join(infer_text("I don't know much about Hungarian underground"))
    print(f"{result=}")
    return

    if rank == 0:
        with gr.Blocks() as demo:
            chatbot = gr.Chatbot()
            msg = gr.Textbox()
            clear = gr.Button("Clear")

            def respond(message, chat_history):
                str_history = "".join([f"<|USER|>{m}<|ASSISTANT|>{b}" for m, b in chat_history])
                str_history = "".join([f"### User: {m}\n\n### Assistant{b}\n\n" for m, b in chat_history])
                # prompt = system_prompt + str_history + f"<|USER|>{message}<|ASSISTANT|>"
                prompt = system_prompt + str_history + f"### User: {message}\n\n### Assistant: "
                for bot_message in infer_text(prompt):
                    print(bot_message, flush=True)
                    new_part = bot_message[len(prompt):]
                    first_message, *rest = new_part.split("\n\n")

                    yield "", chat_history + [(message, first_message)]
                    if rest:
                        break

            msg.submit(respond, [msg, chatbot], [msg, chatbot])
            clear.click(lambda: None, None, chatbot, queue=False)
        demo.queue()
        demo.launch(share=True)
    else:
        while True:
            [*infer_text(None)]


def main(llama: Path, tokenizer: Path, tp_world: int, pp_world: int, save_to: Path, distributed_adam=True):
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["WORLD_SIZE"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    assert gpus_per_node == torch.cuda.device_count()
    print(f"hi from {rank}/{world_size} on {gethostname()}", flush=True)

    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    if tp_world == 1:
        global SP
        SP = False

    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)

    tensor_model_parallel_size = tp_world
    pipeline_model_parallel_size = pp_world
    virtual_pipeline_model_parallel_size = None

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size,
    )

    world_size = torch.distributed.get_world_size()
    data_parallel_size: int = world_size // (
            tensor_model_parallel_size * pipeline_model_parallel_size
    )
    tok = SentencePieceProcessor(str(tokenizer))

    with open(llama / "params.json") as f:
        params = json.load(f)

    # round vocab size to nearest multiple of 256
    vocab_size = 256 * ((tok.vocab_size() + 255) // 256)
    llama_args = ModelArgs(**dict(params, vocab_size=vocab_size))

    if rank == (world_size - 1):
        wandb.init(
            project="tinypar",
            name="llama",
            config=llama_args.__dict__,
        )

    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()

    try:
        state_dict = torch.load(llama / f"consolidated.{tp_rank:02d}.pth", map_location=torch.device('cpu'))
        state_dict = convert_llama_state_dict(
            llama_args,
            state_dict,
            tp_rank,
            tensor_model_parallel_size,
            pp_rank,
            pipeline_model_parallel_size,
            add_new_tokens=vocab_size - 32000
        )
    except FileNotFoundError:
        print(f"no checkpoint found")
        state_dict = None

    torch.backends.cudnn.benchmark = True

    global_batch_size = 256
    micro_batch_size = 1

    setup_microbatch_calculator(
        rank=rank,
        rampup_batch_size=None,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        data_parallel_size=data_parallel_size,
    )

    set_random_seed(2023)

    forward_backward_func = get_forward_backward_func(
        virtual_pipeline_model_parallel_size, pipeline_model_parallel_size
    )
    print(f"{forward_backward_func=}")

    model_kwargs = dict(llama_args=llama_args)
    wrap_with_ddp = True

    print(model_kwargs, flush=True)
    models = build_model(
        model_provider_func,
        wrap_with_ddp,
        virtual_pipeline_model_parallel_size,
        **model_kwargs,
    )
    print("built model", flush=True)

    if state_dict is not None:
        models[0].load_state_dict(state_dict)
        del state_dict
        print("loaded state dict", flush=True)

    local_rank = torch.cuda.current_device()

    # optimizer = torch.optim.AdamW(models[0].parameters(), lr=1e-4, weight_decay=0.1)
    if distributed_adam:
        optimizer = DistributedFusedAdam(
            models[0].parameters(),
            lr=0.0,  # * (global_batch_size / 128),
            weight_decay=1e-6,
            betas=(0.9, 0.95),
            process_group=parallel_state.get_data_parallel_group(),
            dtype=torch.bfloat16,
            # distributed_process_group=torch.distributed.new_group(ranks=[torch.distributed.get_rank()]),
            # redundant_process_group=parallel_state.get_data_parallel_group(),
            store_params=False,
        )
    else:
        optimizer = FusedAdam(
            models[0].parameters(),
            lr=1e-4,  # * (global_batch_size / 128),
            weight_decay=0.1,
            adam_w_mode=True,
        )

    print("constructed opt", flush=True)
    dp_rank = parallel_state.get_data_parallel_rank()
    tp_group = parallel_state.get_tensor_model_parallel_group()
    dp_group = parallel_state.get_data_parallel_group()
    print(f"{rank}: dp:{dp_rank}, tp:{tp_rank}, pp: {pp_rank}")

    total_params_for_rank = sum(p.numel() for p in models[0].parameters())
    total_params_world = torch.tensor(total_params_for_rank).cuda()

    torch.distributed.all_reduce(total_params_world, op=torch.distributed.ReduceOp.SUM)
    total_params = total_params_world.item() / data_parallel_size
    if rank == 0:
        print(f"total params: {total_params / 1e9:.2f}B", flush=True)

    io_shape = (llama_args.max_seq_len, micro_batch_size, llama_args.dim)
    approx_model_flops = 6 * global_batch_size * llama_args.max_seq_len * total_params

    if rank == 0:
        print(f"start {io_shape}", flush=True)

    t = time.time()

    dt = time.time() - t

    data = packed_dataset(tok, "CarperAI/orca-chatgpt-500k-cleaned")

    rank_batch = global_batch_size // data_parallel_size
    total_samples = len(data)
    print(f"{total_samples=}", flush=True)
    total_steps = total_samples // global_batch_size
    step = 0
    num_tokens = 0
    if rank == (world_size - 1):
        wandb.define_metric("num_tokens")
        wandb.define_metric("loss", step_metric="num_tokens")
    lr = 0.0
    true_time = time.time()
    for batch in sample_random_chunks(data, llama_args.max_seq_len + 1, rank_batch, global_batch_size, dp_rank):
        optimizer.zero_grad()
        inputs, labels = batch[0].to(local_rank), batch[1].to(local_rank)
        t = time.time()
        loss = forward_backward_func(
            train_forward_step_func,
            [inputs, labels],
            models,
            forward_only=False,
            tensor_shape=io_shape,
            dtype=torch.bfloat16,
            async_comm=True,  # pp_world == 0,
            sync_batch_comm=False,
            sequence_parallel_enabled=SP,
        )
        num_tokens += inputs.numel() * data_parallel_size

        dt = time.time() - t
        tdt = time.time() - true_time
        if rank == (world_size - 1):
            print(f"step {step}/{total_steps}, est to completion: {tdt * (total_steps-step)}", flush=True)
            true_time = time.time()
            print(f"tflops: {approx_model_flops / (dt * world_size) / 1e12=}", flush=True)
            memory_usage_gb = torch.cuda.max_memory_allocated() / 1e9
            print(f"memory usage: {memory_usage_gb=}", flush=True)
            samples_per_sec = global_batch_size / dt
            print(f"throughput: {samples_per_sec=}", flush=True)
            print(f"{len(loss)=}", flush=True)
            loss = [d["nice_loss"] for d in loss]
            mean_loss = torch.mean(torch.stack(loss).detach())
            print(f"{mean_loss=}", flush=True)
            wandb.log(dict(
                loss=mean_loss.item(),
                throughput=samples_per_sec,
                memory_usage=memory_usage_gb,
                tflops=approx_model_flops / (dt * world_size) / 1e12,
                num_tokens=num_tokens,
                tokens_per_sec=inputs.numel() / dt,
                lr=lr
            ))

        # All-reduce RMSNorm grads over sequence dimension
        rmsnorms = [m for _, m in models[0].named_modules() if isinstance(m, RMSNorm)]
        rmsnorm_grads = [param.grad for rmsnorm in rmsnorms for param in rmsnorm.parameters()]
        rmsnorm_grads = [grad for grad in rmsnorm_grads if grad is not None]
        if rmsnorm_grads:
            coalesced = torch._utils._flatten_dense_tensors(rmsnorm_grads)
            torch.distributed.all_reduce(
                coalesced, group=parallel_state.get_tensor_model_parallel_group()
            )
            for buf, synced in zip(
                    rmsnorm_grads, torch._utils._unflatten_dense_tensors(coalesced, rmsnorm_grads)
            ):
                buf.copy_(synced)
        optimizer.step()

        if step >= total_steps:
            break

        if False:  # step > 0 and step % 100 == 0:
            # print(f"{step=}", flush=True)
            test_prompts = ["My name is"] * data_parallel_size
            inferred = inference(
                models, tok, test_prompts, llama_args, micro_batch_size, rank, forward_backward_func, 32,
                global_batch_size, data_parallel_size)
            print(f"{[*inferred]=}", flush=True)

        if False:  # (step % 1000) == 0 or (step == total_steps):
            torch.distributed.barrier()
            if rank < (tensor_model_parallel_size * pipeline_model_parallel_size):
                print("saving", flush=True)
                os.makedirs(save_to, exist_ok=True)
                torch.save(
                    models[0].state_dict(),
                    save_to / f"ckpt-tp-consolidated.{tp_rank:02d}.pth",
                )
                if rank == 0:
                    print("done", flush=True)
            torch.distributed.barrier()

        step += 1
        # LR Schedule
        warmup = min(1.0, step / 100.0)
        cos_anneal = 0.5 * (np.cos(np.pi * step / total_steps) + 1)
        lr = 3e-5 * warmup * (0.1 + 0.9 * cos_anneal)
        for group in optimizer.param_groups:
            group['lr'] = lr

    print("done", flush=True)
    torch.distributed.barrier()
    if dp_rank == 0:
        # save the state dict to sharded files
        os.makedirs(save_to, exist_ok=True)
        torch.save(
            models[0].state_dict(),
            save_to / f"tp-consolidated.{tp_rank:02d}.{pp_rank:02d}.pth",
        )

    if rank == 0:
        with open(save_to / "params.json", "w") as f:
            json.dump(llama_args.__dict__, f)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--llama", default="/mnt/hdd/llama2/65B/", type=Path)
    parser.add_argument("--tokenizer", default="/mnt/hdd/llama2/tokenizer.model", type=Path)
    parser.add_argument("--tp-world", default=8, type=int)
    parser.add_argument("--pp-world", default=1, type=int)
    parser.add_argument("--server", action="store_true", default=False)
    parser.add_argument("--save-to", default="/mnt/hdd/cc-llama2/65B", type=Path)
    parser.add_argument("--no-distributed-adam", action="store_true", default=False)
    args = parser.parse_args()
    if not args.server:
        main(llama=args.llama, tokenizer=args.tokenizer, tp_world=args.tp_world, pp_world=args.pp_world,
             save_to=args.save_to, distributed_adam=not args.no_distributed_adam)
    else:
        inference_server(llama=args.llama, tokenizer=args.tokenizer, tp_world=args.tp_world, pp_world=args.pp_world)
