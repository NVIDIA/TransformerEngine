# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import math
from typing import Callable, Optional
import torch
import transformer_engine.pytorch as te
from transformer_engine.pytorch.fp8 import DelayedScaling, dist_group_type


def speedometer(
    module: torch.nn.Module,
    input: torch.Tensor,
    output_grad: torch.Tensor,
    forward_kwargs: dict = {},
    fp8_autocast_kwargs: Optional[dict] = None,
    timing_iters: int = 50,
    warmup_iters: int = 50,
) -> None:
    """Measure average run time for a PyTorch module

    Performs forward and backward passes.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    if fp8_autocast_kwargs is None:
        fp8_autocast_kwargs = {"enabled": False}

    # Warmup runs
    torch.cuda.synchronize()
    for _ in range(warmup_iters):
        with te.fp8_autocast(**fp8_autocast_kwargs):
            output = module(input, **forward_kwargs)
        output.backward(output_grad)

    # Timing runs
    start.record()
    for _ in range(timing_iters):
        with te.fp8_autocast(**fp8_autocast_kwargs):
            output = module(input, **forward_kwargs)
        output.backward(output_grad)
    end.record()
    torch.cuda.synchronize()

    print(f"Mean time: {start.elapsed_time(end)/timing_iters} ms")


class DotProductAttention(torch.nn.Module):
    """Attention operation in Transformer layer

    Built with plain PyTorch modules.

    """

    def __init__(
        self,
        num_attention_heads: int,
        kv_channels: int,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        self.projection_size = kv_channels * num_attention_heads
        self.hidden_size_per_attention_head = kv_channels
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        self.dropout = torch.nn.Dropout(attention_dropout)

    def masked_softmax(self, inp: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is not None:
            inp.masked_fill_(mask, -10000.0)
        return torch.nn.Softmax(dim=-1)(inp)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b = query.size(1)
        np = query.size(2)
        sq = query.size(0)
        sk = key.size(0)
        hn = value.size(3)

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query = query.view(sq, b * np, -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key = key.view(sk, b * np, -1)

        bmm1 = (
            torch.bmm(query.transpose(0, 1), key.transpose(0, 1).transpose(1, 2)) / self.norm_factor
        )

        # change view to [b, np, sq, sk]
        attention_scores = bmm1.view(b, np, sq, sk)

        attention_probs = self.masked_softmax(attention_scores, attention_mask)

        attention_probs = self.dropout(attention_probs)

        # change view [sk, b * np, hn]
        value = value.view(sk, b * np, -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(b * np, sq, -1)

        # matmul: [b * np, sq, hn]
        context = torch.bmm(attention_probs, value.transpose(0, 1))

        # change view [b, np, sq, hn]
        context = context.view(b, np, sq, hn)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context = context.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        context = context.view(sq, b, self.projection_size)

        return context


class BasicMLP(torch.nn.Module):
    """Feed-forward network in Transformer layer

    Built with plain PyTorch modules.

    """

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
    ) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(hidden_size, ffn_hidden_size, bias=True)
        self.linear2 = torch.nn.Linear(ffn_hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = torch.nn.functional.gelu(x, approximate="tanh")
        x = self.linear2(x)
        return x


def share_parameters_with_basic_te_model(te_model, basic_model):
    """Initialize parameters for TE Transformer layer with basic modules

    Parameter values are copied from pure PyTorch implementation.

    """
    te_model.ln1.weight = basic_model.ln1.weight
    te_model.ln1.bias = basic_model.ln1.bias
    te_model.qkv_projection.weight = basic_model.qkv_projection.weight
    te_model.qkv_projection.bias = basic_model.qkv_projection.bias
    te_model.projection.weight = basic_model.projection.weight
    te_model.projection.bias = basic_model.projection.bias
    te_model.ln2.weight = basic_model.ln2.weight
    te_model.ln2.bias = basic_model.ln2.bias
    te_model.mlp.linear1.weight = basic_model.mlp.linear1.weight
    te_model.mlp.linear1.bias = basic_model.mlp.linear1.bias
    te_model.mlp.linear2.weight = basic_model.mlp.linear2.weight
    te_model.mlp.linear2.bias = basic_model.mlp.linear2.bias


def share_parameters_with_fused_te_model(te_model, basic_model):
    """Initialize parameters for TE Transformer layer with fused modules

    Parameter values are copied from pure PyTorch implementation.

    """
    te_model.ln_qkv.layer_norm_weight = basic_model.ln1.weight
    te_model.ln_qkv.layer_norm_bias = basic_model.ln1.bias
    te_model.ln_qkv.weight = basic_model.qkv_projection.weight
    te_model.ln_qkv.bias = basic_model.qkv_projection.bias
    te_model.projection.weight = basic_model.projection.weight
    te_model.projection.bias = basic_model.projection.bias
    te_model.ln_mlp.layer_norm_weight = basic_model.ln2.weight
    te_model.ln_mlp.layer_norm_bias = basic_model.ln2.bias
    te_model.ln_mlp.fc1_weight = basic_model.mlp.linear1.weight
    te_model.ln_mlp.fc1_bias = basic_model.mlp.linear1.bias
    te_model.ln_mlp.fc2_weight = basic_model.mlp.linear2.weight
    te_model.ln_mlp.fc2_bias = basic_model.mlp.linear2.bias


def share_parameters_with_transformerlayer_te_model(te_model, basic_model):
    """Initialize parameters for monolithic TE Transformer layer

    Parameter values are copied from pure PyTorch implementation.

    """
    te_model.self_attention.layernorm_qkv.layer_norm_weight = basic_model.ln1.weight
    te_model.self_attention.layernorm_qkv.layer_norm_bias = basic_model.ln1.bias
    te_model.self_attention.layernorm_qkv.weight = basic_model.qkv_projection.weight
    te_model.self_attention.layernorm_qkv.bias = basic_model.qkv_projection.bias
    te_model.self_attention.proj.weight = basic_model.projection.weight
    te_model.self_attention.proj.bias = basic_model.projection.bias
    te_model.layernorm_mlp.layer_norm_weight = basic_model.ln2.weight
    te_model.layernorm_mlp.layer_norm_bias = basic_model.ln2.bias
    te_model.layernorm_mlp.fc1_weight = basic_model.mlp.linear1.weight
    te_model.layernorm_mlp.fc1_bias = basic_model.mlp.linear1.bias
    te_model.layernorm_mlp.fc2_weight = basic_model.mlp.linear2.weight
    te_model.layernorm_mlp.fc2_bias = basic_model.mlp.linear2.bias


def cast_to_representable(inp, scale=1.0, fp8_format="e4m3"):
    import transformer_engine.pytorch.cpp_extensions as texcpp
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.constants import TE_DType

    fp8_type = tex.DType.kFloat8E4M3 if fp8_format == "e4m3" else tex.DType.kFloat8E5M2
    input_type = TE_DType[inp.dtype]
    meta = tex.FP8TensorMeta()
    meta.scale = torch.ones(1, dtype=torch.float32, device="cuda") * scale
    meta.scale_inv = torch.ones(1, dtype=torch.float32, device="cuda") / scale
    meta.amax_history = torch.zeros(1, 1, dtype=torch.float32, device="cuda")
    ret = texcpp.cast_to_fp8(inp, meta, tex.FP8FwdTensors.GEMM1_INPUT, fp8_type)
    ret = texcpp.cast_from_fp8(ret, meta, tex.FP8FwdTensors.GEMM1_INPUT, fp8_type, input_type)
    return ret
