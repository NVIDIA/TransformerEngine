# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Utility functions for Getting Started with Transformer Engine - PyTorch
========================================================================

Helper classes and functions for the getting started examples.
"""

import math
from typing import Optional
import torch
import transformer_engine.pytorch as te


def speedometer(
    module: torch.nn.Module,
    x: torch.Tensor,
    forward_kwargs: dict = {},
    autocast_kwargs: Optional[dict] = None,
    timing_iters: int = 100,
    warmup_iters: int = 10,
    label: str = "benchmark",
) -> float:
    """Measure average forward + backward pass time for a PyTorch module.

    Args:
        module: PyTorch module to benchmark
        x: Input tensor
        forward_kwargs: Additional kwargs for forward pass
        autocast_kwargs: Kwargs for te.autocast context
        timing_iters: Number of timing iterations
        warmup_iters: Number of warmup iterations
        label: Optional label for logging

    Returns:
        Average time per iteration in milliseconds
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    if autocast_kwargs is None:
        autocast_kwargs = {"enabled": False}

    # Warmup runs
    torch.cuda.synchronize()
    for _ in range(warmup_iters):
        with te.autocast(**autocast_kwargs):
            y = module(x, **forward_kwargs)
            loss = y.sum()
        loss.backward()
    torch.cuda.synchronize()

    # Timing runs
    start.record()
    for _ in range(timing_iters):
        with te.autocast(**autocast_kwargs):
            y = module(x, **forward_kwargs)
            loss = y.sum()
        loss.backward()
    end.record()
    torch.cuda.synchronize()

    avg_time = start.elapsed_time(end) / timing_iters
    print(f"Mean time: {avg_time:.3f} ms")
    return avg_time


class DotProductAttention(torch.nn.Module):
    """Attention operation in Transformer layer.

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

        query = query.view(sq, b * np, -1)
        key = key.view(sk, b * np, -1)

        bmm1 = (
            torch.bmm(query.transpose(0, 1), key.transpose(0, 1).transpose(1, 2)) / self.norm_factor
        )

        attention_scores = bmm1.view(b, np, sq, sk)
        attention_probs = self.masked_softmax(attention_scores, attention_mask)
        attention_probs = self.dropout(attention_probs)

        value = value.view(sk, b * np, -1)
        attention_probs = attention_probs.view(b * np, sq, -1)
        context = torch.bmm(attention_probs, value.transpose(0, 1))
        context = context.view(b, np, sq, hn)
        context = context.permute(2, 0, 1, 3).contiguous()
        context = context.view(sq, b, self.projection_size)

        return context
