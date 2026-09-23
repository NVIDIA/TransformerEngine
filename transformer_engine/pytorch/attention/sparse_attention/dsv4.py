# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""DSv4 core and preparation helpers, composed with model-owned projections."""

from typing import Optional

import torch

from ._dsv4_backend import attention as _attention
from ._dsv4_backend import compress, select_blocks

__all__ = ["DSv4Attention", "compress", "select_blocks"]


class DSv4Attention(torch.nn.Module):
    """Experimental causal joint local + compressed attention for DSv4.

    Parameter-free core: the caller owns projections, compression, normalization,
    RoPE, and the learned sink. Use :func:`compress` for gated pooling and
    :func:`select_blocks` for the CSA indexer. Pass selected indices for CSA;
    omit them and supply ``max_compressed_seqlen`` for HCA.

    Inputs are contiguous CUDA BF16 query [T,64,D], local KV [T,D], compressed
    KV [Tc,D], FP32 sink [64], and CUDA INT32 sequence prefixes [B+1]. D is 512
    or 576; values use the first 512 KV channels. Output is [T,64,512], retaining
    the head axis for DSv4's output unrotation. Prefixes start at zero and end
    at valid row counts; compressed lengths are floor(sequence length / ratio).
    CSA indices are distinct global packed compressed-row IDs, or -1 for padding.

    Only full sequences with positions starting at zero are supported. Local
    keys satisfy max(0,q-window_size+1) <= k <= q. Compressed block j becomes
    visible when (j+1)*ratio <= q+1. All visible local and compressed entries
    share one softmax with the sink; its value contribution is zero.

    Requires SM100 and cuDNN Frontend 1.29.0. Forward and first-order backward
    use cuDNN's DSA namespace. No dropout, arbitrary masks, cache/decode, FP8,
    distributed attention, or higher-order gradients. This module does not use
    DPA backend-selection flags. cuDNN operations are loaded on execution.
    """

    def __init__(self, *, window_size: int, ratio: int):
        super().__init__()
        if window_size <= 0 or ratio <= 0:
            raise ValueError("window_size and ratio must be positive.")
        self.window_size = window_size
        self.ratio = ratio

    def forward(
        self,
        query: torch.Tensor,
        local_kv: torch.Tensor,
        compressed_kv: torch.Tensor,
        sink: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_seqlens_comp: torch.Tensor,
        *,
        indices: Optional[torch.Tensor] = None,
        scale: Optional[float] = None,
        max_compressed_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        """Evaluate the joint attention, with scale defaulting to D**-0.5."""
        from transformer_engine.pytorch.quantization import FP8GlobalStateManager

        if FP8GlobalStateManager.is_fp8_enabled():
            raise NotImplementedError(
                "DSv4Attention supports BF16 only; disable TE FP8 autocast."
            )
        return _attention(
            query,
            local_kv,
            compressed_kv,
            sink,
            cu_seqlens,
            cu_seqlens_comp,
            window_size=self.window_size,
            ratio=self.ratio,
            indices=indices,
            scale=scale,
            max_compressed_seqlen=max_compressed_seqlen,
        )
