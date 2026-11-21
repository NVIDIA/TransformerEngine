# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch wrappers for causal MXFP8 attention without future-token scale leakage."""

import torch

from transformer_engine.common.cudnn.mxfp8_utils import OWN_BLOCK
from transformer_engine.common.triton.mxfp8_causal_attention import (
    decode_kernel,
    own_block_attention_kernel,
    token_values_kernel,
)

__all__ = ["OWN_BLOCK", "token_values", "own_block_attention", "decode_operands"]


def _bsh_strides(x: torch.Tensor, fmt: str):
    """(batch, sequence, head) element strides of a bshd/sbhd tensor with unit feature stride."""
    assert x.stride(-1) == 1, "the feature dimension must be contiguous"
    if fmt == "bshd":
        return x.stride(0), x.stride(1), x.stride(2)
    if fmt == "sbhd":
        return x.stride(1), x.stride(0), x.stride(2)
    if fmt == "bhsd":
        return x.stride(0), x.stride(2), x.stride(1)
    raise ValueError(f"Unsupported format {fmt!r}; expected bshd, sbhd or bhsd.")


def _bshd_dims(x: torch.Tensor, fmt: str):
    """(batch, sequence, heads, features) of a bshd/sbhd/bhsd tensor."""
    order = {"bshd": (0, 1, 2, 3), "sbhd": (1, 0, 2, 3), "bhsd": (0, 2, 1, 3)}[fmt]
    return tuple(x.shape[i] for i in order)


def _payload(x) -> torch.Tensor:
    """Rowwise E4M3 payload of an MXFP8 tensor (stored as uint8) typed for Triton."""
    return x._rowwise_data.view(torch.float8_e4m3fn)


def _pack_heads(num_heads: int, num_kv_heads: int) -> int:
    """Query heads sharing a KV head that fill the boundary kernel's 128 MMA rows."""
    return next(p for p in (4, 2, 1) if (num_heads // num_kv_heads) % p == 0)


def token_values(v: torch.Tensor, fmt: str):
    """Token-normalized V for the own-block product and the backward.

    Returns ``u`` (E4M3, ``[B, H_kv, S, 128]``) and ``e`` (int32, ``[B, H_kv, S]``) with
    ``V = u * 2**(e - 8)`` up to the E4M3 rounding of ``u``.
    """
    b, s, h, d = _bshd_dims(v, fmt)
    assert d == 128 and s % OWN_BLOCK == 0
    u = torch.empty((b, h, s, d), dtype=torch.float8_e4m3fn, device=v.device)
    e = torch.empty((b, h, s), dtype=torch.int32, device=v.device)
    token_values_kernel[(s // OWN_BLOCK, h, b)](v, u, e, *_bsh_strides(v, fmt), s, h, num_warps=4)
    return u, e


def own_block_attention(
    q_fp8, k_fp8, u, e, out, stats, attn_scale: float, qkv_format: str, o_format: str
):
    """Merge each query's own-block attention into the completed-block result, in place.

    ``q_fp8`` and ``k_fp8`` are rowwise MXFP8 tensors whose payloads are in ``qkv_format`` and
    whose scales are in the per-(batch, head) F8_128x4 layout; ``out`` (``o_format``) and
    ``stats`` (``[B, H, S, 1]``, natural log) hold cuDNN's ``causal_exclude_own_block=32``
    result on entry and the causal result on exit.
    """
    fmt = qkv_format
    q8, k8 = _payload(q_fp8), _payload(k_fp8)
    b, s, h, d = _bshd_dims(q8, fmt)
    h_kv = _bshd_dims(k8, fmt)[2]
    assert d == 128 and s % 128 == 0 and h % h_kv == 0
    assert stats.is_contiguous() and stats.numel() == b * h * s
    pack = _pack_heads(h, h_kv)
    own_block_attention_kernel[(s // (128 // pack), h // pack, b)](
        q8,
        k8,
        q_fp8._rowwise_scale_inv,
        k_fp8._rowwise_scale_inv,
        u,
        e,
        out,
        stats,
        *_bsh_strides(q8, fmt),
        *_bsh_strides(k8, fmt),
        *_bsh_strides(out, o_format),
        attn_scale,
        s,
        h,
        h_kv,
        pack,
        num_warps=4,
        num_stages=1,
    )


def decode_operands(q_fp8, k_fp8, u, e, dtype: torch.dtype, fmt: str):
    """BF16/FP16 Q, K and V for the backward, shaped like the MXFP8 payloads (``fmt``)."""
    q8, k8 = _payload(q_fp8), _payload(k_fp8)
    b, s, h, _ = _bshd_dims(q8, fmt)
    h_kv = _bshd_dims(k8, fmt)[2]
    q = torch.empty_strided(q8.shape, q8.stride(), dtype=dtype, device=q8.device)
    k = torch.empty_strided(k8.shape, k8.stride(), dtype=dtype, device=k8.device)
    v = torch.empty_like(k)
    block = 64
    decode_kernel[(s // block, b * (h + h_kv))](
        q8,
        q_fp8._rowwise_scale_inv,
        k8,
        k_fp8._rowwise_scale_inv,
        u,
        e,
        q,
        k,
        v,
        *_bsh_strides(q, fmt),
        *_bsh_strides(k, fmt),
        b,
        s,
        h,
        h_kv,
        block,
        num_warps=4,
    )
    return q, k, v
