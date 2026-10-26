# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Deferred fused-QKV RoPE for FP8 current-scaling attention.

``apply_fused_qkv_rotary_pos_emb(..., defer_for_fp8_attention=True)`` returns *placeholder* Q/K/V
tensors tagged with the QKV GEMM output and the RoPE angles instead of running RoPE. If the
DotProductAttention that receives them runs FP8 current-scaling fused attention, RoPE and the
FP8 quantization of Q/K/V happen in one fused kernel straight from the GEMM output and the
placeholders are never read. Otherwise DotProductAttention fills them with TE's regular RoPE
before doing anything else, so results do not depend on which path is taken.
"""

import os
from typing import List, Optional, Tuple

import torch

import transformer_engine_torch as tex

_ENABLED = os.environ.get("NVTE_FUSED_QKV_ROPE_FP8", "1") == "1"
_HEAD_DIM = 128


class _DeferredRoPEState:
    """What the deferred RoPE needs, carried from the RoPE call to the quantize call."""

    __slots__ = ("mixed", "freqs", "split", "cos_sin", "consumed", "materialized")

    def __init__(self, mixed: torch.Tensor, freqs: torch.Tensor, split: List[int]):
        self.mixed = mixed  # released once quantized or materialized
        self.freqs = freqs
        self.split = list(split)
        self.cos_sin = None
        self.consumed = False
        self.materialized = False


class _DeferredQKVRoPEFunc(torch.autograd.Function):
    """Placeholders whose backward is TE's fused QKV RoPE backward."""

    @staticmethod
    def forward(ctx, mixed, state):
        s, b, g, _ = mixed.shape
        d = state.split[1]
        hq = state.split[0] // d
        kwargs = dict(dtype=mixed.dtype, device=mixed.device)
        q = torch.empty(s, b, g * hq, d, **kwargs)
        k = torch.empty(s, b, g, d, **kwargs)
        v = torch.empty(s, b, g, d, **kwargs)
        ctx.freqs = state.freqs
        ctx.split = state.split
        return q, k, v

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v):
        # pylint: disable=import-outside-toplevel
        from .rope import QKVFormat

        grad_input = tex.fused_qkv_rope_backward(
            grad_q.contiguous(),
            grad_k.contiguous(),
            grad_v.contiguous(),
            ctx.freqs,
            ctx.freqs,
            ctx.split,
            QKVFormat["sbhd"],
            False,
            1,
            0,
        )
        return grad_input, None


def get_state(q: torch.Tensor) -> Optional[_DeferredRoPEState]:
    """Deferred-RoPE state of a placeholder, or None for an ordinary tensor."""
    return getattr(q, "_deferred_qkv_rope", None)


def defer_fused_qkv_rope(
    qkv: torch.Tensor,
    q_freqs: torch.Tensor,
    k_freqs: torch.Tensor,
    qkv_split_arg_list: List[int],
    start_positions,
    tensor_format: str,
    interleaved: bool,
    cp_size: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return tagged placeholders for the deferred RoPE, or None if the inputs are unsupported."""
    split = list(qkv_split_arg_list)
    if not (
        _ENABLED
        and start_positions is None
        and tensor_format == "sbhd"
        and not interleaved
        and cp_size == 1
        and qkv.dim() == 4
        and qkv.is_cuda
        and qkv.dtype == torch.bfloat16
        and qkv.is_contiguous()
        and len(split) == 3
        and split[1] == split[2] == _HEAD_DIM
        and split[0] % split[1] == 0
        and qkv.shape[3] == sum(split)
        and q_freqs.data_ptr() == k_freqs.data_ptr()
        and q_freqs.shape == k_freqs.shape
        and q_freqs.stride() == k_freqs.stride()
        and q_freqs.dim() == 4
        and q_freqs.shape[1] == 1
        and q_freqs.shape[2] == 1
        and q_freqs.shape[3] == split[1]
        and q_freqs.shape[0] >= qkv.shape[0]
        and q_freqs.is_cuda
    ):
        return None
    freqs = q_freqs if q_freqs.dtype == torch.float32 else q_freqs.float()
    if not freqs.is_contiguous():
        freqs = freqs.contiguous()
    state = _DeferredRoPEState(qkv, freqs, split)
    q, k, v = _DeferredQKVRoPEFunc.apply(qkv, state)
    for t in (q, k, v):
        t._deferred_qkv_rope = state
    return q, k, v


def materialize(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    """Run TE's regular RoPE into the placeholders (no-op for ordinary tensors)."""
    state = get_state(q)
    if state is None or state.materialized:
        return
    if state.mixed is None:
        raise RuntimeError(
            "Deferred fused-QKV RoPE placeholders were already consumed by the fused FP8 "
            "quantization and cannot be reused (e.g. attention recompute). Disable "
            "defer_for_fp8_attention / NVTE_FUSED_QKV_ROPE_FP8 for this configuration."
        )
    with torch.no_grad():
        q_rot, k_rot, v_rot = tex.fused_qkv_rope_forward(
            state.mixed,
            state.freqs,
            state.freqs,
            None,
            state.split,
            _sbhd(),
            False,
            1,
            0,
        )
        q.copy_(q_rot)
        k.copy_(k_rot)
        v.copy_(v_rot)
    state.consumed = state.materialized = True
    state.mixed = None


def _sbhd():
    # pylint: disable=import-outside-toplevel
    from .rope import QKVFormat

    return QKVFormat["sbhd"]


def can_fuse_with_fp8_attention(
    q: torch.Tensor, qkv_quantizer, qkv_layout: str, is_bwd_fp8: bool
) -> bool:
    """Whether FP8 fused attention with this QKV quantizer can consume the deferred RoPE."""
    # pylint: disable=import-outside-toplevel
    from ..tensor.float8_tensor import Float8CurrentScalingQuantizer
    from ..constants import DType

    state = get_state(q)
    return (
        state is not None
        and not state.consumed
        and is_bwd_fp8
        and type(qkv_quantizer) is Float8CurrentScalingQuantizer
        and not qkv_quantizer.with_amax_reduction
        and qkv_quantizer.dtype == DType.kFloat8E4M3
        and qkv_layout == "sbhd_sbhd_sbhd"
    )


def quantize_deferred(q, k, v, qkv_quantizer):
    """Fused RoPE + FP8 quantization of deferred placeholders.

    Returns ``(q_fp8, k_fp8, v_fp8)`` Float8Tensor views over one packed payload, or None if q is
    not a placeholder (or the quantizer's output layout is unsupported, in which case the
    placeholders are materialized with TE's RoPE and the regular quantization applies).
    """
    # pylint: disable=import-outside-toplevel
    from ..tensor.float8_tensor import Float8Tensor
    from ..triton.rope_fp8 import rope_trig_tables, rope_qkv_quantize

    state = get_state(q)
    if state is None:
        return None
    if state.consumed:
        raise RuntimeError("Deferred fused-QKV RoPE placeholders were already consumed.")
    d = state.split[1]
    hq = state.split[0] // d
    s, b, g, _ = state.mixed.shape
    nq, nk = s * b * g * hq * d, s * b * g * d
    qkv_fp8 = qkv_quantizer.make_empty((nq + 2 * nk,), dtype=q.dtype, device=q.device)
    if qkv_fp8._data is None or qkv_fp8._transpose is not None or qkv_fp8._scale_inv.numel() != 1:
        materialize(q, k, v)
        return None
    data = qkv_fp8._data
    cos, sin = rope_trig_tables(state.freqs)
    rope_qkv_quantize(
        state.mixed,
        cos,
        sin,
        data.view(torch.float8_e4m3fn),
        qkv_fp8._scale_inv,
        hq,
        eps=float(qkv_quantizer.amax_epsilon),
        force_pow_2_scales=bool(qkv_quantizer.force_pow_2_scales),
    )
    views = (
        data[:nq].view(q.shape),
        data[nq : nq + nk].view(k.shape),
        data[nq + nk :].view(v.shape),
    )
    state.consumed = True
    state.mixed = None
    return tuple(Float8Tensor.make_like(qkv_fp8, data=x, dtype=q.dtype) for x in views)
