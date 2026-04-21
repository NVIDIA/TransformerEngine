# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Dispatch layer for the cuDNN frontend Python SDPA (CuTe DSL), head_dim=256.

The kernels in ``cudnn.sdpa`` are Python-only (CuTe DSL), so invocation happens
entirely on the Python side. This module adapts TransformerEngine's
``fused_attn_fwd``/``fused_attn_bwd`` calling convention to the wrappers
``sdpa_fwd_wrapper_sm100_d256``/``sdpa_bwd_wrapper_sm100_d256`` and hides the
layout massaging required by the kernel.
"""

from __future__ import annotations

import functools
from typing import List, Optional, Tuple

import torch


@functools.lru_cache(maxsize=None)
def _sdpa_fwd_wrapper():
    from cudnn import sdpa_fwd_wrapper_sm100_d256  # pylint: disable=no-name-in-module

    return sdpa_fwd_wrapper_sm100_d256


@functools.lru_cache(maxsize=None)
def _sdpa_bwd_wrapper():
    from cudnn import sdpa_bwd_wrapper_sm100_d256  # pylint: disable=no-name-in-module

    return sdpa_bwd_wrapper_sm100_d256


@functools.lru_cache(maxsize=None)
def is_available() -> bool:
    """Whether the cuDNN-FE Python SDPA d=256 kernels can be imported."""
    try:
        _sdpa_fwd_wrapper()
        _sdpa_bwd_wrapper()
    except ImportError:
        return False
    return True


_SUPPORTED_MASKS = (
    "no_mask",
    "causal",
    "causal_bottom_right",
    "padding",
    "padding_causal",
    "padding_causal_bottom_right",
)


def is_supported(
    *,
    head_dim_qk: int,
    head_dim_v: int,
    qkv_dtype: torch.dtype,
    qkv_format: str,
    attn_mask_type: str,
    attn_bias_type: str,
    softmax_type: str,
    dropout: float,
    window_size: Tuple[int, int],
    max_seqlen_q: int,
    max_seqlen_kv: int,
    is_training: bool,
    deterministic: bool,
    device_compute_capability: Tuple[int, int],
    return_max_logit: bool = False,
) -> bool:
    """Whether the cuDNN-FE SDPA d=256 kernel can service this configuration."""
    if device_compute_capability[0] < 10:
        return False
    if head_dim_qk != 256 or head_dim_v != 256:
        return False
    if qkv_dtype not in (torch.float16, torch.bfloat16):
        return False
    if qkv_format not in ("bshd", "thd"):
        return False
    if attn_bias_type != "no_bias":
        return False
    if softmax_type != "vanilla":
        return False
    if dropout != 0.0:
        return False
    if return_max_logit:
        return False
    if attn_mask_type not in _SUPPORTED_MASKS:
        return False
    if qkv_format == "thd" and "padding" not in attn_mask_type:
        return False
    if qkv_format == "bshd" and "padding" in attn_mask_type:
        return False
    # The kernel's causal implementation aligns the end of Q with the end of K
    # (i.e., bottom-right). TE's plain "causal" means top-left, which only
    # matches when max_seqlen_q == max_seqlen_kv. For cross-attention the user
    # must opt in explicitly via a "_bottom_right" mask.
    if attn_mask_type in ("causal", "padding_causal") and max_seqlen_q != max_seqlen_kv:
        return False
    is_causal = "causal" in attn_mask_type
    left, right = window_size
    if not is_causal:
        if (left, right) != (-1, -1):
            return False
    else:
        if right not in (-1, 0):
            return False
    # Backward uses atomic adds on dQ → non-deterministic.
    if is_training and deterministic:
        return False
    if not is_available():
        return False
    return True


def _to_kernel_shape(x: torch.Tensor, qkv_format: str) -> torch.Tensor:
    """Arrange a TE tensor in the (B, H, S, D) view expected by the kernel.

    The kernel accepts either:
      * ``(B, H, S, D)`` where the underlying memory is BSHD-contiguous (i.e.
        a ``.transpose(1, 2)`` view of a BSHD tensor), or
      * ``(T, H, D)`` for variable-length THD.
    """
    if qkv_format == "bshd":
        return x.transpose(1, 2)
    return x


def _from_kernel_shape(x: torch.Tensor, qkv_format: str) -> torch.Tensor:
    """Inverse of :func:`_to_kernel_shape`."""
    if qkv_format == "bshd":
        return x.transpose(1, 2)
    return x


def _causal_and_window(
    attn_mask_type: str, window_size: Tuple[int, int]
) -> Tuple[bool, Tuple[int, int]]:
    is_causal = "causal" in attn_mask_type
    left, right = window_size
    if is_causal:
        return True, (left if left is not None else -1, 0)
    return False, (-1, -1)


def _cum_seqlens_for_kernel(
    cu_seqlens: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if cu_seqlens is None:
        return None
    if cu_seqlens.dtype == torch.int32:
        return cu_seqlens
    return cu_seqlens.to(dtype=torch.int32)


def fused_attn_fwd(
    *,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_kv: Optional[torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qkv_format: str,
    attn_mask_type: str,
    attn_scale: Optional[float],
    window_size: Tuple[int, int],
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Run the cuDNN-FE Python SDPA forward for head_dim=256 on SM100+.

    Returns
    -------
    out : torch.Tensor
        Attention output, same layout as ``q``.
    aux_ctx_tensors : list of torch.Tensor
        ``[softmax_lse, rng_state_placeholder]`` — kept two-element for
        compatibility with TE's aux-context convention. The rng placeholder is
        an empty tensor because this kernel does not support dropout.
    """
    sdpa_fwd = _sdpa_fwd_wrapper()
    is_causal, window = _causal_and_window(attn_mask_type, window_size)

    q_i = _to_kernel_shape(q, qkv_format)
    k_i = _to_kernel_shape(k, qkv_format)
    v_i = _to_kernel_shape(v, qkv_format)

    cum_q = _cum_seqlens_for_kernel(cu_seqlens_q) if qkv_format == "thd" else None
    cum_k = _cum_seqlens_for_kernel(cu_seqlens_kv) if qkv_format == "thd" else None

    current_stream = torch.cuda.current_stream().cuda_stream
    result = sdpa_fwd(
        q_tensor=q_i,
        k_tensor=k_i,
        v_tensor=v_i,
        cum_seqlen_q_tensor=cum_q,
        cum_seqlen_k_tensor=cum_k,
        max_s_q=max_seqlen_q,
        max_s_k=max_seqlen_kv,
        is_causal=is_causal,
        window_size=window,
        scale_softmax=attn_scale,
        current_stream=current_stream,
    )

    o_i = result["o_tensor"]
    lse = result["lse_tensor"]
    out = _from_kernel_shape(o_i, qkv_format)

    # Rng state placeholder (no dropout support on this path); kept so the
    # aux-context shape matches other F16 fused backends' (lse, rng) pair.
    rng_state = torch.empty(2, dtype=torch.int64, device=q.device)
    return out, [lse, rng_state]


def fused_attn_bwd(
    *,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_kv: Optional[torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    d_o: torch.Tensor,
    aux_ctx_tensors: List[torch.Tensor],
    qkv_format: str,
    attn_mask_type: str,
    attn_scale: Optional[float],
    window_size: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the cuDNN-FE Python SDPA backward for head_dim=256 on SM100+."""
    sdpa_bwd = _sdpa_bwd_wrapper()
    is_causal, window = _causal_and_window(attn_mask_type, window_size)

    q_i = _to_kernel_shape(q, qkv_format)
    k_i = _to_kernel_shape(k, qkv_format)
    v_i = _to_kernel_shape(v, qkv_format)
    o_i = _to_kernel_shape(o, qkv_format)
    do_i = _to_kernel_shape(d_o, qkv_format)

    cum_q = _cum_seqlens_for_kernel(cu_seqlens_q) if qkv_format == "thd" else None
    cum_k = _cum_seqlens_for_kernel(cu_seqlens_kv) if qkv_format == "thd" else None

    lse = aux_ctx_tensors[0]

    current_stream = torch.cuda.current_stream().cuda_stream
    result = sdpa_bwd(
        q_tensor=q_i,
        k_tensor=k_i,
        v_tensor=v_i,
        o_tensor=o_i,
        do_tensor=do_i,
        lse_tensor=lse,
        cum_seqlen_q_tensor=cum_q,
        cum_seqlen_k_tensor=cum_k,
        max_s_q=max_seqlen_q,
        max_s_k=max_seqlen_kv,
        is_causal=is_causal,
        window_size=window,
        scale_softmax=attn_scale,
        current_stream=current_stream,
    )

    dq_i = result["dq_tensor"]
    dk_i = result["dk_tensor"]
    dv_i = result["dv_tensor"]

    dq = _from_kernel_shape(dq_i, qkv_format)
    dk = _from_kernel_shape(dk_i, qkv_format)
    dv = _from_kernel_shape(dv_i, qkv_format)
    return dq, dk, dv
