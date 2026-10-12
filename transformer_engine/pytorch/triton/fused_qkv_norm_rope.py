# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch wrapper functions for the fused QKV RMSNorm + RoPE Triton kernels."""

from typing import List, Optional, Tuple

import torch
import triton

from transformer_engine.common.triton.fused_qkv_norm_rope import (
    _qkv_norm_rope_table_kernel,
    _qkv_norm_rope_fwd_kernel,
    _qkv_norm_rope_bwd_kernel,
)

_NUM_WARPS = 4
_BWD_CTAS_PER_SM = 8
_sm_count = {}


def _next_pow2(n: int) -> int:
    return 1 << max(0, n - 1).bit_length()


def _rows_per_program(d: int) -> int:
    """Rows per program so that a program streams about 2 KiB of input per dtype byte."""
    return max(8, 1 << ((2048 // d).bit_length() - 1))


def _sm_count_of(device: torch.device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _sm_count:
        _sm_count[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _sm_count[idx]


def _geometry(
    qkv: torch.Tensor,
    q_freqs: torch.Tensor,
    k_freqs: torch.Tensor,
    qkv_split_arg_list: List[int],
    q_norm_weight: Optional[torch.Tensor],
    k_norm_weight: Optional[torch.Tensor],
    tensor_format: str,
    start_positions: Optional[torch.Tensor],
    cp_size: int,
) -> dict:
    assert qkv.dim() == 4 and qkv.is_contiguous(), "qkv must be a contiguous 4D tensor"
    assert qkv.dtype in (
        torch.float32,
        torch.bfloat16,
        torch.float16,
    ), f"Unsupported qkv dtype {qkv.dtype}"
    assert tensor_format in ("sbhd", "bshd"), f"Unsupported tensor_format: {tensor_format}."
    assert len(qkv_split_arg_list) == 3, "qkv_split_arg_list must have 3 entries"
    q_split, k_split, v_split = qkv_split_arg_list
    d = k_split
    assert v_split == d, "the fused norm path needs value heads of the same width as key heads"
    assert q_split % d == 0 and q_split > 0, "q split must be a multiple of the head size"
    nq = q_split // d
    assert qkv.shape[3] == q_split + k_split + v_split, "qkv last dim must equal the sum of splits"
    for f in (q_freqs, k_freqs):
        assert f.dim() == 4 and f.shape[1] == 1 and f.shape[2] == 1, "freqs must be [s2, 1, 1, d2]"
        assert f.dtype == torch.float32 and f.is_contiguous(), "freqs must be contiguous fp32"
    d2 = q_freqs.shape[3]
    assert k_freqs.shape[3] == d2, "q_freqs and k_freqs must have the same rotary dimension"
    assert (
        0 < d2 <= d and d2 % 2 == 0
    ), "the rotary dimension must be even and at most the head size"
    for w in (q_norm_weight, k_norm_weight):
        if w is not None:
            assert w.numel() == d and w.is_contiguous(), "norm weights must be contiguous [d]"
    sbhd = tensor_format == "sbhd"
    s = qkv.shape[0] if sbhd else qkv.shape[1]
    b = qkv.shape[1] if sbhd else qkv.shape[0]
    h = qkv.shape[2]
    if cp_size > 1:
        assert s % 2 == 0, "context parallelism needs an even local sequence length"
        assert start_positions is None, "start_positions is not supported with cp_size > 1"
    if start_positions is not None:
        assert start_positions.numel() == b and start_positions.dtype in (torch.int32, torch.int64)
    return dict(
        s=s,
        b=b,
        h=h,
        nq=nq,
        d=d,
        d2=d2,
        sbhd=sbhd,
        n_rows=s * b * h * (nq + 2),
        rows=_rows_per_program(d),
        block_h2=_next_pow2(d2 // 2),
        block_p=_next_pow2(d - d2) if d > d2 else 1,
    )


def _rotation_tables(q_freqs: torch.Tensor, k_freqs: torch.Tensor, d2: int) -> torch.Tensor:
    s_q, s_k = q_freqs.shape[0], k_freqs.shape[0]
    s_table = max(s_q, s_k)
    table = torch.empty((4, s_table, d2), dtype=torch.float32, device=q_freqs.device)
    block_s = 64
    _qkv_norm_rope_table_kernel[(triton.cdiv(s_table, block_s),)](
        q_freqs,
        k_freqs,
        table,
        s_q,
        s_k,
        s_table,
        D2=d2,
        BLOCK_S=block_s,
        BLOCK_D2=_next_pow2(d2),
        num_warps=_NUM_WARPS,
    )
    return table


def _common_kwargs(
    g: dict,
    start_positions,
    q_norm_weight,
    k_norm_weight,
    interleaved,
    cp_size,
    zero_centered_gamma,
):
    return dict(
        H=g["h"],
        NQ=g["nq"],
        D=g["d"],
        D2=g["d2"],
        BLOCK_H2=g["block_h2"],
        BLOCK_P=g["block_p"],
        INTERLEAVED=bool(interleaved),
        SBHD=g["sbhd"],
        CP=cp_size > 1,
        HAS_START=start_positions is not None,
        NORM_Q=q_norm_weight is not None,
        NORM_K=k_norm_weight is not None,
        ZERO_CENTERED=bool(zero_centered_gamma),
        ROWS=g["rows"],
        num_warps=_NUM_WARPS,
    )


def _weight_or_dummy(w: Optional[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
    return w if w is not None else like


def fused_qkv_norm_rope_forward(
    qkv: torch.Tensor,
    q_freqs: torch.Tensor,
    k_freqs: torch.Tensor,
    qkv_split_arg_list: List[int],
    q_norm_weight: Optional[torch.Tensor],
    k_norm_weight: Optional[torch.Tensor],
    eps: float,
    zero_centered_gamma: bool,
    tensor_format: str = "sbhd",
    start_positions: Optional[torch.Tensor] = None,
    interleaved: bool = False,
    cp_size: int = 1,
    cp_rank: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-head RMSNorm of the Q and K heads of a combined QKV tensor followed by RoPE.

    Returns ``(q, k, v)`` with ``q`` of shape ``[..., h * nq, d]`` and ``k``, ``v`` of shape
    ``[..., h, d]`` in the token order of ``qkv``. A norm whose weight is ``None`` is skipped.
    """
    g = _geometry(
        qkv,
        q_freqs,
        k_freqs,
        qkv_split_arg_list,
        q_norm_weight,
        k_norm_weight,
        tensor_format,
        start_positions,
        cp_size,
    )
    lead = tuple(qkv.shape[:2])
    q_out = torch.empty(lead + (g["h"] * g["nq"], g["d"]), dtype=qkv.dtype, device=qkv.device)
    k_out = torch.empty(lead + (g["h"], g["d"]), dtype=qkv.dtype, device=qkv.device)
    v_out = torch.empty(lead + (g["h"], g["d"]), dtype=qkv.dtype, device=qkv.device)
    table = _rotation_tables(q_freqs, k_freqs, g["d2"])
    _qkv_norm_rope_fwd_kernel[(triton.cdiv(g["n_rows"], g["rows"]),)](
        qkv,
        _weight_or_dummy(q_norm_weight, qkv),
        _weight_or_dummy(k_norm_weight, qkv),
        start_positions if start_positions is not None else qkv,
        table,
        q_out,
        k_out,
        v_out,
        g["n_rows"],
        g["s"],
        g["b"],
        cp_size,
        cp_rank,
        table.shape[1],
        float(eps),
        **_common_kwargs(
            g,
            start_positions,
            q_norm_weight,
            k_norm_weight,
            interleaved,
            cp_size,
            zero_centered_gamma,
        ),
    )
    return q_out, k_out, v_out


def _grad_strides(grad: torch.Tensor, sbhd: bool) -> Tuple[torch.Tensor, int, int, int]:
    if grad.stride(3) != 1:
        grad = grad.contiguous()
    if sbhd:
        return grad, grad.stride(0), grad.stride(1), grad.stride(2)
    return grad, grad.stride(1), grad.stride(0), grad.stride(2)


def fused_qkv_norm_rope_backward(
    grad_q: torch.Tensor,
    grad_k: torch.Tensor,
    grad_v: torch.Tensor,
    qkv: torch.Tensor,
    q_freqs: torch.Tensor,
    k_freqs: torch.Tensor,
    qkv_split_arg_list: List[int],
    q_norm_weight: Optional[torch.Tensor],
    k_norm_weight: Optional[torch.Tensor],
    eps: float,
    zero_centered_gamma: bool,
    tensor_format: str = "sbhd",
    start_positions: Optional[torch.Tensor] = None,
    interleaved: bool = False,
    cp_size: int = 1,
    cp_rank: int = 0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Backward of :func:`fused_qkv_norm_rope_forward`.

    Returns the gradient w.r.t. ``qkv`` (same layout) and the fp32-accumulated gradients
    w.r.t. the two norm weights (``None`` for a norm that was skipped). The weight gradients
    are reduced in a fixed order and are bitwise reproducible.
    """
    g = _geometry(
        qkv,
        q_freqs,
        k_freqs,
        qkv_split_arg_list,
        q_norm_weight,
        k_norm_weight,
        tensor_format,
        start_positions,
        cp_size,
    )
    lead = tuple(qkv.shape[:2])
    assert grad_q.shape == lead + (g["h"] * g["nq"], g["d"]), "grad_q has an unexpected shape"
    assert grad_k.shape == lead + (g["h"], g["d"]), "grad_k has an unexpected shape"
    assert grad_v.shape == lead + (g["h"], g["d"]), "grad_v has an unexpected shape"
    grad_q, dq_ss, dq_sb, dq_sh = _grad_strides(grad_q, g["sbhd"])
    grad_k, dk_ss, dk_sb, dk_sh = _grad_strides(grad_k, g["sbhd"])
    grad_v, dv_ss, dv_sb, dv_sh = _grad_strides(grad_v, g["sbhd"])
    grad_qkv = torch.empty_like(qkv)
    table = _rotation_tables(q_freqs, k_freqs, g["d2"])
    n_blocks = triton.cdiv(g["n_rows"], g["rows"])
    nprog = min(n_blocks, _sm_count_of(qkv.device) * _BWD_CTAS_PER_SM)
    partial = torch.empty((nprog, 2 * g["d"]), dtype=torch.float32, device=qkv.device)
    _qkv_norm_rope_bwd_kernel[(nprog,)](
        grad_q,
        grad_k,
        grad_v,
        qkv,
        _weight_or_dummy(q_norm_weight, qkv),
        _weight_or_dummy(k_norm_weight, qkv),
        start_positions if start_positions is not None else qkv,
        table,
        grad_qkv,
        partial,
        g["n_rows"],
        n_blocks,
        g["s"],
        g["b"],
        cp_size,
        cp_rank,
        table.shape[1],
        float(eps),
        dq_ss,
        dq_sb,
        dq_sh,
        dk_ss,
        dk_sb,
        dk_sh,
        dv_ss,
        dv_sb,
        dv_sh,
        **_common_kwargs(
            g,
            start_positions,
            q_norm_weight,
            k_norm_weight,
            interleaved,
            cp_size,
            zero_centered_gamma,
        ),
    )
    weight_grads = partial.sum(dim=0)
    grad_q_weight = weight_grads[: g["d"]] if q_norm_weight is not None else None
    grad_k_weight = weight_grads[g["d"] :] if k_norm_weight is not None else None
    return grad_qkv, grad_q_weight, grad_k_weight
