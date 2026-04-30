# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch wrapper for the Triton MLA attention kernels.

Public entrypoint: :func:`mla_attention`. Forward and backward both run on
Triton kernels (FA-2 style). The pure-PyTorch :func:`mla_attention_ref` is
kept as the test reference.

The kernels do NOT use ``tl.atomic_add`` (forward owns one O slice per program;
backward uses two passes — dQ owned by Q-tile programs, dK/dV owned by K-tile
programs). Results are deterministic, so no ``NVTE_ALLOW_NONDETERMINISTIC_ALGO``
gate is needed.
"""

from typing import Optional

import torch
import triton

from transformer_engine.common.triton.mla import (
    _mla_attn_fwd,
    _mla_attn_bwd_preprocess,
    _mla_attn_bwd_dq,
    _mla_attn_bwd_dkv,
    _mla_decode_attn_fwd,
)


_SUPPORTED_QKV_FORMATS = ("bshd", "bhsd", "sbhd")


def _user_to_bhsd(t: torch.Tensor, qkv_format: str) -> torch.Tensor:
    """User-layout tensor -> contiguous BHSD."""
    if qkv_format == "bhsd":
        return t.contiguous() if not t.is_contiguous() else t
    if qkv_format == "bshd":
        # [B, S, H, D] -> [B, H, S, D]
        return t.transpose(1, 2).contiguous()
    if qkv_format == "sbhd":
        # [S, B, H, D] -> [B, H, S, D]
        return t.permute(1, 2, 0, 3).contiguous()
    raise ValueError(
        f"Unsupported qkv_format: {qkv_format!r}. Expected one of {_SUPPORTED_QKV_FORMATS}."
    )


def _bhsd_to_user(t: torch.Tensor, qkv_format: str) -> torch.Tensor:
    """BHSD tensor -> contiguous user layout."""
    if qkv_format == "bhsd":
        return t
    if qkv_format == "bshd":
        # [B, H, S, D] -> [B, S, H, D]
        return t.transpose(1, 2).contiguous()
    if qkv_format == "sbhd":
        # [B, H, S, D] -> [S, B, H, D]
        return t.permute(2, 0, 1, 3).contiguous()
    raise ValueError(
        f"Unsupported qkv_format: {qkv_format!r}. Expected one of {_SUPPORTED_QKV_FORMATS}."
    )


def mla_attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """Pure-PyTorch reference MLA attention in BHSD layout.

    Supports ``head_dim_qk != head_dim_v``. Internal compute is fp32; output is
    cast back to the input dtype. Right-aligned causal mask matches the kernel:
    row ``i`` attends to keys ``j <= i + (S_kv - S_q)``.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    in_dtype = q.dtype
    q32 = q.float()
    k32 = k.float()
    v32 = v.float()

    s = torch.matmul(q32, k32.transpose(-1, -2)) * softmax_scale
    if is_causal:
        s_q, s_kv = s.shape[-2], s.shape[-1]
        row = torch.arange(s_q, device=s.device).unsqueeze(-1)
        col = torch.arange(s_kv, device=s.device).unsqueeze(0)
        mask = col > (row + (s_kv - s_q))
        s = s.masked_fill(mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    out = torch.matmul(p, v32)
    return out.to(in_dtype)


def _launch_mla_fwd(q_bhsd, k_bhsd, v_bhsd, softmax_scale, is_causal):
    B, H, S_q, D_qk = q_bhsd.shape
    S_kv = k_bhsd.shape[2]
    D_v = v_bhsd.shape[3]

    o = torch.empty((B, H, S_q, D_v), device=q_bhsd.device, dtype=q_bhsd.dtype)
    lse = torch.empty((B, H, S_q), device=q_bhsd.device, dtype=torch.float32)

    block_dmodel_qk = max(16, triton.next_power_of_2(D_qk))
    block_dmodel_v = max(16, triton.next_power_of_2(D_v))

    grid = lambda meta: (triton.cdiv(S_q, meta["BLOCK_M"]), B * H)

    _mla_attn_fwd[grid](
        q_bhsd,
        k_bhsd,
        v_bhsd,
        o,
        lse,
        float(softmax_scale),
        B,
        H,
        S_q,
        S_kv,
        D_qk,
        D_v,
        # Q strides
        q_bhsd.stride(0),
        q_bhsd.stride(1),
        q_bhsd.stride(2),
        1,
        # K strides
        k_bhsd.stride(0),
        k_bhsd.stride(1),
        k_bhsd.stride(2),
        1,
        # V strides
        v_bhsd.stride(0),
        v_bhsd.stride(1),
        v_bhsd.stride(2),
        1,
        # O strides
        o.stride(0),
        o.stride(1),
        o.stride(2),
        1,
        # LSE strides
        lse.stride(0),
        lse.stride(1),
        1,
        IS_CAUSAL=is_causal,
        BLOCK_DMODEL_QK=block_dmodel_qk,
        BLOCK_DMODEL_V=block_dmodel_v,
    )
    return o, lse


# Preprocess kernel uses a fixed BLOCK_M (it's I/O-bound and the choice barely
# matters); the dQ / dKV kernels autotune and pick their own BLOCK_M / BLOCK_N.
_BWD_PREPROCESS_BLOCK_M = 128


def _launch_mla_bwd(q_bhsd, k_bhsd, v_bhsd, o_bhsd, lse, do_bhsd, softmax_scale, is_causal):
    """Run the three backward kernels and return ``(dQ, dK, dV)`` in BHSD."""
    B, H, S_q, D_qk = q_bhsd.shape
    S_kv = k_bhsd.shape[2]
    D_v = v_bhsd.shape[3]

    block_dmodel_qk = max(16, triton.next_power_of_2(D_qk))
    block_dmodel_v = max(16, triton.next_power_of_2(D_v))

    delta = torch.empty((B, H, S_q), device=q_bhsd.device, dtype=torch.float32)
    dq = torch.empty_like(q_bhsd)
    dk = torch.empty_like(k_bhsd)
    dv = torch.empty_like(v_bhsd)

    # 1) Delta = rowsum(O * dO)
    grid_pre = (triton.cdiv(S_q, _BWD_PREPROCESS_BLOCK_M), B * H)
    _mla_attn_bwd_preprocess[grid_pre](
        o_bhsd,
        do_bhsd,
        delta,
        B,
        H,
        S_q,
        D_v,
        # O strides
        o_bhsd.stride(0),
        o_bhsd.stride(1),
        o_bhsd.stride(2),
        1,
        # dO strides
        do_bhsd.stride(0),
        do_bhsd.stride(1),
        do_bhsd.stride(2),
        1,
        # Delta strides
        delta.stride(0),
        delta.stride(1),
        1,
        BLOCK_M=_BWD_PREPROCESS_BLOCK_M,
        BLOCK_DMODEL_V=block_dmodel_v,
    )

    common_args = (
        q_bhsd,
        k_bhsd,
        v_bhsd,
        do_bhsd,
        lse,
        delta,
    )
    common_shape_args = (B, H, S_q, S_kv, D_qk, D_v)
    common_strides = (
        # Q
        q_bhsd.stride(0), q_bhsd.stride(1), q_bhsd.stride(2), 1,
        # K
        k_bhsd.stride(0), k_bhsd.stride(1), k_bhsd.stride(2), 1,
        # V
        v_bhsd.stride(0), v_bhsd.stride(1), v_bhsd.stride(2), 1,
        # dO
        do_bhsd.stride(0), do_bhsd.stride(1), do_bhsd.stride(2), 1,
        # LSE
        lse.stride(0), lse.stride(1), 1,
        # Delta
        delta.stride(0), delta.stride(1), 1,
    )

    # 2) dQ
    grid_dq = lambda meta: (triton.cdiv(S_q, meta["BLOCK_M"]), B * H)
    _mla_attn_bwd_dq[grid_dq](
        *common_args,
        dq,
        float(softmax_scale),
        *common_shape_args,
        *common_strides,
        # dQ strides
        dq.stride(0), dq.stride(1), dq.stride(2), 1,
        IS_CAUSAL=is_causal,
        BLOCK_DMODEL_QK=block_dmodel_qk,
        BLOCK_DMODEL_V=block_dmodel_v,
    )

    # 3) dK, dV
    grid_dkv = lambda meta: (triton.cdiv(S_kv, meta["BLOCK_N"]), B * H)
    _mla_attn_bwd_dkv[grid_dkv](
        *common_args,
        dk,
        dv,
        float(softmax_scale),
        *common_shape_args,
        *common_strides,
        # dK strides
        dk.stride(0), dk.stride(1), dk.stride(2), 1,
        # dV strides
        dv.stride(0), dv.stride(1), dv.stride(2), 1,
        IS_CAUSAL=is_causal,
        BLOCK_DMODEL_QK=block_dmodel_qk,
        BLOCK_DMODEL_V=block_dmodel_v,
    )

    return dq, dk, dv


class MLAttentionFn(torch.autograd.Function):
    """Forward and backward via Triton kernels (FA-2 style)."""

    @staticmethod
    def forward(ctx, q, k, v, softmax_scale, is_causal, qkv_format):
        if qkv_format not in _SUPPORTED_QKV_FORMATS:
            raise ValueError(
                f"qkv_format must be one of {_SUPPORTED_QKV_FORMATS}, got {qkv_format!r}"
            )
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"mla_attention requires fp16 or bf16 inputs, got {q.dtype}"
            )
        if not (q.dtype == k.dtype == v.dtype):
            raise ValueError("q, k, v must share the same dtype")
        if not q.is_cuda:
            raise ValueError("mla_attention requires CUDA tensors")

        q_bhsd = _user_to_bhsd(q, qkv_format)
        k_bhsd = _user_to_bhsd(k, qkv_format)
        v_bhsd = _user_to_bhsd(v, qkv_format)

        if q_bhsd.shape[3] != k_bhsd.shape[3]:
            raise ValueError(
                "q.head_dim and k.head_dim must match"
                f" (got {q_bhsd.shape[3]} vs {k_bhsd.shape[3]})"
            )
        if q_bhsd.shape[:2] != k_bhsd.shape[:2] or q_bhsd.shape[:2] != v_bhsd.shape[:2]:
            raise ValueError("q, k, v must share batch and head dimensions")
        if k_bhsd.shape[2] != v_bhsd.shape[2]:
            raise ValueError("k.seq_len and v.seq_len must match")

        o_bhsd, lse = _launch_mla_fwd(q_bhsd, k_bhsd, v_bhsd, softmax_scale, is_causal)

        ctx.save_for_backward(q_bhsd, k_bhsd, v_bhsd, o_bhsd, lse)
        ctx.softmax_scale = softmax_scale
        ctx.is_causal = is_causal
        ctx.qkv_format = qkv_format

        return _bhsd_to_user(o_bhsd, qkv_format)

    @staticmethod
    def backward(ctx, grad_o):
        q_bhsd, k_bhsd, v_bhsd, o_bhsd, lse = ctx.saved_tensors
        grad_o_bhsd = _user_to_bhsd(grad_o, ctx.qkv_format)

        dq_bhsd, dk_bhsd, dv_bhsd = _launch_mla_bwd(
            q_bhsd,
            k_bhsd,
            v_bhsd,
            o_bhsd,
            lse,
            grad_o_bhsd,
            ctx.softmax_scale,
            ctx.is_causal,
        )

        grad_q = _bhsd_to_user(dq_bhsd, ctx.qkv_format)
        grad_k = _bhsd_to_user(dk_bhsd, ctx.qkv_format)
        grad_v = _bhsd_to_user(dv_bhsd, ctx.qkv_format)
        return grad_q, grad_k, grad_v, None, None, None


def mla_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    is_causal: bool = False,
    qkv_format: str = "bshd",
) -> torch.Tensor:
    """Triton MLA attention (FA-2 style forward and backward) for SM80+.

    Supports ``head_dim_qk != head_dim_v`` (DeepSeek-V2/V3 style). Both forward
    and backward run as Triton kernels: the forward saves an fp32 ``LSE``, which
    the backward consumes to recompute softmax probabilities without extra
    memory.

    Parameters
    ----------
    q, k, v
        Tensors in ``qkv_format`` layout. ``q`` and ``k`` must share the last
        dim (head_dim_qk); ``v`` may have a different last dim (head_dim_v).
        Dtype must be fp16 or bf16. Batch and head dims must match across the
        three tensors; ``k`` and ``v`` must share seq_len.
    softmax_scale
        Multiplier applied to ``Q @ K^T`` before softmax. Defaults to
        ``1 / sqrt(head_dim_qk)``.
    is_causal
        If True, applies a right-aligned causal mask: row ``i`` attends to
        keys ``j <= i + (S_kv - S_q)``. With ``S_q == S_kv`` this is the
        standard causal self-attention mask.
    qkv_format
        ``"bshd"`` (default, matches FlashAttention), ``"bhsd"`` (matches
        ``F.scaled_dot_product_attention``), or ``"sbhd"`` (megatron-style).

    Returns
    -------
    torch.Tensor
        Attention output in ``qkv_format`` layout, dtype matching the inputs.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    return MLAttentionFn.apply(q, k, v, softmax_scale, is_causal, qkv_format)


# ---------------------------------------------------------------------------
# Decode (absorbed up-projection over compressed KV cache)
# ---------------------------------------------------------------------------


def mla_decode_attention_ref(
    q_nope_abs: torch.Tensor,
    q_rope: torch.Tensor,
    c_kv: torch.Tensor,
    k_rope: torch.Tensor,
    softmax_scale: float,
    is_causal: bool = False,
) -> torch.Tensor:
    """Pure-PyTorch reference for absorbed MLA decode attention.

    Inputs (BHSD layout for Q-side, BSR layout for compressed cache):

    - ``q_nope_abs`` ``[B, H, S_q, R]`` — Q's nope side already multiplied by
      ``W_uk^T``. ``R`` is ``kv_lora_rank``.
    - ``q_rope`` ``[B, H, S_q, R_rope]`` — Q's rope side.
    - ``c_kv`` ``[B, S_kv, R]`` — compressed KV cache, shared across heads.
    - ``k_rope`` ``[B, S_kv, R_rope]`` — decoupled rope keys, shared across heads.
    - ``softmax_scale`` — typically ``1 / sqrt(head_dim_qk_orig)`` where
      ``head_dim_qk_orig = q_nope_dim + R_rope`` (the *un-absorbed* head dim).

    Returns ``o_inter`` ``[B, H, S_q, R]``. The caller multiplies by ``W_uv``
    (per head) to obtain the final attention output.
    """
    in_dtype = q_nope_abs.dtype
    qn = q_nope_abs.float()
    qr = q_rope.float()
    ck = c_kv.float()
    kr = k_rope.float()

    score_nope = torch.einsum("bhmr,bnr->bhmn", qn, ck)
    score_rope = torch.einsum("bhmr,bnr->bhmn", qr, kr)
    s = (score_nope + score_rope) * softmax_scale

    if is_causal:
        s_q, s_kv = s.shape[-2], s.shape[-1]
        row = torch.arange(s_q, device=s.device).unsqueeze(-1)
        col = torch.arange(s_kv, device=s.device).unsqueeze(0)
        mask = col > (row + (s_kv - s_q))
        s = s.masked_fill(mask, float("-inf"))

    p = torch.softmax(s, dim=-1)
    o_inter = torch.einsum("bhmn,bnr->bhmr", p, ck)
    return o_inter.to(in_dtype)


def _launch_mla_decode_fwd(q_nope_abs, q_rope, c_kv, k_rope, softmax_scale, is_causal):
    B, H, S_q, R = q_nope_abs.shape
    S_kv = c_kv.shape[1]
    R_rope = q_rope.shape[3]

    o_inter = torch.empty(
        (B, H, S_q, R), device=q_nope_abs.device, dtype=q_nope_abs.dtype
    )
    lse = torch.empty((B, H, S_q), device=q_nope_abs.device, dtype=torch.float32)

    block_dmodel_r = max(16, triton.next_power_of_2(R))
    block_dmodel_rr = max(16, triton.next_power_of_2(R_rope))

    grid = lambda meta: (triton.cdiv(S_q, meta["BLOCK_M"]), B * H)

    _mla_decode_attn_fwd[grid](
        q_nope_abs,
        q_rope,
        c_kv,
        k_rope,
        o_inter,
        lse,
        float(softmax_scale),
        B,
        H,
        S_q,
        S_kv,
        R,
        R_rope,
        # Q_nope_abs strides
        q_nope_abs.stride(0), q_nope_abs.stride(1), q_nope_abs.stride(2), 1,
        # Q_rope strides
        q_rope.stride(0), q_rope.stride(1), q_rope.stride(2), 1,
        # c_kv strides (no H)
        c_kv.stride(0), c_kv.stride(1), 1,
        # k_rope strides (no H)
        k_rope.stride(0), k_rope.stride(1), 1,
        # O_inter strides
        o_inter.stride(0), o_inter.stride(1), o_inter.stride(2), 1,
        # LSE strides
        lse.stride(0), lse.stride(1), 1,
        IS_CAUSAL=is_causal,
        BLOCK_DMODEL_R=block_dmodel_r,
        BLOCK_DMODEL_RR=block_dmodel_rr,
    )
    return o_inter, lse


def mla_decode_attention(
    q_nope_abs: torch.Tensor,
    q_rope: torch.Tensor,
    c_kv: torch.Tensor,
    k_rope: torch.Tensor,
    *,
    softmax_scale: float,
    is_causal: bool = False,
) -> torch.Tensor:
    """Triton MLA decode forward over a compressed KV cache (SM80+).

    Implements the FlashMLA-style absorbed up-projection: ``c_kv`` plays the
    role of both K (nope side) and V, so per-head K/V are never materialized
    inside the kernel. The caller is expected to apply ``W_uv`` to the
    returned ``O_inter`` to obtain the final attention output.

    Parameters
    ----------
    q_nope_abs : ``[B, H, S_q, R]``
        Q's nope side after absorbing ``W_uk^T`` (so its last-dim is
        ``kv_lora_rank``, not the original Q nope dim).
    q_rope : ``[B, H, S_q, R_rope]``
        Q's rope side.
    c_kv : ``[B, S_kv, R]``
        Compressed KV cache, shared across heads.
    k_rope : ``[B, S_kv, R_rope]``
        Decoupled rope keys, shared across heads.
    softmax_scale : float
        Required. Typically ``1 / sqrt(q_nope_dim + R_rope)`` — i.e. the
        original head_dim_qk before absorption. **No default**: the absorbed
        Q's last dim (``R``) is *not* the right denominator, so the kernel
        will not guess.
    is_causal : bool
        Right-aligned causal: row ``i`` attends to keys ``j <= i + (S_kv - S_q)``.

    Returns
    -------
    torch.Tensor
        ``O_inter`` of shape ``[B, H, S_q, R]`` in the input dtype. Apply
        ``W_uv`` outside the kernel for the final attention output.
    """
    if q_nope_abs.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"mla_decode_attention requires fp16 or bf16 inputs, got {q_nope_abs.dtype}"
        )
    if not (
        q_nope_abs.dtype == q_rope.dtype == c_kv.dtype == k_rope.dtype
    ):
        raise ValueError("q_nope_abs, q_rope, c_kv, k_rope must share dtype")
    if not q_nope_abs.is_cuda:
        raise ValueError("mla_decode_attention requires CUDA tensors")

    if q_nope_abs.shape[-1] != c_kv.shape[-1]:
        raise ValueError(
            "q_nope_abs.last_dim and c_kv.last_dim must match (kv_lora_rank);"
            f" got {q_nope_abs.shape[-1]} vs {c_kv.shape[-1]}"
        )
    if q_rope.shape[-1] != k_rope.shape[-1]:
        raise ValueError(
            "q_rope.last_dim and k_rope.last_dim must match (rope_dim);"
            f" got {q_rope.shape[-1]} vs {k_rope.shape[-1]}"
        )
    if q_nope_abs.shape[:3] != q_rope.shape[:3]:
        raise ValueError("q_nope_abs and q_rope must share (B, H, S_q)")
    if c_kv.shape[:2] != k_rope.shape[:2]:
        raise ValueError("c_kv and k_rope must share (B, S_kv)")
    if q_nope_abs.shape[0] != c_kv.shape[0]:
        raise ValueError("Q-side and KV-cache batch dims must match")

    qn = q_nope_abs.contiguous() if not q_nope_abs.is_contiguous() else q_nope_abs
    qr = q_rope.contiguous() if not q_rope.is_contiguous() else q_rope
    ck = c_kv.contiguous() if not c_kv.is_contiguous() else c_kv
    kr = k_rope.contiguous() if not k_rope.is_contiguous() else k_rope

    o_inter, _lse = _launch_mla_decode_fwd(qn, qr, ck, kr, softmax_scale, is_causal)
    return o_inter
