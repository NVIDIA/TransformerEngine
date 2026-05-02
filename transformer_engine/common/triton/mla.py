# pylint: disable=missing-function-docstring

# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Triton kernels for MLA (Multi-head Latent Attention).

Two operating modes:

1. **Prefill / training** (``_mla_attn_fwd`` + backward kernels):
   FlashAttention-2 style with non-square head dimensions
   (``head_dim_qk != head_dim_v`` as used by DeepSeek-V2/V3). Inputs are full
   Q, K, V tensors. Backward uses the canonical FA-2 three-pass structure
   (preprocess for ``Delta = rowsum(O * dO)``, then ``dQ`` and ``dK``/``dV``
   kernels). No atomics — each program owns a distinct output slice.

2. **Absorbed-projection decode** (``_mla_decode_attn_fwd``):
   Operates on a compressed KV cache (latent ``c_kv`` of dim ``kv_lora_rank``
   plus a decoupled rope key ``k_rope``). Q's nope side is pre-absorbed via
   ``W_uk^T`` so the kernel never materializes per-head K or V; ``c_kv``
   serves as both the key (nope side) and the value. Output is the
   pre-``W_uv`` intermediate ``O_inter`` of shape ``[B, H, S_q, kv_lora_rank]``;
   the caller applies ``W_uv`` to produce the final attention output.

Tuned for SM80 (A100). Compiles on SM89/SM90 too but is not specialized for them.
"""

import itertools
import os

import triton
import triton.language as tl


def _mla_fwd_configs():
    block_m = [64, 128]
    block_n = [32, 64]
    num_warps = [4, 8]
    num_stages = [2, 3]

    configs = []
    for m, n, w, s in itertools.product(block_m, block_n, num_warps, num_stages):
        configs.append(
            triton.Config(
                {"BLOCK_M": m, "BLOCK_N": n},
                num_warps=w,
                num_stages=s,
            )
        )
    if os.environ.get("NVTE_DISABLE_TRITON_AUTOTUNING", "0") == "1":
        configs = configs[:1]
    return configs


@triton.autotune(
    configs=_mla_fwd_configs(),
    key=["S_q", "S_kv", "D_qk", "D_v", "IS_CAUSAL"],
)
@triton.jit
def _mla_attn_fwd(
    Q_ptr,  # (B, H, S_q,  D_qk)
    K_ptr,  # (B, H, S_kv, D_qk)
    V_ptr,  # (B, H, S_kv, D_v)
    O_ptr,  # (B, H, S_q,  D_v)
    LSE_ptr,  # (B, H, S_q) fp32
    softmax_scale,
    B,
    H,
    S_q,
    S_kv,
    D_qk,
    D_v,
    sQ_b,
    sQ_h,
    sQ_s,
    sQ_d: tl.constexpr,
    sK_b,
    sK_h,
    sK_s,
    sK_d: tl.constexpr,
    sV_b,
    sV_h,
    sV_s,
    sV_d: tl.constexpr,
    sO_b,
    sO_h,
    sO_s,
    sO_d: tl.constexpr,
    sL_b,
    sL_h,
    sL_s: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
):
    """One program -> one BLOCK_M tile of Q rows for one (batch, head).

    Layout: BHSD. Right-aligned causal: row i attends to keys j s.t.
    ``j <= i + (S_kv - S_q)``.
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    tl.assume(pid_m >= 0)
    tl.assume(pid_bh >= 0)

    off_b = pid_bh // H
    off_h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_init = tl.arange(0, BLOCK_N)
    offs_d_qk = tl.arange(0, BLOCK_DMODEL_QK)
    offs_d_v = tl.arange(0, BLOCK_DMODEL_V)

    mask_m = offs_m < S_q
    mask_d_qk = offs_d_qk < D_qk
    mask_d_v = offs_d_v < D_v

    # Load Q tile once (resident across the inner K/V loop).
    q_base = Q_ptr + off_b * sQ_b + off_h * sQ_h
    q_ptrs = q_base + offs_m[:, None] * sQ_s + offs_d_qk[None, :] * sQ_d
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d_qk[None, :], other=0.0)

    k_base = K_ptr + off_b * sK_b + off_h * sK_h
    v_base = V_ptr + off_b * sV_b + off_h * sV_h

    # ``m_i`` is initialized to a finite "very negative" sentinel rather than -inf
    # so that exp(m_i - m_ij) is well-defined even when an entire K block is masked
    # (every qk == -inf, e.g. the rare S_q > S_kv + causal case). -1e6 is large
    # enough that exp(-1e6 - any_realistic_qk) underflows to 0 in fp32.
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1.0e6
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_V], dtype=tl.float32)

    # Loop bound: standard FA-2 early exit for causal. No clamp to 0 needed —
    # ``range`` with a non-positive upper bound is empty.
    if IS_CAUSAL:
        hi = tl.minimum((pid_m + 1) * BLOCK_M + (S_kv - S_q), S_kv)
    else:
        hi = S_kv

    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_init
        mask_n = offs_n < S_kv

        # Load K tile [BLOCK_N, BLOCK_DMODEL_QK].
        k_ptrs = k_base + offs_n[:, None] * sK_s + offs_d_qk[None, :] * sK_d
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d_qk[None, :], other=0.0)

        # qk = Q @ K^T  ->  [BLOCK_M, BLOCK_N]  (fp32 accum)
        qk = tl.dot(q, tl.trans(k))
        qk = qk * softmax_scale

        # Mask invalid keys (K-tail past S_kv) to -inf BEFORE softmax max/exp.
        qk = tl.where(mask_n[None, :], qk, float("-inf"))
        if IS_CAUSAL:
            causal_mask = offs_n[None, :] <= (offs_m[:, None] + (S_kv - S_q))
            qk = tl.where(causal_mask, qk, float("-inf"))

        # Online softmax. ``m_i`` is initialized to a finite sentinel so that
        # exp(m_i - m_ij) is well-defined even when this entire block is masked
        # (all qk == -inf).
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_ij)
        p = tl.exp(qk - m_ij[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        # Load V tile [BLOCK_N, BLOCK_DMODEL_V] and accumulate P @ V.
        v_ptrs = v_base + offs_n[:, None] * sV_s + offs_d_v[None, :] * sV_d
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d_v[None, :], other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)

        m_i = m_ij

    # Epilogue: normalize and store O in input dtype.
    acc = acc / l_i[:, None]
    o_base = O_ptr + off_b * sO_b + off_h * sO_h
    o_ptrs = o_base + offs_m[:, None] * sO_s + offs_d_v[None, :] * sO_d
    tl.store(
        o_ptrs,
        acc.to(O_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_d_v[None, :],
    )

    # Store fp32 LSE = m_i + log(l_i) (used by the analytical backward).
    lse_base = LSE_ptr + off_b * sL_b + off_h * sL_h
    lse_ptrs = lse_base + offs_m * sL_s
    tl.store(lse_ptrs, m_i + tl.log(l_i), mask=mask_m)


# ---------------------------------------------------------------------------
# Backward kernels (FA-2 style, three passes, no atomics)
# ---------------------------------------------------------------------------


def _mla_bwd_configs():
    block_m = [64, 128]
    block_n = [32, 64]
    num_warps = [4, 8]
    num_stages = [2, 3]
    configs = []
    for m, n, w, s in itertools.product(block_m, block_n, num_warps, num_stages):
        configs.append(
            triton.Config(
                {"BLOCK_M": m, "BLOCK_N": n},
                num_warps=w,
                num_stages=s,
            )
        )
    if os.environ.get("NVTE_DISABLE_TRITON_AUTOTUNING", "0") == "1":
        configs = configs[:1]
    return configs


@triton.jit
def _mla_attn_bwd_preprocess(
    O_ptr,  # (B, H, S_q, D_v)
    DO_ptr,  # (B, H, S_q, D_v)
    Delta_ptr,  # (B, H, S_q) fp32 output
    B,
    H,
    S_q,
    D_v,
    sO_b,
    sO_h,
    sO_s,
    sO_d: tl.constexpr,
    sDO_b,
    sDO_h,
    sDO_s,
    sDO_d: tl.constexpr,
    sD_b,
    sD_h,
    sD_s: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
):
    """Compute ``Delta[b,h,m] = sum_d O[b,h,m,d] * dO[b,h,m,d]`` in fp32."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    tl.assume(pid_m >= 0)
    tl.assume(pid_bh >= 0)

    off_b = pid_bh // H
    off_h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL_V)
    mask_m = offs_m < S_q
    mask_d = offs_d < D_v
    mask_md = mask_m[:, None] & mask_d[None, :]

    o_ptrs = O_ptr + off_b * sO_b + off_h * sO_h + offs_m[:, None] * sO_s + offs_d[None, :] * sO_d
    do_ptrs = (
        DO_ptr + off_b * sDO_b + off_h * sDO_h + offs_m[:, None] * sDO_s + offs_d[None, :] * sDO_d
    )
    o = tl.load(o_ptrs, mask=mask_md, other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask_md, other=0.0).to(tl.float32)

    delta = tl.sum(o * do, axis=1)
    delta_ptrs = Delta_ptr + off_b * sD_b + off_h * sD_h + offs_m * sD_s
    tl.store(delta_ptrs, delta, mask=mask_m)


@triton.autotune(
    configs=_mla_bwd_configs(),
    key=["S_q", "S_kv", "D_qk", "D_v", "IS_CAUSAL"],
)
@triton.jit
def _mla_attn_bwd_dq(
    Q_ptr,  # (B, H, S_q,  D_qk)
    K_ptr,  # (B, H, S_kv, D_qk)
    V_ptr,  # (B, H, S_kv, D_v)
    DO_ptr,  # (B, H, S_q,  D_v)
    LSE_ptr,  # (B, H, S_q) fp32
    Delta_ptr,  # (B, H, S_q) fp32
    DQ_ptr,  # (B, H, S_q,  D_qk) output
    softmax_scale,
    B,
    H,
    S_q,
    S_kv,
    D_qk,
    D_v,
    sQ_b,
    sQ_h,
    sQ_s,
    sQ_d: tl.constexpr,
    sK_b,
    sK_h,
    sK_s,
    sK_d: tl.constexpr,
    sV_b,
    sV_h,
    sV_s,
    sV_d: tl.constexpr,
    sDO_b,
    sDO_h,
    sDO_s,
    sDO_d: tl.constexpr,
    sLSE_b,
    sLSE_h,
    sLSE_s: tl.constexpr,
    sD_b,
    sD_h,
    sD_s: tl.constexpr,
    sDQ_b,
    sDQ_h,
    sDQ_s,
    sDQ_d: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
):
    """Compute ``dQ`` for one BLOCK_M row tile of one (batch, head)."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    tl.assume(pid_m >= 0)
    tl.assume(pid_bh >= 0)

    off_b = pid_bh // H
    off_h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_init = tl.arange(0, BLOCK_N)
    offs_d_qk = tl.arange(0, BLOCK_DMODEL_QK)
    offs_d_v = tl.arange(0, BLOCK_DMODEL_V)

    mask_m = offs_m < S_q
    mask_d_qk = offs_d_qk < D_qk
    mask_d_v = offs_d_v < D_v

    # Load Q, dO, LSE, Delta — resident across the K/V loop.
    q_base = Q_ptr + off_b * sQ_b + off_h * sQ_h
    q_ptrs = q_base + offs_m[:, None] * sQ_s + offs_d_qk[None, :] * sQ_d
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d_qk[None, :], other=0.0)

    do_base = DO_ptr + off_b * sDO_b + off_h * sDO_h
    do_ptrs = do_base + offs_m[:, None] * sDO_s + offs_d_v[None, :] * sDO_d
    do = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d_v[None, :], other=0.0)

    # ``other=+inf`` so that exp(qk - lse) underflows to 0 for invalid Q rows
    # and they contribute nothing to dQ.
    lse_base = LSE_ptr + off_b * sLSE_b + off_h * sLSE_h
    lse = tl.load(lse_base + offs_m * sLSE_s, mask=mask_m, other=float("inf"))

    delta_base = Delta_ptr + off_b * sD_b + off_h * sD_h
    delta = tl.load(delta_base + offs_m * sD_s, mask=mask_m, other=0.0)

    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL_QK], dtype=tl.float32)

    if IS_CAUSAL:
        n_hi = tl.minimum((pid_m + 1) * BLOCK_M + (S_kv - S_q), S_kv)
    else:
        n_hi = S_kv

    k_base = K_ptr + off_b * sK_b + off_h * sK_h
    v_base = V_ptr + off_b * sV_b + off_h * sV_h

    for start_n in range(0, n_hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_init
        mask_n = offs_n < S_kv

        k_ptrs = k_base + offs_n[:, None] * sK_s + offs_d_qk[None, :] * sK_d
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d_qk[None, :], other=0.0)
        v_ptrs = v_base + offs_n[:, None] * sV_s + offs_d_v[None, :] * sV_d
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d_v[None, :], other=0.0)

        # Recompute qk and apply same masks as the forward.
        qk = tl.dot(q, tl.trans(k))
        qk = qk * softmax_scale
        qk = tl.where(mask_n[None, :], qk, float("-inf"))
        if IS_CAUSAL:
            causal_mask = offs_n[None, :] <= (offs_m[:, None] + (S_kv - S_q))
            qk = tl.where(causal_mask, qk, float("-inf"))

        p = tl.exp(qk - lse[:, None])
        # dP = dO @ V^T  ->  [BLOCK_M, BLOCK_N]
        dp = tl.dot(do, tl.trans(v))
        # dS = P * (dP - Delta) * scale
        ds = (p * (dp - delta[:, None])) * softmax_scale
        # dQ += dS @ K
        dq += tl.dot(ds.to(k.dtype), k)

    dq_base = DQ_ptr + off_b * sDQ_b + off_h * sDQ_h
    dq_ptrs = dq_base + offs_m[:, None] * sDQ_s + offs_d_qk[None, :] * sDQ_d
    tl.store(
        dq_ptrs,
        dq.to(DQ_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_d_qk[None, :],
    )


@triton.autotune(
    configs=_mla_bwd_configs(),
    key=["S_q", "S_kv", "D_qk", "D_v", "IS_CAUSAL"],
)
@triton.jit
def _mla_attn_bwd_dkv(
    Q_ptr,
    K_ptr,
    V_ptr,
    DO_ptr,
    LSE_ptr,
    Delta_ptr,
    DK_ptr,  # (B, H, S_kv, D_qk) output
    DV_ptr,  # (B, H, S_kv, D_v)  output
    softmax_scale,
    B,
    H,
    S_q,
    S_kv,
    D_qk,
    D_v,
    sQ_b,
    sQ_h,
    sQ_s,
    sQ_d: tl.constexpr,
    sK_b,
    sK_h,
    sK_s,
    sK_d: tl.constexpr,
    sV_b,
    sV_h,
    sV_s,
    sV_d: tl.constexpr,
    sDO_b,
    sDO_h,
    sDO_s,
    sDO_d: tl.constexpr,
    sLSE_b,
    sLSE_h,
    sLSE_s: tl.constexpr,
    sD_b,
    sD_h,
    sD_s: tl.constexpr,
    sDK_b,
    sDK_h,
    sDK_s,
    sDK_d: tl.constexpr,
    sDV_b,
    sDV_h,
    sDV_s,
    sDV_d: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
):
    """Compute ``dK`` and ``dV`` for one BLOCK_N tile of one (batch, head)."""
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    tl.assume(pid_n >= 0)
    tl.assume(pid_bh >= 0)

    off_b = pid_bh // H
    off_h = pid_bh % H

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m_init = tl.arange(0, BLOCK_M)
    offs_d_qk = tl.arange(0, BLOCK_DMODEL_QK)
    offs_d_v = tl.arange(0, BLOCK_DMODEL_V)

    mask_n = offs_n < S_kv
    mask_d_qk = offs_d_qk < D_qk
    mask_d_v = offs_d_v < D_v

    # Load K, V tiles — resident across the Q loop.
    k_base = K_ptr + off_b * sK_b + off_h * sK_h
    k_ptrs = k_base + offs_n[:, None] * sK_s + offs_d_qk[None, :] * sK_d
    k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d_qk[None, :], other=0.0)

    v_base = V_ptr + off_b * sV_b + off_h * sV_h
    v_ptrs = v_base + offs_n[:, None] * sV_s + offs_d_v[None, :] * sV_d
    v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d_v[None, :], other=0.0)

    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL_QK], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL_V], dtype=tl.float32)

    # Causal: only Q rows i with j_max(i) >= n_start contribute. Round the
    # lower bound *down* to a multiple of BLOCK_M so loads stay aligned; the
    # extra rows iterated are masked out per-element.
    if IS_CAUSAL:
        m_lo = pid_n * BLOCK_N - (S_kv - S_q)
        m_lo = tl.maximum(m_lo, 0)
        m_lo = (m_lo // BLOCK_M) * BLOCK_M
    else:
        m_lo = 0

    for start_m in range(m_lo, S_q, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m = start_m + offs_m_init
        mask_m = offs_m < S_q

        q_base = Q_ptr + off_b * sQ_b + off_h * sQ_h
        q_ptrs = q_base + offs_m[:, None] * sQ_s + offs_d_qk[None, :] * sQ_d
        q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d_qk[None, :], other=0.0)

        # Recompute qk with the forward's mask convention.
        qk = tl.dot(q, tl.trans(k))
        qk = qk * softmax_scale
        qk = tl.where(mask_n[None, :], qk, float("-inf"))
        if IS_CAUSAL:
            causal_mask = offs_n[None, :] <= (offs_m[:, None] + (S_kv - S_q))
            qk = tl.where(causal_mask, qk, float("-inf"))

        # Invalid Q rows: load LSE with ``+inf`` so P underflows to 0 and these
        # rows contribute nothing to dK/dV.
        lse_base = LSE_ptr + off_b * sLSE_b + off_h * sLSE_h
        lse = tl.load(lse_base + offs_m * sLSE_s, mask=mask_m, other=float("inf"))
        delta_base = Delta_ptr + off_b * sD_b + off_h * sD_h
        delta = tl.load(delta_base + offs_m * sD_s, mask=mask_m, other=0.0)

        do_base = DO_ptr + off_b * sDO_b + off_h * sDO_h
        do_ptrs = do_base + offs_m[:, None] * sDO_s + offs_d_v[None, :] * sDO_d
        do = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d_v[None, :], other=0.0)

        p = tl.exp(qk - lse[:, None])  # [BLOCK_M, BLOCK_N]

        # dV += P^T @ dO
        dv += tl.dot(tl.trans(p).to(do.dtype), do)

        # dP = dO @ V^T  ->  [BLOCK_M, BLOCK_N]
        dp = tl.dot(do, tl.trans(v))
        ds = (p * (dp - delta[:, None])) * softmax_scale
        # dK += dS^T @ Q
        dk += tl.dot(tl.trans(ds).to(q.dtype), q)

    dk_base = DK_ptr + off_b * sDK_b + off_h * sDK_h
    dk_ptrs = dk_base + offs_n[:, None] * sDK_s + offs_d_qk[None, :] * sDK_d
    tl.store(
        dk_ptrs,
        dk.to(DK_ptr.dtype.element_ty),
        mask=mask_n[:, None] & mask_d_qk[None, :],
    )

    dv_base = DV_ptr + off_b * sDV_b + off_h * sDV_h
    dv_ptrs = dv_base + offs_n[:, None] * sDV_s + offs_d_v[None, :] * sDV_d
    tl.store(
        dv_ptrs,
        dv.to(DV_ptr.dtype.element_ty),
        mask=mask_n[:, None] & mask_d_v[None, :],
    )


# ---------------------------------------------------------------------------
# Decode forward kernel (absorbed up-projection over compressed KV cache)
# ---------------------------------------------------------------------------


def _mla_decode_fwd_configs():
    # Decode tiles are smaller than prefill because the effective K dim is
    # ``kv_lora_rank`` (e.g. 512) which is much larger than a normal head dim,
    # so SMEM pressure is high. Triton's autotune will silently prune configs
    # that exceed the SM80 SMEM budget.
    block_m = [16, 32, 64]
    block_n = [16, 32, 64]
    num_warps = [4, 8]
    num_stages = [2, 3]
    configs = []
    for m, n, w, s in itertools.product(block_m, block_n, num_warps, num_stages):
        configs.append(
            triton.Config(
                {"BLOCK_M": m, "BLOCK_N": n},
                num_warps=w,
                num_stages=s,
            )
        )
    if os.environ.get("NVTE_DISABLE_TRITON_AUTOTUNING", "0") == "1":
        configs = configs[:1]
    return configs


@triton.autotune(
    configs=_mla_decode_fwd_configs(),
    key=["S_q", "S_kv", "R", "R_rope", "IS_CAUSAL"],
)
@triton.jit
def _mla_decode_attn_fwd(
    QN_ptr,  # (B, H,    S_q,  R)        Q_nope_abs  (Q_nope already multiplied by W_uk^T)
    QR_ptr,  # (B, H,    S_q,  R_rope)   Q_rope
    CKV_ptr,  # (B,       S_kv, R)        compressed KV cache (shared across heads)
    KR_ptr,  # (B,       S_kv, R_rope)   decoupled rope key (shared across heads)
    O_ptr,  # (B, H,    S_q,  R)        O_inter (caller applies W_uv)
    softmax_scale,
    B,
    H,
    S_q,
    S_kv,
    R,
    R_rope,
    # Q_nope_abs strides
    sQN_b,
    sQN_h,
    sQN_s,
    sQN_r: tl.constexpr,
    # Q_rope strides
    sQR_b,
    sQR_h,
    sQR_s,
    sQR_r: tl.constexpr,
    # c_kv strides (no H)
    sCKV_b,
    sCKV_s,
    sCKV_r: tl.constexpr,
    # k_rope strides (no H)
    sKR_b,
    sKR_s,
    sKR_r: tl.constexpr,
    # O_inter strides
    sO_b,
    sO_h,
    sO_s,
    sO_r: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL_R: tl.constexpr,  # next_pow2(R)
    BLOCK_DMODEL_RR: tl.constexpr,  # next_pow2(R_rope)
):
    """Absorbed MLA decode forward — one BLOCK_M tile of Q rows for one (b, h).

    Computes ``score = Q_nope_abs @ c_kv^T + Q_rope @ k_rope^T`` (scaled),
    softmax, then ``O_inter = P @ c_kv``. ``c_kv`` is reused as both K (for the
    nope-side score) and V; per-head ``K_nope`` / ``V`` are never materialized.
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    tl.assume(pid_m >= 0)
    tl.assume(pid_bh >= 0)

    off_b = pid_bh // H
    off_h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_init = tl.arange(0, BLOCK_N)
    offs_r = tl.arange(0, BLOCK_DMODEL_R)
    offs_rr = tl.arange(0, BLOCK_DMODEL_RR)

    mask_m = offs_m < S_q
    mask_r = offs_r < R
    mask_rr = offs_rr < R_rope

    # Load Q_nope_abs and Q_rope tiles once.
    qn_base = QN_ptr + off_b * sQN_b + off_h * sQN_h
    qn_ptrs = qn_base + offs_m[:, None] * sQN_s + offs_r[None, :] * sQN_r
    qn = tl.load(qn_ptrs, mask=mask_m[:, None] & mask_r[None, :], other=0.0)

    qr_base = QR_ptr + off_b * sQR_b + off_h * sQR_h
    qr_ptrs = qr_base + offs_m[:, None] * sQR_s + offs_rr[None, :] * sQR_r
    qr = tl.load(qr_ptrs, mask=mask_m[:, None] & mask_rr[None, :], other=0.0)

    ckv_base = CKV_ptr + off_b * sCKV_b
    kr_base = KR_ptr + off_b * sKR_b

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1.0e6
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_R], dtype=tl.float32)

    if IS_CAUSAL:
        hi = tl.minimum((pid_m + 1) * BLOCK_M + (S_kv - S_q), S_kv)
    else:
        hi = S_kv

    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_init
        mask_n = offs_n < S_kv

        # c_kv tile [BLOCK_N, BLOCK_DMODEL_R] — used for both the nope-side
        # score and as the value tile.
        ckv_ptrs = ckv_base + offs_n[:, None] * sCKV_s + offs_r[None, :] * sCKV_r
        ckv = tl.load(ckv_ptrs, mask=mask_n[:, None] & mask_r[None, :], other=0.0)

        # k_rope tile [BLOCK_N, BLOCK_DMODEL_RR]
        kr_ptrs = kr_base + offs_n[:, None] * sKR_s + offs_rr[None, :] * sKR_r
        kr = tl.load(kr_ptrs, mask=mask_n[:, None] & mask_rr[None, :], other=0.0)

        # Scores: nope side is Q_nope_abs @ c_kv^T, rope side is Q_rope @ k_rope^T.
        score_nope = tl.dot(qn, tl.trans(ckv))
        score_rope = tl.dot(qr, tl.trans(kr))
        qk = (score_nope + score_rope) * softmax_scale

        qk = tl.where(mask_n[None, :], qk, float("-inf"))
        if IS_CAUSAL:
            causal_mask = offs_n[None, :] <= (offs_m[:, None] + (S_kv - S_q))
            qk = tl.where(causal_mask, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_ij)
        p = tl.exp(qk - m_ij[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        # Reuse the c_kv tile as V: O_inter += P @ c_kv.
        acc = tl.dot(p.to(ckv.dtype), ckv, acc)

        m_i = m_ij

    acc = acc / l_i[:, None]
    o_base = O_ptr + off_b * sO_b + off_h * sO_h
    o_ptrs = o_base + offs_m[:, None] * sO_s + offs_r[None, :] * sO_r
    tl.store(
        o_ptrs,
        acc.to(O_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_r[None, :],
    )
    # LSE is intentionally not saved here — v1 has no analytical decode
    # backward. If a future change adds one, re-introduce a fp32 LSE buffer
    # and ``m_i + tl.log(l_i)`` store at this point.
