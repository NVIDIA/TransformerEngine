# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused RoPE + FP8 current-scaling quantization of Q/K/V, written with OpenAI Triton.

Input is the QKV GEMM output ``[s, b, g, (hq + 2) * d]``. Two passes are needed because current
scaling requires the global amax before the first cast:

* amax pass: persistent; rotates Q/K (rounded to bf16 like TE's RoPE output) and reduces
  amax(|Q|, |K|, |V|) with one atomic per CTA; the last CTA turns the amax into the scale.
* cast pass: rotates again and writes e4m3/e5m2 into the packed ``[Q | K | V]`` payload, visiting
  tiles in reverse order so the tail of the amax pass is still in L2.

Results are bitwise identical to ``tex.fused_qkv_rope_forward`` followed by current-scaling
quantization.
"""

import triton
import triton.language as tl

from .current_scaling import _finalize_scale

try:
    from triton.language.extra import libdevice
except ImportError:  # older Triton
    from triton.language.extra.cuda import libdevice


@triton.jit
def rope_trig_table_kernel(f_ptr, cos_ptr, sin_ptr, n, BLOCK: tl.constexpr):
    """cos/sin of the RoPE angles with the libdevice functions TE's sincosf resolves to,
    evaluated once per angle instead of once per element of every head."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    f = tl.load(f_ptr + offs, mask=m, other=0.0)
    tl.store(cos_ptr + offs, libdevice.cos(f), mask=m)
    tl.store(sin_ptr + offs, libdevice.sin(f), mask=m)


@triton.jit
def _rot(x, c, xr, s):
    # fma(x, c, xr * s): matches nvcc's contraction in TE's RoPE kernels
    return tl.fma(x, c, xr * s)


@triton.jit
def _rot_fwd(lo, hi, c_lo, s_lo, c_hi, s_hi):
    # TE fused_qkv_rope_block_forward: out[d<h] = x[d]cos[d] + (-x[d+h])sin[d];
    #                                  out[d>=h] = x[d]cos[d] + x[d-h]sin[d]; bf16 output
    y_lo = _rot(lo, c_lo, -hi, s_lo).to(tl.bfloat16).to(tl.float32)
    y_hi = _rot(hi, c_hi, lo, s_hi).to(tl.bfloat16).to(tl.float32)
    return y_lo, y_hi


@triton.jit
def _rope_qkv_token(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    sc,
    tok,
    n_tok,
    B: tl.constexpr,
    G: tl.constexpr,
    HQ: tl.constexpr,
    D: tl.constexpr,
    CAST: tl.constexpr,
):
    """Rotate Q/K of one (s, b) token for all heads. CAST=False returns the token's amax;
    CAST=True writes the scaled FP8 values into the packed [Q | K | V] payload."""
    W: tl.constexpr = (HQ + 2) * D
    HD: tl.constexpr = D // 2
    pos = tok // B
    g = tl.arange(0, G)
    jq = tl.arange(0, HQ)
    dl = tl.arange(0, HD)
    d = tl.arange(0, D)
    tb = pos * D + dl
    c_lo = tl.load(cos_ptr + tb)
    s_lo = tl.load(sin_ptr + tb)
    c_hi = tl.load(cos_ptr + tb + HD)
    s_hi = tl.load(sin_ptr + tb + HD)
    # Q: [g, jq, dl]; K: [g, dl]; V: [g, d]
    row = tok * (G * W) + g * W
    q_src = x_ptr + row[:, None, None] + jq[None, :, None] * D + dl[None, None, :]
    k_src = x_ptr + row[:, None] + HQ * D + dl[None, :]
    v_src = x_ptr + row[:, None] + (HQ + 1) * D + d[None, :]
    q_lo, q_hi = _rot_fwd(
        tl.load(q_src).to(tl.float32),
        tl.load(q_src + HD).to(tl.float32),
        c_lo[None, None, :],
        s_lo[None, None, :],
        c_hi[None, None, :],
        s_hi[None, None, :],
    )
    k_lo, k_hi = _rot_fwd(
        tl.load(k_src).to(tl.float32),
        tl.load(k_src + HD).to(tl.float32),
        c_lo[None, :],
        s_lo[None, :],
        c_hi[None, :],
        s_hi[None, :],
    )
    v = tl.load(v_src).to(tl.float32)
    if CAST:
        nq = n_tok * (G * HQ * D)
        nk = n_tok * (G * D)
        q_dst = (
            out_ptr
            + tok * (G * HQ * D)
            + g[:, None, None] * (HQ * D)
            + jq[None, :, None] * D
            + dl[None, None, :]
        )
        k_dst = out_ptr + nq + tok * (G * D) + g[:, None] * D + dl[None, :]
        v_dst = out_ptr + nq + nk + tok * (G * D) + g[:, None] * D + d[None, :]
        ety = out_ptr.dtype.element_ty
        tl.store(q_dst, (q_lo * sc).to(ety))
        tl.store(q_dst + HD, (q_hi * sc).to(ety))
        tl.store(k_dst, (k_lo * sc).to(ety))
        tl.store(k_dst + HD, (k_hi * sc).to(ety))
        tl.store(v_dst, (v * sc).to(ety))
        return 0.0
    else:
        aq = tl.max(tl.max(tl.max(tl.maximum(tl.abs(q_lo), tl.abs(q_hi)), 2), 1), 0)
        ak = tl.max(tl.max(tl.maximum(tl.abs(k_lo), tl.abs(k_hi)), 1), 0)
        av = tl.max(tl.max(tl.abs(v), 1), 0)
        return tl.maximum(tl.maximum(aq, ak), av)


@triton.jit
def rope_qkv_quantize_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    ws_ptr,
    scale_inv_ptr,
    n_tok,
    eps,
    B: tl.constexpr,
    G: tl.constexpr,
    HQ: tl.constexpr,
    D: tl.constexpr,
    CAST: tl.constexpr,
    FP8MAX: tl.constexpr,
    POW2: tl.constexpr,
    NUM_STAGES: tl.constexpr = 1,
):
    """CAST=False: persistent amax pass (one atomic per CTA, last CTA computes the scale).
    CAST=True: one token per CTA, in the reverse order of the amax pass."""
    n_prog = tl.num_programs(0)
    pid = tl.program_id(0)
    if CAST:
        sc = tl.load(ws_ptr + 2)
        _rope_qkv_token(
            x_ptr, cos_ptr, sin_ptr, out_ptr, sc, n_prog - 1 - pid, n_tok, B, G, HQ, D, True
        )
    else:
        amax = 0.0
        for tok in tl.range(pid, n_tok, n_prog, num_stages=NUM_STAGES):
            a = _rope_qkv_token(
                x_ptr, cos_ptr, sin_ptr, out_ptr, 1.0, tok, n_tok, B, G, HQ, D, False
            )
            amax = tl.maximum(amax, a)
        tl.atomic_max(ws_ptr, amax)
        _finalize_scale(ws_ptr, scale_inv_ptr, n_prog, eps, FP8MAX, POW2)
