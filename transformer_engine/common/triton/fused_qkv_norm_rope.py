# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused per-head RMSNorm + RoPE on a combined QKV tensor, written with OpenAI Triton.

The input is the output of a single QKV projection, ``[s, b, h, (nq + 2) * d]`` or
``[b, s, h, (nq + 2) * d]``: for each of the ``h`` query groups, ``nq`` query heads,
one key head and one value head of width ``d`` stored back to back. Every head row
is one block row of the kernels, so a program streams ``ROWS`` consecutive rows and
the slot of a row inside its group selects the norm weight, the rotation table and
the destination. Q and K rows are normalized (RMSNorm with a learnable weight, in
fp32) and rotated in one pass with a single rounding to the output dtype; V rows
are copied.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["s_q", "s_k", "s_table"])
def _qkv_norm_rope_table_kernel(
    q_freqs_ptr,
    k_freqs_ptr,
    table_ptr,
    s_q,
    s_k,
    s_table,
    D2: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D2: tl.constexpr,
):
    """Writes the fp32 table ``[4, s_table, D2]``: cos(q), sin(q), cos(k), sin(k)."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    cols = tl.arange(0, BLOCK_D2)
    col_mask = (cols < D2)[None, :]
    off = rows[:, None] * D2 + cols[None, :]
    q_mask = (rows < s_q)[:, None] & col_mask
    k_mask = (rows < s_k)[:, None] & col_mask
    angle = tl.load(q_freqs_ptr + off, mask=q_mask, other=0.0)
    tl.store(table_ptr + off, tl.cos(angle), mask=q_mask)
    tl.store(table_ptr + s_table * D2 + off, tl.sin(angle), mask=q_mask)
    angle = tl.load(k_freqs_ptr + off, mask=k_mask, other=0.0)
    tl.store(table_ptr + 2 * s_table * D2 + off, tl.cos(angle), mask=k_mask)
    tl.store(table_ptr + 3 * s_table * D2 + off, tl.sin(angle), mask=k_mask)


@triton.jit
def _row_geometry(
    row,
    row_mask,
    s,
    b,
    cp_size,
    cp_rank,
    start_positions_ptr,
    H: tl.constexpr,
    NQ: tl.constexpr,
    SBHD: tl.constexpr,
    CP: tl.constexpr,
    HAS_START: tl.constexpr,
):
    """Maps block rows to (token*H + group, slot, s_id, b_id, rope position)."""
    NH: tl.constexpr = NQ + 2
    token = row // (H * NH)
    group = (row // NH) % H
    slot = row % NH
    if SBHD:
        s_id = token // b
        b_id = token % b
    else:
        b_id = token // s
        s_id = token % s
    if CP:
        half = s // 2
        pos = tl.where(
            s_id < half,
            s_id + cp_rank * half,
            s * cp_size - (cp_rank + 1) * half + s_id - half,
        )
    else:
        pos = s_id
    if HAS_START:
        pos = pos + tl.load(start_positions_ptr + b_id, mask=row_mask, other=0)
    return token * H + group, slot, s_id, b_id, pos


@triton.jit
def _load_weight(
    weight_ptr,
    lo_idx,
    hi_idx,
    p_idx,
    half_mask,
    pass_mask,
    ENABLED: tl.constexpr,
    ZERO_CENTERED: tl.constexpr,
    HAS_PASS: tl.constexpr,
    BLOCK_H2: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """Loads one norm weight as fp32 (lo half, hi half, pass-through dims); ones if disabled."""
    if ENABLED:
        w_lo = tl.load(weight_ptr + lo_idx, mask=half_mask, other=0.0).to(tl.float32)
        w_hi = tl.load(weight_ptr + hi_idx, mask=half_mask, other=0.0).to(tl.float32)
        if HAS_PASS:
            w_p = tl.load(weight_ptr + p_idx, mask=pass_mask, other=0.0).to(tl.float32)
        else:
            w_p = tl.zeros([BLOCK_P], dtype=tl.float32)
        if ZERO_CENTERED:
            w_lo = w_lo + 1.0
            w_hi = w_hi + 1.0
            w_p = w_p + 1.0
    else:
        w_lo = tl.full([BLOCK_H2], 1.0, tl.float32)
        w_hi = tl.full([BLOCK_H2], 1.0, tl.float32)
        w_p = tl.full([BLOCK_P], 1.0, tl.float32)
    return w_lo, w_hi, w_p


@triton.jit
def _select_rows(is_q, is_k, q_val, k_val):
    """Per-row selection of the Q or K per-element vector (ones for V rows)."""
    return tl.where(is_q[:, None], q_val[None, :], tl.where(is_k[:, None], k_val[None, :], 1.0))


@triton.jit
def _normed_rows(is_q, is_k, NORM_Q: tl.constexpr, NORM_K: tl.constexpr):
    """Mask of the rows that are normalized."""
    if NORM_Q:
        if NORM_K:
            return is_q | is_k
        return is_q
    if NORM_K:
        return is_k
    return is_q != is_q


@triton.jit
def _load_rotation(table_ptr, is_q, pos, s_table, lo_idx, hi_idx, rot_mask, D2: tl.constexpr):
    """cos/sin at the lo and hi element of every rotary pair, from the Q or the K table."""
    base = table_ptr + ((tl.where(is_q, 0, 2) * s_table + pos).to(tl.int64))[:, None] * D2
    c_lo = tl.load(base + lo_idx[None, :], mask=rot_mask, other=1.0)
    c_hi = tl.load(base + hi_idx[None, :], mask=rot_mask, other=1.0)
    s_lo = tl.load(base + s_table * D2 + lo_idx[None, :], mask=rot_mask, other=0.0)
    s_hi = tl.load(base + s_table * D2 + hi_idx[None, :], mask=rot_mask, other=0.0)
    return c_lo, c_hi, s_lo, s_hi


@triton.jit(do_not_specialize=["n_rows", "s", "b", "cp_size", "cp_rank", "s_table"])
def _qkv_norm_rope_fwd_kernel(
    qkv_ptr,
    q_weight_ptr,
    k_weight_ptr,
    start_positions_ptr,
    table_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    n_rows,
    s,
    b,
    cp_size,
    cp_rank,
    s_table,
    eps,
    H: tl.constexpr,
    NQ: tl.constexpr,
    D: tl.constexpr,
    D2: tl.constexpr,
    BLOCK_H2: tl.constexpr,
    BLOCK_P: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    SBHD: tl.constexpr,
    CP: tl.constexpr,
    HAS_START: tl.constexpr,
    NORM_Q: tl.constexpr,
    NORM_K: tl.constexpr,
    ZERO_CENTERED: tl.constexpr,
    ROWS: tl.constexpr,
):
    """Forward: q_out[..., h*nq, d], k_out[..., h, d] normalized + rotated, v_out copied."""
    H2: tl.constexpr = D2 // 2
    HAS_PASS: tl.constexpr = D2 < D
    pid = tl.program_id(0)
    row = pid * ROWS + tl.arange(0, ROWS)
    row_mask = row < n_rows
    tg, slot, s_id, b_id, pos = _row_geometry(
        row, row_mask, s, b, cp_size, cp_rank, start_positions_ptr, H, NQ, SBHD, CP, HAS_START
    )
    is_q = slot < NQ
    is_k = slot == NQ
    is_v = slot == NQ + 1

    i = tl.arange(0, BLOCK_H2)
    half_mask = i < H2
    if INTERLEAVED:
        lo_idx = 2 * i
        hi_idx = 2 * i + 1
    else:
        lo_idx = i
        hi_idx = i + H2
    p = tl.arange(0, BLOCK_P)
    pass_mask = p < (D - D2)
    p_idx = D2 + p
    m_half = row_mask[:, None] & half_mask[None, :]
    m_pass = row_mask[:, None] & pass_mask[None, :]

    src = qkv_ptr + row.to(tl.int64)[:, None] * D
    x_lo = tl.load(src + lo_idx[None, :], mask=m_half, other=0.0).to(tl.float32)
    x_hi = tl.load(src + hi_idx[None, :], mask=m_half, other=0.0).to(tl.float32)
    sumsq = tl.sum(x_lo * x_lo, axis=1) + tl.sum(x_hi * x_hi, axis=1)
    if HAS_PASS:
        x_p = tl.load(src + p_idx[None, :], mask=m_pass, other=0.0).to(tl.float32)
        sumsq += tl.sum(x_p * x_p, axis=1)
    else:
        x_p = tl.zeros([ROWS, BLOCK_P], dtype=tl.float32)
    normed = _normed_rows(is_q, is_k, NORM_Q, NORM_K)
    rstd = tl.where(normed, tl.rsqrt(sumsq * (1.0 / D) + eps), 1.0)[:, None]

    wq_lo, wq_hi, wq_p = _load_weight(
        q_weight_ptr,
        lo_idx,
        hi_idx,
        p_idx,
        half_mask,
        pass_mask,
        NORM_Q,
        ZERO_CENTERED,
        HAS_PASS,
        BLOCK_H2,
        BLOCK_P,
    )
    wk_lo, wk_hi, wk_p = _load_weight(
        k_weight_ptr,
        lo_idx,
        hi_idx,
        p_idx,
        half_mask,
        pass_mask,
        NORM_K,
        ZERO_CENTERED,
        HAS_PASS,
        BLOCK_H2,
        BLOCK_P,
    )
    y_lo = x_lo * rstd * _select_rows(is_q, is_k, wq_lo, wk_lo)
    y_hi = x_hi * rstd * _select_rows(is_q, is_k, wq_hi, wk_hi)
    y_p = x_p * rstd * _select_rows(is_q, is_k, wq_p, wk_p)

    rot_mask = (row_mask & (is_q | is_k))[:, None] & half_mask[None, :]
    c_lo, c_hi, s_lo, s_hi = _load_rotation(
        table_ptr, is_q, pos, s_table, lo_idx, hi_idx, rot_mask, D2
    )
    o_lo = y_lo * c_lo - y_hi * s_lo
    o_hi = y_hi * c_hi + y_lo * s_hi

    out_dtype = q_out_ptr.dtype.element_ty
    tg64 = tg.to(tl.int64)
    q_row = q_out_ptr + ((tg64 * NQ + slot) * D)[:, None]
    k_row = k_out_ptr + (tg64 * D)[:, None]
    v_row = v_out_ptr + (tg64 * D)[:, None]
    m_q = (row_mask & is_q)[:, None]
    m_k = (row_mask & is_k)[:, None]
    m_v = (row_mask & is_v)[:, None]
    hm = half_mask[None, :]
    pm = pass_mask[None, :]
    tl.store(q_row + lo_idx[None, :], o_lo.to(out_dtype), mask=m_q & hm)
    tl.store(q_row + hi_idx[None, :], o_hi.to(out_dtype), mask=m_q & hm)
    tl.store(k_row + lo_idx[None, :], o_lo.to(out_dtype), mask=m_k & hm)
    tl.store(k_row + hi_idx[None, :], o_hi.to(out_dtype), mask=m_k & hm)
    tl.store(v_row + lo_idx[None, :], o_lo.to(out_dtype), mask=m_v & hm)
    tl.store(v_row + hi_idx[None, :], o_hi.to(out_dtype), mask=m_v & hm)
    if HAS_PASS:
        tl.store(q_row + p_idx[None, :], y_p.to(out_dtype), mask=m_q & pm)
        tl.store(k_row + p_idx[None, :], y_p.to(out_dtype), mask=m_k & pm)
        tl.store(v_row + p_idx[None, :], y_p.to(out_dtype), mask=m_v & pm)


@triton.jit(
    do_not_specialize=[
        "n_rows",
        "n_blocks",
        "s",
        "b",
        "cp_size",
        "cp_rank",
        "s_table",
        "dq_ss",
        "dq_sb",
        "dq_sh",
        "dk_ss",
        "dk_sb",
        "dk_sh",
        "dv_ss",
        "dv_sb",
        "dv_sh",
    ]
)
def _qkv_norm_rope_bwd_kernel(
    grad_q_ptr,
    grad_k_ptr,
    grad_v_ptr,
    qkv_ptr,
    q_weight_ptr,
    k_weight_ptr,
    start_positions_ptr,
    table_ptr,
    grad_qkv_ptr,
    partial_ptr,
    n_rows,
    n_blocks,
    s,
    b,
    cp_size,
    cp_rank,
    s_table,
    eps,
    dq_ss,
    dq_sb,
    dq_sh,
    dk_ss,
    dk_sb,
    dk_sh,
    dv_ss,
    dv_sb,
    dv_sh,
    H: tl.constexpr,
    NQ: tl.constexpr,
    D: tl.constexpr,
    D2: tl.constexpr,
    BLOCK_H2: tl.constexpr,
    BLOCK_P: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    SBHD: tl.constexpr,
    CP: tl.constexpr,
    HAS_START: tl.constexpr,
    NORM_Q: tl.constexpr,
    NORM_K: tl.constexpr,
    ZERO_CENTERED: tl.constexpr,
    ROWS: tl.constexpr,
):
    """Backward: grad_qkv in the input layout; per-program weight-gradient partials
    ``partial[pid, 0:D]`` (Q) and ``partial[pid, D:2D]`` (K). Persistent grid."""
    H2: tl.constexpr = D2 // 2
    HAS_PASS: tl.constexpr = D2 < D
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)

    i = tl.arange(0, BLOCK_H2)
    half_mask = i < H2
    if INTERLEAVED:
        lo_idx = 2 * i
        hi_idx = 2 * i + 1
    else:
        lo_idx = i
        hi_idx = i + H2
    p = tl.arange(0, BLOCK_P)
    pass_mask = p < (D - D2)
    p_idx = D2 + p
    hm = half_mask[None, :]
    pm = pass_mask[None, :]

    wq_lo, wq_hi, wq_p = _load_weight(
        q_weight_ptr,
        lo_idx,
        hi_idx,
        p_idx,
        half_mask,
        pass_mask,
        NORM_Q,
        ZERO_CENTERED,
        HAS_PASS,
        BLOCK_H2,
        BLOCK_P,
    )
    wk_lo, wk_hi, wk_p = _load_weight(
        k_weight_ptr,
        lo_idx,
        hi_idx,
        p_idx,
        half_mask,
        pass_mask,
        NORM_K,
        ZERO_CENTERED,
        HAS_PASS,
        BLOCK_H2,
        BLOCK_P,
    )

    acc_q_lo = tl.zeros([ROWS, BLOCK_H2], dtype=tl.float32)
    acc_q_hi = tl.zeros([ROWS, BLOCK_H2], dtype=tl.float32)
    acc_q_p = tl.zeros([ROWS, BLOCK_P], dtype=tl.float32)
    acc_k_lo = tl.zeros([ROWS, BLOCK_H2], dtype=tl.float32)
    acc_k_hi = tl.zeros([ROWS, BLOCK_H2], dtype=tl.float32)
    acc_k_p = tl.zeros([ROWS, BLOCK_P], dtype=tl.float32)

    for blk in range(pid, n_blocks, nprog):
        row = blk * ROWS + tl.arange(0, ROWS)
        row_mask = row < n_rows
        tg, slot, s_id, b_id, pos = _row_geometry(
            row, row_mask, s, b, cp_size, cp_rank, start_positions_ptr, H, NQ, SBHD, CP, HAS_START
        )
        is_q = slot < NQ
        is_k = slot == NQ
        is_v = slot == NQ + 1
        m_half = row_mask[:, None] & hm
        m_pass = row_mask[:, None] & pm
        m_q = (row_mask & is_q)[:, None]
        m_k = (row_mask & is_k)[:, None]
        m_v = (row_mask & is_v)[:, None]

        src = qkv_ptr + row.to(tl.int64)[:, None] * D
        x_lo = tl.load(src + lo_idx[None, :], mask=m_half, other=0.0).to(tl.float32)
        x_hi = tl.load(src + hi_idx[None, :], mask=m_half, other=0.0).to(tl.float32)
        sumsq = tl.sum(x_lo * x_lo, axis=1) + tl.sum(x_hi * x_hi, axis=1)
        if HAS_PASS:
            x_p = tl.load(src + p_idx[None, :], mask=m_pass, other=0.0).to(tl.float32)
            sumsq += tl.sum(x_p * x_p, axis=1)
        else:
            x_p = tl.zeros([ROWS, BLOCK_P], dtype=tl.float32)
        normed = _normed_rows(is_q, is_k, NORM_Q, NORM_K)
        rstd = tl.where(normed, tl.rsqrt(sumsq * (1.0 / D) + eps), 1.0)[:, None]
        xh_lo = x_lo * rstd
        xh_hi = x_hi * rstd
        xh_p = x_p * rstd

        s64 = s_id.to(tl.int64)
        b64 = b_id.to(tl.int64)
        g64 = (tg - (tg // H) * H).to(tl.int64)
        dq_row = grad_q_ptr + (s64 * dq_ss + b64 * dq_sb + (g64 * NQ + slot) * dq_sh)[:, None]
        dk_row = grad_k_ptr + (s64 * dk_ss + b64 * dk_sb + g64 * dk_sh)[:, None]
        dv_row = grad_v_ptr + (s64 * dv_ss + b64 * dv_sb + g64 * dv_sh)[:, None]
        d_lo = (
            tl.load(dq_row + lo_idx[None, :], mask=m_q & hm, other=0.0).to(tl.float32)
            + tl.load(dk_row + lo_idx[None, :], mask=m_k & hm, other=0.0).to(tl.float32)
            + tl.load(dv_row + lo_idx[None, :], mask=m_v & hm, other=0.0).to(tl.float32)
        )
        d_hi = (
            tl.load(dq_row + hi_idx[None, :], mask=m_q & hm, other=0.0).to(tl.float32)
            + tl.load(dk_row + hi_idx[None, :], mask=m_k & hm, other=0.0).to(tl.float32)
            + tl.load(dv_row + hi_idx[None, :], mask=m_v & hm, other=0.0).to(tl.float32)
        )
        if HAS_PASS:
            d_p = (
                tl.load(dq_row + p_idx[None, :], mask=m_q & pm, other=0.0).to(tl.float32)
                + tl.load(dk_row + p_idx[None, :], mask=m_k & pm, other=0.0).to(tl.float32)
                + tl.load(dv_row + p_idx[None, :], mask=m_v & pm, other=0.0).to(tl.float32)
            )
        else:
            d_p = tl.zeros([ROWS, BLOCK_P], dtype=tl.float32)

        rot_mask = (row_mask & (is_q | is_k))[:, None] & hm
        c_lo, c_hi, s_lo, s_hi = _load_rotation(
            table_ptr, is_q, pos, s_table, lo_idx, hi_idx, rot_mask, D2
        )
        dy_lo = d_lo * c_lo + d_hi * s_hi
        dy_hi = d_hi * c_hi - d_lo * s_lo
        dy_p = d_p

        if NORM_Q:
            acc_q_lo += tl.where(m_q, dy_lo * xh_lo, 0.0)
            acc_q_hi += tl.where(m_q, dy_hi * xh_hi, 0.0)
            acc_q_p += tl.where(m_q, dy_p * xh_p, 0.0)
        if NORM_K:
            acc_k_lo += tl.where(m_k, dy_lo * xh_lo, 0.0)
            acc_k_hi += tl.where(m_k, dy_hi * xh_hi, 0.0)
            acc_k_p += tl.where(m_k, dy_p * xh_p, 0.0)

        dh_lo = dy_lo * _select_rows(is_q, is_k, wq_lo, wk_lo)
        dh_hi = dy_hi * _select_rows(is_q, is_k, wq_hi, wk_hi)
        dh_p = dy_p * _select_rows(is_q, is_k, wq_p, wk_p)
        mean = (
            tl.sum(dh_lo * xh_lo, axis=1)
            + tl.sum(dh_hi * xh_hi, axis=1)
            + tl.sum(dh_p * xh_p, axis=1)
        ) * (1.0 / D)
        mean = mean[:, None]
        nm = normed[:, None]
        dx_lo = tl.where(nm, rstd * (dh_lo - mean * xh_lo), dy_lo)
        dx_hi = tl.where(nm, rstd * (dh_hi - mean * xh_hi), dy_hi)
        dx_p = tl.where(nm, rstd * (dh_p - mean * xh_p), dy_p)

        in_dtype = grad_qkv_ptr.dtype.element_ty
        dst = grad_qkv_ptr + row.to(tl.int64)[:, None] * D
        tl.store(dst + lo_idx[None, :], dx_lo.to(in_dtype), mask=m_half)
        tl.store(dst + hi_idx[None, :], dx_hi.to(in_dtype), mask=m_half)
        if HAS_PASS:
            tl.store(dst + p_idx[None, :], dx_p.to(in_dtype), mask=m_pass)

    part = partial_ptr + pid.to(tl.int64) * (2 * D)
    tl.store(part + lo_idx, tl.sum(acc_q_lo, axis=0), mask=half_mask)
    tl.store(part + hi_idx, tl.sum(acc_q_hi, axis=0), mask=half_mask)
    tl.store(part + D + lo_idx, tl.sum(acc_k_lo, axis=0), mask=half_mask)
    tl.store(part + D + hi_idx, tl.sum(acc_k_hi, axis=0), mask=half_mask)
    if HAS_PASS:
        tl.store(part + p_idx, tl.sum(acc_q_p, axis=0), mask=pass_mask)
        tl.store(part + D + p_idx, tl.sum(acc_k_p, axis=0), mask=pass_mask)
