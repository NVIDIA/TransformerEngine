# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""DSv4 RoPE kernels copied from NVIDIA/Megatron-LM.

Source: megatron/core/fusions/fused_mla_yarn_rope_apply.py at
c4129afff4867ca89b1fb7b4b911128424cdfbe8. Kernel bodies and autotuning are
retained; TE owns tensor preparation and autograd.
"""

import triton
import triton.language as tl


@triton.jit
def _get_thd_token_idx(cu_seqlens, pid_m, seq_num, cp_rank, cp_size):
    # Cast ``pid_m`` and ``cu_seqlens`` loads to a single shared dtype so
    # the loop-body reassignments don't surface as
    # "initial value is int32 but redefined as int64" in newer Triton
    # versions (which promote ``// Python_int`` to int64).
    pid_m = pid_m.to(tl.int64)
    token_idx = tl.full((), -1, dtype=tl.int64)
    this_seq_len = tl.full((), 0, dtype=tl.int64)
    seq_idx = 0
    last_cum_seqlen = tl.load(cu_seqlens).to(tl.int64) // cp_size
    while seq_idx < seq_num:
        cur_cum_seqlen = tl.load(cu_seqlens + seq_idx + 1).to(tl.int64) // cp_size
        if token_idx == -1 and cur_cum_seqlen > pid_m:
            token_idx = pid_m - last_cum_seqlen
            this_seq_len = cur_cum_seqlen - last_cum_seqlen
        last_cum_seqlen = cur_cum_seqlen
        seq_idx += 1
    # Padding tokens beyond cu_seqlens[-1] (from THD CUDA-graph padding)
    # never match any sequence, leaving token_idx == -1.  Clamp to 0 so
    # the cos/sin table loads stay in-bounds; the wrong RoPE result is
    # harmless because padding positions are excluded by loss_mask.
    if token_idx == -1:
        token_idx = tl.full((), 0, dtype=tl.int64)
    if cp_size > 1:
        if token_idx < this_seq_len // 2:
            token_idx = token_idx + cp_rank * this_seq_len // 2
        else:
            token_idx = (token_idx - this_seq_len // 2) + (
                2 * cp_size - cp_rank - 1
            ) * this_seq_len // 2
    return token_idx


@triton.jit
def _mla_rope_fwd_inplace_kernel(
    Q,
    COS,
    SIN,
    nope_dim,
    emb_dim: tl.constexpr,
    head_num: tl.constexpr,
    batch_size,
    seq_num,
    cu_seqlens_q,
    position_ids,
    stride_x_seq,
    stride_x_nheads,
    stride_cos_seq,
    stride_sin_seq,
    cp_rank,
    cp_size,
    INVERSE: tl.constexpr,
    REMOVE_INTERLEAVING: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """
    Forward pass: apply RoPE inplace to the trailing emb_dim elements.
    Reads from interleaved layout, writes back to interleaved layout.

    Input:
        Q: [seq_len, batch_size, head_num, nope_dim + emb_dim]
            or [total_seq_len, head_num, nope_dim + emb_dim]
        COS/SIN: [max_seq_len, emb_dim]

        batch_size: batch size for sbhd format, not used for thd format
        seq_num: number of sequences for thd format, not used for sbhd format
        cu_seqlens_q: [seq_num + 1] accumulated sequence lengths for thd format
    """
    pid_m = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)

    if position_ids is not None:
        token_idx = tl.load(position_ids + pid_m)
    elif cu_seqlens_q is None:
        token_idx = pid_m // batch_size
    else:
        token_idx = _get_thd_token_idx(cu_seqlens_q, pid_m, seq_num, cp_rank, cp_size)

    cos_left = tl.load(COS + token_idx * stride_cos_seq + tl.arange(0, emb_dim // 2))
    sin_left = tl.load(SIN + token_idx * stride_sin_seq + tl.arange(0, emb_dim // 2))
    cos_right = tl.load(
        COS + token_idx * stride_cos_seq + emb_dim // 2 + tl.arange(0, emb_dim // 2)
    )
    sin_right = tl.load(
        SIN + token_idx * stride_sin_seq + emb_dim // 2 + tl.arange(0, emb_dim // 2)
    )
    if INVERSE:
        sin_left = -sin_left
        sin_right = -sin_right
    cos_left = cos_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    sin_left = sin_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    cos_right = cos_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    sin_right = sin_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)

    Q = Q + pid_m * stride_x_seq + pid_head * BLOCK_H * stride_x_nheads

    x_off = tl.arange(0, BLOCK_H)[:, None] * stride_x_nheads + nope_dim
    mask = (pid_head * BLOCK_H + tl.arange(0, BLOCK_H))[:, None] < head_num
    # x1 = t[..., 0::2], x2 = t[..., 1::2]
    x_1_off = x_off + tl.arange(0, emb_dim // 2)[None, :] * 2
    x_2_off = x_1_off + 1
    x_1 = tl.load(Q + x_1_off, mask=mask)
    x_2 = tl.load(Q + x_2_off, mask=mask)

    x_left = x_1 * cos_left - x_2 * sin_left
    x_right = x_2 * cos_right + x_1 * sin_right

    if REMOVE_INTERLEAVING:
        tl.store(Q + x_1_off, x_left, mask=mask)
        tl.store(Q + x_2_off, x_right, mask=mask)
    else:
        # The interleaved input and split output layouts alias. Finish all loads
        # before any warp stores to the overlapping destination addresses.
        tl.debug_barrier()
        x_left_off = x_off + tl.arange(0, emb_dim // 2)[None, :]
        x_right_off = x_left_off + emb_dim // 2
        tl.store(Q + x_left_off, x_left, mask=mask)
        tl.store(Q + x_right_off, x_right, mask=mask)


_autotuned_mla_rope_fwd_inplace_kernel = triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 1}),
        triton.Config({"BLOCK_H": 2}),
        triton.Config({"BLOCK_H": 4}),
        triton.Config({"BLOCK_H": 8}),
        triton.Config({"BLOCK_H": 16}),
        triton.Config({"BLOCK_H": 32}),
        triton.Config({"BLOCK_H": 64}),
        triton.Config({"BLOCK_H": 128}),
    ],
    key=["emb_dim", "head_num"],
    restore_value=["Q"],
)(_mla_rope_fwd_inplace_kernel)


@triton.jit
def _mla_rope_bwd_inplace_kernel(
    DO,
    COS,
    SIN,
    nope_dim,
    emb_dim: tl.constexpr,
    head_num: tl.constexpr,
    batch_size,
    seq_num,
    cu_seqlens_q,
    position_ids,
    stride_x_seq,
    stride_x_nheads,
    stride_cos_seq,
    stride_sin_seq,
    cp_rank,
    cp_size,
    INVERSE: tl.constexpr,
    REMOVE_INTERLEAVING: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """
    Backward pass: inverse RoPE inplace on the trailing emb_dim elements.
    Reads from interleaved layout, writes to interleaved layout.

    Input:
        DO: [seq_len, batch_size, head_num, nope_dim + emb_dim]
            or [total_seq_len, head_num, nope_dim + emb_dim]
        COS/SIN: [max_seq_len, emb_dim]

        batch_size, seq_num, and cu_seqlens_q are the same as in the forward pass
    """
    pid_m = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)

    if position_ids is not None:
        token_idx = tl.load(position_ids + pid_m)
    elif cu_seqlens_q is None:
        token_idx = pid_m // batch_size
    else:
        token_idx = _get_thd_token_idx(cu_seqlens_q, pid_m, seq_num, cp_rank, cp_size)

    cos_left = tl.load(COS + token_idx * stride_cos_seq + tl.arange(0, emb_dim // 2))
    sin_left = tl.load(SIN + token_idx * stride_sin_seq + tl.arange(0, emb_dim // 2))
    cos_right = tl.load(
        COS + token_idx * stride_cos_seq + emb_dim // 2 + tl.arange(0, emb_dim // 2)
    )
    sin_right = tl.load(
        SIN + token_idx * stride_sin_seq + emb_dim // 2 + tl.arange(0, emb_dim // 2)
    )
    if INVERSE:
        sin_left = -sin_left
        sin_right = -sin_right
    cos_left = cos_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    sin_left = sin_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    cos_right = cos_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    sin_right = sin_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)

    DO = DO + pid_m * stride_x_seq + pid_head * BLOCK_H * stride_x_nheads

    x_off = tl.arange(0, BLOCK_H)[:, None] * stride_x_nheads + nope_dim
    mask = (pid_head * BLOCK_H + tl.arange(0, BLOCK_H))[:, None] < head_num
    if REMOVE_INTERLEAVING:
        x_1_off = x_off + tl.arange(0, emb_dim // 2)[None, :] * 2
        x_2_off = x_1_off + 1
        x_left = tl.load(DO + x_1_off, mask=mask)
        x_right = tl.load(DO + x_2_off, mask=mask)
    else:
        x_left_off = x_off + tl.arange(0, emb_dim // 2)[None, :]
        x_right_off = x_left_off + emb_dim // 2
        x_left = tl.load(DO + x_left_off, mask=mask)
        x_right = tl.load(DO + x_right_off, mask=mask)
        x_1_off = x_off + tl.arange(0, emb_dim // 2)[None, :] * 2
        x_2_off = x_1_off + 1

    x_1 = x_left * cos_left + x_right * sin_right
    x_2 = -x_left * sin_left + x_right * cos_right

    if not REMOVE_INTERLEAVING:
        # The split input and interleaved output layouts alias. Finish all loads
        # before any warp stores to the overlapping destination addresses.
        tl.debug_barrier()
    tl.store(DO + x_1_off, x_1, mask=mask)
    tl.store(DO + x_2_off, x_2, mask=mask)


_autotuned_mla_rope_bwd_inplace_kernel = triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 1}),
        triton.Config({"BLOCK_H": 2}),
        triton.Config({"BLOCK_H": 4}),
        triton.Config({"BLOCK_H": 8}),
        triton.Config({"BLOCK_H": 16}),
        triton.Config({"BLOCK_H": 32}),
        triton.Config({"BLOCK_H": 64}),
        triton.Config({"BLOCK_H": 128}),
    ],
    key=["emb_dim", "head_num"],
    restore_value=["DO"],
)(_mla_rope_bwd_inplace_kernel)
