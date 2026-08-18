# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused MLA RoPE kernels (DeepSeekV3-style decoupled RoPE/NoPE).

Triton forward/backward kernels adapted from Megatron-LM
``megatron/core/fusions/fused_mla_yarn_rope_apply.py``. The query kernel
rotates the trailing ``head_dim_rope`` slice in place (no concat); the KV
kernel builds the final key (nope | broadcast-rotated shared rope head) and
value tensors in a single pass. Falls back to pure PyTorch when Triton is
unavailable or for the ``bshd`` layout (the Triton path is ``sbhd``-only).

Rotation convention: the rope slice is read interleaved (as stored in
HF/Megatron DeepSeekV3 checkpoints) and written in NeoX half-split layout,
matching the Megatron fused kernel semantics.
"""

from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

__all__ = ["build_rope_tables", "apply_mla_rope_q", "apply_mla_rope_kv"]


def build_rope_tables(
    seq_len: int,
    emb_dim: int,
    base: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape ``[seq_len, emb_dim]`` (fp32, NeoX duplicated halves)."""
    inv_freq = 1.0 / (
        base ** (torch.arange(0, emb_dim, 2, dtype=torch.float32, device=device) / emb_dim)
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    freqs = torch.cat([freqs, freqs], dim=-1)
    return torch.cos(freqs).contiguous(), torch.sin(freqs).contiguous()


if HAVE_TRITON:

    # Not used for non-packed batches; kept for THD compatibility.
    @triton.jit
    def _get_thd_token_idx(cu_seqlens, pid_m, seq_num, cp_rank, cp_size):
        token_idx = -1
        this_seq_len = 0
        seq_idx = 0
        last_cum_seqlen = tl.load(cu_seqlens) // cp_size
        while seq_idx < seq_num:
            cur_cum_seqlen = tl.load(cu_seqlens + seq_idx + 1) // cp_size
            if token_idx == -1 and cur_cum_seqlen > pid_m:
                token_idx = pid_m - last_cum_seqlen
                this_seq_len = cur_cum_seqlen - last_cum_seqlen
            last_cum_seqlen = cur_cum_seqlen
            seq_idx += 1
        if cp_size > 1:
            if token_idx < this_seq_len // 2:
                token_idx = token_idx + cp_rank * this_seq_len // 2
            else:
                token_idx = (token_idx - this_seq_len // 2) + (
                    2 * cp_size - cp_rank - 1
                ) * this_seq_len // 2
        return token_idx

    _AUTOTUNE_CONFIGS = [triton.Config({"BLOCK_H": h}) for h in (1, 2, 4, 8, 16, 32, 64, 128)]

    @triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["emb_dim", "head_num"], restore_value=["Q"])
    @triton.jit
    def rotary_fwd_q_kernel(
        Q,
        COS,
        SIN,
        qk_head_dim,
        emb_dim: tl.constexpr,
        head_num: tl.constexpr,
        batch_size,
        seq_num,
        cu_seqlens_q,
        stride_x_seq,
        stride_x_nheads,
        cp_rank,
        cp_size,
        BLOCK_H: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_head = tl.program_id(axis=1)
        if cu_seqlens_q is None:
            token_idx = pid_m // batch_size
        else:
            token_idx = _get_thd_token_idx(cu_seqlens_q, pid_m, seq_num, cp_rank, cp_size)
        cos_left = tl.load(COS + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
        sin_left = tl.load(SIN + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
        cos_right = tl.load(COS + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
        sin_right = tl.load(SIN + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
        cos_left = cos_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
        sin_left = sin_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
        cos_right = cos_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
        sin_right = sin_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
        head_offsets = pid_head * BLOCK_H + tl.arange(0, BLOCK_H)
        Q = Q + pid_m * stride_x_seq
        x_off = head_offsets[:, None] * stride_x_nheads + qk_head_dim
        mask = head_offsets[:, None] < head_num
        x_1_off = x_off + tl.arange(0, emb_dim // 2)[None, :] * 2
        x_2_off = x_1_off + 1
        x_1 = tl.load(Q + x_1_off, mask=mask)
        x_2 = tl.load(Q + x_2_off, mask=mask)
        x_left = x_1 * cos_left - x_2 * sin_left
        x_right = x_2 * cos_right + x_1 * sin_right
        x_left_off = x_off + tl.arange(0, emb_dim // 2)[None, :]
        x_right_off = x_left_off + emb_dim // 2
        tl.store(Q + x_left_off, x_left, mask=mask)
        tl.store(Q + x_right_off, x_right, mask=mask)

    @triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["emb_dim", "head_num"], restore_value=["DO"])
    @triton.jit
    def rotary_bwd_q_kernel(
        DO,
        COS,
        SIN,
        qk_head_dim,
        emb_dim: tl.constexpr,
        head_num: tl.constexpr,
        batch_size,
        seq_num,
        cu_seqlens_q,
        stride_x_seq,
        stride_x_nheads,
        cp_rank,
        cp_size,
        BLOCK_H: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_head = tl.program_id(axis=1)
        if cu_seqlens_q is None:
            token_idx = pid_m // batch_size
        else:
            token_idx = _get_thd_token_idx(cu_seqlens_q, pid_m, seq_num, cp_rank, cp_size)
        cos_left = tl.load(COS + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
        sin_left = tl.load(SIN + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
        cos_right = tl.load(COS + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
        sin_right = tl.load(SIN + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
        cos_left = cos_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
        sin_left = sin_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
        cos_right = cos_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
        sin_right = sin_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
        head_offsets = pid_head * BLOCK_H + tl.arange(0, BLOCK_H)
        DO = DO + pid_m * stride_x_seq
        x_off = head_offsets[:, None] * stride_x_nheads + qk_head_dim
        mask = head_offsets[:, None] < head_num
        x_left_off = x_off + tl.arange(0, emb_dim // 2)[None, :]
        x_right_off = x_left_off + emb_dim // 2
        x_left = tl.load(DO + x_left_off, mask=mask)
        x_right = tl.load(DO + x_right_off, mask=mask)
        x_1 = x_left * cos_left + x_right * sin_right
        x_2 = -x_left * sin_left + x_right * cos_right
        x_1_off = x_off + tl.arange(0, emb_dim // 2)[None, :] * 2
        x_2_off = x_1_off + 1
        tl.store(DO + x_1_off, x_1, mask=mask)
        tl.store(DO + x_2_off, x_2, mask=mask)

    @triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["emb_dim", "k_dim", "v_dim", "head_num"])
    @triton.jit
    def rotary_fwd_kv_kernel(
        KV,
        K_POS_EMB,
        O_KEY,
        O_VALUE,
        COS,
        SIN,
        emb_dim: tl.constexpr,
        k_dim: tl.constexpr,
        v_dim: tl.constexpr,
        head_num: tl.constexpr,
        batch_size,
        seq_num,
        cu_seqlens_kv,
        stride_kv_seq,
        stride_kv_nheads,
        stride_emb_seq,
        stride_k_seq,
        stride_k_nheads,
        stride_v_seq,
        stride_v_nheads,
        cp_rank,
        cp_size,
        BLOCK_H: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_head = tl.program_id(axis=1)
        if cu_seqlens_kv is None:
            token_idx = pid_m // batch_size
        else:
            token_idx = _get_thd_token_idx(cu_seqlens_kv, pid_m, seq_num, cp_rank, cp_size)
        cos_left = tl.load(COS + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
        sin_left = tl.load(SIN + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
        cos_right = tl.load(COS + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
        sin_right = tl.load(SIN + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
        head_offsets = pid_head * BLOCK_H + tl.arange(0, BLOCK_H)
        KV_ptr = KV + pid_m * stride_kv_seq
        kv_off = head_offsets[:, None] * stride_kv_nheads
        mask = head_offsets[:, None] < head_num
        k_in_off = kv_off + tl.arange(0, k_dim)[None, :]
        v_in_off = kv_off + k_dim + tl.arange(0, v_dim)[None, :]
        k = tl.load(KV_ptr + k_in_off, mask=mask)
        v = tl.load(KV_ptr + v_in_off, mask=mask)
        K_ptr = O_KEY + pid_m * stride_k_seq + pid_head * BLOCK_H * stride_k_nheads
        V_ptr = O_VALUE + pid_m * stride_v_seq + pid_head * BLOCK_H * stride_v_nheads
        k_out_off = tl.arange(0, BLOCK_H)[:, None] * stride_k_nheads + tl.arange(0, k_dim)[None, :]
        v_out_off = tl.arange(0, BLOCK_H)[:, None] * stride_v_nheads + tl.arange(0, v_dim)[None, :]
        tl.store(K_ptr + k_out_off, k, mask=mask)
        tl.store(V_ptr + v_out_off, v, mask=mask)
        EMB = K_POS_EMB + pid_m * stride_emb_seq
        x_1 = tl.load(EMB + tl.arange(0, emb_dim // 2) * 2)
        x_2 = tl.load(EMB + tl.arange(0, emb_dim // 2) * 2 + 1)
        x_left = x_1 * cos_left - x_2 * sin_left
        x_right = x_2 * cos_right + x_1 * sin_right
        x_left = x_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
        x_right = x_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
        x_left_off = (
            tl.arange(0, BLOCK_H)[:, None] * stride_k_nheads
            + k_dim
            + tl.arange(0, emb_dim // 2)[None, :]
        )
        x_right_off = x_left_off + emb_dim // 2
        tl.store(K_ptr + x_left_off, x_left, mask=mask)
        tl.store(K_ptr + x_right_off, x_right, mask=mask)

    @triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["emb_dim", "k_dim", "v_dim", "head_num"])
    @triton.jit
    def rotary_bwd_kv_kernel(
        dK,
        dV,
        dKV,
        dEMB,
        COS,
        SIN,
        emb_dim: tl.constexpr,
        k_dim: tl.constexpr,
        v_dim: tl.constexpr,
        head_num: tl.constexpr,
        batch_size,
        seq_num,
        cu_seqlens_kv,
        stride_dk_seq,
        stride_dk_nheads,
        stride_dv_seq,
        stride_dv_nheads,
        stride_dkv_seq,
        stride_dkv_nheads,
        stride_demb_seq,
        cp_rank,
        cp_size,
        BLOCK_H: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_head = tl.program_id(axis=1)
        if cu_seqlens_kv is None:
            token_idx = pid_m // batch_size
        else:
            token_idx = _get_thd_token_idx(cu_seqlens_kv, pid_m, seq_num, cp_rank, cp_size)
        head_offsets = pid_head * BLOCK_H + tl.arange(0, BLOCK_H)
        dKV_ptr = dKV + pid_m * stride_dkv_seq
        dkv_off = head_offsets[:, None] * stride_dkv_nheads
        mask = head_offsets[:, None] < head_num
        dk_out_off = dkv_off + tl.arange(0, k_dim)[None, :]
        dv_out_off = dkv_off + k_dim + tl.arange(0, v_dim)[None, :]
        dK_ptr = dK + pid_m * stride_dk_seq + pid_head * BLOCK_H * stride_dk_nheads
        dV_ptr = dV + pid_m * stride_dv_seq + pid_head * BLOCK_H * stride_dv_nheads
        dk_in_off = tl.arange(0, BLOCK_H)[:, None] * stride_dk_nheads + tl.arange(0, k_dim)[None, :]
        dv_in_off = tl.arange(0, BLOCK_H)[:, None] * stride_dv_nheads + tl.arange(0, v_dim)[None, :]
        dk = tl.load(dK_ptr + dk_in_off, mask=mask)
        dv = tl.load(dV_ptr + dv_in_off, mask=mask)
        tl.store(dKV_ptr + dk_out_off, dk, mask=mask)
        tl.store(dKV_ptr + dv_out_off, dv, mask=mask)
        if pid_head == 0:
            x_left_accum = tl.zeros((BLOCK_H, emb_dim // 2), dtype=tl.float32)
            x_right_accum = tl.zeros((BLOCK_H, emb_dim // 2), dtype=tl.float32)
            for i in tl.static_range(triton.cdiv(head_num, BLOCK_H)):
                head_offsets_i = i * BLOCK_H + tl.arange(0, BLOCK_H)
                dK_ptr_i = dK + pid_m * stride_dk_seq
                x_off = head_offsets_i[:, None] * stride_dk_nheads + k_dim
                mask_i = head_offsets_i[:, None] < head_num
                x_left_off = x_off + tl.arange(0, emb_dim // 2)[None, :]
                x_right_off = x_left_off + emb_dim // 2
                x_left_accum += tl.load(dK_ptr_i + x_left_off, mask=mask_i)
                x_right_accum += tl.load(dK_ptr_i + x_right_off, mask=mask_i)
            x_left_accum = tl.sum(x_left_accum, axis=0)
            x_right_accum = tl.sum(x_right_accum, axis=0)
            x_left_accum = x_left_accum.to(dEMB.dtype.element_ty)
            x_right_accum = x_right_accum.to(dEMB.dtype.element_ty)
            cos_left = tl.load(COS + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
            sin_left = tl.load(SIN + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
            cos_right = tl.load(
                COS + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2)
            )
            sin_right = tl.load(
                SIN + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2)
            )
            x_1 = x_left_accum * cos_left + x_right_accum * sin_right
            x_2 = -x_left_accum * sin_left + x_right_accum * cos_right
            dEMB_ptr = dEMB + pid_m * stride_demb_seq
            tl.store(dEMB_ptr + tl.arange(0, emb_dim // 2) * 2, x_1)
            tl.store(dEMB_ptr + tl.arange(0, emb_dim // 2) * 2 + 1, x_2)

    def _token_stride(tensor: torch.Tensor) -> int:
        return tensor.stride(1) if tensor.dim() == 4 else tensor.stride(0)

    class _MLARoPEQTriton(torch.autograd.Function):
        """In-place RoPE on the trailing rope slice of q [s, b, h, nope+rope]."""

        @staticmethod
        def forward(ctx, q, cos, sin, head_dim_nope, head_dim_rope):
            if not q.is_contiguous():
                q = q.contiguous()
            s, b, nheads, _ = q.shape
            grid = lambda META: (s * b, triton.cdiv(nheads, META["BLOCK_H"]))
            rotary_fwd_q_kernel[grid](
                q,
                cos,
                sin,
                head_dim_nope,
                head_dim_rope,
                nheads,
                b,
                None,
                None,
                _token_stride(q),
                q.stride(2),
                0,
                1,
            )
            ctx.save_for_backward(cos, sin)
            ctx.dims = (s, b, nheads, head_dim_nope, head_dim_rope)
            return q

        @staticmethod
        def backward(ctx, dq):
            cos, sin = ctx.saved_tensors
            # attention backward may hand over a strided grad; the kernel
            # assumes a contiguous [s, b, h, d] layout
            dq = dq.contiguous()
            s, b, nheads, head_dim_nope, head_dim_rope = ctx.dims
            grid = lambda META: (s * b, triton.cdiv(nheads, META["BLOCK_H"]))
            rotary_bwd_q_kernel[grid](
                dq,
                cos,
                sin,
                head_dim_nope,
                head_dim_rope,
                nheads,
                b,
                None,
                None,
                _token_stride(dq),
                dq.stride(2),
                0,
                1,
            )
            return dq, None, None, None, None

    class _MLARoPEKVTriton(torch.autograd.Function):
        """kv [s, b, h, nope+v] + shared rope head [s, b, 1, rope] -> (k, v)."""

        @staticmethod
        def forward(ctx, kv, k_pos_emb, cos, sin, head_dim_nope, head_dim_rope, head_dim_v):
            if not kv.is_contiguous():
                kv = kv.contiguous()
            s, b, nheads, _ = kv.shape
            o_key = kv.new_empty(s, b, nheads, head_dim_nope + head_dim_rope)
            o_value = kv.new_empty(s, b, nheads, head_dim_v)
            grid = lambda META: (s * b, triton.cdiv(nheads, META["BLOCK_H"]))
            rotary_fwd_kv_kernel[grid](
                kv,
                k_pos_emb,
                o_key,
                o_value,
                cos,
                sin,
                head_dim_rope,
                head_dim_nope,
                head_dim_v,
                nheads,
                b,
                None,
                None,
                _token_stride(kv),
                kv.stride(2),
                _token_stride(k_pos_emb),
                _token_stride(o_key),
                o_key.stride(2),
                _token_stride(o_value),
                o_value.stride(2),
                0,
                1,
            )
            ctx.save_for_backward(cos, sin)
            ctx.dims = (s, b, nheads, head_dim_nope, head_dim_rope, head_dim_v)
            return o_key, o_value

        @staticmethod
        def backward(ctx, dk_out, dv_out):
            cos, sin = ctx.saved_tensors
            s, b, nheads, ndp, ndr, ndv = ctx.dims
            dk_out = dk_out.contiguous()
            dv_out = dv_out.contiguous()
            d_kv = dk_out.new_empty(s, b, nheads, ndp + ndv)
            d_emb = dk_out.new_empty(s, b, 1, ndr)
            grid = lambda META: (s * b, triton.cdiv(nheads, META["BLOCK_H"]))
            rotary_bwd_kv_kernel[grid](
                dk_out,
                dv_out,
                d_kv,
                d_emb,
                cos,
                sin,
                ndr,
                ndp,
                ndv,
                nheads,
                b,
                None,
                None,
                _token_stride(dk_out),
                dk_out.stride(2),
                _token_stride(dv_out),
                dv_out.stride(2),
                _token_stride(d_kv),
                d_kv.stride(2),
                _token_stride(d_emb),
                0,
                1,
            )
            return d_kv, d_emb, None, None, None, None, None


def _rotate_interleaved_to_neox(x, cos_table, sin_table, seq_dim):
    shape = [1, 1, 1, cos_table.shape[-1]]
    shape[seq_dim] = cos_table.shape[0]
    cos_ = cos_table.view(shape).to(x.dtype)
    sin_ = sin_table.view(shape).to(x.dtype)
    half = x.shape[-1] // 2
    x_1 = x[..., 0::2]
    x_2 = x[..., 1::2]
    x_left = x_1 * cos_[..., :half] - x_2 * sin_[..., :half]
    x_right = x_2 * cos_[..., half:] + x_1 * sin_[..., half:]
    return torch.cat((x_left, x_right), dim=-1)


def apply_mla_rope_q(
    q: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    head_dim_nope: int,
    head_dim_rope: int,
    tensor_format: str = "sbhd",
) -> torch.Tensor:
    """RoPE on the trailing ``head_dim_rope`` slice of q; in place on the Triton path."""
    if HAVE_TRITON and tensor_format == "sbhd":
        return _MLARoPEQTriton.apply(q, cos_table, sin_table, head_dim_nope, head_dim_rope)
    seq_dim = 0 if tensor_format == "sbhd" else 1
    q_rope = _rotate_interleaved_to_neox(q[..., head_dim_nope:], cos_table, sin_table, seq_dim)
    return torch.cat((q[..., :head_dim_nope], q_rope), dim=-1)


def apply_mla_rope_kv(
    kv: torch.Tensor,
    k_pos_emb: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    head_dim_nope: int,
    head_dim_rope: int,
    head_dim_v: int,
    tensor_format: str = "sbhd",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build (k, v) from kv ``[.., h, nope+v]`` and the shared rope head ``[.., 1, rope]``."""
    if HAVE_TRITON and tensor_format == "sbhd":
        return _MLARoPEKVTriton.apply(
            kv, k_pos_emb, cos_table, sin_table, head_dim_nope, head_dim_rope, head_dim_v
        )
    seq_dim = 0 if tensor_format == "sbhd" else 1
    k_nope = kv[..., :head_dim_nope]
    v = kv[..., head_dim_nope : head_dim_nope + head_dim_v]
    k_rope = _rotate_interleaved_to_neox(k_pos_emb, cos_table, sin_table, seq_dim)
    k_rope = k_rope.expand(*k_nope.shape[:-1], -1)
    return torch.cat((k_nope, k_rope), dim=-1), v.contiguous()
