# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MLA RoPE for DSv3 671B - Triton forward and backward kernels.

Source: Megatron-LM megatron/core/fusions/fused_mla_yarn_rope_apply.py
Falls back to pure PyTorch when Triton is unavailable.

Note: DSv3 uses YaRN-scaled RoPE for long-context extrapolation. This test
intentionally uses plain RoPE (base=10000) because it only validates MXFP8
attention path wiring, tensor shapes, forward/backward flow, and relative BF16
vs MXFP8 behavior. Both reference and MXFP8 paths use the same RoPE tables.
"""

import torch

try:
    import triton
    from transformer_engine.common.triton.dsa_rope import (
        rotary_fwd_q_kernel,
        rotary_bwd_q_kernel,
        rotary_fwd_kv_kernel,
        rotary_bwd_kv_kernel,
    )

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

HEAD_DIM_ROPE = 64
HEAD_DIM_NOPE = 128
HEAD_DIM_V = 128
ROTARY_BASE = 10000


def build_rope_tables(
    seq_len: int,
    emb_dim: int = HEAD_DIM_ROPE,
    base: int = ROTARY_BASE,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        base ** (torch.arange(0, emb_dim, 2, dtype=torch.float32, device=device) / emb_dim)
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    freqs = torch.cat([freqs, freqs], dim=-1)
    return torch.cos(freqs).contiguous(), torch.sin(freqs).contiguous()


if HAVE_TRITON:

    def _flattened_token_stride(tensor: torch.Tensor) -> int:
        if tensor.dim() == 4:
            return tensor.stride(1)
        return tensor.stride(0)

    class _MLARoPEQTriton(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, cos, sin, head_dim_nope, head_dim_rope):
            s, b, nheads, _ = q.shape
            total = s * b

            grid_q = lambda META: (total, triton.cdiv(nheads, META["BLOCK_H"]))
            rotary_fwd_q_kernel[grid_q](
                q,
                cos,
                sin,
                head_dim_nope,
                head_dim_rope,
                nheads,
                b,
                None,
                None,
                None,
                _flattened_token_stride(q),
                q.stride(2),
                cos.stride(0),
                sin.stride(0),
                0,
                1,
                INVERSE=False,
                REMOVE_INTERLEAVING=False,
            )

            ctx.save_for_backward(cos, sin)
            ctx.head_dim_nope = head_dim_nope
            ctx.head_dim_rope = head_dim_rope
            ctx.nheads = nheads
            ctx.s = s
            ctx.b = b
            return q

        @staticmethod
        def backward(ctx, dq):
            cos, sin = ctx.saved_tensors
            s, b, nheads = ctx.s, ctx.b, ctx.nheads
            total = s * b

            grid_q = lambda META: (total, triton.cdiv(nheads, META["BLOCK_H"]))
            rotary_bwd_q_kernel[grid_q](
                dq,
                cos,
                sin,
                ctx.head_dim_nope,
                ctx.head_dim_rope,
                nheads,
                b,
                None,
                None,
                None,
                _flattened_token_stride(dq),
                dq.stride(2),
                cos.stride(0),
                sin.stride(0),
                0,
                1,
                INVERSE=False,
                REMOVE_INTERLEAVING=False,
            )
            return dq, None, None, None, None

    class _MLARoPEKVTriton(torch.autograd.Function):
        @staticmethod
        def forward(ctx, kv, k_pos_emb, cos, sin, head_dim_nope, head_dim_rope, head_dim_v):
            s, b, nheads, _ = kv.shape
            total = s * b

            o_key = kv.new_empty(s, b, nheads, head_dim_nope + head_dim_rope)
            o_value = kv.new_empty(s, b, nheads, head_dim_v)
            grid_kv = lambda META: (total, triton.cdiv(nheads, META["BLOCK_H"]))
            rotary_fwd_kv_kernel[grid_kv](
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
                _flattened_token_stride(kv),
                kv.stride(2),
                _flattened_token_stride(k_pos_emb),
                _flattened_token_stride(o_key),
                o_key.stride(2),
                _flattened_token_stride(o_value),
                o_value.stride(2),
                0,
                1,
            )

            ctx.save_for_backward(cos, sin)
            ctx.head_dim_nope = head_dim_nope
            ctx.head_dim_rope = head_dim_rope
            ctx.head_dim_v = head_dim_v
            ctx.nheads = nheads
            ctx.s = s
            ctx.b = b
            return o_key, o_value

        @staticmethod
        def backward(ctx, dk_out, dv_out):
            cos, sin = ctx.saved_tensors
            s, b, nheads = ctx.s, ctx.b, ctx.nheads
            ndp, ndr, ndv = ctx.head_dim_nope, ctx.head_dim_rope, ctx.head_dim_v
            total = s * b

            d_kv = dk_out.new_empty(s, b, nheads, ndp + ndv)
            d_emb = dk_out.new_empty(s, b, 1, ndr)
            grid_kv = lambda META: (total, triton.cdiv(nheads, META["BLOCK_H"]))
            rotary_bwd_kv_kernel[grid_kv](
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
                _flattened_token_stride(dk_out),
                dk_out.stride(2),
                _flattened_token_stride(dv_out),
                dv_out.stride(2),
                _flattened_token_stride(d_kv),
                d_kv.stride(2),
                _flattened_token_stride(d_emb),
                0,
                1,
            )
            return d_kv, d_emb, None, None, None, None, None


def _apply_mla_rope_q_with_tables(
    q: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    head_dim_nope: int = HEAD_DIM_NOPE,
    head_dim_rope: int = HEAD_DIM_ROPE,
) -> torch.Tensor:
    if HAVE_TRITON:
        return _MLARoPEQTriton.apply(
            q,
            cos_table,
            sin_table,
            head_dim_nope,
            head_dim_rope,
        )
    return _apply_pytorch_q(q, cos_table, sin_table, head_dim_nope, head_dim_rope)


def _apply_mla_rope_kv_with_tables(
    kv: torch.Tensor,
    k_pos_emb: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    head_dim_nope: int = HEAD_DIM_NOPE,
    head_dim_rope: int = HEAD_DIM_ROPE,
    head_dim_v: int = HEAD_DIM_V,
) -> tuple[torch.Tensor, torch.Tensor]:
    if HAVE_TRITON:
        return _MLARoPEKVTriton.apply(
            kv,
            k_pos_emb,
            cos_table,
            sin_table,
            head_dim_nope,
            head_dim_rope,
            head_dim_v,
        )
    return _apply_pytorch_kv(
        kv,
        k_pos_emb,
        cos_table,
        sin_table,
        head_dim_nope,
        head_dim_rope,
        head_dim_v,
    )


def apply_mla_rope_q(
    q: torch.Tensor,
    head_dim_nope: int = HEAD_DIM_NOPE,
    head_dim_rope: int = HEAD_DIM_ROPE,
    base: int = ROTARY_BASE,
    cos_table: torch.Tensor | None = None,
    sin_table: torch.Tensor | None = None,
) -> torch.Tensor:
    if cos_table is None or sin_table is None:
        s = q.shape[0]
        cos_table, sin_table = build_rope_tables(
            s,
            emb_dim=head_dim_rope,
            base=base,
            device=q.device,
        )
    return _apply_mla_rope_q_with_tables(
        q,
        cos_table,
        sin_table,
        head_dim_nope,
        head_dim_rope,
    )


def apply_mla_rope_kv(
    kv: torch.Tensor,
    k_pos_emb: torch.Tensor,
    head_dim_nope: int = HEAD_DIM_NOPE,
    head_dim_rope: int = HEAD_DIM_ROPE,
    head_dim_v: int = HEAD_DIM_V,
    base: int = ROTARY_BASE,
    cos_table: torch.Tensor | None = None,
    sin_table: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos_table is None or sin_table is None:
        s = kv.shape[0]
        cos_table, sin_table = build_rope_tables(
            s,
            emb_dim=head_dim_rope,
            base=base,
            device=kv.device,
        )
    return _apply_mla_rope_kv_with_tables(
        kv,
        k_pos_emb,
        cos_table,
        sin_table,
        head_dim_nope,
        head_dim_rope,
        head_dim_v,
    )


def apply_mla_rope(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_pos_emb: torch.Tensor,
    head_dim_nope: int = HEAD_DIM_NOPE,
    head_dim_rope: int = HEAD_DIM_ROPE,
    head_dim_v: int = HEAD_DIM_V,
    base: int = ROTARY_BASE,
    cos_table: torch.Tensor | None = None,
    sin_table: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if cos_table is None or sin_table is None:
        s = q.shape[0]
        cos_table, sin_table = build_rope_tables(
            s,
            emb_dim=head_dim_rope,
            base=base,
            device=q.device,
        )
    q = _apply_mla_rope_q_with_tables(q, cos_table, sin_table, head_dim_nope, head_dim_rope)
    k, v = _apply_mla_rope_kv_with_tables(
        kv,
        k_pos_emb,
        cos_table,
        sin_table,
        head_dim_nope,
        head_dim_rope,
        head_dim_v,
    )
    return q, k, v


def _rotate_interleaved_to_neox(
    x: torch.Tensor, cos_table: torch.Tensor, sin_table: torch.Tensor
) -> torch.Tensor:
    cos_ = cos_table[:, None, None, :].to(x.dtype)
    sin_ = sin_table[:, None, None, :].to(x.dtype)
    half_dim = x.shape[-1] // 2
    x_1 = x[..., 0::2]
    x_2 = x[..., 1::2]
    x_left = x_1 * cos_[..., :half_dim] - x_2 * sin_[..., :half_dim]
    x_right = x_2 * cos_[..., half_dim:] + x_1 * sin_[..., half_dim:]
    return torch.cat((x_left, x_right), dim=-1)


def _apply_pytorch_q(q, cos_table, sin_table, head_dim_nope, head_dim_rope):
    q_nope = q[..., :head_dim_nope]
    q_rope = q[..., head_dim_nope : head_dim_nope + head_dim_rope]
    q_rope = _rotate_interleaved_to_neox(q_rope, cos_table, sin_table)
    return torch.cat((q_nope, q_rope), dim=-1)


def _apply_pytorch_kv(
    kv,
    k_pos_emb,
    cos_table,
    sin_table,
    head_dim_nope,
    head_dim_rope,
    head_dim_v,
):
    k_nope = kv[..., :head_dim_nope]
    v = kv[..., head_dim_nope : head_dim_nope + head_dim_v]
    k_rope = _rotate_interleaved_to_neox(k_pos_emb, cos_table, sin_table).expand(
        -1, -1, kv.shape[2], -1
    )
    return torch.cat((k_nope, k_rope), dim=-1), v
