# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Unit tests for FusedMLAQUpProjRopeQuant.

Run:
    pytest tests/pytorch/attention/test_fused_mla_q_uproj.py -v
"""

import pytest
import torch

import transformer_engine.pytorch  # registers transformer_engine_torch
import transformer_engine_torch as tex
from transformer_engine.pytorch.attention import FusedMLAQUpProjRopeQuant
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor

# DSv3 671B MLA dims
NUM_HEADS = 128
HEAD_DIM_NOPE = 128
HEAD_DIM_ROPE = 64
HEAD_DIM = HEAD_DIM_NOPE + HEAD_DIM_ROPE  # 192
Q_LORA_RANK = 1536
PROJ_DIM = NUM_HEADS * HEAD_DIM  # 24576

SEED = 42

fused_supported, reason_not_supported = (
    (True, "")
    if FusedMLAQUpProjRopeQuant.is_supported()
    else (
        False,
        (
            "FusedMLAQUpProjRopeQuant.is_supported() returned False "
            "(SM100+, cudnn-frontend >= 1.27.0, and NVTE_FUSED_MLA_Q_UPROJ=1 required)"
        ),
    )
)


def _dequantize_fused_output(query: MXFP8Tensor, s: int, b: int) -> torch.Tensor:
    """Dequantize the rowwise fused output to bf16 [s, b, nh, head_dim].

    TE's C++ dequantize kernel requires 2D layout, so reshape before calling dequantize().
    """
    tokens = s * b
    q_2d = MXFP8Tensor(
        shape=(tokens, PROJ_DIM),
        dtype=torch.bfloat16,
        rowwise_data=query._rowwise_data.view(tokens, PROJ_DIM),
        rowwise_scale_inv=query._rowwise_scale_inv.view(tokens, PROJ_DIM // 32),
        columnwise_data=None,
        columnwise_scale_inv=None,
        quantizer=query._quantizer,
        requires_grad=False,
        fp8_dtype=query._fp8_dtype,
        with_gemm_swizzled_scales=False,
    )
    return q_2d.dequantize().to(torch.bfloat16).view(s, b, NUM_HEADS, HEAD_DIM)


def _reference_q_uproj(
    x: torch.Tensor,
    w_mxfp8: MXFP8Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    s: int,
    b: int,
) -> torch.Tensor:
    """Unfused bf16 reference: dequantize-then-GEMM + RoPE. Returns [s, b, nh, head_dim] bf16."""
    x_dq = (
        MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)(x)
        .dequantize()
        .to(torch.bfloat16)
    )
    w_dq = w_mxfp8.dequantize().to(torch.bfloat16)
    out = (x_dq @ w_dq.t()).view(s, b, NUM_HEADS, HEAD_DIM)

    q_nope = out[..., :HEAD_DIM_NOPE]
    q_rope = out[..., HEAD_DIM_NOPE:]
    cos_ = cos[:, None, None, :].to(q_rope.dtype)
    sin_ = sin[:, None, None, :].to(q_rope.dtype)
    half = HEAD_DIM_ROPE // 2
    x1, x2 = q_rope[..., 0::2], q_rope[..., 1::2]
    q_rope_out = torch.cat(
        [
            x1 * cos_[..., :half] - x2 * sin_[..., :half],
            x2 * cos_[..., half:] + x1 * sin_[..., half:],
        ],
        dim=-1,
    )
    return torch.cat([q_nope, q_rope_out], dim=-1)


def _build_rope_tables(tokens: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        10000
        ** (torch.arange(0, HEAD_DIM_ROPE, 2, dtype=torch.float32, device=device) / HEAD_DIM_ROPE)
    )
    freqs = torch.cat(
        [torch.outer(torch.arange(tokens, device=device, dtype=torch.float32), inv_freq)] * 2,
        dim=-1,
    )
    return freqs.cos().to(torch.bfloat16), freqs.sin().to(torch.bfloat16)


@pytest.mark.skipif(not fused_supported, reason=reason_not_supported)
@pytest.mark.parametrize("tokens", [256])
def test_fused_mla_q_uproj(tokens: int) -> None:
    """Forward numerics and x_saved properties for FusedMLAQUpProjRopeQuant.run().

    Full forward+backward autograd testing (via _FusedMLAQUpProjFunction) lives in
    Megatron-Core.
    """
    s, b = tokens, 1
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    x = torch.randn(tokens, Q_LORA_RANK, dtype=torch.bfloat16, device=device)
    w = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)(
        torch.randn(PROJ_DIM, Q_LORA_RANK, dtype=torch.bfloat16, device=device)
    )
    cos, sin = _build_rope_tables(tokens, device)

    query, x_saved = FusedMLAQUpProjRopeQuant.run(x, w, cos, sin, s, b)

    # Forward numerics: FP8 GEMM + output quantize introduce ~10% relative error.
    fused_dq = _dequantize_fused_output(query, s, b)
    ref_dq = _reference_q_uproj(x, w, cos, sin, s, b)
    torch.testing.assert_close(fused_dq, ref_dq, atol=0.5, rtol=0.1)

    # x_saved: must be MXFP8 with only columnwise data retained for wgrad.
    assert isinstance(x_saved, MXFP8Tensor)
    assert x_saved._columnwise_data is not None, "x_saved must retain columnwise data for wgrad"
    assert x_saved._rowwise_data is None, "x_saved rowwise data should be dropped after forward"
