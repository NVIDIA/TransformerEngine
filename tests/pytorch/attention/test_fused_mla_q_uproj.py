# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Unit tests for FusedMLAQUpProjRopeQuant.

Validates the fused MXFP8 Q up-proj + per-head RoPE + dual-direction MXFP8 quantize kernel
against an unfused reference: MXFP8 quantize inputs -> bf16 GEMM -> RoPE -> MXFP8 quantize.

DSv3 671B MLA dimensions throughout; tokens must be a multiple of TILE_M (128).

Run:
    pytest tests/pytorch/attention/test_fused_mla_q_uproj.py -v
"""

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.attention import FusedMLAQUpProjRopeQuant
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor

# DSv3 671B MLA dims
NUM_HEADS = 128
HEAD_DIM_NOPE = 128
HEAD_DIM_ROPE = 64
HEAD_DIM = HEAD_DIM_NOPE + HEAD_DIM_ROPE  # 192
Q_LORA_RANK = 1536                         # K dimension of the up-proj GEMM
PROJ_DIM = NUM_HEADS * HEAD_DIM            # 24576

SEED = 42

fused_supported, reason_not_supported = (
    (True, "") if FusedMLAQUpProjRopeQuant.is_supported()
    else (False, "FusedMLAQUpProjRopeQuant.is_supported() returned False "
                 "(SM100+, cudnn-frontend >= 1.27.0, and NVTE_FUSED_MLA_Q_UPROJ=1 required)")
)


def _build_mxfp8_weight(proj_dim: int, k: int, device: torch.device) -> MXFP8Tensor:
    """Quantize a random bf16 weight to MXFP8Tensor (rowwise only, matching the primary param)."""
    w_bf16 = torch.randn(proj_dim, k, dtype=torch.bfloat16, device=device)
    quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)
    return quantizer(w_bf16)


def _dequantize_fused_output(query: MXFP8Tensor, s: int, b: int) -> torch.Tensor:
    """Dequantize the rowwise fused output to bf16 [s, b, nh, head_dim].

    query._rowwise_data is a plain float8_e4m3fn tensor — TE's C++ dequantize
    kernel requires 2D layout, so we reshape before calling MXFP8Tensor.dequantize().
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
    """Unfused bf16 reference: dequantize-then-GEMM, RoPE. Returns [s, b, nh, head_dim] bf16.

    We compute in bf16 (dequantizing x and w) rather than re-quantizing the output,
    so we compare against the fused kernel's dequantized output directly.
    The fused kernel's MXFP8 GEMM + output quantize will naturally introduce some error.
    """
    x_dq = (
        MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)(x)
        .dequantize()
        .to(torch.bfloat16)
    )
    w_dq = w_mxfp8.dequantize().to(torch.bfloat16)

    out = (x_dq @ w_dq.t()).view(s, b, NUM_HEADS, HEAD_DIM)

    # Per-head RoPE on the trailing HEAD_DIM_ROPE features, interleaved convention.
    q_nope = out[..., :HEAD_DIM_NOPE]
    q_rope = out[..., HEAD_DIM_NOPE:]
    cos_ = cos[:, None, None, :].to(q_rope.dtype)
    sin_ = sin[:, None, None, :].to(q_rope.dtype)
    half = HEAD_DIM_ROPE // 2
    x1 = q_rope[..., 0::2]
    x2 = q_rope[..., 1::2]
    x_left  = x1 * cos_[..., :half] - x2 * sin_[..., :half]
    x_right = x2 * cos_[..., half:] + x1 * sin_[..., half:]
    q_rope_out = torch.cat([x_left, x_right], dim=-1)
    return torch.cat([q_nope, q_rope_out], dim=-1)


@pytest.mark.skipif(not fused_supported, reason=reason_not_supported)
@pytest.mark.parametrize(
    "s, b",
    [
        (128, 1),   # minimum tile (TILE_M=128)
        (256, 1),
        (128, 2),
    ],
    ids=["s128_b1", "s256_b1", "s128_b2"],
)
def test_fused_mla_q_uproj_output_shapes(s: int, b: int) -> None:
    """Fused kernel output tensors must have the shapes wrap_mxfp8 promises."""
    device = torch.device("cuda")
    tokens = s * b
    torch.manual_seed(SEED)

    x = torch.randn(tokens, Q_LORA_RANK, dtype=torch.bfloat16, device=device)
    w = _build_mxfp8_weight(PROJ_DIM, Q_LORA_RANK, device)
    cos, sin = _build_rope_tables(tokens, device)

    query, x_saved = FusedMLAQUpProjRopeQuant.run(x, w, cos, sin, s, b)

    assert isinstance(query, MXFP8Tensor), f"Expected MXFP8Tensor, got {type(query)}"
    assert query.shape == (s, b, NUM_HEADS, HEAD_DIM), (
        f"Expected query shape {(s, b, NUM_HEADS, HEAD_DIM)}, got {query.shape}"
    )
    assert query._rowwise_data is not None,    "Missing rowwise data"
    assert query._columnwise_data is not None, "Missing columnwise data"

    blk = 32
    assert query._rowwise_scale_inv.shape    == (s, b, NUM_HEADS, HEAD_DIM // blk)
    assert query._columnwise_scale_inv.shape == (s // blk, b, NUM_HEADS, HEAD_DIM)


@pytest.mark.skipif(not fused_supported, reason=reason_not_supported)
@pytest.mark.parametrize("tokens", [128, 256], ids=["tokens128", "tokens256"])
def test_fused_mla_q_uproj_numerics(tokens: int) -> None:
    """Fused kernel output must be close to the unfused MXFP8-precision reference.

    Tolerance reflects the double quantization noise (activation + output quantize).
    """
    s, b = tokens, 1
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    x = torch.randn(tokens, Q_LORA_RANK, dtype=torch.bfloat16, device=device)
    w = _build_mxfp8_weight(PROJ_DIM, Q_LORA_RANK, device)
    cos, sin = _build_rope_tables(tokens, device)

    query, _ = FusedMLAQUpProjRopeQuant.run(x, w, cos, sin, s, b)

    fused_dq = _dequantize_fused_output(query, s, b)
    ref_dq = _reference_q_uproj(x, w, cos, sin, s, b)

    # FP8 GEMM + output quantize introduce ~10-20% relative error.
    torch.testing.assert_close(fused_dq, ref_dq, atol=0.5, rtol=0.1)


@pytest.mark.skipif(not fused_supported, reason=reason_not_supported)
def test_fused_mla_q_uproj_x_saved_is_mxfp8() -> None:
    """In the MXFP8 weight path, x_saved must be an MXFP8Tensor with columnwise data."""
    s, b = 128, 1
    device = torch.device("cuda")
    torch.manual_seed(SEED)

    x = torch.randn(s * b, Q_LORA_RANK, dtype=torch.bfloat16, device=device)
    w = _build_mxfp8_weight(PROJ_DIM, Q_LORA_RANK, device)
    cos, sin = _build_rope_tables(s * b, device)

    _, x_saved = FusedMLAQUpProjRopeQuant.run(x, w, cos, sin, s, b)

    assert isinstance(x_saved, MXFP8Tensor), (
        f"x_saved should be MXFP8Tensor for MXFP8 weight path, got {type(x_saved)}"
    )
    assert x_saved._columnwise_data is not None, "x_saved must retain columnwise data for wgrad"
    assert x_saved._rowwise_data is None, "x_saved rowwise data should be dropped after forward"


def _build_rope_tables(
    tokens: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain cos/sin tables for HEAD_DIM_ROPE, interleaved [tokens, rope_dim] bf16."""
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, HEAD_DIM_ROPE, 2, dtype=torch.float32, device=device)
                  / HEAD_DIM_ROPE)
    )
    t = torch.arange(tokens, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    freqs = torch.cat([freqs, freqs], dim=-1)
    return freqs.cos().to(torch.bfloat16), freqs.sin().to(torch.bfloat16)
