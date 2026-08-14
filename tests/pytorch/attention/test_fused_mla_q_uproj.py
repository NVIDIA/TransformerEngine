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
from transformer_engine.pytorch.attention import FusedMLAQUpProjFunction, FusedMLAQUpProjRopeQuant
from transformer_engine.pytorch.cpp_extensions import general_gemm as _fused_general_gemm
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor

# All tests in this file require GB200/SM100 hardware (cuDNN FE fused kernel).
pytestmark = pytest.mark.launch_on_gb200

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


_MXFP8_BLOCK = 32  # E8M0 block size; matches MXFP8_BLOCK_SCALING_SIZE in TE constants
_E8M0_BIAS = 127


def _dequantize_fused_output(query: MXFP8Tensor, s: int, b: int) -> torch.Tensor:
    """Dequantize the rowwise fused output to bf16 [s, b, nh, head_dim].

    TE's C++ dequantize kernel requires 2D layout, so reshape before calling dequantize().
    """
    tokens = s * b
    q_2d = MXFP8Tensor(
        shape=(tokens, PROJ_DIM),
        dtype=torch.bfloat16,
        rowwise_data=query._rowwise_data.view(tokens, PROJ_DIM),
        rowwise_scale_inv=query._rowwise_scale_inv.view(tokens, PROJ_DIM // _MXFP8_BLOCK),
        columnwise_data=None,
        columnwise_scale_inv=None,
        quantizer=query._quantizer,
        requires_grad=False,
        fp8_dtype=query._fp8_dtype,
        with_gemm_swizzled_scales=False,
    )
    return q_2d.dequantize().to(torch.bfloat16).view(s, b, NUM_HEADS, HEAD_DIM)


def _dequantize_fused_output_col(query: MXFP8Tensor, s: int, b: int) -> torch.Tensor:
    """Dequantize the columnwise fused output to bf16 [s, b, nh, head_dim].

    The columnwise layout groups _MXFP8_BLOCK consecutive tokens per (head, head_dim)
    E8M0 scale.  Matches the dequantization formula used in the cuDNN kernel's own
    reference (test_gemm_proj_rope_mxfp8_utils._deq_col).
    """
    tokens = s * b
    assert tokens % _MXFP8_BLOCK == 0, f"tokens ({tokens}) must be divisible by {_MXFP8_BLOCK}"
    # Flatten batch into the token dimension to match the cuDNN kernel's native 3-D layout.
    fp8_col = query._columnwise_data.view(tokens, NUM_HEADS, HEAD_DIM)
    scale_col = query._columnwise_scale_inv.view(tokens // _MXFP8_BLOCK, NUM_HEADS, HEAD_DIM)
    inv = torch.pow(2.0, scale_col.to(torch.float32) - _E8M0_BIAS)  # (tok//32, NH, HD)
    dq = (
        fp8_col.to(torch.float32).view(tokens // _MXFP8_BLOCK, _MXFP8_BLOCK, NUM_HEADS, HEAD_DIM)
        * inv.unsqueeze(1)
    )
    return dq.reshape(s, b, NUM_HEADS, HEAD_DIM).to(torch.bfloat16)


def _reference_q_uproj(
    x: torch.Tensor,
    w,  # MXFP8Tensor (fp8_weight=True) or bf16 torch.Tensor (fp8_weight=False)
    cos: torch.Tensor,
    sin: torch.Tensor,
    s: int,
    b: int,
) -> torch.Tensor:
    """Unfused bf16 reference: dequantize-then-GEMM + RoPE. Returns [s, b, nh, head_dim] bf16."""
    if isinstance(w, MXFP8Tensor):
        x_dq = (
            MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)(x)
            .dequantize()
            .to(torch.bfloat16)
        )
        w_dq = w.dequantize().to(torch.bfloat16)
    else:
        x_dq = x.to(torch.bfloat16)
        w_dq = w.to(torch.bfloat16)
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
    t = torch.arange(tokens, device=device, dtype=torch.float32)
    # Asymmetric table: left half uses standard inv_freq, right half uses a different
    # scale so cos[:, :32] != cos[:, 32:].  This exposes any forward or backward kernel
    # that accidentally treats the two halves as identical (which the symmetric table
    # would silently hide).
    freqs = torch.cat([torch.outer(t, inv_freq), torch.outer(t, inv_freq * 0.5)], dim=-1)
    return freqs.cos().to(torch.bfloat16), freqs.sin().to(torch.bfloat16)


@pytest.mark.skipif(not fused_supported, reason=reason_not_supported)
@pytest.mark.parametrize("tokens", [256])
@pytest.mark.parametrize("fp8_weight", [True, False], ids=["fp8", "bf16"])
def test_fused_mla_q_uproj(tokens: int, fp8_weight: bool) -> None:
    """Forward numerics and x_saved properties for FusedMLAQUpProjRopeQuant.run().

    fp8_weight=True  exercises the MXFP8 Q-projection GEMM branch (production path).
    fp8_weight=False exercises the BF16 weight branch.
    """
    s, b = tokens, 1
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    x = torch.randn(tokens, Q_LORA_RANK, dtype=torch.bfloat16, device=device)
    w_raw = torch.randn(PROJ_DIM, Q_LORA_RANK, dtype=torch.bfloat16, device=device)
    if fp8_weight:
        w = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)(w_raw)
    else:
        w = w_raw
    cos, sin = _build_rope_tables(tokens, device)

    query, x_saved = FusedMLAQUpProjRopeQuant.run(x, w, cos, sin, s, b)

    # Forward numerics: E4M3 output quantization gives ≤6.25% relative error (= 2^-4) per
    # normalized element. rtol=0.07 sits just above this floor with margin for subnormals.
    fused_dq = _dequantize_fused_output(query, s, b)
    ref_dq = _reference_q_uproj(x, w, cos, sin, s, b)
    torch.testing.assert_close(fused_dq, ref_dq, atol=0.5, rtol=0.07)

    # Columnwise output: same E4M3 tolerance as rowwise.  The columnwise Q feeds the dK
    # GEMM in the attention backward, so a bug here causes wrong K-path gradients without
    # any other test catching it.
    fused_dq_col = _dequantize_fused_output_col(query, s, b)
    torch.testing.assert_close(fused_dq_col, ref_dq, atol=0.5, rtol=0.07)

    # x_saved shape and type depend on which branch ran.
    if fp8_weight:
        # FP8 path: x_saved must be MXFP8 with only columnwise data retained for wgrad.
        assert isinstance(x_saved, MXFP8Tensor), "FP8-weight path must save MXFP8 activation"
        assert x_saved._columnwise_data is not None, "x_saved must retain columnwise data for wgrad"
        assert x_saved._rowwise_data is None, "x_saved rowwise data should be dropped after forward"
    else:
        assert not isinstance(x_saved, MXFP8Tensor), "BF16-weight path must save plain tensor"


@pytest.mark.skipif(not fused_supported, reason=reason_not_supported)
def test_fused_mla_q_uproj_autograd() -> None:
    """The real autograd path must produce correct input and weight gradients.

    Verifies wiring: Triton RoPE backward + TE linear backward produce the same
    result as the reference computed with the same kernels. For an independent
    pure-PyTorch cross-check (reviewer #15), see test_fused_mla_q_uproj_autograd_pytorch_ref.
    """
    import triton

    from transformer_engine.pytorch.attention.fused_mla_q_uproj import rotary_bwd_q_kernel

    tokens, s, b = 256, 256, 1
    device = torch.device("cuda")
    torch.manual_seed(SEED)

    x = torch.randn(s, b, Q_LORA_RANK, dtype=torch.bfloat16, device=device, requires_grad=True)
    w_bf16 = torch.randn(
        PROJ_DIM, Q_LORA_RANK, dtype=torch.bfloat16, device=device, requires_grad=True
    )
    w = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)(w_bf16)
    cos, sin = _build_rope_tables(tokens, device)
    cos_flat = cos.reshape(s, -1).contiguous()
    sin_flat = sin.reshape(s, -1).contiguous()
    _, x_saved = FusedMLAQUpProjRopeQuant.run(
        x.detach().reshape(tokens, Q_LORA_RANK), w.detach(), cos_flat, sin_flat, s, b
    )
    grad_out = torch.randn(s, b, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)

    query = FusedMLAQUpProjFunction.apply(
        x,
        w,
        cos[:, None, None, :],
        sin[:, None, None, :],
        None,
        False,
        NUM_HEADS,
        HEAD_DIM,
        HEAD_DIM_NOPE,
        HEAD_DIM_ROPE,
        s,
        b,
        None,
        False,
    )
    assert query.requires_grad
    assert query.grad_fn is not None
    torch.autograd.backward(query, grad_out.clone())
    assert x.grad is not None
    assert w_bf16.grad is not None

    dq3 = grad_out.reshape(tokens, NUM_HEADS, HEAD_DIM).clone().contiguous()
    grid = lambda META: (tokens, triton.cdiv(NUM_HEADS, META["BLOCK_H"]))
    rotary_bwd_q_kernel[grid](
        dq3,
        cos_flat,
        sin_flat,
        HEAD_DIM_NOPE,
        HEAD_DIM_ROPE,
        NUM_HEADS,
        1,
        None,
        None,
        dq3.stride(0),
        dq3.stride(1),
        0,
        1,
    )
    dq2d = dq3.reshape(tokens, PROJ_DIM).contiguous()
    gy_quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
    gy_quantizer.optimize_for_gemm = True
    gy = gy_quantizer(dq2d)

    w.update_usage(rowwise_usage=True, columnwise_usage=True)
    grad_x_ref = _fused_general_gemm(
        w, gy, layout="NN", grad=True, out_dtype=torch.bfloat16, use_split_accumulator=True
    )[0]
    grad_w_ref = _fused_general_gemm(
        x_saved,
        gy,
        layout="NT",
        grad=True,
        out_dtype=torch.bfloat16,
        use_split_accumulator=True,
    )[0]
    torch.testing.assert_close(x.grad.reshape(tokens, Q_LORA_RANK), grad_x_ref, atol=0.5, rtol=0.1)
    torch.testing.assert_close(w_bf16.grad, grad_w_ref, atol=0.5, rtol=0.1)

