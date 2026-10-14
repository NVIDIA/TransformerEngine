# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Unit tests for the fused multi-latent attention (MLA) kernels.

- FusedMLAQUpProjRopeQuant: query up-projection, RoPE and MXFP8 quantization (cuDNN FE).
- fused_mla_kv_rope_mxfp8: key/value split, key RoPE and MXFP8 quantization. The reference
  applies Megatron-LM's MLA key/value RoPE (the Triton kernel in mla_rope_utils) and quantizes
  key and value with mxfp8_quantize_only; the fused kernel must produce the same bytes.

Run:
    pytest tests/pytorch/attention/test_fused_mla.py -v
"""

import os
import pathlib
import sys

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.attention import FusedMLAQUpProjFunction, FusedMLAQUpProjRopeQuant
from transformer_engine.pytorch.attention.dot_product_attention.utils import (
    fused_mla_kv_rope_mxfp8,
    mxfp8_quantize_only,
)
from transformer_engine.pytorch.cpp_extensions import general_gemm as _fused_general_gemm
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor

_current_file = pathlib.Path(__file__).resolve()
sys.path = [str(_current_file.parent)] + sys.path
from mla_rope_utils import HAVE_TRITON, apply_mla_rope_kv, build_rope_tables

# All tests in this file require GB200/SM100 hardware.
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
    dq = fp8_col.to(torch.float32).view(
        tokens // _MXFP8_BLOCK, _MXFP8_BLOCK, NUM_HEADS, HEAD_DIM
    ) * inv.unsqueeze(1)
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
        x_run = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)(x)
    else:
        w = w_raw
        x_run = x
    cos, sin = _build_rope_tables(tokens, device)

    query, x_saved = FusedMLAQUpProjRopeQuant.run(x_run, w, cos, sin, s, b)

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

    Verifies wiring: Triton RoPE backward + TE linear and RMSNorm backward produce
    the same result as the reference computed with the same kernels.
    """
    import triton

    from transformer_engine.pytorch.attention.fused_mla_q_uproj import rotary_bwd_q_kernel

    tokens, s, b = 256, 256, 1
    device = torch.device("cuda")
    torch.manual_seed(SEED)

    x = torch.randn(s, b, Q_LORA_RANK, dtype=torch.bfloat16, device=device, requires_grad=True)
    gamma = torch.ones(Q_LORA_RANK, dtype=torch.bfloat16, device=device, requires_grad=True)
    w_bf16 = torch.randn(
        PROJ_DIM, Q_LORA_RANK, dtype=torch.bfloat16, device=device, requires_grad=True
    )
    w = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)(w_bf16)
    eps = 1.0e-6
    normalization = "RMSNorm"
    zero_centered_gamma = False
    cos, sin = _build_rope_tables(tokens, device)
    cos_flat = cos.reshape(s, -1).contiguous()
    sin_flat = sin.reshape(s, -1).contiguous()

    from transformer_engine.pytorch.module._common import apply_normalization

    x2d = x.detach().reshape(tokens, Q_LORA_RANK).contiguous()
    x_quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
    ln_out, _, rsigma = apply_normalization(
        x2d,
        None,
        gamma.detach(),
        None,
        eps,
        x_quantizer,
        x2d.dtype,
        normalization,
        int(os.getenv("NVTE_FWD_LAYERNORM_SM_MARGIN", "0")),
        zero_centered_gamma,
    )
    _, x_saved = FusedMLAQUpProjRopeQuant.run(ln_out, w.detach(), cos_flat, sin_flat, s, b)
    grad_out = torch.randn(s, b, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)

    query = FusedMLAQUpProjFunction.apply(
        x,
        gamma,
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
        eps,
        normalization,
        zero_centered_gamma,
    )
    assert query.requires_grad
    assert query.grad_fn is not None
    torch.autograd.backward(query, grad_out.clone())
    assert x.grad is not None
    assert gamma.grad is not None
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
    grad_ln_out_ref = _fused_general_gemm(
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
    grad_x_ref, grad_gamma_ref = tex.rmsnorm_bwd(
        grad_ln_out_ref.reshape(x2d.shape),
        x2d,
        rsigma,
        gamma.detach(),
        int(os.getenv("NVTE_BWD_LAYERNORM_SM_MARGIN", "0")),
        zero_centered_gamma,
    )
    torch.testing.assert_close(x.grad.reshape(tokens, Q_LORA_RANK), grad_x_ref, atol=0.5, rtol=0.1)
    torch.testing.assert_close(gamma.grad, grad_gamma_ref, atol=0.5, rtol=0.1)
    torch.testing.assert_close(w_bf16.grad, grad_w_ref, atol=0.5, rtol=0.1)


# --- fused_mla_kv_rope_mxfp8 ---------------------------------------------------------------------

mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
sm100 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10
if not mxfp8_available:
    _kv_rope_skip_reason = reason_for_no_mxfp8
elif not sm100:
    _kv_rope_skip_reason = "The fused kernel requires SM 10.0+."
elif not HAVE_TRITON:
    _kv_rope_skip_reason = "The reference MLA RoPE needs Triton."
else:
    _kv_rope_skip_reason = ""
requires_kv_rope_kernel = pytest.mark.skipif(
    bool(_kv_rope_skip_reason), reason=_kv_rope_skip_reason
)

HEAD_DIM_V = 128
KV_LORA_RANK = 512
MXFP8_BUFFERS = ("_rowwise_data", "_rowwise_scale_inv", "_columnwise_data", "_columnwise_scale_inv")


def _kv_quantizer():
    return MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)


def _kv_rope_inputs(s, b, h, cos_dtype, seed=1234):
    torch.manual_seed(seed)
    kv = (torch.randn(s, b, h, HEAD_DIM_NOPE + HEAD_DIM_V, device="cuda") * 4).to(torch.bfloat16)
    # An all-zero scaling block and a value that saturates the scale of its blocks.
    kv[5, 0, 0, :32] = 0
    kv[7, 0, h - 1, 3] = 3.0e4
    kv[9, b - 1, 1, HEAD_DIM_NOPE + 17] = -2.5e4
    # k_rope is a slice of the key/value down-projection output, as in MLA.
    kv_down = (torch.randn(s, b, KV_LORA_RANK + HEAD_DIM_ROPE, device="cuda") * 4).to(
        torch.bfloat16
    )
    k_rope = kv_down[..., KV_LORA_RANK:].unsqueeze(-2)
    cos, sin = build_rope_tables(s, emb_dim=HEAD_DIM_ROPE, device="cuda")
    # A magnitude scale as in YaRN, so that the products are not exact.
    cos, sin = (cos * 1.2).to(cos_dtype), (sin * 1.2).to(cos_dtype)
    return kv, k_rope, cos, sin


def _kv_rope_reference(kv, k_rope, cos, sin):
    key, value = apply_mla_rope_kv(kv, k_rope, cos_table=cos, sin_table=sin)
    return mxfp8_quantize_only(
        [(key.contiguous(), _kv_quantizer()), (value.contiguous(), _kv_quantizer())], "sbhd"
    )


def _assert_identical(actual, expected, name):
    assert actual.shape == expected.shape, f"{name}: shape {actual.shape} vs {expected.shape}"
    for buffer in MXFP8_BUFFERS:
        a, e = getattr(actual, buffer), getattr(expected, buffer)
        assert a.shape == e.shape, f"{name}.{buffer}: shape {tuple(a.shape)} vs {tuple(e.shape)}"
        a, e = a.view(torch.uint8), e.view(torch.uint8)
        mismatches = int((a != e).sum())
        assert mismatches == 0, f"{name}.{buffer}: {mismatches} of {e.numel()} bytes differ"


@requires_kv_rope_kernel
@pytest.mark.parametrize(
    "s, b, h", [(4096, 1, 128), (2048, 2, 64), (512, 1, 16), (256, 3, 8), (128, 2, 2)]
)
@pytest.mark.parametrize("cos_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_matches_rope_then_quantize(s, b, h, cos_dtype):
    """Same bytes as the RoPE / split kernel followed by MXFP8 quantization."""
    kv, k_rope, cos, sin = _kv_rope_inputs(s, b, h, cos_dtype)
    ref_key, ref_value = _kv_rope_reference(kv, k_rope, cos, sin)
    cos4, sin4 = cos.view(s, 1, 1, HEAD_DIM_ROPE), sin.view(s, 1, 1, HEAD_DIM_ROPE)
    key, value = fused_mla_kv_rope_mxfp8(kv, k_rope, cos4, sin4, HEAD_DIM_V)
    _assert_identical(key, ref_key, "key")
    _assert_identical(value, ref_value, "value")


@requires_kv_rope_kernel
def test_contiguous_k_rope_and_longer_rope_tables():
    """A contiguous [s, b, emb_dim] k_rope and cos / sin with more rows than tokens."""
    s, b, h = 256, 2, 4
    kv, k_rope, _, _ = _kv_rope_inputs(s, b, h, torch.bfloat16)
    cos, sin = build_rope_tables(2 * s, emb_dim=HEAD_DIM_ROPE, device="cuda")
    cos, sin = cos.to(torch.bfloat16), sin.to(torch.bfloat16)
    ref_key, ref_value = _kv_rope_reference(kv, k_rope, cos[:s], sin[:s])
    key, value = fused_mla_kv_rope_mxfp8(kv, k_rope.squeeze(-2).contiguous(), cos, sin, 128)
    _assert_identical(key, ref_key, "key")
    _assert_identical(value, ref_value, "value")


@requires_kv_rope_kernel
def test_non_finite_values():
    """Inf and NaN inputs give the quantizer's scaling factors and data."""
    s, b, h = 128, 1, 4
    kv, k_rope, cos, sin = _kv_rope_inputs(s, b, h, torch.bfloat16)
    kv[3, 0, 0, 40] = float("inf")
    kv[40, 0, 1, HEAD_DIM_NOPE + 5] = float("-inf")
    kv[64, 0, 2, 100] = float("nan")
    kv[96:128, 0, 3, HEAD_DIM_NOPE + 32 : HEAD_DIM_NOPE + 64] = float("nan")
    ref_key, ref_value = _kv_rope_reference(kv, k_rope, cos, sin)
    key, value = fused_mla_kv_rope_mxfp8(kv, k_rope, cos, sin, HEAD_DIM_V)
    _assert_identical(key, ref_key, "key")
    _assert_identical(value, ref_value, "value")


@requires_kv_rope_kernel
@pytest.mark.parametrize(
    "s, h, message", [(96, 4, "multiple of 128"), (128, 3, "must be even")], ids=["s", "h"]
)
def test_unsupported_shapes(s, h, message):
    kv, k_rope, cos, sin = _kv_rope_inputs(s, 1, h, torch.bfloat16)
    with pytest.raises((RuntimeError, ValueError), match=message):
        fused_mla_kv_rope_mxfp8(kv, k_rope, cos, sin, HEAD_DIM_V)


@requires_kv_rope_kernel
def test_2d_quantizer_rejected():
    """The kernel quantizes 1D blocks, so 2D MXFP8 quantizers are rejected."""
    kv, k_rope, cos, sin = _kv_rope_inputs(128, 1, 4, torch.bfloat16)
    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True, with_2d_quantization=True
    )
    with pytest.raises(RuntimeError, match="2D MXFP8 quantization"):
        fused_mla_kv_rope_mxfp8(kv, k_rope, cos, sin, HEAD_DIM_V, quantizer, quantizer)


@requires_kv_rope_kernel
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Needs two GPUs.")
def test_inputs_on_non_current_device():
    """Outputs are allocated, and the kernel runs, on the inputs' device."""
    with torch.cuda.device(1):
        kv, k_rope, cos, sin = _kv_rope_inputs(128, 2, 4, torch.bfloat16)
        ref_key, ref_value = _kv_rope_reference(kv, k_rope, cos, sin)
    assert torch.cuda.current_device() == 0
    key, value = fused_mla_kv_rope_mxfp8(kv, k_rope, cos, sin, HEAD_DIM_V)
    assert key._rowwise_data.device == kv.device and value._rowwise_data.device == kv.device
    _assert_identical(key, ref_key, "key")
    _assert_identical(value, ref_value, "value")
    with pytest.raises(RuntimeError, match="same device"):
        fused_mla_kv_rope_mxfp8(kv, k_rope.to("cuda:0"), cos, sin, HEAD_DIM_V)
