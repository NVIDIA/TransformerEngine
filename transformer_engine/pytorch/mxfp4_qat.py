# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MXFP4 weight fake-quantization for quantization-aware training.

Projects a weight onto the MXFP4 (E2M1) grid with 1x32 power-of-two (UE8M0)
block scales and returns the dequantized result in the input dtype — i.e. the
fused form of ``bf16 -> mxfp4(row) -> dequantize -> bf16``. The returned values
are exactly representable in bf16/fp32 (E2M1 needs at most two significant
mantissa bits and the scales are powers of two), so composing with the base
recipe's weight quantization stays lossless:

* MXFP8 (1x32, E8M0): MXFP4 blocks are 1x32-aligned and the E2M1 grid is a
  subset of E4M3 — the rowwise MXFP8 encoding of the projected weight is exact.
* Float8 block scaling (128x128, power-of-two scales): exact whenever every
  1x32 MXFP4 scale within a tile lies within the FP8 dynamic-range headroom of
  the tile's maximum scale.

Three bit-identical implementations are dispatched via ``NVTE_MXFP4_QAT_IMPL``
(``auto``/``cuda``/``cute_dsl``/``torch``, default ``auto`` = CUDA binding,
then CuTe DSL, then the PyTorch reference): the CUDA kernel in
``common/cast/mxfp4/fake_quantize_mxfp4.cuh`` (``tex.mxfp4_fake_quantize``),
the CuTe DSL kernel in ``mxfp4_qat_cute_dsl.py``, and
``_mxfp4_fake_quantize_torch`` below.

Value-domain semantics (identical in all implementations):
* zero block (amax == 0): scale 1, output zeros — exact.
* tiny amax (<= 6*2^-126): the scale floors at 2^-126 (TileKernels deployment
  contract). The 0.5 payload then dequantizes to 2^-127, a bf16/fp32
  subnormal. NOTE: TE's MXFP8 software dequantize flushes that grid point to
  zero until rebuilt with the ptx.cuh UE8M0 code-0 fix (raw encoding and
  hardware GEMM are exact).
* non-finite amax (inf/NaN anywhere in a 1x32 block): the WHOLE block becomes
  NaN. A single inf must not silently rescale its 31 neighbors to zero, and
  NaN must not silently fall back to scale 1 — corruption is made visible.
* huge amax (> 6*2^125): the scale is capped at 2^125 so the dequantized grid
  stays representable in bf16/fp32; values saturate at 6*2^125 (satfinite).
* fp16 weights are rejected: even capped scales overflow fp16's range.
"""
import os
import warnings

import torch

__all__ = ["mxfp4_fake_quantize"]

_E2M1_MAX = 6.0
_MXFP4_BLOCK = 32

_IMPL_CHOICE = os.getenv("NVTE_MXFP4_QAT_IMPL", "auto")
_torch_path_forced = os.getenv("NVTE_MXFP4_QAT_DISABLE_CUDA_KERNEL", "0") == "1"
_kernel_required = os.getenv("NVTE_MXFP4_QAT_REQUIRE_KERNEL", "0") == "1"
_missing_kernel_warned = False
_cute_dsl_import_error = None


class _MXFP4FakeQuantizeSTE(torch.autograd.Function):
    """Straight-through estimator: identity gradient through the projection."""

    @staticmethod
    def forward(ctx, weight, impl):  # pylint: disable=arguments-differ
        return impl(weight)

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ
        return grad_output, None


def _round_to_e2m1_grid(y: torch.Tensor) -> torch.Tensor:
    """RTNE onto the E2M1 magnitude grid {0, .5, 1, 1.5, 2, 3, 4, 6}.

    Input must be non-negative and <= 6. Within each binade the grid is
    uniform, so torch.round (RTNE) on the rescaled value reproduces IEEE RTNE
    exactly, including ties across binade boundaries (2.5 -> 2, 3.5 -> 4, 5 -> 4).
    """
    fine = torch.round(y * 2.0) * 0.5
    mid = torch.round(y)
    coarse = torch.round(y * 0.5) * 2.0
    return torch.where(y <= 2.0, fine, torch.where(y <= 4.0, mid, coarse))


def _mxfp4_fake_quantize_torch(weight: torch.Tensor) -> torch.Tensor:
    """Bit-identical PyTorch reference for the CUDA and CuTe DSL kernels."""
    rows, cols = weight.shape
    w32 = weight.contiguous().to(torch.float32).view(rows, cols // _MXFP4_BLOCK, _MXFP4_BLOCK)
    amax = w32.abs().amax(dim=-1, keepdim=True)
    nonfinite = ~torch.isfinite(amax)

    bits = amax.view(torch.int32)
    exp_field = bits >> 23
    mantissa = bits & 0x7FFFFF
    exp = exp_field - 129 + (mantissa > 0x400000).to(torch.int32)
    exp = torch.where(exp_field > 0, exp, torch.full_like(exp, -126))
    exp = exp.clamp(min=-126, max=125)
    scale = torch.ldexp(torch.ones_like(amax), exp)
    scale = torch.where(amax > 0, scale, torch.ones_like(scale))

    y = (w32 / scale).abs().clamp(max=_E2M1_MAX)
    q = _round_to_e2m1_grid(torch.where(nonfinite, torch.zeros_like(y), y))
    q = torch.copysign(q, w32)
    out = q * scale
    out = torch.where(nonfinite.expand_as(out), torch.full_like(out, float("nan")), out)
    return out.view(rows, cols).to(weight.dtype)


def _cuda_impl():
    import transformer_engine_torch as tex

    if not hasattr(tex, "mxfp4_fake_quantize"):
        return None

    def impl(w):
        w = w.contiguous()
        if w.data_ptr() % 16 != 0:
            w = w.clone()
        return tex.mxfp4_fake_quantize(w)

    return impl


def _cute_dsl_impl():
    global _cute_dsl_import_error
    if _cute_dsl_import_error is not None:
        return None
    try:
        from .mxfp4_qat_cute_dsl import mxfp4_fake_quantize_cute_dsl
    except Exception as exc:  # pylint: disable=broad-except
        _cute_dsl_import_error = exc
        return None
    return mxfp4_fake_quantize_cute_dsl


def _resolve_impl(weight: torch.Tensor):
    if not weight.is_cuda:
        return _mxfp4_fake_quantize_torch
    if _IMPL_CHOICE == "torch":
        return _mxfp4_fake_quantize_torch
    if _IMPL_CHOICE == "cuda":
        impl = _cuda_impl()
        if impl is None:
            raise RuntimeError(
                "NVTE_MXFP4_QAT_IMPL=cuda but transformer_engine_torch was built "
                "without mxfp4_fake_quantize. Rebuild Transformer Engine."
            )
        return impl
    if _IMPL_CHOICE == "cute_dsl":
        impl = _cute_dsl_impl()
        if impl is None:
            raise RuntimeError(
                "NVTE_MXFP4_QAT_IMPL=cute_dsl but the cutlass CuTe DSL is "
                f"unavailable: {_cute_dsl_import_error!r}"
            )
        return impl
    if _IMPL_CHOICE != "auto":
        raise ValueError(
            f"NVTE_MXFP4_QAT_IMPL={_IMPL_CHOICE!r} is not one of "
            "'auto', 'cuda', 'cute_dsl', 'torch'"
        )
    if _torch_path_forced:
        return _mxfp4_fake_quantize_torch
    impl = _cuda_impl()
    if impl is not None:
        return impl
    impl = _cute_dsl_impl()
    if impl is not None:
        return impl
    if _kernel_required:
        raise RuntimeError(
            "NVTE_MXFP4_QAT_REQUIRE_KERNEL=1 but neither the CUDA binding nor "
            "the CuTe DSL kernel is available. Rebuild Transformer Engine or "
            "install the cutlass CuTe DSL."
        )
    global _missing_kernel_warned
    if not _missing_kernel_warned:
        _missing_kernel_warned = True
        warnings.warn(
            "transformer_engine_torch was built without mxfp4_fake_quantize and "
            "the CuTe DSL is unavailable; falling back to the (slower) PyTorch "
            "reference. Rebuild Transformer Engine to use the CUDA kernel.",
            stacklevel=2,
        )
    return _mxfp4_fake_quantize_torch


def mxfp4_fake_quantize(weight: torch.Tensor) -> torch.Tensor:
    """Project ``weight`` onto the MXFP4 grid and return it in the input dtype.

    Per 1x32 block: scale = 2^ceil(log2(amax / 6)) (all-zero blocks use 1.0),
    values are divided by the scale, rounded RTNE onto the E2M1 grid with
    saturation at +-6, and multiplied back. Blocks containing non-finite
    values become NaN. The gradient is the straight-through estimator
    (identity), the QAT-correct semantics.

    Env knobs: ``NVTE_MXFP4_QAT_IMPL`` picks the implementation
    (``auto``/``cuda``/``cute_dsl``/``torch``);
    ``NVTE_MXFP4_QAT_REQUIRE_KERNEL=1`` makes a kernel-less fallback a hard
    error instead of a warn-once (recommended in production so a stale wheel
    cannot silently train on the slow path).
    """
    if weight.dim() != 2:
        raise ValueError(f"MXFP4 QAT expects a 2D weight, got {tuple(weight.shape)}")
    if weight.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(
            f"MXFP4 QAT supports bf16/fp32 weights, got {weight.dtype} "
            "(fp16 cannot represent the full MXFP4 scale range)"
        )
    rows, cols = weight.shape
    if cols % _MXFP4_BLOCK != 0:
        raise ValueError(
            f"MXFP4 QAT needs the weight inner dim divisible by {_MXFP4_BLOCK}, got {cols}"
        )

    impl = _resolve_impl(weight)
    if torch.is_grad_enabled() and weight.requires_grad:
        return _MXFP4FakeQuantizeSTE.apply(weight, impl)
    return impl(weight)
