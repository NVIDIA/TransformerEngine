# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MXFP4 weight fake-quantization for quantization-aware training.

Projects weights onto the MXFP4 (E2M1) grid with 1x32 power-of-two block
scales and returns the dequantized values in the input dtype. The output is
exact in bf16/fp32, and the host recipe's weight quantization (MXFP8 rowwise;
128x128 blockwise within the tile scale-spread bound) is DECODED-VALUE exact
on this finite QAT domain. It is not raw-tuple identical to a direct
MXFP4->MXFP8 converter: the host encoder re-canonicalizes (payload, scale)
per block, so equal values may carry different E4M3/UE8M0 bytes, and the
original MXFP4 scale metadata is not preserved. Scale floor 2^-126 / cap 2^125
(TileKernels deployment contract); non-finite values NaN-poison their 1x32
block; fp16 is rejected. Implementation picked by NVTE_MXFP4_QAT_IMPL
(auto/cuda/cute_dsl/torch).
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
    """RTNE onto the E2M1 magnitude grid {0, .5, 1, 1.5, 2, 3, 4, 6}; input in [0, 6]."""
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

    Per 1x32 block: scale = 2^clamp(ceil(log2(amax / 6)), -126, 125), RTNE onto
    E2M1, saturate at 6, multiply back. Gradient is the straight-through
    estimator. ``NVTE_MXFP4_QAT_REQUIRE_KERNEL=1`` forbids the torch fallback.
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
