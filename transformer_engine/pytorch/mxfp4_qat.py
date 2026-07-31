# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MXFP4 weight fake-quantization for quantization-aware training.

Projects weights onto the MXFP4 (E2M1) grid with 1x32 power-of-two block
scales and returns the dequantized values in the input dtype. The output is
exact in bf16/fp32, and the host recipe's weight quantization (MXFP8 rowwise;
1x128 or 128x128 blockwise within the block scale-spread bound) is
DECODED-VALUE exact on this finite QAT domain. It is not raw-tuple identical to a direct
MXFP4->MXFP8 converter: the host encoder re-canonicalizes (payload, scale)
per block, so equal values may carry different E4M3/UE8M0 bytes, and the
original MXFP4 scale metadata is not preserved. Scale floor 2^-126 / cap 2^125
(deployment contract); non-finite values NaN-poison their 1x32 block; fp16 is
rejected. The numerical specification lives in
tests/pytorch/references/mxfp4_qat_reference.py.
"""
import torch

__all__ = ["mxfp4_fake_quantize"]

_MXFP4_BLOCK = 32


class _MXFP4FakeQuantizeSTE(torch.autograd.Function):
    """Straight-through estimator: identity gradient through the projection."""

    @staticmethod
    def forward(ctx, weight):  # pylint: disable=arguments-differ
        import transformer_engine_torch as tex

        if not hasattr(tex, "mxfp4_fake_quantize"):
            raise RuntimeError(
                "transformer_engine_torch was built without mxfp4_fake_quantize; "
                "rebuild Transformer Engine."
            )
        w = weight.contiguous()
        if w.data_ptr() % 16 != 0:
            # logically-contiguous storage-offset views can be misaligned for
            # the kernel's 16-byte vector accesses
            w = w.clone()
        return tex.mxfp4_fake_quantize(w)

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ
        return grad_output


def mxfp4_fake_quantize(weight: torch.Tensor) -> torch.Tensor:
    """Project ``weight`` onto the MXFP4 grid and return it in the input dtype.

    Per 1x32 block: scale = 2^clamp(ceil(log2(amax / 6)), -126, 125), RTNE onto
    E2M1, saturate at 6, multiply back. The gradient is the straight-through
    estimator (identity).
    """
    if not weight.is_cuda:
        raise ValueError("MXFP4 QAT expects a CUDA weight tensor")
    if weight.dim() != 2:
        raise ValueError(f"MXFP4 QAT expects a 2D weight, got {tuple(weight.shape)}")
    if weight.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(
            f"MXFP4 QAT supports bf16/fp32 weights, got {weight.dtype} "
            "(fp16 cannot represent the full MXFP4 scale range)"
        )
    if weight.shape[1] % _MXFP4_BLOCK != 0:
        raise ValueError(
            "MXFP4 QAT needs the weight inner dim divisible by "
            f"{_MXFP4_BLOCK}, got {weight.shape[1]}"
        )
    return _MXFP4FakeQuantizeSTE.apply(weight)
