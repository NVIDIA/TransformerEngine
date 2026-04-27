# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Bit-exact tests for the NVFP4 rowwise 1x64 local-encode CUDA kernel.

Methodology mirrors ``test_nvfp4_quantize_exact.py``: invoke the kernel
through ``NVFP4Quantizer`` (with the ``NVTE_NVFP4_ROWWISE_1X64_LOCAL_ENCODE=1``
env flag enabling the alternate dispatch), and compare the resulting
``(qx, sx, amax)`` triple against the pure-PyTorch oracle
``NVFP4Quantizer1x64Ref`` byte-for-byte (``atol=rtol=0``).
"""

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.custom_recipes.quantization_nvfp4_1x64 import (
    NVFP4Quantizer1x64Ref,
)


recipe_available, reason_for_no_recipe = te.is_nvfp4_available(return_reason=True)


def unpack_fp4(x: torch.Tensor) -> torch.Tensor:
    """Unpack two FP4 values per byte into one ``uint8`` value per element.

    Identical to the helper in ``test_nvfp4_quantize_exact.py`` -- duplicated
    here so the two test suites stay independent.
    """
    repeated = x.repeat_interleave(2, dim=1)
    repeated[:, 0::2] &= 0x0F
    repeated[:, 1::2] >>= 4
    return repeated


def _check_quantization_1x64_versus_reference(
    x_dtype: torch.dtype,
    M: int,
    N: int,
) -> None:
    """Quantize ``(M, N)`` random input through both kernel and reference and
    assert the rowwise data, scale, and global amax all match bit-exactly."""
    te_dtype = tex.DType.kFloat4E2M1
    device = "cuda"

    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    x = torch.randn((M, N), dtype=x_dtype, device=device)

    # Kernel path. ``with_rht=False`` and ``with_2d_quantization=False`` are
    # required by the 1x64 dispatch's preconditions; ``columnwise=False``
    # because the kernel does not produce a transposed output.
    quantizer = NVFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=False,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=False,
    )
    x_nvfp4_sut = quantizer(x)

    qx = x_nvfp4_sut._rowwise_data.view(dtype=torch.uint8)
    sx = x_nvfp4_sut._rowwise_scale_inv
    qx_amax = x_nvfp4_sut._amax_rowwise

    # Reference path.
    ref = NVFP4Quantizer1x64Ref().quantize(x)
    qx_ref = unpack_fp4(ref.data.view(dtype=torch.uint8))
    sx_ref = ref.scale.view(dtype=torch.uint8)
    ref_amax = ref.global_amax_row

    qx = unpack_fp4(qx)

    # The kernel may pad qx/sx to alignment boundaries; only the unpadded
    # prefix is meaningfully written. Slice both sides down to the reference
    # shape before comparing (the reference returns un-padded tensors).
    qx_valid = qx[: qx_ref.shape[0], : qx_ref.shape[1]]
    sx_valid = sx[: sx_ref.shape[0], : sx_ref.shape[1]]

    torch.testing.assert_close(qx_valid, qx_ref, atol=0.0, rtol=0.0)
    torch.testing.assert_close(sx_valid, sx_ref, atol=0.0, rtol=0.0)
    torch.testing.assert_close(qx_amax, ref_amax, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, N",
    [
        # K is a multiple of WINDOW_K=64
        (128, 128),
        (256, 256),
        (256, 512),
        (1024, 256),
        (256, 1024),
        (2048, 2048),
        (1024, 2048),
        # K is a multiple of BLOCK_K=16 but not of WINDOW_K -- exercises the
        # partial-last-window path in the kernel.
        (256, 80),
        (256, 272),
        (256, 336),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
def test_nvfp4_1x64_quantize_versus_reference(
    monkeypatch,
    x_dtype: torch.dtype,
    M: int,
    N: int,
) -> None:
    """Random-input bit-exact test across a representative shape grid."""
    monkeypatch.setenv("NVTE_NVFP4_ROWWISE_1X64_LOCAL_ENCODE", "1")
    monkeypatch.setenv("NVTE_NVFP4_DISABLE_RHT", "1")
    _check_quantization_1x64_versus_reference(x_dtype, M, N)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
def test_nvfp4_1x64_quantize_extrema(monkeypatch, x_dtype: torch.dtype) -> None:
    """Stress the saturating-cast and ``S_enc`` clamps with extreme inputs.

    Inputs:
      * an outlier-heavy row (one ~FP4_MAX value per 1x64 window plus tiny
        noise) -- exercises the ``s_dec`` saturation branch;
      * an all-zero region -- exercises the ``tile_amax == 0`` fallback that
        promotes ``S_enc`` to 1.0;
      * a uniform constant row -- ``s_dec`` should be the same E4M3 byte for
        every block in the row.
    """
    monkeypatch.setenv("NVTE_NVFP4_ROWWISE_1X64_LOCAL_ENCODE", "1")
    monkeypatch.setenv("NVTE_NVFP4_DISABLE_RHT", "1")

    device = "cuda"
    M, N = 64, 256

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    x = torch.randn((M, N), dtype=x_dtype, device=device) * 0.01

    # Row 0: per-window outliers.
    for w in range(N // 64):
        x[0, w * 64] = 5.5

    # Row 1: all zero (degenerate window amax path).
    x[1] = 0.0

    # Row 2: uniform constant.
    x[2] = 0.75

    _check_quantization_1x64_versus_reference_with_input(x)


def _check_quantization_1x64_versus_reference_with_input(x: torch.Tensor) -> None:
    """Variant of the random-input checker that consumes a caller-provided
    tensor (used by the extrema test)."""
    te_dtype = tex.DType.kFloat4E2M1

    quantizer = NVFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=False,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=False,
    )
    x_nvfp4_sut = quantizer(x)

    qx = x_nvfp4_sut._rowwise_data.view(dtype=torch.uint8)
    sx = x_nvfp4_sut._rowwise_scale_inv
    qx_amax = x_nvfp4_sut._amax_rowwise

    ref = NVFP4Quantizer1x64Ref().quantize(x)
    qx_ref = unpack_fp4(ref.data.view(dtype=torch.uint8))
    sx_ref = ref.scale.view(dtype=torch.uint8)
    ref_amax = ref.global_amax_row

    qx = unpack_fp4(qx)

    qx_valid = qx[: qx_ref.shape[0], : qx_ref.shape[1]]
    sx_valid = sx[: sx_ref.shape[0], : sx_ref.shape[1]]

    torch.testing.assert_close(qx_valid, qx_ref, atol=0.0, rtol=0.0)
    torch.testing.assert_close(sx_valid, sx_ref, atol=0.0, rtol=0.0)
    torch.testing.assert_close(qx_amax, ref_amax, atol=0.0, rtol=0.0)
