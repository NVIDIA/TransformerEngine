# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Bit-exact tests for the NVFP4 hierarchical 1x64 cast CUDA kernel.

Methodology mirrors ``test_nvfp4_quantize_exact.py``: invoke the kernel
through ``NVFP4Quantizer`` (with the ``NVTE_NVFP4_ROWWISE_1X64_LOCAL_ENCODE=1``
env flag enabling the alternate dispatch), and compare every output buffer
against a pure-PyTorch oracle (``NVFP4Quantizer1x64Ref``) byte-for-byte
(``atol=rtol=0``).

The kernel writes up to six tensors per call:

* rowwise FP4 data, rowwise per-1x16 E4M3 ``s_dec``, rowwise per-1x64
  window amax;
* columnwise (transposed) FP4 data, columnwise per-1x16 E4M3 ``s_dec``,
  columnwise per-1x64 window amax.

This file covers the rowwise-only, columnwise-only, and rowwise+columnwise
configurations -- the latter being the production-equivalent fused mode.
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
    """Unpack two FP4 values per byte into one ``uint8`` value per element."""
    repeated = x.repeat_interleave(2, dim=1)
    repeated[:, 0::2] &= 0x0F
    repeated[:, 1::2] >>= 4
    return repeated


def _check_quantization_1x64_versus_reference_with_input(
    x: torch.Tensor,
    *,
    rowwise: bool,
    columnwise: bool,
) -> None:
    """Quantize ``x`` through both kernel and reference and assert that every
    requested output (data, scale, window-amax, on each requested direction)
    matches bit-exactly.
    """
    te_dtype = tex.DType.kFloat4E2M1

    quantizer = NVFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=rowwise,
        columnwise=columnwise,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=False,
    )
    sut = quantizer(x)
    ref = NVFP4Quantizer1x64Ref(rowwise=rowwise, columnwise=columnwise).quantize(x)

    if rowwise:
        qx_sut = unpack_fp4(sut._rowwise_data.view(dtype=torch.uint8))
        qx_ref = unpack_fp4(ref.data.view(dtype=torch.uint8))
        sx_sut = sut._rowwise_scale_inv.view(dtype=torch.uint8)
        sx_ref = ref.scale.view(dtype=torch.uint8)
        # The kernel may pad qx/sx to alignment boundaries; only the
        # un-padded prefix is meaningfully written.
        torch.testing.assert_close(
            qx_sut[: qx_ref.shape[0], : qx_ref.shape[1]], qx_ref, atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            sx_sut[: sx_ref.shape[0], : sx_ref.shape[1]], sx_ref, atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(sut._amax_rowwise, ref.window_amax_row, atol=0.0, rtol=0.0)

    if columnwise:
        qxt_sut = unpack_fp4(sut._columnwise_data.view(dtype=torch.uint8))
        qxt_ref = unpack_fp4(ref.columnwise_data.view(dtype=torch.uint8))
        sxt_sut = sut._columnwise_scale_inv.view(dtype=torch.uint8)
        sxt_ref = ref.columnwise_scale.view(dtype=torch.uint8)
        torch.testing.assert_close(
            qxt_sut[: qxt_ref.shape[0], : qxt_ref.shape[1]], qxt_ref, atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            sxt_sut[: sxt_ref.shape[0], : sxt_ref.shape[1]], sxt_ref, atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            sut._amax_columnwise, ref.window_amax_col, atol=0.0, rtol=0.0
        )


def _check_random(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    *,
    rowwise: bool,
    columnwise: bool,
) -> None:
    """Random-input variant. Seeds are fixed so failures reproduce."""
    device = "cuda"
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    x = torch.randn((M, N), dtype=x_dtype, device=device)
    _check_quantization_1x64_versus_reference_with_input(
        x, rowwise=rowwise, columnwise=columnwise
    )


# Shapes where both M and N are multiples of 64 -- the 1x64 hierarchy's
# strict alignment requirement (enforced by the dispatcher).
_SHAPES_64x64_MULTIPLE = [
    (64, 64),
    (128, 128),
    (256, 256),
    (256, 512),
    (1024, 256),
    (256, 1024),
    (2048, 2048),
    (1024, 2048),
    # Non-square shapes that exercise distinct row/col tile counts -- the
    # columnwise pass uses M/64 windows, the rowwise uses N/64.
    (64, 256),
    (256, 64),
    (192, 384),
    (384, 192),
]


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("M, N", _SHAPES_64x64_MULTIPLE)
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
def test_nvfp4_1x64_quantize_rowwise(
    monkeypatch, x_dtype: torch.dtype, M: int, N: int
) -> None:
    """Rowwise-only configuration -- preserves the original PR's coverage."""
    monkeypatch.setenv("NVTE_NVFP4_ROWWISE_1X64_LOCAL_ENCODE", "1")
    monkeypatch.setenv("NVTE_NVFP4_DISABLE_RHT", "1")
    _check_random(x_dtype, M, N, rowwise=True, columnwise=False)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("M, N", _SHAPES_64x64_MULTIPLE)
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
def test_nvfp4_1x64_quantize_columnwise(
    monkeypatch, x_dtype: torch.dtype, M: int, N: int
) -> None:
    """Columnwise-only -- exercises the transposed output path on its own."""
    monkeypatch.setenv("NVTE_NVFP4_ROWWISE_1X64_LOCAL_ENCODE", "1")
    monkeypatch.setenv("NVTE_NVFP4_DISABLE_RHT", "1")
    _check_random(x_dtype, M, N, rowwise=False, columnwise=True)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("M, N", _SHAPES_64x64_MULTIPLE)
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
def test_nvfp4_1x64_quantize_rowwise_columnwise(
    monkeypatch, x_dtype: torch.dtype, M: int, N: int
) -> None:
    """Fused rowwise+columnwise -- the production-equivalent configuration."""
    monkeypatch.setenv("NVTE_NVFP4_ROWWISE_1X64_LOCAL_ENCODE", "1")
    monkeypatch.setenv("NVTE_NVFP4_DISABLE_RHT", "1")
    _check_random(x_dtype, M, N, rowwise=True, columnwise=True)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize(
    "rowwise, columnwise",
    [(True, False), (False, True), (True, True)],
    ids=["row", "col", "rowcol"],
)
def test_nvfp4_1x64_quantize_extrema(
    monkeypatch, x_dtype: torch.dtype, rowwise: bool, columnwise: bool
) -> None:
    """Stress the saturating cast and ``S_enc`` clamps on each direction.

    Inputs:
      * an outlier-heavy row (one ~FP4_MAX value per 1x64 K-window plus tiny
        noise) -- exercises the rowwise ``s_dec`` saturation branch;
      * an outlier-heavy column (one ~FP4_MAX value per 1x64 M-window plus
        tiny noise) -- the columnwise mirror of the above;
      * an all-zero region -- exercises the ``tile_amax == 0`` fallback that
        promotes ``S_enc`` to 1.0 and the kernel's ``s_dec == 0``
        short-circuit (without which ``cvt.rn.satfinite.e2m1x4.f32(NaN)``
        on SM10 would saturate to ``+FP4_MAX = 0x7`` instead of ``0x0``);
      * a uniform constant block -- ``s_dec`` should be the same E4M3 byte
        for every block in the affected window.
    """
    monkeypatch.setenv("NVTE_NVFP4_ROWWISE_1X64_LOCAL_ENCODE", "1")
    monkeypatch.setenv("NVTE_NVFP4_DISABLE_RHT", "1")

    device = "cuda"
    M, N = 128, 256

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    x = torch.randn((M, N), dtype=x_dtype, device=device) * 0.01

    # Row 0: per-1x64-K-window outlier on the rowwise side.
    for w in range(N // 64):
        x[0, w * 64] = 5.5

    # Col 0: per-1x64-M-window outlier on the columnwise side. Distinct row
    # from row 0 so the two outlier patterns do not overlap.
    for w in range(M // 64):
        x[w * 64 + 1, 0] = 5.25  # row index != 0, col 0

    # Row 1: all zero (degenerate window amax path on the rowwise side).
    # We've already touched (1, 0) above so re-zero just (1, 1:) -- the
    # block at (1, 0..15) still contains a single non-zero outlier 5.25,
    # which is fine; the all-zero stress remains in (1, 16:).
    x[1, 16:] = 0.0

    # Col 1: all zero (degenerate window amax path on the columnwise side).
    x[16:, 1] = 0.0

    # Row 2: uniform constant -- exercises both directions' constant-block
    # path simultaneously for that row's columns it spans.
    x[2] = 0.75

    _check_quantization_1x64_versus_reference_with_input(
        x, rowwise=rowwise, columnwise=columnwise
    )
