# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fidelity tests for NVFP4 (non-RHT) 2D-quantize swizzled SF output.

Mirrors ``test_nvfp4_rht_quantize_swizzle_fusion.py`` but for the plain
2D-block-scaling quantize kernel used by cached weights: when the quantizer
has ``optimize_for_gemm=True``, ``with_2d_quantization=True``, ``with_rht=False``
and the shape is 128-aligned (so no SF padding is needed), the quantize kernel
should emit scale factors directly in the GEMM-swizzled layout ``cuBLAS LT``
consumes, eliminating the otherwise-required ``nvte_swizzle_scaling_factors``
pass between quantize and GEMM.

The fidelity contract is: ``quantizer_swizzle_fusion(x)`` produces SF that are
byte-equal to ``swizzle_nvfp4_scale(quantizer(x).sx)``. The ``_rowwise_data`` /
``_columnwise_data`` quantized FP4 buffers and amaxes must also be byte-equal
(the swizzle optimization changes only the SF layout, not the FP4 data itself).
"""

from typing import Tuple

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex  # noqa: F401
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.tensor.storage.nvfp4_tensor_storage import (
    NVFP4TensorStorage,
)

from nvfp4_utils import (
    swizzle_nvfp4_scale,
    get_nvfp4_scale_shape_no_padding,
)

recipe_available, reason_for_no_recipe = te.is_nvfp4_available(return_reason=True)


def _unpack_quantized_tensor(
    quantized_tensor: NVFP4TensorStorage,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Extract ``(qx, sx, amax_row, qx_t, sx_t, amax_col)`` for byte comparison."""
    qx, sx, amax_row = None, None, None
    qx_t, sx_t, amax_col = None, None, None
    if quantized_tensor._rowwise_data is not None:
        qx = quantized_tensor._rowwise_data.view(dtype=torch.uint8)
    if quantized_tensor._rowwise_scale_inv is not None:
        sx = quantized_tensor._rowwise_scale_inv
    if quantized_tensor._amax_rowwise is not None:
        amax_row = quantized_tensor._amax_rowwise
    if quantized_tensor._columnwise_data is not None:
        qx_t = quantized_tensor._columnwise_data.view(dtype=torch.uint8)
    if quantized_tensor._columnwise_scale_inv is not None:
        sx_t = quantized_tensor._columnwise_scale_inv
    if quantized_tensor._amax_columnwise is not None:
        amax_col = quantized_tensor._amax_columnwise
    return qx, sx, amax_row, qx_t, sx_t, amax_col


def _make_quantizer(return_rowwise: bool, return_transpose: bool) -> NVFP4Quantizer:
    return NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=return_rowwise,
        columnwise=return_transpose,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=True,
    )


def _check_nvfp4_2d_quantize_swizzle_fusion(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    return_rowwise: bool,
    return_transpose: bool,
) -> None:
    device = "cuda"
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    x = torch.randn((M, N), dtype=x_dtype, device=device)

    # Reference quantizer: compact SF (default, no optimize_for_gemm).
    quantizer = _make_quantizer(return_rowwise, return_transpose)

    # SUT: same quantizer but with swizzled-SF emission enabled. The kernel must
    # bake the swizzled layout in directly (no standalone swizzle pass).
    quantizer_swizzle_fusion = quantizer.copy()
    quantizer_swizzle_fusion.optimize_for_gemm = True

    sut = quantizer_swizzle_fusion(x)
    assert sut._with_gemm_swizzled_scales, (
        f"expected in-kernel swizzled SF for aligned shape ({M}, {N}); "
        "the 2D quantize kernel should have baked the swizzled layout in"
    )

    qx_swf, sx_swf, amax_row_swf, qx_t_swf, sx_t_swf, amax_col_swf = _unpack_quantized_tensor(sut)
    qx_ref, sx_ref, amax_row_ref, qx_t_ref, sx_t_ref, amax_col_ref = _unpack_quantized_tensor(
        quantizer(x)
    )

    if return_rowwise:
        # FP4 data buffer and amax must be byte-equal (swizzle only changes SF
        # layout, not the quantized data).
        torch.testing.assert_close(qx_swf, qx_ref, atol=0.0, rtol=0.0)
        torch.testing.assert_close(amax_row_swf, amax_row_ref, atol=0.0, rtol=0.0)

        # SF tensor must match the swizzle of the reference compact SF.
        valid_scale_shape = get_nvfp4_scale_shape_no_padding(x.shape, False)
        assert valid_scale_shape == sx_swf.shape, (
            "rowwise SF shape mismatch; this test assumes the input shape needs no "
            f"SF padding (got valid={valid_scale_shape}, got_swf={sx_swf.shape})."
        )
        sx_ref_swizzled = swizzle_nvfp4_scale(M, N, sx_ref, columnwise=False)
        torch.testing.assert_close(sx_swf, sx_ref_swizzled, atol=0.0, rtol=0.0)

    if return_transpose:
        torch.testing.assert_close(qx_t_swf, qx_t_ref, atol=0.0, rtol=0.0)
        torch.testing.assert_close(amax_col_swf, amax_col_ref, atol=0.0, rtol=0.0)

        valid_scale_shape = get_nvfp4_scale_shape_no_padding(x.shape, True)
        assert valid_scale_shape == sx_t_swf.shape, (
            "columnwise SF shape mismatch; this test assumes the input shape needs no "
            f"SF padding (got valid={valid_scale_shape}, got_swf={sx_t_swf.shape})."
        )
        sx_t_ref_swizzled = swizzle_nvfp4_scale(M, N, sx_t_ref, columnwise=True)
        torch.testing.assert_close(sx_t_swf, sx_t_ref_swizzled, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, N",
    [
        # 128-aligned shapes (no SF padding needed) -> in-kernel swizzle eligible.
        (128, 128),
        (256, 256),
        (1024, 1024),
        (2048, 2048),
        (8192, 1024),
        (8192, 5120),
        (8192, 10240),
        (16384, 8192),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize("quantize_mode", ["rowwise_only", "both_directions", "columnwise_only"])
def test_nvfp4_2d_quantize_swizzle_fusion(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    quantize_mode: str,
) -> None:
    if quantize_mode == "rowwise_only":
        return_rowwise, return_transpose = True, False
    elif quantize_mode == "both_directions":
        return_rowwise, return_transpose = True, True
    elif quantize_mode == "columnwise_only":
        return_rowwise, return_transpose = False, True
    else:
        raise ValueError(f"Invalid quantize mode: {quantize_mode}")

    _check_nvfp4_2d_quantize_swizzle_fusion(
        x_dtype=x_dtype,
        M=M,
        N=N,
        return_rowwise=return_rowwise,
        return_transpose=return_transpose,
    )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, N, expected_swizzled",
    [
        # Eligible: both dims % 128 == 0 -> 2D quantize kernel bakes swizzled SF.
        (128, 128, True),
        (256, 512, True),
        # Ineligible (cols % 128 != 0) -> kernel emits compact SF; create_tensor
        # reports swizzled=False (the fused kernel did not bake it in).
        (128, 192, False),
        # Ineligible (rows % 128 != 0).
        (64, 128, False),
    ],
)
def test_nvfp4_2d_swizzle_fusion_shape_gate(M: int, N: int, expected_swizzled: bool) -> None:
    """``create_tensor`` must report in-kernel swizzle only for 128-aligned shapes.

    ``make_empty`` runs ``create_tensor`` only (no quantize kernel, no
    post-quantize ``inplace_swizzle_scale_for_gemm`` fallback), so its
    ``_with_gemm_swizzled_scales`` observes the in-kernel gate in isolation.
    """
    quantizer = _make_quantizer(return_rowwise=True, return_transpose=True)
    quantizer.optimize_for_gemm = True
    out = quantizer.make_empty([M, N], dtype=torch.bfloat16)
    assert out._with_gemm_swizzled_scales is expected_swizzled, (
        f"2D-kernel shape gate expected _with_gemm_swizzled_scales={expected_swizzled} "
        f"for shape ({M}, {N}), got {out._with_gemm_swizzled_scales}"
    )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, N",
    [
        (128, 128),
        (256, 512),
        # Ineligible shapes still end up swizzled via the post-quantize fallback.
        (128, 192),
        (64, 128),
    ],
)
def test_nvfp4_2d_swizzle_fusion_end_to_end_swizzled(M: int, N: int) -> None:
    """End-to-end quantize must never crash and must always yield swizzled SF.

    Eligible (128-aligned) shapes get swizzled SF directly from the 2D quantize
    kernel; ineligible shapes are gated off the in-kernel path and get swizzled
    by the post-quantize ``inplace_swizzle_scale_for_gemm`` fallback. Either way
    the final ``_with_gemm_swizzled_scales`` must be True.
    """
    quantizer = _make_quantizer(return_rowwise=True, return_transpose=True)
    quantizer.optimize_for_gemm = True
    x = torch.randn((M, N), dtype=torch.bfloat16, device="cuda")

    result = quantizer(x)
    assert result._with_gemm_swizzled_scales is True, (
        "End-to-end quantize expected _with_gemm_swizzled_scales=True for shape "
        f"({M}, {N}) with optimize_for_gemm=True + with_2d_quantization=True, "
        f"got {result._with_gemm_swizzled_scales}"
    )
