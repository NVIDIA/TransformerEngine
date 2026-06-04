# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fidelity tests for NVFP4 single-tensor RHT cast-fusion swizzled SF output.

Mirrors ``tests/pytorch/mxfp8/test_mxfp8_quantize_swizzle_fusion.py``: when
the quantizer has ``optimize_for_gemm=True`` and the input is eligible for
the RHT cast-fusion kernel (bf16, rows%64==0, cols%128==0, SM 100/110), the
kernel should emit scale factors directly in the GEMM-swizzled layout
``cuBLAS LT`` consumes, eliminating the otherwise-required
``nvte_swizzle_scaling_factors`` pass between quantize and GEMM.

The fidelity contract is: ``quantizer_swizzle_fusion(x)`` produces SF that
are byte-equal to ``swizzle_nvfp4_scale(quantizer(x).sx)``. The
``_rowwise_data`` / ``_columnwise_data`` quantized FP4 buffers and amaxes
must also be byte-equal (the swizzle optimization changes only the SF
layout, not the FP4 data itself).
"""

import transformer_engine.pytorch as te
import transformer_engine_torch as tex  # noqa: F401
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.tensor.storage.nvfp4_tensor_storage import (
    NVFP4TensorStorage,
)

import pytest
import torch

from typing import Tuple

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
    """Extract the six tensors we want to compare in byte-equal form.

    Returns ``(qx, sx, amax_row, qx_t, sx_t, amax_col)``, with any of these
    set to ``None`` when the quantizer did not request that direction.
    """
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


def _check_nvfp4_rht_quantize_swizzle_fusion(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    return_rowwise: bool,
    return_transpose: bool,
    with_random_sign_mask: bool,
) -> None:
    te_dtype = tex.DType.kFloat4E2M1

    device = "cuda"
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    x = torch.randn((M, N), dtype=x_dtype, device=device)

    # Reference quantizer (compact SF, default behavior).
    quantizer = NVFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=return_rowwise,
        columnwise=return_transpose,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=True,
        with_post_rht_amax=True,
        with_random_sign_mask=with_random_sign_mask,
    )

    # SUT: same quantizer but with swizzled-SF emission enabled.
    quantizer_swizzle_fusion = quantizer.copy()
    quantizer_swizzle_fusion.optimize_for_gemm = True

    qx_swf, sx_swf, amax_row_swf, qx_t_swf, sx_t_swf, amax_col_swf = _unpack_quantized_tensor(
        quantizer_swizzle_fusion(x)
    )
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
        # Full tile cases (rows%64==0 and cols%128==0 to be eligible for the
        # RHT cast-fusion kernel; also sized so no SF padding is needed).
        (128, 128),
        (256, 256),
        (1024, 256),
        # Production-like shapes.
        (2048, 2048),
        (8192, 1024),
        (8192, 5120),
        (8192, 10240),
        (16384, 8192),
        (16384, 16384),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize("quantize_mode", ["rowwise_only", "both_directions", "columnwise_only"])
@pytest.mark.parametrize(
    "with_random_sign_mask", [True, False], ids=["with_random_sign_mask", "no_random_sign_mask"]
)
def test_nvfp4_rht_quantize_swizzle_fusion(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    quantize_mode: str,
    with_random_sign_mask: bool,
) -> None:
    if quantize_mode == "rowwise_only":
        return_rowwise = True
        return_transpose = False
    elif quantize_mode == "both_directions":
        return_rowwise = True
        return_transpose = True
    elif quantize_mode == "columnwise_only":
        return_rowwise = False
        return_transpose = True
    else:
        raise ValueError(f"Invalid quantize mode: {quantize_mode}")

    _check_nvfp4_rht_quantize_swizzle_fusion(
        x_dtype=x_dtype,
        M=M,
        N=N,
        return_rowwise=return_rowwise,
        return_transpose=return_transpose,
        with_random_sign_mask=with_random_sign_mask,
    )


def _make_swizzle_fusion_quantizer() -> NVFP4Quantizer:
    """RHT NVFP4 quantizer configured to request GEMM-swizzled SF output."""
    quantizer = NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=True,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=True,
        with_post_rht_amax=True,
        with_random_sign_mask=True,
    )
    quantizer.optimize_for_gemm = True
    return quantizer


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, N, expected_swizzled",
    [
        # Eligible: rows%64==0 AND cols%128==0 -> the fused RHT cast-fusion
        # kernel can bake the GEMM-swizzled SF in directly.
        (64, 128, True),
        (128, 256, True),
        # Ineligible (cols%128 != 0) -> the fused kernel is gated off, so
        # create_tensor reports swizzled=False (the fused kernel did not bake
        # it in). This is the case hit in production at irregular shapes like
        # (8192, 11328) where 11328%128==64.
        (64, 144, False),
        (128, 144, False),
        # Ineligible (rows%64 != 0) -> same gating.
        (48, 128, False),
    ],
)
def test_nvfp4_rht_swizzle_fusion_shape_gate(M: int, N: int, expected_swizzled: bool) -> None:
    """The fused-kernel shape gate must reflect cast-fusion eligibility.

    Only ``row_cast_col_hadamard_transform_cast_fusion.cu`` can bake
    GEMM-swizzled SF directly, and only for eligible shapes (rows%64==0 AND
    cols%128==0 on SM 100/110). ``NVFP4Quantizer::create_tensor`` carries this
    gate in ``_with_gemm_swizzled_scales``: it must be True only when the fused
    kernel is selectable, and False on ineligible shapes so dispatch routes
    them to the unfused path (which is what keeps the defense-in-depth
    ``NVTE_CHECK`` in ``quantize_with_rht_unfused_helper`` from hard-aborting
    on irregular production shapes such as ``(8192, 11328)``).

    ``make_empty`` runs ``create_tensor`` only -- no quantize kernel, and in
    particular no post-quantize ``inplace_swizzle_scale_for_gemm`` fallback --
    so its ``_with_gemm_swizzled_scales`` observes the fused-kernel gate in
    isolation. (End-to-end ``quantizer(x)`` differs: the fallback then swizzles
    ineligible shapes too and flips the flag back to True; see
    ``test_nvfp4_rht_swizzle_fusion_end_to_end_swizzled``.)
    """
    quantizer = _make_swizzle_fusion_quantizer()
    out = quantizer.make_empty([M, N], dtype=torch.bfloat16)
    assert out._with_gemm_swizzled_scales is expected_swizzled, (
        f"Fused-kernel shape gate expected _with_gemm_swizzled_scales={expected_swizzled} "
        f"for shape ({M}, {N}) with optimize_for_gemm=True + with_rht=True, "
        f"got {out._with_gemm_swizzled_scales}"
    )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, N",
    [
        # Eligible shapes (fused cast-fusion kernel bakes swizzled SF in).
        (64, 128),
        (128, 256),
        # Ineligible shapes (cols%128 != 0 / rows%64 != 0): the fused kernel is
        # gated off, so they take the unfused quantize path and rely on the
        # post-quantize swizzle fallback. (8192, 11328)-class production shapes.
        (64, 144),
        (128, 144),
        (48, 128),
    ],
)
def test_nvfp4_rht_swizzle_fusion_end_to_end_swizzled(M: int, N: int) -> None:
    """End-to-end quantize must never crash and must always yield swizzled SF.

    With ``optimize_for_gemm=True`` the contract is that the produced tensor
    carries GEMM-swizzled SF regardless of shape, so the GEMM can skip its own
    swizzle pass. Eligible shapes get this from the fused cast-fusion kernel in
    one pass; ineligible shapes are gated off the fused kernel (avoiding the
    ``NVTE_CHECK`` hard-abort in ``quantize_with_rht_unfused_helper``) and
    instead get swizzled by the post-quantize ``inplace_swizzle_scale_for_gemm``
    fallback added in mainline #3076. Either way ``quantizer(x)`` must not raise
    and the final ``_with_gemm_swizzled_scales`` must be True.
    """
    quantizer = _make_swizzle_fusion_quantizer()
    x = torch.randn((M, N), dtype=torch.bfloat16, device="cuda")

    # Must not raise on ineligible shapes (gate keeps them off the fused
    # kernel; the unfused path + swizzle fallback run cleanly).
    result = quantizer(x)
    assert result._with_gemm_swizzled_scales is True, (
        f"End-to-end quantize expected _with_gemm_swizzled_scales=True for shape "
        f"({M}, {N}) with optimize_for_gemm=True + with_rht=True, "
        f"got {result._with_gemm_swizzled_scales}"
    )
