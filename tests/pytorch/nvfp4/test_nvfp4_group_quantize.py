# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# NOTE: This file is dependent on the success of test_nvfp4_quantize_exact.py
# and also the test_nvfp4_rht_quantize_exact.py.
# Separate to make sure all the functionalities are working as expected.
# Otherwise reference implementation will get messy.

# Due to the structure of NVFP4Quantizer, we need to test the RHT functionality
# together with the quantization functionality.

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.custom_recipes.quantization_ref_nvfp4 import NVFP4QuantizerRef
from transformer_engine.pytorch.custom_recipes import utils
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.common.recipe import NVFP4BlockScaling

import pytest
import torch
import random
import math

from nvfp4_utils import (
    get_nvfp4_scale_shape_no_padding,
    generate_split_sections,
    assert_same_shape_and_dtype,
    reference_group_quantize,
    swizzle_nvfp4_scale,
)

recipe_available, reason_for_no_recipe = te.is_nvfp4_available(return_reason=True)


def check_group_quantization_nvfp4_versus_reference(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    return_rowwise: bool,
    return_transpose: bool,
    split_sections: list[int],
    with_rht: bool = True,
    with_post_rht_amax: bool = True,
    with_random_sign_mask: bool = True,
    optimize_for_gemm: bool = False,
) -> None:

    te_dtype = tex.DType.kFloat4E2M1

    # Setup device and random seed
    device = "cuda"
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Input
    x = torch.randn((M, N), dtype=x_dtype, device=device)
    num_chunks = len(split_sections)

    x_splits = torch.split(x, split_sections)

    # Reference quantizers (compact SF, default optimize_for_gemm=False).
    quantizers = [
        NVFP4Quantizer(
            fp4_dtype=te_dtype,
            rowwise=return_rowwise,
            columnwise=return_transpose,
            with_amax_reduction=False,
            amax_reduction_group=None,
            with_rht=with_rht,
            with_post_rht_amax=with_post_rht_amax,
            with_random_sign_mask=with_random_sign_mask,
        )
        for _ in range(len(split_sections))
    ]
    x_qx_ref, x_sx_ref, x_amax_rowwise_ref, x_qx_t_ref, x_sx_t_ref, x_amax_colwise_ref = (
        reference_group_quantize(x, quantizers, split_sections, return_rowwise, return_transpose)
    )

    # SUT quantizers: same as reference, but with optimize_for_gemm toggled to
    # request direct swizzled SF emission from the RHT cast-fusion kernel.
    sut_quantizers = [q.copy() for q in quantizers]
    for q in sut_quantizers:
        q.optimize_for_gemm = optimize_for_gemm

    split_quantize_outputs = tex.split_quantize(x, split_sections, sut_quantizers)

    if return_rowwise:
        x_qx = [output._rowwise_data.view(dtype=torch.uint8) for output in split_quantize_outputs]
        x_sx = [output._rowwise_scale_inv for output in split_quantize_outputs]
        x_amax_rowwise = [output._amax_rowwise for output in split_quantize_outputs]

        for i in range(len(x_qx)):
            if split_sections[i] == 0:
                # then just assert the same shape and dtype because the buffer won't be zero out
                assert_same_shape_and_dtype(x_amax_rowwise[i], x_amax_rowwise_ref[i])
                assert_same_shape_and_dtype(x_qx[i], x_qx_ref[i])
                assert_same_shape_and_dtype(x_sx[i], x_sx_ref[i])
            else:
                torch.testing.assert_close(
                    x_amax_rowwise[i], x_amax_rowwise_ref[i], atol=0.0, rtol=0.0
                )
                torch.testing.assert_close(x_qx[i], x_qx_ref[i], atol=0.0, rtol=0.0)
                valid_scale_shape = get_nvfp4_scale_shape_no_padding(x_splits[i].shape, False)
                x_sx_valid = x_sx[i][: valid_scale_shape[0], : valid_scale_shape[1]]
                x_sx_ref_valid = x_sx_ref[i][: valid_scale_shape[0], : valid_scale_shape[1]]
                if optimize_for_gemm:
                    # SUT emits SF in the GEMM-swizzled layout directly; swizzle
                    # the reference compact SF for byte-equal comparison.
                    x_sx_ref_valid = swizzle_nvfp4_scale(
                        split_sections[i], N, x_sx_ref_valid, columnwise=False
                    )
                torch.testing.assert_close(x_sx_valid, x_sx_ref_valid, atol=0.0, rtol=0.0)

    if return_transpose:
        x_qx_t = [
            output._columnwise_data.view(dtype=torch.uint8) for output in split_quantize_outputs
        ]
        x_sx_t = [output._columnwise_scale_inv for output in split_quantize_outputs]
        x_amax_colwise = [output._amax_columnwise for output in split_quantize_outputs]
        # assert with zero tolerance
        for i in range(len(x_qx_t)):
            if split_sections[i] == 0:
                # then just assert the same shape and dtype because the buffer won't be zero out
                assert_same_shape_and_dtype(x_amax_colwise[i], x_amax_colwise_ref[i])
                assert_same_shape_and_dtype(x_qx_t[i], x_qx_t_ref[i])
                assert_same_shape_and_dtype(x_sx_t[i], x_sx_t_ref[i])
            else:
                torch.testing.assert_close(
                    x_amax_colwise[i], x_amax_colwise_ref[i], atol=0.0, rtol=0.0
                )
                torch.testing.assert_close(x_qx_t[i], x_qx_t_ref[i], atol=0.0, rtol=0.0)
                valid_scale_shape = get_nvfp4_scale_shape_no_padding(x_splits[i].shape, True)
                x_sx_t_valid = x_sx_t[i][: valid_scale_shape[0], : valid_scale_shape[1]]
                x_sx_t_ref_valid = x_sx_t_ref[i][: valid_scale_shape[0], : valid_scale_shape[1]]
                if optimize_for_gemm:
                    x_sx_t_ref_valid = swizzle_nvfp4_scale(
                        split_sections[i], N, x_sx_t_ref_valid, columnwise=True
                    )
                torch.testing.assert_close(x_sx_t_valid, x_sx_t_ref_valid, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, N",
    [
        # edge case, zero tokens for all
        (0, 512),
        # edge case, not 128 multiple hidden dimension
        (1024, 320),
        # full tile cases
        (256, 1024),
        (1024, 256),
        # larger sizes
        (8192, 1024),
        (16384, 8192),
        (16384, 16384),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize(
    "edge_cases",
    [
        "regular",
        "zero_tokens_front",
        "zero_tokens_end",
        "zero_tokens_middle",
        "random_uneven_split",
    ],
)
@pytest.mark.parametrize("quantize_mode", ["rowwise_only", "both_directions", "columnwise_only"])
@pytest.mark.parametrize(
    "with_random_sign_mask", [True, False], ids=["with_random_sign_mask", "no_random_sign_mask"]
)
@pytest.mark.parametrize("with_rht", [True, False], ids=["with_rht", "no_rht"])
@pytest.mark.parametrize(
    "optimize_for_gemm",
    [False, True],
    ids=["compact_sf", "swizzled_sf"],
)
def test_rht_with_quantization_block_tiling_versus_reference(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    edge_cases: str,
    quantize_mode: str,
    with_random_sign_mask: bool,
    with_rht: bool,
    optimize_for_gemm: bool,
) -> None:

    # The "quantize writes swizzled SF" fast-path is gated in the C++ framework
    # on ``optimize_for_gemm && with_rht`` (see NVFP4Quantizer::create_tensor and
    # bulk_allocate_nvfp4_tensors in transformer_engine/pytorch/csrc). Without
    # ``with_rht=True`` the flag is silently dropped, so the swizzled_sf row
    # would just duplicate the compact_sf row — skip it instead of flooding the
    # matrix with redundant cases.
    if optimize_for_gemm and not with_rht:
        pytest.skip("optimize_for_gemm requires with_rht=True (framework gate)")

    # The grouped RHT cast-fusion kernel that honors with_gemm_swizzled_scales
    # (group_row_cast_col_hadamard_transform_cast_fusion.cu) is only dispatched
    # when:
    #   - cols are a 128 multiple (RHT cast-fusion eligibility), AND
    #   - every split section is a 128 multiple (all_aligned_token_dim path in
    #     split_quantize_nvfp4_impl_with_rht_helper).
    # For other shapes the C++ side falls back to the unfused row/col split,
    # which does NOT (yet) emit swizzled SF; we'd hit either an NVTE_CHECK or
    # silent SF-layout corruption. Restrict the swizzled coverage to the fused
    # path; the unfused fallback is covered by the optimize_for_gemm=False
    # baseline already exercised above.
    if optimize_for_gemm and N % 128 != 0:
        pytest.skip("RHT cast-fusion requires N % 128 == 0")

    # generate_split_sections hard-codes num_chunks=4 and requires every chunk
    # to be a least_multiple multiple. When optimize_for_gemm forces
    # least_multiple from 64 to 128, the test needs M >= 4*128 = 512 (and a
    # multiple of 512 for the regular/zero_tokens patterns, which the existing
    # M shapes already satisfy: 0, 1024, 8192, 16384). The small M=256 shape
    # cannot satisfy this and is exercised by the optimize_for_gemm=False rows.
    if optimize_for_gemm and 0 < M < 4 * 128:
        pytest.skip("optimize_for_gemm requires M==0 or M>=512 for 4-chunk 128-aligned split")

    least_multiple = 128 if optimize_for_gemm else 64
    split_sections = generate_split_sections(M, N, edge_cases, least_multiple=least_multiple)

    # currently disable pre-RHT amax
    with_post_rht_amax = with_rht

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

    check_group_quantization_nvfp4_versus_reference(
        x_dtype=x_dtype,
        M=M,
        N=N,
        return_rowwise=return_rowwise,
        return_transpose=return_transpose,
        split_sections=split_sections,
        with_rht=with_rht,
        with_post_rht_amax=with_post_rht_amax,
        with_random_sign_mask=with_random_sign_mask,
        optimize_for_gemm=optimize_for_gemm,
    )
