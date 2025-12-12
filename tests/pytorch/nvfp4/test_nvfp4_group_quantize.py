# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from transformer_engine.pytorch.custom_recipes.quantization_nvfp4 import NVFP4QuantizerRef
from transformer_engine.pytorch.custom_recipes import utils
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.common.recipe import NVFP4BlockScaling

import pytest
import torch
import random
import math

recipe_available, reason_for_no_recipe = te.is_nvfp4_available(return_reason=True)


def generate_random_multiples_sum(total=8192, n=4, multiple=64):
    if total % multiple != 0:
        raise ValueError(f"Total ({total}) must be a multiple of {multiple}")
    if (total // multiple) < n:
        raise ValueError("Total too small for given n and multiple.")

    # Work in units of multiples
    total_units = total // multiple

    # choose n−1 random cut points in [1, total_units−1)
    cuts = sorted(random.sample(range(1, total_units), n - 1))

    # convert to segment lengths
    parts = (
        [cuts[0]] + [cuts[i] - cuts[i - 1] for i in range(1, len(cuts))] + [total_units - cuts[-1]]
    )

    # convert back to multiples
    return [p * multiple for p in parts]


def generate_split_sections(M: int, N: int, edge_cases: str) -> list[int]:
    least_multiple = 64
    num_chunks = 4
    split_sections = None

    avg_split = M // num_chunks

    if M == 0 or N == 0:
        # all zeros
        return [0] * num_chunks
    if edge_cases == "regular":
        split_sections = [avg_split] * num_chunks
    elif edge_cases == "zero_tokens_front":
        split_sections = [0] + [avg_split] * (num_chunks - 2) + [avg_split * 2]
    elif edge_cases == "zero_tokens_end":
        split_sections = [avg_split * 2] + [avg_split] * (num_chunks - 2) + [0]
    elif edge_cases == "zero_tokens_middle":
        split_sections = [avg_split] * (num_chunks - 2) + [0] + [avg_split * 2]
    elif edge_cases == "random_uneven_split":
        split_sections = generate_random_multiples_sum(M, num_chunks, least_multiple)
    else:
        raise ValueError(f"Invalid edge case: {edge_cases}")

    # adds up the split_sections to make it M
    assert sum(split_sections) == M, "The split_sections do not add up to M"

    # make sure every split_section is a multiple of least_multiple
    for split_section in split_sections:
        assert (
            split_section % least_multiple == 0
        ), "The split_sections are not multiples of least_multiple"

    return split_sections


# Calculate the shape of the scaling tensor for NVFP4 1D blockwise quantization without padding
def get_nvfp4_scale_shape_no_padding(shape, columnwise):
    M, K = 1, 1
    M = math.prod(shape[:-1])
    K = shape[-1]

    if columnwise:
        outer = K
        inner = math.ceil(M / 16)
        return (outer, inner)
    # rowwise
    outer = M
    inner = math.ceil(K / 16)
    return (outer, inner)


def reference_group_quantize(
    x: torch.Tensor,
    quantizers: list[NVFP4Quantizer],
    split_sections: list[int],
    return_identity: bool,
    return_transpose: bool,
) -> torch.Tensor:
    x_view = x.reshape(-1, x.size(-1))
    x_chunks = torch.split(x, split_sections)

    # rowwise quantization
    x_qx = []
    x_sx = []
    x_amax_rowwise = []
    # columnwise quantization
    x_qx_t = []
    x_sx_t = []
    x_amax_colwise = []

    for i in range(len(x_chunks)):
        x_chunk = x_chunks[i]
        x_nvfp4_res = quantizers[i](x_chunk)
        if return_identity:
            x_qx.append(x_nvfp4_res._rowwise_data.view(dtype=torch.uint8))
            x_sx.append(x_nvfp4_res._rowwise_scale_inv)
            x_amax_rowwise.append(x_nvfp4_res._amax_rowwise)
        else:
            x_qx.append(None)
            x_sx.append(None)
            x_amax_rowwise.append(None)
        if return_transpose:
            x_qx_t.append(x_nvfp4_res._columnwise_data.view(dtype=torch.uint8))
            x_sx_t.append(x_nvfp4_res._columnwise_scale_inv)
            x_amax_colwise.append(x_nvfp4_res._amax_columnwise)
        else:
            x_qx_t.append(None)
            x_sx_t.append(None)
            x_amax_colwise.append(None)

    return x_qx, x_sx, x_amax_rowwise, x_qx_t, x_sx_t, x_amax_colwise


def assert_same_shape_and_dtype(x: torch.Tensor, y: torch.Tensor) -> None:
    assert x.shape == y.shape
    assert x.dtype == y.dtype


def check_group_quantization_nvfp4_versus_reference(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    return_identity: bool,
    return_transpose: bool,
    split_sections: list[int],
    with_rht: bool = True,
    with_post_rht_amax: bool = True,
    with_random_sign_mask: bool = True,
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

    # Quantize
    quantizers = [
        NVFP4Quantizer(
            fp4_dtype=te_dtype,
            rowwise=return_identity,
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
        reference_group_quantize(x, quantizers, split_sections, return_identity, return_transpose)
    )

    split_quantize_outputs = tex.split_quantize(x, split_sections, quantizers)

    if return_identity:
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
                torch.testing.assert_close(x_sx_t_valid, x_sx_t_ref_valid, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, N",
    [
        # edge case, zero tokens for all
        (0, 512),
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
@pytest.mark.parametrize(
    "quantize_mode", ["quantize", "quantize_transpose", "quantize_colwise_only"]
)
@pytest.mark.parametrize(
    "with_random_sign_mask", [True, False], ids=["with_random_sign_mask", "no_random_sign_mask"]
)
@pytest.mark.parametrize("with_rht", [True, False], ids=["with_rht", "no_rht"])
def test_rht_with_quantization_block_tiling_versus_reference(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    edge_cases: str,
    quantize_mode: str,
    with_random_sign_mask: bool,
    with_rht: bool,
) -> None:

    split_sections = generate_split_sections(M, N, edge_cases)

    # currently disable pre-RHT amax
    with_post_rht_amax = with_rht

    if quantize_mode == "quantize":
        return_identity = True
        return_transpose = False
    elif quantize_mode == "quantize_transpose":
        return_identity = True
        return_transpose = True
    elif quantize_mode == "quantize_colwise_only":
        return_identity = False
        return_transpose = True
    else:
        raise ValueError(f"Invalid quantize mode: {quantize_mode}")

    check_group_quantization_nvfp4_versus_reference(
        x_dtype=x_dtype,
        M=M,
        N=N,
        return_identity=return_identity,
        return_transpose=return_transpose,
        split_sections=split_sections,
        with_rht=with_rht,
        with_post_rht_amax=with_post_rht_amax,
        with_random_sign_mask=with_random_sign_mask,
    )
