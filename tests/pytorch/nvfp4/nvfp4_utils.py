# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer

import torch
import math
import random


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


def _rowwise_swizzle_nvfp4_scale(input_M, input_N, scale: torch.Tensor) -> torch.Tensor:
    assert scale.dim() == 2
    assert input_M == scale.shape[0]
    assert input_N // 16 == scale.shape[1]

    x = scale.view(input_M // 128, 4, 32, input_N // 64, 4)
    x = x.permute(0, 3, 2, 1, 4)
    x = x.contiguous()
    # View back as original 2D shape
    x = x.view(input_M, input_N // 16)
    return x


# TN-only layout for NVFP4 means that there is only rowwise swizzle
# just need to switch the M, N which means transposing the input
def swizzle_nvfp4_scale(input_M, input_N, scale: torch.Tensor, columnwise: bool) -> torch.Tensor:
    if not columnwise:
        return _rowwise_swizzle_nvfp4_scale(input_M, input_N, scale)
    else:
        return _rowwise_swizzle_nvfp4_scale(input_N, input_M, scale)


# Helper function to generate random multiples sum
def _generate_random_multiples_sum(total=8192, n=4, multiple=64):
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


# Generate split sections for NVFP4 1D blockwise quantization
def generate_split_sections(
    M: int, N: int, edge_cases: str, least_multiple: int = 128
) -> list[int]:
    num_chunks = 4
    split_sections = None

    avg_split = M // num_chunks

    if M == 0 or N == 0:
        # all zeros
        return [0] * num_chunks
    if edge_cases == "regular":
        split_sections = [avg_split] * num_chunks
    elif edge_cases == "zero_tokens_all":
        split_sections = [0] * num_chunks
    elif edge_cases == "zero_tokens_front":
        split_sections = [0] + [avg_split] * (num_chunks - 2) + [avg_split * 2]
    elif edge_cases == "zero_tokens_end":
        split_sections = [avg_split * 2] + [avg_split] * (num_chunks - 2) + [0]
    elif edge_cases == "zero_tokens_middle":
        split_sections = [avg_split] * (num_chunks - 2) + [0] + [avg_split * 2]
    elif edge_cases == "random_uneven_split":
        split_sections = _generate_random_multiples_sum(M, num_chunks, least_multiple)
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


# Reference implementation of group quantization for NVFP4 1D blockwise quantization
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


# Function to assert that two tensors have the same shape and dtype
def assert_same_shape_and_dtype(x: torch.Tensor, y: torch.Tensor) -> None:
    assert x.shape == y.shape
    assert x.dtype == y.dtype
