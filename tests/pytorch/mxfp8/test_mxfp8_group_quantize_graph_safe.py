# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.


import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import MXFP8Quantizer

import pytest
import torch
import random
import math

from mxfp8_utils import swizzle_mxfp8_scale, get_mxfp8_scale_shape_no_padding

recipe_available, reason_for_no_recipe = te.is_mxfp8_available(return_reason=True)


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
    least_multiple = 128
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


def reference_group_quantize(
    x: torch.Tensor,
    quantizers: list[MXFP8Quantizer],
    split_sections: list[int],
    return_rowwise: bool,
    return_transpose: bool,
) -> torch.Tensor:
    x_chunks = torch.split(x, split_sections)

    # rowwise quantization
    x_qx = []
    x_sx = []
    # columnwise quantization
    x_qx_t = []
    x_sx_t = []

    for i in range(len(x_chunks)):
        x_chunk = x_chunks[i]
        x_mxfp8_res = quantizers[i](x_chunk)
        if return_rowwise:
            x_qx.append(x_mxfp8_res._rowwise_data.view(dtype=torch.uint8))
            x_sx.append(x_mxfp8_res._rowwise_scale_inv)
        else:
            x_qx.append(None)
            x_sx.append(None)
        if return_transpose:
            x_qx_t.append(x_mxfp8_res._columnwise_data.view(dtype=torch.uint8))
            x_sx_t.append(x_mxfp8_res._columnwise_scale_inv)
        else:
            x_qx_t.append(None)
            x_sx_t.append(None)

    return x_qx, x_sx, x_qx_t, x_sx_t


def fused_grouped_quantize(
    x: torch.Tensor, split_section_tensor: torch.Tensor, quantizer: MXFP8Quantizer
):

    # view x as a 2D tensor
    hidden_dim = x.shape[-1]
    x = x.view(-1, hidden_dim)
    num_tensors = split_section_tensor.shape[0]

    grouped_output = tex.group_quantize(x, quantizer, num_tensors, split_section_tensor)

    return grouped_output


def assert_same_shape_and_dtype(x: torch.Tensor, y: torch.Tensor) -> None:
    assert x.shape == y.shape
    assert x.dtype == y.dtype


def check_grouped_tensor_mxfp8_versus_reference(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    return_rowwise: bool,
    return_transpose: bool,
    split_sections: list[int],
    optimize_for_gemm: bool = False,
) -> None:

    te_dtype = te.DType.kFloat8E4M3

    split_section_tensor = torch.tensor(split_sections, dtype=torch.int64, device="cuda")

    # Setup device and random seed
    device = "cuda"
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Input
    x = torch.randn((M, N), dtype=x_dtype, device=device)
    x_splits = torch.split(x, split_sections)

    # Quantize
    quantizers = [
        MXFP8Quantizer(
            fp8_dtype=te_dtype,
            rowwise=return_rowwise,
            columnwise=return_transpose,
        )
        for _ in range(len(split_sections))
    ]

    grouped_quantizer = quantizers[0].copy()
    # configure grouped quantizer with swizzle fusion
    # and compare with reference without swizzle fusion
    grouped_quantizer.optimize_for_gemm = optimize_for_gemm

    x_qx_ref, x_sx_ref, x_qx_t_ref, x_sx_t_ref = reference_group_quantize(
        x, quantizers, split_sections, return_rowwise, return_transpose
    )

    group_quantized_output = fused_grouped_quantize(x, split_section_tensor, grouped_quantizer)
    # get a list of MXFP8 quantized tensors for testing
    split_quantize_outputs = group_quantized_output.split_into_quantized_tensors()

    if return_rowwise:
        x_qx = [output._rowwise_data.view(dtype=torch.uint8) for output in split_quantize_outputs]
        x_sx = [output._rowwise_scale_inv for output in split_quantize_outputs]

        for i in range(len(x_qx)):
            if split_sections[i] == 0:
                # then just assert the same shape and dtype because the buffer won't be zero out
                assert_same_shape_and_dtype(x_qx[i], x_qx_ref[i])
                assert_same_shape_and_dtype(x_sx[i], x_sx_ref[i])
            else:
                torch.testing.assert_close(x_qx[i], x_qx_ref[i], atol=0.0, rtol=0.0)
                valid_scale_shape = get_mxfp8_scale_shape_no_padding(x_splits[i].shape, False)
                assert (
                    valid_scale_shape == x_sx[i].shape
                ), "The scale shape is not correctly aligned"
                x_sx_i = x_sx[i].clone()
                x_sx_ref_i = x_sx_ref[i].clone()
                if optimize_for_gemm:
                    x_sx_ref_i = swizzle_mxfp8_scale(
                        split_sections[i], N, x_sx_ref_i, columnwise=False
                    )
                torch.testing.assert_close(x_sx_i, x_sx_ref_i, atol=0.0, rtol=0.0)

    if return_transpose:
        x_qx_t = [
            output._columnwise_data.view(dtype=torch.uint8) for output in split_quantize_outputs
        ]
        x_sx_t = [output._columnwise_scale_inv for output in split_quantize_outputs]
        # assert with zero tolerance
        for i in range(len(x_qx_t)):
            if split_sections[i] == 0:
                # then just assert the same shape and dtype because the buffer won't be zero out
                assert_same_shape_and_dtype(x_qx_t[i], x_qx_t_ref[i])
                assert_same_shape_and_dtype(x_sx_t[i], x_sx_t_ref[i])
            else:
                torch.testing.assert_close(x_qx_t[i], x_qx_t_ref[i], atol=0.0, rtol=0.0)
                valid_scale_shape = get_mxfp8_scale_shape_no_padding(x_splits[i].shape, True)
                assert (
                    valid_scale_shape == x_sx_t[i].shape
                ), "The scale shape is not correctly aligned"
                x_sx_t_i = x_sx_t[i].clone()
                x_sx_t_ref_i = x_sx_t_ref[i].clone()
                if optimize_for_gemm:
                    x_sx_t_ref_i = swizzle_mxfp8_scale(
                        split_sections[i], N, x_sx_t_ref_i, columnwise=True
                    )
                torch.testing.assert_close(x_sx_t_i, x_sx_t_ref_i, atol=0.0, rtol=0.0)


def check_grouped_tensor_mxfp8_with_paged_stashing(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    return_rowwise: bool,
    return_transpose: bool,
    split_sections: list[int],
    valid_M: int = None,
    optimize_for_gemm: bool = False,
) -> None:

    te_dtype = te.DType.kFloat8E4M3

    assert valid_M is not None, "valid_M must be provided when with_paged_stashing is True"
    assert valid_M < M, "valid_M must be less than M when with_paged_stashing is True"

    split_section_tensor = torch.tensor(split_sections, dtype=torch.int64, device="cuda")

    # Setup device and random seed
    device = "cuda"
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Input (fill the entire tensor with garbage too)
    x = torch.randn((M, N), dtype=x_dtype, device=device)
    valid_x = x[:valid_M, :].clone()
    x_splits = torch.split(valid_x, split_sections)

    # Quantize
    quantizers = [
        MXFP8Quantizer(
            fp8_dtype=te_dtype,
            rowwise=return_rowwise,
            columnwise=return_transpose,
        )
        for _ in range(len(split_sections))
    ]

    grouped_quantizer = quantizers[0].copy()
    # configure grouped quantizer with swizzle fusion
    # and compare with reference without swizzle fusion
    grouped_quantizer.optimize_for_gemm = optimize_for_gemm

    x_qx_ref, x_sx_ref, x_qx_t_ref, x_sx_t_ref = reference_group_quantize(
        valid_x, quantizers, split_sections, return_rowwise, return_transpose
    )

    # Note: for grouped quantize with paged stashing
    # it's expected that we can just pass in the regular input x, not the valid_x
    # the kernel is expected to porcess it correctly by becoming no-op for cuda graph
    group_quantized_output = fused_grouped_quantize(x, split_section_tensor, grouped_quantizer)

    # get a list of MXFP8 quantized tensors for testing
    split_quantize_outputs = group_quantized_output.split_into_quantized_tensors()

    if return_rowwise:
        x_qx = [output._rowwise_data.view(dtype=torch.uint8) for output in split_quantize_outputs]
        x_sx = [output._rowwise_scale_inv for output in split_quantize_outputs]

        for i in range(len(x_qx)):
            if split_sections[i] == 0:
                # then just assert the same shape and dtype because the buffer won't be zero out
                assert_same_shape_and_dtype(x_qx[i], x_qx_ref[i])
                assert_same_shape_and_dtype(x_sx[i], x_sx_ref[i])
            else:
                torch.testing.assert_close(x_qx[i], x_qx_ref[i], atol=0.0, rtol=0.0)
                valid_scale_shape = get_mxfp8_scale_shape_no_padding(x_splits[i].shape, False)
                assert (
                    valid_scale_shape == x_sx[i].shape
                ), "The scale shape is not correctly aligned"
                x_sx_i = x_sx[i].clone()
                x_sx_ref_i = x_sx_ref[i].clone()
                if optimize_for_gemm:
                    x_sx_ref_i = swizzle_mxfp8_scale(
                        split_sections[i], N, x_sx_ref_i, columnwise=False
                    )
                torch.testing.assert_close(x_sx_i, x_sx_ref_i, atol=0.0, rtol=0.0)

    if return_transpose:
        x_qx_t = [
            output._columnwise_data.view(dtype=torch.uint8) for output in split_quantize_outputs
        ]
        x_sx_t = [output._columnwise_scale_inv for output in split_quantize_outputs]
        # assert with zero tolerance
        for i in range(len(x_qx_t)):
            if split_sections[i] == 0:
                # then just assert the same shape and dtype because the buffer won't be zero out
                assert_same_shape_and_dtype(x_qx_t[i], x_qx_t_ref[i])
                assert_same_shape_and_dtype(x_sx_t[i], x_sx_t_ref[i])
            else:
                torch.testing.assert_close(x_qx_t[i], x_qx_t_ref[i], atol=0.0, rtol=0.0)
                valid_scale_shape = get_mxfp8_scale_shape_no_padding(x_splits[i].shape, True)
                assert (
                    valid_scale_shape == x_sx_t[i].shape
                ), "The scale shape is not correctly aligned"
                x_sx_t_i = x_sx_t[i].clone()
                x_sx_t_ref_i = x_sx_t_ref[i].clone()
                if optimize_for_gemm:
                    x_sx_t_ref_i = swizzle_mxfp8_scale(
                        split_sections[i], N, x_sx_t_ref_i, columnwise=True
                    )
                torch.testing.assert_close(x_sx_t_i, x_sx_t_ref_i, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, N",
    [
        # edge case, zero tokens for all
        (0, 512),
        # full tile cases
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
    "optimize_for_gemm", [True, False], ids=["optimize_for_gemm", "no_optimize_for_gemm"]
)
def test_grouped_tensor_mxfp8_versus_reference(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    edge_cases: str,
    quantize_mode: str,
    optimize_for_gemm: bool,
) -> None:

    split_sections = generate_split_sections(M, N, edge_cases)

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

    check_grouped_tensor_mxfp8_versus_reference(
        x_dtype=x_dtype,
        M=M,
        N=N,
        return_rowwise=return_rowwise,
        return_transpose=return_transpose,
        split_sections=split_sections,
        optimize_for_gemm=optimize_for_gemm,
    )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, N",
    [
        # M won't be empty in paged stashing
        # full tile cases
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
        # even if buffer is not empty, but the token splits are all zero
        "zero_tokens_all",
        # partially zero tokens
        "zero_tokens_front",
        "zero_tokens_end",
        "zero_tokens_middle",
        "random_uneven_split",
    ],
)
@pytest.mark.parametrize("quantize_mode", ["rowwise_only", "both_directions", "columnwise_only"])
@pytest.mark.parametrize(
    "optimize_for_gemm", [True, False], ids=["optimize_for_gemm", "no_optimize_for_gemm"]
)
def test_grouped_tensor_mxfp8_with_paged_stashing(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    edge_cases: str,
    quantize_mode: str,
    optimize_for_gemm: bool,
) -> None:

    # paged stashing means that the sum of total tokens is less than
    # or equal to the buffer size, you can have buffer [2048, 1024]
    # and when you only receive 1024 tokens, the last half is garbage
    # so input has shape [2048, 1024]
    # split sections can be [256, 256, 256, 256], sums to 1024
    valid_M = 0 if edge_cases == "zero_tokens_all" else M // 2
    split_sections = generate_split_sections(valid_M, N, edge_cases)

    # sanity check
    if edge_cases == "zero_tokens_all":
        assert valid_M == 0, "valid_M must be 0 when edge_cases is zero_tokens_all"
    else:
        assert valid_M == M // 2, "valid_M must be M // 2 when edge_cases is not zero_tokens_all"

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

    check_grouped_tensor_mxfp8_with_paged_stashing(
        x_dtype=x_dtype,
        M=M,
        N=N,
        return_rowwise=return_rowwise,
        return_transpose=return_transpose,
        split_sections=split_sections,
        valid_M=valid_M,
        optimize_for_gemm=optimize_for_gemm,
    )


# ---------------------------------------------------------------------------------------------
# Pre-quantized MXFP8 input (FP8 token dispatch)
#
# tex.group_requantize_inplace takes a grouped tensor that arrives
# ALREADY rowwise-quantized (its high-precision form no longer exists), and makes it GEMM-ready
# in both directions: the rowwise data passes through verbatim with its scales swizzled, and the
# columnwise copy is rebuilt via dequantize + columnwise-only requantize.
#
# These mirror the edge-case matrices of test_grouped_tensor_mxfp8_versus_reference and
# test_grouped_tensor_mxfp8_with_paged_stashing so the same shapes, zero-token placements and
# uneven splits exercise this path.
# ---------------------------------------------------------------------------------------------


def make_prequantized_wire_tensor(x: torch.Tensor, split_section_tensor: torch.Tensor):
    """Rowwise-only, unswizzled grouped tensor, as FP8 dispatch delivers it."""
    wire_quantizer = MXFP8Quantizer(fp8_dtype=te.DType.kFloat8E4M3, rowwise=True, columnwise=False)
    # Must stay unswizzled: the requantize path asserts on it and dequantize needs compact scales.
    wire_quantizer.optimize_for_gemm = False
    wire = fused_grouped_quantize(x, split_section_tensor, wire_quantizer)
    assert wire.columnwise_data is None
    assert not wire._with_gemm_swizzled_scales
    return wire


def make_op_quantizer(columnwise: bool):
    """The op's input quantizer, configured the way the ops layer configures it.

    ``columnwise`` mirrors ``weight_requires_grad``: it tells the helper whether a wgrad GEMM
    will consume a columnwise copy.
    """
    quantizer = MXFP8Quantizer(fp8_dtype=te.DType.kFloat8E4M3, rowwise=True, columnwise=columnwise)
    quantizer.optimize_for_gemm = True
    return quantizer


def make_gemm_ready_tensor(x: torch.Tensor, split_section_tensor: torch.Tensor):
    """Grouped tensor already GEMM-ready in both directions (swizzled scales)."""
    quantizer = make_op_quantizer(columnwise=True)
    tensor = fused_grouped_quantize(x, split_section_tensor, quantizer)
    assert tensor.columnwise_data is not None
    assert tensor._with_gemm_swizzled_scales
    return tensor


def check_prequantized_requantize_versus_reference(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    split_sections: list[int],
) -> None:
    """Run the pre-quantized requantize path and check both directions against a reference.

    The reference is derived from dequantize(wire), not from the original high-precision x: that
    is the only data a consumer can see after dispatch, and MXFP8 rowwise requantization is
    idempotent, so it is an exact reference rather than an approximate one.
    """
    device = "cuda"
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # The buffer is always M rows. Paged stashing is just the case where the groups cover fewer
    # than M of them (valid_M < M) and the tail holds garbage the kernels must leave alone; the
    # non-paged case is the same code path with sum(split_sections) == M.
    x = torch.randn((M, N), dtype=x_dtype, device=device)
    split_section_tensor = torch.tensor(split_sections, dtype=torch.int64, device=device)
    num_groups = len(split_sections)
    # Rows the groups actually cover. Beyond this the buffers hold whatever the allocator handed
    # out, so nothing past it may be compared.
    valid_rows = sum(split_sections)

    wire = make_prequantized_wire_tensor(x, split_section_tensor)

    # Snapshot what must survive verbatim, plus the compact scales the reference swizzles.
    rowwise_data_before = wire.rowwise_data.clone()
    wire_splits_before = [
        (t._rowwise_data.view(dtype=torch.uint8).clone(), t._rowwise_scale_inv.clone())
        for t in wire.split_into_quantized_tensors()
    ]

    # Reference high-precision input: everything downstream is derived from this. Only the live
    # rows are kept -- dequantize allocates M rows but writes only the ones the groups cover.
    dequantized_ref = (
        tex.group_dequantize(wire, te.DType.kBFloat16)
        .rowwise_data.view(M, N)[:valid_rows, :]
        .clone()
    )

    # Reference columnwise copy, quantized per group from the dequantized data.
    colwise_quantizers = [
        MXFP8Quantizer(fp8_dtype=te.DType.kFloat8E4M3, rowwise=False, columnwise=True)
        for _ in range(num_groups)
    ]
    _, _, colwise_data_ref, colwise_scale_ref = reference_group_quantize(
        dequantized_ref,
        colwise_quantizers,
        split_sections,
        return_rowwise=False,
        return_transpose=True,
    )

    # ---- the code under test ----
    # The op's quantizer, configured as the ops layer does: columnwise_usage says a wgrad GEMM
    # will consume the columnwise copy, so the helper builds it and switches rowwise off itself.
    dequantized = tex.group_requantize_inplace(
        wire,
        make_op_quantizer(columnwise=True),
        num_groups,
        split_section_tensor,
        te.DType.kBFloat16,
        return_dequantized=True,
    )

    assert wire.columnwise_data is not None, "columnwise data must be built"
    assert wire.columnwise_scale_inv is not None, "columnwise scales must survive the swizzle"

    # The returned dequantized tensor is what bias gradients are reduced from. Compare only the
    # live rows: both this and the reference allocate M rows but write only the covered ones, and
    # their tails are separate uninitialized allocations.
    torch.testing.assert_close(dequantized[:valid_rows, :], dequantized_ref, atol=0.0, rtol=0.0)

    # The rowwise DATA must pass through untouched; only its scales are re-laid-out.
    torch.testing.assert_close(wire.rowwise_data, rowwise_data_before, atol=0.0, rtol=0.0)

    if valid_rows > 0:
        # A tensor whose groups are all empty has no scales to lay out, so the swizzle is a no-op
        # and leaves the flag unset; every other case must come back swizzled.
        assert wire._with_gemm_swizzled_scales, "rowwise scales must be marked swizzled"

    # Per-group comparison, same structure as check_grouped_tensor_mxfp8_versus_reference.
    outputs = wire.split_into_quantized_tensors()
    x_splits = torch.split(dequantized_ref, split_sections)

    for i, out in enumerate(outputs):
        rows_i = split_sections[i]
        scale_before = wire_splits_before[i][1]
        colwise_data = out._columnwise_data.view(dtype=torch.uint8)
        colwise_scale = out._columnwise_scale_inv

        if rows_i == 0:
            # Buffers for empty groups are never written, so only shape and dtype are meaningful.
            assert_same_shape_and_dtype(colwise_data, colwise_data_ref[i])
            assert_same_shape_and_dtype(colwise_scale, colwise_scale_ref[i])
            continue

        # Rowwise scales: the swizzled form of the compact scales this group arrived with. The
        # rowwise DATA is covered by the whole-buffer identity check above.
        torch.testing.assert_close(
            out._rowwise_scale_inv,
            swizzle_mxfp8_scale(rows_i, N, scale_before, columnwise=False),
            atol=0.0,
            rtol=0.0,
        )

        # Columnwise: rebuilt from the dequantized data, and swizzled by the quantize kernel
        # because the caller sets optimize_for_gemm.
        torch.testing.assert_close(colwise_data, colwise_data_ref[i], atol=0.0, rtol=0.0)
        valid_scale_shape = get_mxfp8_scale_shape_no_padding(x_splits[i].shape, True)
        assert (
            valid_scale_shape == colwise_scale.shape
        ), "The columnwise scale shape is not correctly aligned"
        torch.testing.assert_close(
            colwise_scale,
            swizzle_mxfp8_scale(rows_i, N, colwise_scale_ref[i], columnwise=True),
            atol=0.0,
            rtol=0.0,
        )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, N",
    [
        # edge case, zero tokens for all
        (0, 512),
        # full tile cases
        (1024, 256),
        # larger sizes
        (8192, 1024),
        (16384, 8192),
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
def test_prequantized_requantize_versus_reference(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    edge_cases: str,
) -> None:
    split_sections = generate_split_sections(M, N, edge_cases)
    check_prequantized_requantize_versus_reference(
        x_dtype=x_dtype,
        M=M,
        N=N,
        split_sections=split_sections,
    )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, N",
    [
        # M won't be empty in paged stashing
        (1024, 256),
        (8192, 1024),
        (16384, 8192),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.bfloat16], ids=str)
@pytest.mark.parametrize(
    "edge_cases",
    [
        "regular",
        "zero_tokens_all",
        "zero_tokens_front",
        "zero_tokens_end",
        "zero_tokens_middle",
        "random_uneven_split",
    ],
)
def test_prequantized_requantize_with_paged_stashing(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    edge_cases: str,
) -> None:
    # Paged stashing: the buffer holds M rows but only valid_M carry live tokens; the rest is
    # garbage the kernels must not touch.
    valid_M = 0 if edge_cases == "zero_tokens_all" else M // 2
    split_sections = generate_split_sections(valid_M, N, edge_cases)
    assert sum(split_sections) == valid_M

    check_prequantized_requantize_versus_reference(
        x_dtype=x_dtype,
        M=M,
        N=N,
        split_sections=split_sections,
    )


def _requantize_setup(M: int = 1024, N: int = 256):
    """Common inputs for the state-dispatch tests below."""
    torch.manual_seed(0)
    split_sections = [M // 4] * 4
    x = torch.randn((M, N), dtype=torch.bfloat16, device="cuda")
    return x, torch.tensor(split_sections, dtype=torch.int64, device="cuda"), len(split_sections)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_prequantized_requantize_passes_through_gemm_ready_input():
    """A tensor already GEMM-ready in both directions is left untouched."""
    x, splits, num_groups = _requantize_setup()
    tensor = make_gemm_ready_tensor(x, splits)
    rowwise_before = tensor.rowwise_data.clone()
    columnwise_before = tensor.columnwise_data.clone()
    scale_before = tensor.scale_inv.clone()

    out = tex.group_requantize_inplace(
        tensor, make_op_quantizer(columnwise=True), num_groups, splits, te.DType.kBFloat16
    )

    assert out is None
    assert tensor._with_gemm_swizzled_scales
    torch.testing.assert_close(tensor.rowwise_data, rowwise_before, atol=0.0, rtol=0.0)
    torch.testing.assert_close(tensor.columnwise_data, columnwise_before, atol=0.0, rtol=0.0)
    torch.testing.assert_close(tensor.scale_inv, scale_before, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_prequantized_requantize_skips_columnwise_when_not_needed():
    """columnwise_usage=False (frozen weights) swizzles rowwise without building columnwise."""
    x, splits, num_groups = _requantize_setup()
    wire = make_prequantized_wire_tensor(x, splits)

    out = tex.group_requantize_inplace(
        wire, make_op_quantizer(columnwise=False), num_groups, splits, te.DType.kBFloat16
    )

    assert out is None
    assert wire._with_gemm_swizzled_scales, "the GEMM still needs swizzled rowwise scales"
    assert wire.columnwise_data is None, "no wgrad GEMM, so no columnwise copy should be built"


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_prequantized_requantize_rejects_dequantized_from_gemm_ready_input():
    """Bias grads cannot be served from an already-swizzled input, so this must raise."""
    x, splits, num_groups = _requantize_setup()
    tensor = make_gemm_ready_tensor(x, splits)

    with pytest.raises(RuntimeError, match="compact format"):
        tex.group_requantize_inplace(
            tensor,
            make_op_quantizer(columnwise=True),
            num_groups,
            splits,
            te.DType.kBFloat16,
            return_dequantized=True,
        )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_prequantized_requantize_rejects_swizzled_without_columnwise():
    """Swizzled rowwise scales with no columnwise copy: it can no longer be rebuilt."""
    x, splits, num_groups = _requantize_setup()
    wire = make_prequantized_wire_tensor(x, splits)
    # Swizzle in place, leaving the tensor rowwise-only.
    tex.grouped_swizzle_for_gemm(wire, True, False)

    with pytest.raises(RuntimeError, match="cannot be rebuilt"):
        tex.group_requantize_inplace(
            wire, make_op_quantizer(columnwise=True), num_groups, splits, te.DType.kBFloat16
        )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_prequantized_requantize_rejects_dtype_mismatch():
    """The helper keeps the input's format; it does not convert between formats."""
    x, splits, num_groups = _requantize_setup()
    wire = make_prequantized_wire_tensor(x, splits)
    mismatched = MXFP8Quantizer(fp8_dtype=te.DType.kFloat8E5M2, rowwise=True, columnwise=True)
    mismatched.optimize_for_gemm = True

    with pytest.raises(RuntimeError, match="dtype"):
        tex.group_requantize_inplace(wire, mismatched, num_groups, splits, te.DType.kBFloat16)
