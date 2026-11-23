# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
"""Block128 grouped requant must match BF16 grouped quantization exactly."""

import pytest
import torch
import transformer_engine.pytorch as te
import transformer_engine_torch as tex

from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="Block128 grouped requant requires Hopper or newer",
)


@pytest.mark.parametrize("hidden", [128, 256, 384, 512, 640])
@pytest.mark.parametrize(
    "shape_rep,group_rows",
    [
        ("same", [128, 128, 128]),
        ("same", [256, 256]),
        ("varying", [128, 128, 128]),
        ("varying", [128, 256, 384, 128]),
        ("varying", [256, 128]),
        ("varying", [0, 128, 0, 256, 0]),
    ],
)
@pytest.mark.parametrize("power_of_two", [False, True])
@pytest.mark.parametrize("return_dequantized", [False, True])
@pytest.mark.parametrize("compact_input", [False, True])
def test_block128_requant_matches_grouped_tma(
    hidden, shape_rep, group_rows, power_of_two, return_dequantized, compact_input
):
    counts = torch.tensor(group_rows, dtype=torch.int64, device="cuda")
    rows = int(counts.sum())
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0))) * hidden
    torch.manual_seed(9182)
    source = torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda")
    quantizer = Float8BlockQuantizer(
        te.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
        block_scaling_dim=1,
        force_pow_2_scales=power_of_two,
    )
    direct = tex.group_quantize(source, quantizer, counts.numel(), counts, tensor_offsets=offsets)
    input_scales = direct.scale_inv.clone().flatten()
    if compact_input:
        blocks = hidden // 128
        input_scales = torch.cat(
            [
                scales.view(blocks, count).T.contiguous().flatten()
                for scales, count in zip(
                    input_scales.split([count * blocks for count in group_rows]), group_rows
                )
            ]
        )
    first_dims = None if shape_rep == "same" else counts
    tensor_offsets = None if shape_rep == "same" else offsets
    requant = GroupedTensor(
        (rows, hidden),
        torch.bfloat16,
        num_tensors=counts.numel(),
        quantizer=quantizer,
        data=direct.rowwise_data.clone().flatten(),
        scale_inv=input_scales,
        first_dims=first_dims,
        tensor_offsets=tensor_offsets,
        with_gemm_swizzled_scales=not compact_input,
    )
    reconstructed = tex.group_requantize_inplace(
        requant,
        quantizer,
        counts.numel(),
        first_dims,
        te.DType.kBFloat16,
        tensor_offsets,
        return_dequantized,
    )
    if not return_dequantized:
        reconstructed = tex.group_requantize_inplace(
            requant,
            quantizer,
            counts.numel(),
            first_dims,
            te.DType.kBFloat16,
            tensor_offsets,
            True,
        )
    compact_scales = (
        input_scales.view(rows, hidden // 128)
        if compact_input
        else torch.cat(
            [
                scales.view(hidden // 128, count).T.contiguous()
                for scales, count in zip(
                    input_scales.split([count * (hidden // 128) for count in group_rows]),
                    group_rows,
                )
            ]
        )
    )
    independent_bf16 = (
        direct.rowwise_data.view(rows, hidden).view(torch.float8_e4m3fn).float()
        * compact_scales.repeat_interleave(128, dim=1)
    ).bfloat16()
    torch.testing.assert_close(reconstructed, independent_bf16, rtol=0, atol=0)
    expected = tex.group_quantize(
        reconstructed, quantizer, counts.numel(), counts, tensor_offsets=offsets
    )
    torch.testing.assert_close(requant.rowwise_data, direct.rowwise_data, rtol=0, atol=0)
    torch.testing.assert_close(requant.scale_inv, direct.scale_inv, rtol=0, atol=0)
    torch.testing.assert_close(
        requant.columnwise_data.flatten(), expected.columnwise_data, rtol=0, atol=0
    )
    torch.testing.assert_close(
        requant.columnwise_scale_inv.flatten(), expected.columnwise_scale_inv, rtol=0, atol=0
    )


@pytest.mark.parametrize("shape_rep", ["same", "varying"])
@pytest.mark.parametrize("hidden", [384, 4096, 8192])
@pytest.mark.parametrize("columnwise", [False, True])
@pytest.mark.parametrize("return_dequantized", [False, True])
@pytest.mark.parametrize("graph_replay", [False, True])
def test_block128_requant_output_modes(
    shape_rep, hidden, columnwise, return_dequantized, graph_replay
):
    group_rows = [128, 128, 128] if shape_rep == "same" else [128, 0, 256]
    rows = sum(group_rows)
    counts = torch.tensor(group_rows, device="cuda", dtype=torch.int64)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0))) * hidden
    q = Float8BlockQuantizer(
        te.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=columnwise,
        block_scaling_dim=1,
        force_pow_2_scales=True,
        rowwise_scale_layout=tex.FP8BlockwiseRowwiseOption.ROWWISE_COMPACT,
    )
    torch.manual_seed(271)
    source = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    wire = q(source)
    data, scales = wire._rowwise_data, wire._rowwise_scale_inv
    expected_bf16 = (
        data.view(torch.float8_e4m3fn).float() * scales.repeat_interleave(128, 1)
    ).bfloat16()
    expected_scales = torch.cat(
        [part.T.contiguous().flatten() for part in scales.split(group_rows)]
    )
    grouped = GroupedTensor(
        (rows, hidden),
        torch.bfloat16,
        num_tensors=len(group_rows),
        quantizer=q,
        data=data.flatten(),
        scale_inv=scales.flatten(),
        first_dims=None if shape_rep == "same" else counts,
        tensor_offsets=None if shape_rep == "same" else offsets,
        with_gemm_swizzled_scales=False,
    )

    def run():
        # Also exercise binding fallback to the tensor's own first_dims metadata.
        return tex.group_requantize_inplace(
            grouped,
            q,
            len(group_rows),
            None,
            te.DType.kBFloat16,
            None if shape_rep == "same" else offsets,
            return_dequantized,
        )

    if graph_replay:
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            actual = run()
        graph.replay()
    else:
        actual = run()
    torch.cuda.synchronize()
    torch.testing.assert_close(grouped.rowwise_data.flatten(), data.flatten(), rtol=0, atol=0)
    torch.testing.assert_close(grouped.scale_inv.flatten(), expected_scales, rtol=0, atol=0)
    if return_dequantized:
        torch.testing.assert_close(actual, expected_bf16, rtol=0, atol=0)
    else:
        assert actual is None
    if columnwise:
        reference = tex.group_quantize(
            expected_bf16,
            q,
            len(group_rows),
            counts,
            tensor_offsets=offsets,
        )
        torch.testing.assert_close(
            grouped.columnwise_data.flatten(), reference.columnwise_data.flatten(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            grouped.columnwise_scale_inv.flatten(),
            reference.columnwise_scale_inv.flatten(),
            rtol=0,
            atol=0,
        )
    else:
        assert grouped.columnwise_data is None
