# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
"""Unaligned expert row counts must be rejected, including the final expert."""

import subprocess
import sys

import pytest
import torch
import transformer_engine.pytorch
import transformer_engine_torch as tex

from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="EP block128 requant requires Hopper or newer",
)


def _compact_payload(source):
    quantizer = Float8BlockQuantizer(
        tex.DType.kFloat8E4M3, rowwise=True, columnwise=False,
        block_scaling_dim=1, force_pow_2_scales=True,
        rowwise_scale_layout=tex.FP8BlockwiseRowwiseOption.ROWWISE_COMPACT,
    )
    tensor = quantizer.quantize(source)
    wire = tensor._rowwise_scale_inv
    return tensor._rowwise_data, wire, wire.T.contiguous()


def _columnwise_reference(data, wire):
    # These cases have at most 128 live rows per expert. Compute amax only
    # over that expert, after the BF16 reconstruction consumed by grouped cast.
    reconstructed = (
        data.view(torch.float8_e4m3fn).float() * wire.repeat_interleave(128, dim=1)
    ).bfloat16()
    amax = reconstructed.float().abs().amax(dim=0)
    multiplier = torch.where(amax == 0, torch.ones_like(amax), 448.0 / amax)
    # Clear the FP32 mantissa, matching power-of-two block scaling.
    multiplier = (multiplier.view(torch.int32) & -8388608).view(torch.float32)
    columnwise = (reconstructed.float() * multiplier).to(torch.float8_e4m3fn)
    return columnwise.T.contiguous().view(torch.uint8), multiplier.reciprocal(), reconstructed


def _partial_final_group(last_rows, hidden):
    torch.manual_seed(123)
    capacity = 384
    live_rows = 128 + last_rows
    blocks = hidden // 128
    parts = [torch.randn(n, hidden, device="cuda", dtype=torch.bfloat16)
             for n in (128, last_rows)]
    packed = [_compact_payload(x) for x in parts]
    reference = [_columnwise_reference(p[0], p[1]) for p in packed]
    row_data = torch.full((capacity, hidden), 126, device="cuda", dtype=torch.uint8)
    row_scale = torch.full((capacity * blocks,), 1000.0, device="cuda")
    start = 0
    for n, (data, _, gemm) in zip((128, last_rows), packed):
        row_data[start:start + n] = data
        row_scale[start * blocks:(start + n) * blocks] = gemm[:, :n].flatten()
        start += n
    counts = torch.tensor([128, 0, last_rows, 0], device="cuda", dtype=torch.int64)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0))) * hidden
    quantizer = Float8BlockQuantizer(
        tex.DType.kFloat8E4M3, rowwise=True, columnwise=True,
        block_scaling_dim=1, force_pow_2_scales=True,
    )
    grouped = GroupedTensor(
        (capacity, hidden), torch.bfloat16, num_tensors=4, quantizer=quantizer,
        data=row_data.flatten(), scale_inv=row_scale, first_dims=counts,
        tensor_offsets=offsets, with_gemm_swizzled_scales=True,
    )
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        bf16 = tex.group_requantize_inplace(
            grouped, quantizer, 4, counts, tex.DType.kBFloat16, offsets, True,
        )
    cw = grouped.columnwise_data.flatten()
    scales = grouped.columnwise_scale_inv.flatten()
    # Initialize captured output allocations to detect writes outside group bounds.
    cw.fill_(123)
    scales.fill_(-1)
    bf16.fill_(42)
    graph.replay()
    torch.cuda.synchronize()
    expected_cw = torch.cat([r[0].flatten() for r in reference])
    expected_scales = torch.cat([r[1].flatten() for r in reference])
    expected_bf16 = torch.cat([r[2] for r in reference])
    torch.testing.assert_close(cw[:live_rows * hidden], expected_cw, rtol=0, atol=0)
    torch.testing.assert_close(scales[:2 * hidden], expected_scales, rtol=0, atol=0)
    torch.testing.assert_close(bf16[:live_rows], expected_bf16, rtol=0, atol=0)
    assert torch.all(cw[live_rows * hidden:] == 123)
    assert torch.all(scales[2 * hidden:] == -1)
    assert torch.all(bf16[live_rows:] == 42)


def _small_capacity_group(rows, hidden):
    """Grouped scales use [hidden/128, rows], not wire [rows, scale_cols]."""
    torch.manual_seed(321)
    source = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    data, wire, gemm = _compact_payload(source)
    reference_cw, reference_scales, reference_bf16 = (
        _columnwise_reference(data, wire)
    )
    counts = torch.tensor([0, rows, 0], device="cuda", dtype=torch.int64)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0))) * hidden
    quantizer = Float8BlockQuantizer(
        tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
        block_scaling_dim=1,
        force_pow_2_scales=True,
    )
    grouped = GroupedTensor(
        (rows, hidden),
        torch.bfloat16,
        num_tensors=3,
        quantizer=quantizer,
        data=data.flatten(),
        scale_inv=gemm[:, :rows].contiguous().flatten(),
        first_dims=counts,
        tensor_offsets=offsets,
        with_gemm_swizzled_scales=True,
    )
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        bf16 = tex.group_requantize_inplace(
            grouped, quantizer, 3, counts, tex.DType.kBFloat16, offsets, True,
        )
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        grouped.columnwise_data.flatten(), reference_cw.flatten(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        grouped.columnwise_scale_inv.flatten(), reference_scales.flatten(), rtol=0, atol=0
    )
    torch.testing.assert_close(bf16, reference_bf16, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [1, 63, 127])
@pytest.mark.parametrize("hidden", [384, 4096, 8192])
def test_rejects_partial_final_group(rows, hidden):
    _expect_unaligned_rejection("partial", rows, hidden)


@pytest.mark.parametrize("rows", [1, 7, 31])
@pytest.mark.parametrize("hidden", [4096, 8192])
def test_rejects_small_capacity_group(rows, hidden):
    _expect_unaligned_rejection("small", rows, hidden)


def _expect_unaligned_rejection(case, rows, hidden):
    child = subprocess.run(
        [sys.executable, __file__, case, str(rows), str(hidden)],
        capture_output=True, text=True, timeout=60,
    )
    assert child.returncode != 0
    assert "First dimension of each tensor in a group must be divisible by 128" in (
        child.stdout + child.stderr
    )


if __name__ == "__main__":
    run = _partial_final_group if sys.argv[1] == "partial" else _small_capacity_group
    run(int(sys.argv[2]), int(sys.argv[3]))
