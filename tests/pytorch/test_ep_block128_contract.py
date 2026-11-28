# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
"""Compact producer and grouped requant contract tests."""

import os
import weakref

import pytest
import torch
import transformer_engine.pytorch  # Load the common library before the binding.
import transformer_engine_torch as tex
from transformer_engine.common.recipe import Float8BlockScaling
from transformer_engine.pytorch.ep import _quantize_float8_blockwise
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor


def check_columnwise(data, scales, source, power_of_two=False, epsilon=0.0):
    rows, hidden = source.shape
    assert data.shape == (hidden, rows)
    scales = scales.flatten()
    for start in range(0, rows, 128):
        block = source[start : start + 128].float()
        amax = block.abs().amax(dim=0).clamp_min(epsilon)
        scale = torch.where(amax > 0, torch.full_like(amax, 448.0) / amax, torch.ones_like(amax))
        if power_of_two:
            scale = torch.exp2(torch.floor(torch.log2(scale)))
        expected = (block * scale).to(torch.float8_e4m3fn).float()
        actual = data[:, start : start + 128].T.contiguous().view(torch.float8_e4m3fn).float()
        scale_begin = (start // 128) * hidden
        torch.testing.assert_close(
            scales[scale_begin : scale_begin + hidden], scale.reciprocal(), rtol=1e-6, atol=0
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def quantize_payload(source, epsilon, power_of_two, columnwise):
    quantizer = Float8BlockQuantizer(
        tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=columnwise,
        block_scaling_dim=1,
        amax_epsilon=epsilon,
        force_pow_2_scales=power_of_two,
        rowwise_scale_layout=tex.FP8BlockwiseRowwiseOption.ROWWISE_COMPACT,
    )
    carrier = quantizer.quantize(source)
    wire = carrier._rowwise_scale_inv
    payload = (carrier._rowwise_data, wire, wire.T.contiguous())
    if columnwise:
        payload += (carrier._columnwise_data, carrier._columnwise_scale_inv)
    return payload


def main():
    torch.manual_seed(123)
    source = torch.randn(256, 4096, device="cuda", dtype=torch.bfloat16)
    source_ref = weakref.ref(source)
    carrier, wire = _quantize_float8_blockwise(source, Float8BlockScaling(), columnwise=True)
    assert not hasattr(carrier, "_original_tensor")
    assert carrier._columnwise_data is not None
    assert carrier._rowwise_scale_inv is wire
    check_columnwise(
        carrier._columnwise_data,
        carrier._columnwise_scale_inv,
        source,
        power_of_two=carrier._quantizer.force_pow_2_scales,
        epsilon=carrier._quantizer.amax_epsilon,
    )
    del source
    assert source_ref() is None
    print("PASS producer wrapper does not retain BF16 source", flush=True)
    for hidden in (4096, 8192):
        for rows in (0, 1, 127, 128, 129, 256):
            x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
            row_only = quantize_payload(x, 0.0, False, False)
            both = quantize_payload(x, 0.0, False, True)
            assert len(row_only) == 3 and len(both) == 5
            for old, new in zip(row_only, both[:3]):
                torch.testing.assert_close(old, new, rtol=0, atol=0)
            check_columnwise(both[3], both[4], x)
            row_data, wire, _ = row_only
            reconstructed = row_data.view(torch.float8_e4m3fn).float()
            reconstructed *= wire[:, : hidden // 128].repeat_interleave(128, dim=1)
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                captured = quantize_payload(x, 0.0, False, True)
            graph.replay()
            torch.cuda.synchronize()
            for expected, actual in zip(both, captured):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            print(f"PASS rows={rows} hidden={hidden}", flush=True)
    rows, hidden = 256, 4096
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    data, wire, gemm = quantize_payload(x, 0.0, True, False)
    quantizer = Float8BlockQuantizer(
        tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
        block_scaling_dim=1,
        force_pow_2_scales=True,
    )
    counts = torch.tensor([rows], device="cuda", dtype=torch.int64)
    offsets = torch.tensor([0, rows * hidden], device="cuda", dtype=torch.int64)
    grouped = GroupedTensor(
        (rows, hidden),
        torch.bfloat16,
        num_tensors=1,
        quantizer=quantizer,
        data=data.flatten(),
        scale_inv=gemm.flatten(),
        first_dims=counts,
        tensor_offsets=offsets,
        with_gemm_swizzled_scales=True,
    )
    bf16 = tex.group_requantize_inplace(
        grouped, quantizer, 1, counts, tex.DType.kBFloat16, offsets, True
    )
    expected = data.view(torch.float8_e4m3fn).float() * wire.repeat_interleave(128, dim=1)
    torch.testing.assert_close(bf16, expected.bfloat16(), rtol=0, atol=0)
    print("PASS unified single-group GEMM-layout BF16 return", flush=True)

    for power_of_two in (False, True):
        for epsilon in (0.0, 16.0):
            config_quantizer = Float8BlockQuantizer(
                tex.DType.kFloat8E4M3,
                rowwise=True,
                columnwise=True,
                block_scaling_dim=1,
                force_pow_2_scales=power_of_two,
                amax_epsilon=epsilon,
            )
            configured = GroupedTensor(
                (rows, hidden),
                torch.bfloat16,
                num_tensors=1,
                quantizer=config_quantizer,
                data=data.flatten(),
                scale_inv=gemm.flatten(),
                first_dims=counts,
                tensor_offsets=offsets,
                with_gemm_swizzled_scales=True,
            )
            tex.group_requantize_inplace(
                configured, config_quantizer, 1, counts, tex.DType.kBFloat16, offsets
            )
            check_columnwise(
                configured.columnwise_data.view(hidden, rows),
                configured.columnwise_scale_inv,
                expected,
                power_of_two=power_of_two,
                epsilon=epsilon,
            )
    print("PASS unified power-of-two and epsilon configuration", flush=True)

    # Non-power-of-two row scales make the intermediate rounding observable.
    fast_data, fast_wire, fast_gemm = quantize_payload(x, 0.0, False, False)
    precise = fast_data.view(torch.float8_e4m3fn).float() * fast_wire.repeat_interleave(128, 1)
    rounded = precise.bfloat16().float()
    assert torch.any(precise != rounded)
    previous_fast_math = os.environ.get("NVTE_USE_FAST_MATH")
    try:
        for fast_math in (False, True):
            os.environ["NVTE_USE_FAST_MATH"] = str(int(fast_math))
            configured = GroupedTensor(
                (rows, hidden),
                torch.bfloat16,
                num_tensors=1,
                quantizer=quantizer,
                data=fast_data.flatten(),
                scale_inv=fast_gemm.flatten(),
                first_dims=counts,
                tensor_offsets=offsets,
                with_gemm_swizzled_scales=True,
            )
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                result = tex.group_requantize_inplace(
                    configured, quantizer, 1, counts, tex.DType.kBFloat16, offsets, True
                )
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(result, precise.bfloat16(), rtol=0, atol=0)
            check_columnwise(
                configured.columnwise_data.view(hidden, rows),
                configured.columnwise_scale_inv,
                rounded,
                power_of_two=True,
            )
    finally:
        if previous_fast_math is None:
            os.environ.pop("NVTE_USE_FAST_MATH", None)
        else:
            os.environ["NVTE_USE_FAST_MATH"] = previous_fast_math
    print("PASS FP32 and BF16-intermediate requant with graph replay", flush=True)

    saved_cw = grouped.columnwise_data
    saved_scale = grouped.columnwise_scale_inv
    cw_before = saved_cw.clone()
    scale_before = saved_scale.clone()
    returned = tex.group_requantize_inplace(
        grouped, quantizer, 1, counts, tex.DType.kBFloat16, offsets, True
    )
    torch.testing.assert_close(returned, expected.bfloat16(), rtol=0, atol=0)
    assert grouped.columnwise_data is saved_cw
    assert grouped.columnwise_scale_inv is saved_scale
    torch.testing.assert_close(saved_cw, cw_before, rtol=0, atol=0)
    torch.testing.assert_close(saved_scale, scale_before, rtol=0, atol=0)
    quantizer.set_usage(rowwise=True, columnwise=False)
    row_only_group = GroupedTensor(
        (rows, hidden),
        torch.bfloat16,
        num_tensors=1,
        quantizer=quantizer,
        data=data.flatten(),
        scale_inv=gemm.flatten(),
        first_dims=counts,
        tensor_offsets=offsets,
        with_gemm_swizzled_scales=True,
    )
    returned = tex.group_requantize_inplace(
        row_only_group, quantizer, 1, counts, tex.DType.kBFloat16, offsets, True
    )
    torch.testing.assert_close(returned, expected.bfloat16(), rtol=0, atol=0)
    assert row_only_group.columnwise_data is None
    assert row_only_group.columnwise_scale_inv is None
    quantizer.set_usage(rowwise=True, columnwise=True)
    print("PASS BF16-only return preserves existing CW and disabled CW usage", flush=True)

    graph_group = GroupedTensor(
        (rows, hidden),
        torch.bfloat16,
        num_tensors=1,
        quantizer=quantizer,
        data=data.clone(),
        scale_inv=gemm.clone(),
        first_dims=counts,
        tensor_offsets=offsets,
        with_gemm_swizzled_scales=True,
    )
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        graph_bf16 = tex.group_requantize_inplace(
            graph_group, quantizer, 1, counts, tex.DType.kBFloat16, offsets, True
        )
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_bf16, expected.bfloat16(), rtol=0, atol=0)
    print("PASS unified grouped requant CUDA graph replay", flush=True)

    parts = [
        torch.randn(128, hidden, device="cuda", dtype=torch.bfloat16),
        torch.randn(256, hidden, device="cuda", dtype=torch.bfloat16),
    ]
    packed = [quantize_payload(p, 0.0, True, False) for p in parts]
    data = torch.cat([p[0].flatten() for p in packed])
    gemm = torch.cat([p[2].flatten() for p in packed])
    counts = torch.tensor([128, 256], device="cuda", dtype=torch.int64)
    offsets = torch.tensor([0, 128 * hidden, 384 * hidden], device="cuda", dtype=torch.int64)
    grouped = GroupedTensor(
        (384, hidden),
        torch.bfloat16,
        num_tensors=2,
        quantizer=quantizer,
        data=data,
        scale_inv=gemm,
        first_dims=counts,
        tensor_offsets=offsets,
        with_gemm_swizzled_scales=True,
    )
    bf16 = tex.group_requantize_inplace(
        grouped, quantizer, 2, counts, tex.DType.kBFloat16, offsets, True
    )
    expected = torch.cat(
        [
            p[0].view(torch.float8_e4m3fn).float()
            * p[1][:, : hidden // 128].repeat_interleave(128, 1)
            for p in packed
        ]
    )
    torch.testing.assert_close(bf16, expected.bfloat16(), rtol=0, atol=0)
    print("PASS unified grouped offsets BF16 return", flush=True)

    # Capacity-tail and empty-group coverage: only the first 256 rows belong to
    # groups, while the backing storage has 512 rows. Device offsets must skip
    # the empty middle group without changing the group-relative scale layout.
    capacity = 512
    parts = [
        torch.randn(128, hidden, device="cuda", dtype=torch.bfloat16),
        torch.empty(0, hidden, device="cuda", dtype=torch.bfloat16),
        torch.randn(128, hidden, device="cuda", dtype=torch.bfloat16),
    ]
    packed = [quantize_payload(p, 0.0, True, False) for p in parts if p.numel()]
    row_data = torch.zeros((capacity, hidden), device="cuda", dtype=torch.uint8)
    row_scales = torch.zeros(capacity * (hidden // 128), device="cuda", dtype=torch.float32)
    row_data[:128] = packed[0][0]
    row_data[128:256] = packed[1][0]
    blocks = hidden // 128
    row_scales[: 128 * blocks] = packed[0][2].flatten()
    row_scales[128 * blocks : 256 * blocks] = packed[1][2].flatten()
    counts = torch.tensor([128, 0, 128], device="cuda", dtype=torch.int64)
    offsets = torch.tensor(
        [0, 128 * hidden, 128 * hidden, 256 * hidden], device="cuda", dtype=torch.int64
    )

    def make_capacity_group():
        return GroupedTensor(
            (capacity, hidden),
            torch.bfloat16,
            num_tensors=3,
            quantizer=quantizer,
            data=row_data.clone().flatten(),
            scale_inv=row_scales.clone(),
            first_dims=counts,
            tensor_offsets=offsets,
            with_gemm_swizzled_scales=True,
        )

    live_group = make_capacity_group()
    live_bf16 = tex.group_requantize_inplace(
        live_group, quantizer, 3, counts, tex.DType.kBFloat16, offsets, True
    )
    expected0 = packed[0][0].view(torch.float8_e4m3fn).float() * packed[0][2].T.repeat_interleave(
        128, 1
    )
    expected2 = packed[1][0].view(torch.float8_e4m3fn).float() * packed[1][2].T.repeat_interleave(
        128, 1
    )
    torch.testing.assert_close(live_bf16[:128], expected0.bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(live_bf16[128:256], expected2.bfloat16(), rtol=0, atol=0)
    live_cw = live_group.columnwise_data.flatten()
    for group_index, source in enumerate((expected0, expected2)):
        begin = group_index * 128 * hidden
        scale_stride = (hidden + 3) // 4 * 4
        scale_begin = group_index * scale_stride
        check_columnwise(
            live_cw[begin : begin + 128 * hidden].view(hidden, 128),
            live_group.columnwise_scale_inv[scale_begin : scale_begin + scale_stride],
            source,
            power_of_two=True,
        )
    print("PASS padded capacity, empty group BF16 and CW", flush=True)

    for capacity in (0, 256):
        for request_cw in (False, True):
            empty_quantizer = Float8BlockQuantizer(
                tex.DType.kFloat8E4M3,
                rowwise=True,
                columnwise=request_cw,
                block_scaling_dim=1,
            )
            empty_counts = torch.zeros(3, device="cuda", dtype=torch.int64)
            empty_offsets = torch.zeros(4, device="cuda", dtype=torch.int64)
            empty = GroupedTensor(
                (capacity, hidden),
                torch.bfloat16,
                num_tensors=3,
                quantizer=empty_quantizer,
                data=torch.zeros(capacity * hidden, device="cuda", dtype=torch.uint8),
                scale_inv=torch.ones(capacity * (hidden // 128), device="cuda"),
                first_dims=empty_counts,
                tensor_offsets=empty_offsets,
                with_gemm_swizzled_scales=True,
            )
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                result = tex.group_requantize_inplace(
                    empty,
                    empty_quantizer,
                    3,
                    empty_counts,
                    tex.DType.kBFloat16,
                    empty_offsets,
                    True,
                )
            graph.replay()
            torch.cuda.synchronize()
            assert result.shape == (capacity, hidden)
            assert result.dtype == torch.bfloat16
            if request_cw:
                assert empty.columnwise_data.numel() == capacity * hidden
            else:
                assert empty.columnwise_data is None
            # No live values exist; capacity tails are deliberately undefined.
    print("PASS zero-capacity and all-zero groups with graph replay", flush=True)

    capacity, hidden = 256, 4096
    source = torch.randn(capacity, hidden, device="cuda", dtype=torch.bfloat16)
    data, _, gemm = quantize_payload(source, 0.0, True, False)
    counts = torch.tensor([capacity], device="cuda", dtype=torch.int64)
    offsets = torch.tensor([0, capacity * hidden], device="cuda", dtype=torch.int64)
    bad_offsets = (
        (offsets[:1], "num_tensors + 1"),
        (offsets.view(1, 2), "contiguous vector"),
        (offsets.to(torch.int32), "CUDA int64"),
        (offsets.cpu(), "CUDA int64"),
        (torch.zeros(4, device="cuda", dtype=torch.int64)[::2], "contiguous vector"),
    )
    for invalid, message in bad_offsets:
        grouped = GroupedTensor(
            (capacity, hidden),
            torch.bfloat16,
            num_tensors=1,
            quantizer=quantizer,
            data=data.flatten(),
            scale_inv=gemm.flatten(),
            first_dims=counts,
            tensor_offsets=offsets,
            with_gemm_swizzled_scales=True,
        )
        try:
            tex.group_requantize_inplace(
                grouped, quantizer, 1, counts, tex.DType.kBFloat16, invalid, True
            )
        except RuntimeError as error:
            assert message in str(error), str(error)
        else:
            raise AssertionError("Invalid block128 offsets were accepted")
    print("PASS invalid offsets rejected before kernel launch", flush=True)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="EP block128 producer requires Hopper or newer",
)
def test_producer_and_unified_requant_contract():
    main()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="EP block128 requant requires Hopper or newer",
)
def test_compact_block128_scales_are_transformed_inside_group_requantize():
    rows, hidden = 256, 256
    blocks = hidden // 128
    source = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    data, wire, gemm = quantize_payload(source, 0.0, True, False)
    counts = torch.tensor([128, 128], device="cuda", dtype=torch.int64)
    offsets = torch.tensor([0, 128, 256], device="cuda", dtype=torch.int64) * hidden
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
        num_tensors=2,
        quantizer=quantizer,
        data=data.flatten(),
        scale_inv=wire.flatten(),
        first_dims=counts,
        tensor_offsets=offsets,
    )

    dequantized = tex.group_requantize_inplace(
        grouped, quantizer, 2, counts, tex.DType.kBFloat16, offsets, True
    )
    expected_scales = torch.cat((gemm[:, :128].flatten(), gemm[:, 128:].flatten()))
    expected_dequantized = data.view(torch.float8_e4m3fn).float() * wire[
        :, :blocks
    ].repeat_interleave(128, dim=1)
    torch.testing.assert_close(grouped.scale_inv, expected_scales, rtol=0, atol=0)
    torch.testing.assert_close(dequantized, expected_dequantized.bfloat16(), rtol=0, atol=0)
    assert grouped._with_gemm_swizzled_scales
    assert grouped.columnwise_data is not None
    assert grouped.columnwise_scale_inv is not None


if __name__ == "__main__":
    main()
