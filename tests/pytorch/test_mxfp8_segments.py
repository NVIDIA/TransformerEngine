# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Exact MXFP8 payload parity for segmented projection gradients."""

import math

import pytest
import torch

from transformer_engine.pytorch import DType, MXFP8Quantizer


def assert_payload_equal(actual, expected):
    rows = math.prod(expected.shape[:-1])
    columns = expected.shape[-1]
    for name in (
        "_rowwise_data",
        "_rowwise_scale_inv",
        "_columnwise_data",
        "_columnwise_scale_inv",
    ):
        before, after = getattr(expected, name), getattr(actual, name)
        assert (before is None) == (after is None)
        if before is None:
            continue
        assert before.shape == after.shape
        before, after = before.view(torch.uint8), after.view(torch.uint8)
        if "scale_inv" in name:
            columnwise = name.startswith("_columnwise")
            outer, inner = (columns, rows // 32) if columnwise else (rows, columns // 32)
            if expected._with_gemm_swizzled_scales:
                i = torch.arange(outer, device=before.device).view(-1, 1)
                j = torch.arange(inner, device=before.device).view(1, -1)
                index = ((i // 128) * math.ceil(inner / 4) + j // 4) * 512
                index = index + (i % 32) * 16 + ((i % 128) // 32) * 4 + j % 4
                before, after = before.flatten()[index], after.flatten()[index]
            elif columnwise:
                before, after = before[:inner, :outer], after[:inner, :outer]
            else:
                before, after = before[:outer, :inner], after[:outer, :inner]
        if not torch.equal(before, after):
            mismatch = (before != after).nonzero()[:8]
            indexes = tuple(mismatch[:, i] for i in range(mismatch.shape[1]))
            pytest.fail(
                f"{name}: positions={mismatch.tolist()}, expected={before[indexes].tolist()},"
                f" actual={after[indexes].tolist()}"
            )


@pytest.fixture(autouse=True)
def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("MXFP8 requires Blackwell")


@pytest.mark.parametrize("usage", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("swizzle", [False, True])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "singleton", "fallback", "broadcast"])
@pytest.mark.parametrize(
    "pattern", ["random", "signed_zero", "extreme", "all_nan", "all_inf", "all_zero"]
)
def test_segmented_payload(usage, swizzle, layout, pattern):
    torch.manual_seed(1234)
    rows = 128 if layout != "fallback" else 96
    widths = (256, 128, 128)
    source = torch.randn(rows, sum(widths), device="cuda", dtype=torch.bfloat16)
    if pattern == "signed_zero":
        source[:, ::2] = 0.0
        source[:, 1::2] = -0.0
    elif pattern == "extreme":
        values = torch.tensor(
            [0.0, -0.0, 2**-100, -(2**-100), 2**100, -(2**100), float("inf"), float("nan")],
            device="cuda",
            dtype=torch.bfloat16,
        )
        source.copy_(values.repeat(rows, sum(widths) // values.numel()))
    if pattern in ("all_nan", "all_inf", "all_zero"):
        source.fill_({"all_nan": float("nan"), "all_inf": float("inf"), "all_zero": 0.0}[pattern])
    parts = torch.split(source, widths, dim=-1)
    if layout == "contiguous":
        parts = tuple(p.contiguous() for p in parts)
    elif layout == "singleton":
        parts = tuple(p[:, None, None, :] for p in parts)
    elif layout == "broadcast":
        parts = tuple(p[:1].expand(rows, -1) for p in parts)
    quantizer = MXFP8Quantizer(fp8_dtype=DType.kFloat8E4M3, rowwise=usage[0], columnwise=usage[1])
    quantizer.optimize_for_gemm = swizzle
    reference = quantizer(torch.cat(parts, dim=-1))
    actual = quantizer.quantize_segments(parts)
    assert_payload_equal(actual, reference)
    replay = quantizer.quantize_segments(parts)
    assert_payload_equal(replay, actual)


@pytest.mark.parametrize("segments", [1, 3])
def test_segmented_stream_and_graph(segments):
    parts = tuple(
        torch.randn(128, 128, device="cuda", dtype=torch.bfloat16) for _ in range(segments)
    )
    quantizer = MXFP8Quantizer(fp8_dtype=DType.kFloat8E4M3, rowwise=True, columnwise=True)
    quantizer.optimize_for_gemm = True
    expected = quantizer(torch.cat(parts, dim=-1))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = quantizer.quantize_segments(parts)
    torch.cuda.current_stream().wait_stream(stream)
    assert_payload_equal(actual, expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = quantizer.quantize_segments(parts)
    graph.replay()
    torch.cuda.synchronize()
    assert_payload_equal(captured, expected)


def test_segmented_rejects_invalid_shapes():
    quantizer = MXFP8Quantizer(fp8_dtype=DType.kFloat8E4M3, rowwise=True, columnwise=True)
    part = torch.empty(128, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="one or three"):
        quantizer.quantize_segments((part, part))
    with pytest.raises(ValueError, match="output shape"):
        quantizer.quantize_segments((part,), (128, 256))
    with pytest.raises(ValueError, match="row counts"):
        quantizer.quantize_segments((part, part[:96], part))


@pytest.mark.parametrize("usage", [(True, False), (False, True)])
def test_native_segmented_binding_rejects_unqualified_usage(usage):
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("The direct segmented binding is qualified for SM103")
    import transformer_engine_torch as tex

    part = torch.ones(128, 128, device="cuda", dtype=torch.bfloat16)
    quantizer = MXFP8Quantizer(DType.kFloat8E4M3, rowwise=usage[0], columnwise=usage[1])
    quantizer.optimize_for_gemm = True
    with pytest.raises(RuntimeError, match="both orientations"):
        tex.quantize_mxfp8_segments((part,), quantizer, tuple(part.shape))
