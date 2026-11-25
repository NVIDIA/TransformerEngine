# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer
from nvfp4_utils import swizzle_nvfp4_scale

pytestmark = pytest.mark.skipif(not te.is_nvfp4_available(), reason="NVFP4 requires Blackwell")


def _quantizer(directions, swizzled):
    q = NVFP4Quantizer(rowwise=directions[0], columnwise=directions[1], with_2d_quantization=True)
    q.optimize_for_gemm = swizzled
    return q


def _check(x, splits, grouped, directions, swizzled):
    # Debug views cache host metadata. Refresh only outside graph capture/replay.
    grouped.tensor_shapes = None
    grouped.offsets = None
    results = grouped.split_into_quantized_tensors()
    reference_quantizer = _quantizer(directions, False)
    start = 0
    for rows, result in zip(splits, results):
        part = x[start : start + rows]
        start += rows
        if rows == 0:
            for enabled, name in zip(directions, ("_amax_rowwise", "_amax_columnwise")):
                if enabled:
                    assert torch.equal(getattr(result, name), torch.zeros(1, device=x.device))
            continue
        reference = reference_quantizer(part)
        for enabled, direction, amax_name, shape in (
            (directions[0], "rowwise", "_amax_rowwise", (rows, x.shape[1])),
            (directions[1], "columnwise", "_amax_columnwise", (x.shape[1], rows)),
        ):
            if not enabled:
                continue
            assert torch.equal(
                getattr(result, f"_{direction}_data").view(torch.uint8),
                getattr(reference, f"_{direction}_data").view(torch.uint8),
            )
            assert torch.equal(getattr(result, amax_name), getattr(reference, amax_name))
            scales = getattr(result, f"_{direction}_scale_inv").view(torch.uint8)
            expected = getattr(reference, f"_{direction}_scale_inv").view(torch.uint8)
            expected = expected[: shape[0], : shape[1] // 16]
            if swizzled:
                expected = swizzle_nvfp4_scale(shape[0], shape[1], expected, False)
            assert torch.equal(scales, expected)


@pytest.mark.parametrize("cols", [128, 256, 384, 1024])
@pytest.mark.parametrize("directions", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("swizzled", [False, True])
@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize(
    "splits",
    [(128, 128, 128), (0, 0, 0), (0, 128, 256), (256, 128, 0), (128, 0, 256), (256, 384, 128)],
    ids=[
        "regular",
        "zero_tokens_all",
        "zero_tokens_front",
        "zero_tokens_end",
        "zero_tokens_middle",
        "uneven",
    ],
)
def test_grouped_2d_bitwise(cols, directions, swizzled, paged, splits):
    torch.manual_seed(32)
    capacity = sum(splits) + (256 if paged else 0)
    x = torch.randn(capacity, cols, device="cuda", dtype=torch.bfloat16)
    # A capacity tail must not affect any expert's scalar amax or quantized values.
    x[sum(splits) :] = 10000
    metadata = torch.tensor(splits, device="cuda", dtype=torch.int64)
    q = _quantizer(directions, swizzled)
    result = tex.group_quantize(x, q, len(splits), metadata)
    _check(x, splits, result, directions, swizzled)


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("directions", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("swizzled", [False, True])
def test_grouped_2d_graph_replay(paged, directions, swizzled):
    x = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
    metadata = torch.tensor([128, 128, 256], device="cuda", dtype=torch.int64)
    q = _quantizer(directions, swizzled)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        tex.group_quantize(x, q, 3, metadata)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = tex.group_quantize(x, q, 3, metadata)
    cases = (
        [(128, 128, 256), (256, 0, 128), (0, 0, 0), (128, 256, 128)]
        if paged
        else [(128, 128, 256), (256, 0, 256), (0, 384, 128), (128, 256, 128)]
    )
    for splits in cases:
        metadata.copy_(torch.tensor(splits, device="cuda", dtype=torch.int64))
        x.normal_()
        x[sum(splits) :] = 10000
        graph.replay()
        _check(x, splits, output, directions, swizzled)


def test_grouped_2d_noop_and_uniform_weight_reuse():
    x = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
    q = _quantizer((True, True), True)
    output = tex.group_quantize(x, q, 4, None)
    flag = torch.ones(1, device="cuda", dtype=torch.float32)
    names = (
        "rowwise_data",
        "columnwise_data",
        "scale_inv",
        "columnwise_scale_inv",
        "amax",
        "columnwise_amax",
    )
    before = {name: getattr(output, name).clone() for name in names}
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tex.group_quantize(x, q, 4, None, noop_flag=flag, output=output)
    x.normal_()
    graph.replay()
    for name in names:
        assert torch.equal(getattr(output, name), before[name])
    flag.zero_()
    graph.replay()
    _check(x, [128] * 4, output, (True, True), True)


@pytest.mark.parametrize(
    "option",
    [
        "with_2d_quantization",
        "stochastic_rounding",
        "with_amax_reduction",
        "disable_second_level_scale",
    ],
)
def test_grouped_2d_rejects_unsupported_recipe(option):
    q = _quantizer((True, True), False)
    setattr(q, option, option != "with_2d_quantization")
    x = torch.zeros(256, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="not implemented|local per-tensor amax"):
        tex.group_quantize(x, q, 2, None)
