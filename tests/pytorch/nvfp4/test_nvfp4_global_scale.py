# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Tests for the fused NVFP4 global-scale / GEMM-alpha Triton kernel.

The kernel replaces the eager ``amax.float() / (6 * scale_max)`` divisions and the
product / square root of the two operand scales used by the grouped MLP. It must be
bit-identical to the eager ops, including with a broadcast (single-group) amax, and must
be safe to capture in a CUDA graph."""

import pytest
import torch

from transformer_engine.pytorch.triton.nvfp4_global_scale import nvfp4_global_scale_pair

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

_DENOMS = [6.0 * 448.0, 6.0 * 256.0, 6.0 * 114688.0]


def _random_amax(n, device):
    # Wide dynamic range plus exact zeros (empty experts) and a few special magnitudes.
    amax = torch.exp2(torch.empty(n, device=device).uniform_(-40.0, 40.0))
    amax *= torch.rand(n, device=device) + 0.5
    if n >= 4:
        amax[0] = 0.0
        amax[1] = 1.0
        amax[2] = torch.finfo(torch.float32).tiny
        amax[3] = 448.0 * 6.0
    return amax


def _reference(amax_a, amax_b, denom_a, denom_b, combine):
    scale_a = amax_a.to(torch.float32) / denom_a
    scale_b = amax_b.to(torch.float32) / denom_b
    if combine is None:
        return scale_a, scale_b
    if combine == "sqrt_product":
        return torch.sqrt(scale_a * scale_b)
    return (scale_a * scale_b).to(torch.float32)


def _assert_bitwise_equal(out, ref):
    if isinstance(ref, tuple):
        assert isinstance(out, tuple) and len(out) == len(ref)
        for o, r in zip(out, ref):
            _assert_bitwise_equal(o, r)
        return
    assert out.dtype == torch.float32
    assert out.shape == ref.shape
    assert torch.equal(out.view(torch.int32), ref.contiguous().view(torch.int32))


@pytest.mark.parametrize("n", [1, 3, 8, 32, 257, 4096])
@pytest.mark.parametrize("combine", [None, "product", "sqrt_product"])
@pytest.mark.parametrize("denoms", [(_DENOMS[0], _DENOMS[0]), (_DENOMS[1], _DENOMS[2])])
@pytest.mark.parametrize("broadcast", ["none", "a", "b"])
def test_matches_eager(n, combine, denoms, broadcast):
    torch.manual_seed(n)
    device = torch.device("cuda")
    amax_a = _random_amax(1 if broadcast == "a" else n, device)
    amax_b = _random_amax(1 if broadcast == "b" else n, device)
    out = nvfp4_global_scale_pair(amax_a, amax_b, denoms[0], denoms[1], combine=combine)
    if combine is None and amax_a.numel() != amax_b.numel():
        # Separate scales keep their own shapes, so the caller falls back to eager.
        assert out is None
        return
    assert out is not None
    _assert_bitwise_equal(out, _reference(amax_a, amax_b, denoms[0], denoms[1], combine))


def test_falls_back_on_unsupported_inputs():
    device = torch.device("cuda")
    amax = torch.ones(8, device=device)
    assert nvfp4_global_scale_pair(amax.half(), amax, 1.0, 1.0) is None
    assert nvfp4_global_scale_pair(amax, amax[:3], 1.0, 1.0) is None
    assert nvfp4_global_scale_pair(amax.cpu(), amax.cpu(), 1.0, 1.0) is None
    big = torch.ones(4097, device=device)
    assert nvfp4_global_scale_pair(big, big, 1.0, 1.0) is None


@pytest.mark.parametrize("combine", [None, "product", "sqrt_product"])
def test_cuda_graph_replay(combine):
    device = torch.device("cuda")
    n = 32
    denom = _DENOMS[0]
    amax_a = _random_amax(n, device)
    amax_b = _random_amax(n, device)

    # Warm up (compiles the kernel) on a side stream before capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        nvfp4_global_scale_pair(amax_a, amax_b, denom, denom, combine=combine)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = nvfp4_global_scale_pair(amax_a, amax_b, denom, denom, combine=combine)

    for seed in range(3):
        torch.manual_seed(seed)
        amax_a.copy_(_random_amax(n, device))
        amax_b.copy_(_random_amax(n, device))
        graph.replay()
        torch.cuda.synchronize()
        _assert_bitwise_equal(out, _reference(amax_a, amax_b, denom, denom, combine))
