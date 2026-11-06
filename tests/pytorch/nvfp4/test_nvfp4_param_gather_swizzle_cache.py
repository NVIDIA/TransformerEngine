# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Tests for NVTE_NVFP4_WEIGHT_SWIZZLE_CACHE.

NVFP4 primary weights refreshed by ``post_all_gather_processing`` keep compact scales.
With the cache enabled, their GEMM-swizzled scales are built once per all-gather and
reused by ``general_gemm``, the grouped GEMM with discrete weights and the fused grouped
MLP. These tests check that (a) cached GEMMs are bit-identical to uncached ones and launch
fewer swizzle kernels, (b) the cache follows weight updates made through
``post_all_gather_processing``, and (c) a captured CUDA graph reads the refreshed scales."""

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.cpp_extensions import gemm as gemm_mod
from transformer_engine.pytorch.cpp_extensions import general_gemm
from transformer_engine.pytorch.ops.fused import grouped_mlp as grouped_mlp_mod
from transformer_engine.pytorch.tensor import utils as tensor_utils
from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

recipe_available, reason_for_no_recipe = te.is_nvfp4_available(return_reason=True)

pytestmark = pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)


@pytest.fixture
def swizzle_cache(monkeypatch):
    """Toggle the cache; tests switch it off to compute uncached references."""

    def set_enabled(enabled):
        monkeypatch.setattr(tensor_utils, "_NVFP4_WEIGHT_SWIZZLE_CACHE", enabled)

    set_enabled(True)
    yield set_enabled


def _swizzle_kernel_count(fn):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(1 for e in prof.events() if "swizzle" in e.name)


def _make_weights(num, n, k):
    quantizer = NVFP4Quantizer(rowwise=True, columnwise=True, with_2d_quantization=True)
    weights = [
        quantizer(torch.randn(n, k, device="cuda", dtype=torch.bfloat16)) for _ in range(num)
    ]
    for w in weights:
        assert not w._with_gemm_swizzled_scales
    return weights


def _update_weights(weights):
    """Simulate an optimizer step plus parameter all-gather."""
    for w in weights:
        w.quantize_(torch.randn(w.shape, device="cuda", dtype=torch.bfloat16))
    tensor_utils.post_all_gather_processing(weights)


def _quantized_input(m, k):
    quantizer = NVFP4Quantizer(rowwise=True, columnwise=True)
    return quantizer(torch.randn(m, k, device="cuda", dtype=torch.bfloat16))


def test_general_gemm(swizzle_cache):
    torch.manual_seed(0)
    (weight,) = _make_weights(1, 512, 1024)
    x = _quantized_input(256, 1024)

    def gemm():
        return general_gemm(weight, x, out_dtype=torch.bfloat16)[0]

    swizzle_cache(False)
    ref = gemm()
    swizzle_cache(True)
    out = gemm()  # marks the weight
    assert weight._gsw_wanted
    torch.testing.assert_close(out, ref, rtol=0, atol=0)

    tensor_utils.post_all_gather_processing([weight])
    assert weight._gsw_view is not None
    torch.testing.assert_close(gemm(), ref, rtol=0, atol=0)

    _update_weights([weight])
    swizzle_cache(False)
    ref2 = gemm()
    swizzle_cache(True)
    torch.testing.assert_close(gemm(), ref2, rtol=0, atol=0)
    assert not torch.equal(ref, ref2)


@pytest.mark.parametrize("layout", ["TN", "NN"])
def test_grouped_gemm_discrete_weights(swizzle_cache, layout):
    torch.manual_seed(0)
    num_groups, n, k = 4, 512, 1024
    m_splits = (256, 0, 128, 384)
    weights = _make_weights(num_groups, n, k)

    first_dims = torch.tensor(m_splits, dtype=torch.int64, device="cuda")
    m_total = sum(m_splits)
    in_features = k if layout == "TN" else n
    out_features = n if layout == "TN" else k
    x = torch.randn(m_total, in_features, device="cuda", dtype=torch.bfloat16)
    x_quantizer = NVFP4Quantizer(
        rowwise=True, columnwise=True, with_rht=True, with_post_rht_amax=True
    )
    x_quantizer.optimize_for_gemm = True
    grouped_x = tex.group_quantize(x, x_quantizer, num_groups, first_dims)

    def make_out():
        return GroupedTensor.make_grouped_tensor(
            num_tensors=num_groups,
            first_dims=first_dims,
            last_dims=None,
            logical_first_dim=m_total,
            logical_last_dim=out_features,
            quantizer=None,
            device="cuda",
            dtype=torch.bfloat16,
        )

    def gemm(out):
        gemm_mod.general_grouped_gemm_for_grouped_tensor(weights, grouped_x, out, layout=layout)
        return out.rowwise_data

    def reference():
        swizzle_cache(False)
        ref = gemm(make_out()).clone()
        swizzle_cache(True)
        return ref

    swizzle_cache(False)
    ref_out = make_out()
    num_ref = _swizzle_kernel_count(lambda: gemm(ref_out))
    swizzle_cache(True)

    # The first call marks the weights; the post-all-gather refresh builds the views.
    gemm(make_out())
    assert all(getattr(w, "_gsw_wanted", False) for w in weights)
    tensor_utils.post_all_gather_processing(weights)
    assert all(getattr(w, "_gsw_view", None) is not None for w in weights)
    out = make_out()
    num_cached = _swizzle_kernel_count(lambda: gemm(out))
    torch.testing.assert_close(out.rowwise_data, ref_out.rowwise_data, rtol=0, atol=0)
    assert 0 < num_ref and num_cached < num_ref, (num_ref, num_cached)

    # The refresh writes in place, so results follow the weights.
    _update_weights(weights)
    torch.testing.assert_close(gemm(make_out()), reference(), rtol=0, atol=0)

    # A captured CUDA graph reads the refreshed scales on replay.
    static_out = make_out()
    gemm(static_out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gemm(static_out)
    _update_weights(weights)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(static_out.rowwise_data, reference(), rtol=0, atol=0)


def test_swizzle_pack_discrete_follows_param_gather(swizzle_cache, monkeypatch):
    calls = []

    def fake_pack(datas, scales, fmt, device):
        calls.append(fmt)
        return object()

    monkeypatch.setattr(
        tex.grouped_mlp_experimental,
        "swizzle_scales_and_pack_ptrs_for_discrete_weights",
        fake_pack,
    )
    monkeypatch.setattr(grouped_mlp_mod, "_DISCRETE_WEIGHT_SWIZZLE_CACHE", {})
    datas = [torch.empty(16, device="cuda") for _ in range(2)]
    scales = [torch.empty(16, device="cuda") for _ in range(2)]
    pack = grouped_mlp_mod._swizzle_pack_discrete

    first = pack(datas, scales, "nvfp4", "cuda")
    assert pack(datas, scales, "nvfp4", "cuda") is first
    assert pack(datas, scales, "mxfp8_rowwise", "cuda") is not first
    assert len(calls) == 2

    tensor_utils.post_all_gather_processing([])
    assert pack(datas, scales, "nvfp4", "cuda") is not first
    assert len(calls) == 3

    swizzle_cache(False)
    pack(datas, scales, "nvfp4", "cuda")
    pack(datas, scales, "nvfp4", "cuda")
    assert len(calls) == 5
