# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Tests for NVTE_NVFP4_WEIGHT_SWIZZLE_CACHE.

NVFP4 primary weights refreshed by ``post_all_gather_processing`` keep compact scales.
With the cache enabled, their GEMM-swizzled scales are built once per all-gather and
reused by ``general_gemm``, the grouped GEMM with discrete weights and the fused grouped
MLP. These tests check that (a) cached GEMMs are bit-identical to uncached ones and launch
fewer swizzle kernels, (b) the cache follows weight updates made through
``post_all_gather_processing``, (c) a captured CUDA graph reads the refreshed scales,
including the fused grouped MLP, and (d) cached buffers are released with their weights."""

import gc
import weakref

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch import QuantizedTensor
from transformer_engine.pytorch.cpp_extensions import gemm as gemm_mod
from transformer_engine.pytorch.cpp_extensions import general_gemm
from transformer_engine.common.recipe import NVFP4BlockScaling
from transformer_engine.pytorch.ops.fused import GroupedMLP_CuTeGEMMGLU
from transformer_engine.pytorch.tensor import utils as tensor_utils
from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer
from transformer_engine.pytorch.utils import get_device_compute_capability

# Import after transformer_engine.pytorch, which loads the core library it links against.
import transformer_engine_torch as tex  # pylint: disable=wrong-import-order

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


def _cached_operands(weights, groups):
    """(columnwise, group) pairs of the cache entries for the given weight groups."""
    return sorted(
        (key[0], name)
        for name, group in groups.items()
        for key in _discrete_cache_keys(group)
        if set(key[1]) <= {id(w) for w in group}
    )


def _discrete_cache_keys(weights):
    """Keys of the discrete-weight cache entries that hold any of ``weights``."""
    ids = {id(w) for w in weights}
    return [
        key for key in tensor_utils._DISCRETE_WEIGHT_SWIZZLE_CACHE if any(i in ids for i in key[1])
    ]


def _pack_uncached(weights, columnwise):
    fn = tex.grouped_mlp_experimental.swizzle_scales_and_pack_ptrs_for_discrete_weights
    datas, scales = tensor_utils._discrete_weight_operands(weights, columnwise)
    return fn(datas, scales, "nvfp4", torch.device("cuda"))


@pytest.mark.parametrize("columnwise", [False, True])
def test_discrete_weight_cache_follows_param_gather(swizzle_cache, columnwise):
    torch.manual_seed(0)
    weights = _make_weights(4, 256, 512)

    def pack():
        return tensor_utils.swizzle_scales_and_pack_ptrs_for_discrete_weights(
            weights, "nvfp4", "cuda", columnwise=columnwise
        )

    # Weights not refreshed by post_all_gather_processing (e.g. quantized in every
    # forward) are not cached.
    assert pack() is not pack()
    assert not _discrete_cache_keys(weights)

    tensor_utils.post_all_gather_processing(weights)
    cached = pack()
    assert pack() is cached
    assert len(_discrete_cache_keys(weights)) == 1
    torch.testing.assert_close(cached[2], _pack_uncached(weights, columnwise)[2], rtol=0, atol=0)
    old = cached[2].clone()

    # The refresh rewrites the cached buffers in place.
    _update_weights(weights)
    assert pack() is cached
    torch.testing.assert_close(cached[2], _pack_uncached(weights, columnwise)[2], rtol=0, atol=0)
    assert not torch.equal(cached[2], old)

    # Weights gathered in separate calls (e.g. buckets) refresh only their own slots.
    _update_weights(weights[1:3])
    torch.testing.assert_close(cached[2], _pack_uncached(weights, columnwise)[2], rtol=0, atol=0)

    swizzle_cache(False)
    assert pack() is not cached


@pytest.mark.usefixtures("swizzle_cache")
def test_discrete_weight_cache_miss_during_capture():
    """A miss during graph capture keeps the swizzle in the graph and is not cached."""
    torch.manual_seed(0)
    weights = _make_weights(2, 256, 512)
    tensor_utils.post_all_gather_processing(weights)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = tensor_utils.swizzle_scales_and_pack_ptrs_for_discrete_weights(
            weights, "nvfp4", "cuda", columnwise=False
        )
    assert not _discrete_cache_keys(weights)
    _update_weights(weights)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured[2], _pack_uncached(weights, False)[2], rtol=0, atol=0)


@pytest.mark.parametrize("free_all", [True, False])
@pytest.mark.usefixtures("swizzle_cache")
def test_discrete_weight_cache_released_with_weights(free_all):
    torch.manual_seed(0)
    weights = _make_weights(3, 256, 512)
    tensor_utils.post_all_gather_processing(weights)
    result = tensor_utils.swizzle_scales_and_pack_ptrs_for_discrete_weights(
        weights, "nvfp4", "cuda", columnwise=False
    )
    (key,) = _discrete_cache_keys(weights)
    buffer_refs = [weakref.ref(t) for t in result]
    del result
    if free_all:
        del weights
    else:
        del weights[1]
    gc.collect()
    assert key not in tensor_utils._DISCRETE_WEIGHT_SWIZZLE_CACHE
    assert all(ref() is None for ref in buffer_refs)


@pytest.fixture
def fused_grouped_mlp():
    """Skip unless the fused grouped MLP (Blackwell/Rubin, cuDNN frontend) is available."""
    if get_device_compute_capability()[0] != 10 or not GroupedMLP_CuTeGEMMGLU.is_supported():
        pytest.skip("Fused grouped MLP needs Blackwell/Rubin and a supported cuDNN frontend")


_NUM_GROUPS = 4
_HIDDEN = 256
_SPLITS = (256, 512, 0, 768)


def _mlp_recipe():
    return NVFP4BlockScaling(disable_rht=False, disable_stochastic_rounding=True)


def _make_mlp(quantized_weights):
    """Fused grouped MLP (discrete FC1/FC2 expert weights) and its weight params."""
    recipe = _mlp_recipe()
    with te.quantized_model_init(enabled=quantized_weights, recipe=recipe):
        fc1 = te.ops.GroupedLinear(
            _NUM_GROUPS,
            _HIDDEN,
            2 * _HIDDEN,
            bias=False,
            device="cuda",
            dtype=torch.bfloat16,
            split_alignment=256,
        )
        fc2 = te.ops.GroupedLinear(
            _NUM_GROUPS,
            _HIDDEN,
            _HIDDEN,
            bias=False,
            device="cuda",
            dtype=torch.bfloat16,
            split_alignment=256,
        )
    module = te.ops.Sequential(fc1, te.ops.ScaledSwiGLU(glu_interleave_size=32), fc2)
    weights = [getattr(fc, f"weight{i}") for fc in (fc1, fc2) for i in range(_NUM_GROUPS)]
    return module, weights


def _set_mlp_weights(weights, seed):
    """Deterministic weight update followed by the post-all-gather refresh."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        for w in weights:
            value = torch.rand(w.shape, generator=gen, device="cuda", dtype=torch.float32)
            # Random per-row magnitudes, so block scales differ between weight updates.
            magnitude = torch.rand(w.shape[0], 1, generator=gen, device="cuda") * 0.5 + 0.01
            value = ((value - 0.5) * magnitude).to(torch.bfloat16)
            if isinstance(w, QuantizedTensor):
                w.quantize_(value)
            else:
                w.copy_(value)
    tensor_utils.post_all_gather_processing(weights)


def _mlp_inputs(seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    tokens = sum(_SPLITS)
    x = torch.rand(tokens, _HIDDEN, generator=gen, device="cuda") * 0.5 - 0.25
    probs = torch.rand(tokens, generator=gen, device="cuda")
    dy = torch.rand(tokens, _HIDDEN, generator=gen, device="cuda") * 0.5 - 0.25
    return x.to(torch.bfloat16), probs.to(torch.bfloat16), dy.to(torch.bfloat16)


def _assert_fused(module):
    forward_ops = module._module_groups[0]._forward_ops
    assert len(forward_ops) == 1 and isinstance(forward_ops[0][0], GroupedMLP_CuTeGEMMGLU)


def _run_mlp_eager(module, weights, seed):
    x, probs, dy = _mlp_inputs(seed)
    x.requires_grad_(True)
    probs.requires_grad_(True)
    splits = torch.tensor(_SPLITS, dtype=torch.int, device="cuda")
    with te.autocast(enabled=True, recipe=_mlp_recipe()):
        y = module(x, splits, probs, splits)
    grads = torch.autograd.grad(y, [x, probs] + weights, dy)
    return [y.detach()] + [g.detach() for g in grads]


def _assert_bitwise(outs, refs):
    assert len(outs) == len(refs)
    for i, (out, ref) in enumerate(zip(outs, refs)):
        assert torch.equal(out, ref), f"tensor {i} differs"


@pytest.mark.usefixtures("fused_grouped_mlp")
def test_fused_grouped_mlp_eager(swizzle_cache):
    """Cached fused MLP matches the uncached one; entries go away with the module."""

    def run(cache_on):
        swizzle_cache(cache_on)
        torch.manual_seed(0)
        module, weights = _make_mlp(quantized_weights=True)
        _set_mlp_weights(weights, seed=1)
        outs = [_run_mlp_eager(module, weights, seed=3)]
        _assert_fused(module)
        _set_mlp_weights(weights, seed=2)
        outs.append(_run_mlp_eager(module, weights, seed=4))
        outs.append(_run_mlp_eager(module, weights, seed=5))
        return module, weights, outs

    _, _, refs = run(False)
    module, weights, outs = run(True)
    for out, ref in zip(outs, refs):
        _assert_bitwise(out, ref)
    keys = _discrete_cache_keys(weights)
    groups = {"fc1": weights[:_NUM_GROUPS], "fc2": weights[_NUM_GROUPS:]}
    cached = _cached_operands(weights, groups)
    print("cached discrete operands (columnwise, layer):", cached)
    # At least one forward (rowwise) and one backward (columnwise) operand is cached.
    assert {columnwise for columnwise, _ in cached} == {False, True}, cached

    del module, weights, outs, groups
    gc.collect()
    assert not any(key in tensor_utils._DISCRETE_WEIGHT_SWIZZLE_CACHE for key in keys)


@pytest.mark.usefixtures("swizzle_cache", "fused_grouped_mlp")
def test_fused_grouped_mlp_bf16_weights_not_cached():
    """BF16 expert weights quantized in every forward never enter the cache."""
    module, weights = _make_mlp(quantized_weights=False)
    _set_mlp_weights(weights, seed=1)
    before = len(tensor_utils._DISCRETE_WEIGHT_SWIZZLE_CACHE)
    for seed in (3, 4):
        _run_mlp_eager(module, weights, seed=seed)
    _assert_fused(module)
    assert len(tensor_utils._DISCRETE_WEIGHT_SWIZZLE_CACHE) == before


@pytest.mark.usefixtures("fused_grouped_mlp")
def test_fused_grouped_mlp_cuda_graph_replay(swizzle_cache):
    """A compute-only graph captured after eager warmup replays with updated weights.

    Warmup runs eagerly and fills the cache, so the capture hits it and records no swizzle.
    The weights are then updated through ``post_all_gather_processing`` and the graph is
    replayed: outputs and gradients must match the uncached path bit for bit.
    """
    tokens = sum(_SPLITS)

    def run(cache_on):
        swizzle_cache(cache_on)
        torch.manual_seed(0)
        module, weights = _make_mlp(quantized_weights=True)
        _set_mlp_weights(weights, seed=1)
        static_x = torch.zeros(tokens, _HIDDEN, device="cuda", dtype=torch.bfloat16)
        static_x.requires_grad_(True)
        static_probs = torch.zeros(tokens, device="cuda", dtype=torch.bfloat16)
        static_probs.requires_grad_(True)
        # FC1 and FC2 must see the same split tensor for the ops to fuse.
        splits = torch.tensor(_SPLITS, dtype=torch.int, device="cuda")
        with torch.no_grad():
            x, probs, _ = _mlp_inputs(seed=3)
            static_x.copy_(x)
            static_probs.copy_(probs)
        graphed = te.make_graphed_callables(
            module,
            (static_x, splits, static_probs, splits),
            num_warmup_iters=3,
            enabled=True,
            recipe=_mlp_recipe(),
        )
        _assert_fused(module)
        if cache_on:
            cached = {key[0] for key in _discrete_cache_keys(weights)}
            assert cached == {False, True}, cached

        def replay(input_seed):
            x, probs, dy = _mlp_inputs(input_seed)
            with torch.no_grad():
                static_x.copy_(x)
                static_probs.copy_(probs)
            static_x.grad = None
            static_probs.grad = None
            for w in weights:
                w.grad = None
            with te.autocast(enabled=True, recipe=_mlp_recipe()):
                y = graphed(static_x, splits, static_probs, splits)
            y.backward(dy)
            torch.cuda.synchronize()
            return [y.detach().clone(), static_x.grad.clone(), static_probs.grad.clone()] + [
                w.grad.clone() for w in weights
            ]

        outs = [replay(4)]
        _set_mlp_weights(weights, seed=2)
        outs.append(replay(4))
        outs.append(replay(5))
        return outs

    refs = run(False)
    outs = run(True)
    # The update must be visible, or the test could not detect stale scales.
    assert not torch.equal(refs[0][0], refs[1][0])
    for out, ref in zip(outs, refs):
        _assert_bitwise(out, ref)
