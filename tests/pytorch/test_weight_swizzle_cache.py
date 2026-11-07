# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""
Tests for caching GEMM-swizzled scales of primary quantized weights.

Primary MXFP8/NVFP4 weights (``quantized_model_init``) keep compact scales for
all-gather, dequantization and checkpointing. With ``is_first_microbatch`` set,
the modules swizzle those scales once into a cached view on the first
micro-batch and reuse it for the remaining micro-batches.
"""

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.common.recipe import MXFP8BlockScaling, NVFP4BlockScaling
from transformer_engine.pytorch.module.grouped_linear import is_module_grouped_tensor_path_supported
from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor


mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
nvfp4_available, reason_for_no_nvfp4 = te.is_nvfp4_available(return_reason=True)

_RECIPES = [
    pytest.param(
        "mxfp8",
        marks=pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8),
    ),
    pytest.param(
        "nvfp4",
        marks=pytest.mark.skipif(not nvfp4_available, reason=reason_for_no_nvfp4),
    ),
]

_LAYER_TYPES = ["Linear", "LayerNormLinear", "LayerNormMLP", "GroupedLinear"]

_FEATURES = 1024
_BATCH = 256
_NUM_GEMMS = 2


def _make_recipe(recipe_name):
    if recipe_name == "mxfp8":
        return MXFP8BlockScaling()
    return NVFP4BlockScaling(disable_stochastic_rounding=True)


def _make_module(layer_type, recipe):
    common = dict(bias=True, params_dtype=torch.bfloat16, device="cuda")
    with te.quantized_model_init(enabled=True, recipe=recipe):
        if layer_type == "Linear":
            return te.Linear(_FEATURES, _FEATURES, **common)
        if layer_type == "LayerNormLinear":
            return te.LayerNormLinear(_FEATURES, _FEATURES, **common)
        if layer_type == "LayerNormMLP":
            return te.LayerNormMLP(_FEATURES, _FEATURES, **common)
        if layer_type == "GroupedLinear":
            return te.GroupedLinear(_NUM_GEMMS, _FEATURES, _FEATURES, **common)
    raise ValueError(f"unknown layer type {layer_type}")


def _quantized_weights(module):
    return [p for _, p in module.named_parameters() if hasattr(p, "_with_gemm_swizzled_scales")]


def _run_step(module, x, is_first_microbatch, recipe):
    """One fwd+bwd; returns (output, dgrad, [wgrads]) with grads reset per call."""
    x = x.detach().clone().requires_grad_(True)
    module.zero_grad(set_to_none=True)
    with te.autocast(enabled=True, recipe=recipe):
        if isinstance(module, te.GroupedLinear):
            m_splits = [_BATCH // _NUM_GEMMS] * _NUM_GEMMS
            if module.single_grouped_weight:
                # A single grouped weight requires device-side splits.
                m_splits = torch.tensor(m_splits, dtype=torch.int64, device="cuda")
            out = module(x, m_splits, is_first_microbatch=is_first_microbatch)
        else:
            out = module(x, is_first_microbatch=is_first_microbatch)
    out.sum().backward()
    wgrads = [p.grad.detach().clone() for _, p in module.named_parameters() if p.grad is not None]
    return out.detach().clone(), x.grad.detach().clone(), wgrads


def _assert_steps_equal(actual, expected, what):
    a_out, a_dgrad, a_wgrads = actual
    e_out, e_dgrad, e_wgrads = expected
    torch.testing.assert_close(a_out, e_out, atol=0, rtol=0, msg=f"fprop mismatch ({what})")
    torch.testing.assert_close(a_dgrad, e_dgrad, atol=0, rtol=0, msg=f"dgrad mismatch ({what})")
    assert len(a_wgrads) == len(e_wgrads)
    for i, (a, e) in enumerate(zip(a_wgrads, e_wgrads)):
        torch.testing.assert_close(a, e, atol=0, rtol=0, msg=f"wgrad[{i}] mismatch ({what})")


def _scale_snapshot(weight):
    return [
        (s.data_ptr(), s.clone())
        for s in (weight._rowwise_scale_inv, weight._columnwise_scale_inv)
        if s is not None
    ]


def _count_swizzle_kernels(fn):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(1 for e in prof.events() if "swizzle" in e.name.lower())


@pytest.mark.parametrize("layer_type", _LAYER_TYPES)
@pytest.mark.parametrize("recipe_name", _RECIPES)
def test_cached_view_matches_uncached(layer_type, recipe_name):
    """True/False micro-batches match swizzling in every GEMM, and the parameter is untouched."""
    torch.manual_seed(1234)
    recipe = _make_recipe(recipe_name)
    module = _make_module(layer_type, recipe)
    weights = _quantized_weights(module)
    assert weights, "expected quantized weight parameters"
    snapshots = [_scale_snapshot(w) for w in weights]

    inputs = [torch.randn(_BATCH, _FEATURES, dtype=torch.bfloat16, device="cuda") for _ in range(3)]
    for mb, x in enumerate(inputs):
        expected = _run_step(module, x, None, recipe)
        actual = _run_step(module, x, mb == 0, recipe)
        _assert_steps_equal(actual, expected, f"microbatch {mb}")

    # One cached swizzled view per quantized weight.
    views = list(module._fp8_workspaces.values())
    assert len(views) == len(weights)
    for view in views:
        assert view._with_gemm_swizzled_scales is True

    # The parameters keep their compact scales, in place.
    for w, snapshot in zip(weights, snapshots):
        assert w._with_gemm_swizzled_scales is False
        for (ptr, values), scales in zip(
            snapshot, (s for s in (w._rowwise_scale_inv, w._columnwise_scale_inv) if s is not None)
        ):
            assert scales.data_ptr() == ptr
            torch.testing.assert_close(scales, values, atol=0, rtol=0)


@pytest.mark.parametrize("layer_type", _LAYER_TYPES)
@pytest.mark.parametrize("recipe_name", _RECIPES)
def test_no_weight_swizzle_on_followup_microbatch(layer_type, recipe_name):
    """Micro-batches after the first launch fewer swizzle kernels than the uncached path."""
    torch.manual_seed(1234)
    recipe = _make_recipe(recipe_name)
    module = _make_module(layer_type, recipe)
    x = torch.randn(_BATCH, _FEATURES, dtype=torch.bfloat16, device="cuda")

    _run_step(module, x, True, recipe)
    cached = _count_swizzle_kernels(lambda: _run_step(module, x, False, recipe))
    uncached = _count_swizzle_kernels(lambda: _run_step(module, x, None, recipe))
    # Uncached, each weight is swizzled in fprop and dgrad; GroupedLinear batches all
    # experts into one launch per GEMM.
    num_weight_launches = 1 if layer_type == "GroupedLinear" else len(_quantized_weights(module))
    assert uncached - cached >= 2 * num_weight_launches, (cached, uncached)


@pytest.mark.parametrize("layer_type", _LAYER_TYPES)
@pytest.mark.parametrize("recipe_name", _RECIPES)
def test_refresh_after_inplace_update(layer_type, recipe_name):
    """After an in-place weight update, the first micro-batch refreshes the cached scales."""
    torch.manual_seed(1234)
    recipe = _make_recipe(recipe_name)
    module = _make_module(layer_type, recipe)
    x = torch.randn(_BATCH, _FEATURES, dtype=torch.bfloat16, device="cuda")

    _run_step(module, x, True, recipe)
    _run_step(module, x, False, recipe)
    views_before = {k: v for k, v in module._fp8_workspaces.items()}

    # Optimizer-style in-place update of every quantized weight.
    with torch.no_grad():
        for w in _quantized_weights(module):
            w.quantize_(torch.randn(w.shape, dtype=torch.bfloat16, device="cuda"))

    actual = _run_step(module, x, True, recipe)
    expected = _run_step(module, x, None, recipe)
    _assert_steps_equal(actual, expected, "first micro-batch after update")
    _assert_steps_equal(
        _run_step(module, x, False, recipe), expected, "second micro-batch after update"
    )
    # The refresh reuses the cached views and their buffers.
    for key, view in module._fp8_workspaces.items():
        assert view is views_before[key]


@pytest.mark.parametrize("recipe_name", _RECIPES)
def test_view_rebuilt_on_pointer_change(recipe_name):
    """Replacing a weight's data buffer rebuilds the cached view instead of reading old memory."""
    torch.manual_seed(1234)
    recipe = _make_recipe(recipe_name)
    module = _make_module("Linear", recipe)
    x = torch.randn(_BATCH, _FEATURES, dtype=torch.bfloat16, device="cuda")

    _run_step(module, x, True, recipe)
    old_view = module._fp8_workspaces["weight"]
    weight = module.weight
    weight._rowwise_data = torch.zeros_like(weight._rowwise_data)

    actual = _run_step(module, x, False, recipe)
    expected = _run_step(module, x, None, recipe)
    _assert_steps_equal(actual, expected, "after buffer replacement")
    assert module._fp8_workspaces["weight"] is not old_view


@pytest.mark.parametrize("recipe_name", _RECIPES)
def test_swizzle_scales_into_noop(recipe_name):
    """``swizzle_scales_into_`` honors the device-side noop flag."""
    torch.manual_seed(1234)
    recipe = _make_recipe(recipe_name)
    module = _make_module("Linear", recipe)
    weight = module.weight
    x = torch.randn(_BATCH, _FEATURES, dtype=torch.bfloat16, device="cuda")
    _run_step(module, x, True, recipe)
    view = module._fp8_workspaces["weight"]
    view_scales = [
        s for s in (view._rowwise_scale_inv, view._columnwise_scale_inv) if s is not None
    ]
    reference = [s.clone() for s in view_scales]

    for s in view_scales:
        s.fill_(0xAB)
    tex.swizzle_scales_into_(weight, view, torch.ones(1, dtype=torch.float32, device="cuda"))
    for s in view_scales:
        assert bool((s == 0xAB).all()), "noop=1 must leave the destination untouched"

    tex.swizzle_scales_into_(weight, view, torch.zeros(1, dtype=torch.float32, device="cuda"))
    for s, ref in zip(view_scales, reference):
        torch.testing.assert_close(s, ref, atol=0, rtol=0)


@pytest.mark.parametrize("recipe_name", _RECIPES)
def test_grouped_refresh_is_batched(recipe_name):
    """A first-micro-batch refresh of GroupedLinear experts uses no more swizzle launches than
    the uncached path, which swizzles all experts in one launch per GEMM."""
    torch.manual_seed(1234)
    recipe = _make_recipe(recipe_name)
    module = _make_module("GroupedLinear", recipe)
    x = torch.randn(_BATCH, _FEATURES, dtype=torch.bfloat16, device="cuda")

    _run_step(module, x, True, recipe)
    refresh = _count_swizzle_kernels(lambda: _run_step(module, x, True, recipe))
    uncached = _count_swizzle_kernels(lambda: _run_step(module, x, None, recipe))
    assert refresh <= uncached, (refresh, uncached)


@pytest.mark.parametrize("recipe_name", _RECIPES)
def test_multi_tensor_swizzle_scales_into_noop(recipe_name):
    """``multi_tensor_swizzle_scales_into_`` honors the device-side noop flag for all tensors."""
    torch.manual_seed(1234)
    recipe = _make_recipe(recipe_name)
    module = _make_module("GroupedLinear", recipe)
    x = torch.randn(_BATCH, _FEATURES, dtype=torch.bfloat16, device="cuda")
    _run_step(module, x, True, recipe)
    weights = _quantized_weights(module)
    views = [module._fp8_workspaces[f"weight{i}"] for i in range(len(weights))]
    view_scales = [
        s
        for view in views
        for s in (view._rowwise_scale_inv, view._columnwise_scale_inv)
        if s is not None
    ]
    reference = [s.clone() for s in view_scales]

    for s in view_scales:
        s.fill_(0xAB)
    tex.multi_tensor_swizzle_scales_into_(
        weights, views, torch.ones(1, dtype=torch.float32, device="cuda")
    )
    for s in view_scales:
        assert bool((s == 0xAB).all()), "noop=1 must leave the destinations untouched"

    tex.multi_tensor_swizzle_scales_into_(
        weights, views, torch.zeros(1, dtype=torch.float32, device="cuda")
    )
    for s, ref in zip(view_scales, reference):
        torch.testing.assert_close(s, ref, atol=0, rtol=0)


_single_grouped_weight_supported = mxfp8_available and is_module_grouped_tensor_path_supported(
    MXFP8BlockScaling(), torch.bfloat16
)


def _make_single_grouped_weight_module():
    with te.quantized_model_init(enabled=True, recipe=MXFP8BlockScaling()):
        return te.GroupedLinear(
            _NUM_GEMMS,
            _FEATURES,
            _FEATURES,
            bias=False,
            params_dtype=torch.bfloat16,
            device="cuda",
            single_grouped_weight=True,
            use_grouped_tensor=True,
        )


def _grouped_scales(tensor):
    return [s for s in (tensor.scale_inv, tensor.columnwise_scale_inv) if s is not None]


@pytest.mark.skipif(
    not _single_grouped_weight_supported, reason="MXFP8 GroupedTensor GEMM is unavailable."
)
def test_single_grouped_weight_matches_uncached():
    """A primary MXFP8 single grouped weight caches one swizzled GroupedTensor view."""
    torch.manual_seed(1234)
    recipe = MXFP8BlockScaling()
    module = _make_single_grouped_weight_module()
    weight = module.weight
    snapshot = [(s.data_ptr(), s.clone()) for s in _grouped_scales(weight)]

    inputs = [torch.randn(_BATCH, _FEATURES, dtype=torch.bfloat16, device="cuda") for _ in range(3)]
    for mb, x in enumerate(inputs):
        expected = _run_step(module, x, None, recipe)
        actual = _run_step(module, x, mb == 0, recipe)
        _assert_steps_equal(actual, expected, f"microbatch {mb}")

    view = module._fp8_workspaces["weight"]
    assert isinstance(view, GroupedTensor) and view._with_gemm_swizzled_scales
    assert view.rowwise_data.data_ptr() == weight.rowwise_data.data_ptr()
    assert weight._with_gemm_swizzled_scales is False
    for (ptr, values), scales in zip(snapshot, _grouped_scales(weight)):
        assert scales.data_ptr() == ptr
        torch.testing.assert_close(scales, values, atol=0, rtol=0)


@pytest.mark.skipif(
    not _single_grouped_weight_supported, reason="MXFP8 GroupedTensor GEMM is unavailable."
)
def test_single_grouped_weight_no_swizzle_on_followup_microbatch():
    """Micro-batches after the first skip the grouped weight swizzle."""
    torch.manual_seed(1234)
    recipe = MXFP8BlockScaling()
    module = _make_single_grouped_weight_module()
    x = torch.randn(_BATCH, _FEATURES, dtype=torch.bfloat16, device="cuda")

    _run_step(module, x, True, recipe)
    refresh = _count_swizzle_kernels(lambda: _run_step(module, x, True, recipe))
    cached = _count_swizzle_kernels(lambda: _run_step(module, x, False, recipe))
    uncached = _count_swizzle_kernels(lambda: _run_step(module, x, None, recipe))
    # Uncached, the weight is swizzled once in fprop and once in dgrad.
    assert uncached - cached >= 2, (cached, uncached)
    assert refresh <= uncached, (refresh, uncached)


@pytest.mark.skipif(
    not _single_grouped_weight_supported, reason="MXFP8 GroupedTensor GEMM is unavailable."
)
def test_single_grouped_weight_refresh_after_inplace_update():
    """After an in-place update of the grouped weight, the first micro-batch refreshes the view."""
    torch.manual_seed(1234)
    recipe = MXFP8BlockScaling()
    module = _make_single_grouped_weight_module()
    weight = module.weight
    x = torch.randn(_BATCH, _FEATURES, dtype=torch.bfloat16, device="cuda")

    _run_step(module, x, True, recipe)
    _run_step(module, x, False, recipe)
    view = module._fp8_workspaces["weight"]

    with torch.no_grad():
        source = torch.randn(weight.logical_shape, dtype=torch.bfloat16, device="cuda")
        tex.group_quantize(source, weight.quantizer, _NUM_GEMMS, None, output=weight)

    expected = _run_step(module, x, None, recipe)
    _assert_steps_equal(_run_step(module, x, True, recipe), expected, "first after update")
    _assert_steps_equal(_run_step(module, x, False, recipe), expected, "second after update")
    assert module._fp8_workspaces["weight"] is view


@pytest.mark.skipif(
    not _single_grouped_weight_supported, reason="MXFP8 GroupedTensor GEMM is unavailable."
)
def test_grouped_swizzle_scales_into_noop():
    """``grouped_swizzle_scales_into_`` honors the device-side noop flag."""
    torch.manual_seed(1234)
    recipe = MXFP8BlockScaling()
    module = _make_single_grouped_weight_module()
    x = torch.randn(_BATCH, _FEATURES, dtype=torch.bfloat16, device="cuda")
    _run_step(module, x, True, recipe)
    weight, view = module.weight, module._fp8_workspaces["weight"]
    view_scales = _grouped_scales(view)
    reference = [s.clone() for s in view_scales]

    for s in view_scales:
        s.fill_(0xAB)
    tex.grouped_swizzle_scales_into_(
        weight, view, torch.ones(1, dtype=torch.float32, device="cuda")
    )
    for s in view_scales:
        assert bool((s == 0xAB).all()), "noop=1 must leave the destination untouched"

    tex.grouped_swizzle_scales_into_(
        weight, view, torch.zeros(1, dtype=torch.float32, device="cuda")
    )
    for s, ref in zip(view_scales, reference):
        torch.testing.assert_close(s, ref, atol=0, rtol=0)
