# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""
Tests for eager GEMM-swizzling of weight scale factors during quantization.
Except for the case when primary weights are in fp8, we preswizzle the weights during quantization
instead of lazily swizzling inside every GEMM.
"""

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import MXFP8BlockScaling, NVFP4BlockScaling


mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
nvfp4_available, reason_for_no_nvfp4 = te.is_nvfp4_available(return_reason=True)

# Every recipe whose scale factors require a GEMM swizzle. Each entry is
# skipped individually when the hardware/recipe is unavailable.
_SWIZZLING_RECIPES = [
    pytest.param(
        MXFP8BlockScaling,
        marks=pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8),
        id="mxfp8",
    ),
    pytest.param(
        NVFP4BlockScaling,
        marks=pytest.mark.skipif(not nvfp4_available, reason=reason_for_no_nvfp4),
        id="nvfp4",
    ),
]

_LAYER_TYPES = ["Linear", "LayerNormLinear", "LayerNormMLP", "GroupedLinear"]


def _make_module(layer_type, in_features, out_features, device, num_gemms=1):
    common = dict(bias=True, params_dtype=torch.bfloat16)
    if layer_type == "Linear":
        return te.Linear(in_features, out_features, **common).to(device)
    if layer_type == "LayerNormLinear":
        return te.LayerNormLinear(in_features, out_features, **common).to(device)
    if layer_type == "LayerNormMLP":
        # fc1 (in->ffn) and fc2 (ffn->in) each own a weight, so two of each.
        return te.LayerNormMLP(in_features, out_features, **common).to(device)
    if layer_type == "GroupedLinear":
        return te.GroupedLinear(num_gemms, in_features, out_features, **common).to(device)
    raise ValueError(f"unknown layer type {layer_type}")


def _expected_num_weights(layer_type, num_gemms):
    """Number of quantized weights a module owns (workspaces / weight params)."""
    if layer_type == "GroupedLinear":
        return num_gemms  # one per expert
    if layer_type == "LayerNormMLP":
        return 2  # fc1 + fc2
    return 1


def _grouped_m_splits(layer_type, batch, num_gemms):
    """m_splits for GroupedLinear (even token split across experts), else None."""
    if layer_type != "GroupedLinear":
        return None
    return [batch // num_gemms] * num_gemms


def _make_recipe(recipe_cls):
    """Instantiate a recipe with run-to-run nondeterminism disabled where it
    exists (NVFP4 stochastic rounding); MXFP8 has none."""
    if recipe_cls is NVFP4BlockScaling:
        return recipe_cls(disable_stochastic_rounding=True)
    return recipe_cls()


def _forward_backward(module, x, is_first_microbatch, recipe, m_splits):
    """Run one fwd+bwd and return the weight quantizers (captured inside the
    autocast, where the module reports fp8 as enabled)."""
    with te.autocast(enabled=True, recipe=recipe):
        if m_splits is None:
            out = module(x, is_first_microbatch=is_first_microbatch)
        else:
            out = module(x, m_splits, is_first_microbatch=is_first_microbatch)
        weight_quantizers = module._get_weight_quantizers()
    out.sum().backward()
    return weight_quantizers


def _clone_params(src, dst):
    """Copy src's parameters into dst so both modules start identical."""
    with torch.no_grad():
        dst_params = dict(dst.named_parameters())
        for name, param in src.named_parameters():
            dst_params[name].copy_(param)


def _run_step(module, x, is_first_microbatch, recipe, m_splits):
    """One fwd+bwd; returns (output, dgrad, [wgrads]). Grads are reset each call
    so each microbatch is compared independently (no accumulation)."""
    x = x.detach().clone().requires_grad_(True)
    module.zero_grad(set_to_none=True)
    with te.autocast(enabled=True, recipe=recipe):
        if m_splits is None:
            out = module(x, is_first_microbatch=is_first_microbatch)
        else:
            out = module(x, m_splits, is_first_microbatch=is_first_microbatch)
    out.sum().backward()
    wgrads = [p.grad.detach().clone() for _, p in module.named_parameters() if p.grad is not None]
    return out.detach().clone(), x.grad.detach().clone(), wgrads


@pytest.mark.parametrize("layer_type", _LAYER_TYPES)
@pytest.mark.parametrize("recipe_cls", _SWIZZLING_RECIPES)
def test_weight_swizzling_with_workspace_caching(layer_type, recipe_cls):
    """When direct swizzle fusion is supported, cached weights must pre-swizzle scales.
    Generic across every swizzling recipe (MXFP8, NVFP4) and module type. Uses
    128-aligned weight shapes so NVFP4 2D swizzle fusion is eligible.
    """
    torch.manual_seed(1234)
    device = "cuda"
    in_features, out_features = 1024, 1024
    batch = 512
    num_gemms = 2 if layer_type == "GroupedLinear" else 1
    m_splits = _grouped_m_splits(layer_type, batch, num_gemms)
    recipe = recipe_cls()

    module = _make_module(layer_type, in_features, out_features, device, num_gemms)
    x = torch.randn(batch, in_features, dtype=torch.bfloat16, device=device, requires_grad=True)

    # is_first_microbatch=True caches the quantized weight.
    weight_quantizers = _forward_backward(module, x, True, recipe, m_splits)

    for weight_quantizer in weight_quantizers:
        assert weight_quantizer is not None
        assert (
            weight_quantizer.optimize_for_gemm is True
        ), f"optimize_for_gemm must be enabled for cached {layer_type} weights"

    workspaces = module._fp8_workspaces
    assert len(workspaces) == _expected_num_weights(
        layer_type, num_gemms
    ), f"unexpected cached weight workspace count for {layer_type}: {len(workspaces)}"
    for name, ws in workspaces.items():
        assert (
            getattr(ws, "_with_gemm_swizzled_scales", False) is True
        ), f"cached weight workspace {name!r} scales were not pre-swizzled"


@pytest.mark.parametrize("layer_type", _LAYER_TYPES)
@pytest.mark.parametrize("recipe_cls", _SWIZZLING_RECIPES)
def test_weight_swizzling_with_primary_fp8_weights(layer_type, recipe_cls):
    """With quantized_model_init the weight parameter is itself quantized and is
    all-gathered (FSDP2) / optimizer-updated in its unswizzled layout, so the
    eager-swizzle optimization must stay off: the weight quantizer must keep
    ``optimize_for_gemm`` disabled and the weight parameter must not have
    GEMM-swizzled scales. Workspace caching is not a use case here, so
    ``is_first_microbatch`` is irrelevant and left unset.
    """
    torch.manual_seed(1234)
    device = "cuda"
    in_features, out_features = 1024, 1024
    batch = 512
    num_gemms = 2 if layer_type == "GroupedLinear" else 1
    m_splits = _grouped_m_splits(layer_type, batch, num_gemms)
    recipe = recipe_cls()

    with te.quantized_model_init(enabled=True, recipe=recipe):
        module = _make_module(layer_type, in_features, out_features, device, num_gemms)
    x = torch.randn(batch, in_features, dtype=torch.bfloat16, device=device, requires_grad=True)

    weight_quantizers = _forward_backward(module, x, None, recipe, m_splits)

    for weight_quantizer in weight_quantizers:
        assert weight_quantizer is not None
        assert (
            weight_quantizer.optimize_for_gemm is False
        ), "quantized_model_init must not enable optimize_for_gemm on the weight quantizer"

    weight_params = [
        p for _, p in module.named_parameters() if hasattr(p, "_with_gemm_swizzled_scales")
    ]
    assert len(weight_params) == _expected_num_weights(
        layer_type, num_gemms
    ), f"unexpected quantized weight param count for {layer_type}: {len(weight_params)}"
    for w in weight_params:
        assert (
            w._with_gemm_swizzled_scales is False
        ), "quantized_model_init weight parameter must not have GEMM-swizzled scales"


@pytest.mark.parametrize("layer_type", ["Linear", "LayerNormLinear", "GroupedLinear"])
@pytest.mark.skipif(not nvfp4_available, reason=reason_for_no_nvfp4)
def test_weight_optimize_for_gemm_disabled_without_swizzle_fusion(layer_type):
    """NVFP4 weights that cannot use in-kernel swizzle fusion must keep compact cached scales."""
    torch.manual_seed(1234)
    device = "cuda"
    # Valid for NVFP4 quantization, but not aligned to 128x128 swizzle tiles.
    in_features, out_features = 1056, 1056
    batch = 512
    num_gemms = 2 if layer_type == "GroupedLinear" else 1
    m_splits = _grouped_m_splits(layer_type, batch, num_gemms)
    recipe = NVFP4BlockScaling(disable_stochastic_rounding=True)

    module = _make_module(layer_type, in_features, out_features, device, num_gemms)
    x = torch.randn(batch, in_features, dtype=torch.bfloat16, device=device, requires_grad=True)

    weight_quantizers = _forward_backward(module, x, True, recipe, m_splits)

    for weight_quantizer in weight_quantizers:
        assert weight_quantizer is not None
        assert weight_quantizer.optimize_for_gemm is False

    for _, ws in module._fp8_workspaces.items():
        assert getattr(ws, "_with_gemm_swizzled_scales", False) is False


@pytest.mark.skipif(not nvfp4_available, reason=reason_for_no_nvfp4)
def test_weight_optimize_for_gemm_disabled_without_nvfp4_2d_quantization():
    """NVFP4 weights with 2D quantization disabled cannot preswizzle at quantize time."""
    torch.manual_seed(1234)
    device = "cuda"
    in_features, out_features = 1024, 1024
    batch = 512
    recipe = NVFP4BlockScaling(
        disable_stochastic_rounding=True,
        disable_2d_quantization=True,
    )

    module = te.Linear(in_features, out_features, bias=True, params_dtype=torch.bfloat16).to(device)
    x = torch.randn(batch, in_features, dtype=torch.bfloat16, device=device, requires_grad=True)

    weight_quantizers = _forward_backward(module, x, True, recipe, None)

    assert weight_quantizers[0].optimize_for_gemm is False
    for _, ws in module._fp8_workspaces.items():
        assert getattr(ws, "_with_gemm_swizzled_scales", False) is False


@pytest.mark.parametrize("layer_type", _LAYER_TYPES)
@pytest.mark.parametrize("recipe_cls", _SWIZZLING_RECIPES)
def test_weight_caching_matches_no_caching(layer_type, recipe_cls):
    """Caching the quantized (pre-swizzled) weight across microbatches must be
    numerically identical to the uncached flow that re-quantizes the weight every
    microbatch. Verified per microbatch for fprop output, dgrad and wgrad, across
    every swizzling recipe (MXFP8, NVFP4) and module type.

    The two paths are bit-comparable: the weights are constant (no optimizer
    step), so re-quantization yields the same FP4/MXFP8 values, and swizzling is
    a pure layout permutation -- the only difference is whether the swizzled
    weight is computed once and cached or recomputed every microbatch.
    """
    torch.manual_seed(1234)
    device = "cuda"
    in_features, out_features = 1024, 1024
    batch = 512
    microbatches = 4
    num_gemms = 2 if layer_type == "GroupedLinear" else 1
    m_splits = _grouped_m_splits(layer_type, batch, num_gemms)
    recipe = _make_recipe(recipe_cls)

    # Identical modules: one drives the cached path, one the uncached path.
    cached = _make_module(layer_type, in_features, out_features, device, num_gemms)
    uncached = _make_module(layer_type, in_features, out_features, device, num_gemms)
    _clone_params(cached, uncached)

    # Distinct inputs per microbatch (mirrors gradient accumulation: different
    # data each microbatch, same weight).
    inputs = [
        torch.randn(batch, in_features, dtype=torch.bfloat16, device=device)
        for _ in range(microbatches)
    ]

    for mb in range(microbatches):
        # cached: is_first_microbatch=True on mb 0 quantizes+swizzles the weight
        # once and caches it; later microbatches reuse the cached workspace.
        # uncached: is_first_microbatch=None re-quantizes+re-swizzles every step.
        c_out, c_dgrad, c_wgrads = _run_step(cached, inputs[mb], mb == 0, recipe, m_splits)
        u_out, u_dgrad, u_wgrads = _run_step(uncached, inputs[mb], None, recipe, m_splits)
        torch.testing.assert_close(
            c_out, u_out, atol=0, rtol=0, msg=f"fprop output mismatch at microbatch {mb}"
        )
        torch.testing.assert_close(
            c_dgrad, u_dgrad, atol=0, rtol=0, msg=f"dgrad mismatch at microbatch {mb}"
        )
        assert len(c_wgrads) == len(u_wgrads)
        for i, (cg, ug) in enumerate(zip(c_wgrads, u_wgrads)):
            torch.testing.assert_close(
                cg, ug, atol=0, rtol=0, msg=f"wgrad[{i}] mismatch at microbatch {mb}"
            )
