# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Tests for cached NVFP4 weight scale swizzle (in-kernel swizzled SF output).

The NVFP4 2D quantize kernel writes GEMM-swizzled scale factors directly into
the cached weight's scale buffer, so there is no separate out-of-place swizzle
pass that could reallocate and rebind the scale pointer on a cache hit. These
tests guard that (a) the cached scale buffers keep a stable address across
weight updates, (b) weight caching stays numerically correct, and (c) a CUDA
graph that captured a GEMM against the cached weight survives an eager
re-quantize (the original hang/corruption)."""

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import NVFP4BlockScaling


recipe_available, reason_for_no_recipe = te.is_nvfp4_available(return_reason=True)

pytestmark = pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)


def _make_module(kind, in_features, out_features, device, num_gemms=1):
    common = dict(bias=True, params_dtype=torch.bfloat16)
    if kind == "Linear":
        return te.Linear(in_features, out_features, **common).to(device)
    if kind == "LayerNormLinear":
        return te.LayerNormLinear(in_features, out_features, **common).to(device)
    if kind == "LayerNormMLP":
        return te.LayerNormMLP(in_features, out_features, **common).to(device)
    if kind == "GroupedLinear":
        return te.GroupedLinear(num_gemms, in_features, out_features, **common).to(device)
    raise ValueError(f"unknown module kind {kind}")


def _clone_params(src, dst):
    with torch.no_grad():
        dst_params = dict(dst.named_parameters())
        for name, param in src.named_parameters():
            dst_params[name].copy_(param)


def _step(module, x, is_first, recipe, m_splits=None):
    x = x.detach().clone().requires_grad_(True)
    module.zero_grad(set_to_none=True)
    with te.autocast(enabled=True, recipe=recipe):
        if m_splits is None:
            out = module(x, is_first_microbatch=is_first)
        else:
            out = module(x, m_splits, is_first_microbatch=is_first)
    out.sum().backward()
    return out.detach().float(), x.grad.detach().float()


_MODULE_KINDS = ["Linear", "LayerNormLinear", "LayerNormMLP", "GroupedLinear"]


def _grouped_m_splits(kind, batch, num_gemms):
    if kind != "GroupedLinear":
        return None
    return [batch // num_gemms] * num_gemms


@pytest.mark.parametrize("kind", ["Linear", "LayerNormMLP"])
def test_cached_weight_scale_pointers_stable(kind):
    """Cached weight scale buffers must not be reallocated across weight updates."""
    torch.manual_seed(0)
    device = "cuda"
    batch = 512
    recipe = NVFP4BlockScaling(disable_stochastic_rounding=True)
    module = _make_module(kind, 1024, 1024, device)
    m_splits = _grouped_m_splits(kind, batch, 1)

    x0 = torch.randn(batch, 1024, dtype=torch.bfloat16, device=device)
    x1 = torch.randn(batch, 1024, dtype=torch.bfloat16, device=device)

    _step(module, x0, True, recipe, m_splits)
    workspaces = module._fp8_workspaces
    assert workspaces, "expected cached weight workspaces after first microbatch"

    ptrs_before = {}
    for name, ws in workspaces.items():
        assert getattr(ws, "_with_gemm_swizzled_scales", False)
        ptrs_before[name] = (
            ws._rowwise_scale_inv.data_ptr(),
            ws._columnwise_scale_inv.data_ptr(),
        )

    with torch.no_grad():
        for param in module.parameters():
            if param.ndim >= 2:
                param.add_(torch.randn_like(param) * 1e-3)

    _step(module, x1, True, recipe, m_splits)

    for name, ws in workspaces.items():
        ptrs_after = (
            ws._rowwise_scale_inv.data_ptr(),
            ws._columnwise_scale_inv.data_ptr(),
        )
        assert ptrs_after == ptrs_before[name], (
            f"cached weight workspace {name!r} reallocated scale buffers on weight update"
        )
        assert getattr(ws, "_with_gemm_swizzled_scales", False)


@pytest.mark.parametrize("kind", _MODULE_KINDS)
@pytest.mark.parametrize("microbatches", [1, 4])
def test_weight_swizzle_cache_numerics(kind, microbatches):
    torch.manual_seed(1234)
    device = "cuda"
    in_features, out_features = 1024, 1024
    batch = 512
    num_gemms = 2 if kind == "GroupedLinear" else 1
    m_splits = _grouped_m_splits(kind, batch, num_gemms)
    recipe = NVFP4BlockScaling(disable_stochastic_rounding=True)

    ref = _make_module(kind, in_features, out_features, device, num_gemms)
    opt = _make_module(kind, in_features, out_features, device, num_gemms)
    _clone_params(ref, opt)

    inputs = [
        torch.randn(batch, in_features, dtype=torch.bfloat16, device=device)
        for _ in range(microbatches)
    ]

    atol, rtol = 1e-3, 1e-3
    for mb in range(microbatches):
        ref_out, ref_dgrad = _step(ref, inputs[mb], None, recipe, m_splits)
        opt_out, opt_dgrad = _step(opt, inputs[mb], mb == 0, recipe, m_splits)
        torch.testing.assert_close(opt_out, ref_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(opt_dgrad, ref_dgrad, atol=atol, rtol=rtol)


@pytest.mark.parametrize("kind", ["Linear", "LayerNormLinear"])
def test_cached_weight_swizzle_scale_cuda_graph_replay(kind):
    """A captured CUDA graph must stay valid after an eager cached-weight update.

    Reproduces the reviewer's hang/corruption: a CUDA graph captures a forward
    that consumes the cached weight (``is_first_microbatch=False`` -> no
    re-quantize inside the graph), so the captured GEMM hard-codes the swizzled
    scale address. The next batch's first microbatch re-quantizes the same cached
    tensor eagerly (outside the graph). On the buggy path this reallocates the
    swizzled scale and frees the captured address, so graph replay reads stale
    memory. With persistent scale buffers the address is fixed and replay is
    unaffected.
    """
    torch.manual_seed(0)
    device = "cuda"
    batch = 512
    recipe = NVFP4BlockScaling(disable_stochastic_rounding=True)
    module = _make_module(kind, 1024, 1024, device)

    # Populate the weight cache (first microbatch).
    x_warm = torch.randn(batch, 1024, dtype=torch.bfloat16, device=device)
    with te.autocast(enabled=True, recipe=recipe):
        module(x_warm, is_first_microbatch=True)

    ws = module._fp8_workspaces["weight"]
    addr_before = ws._rowwise_scale_inv.data_ptr()

    # Static input the captured graph reads from.
    x_static = torch.randn(batch, 1024, dtype=torch.bfloat16, device=device)

    # Warmup on a side stream (required before graph capture), then capture a
    # forward that consumes the cached weight without re-quantizing it.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            with te.autocast(enabled=True, recipe=recipe):
                module(x_static, is_first_microbatch=False)
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with te.autocast(enabled=True, recipe=recipe):
            y_static = module(x_static, is_first_microbatch=False)

    graph.replay()
    torch.cuda.synchronize()

    # Next batch's first microbatch: eager weight update + re-quantize of the
    # cached weight. With persistent buffers this rewrites the SAME scale buffer
    # in place; the buggy path reallocates it and frees the captured address.
    with torch.no_grad():
        module.weight.add_(torch.randn_like(module.weight) * 0.5)
    with te.autocast(enabled=True, recipe=recipe):
        module(
            torch.randn(batch, 1024, dtype=torch.bfloat16, device=device),
            is_first_microbatch=True,
        )
    addr_after = ws._rowwise_scale_inv.data_ptr()

    # Ground truth for the updated weights: an eager forward that consumes the
    # (now updated) cached weight, i.e. what graph replay must reproduce.
    with te.autocast(enabled=True, recipe=recipe):
        expected = module(x_static, is_first_microbatch=False).detach().float().clone()

    # Churn the allocator so a freed scale buffer would be reused, turning a
    # dangling captured pointer into observable corruption on replay.
    _churn = [torch.randn(4096, 1024, dtype=torch.bfloat16, device=device) for _ in range(8)]
    torch.cuda.synchronize()

    graph.replay()
    torch.cuda.synchronize()
    replayed = y_static.detach().float().clone()

    assert addr_after == addr_before, (
        "cached weight swizzled scale buffer moved on eager re-quantize; a captured "
        "CUDA graph would replay against a stale address"
    )
    # Replay reads the persistent scale buffer in place, so it must match an
    # eager forward with the updated weights. On the buggy path it instead reads
    # freed/reused memory and diverges.
    torch.testing.assert_close(replayed, expected, atol=1e-2, rtol=1e-2)
    del graph
