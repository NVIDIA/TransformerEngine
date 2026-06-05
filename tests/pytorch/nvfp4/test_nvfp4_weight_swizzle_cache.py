# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Tests for the cached-weight scale-swizzle optimization.

For block-scaled NVFP4 a weight participates in two GEMMs per step:

  * fprop: ``y = x @ Wt``   -> consumes the weight's **rowwise** scale factors
  * dgrad: ``dx = dY @ W``  -> consumes the weight's **columnwise** scale factors

cuBLAS/CUTLASS needs those scale factors in a GEMM-"swizzled" layout. Without
``optimize_for_gemm`` on the *weight* quantizer that swizzle is recomputed
lazily inside every GEMM and discarded, so with ``N`` micro-batches the weight
scale swizzle runs ``2*N`` times per step even though the weight is quantized
once. When the quantized weight is cached across micro-batches
(``is_first_microbatch`` is not ``None``) and FSDP is not in use, the module
sets ``weight_quantizer.optimize_for_gemm = True`` so the swizzle is done once
at quantize time, persisted on the cached workspace
(``_with_gemm_swizzled_scales = True``), and reused by every GEMM -> ``2``
swizzles per step instead of ``2*N``.

These tests verify that:

1. The optimization is **numerically a no-op**: swizzling is a pure layout
   permutation of the scale factors, so the cached (eager-swizzle) path must
   produce the same fprop output and dgrad as the un-cached (lazy-swizzle)
   baseline, for every distinct micro-batch.
2. The ``_with_gemm_swizzled_scales`` flag is actually set and persisted on the
   cached weight workspace.
"""

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import NVFP4BlockScaling


recipe_available, reason_for_no_recipe = te.is_nvfp4_available(return_reason=True)

pytestmark = pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)


def _make_module(kind, in_features, out_features, device):
    if kind == "Linear":
        return te.Linear(in_features, out_features, bias=True, params_dtype=torch.bfloat16).to(
            device
        )
    if kind == "LayerNormLinear":
        return te.LayerNormLinear(
            in_features, out_features, bias=True, params_dtype=torch.bfloat16
        ).to(device)
    raise ValueError(f"unknown module kind {kind}")


def _clone_params(src, dst):
    """Copy src's parameters into dst so both modules start identical."""
    with torch.no_grad():
        dst_params = dict(dst.named_parameters())
        for name, param in src.named_parameters():
            dst_params[name].copy_(param)


def _step(module, x, is_first, recipe):
    x = x.detach().clone().requires_grad_(True)
    module.zero_grad(set_to_none=True)  # per-micro-batch grads (no accumulation)
    with te.autocast(enabled=True, recipe=recipe):
        out = module(x, is_first_microbatch=is_first)
    out.sum().backward()
    return out.detach().float(), x.grad.detach().float()


@pytest.mark.parametrize("kind", ["Linear", "LayerNormLinear"])
@pytest.mark.parametrize("microbatches", [1, 4])
@pytest.mark.parametrize("shape", [(1024, 1024), (2048, 512)], ids=["1024x1024", "2048x512"])
def test_weight_swizzle_cache_numerics(kind, microbatches, shape):
    """Cached eager-swizzle path == lazy-swizzle baseline (fprop + dgrad)."""
    torch.manual_seed(1234)
    device = "cuda"
    in_features, out_features = shape
    batch = 512

    # Stochastic rounding is the only run-to-run nondeterminism source (RHT uses
    # a fixed sign mask) and it is applied to the bwd grad regardless of this
    # optimization, so disable it to make eager-vs-lazy weight swizzle
    # bit-comparable. The swizzle is a pure layout transform, so with SR off the
    # two paths must match tightly.
    recipe = NVFP4BlockScaling(disable_stochastic_rounding=True)

    # ref: always lazy-swizzle (is_first_microbatch=None => no weight cache =>
    # optimize_for_gemm stays False). opt: cached eager-swizzle path. Identical
    # weights so per-micro-batch outputs are directly comparable.
    ref = _make_module(kind, in_features, out_features, device)
    opt = _make_module(kind, in_features, out_features, device)
    _clone_params(ref, opt)

    # Distinct inputs per micro-batch (mirrors gradient accumulation: different
    # data each micro-batch, same weight).
    inputs = [
        torch.randn(batch, in_features, dtype=torch.bfloat16, device=device)
        for _ in range(microbatches)
    ]

    atol, rtol = 1e-3, 1e-3
    for mb in range(microbatches):
        ref_out, ref_dgrad = _step(ref, inputs[mb], None, recipe)
        opt_out, opt_dgrad = _step(opt, inputs[mb], mb == 0, recipe)
        torch.testing.assert_close(
            opt_out, ref_out, atol=atol, rtol=rtol, msg=f"fprop mismatch at mb {mb}"
        )
        torch.testing.assert_close(
            opt_dgrad, ref_dgrad, atol=atol, rtol=rtol, msg=f"dgrad mismatch at mb {mb}"
        )

    # The swizzled flag must be set & persisted on the cached weight workspace.
    workspaces = opt._fp8_workspaces
    assert workspaces, "no cached weight workspace was created on the optimized module"
    for name, ws in workspaces.items():
        assert getattr(ws, "_with_gemm_swizzled_scales", False) is True, (
            f"cached weight workspace {name!r} scales were not pre-swizzled "
            "(optimize_for_gemm not applied)"
        )


@pytest.mark.parametrize("microbatches", [1, 4])
@pytest.mark.parametrize("num_gemms", [1, 2])
def test_grouped_linear_weight_swizzle_cache_numerics(microbatches, num_gemms):
    """GroupedLinear (MoE expert path): cached eager-swizzle == lazy baseline."""
    torch.manual_seed(1234)
    device = "cuda"
    in_features, out_features = 1024, 1024
    tokens_per_gemm = 256
    batch = tokens_per_gemm * num_gemms
    m_splits = [tokens_per_gemm] * num_gemms

    recipe = NVFP4BlockScaling(disable_stochastic_rounding=True)

    ref = te.GroupedLinear(
        num_gemms, in_features, out_features, bias=True, params_dtype=torch.bfloat16
    ).to(device)
    opt = te.GroupedLinear(
        num_gemms, in_features, out_features, bias=True, params_dtype=torch.bfloat16
    ).to(device)
    _clone_params(ref, opt)

    inputs = [
        torch.randn(batch, in_features, dtype=torch.bfloat16, device=device)
        for _ in range(microbatches)
    ]

    def grouped_step(module, x, is_first):
        x = x.detach().clone().requires_grad_(True)
        module.zero_grad(set_to_none=True)
        with te.autocast(enabled=True, recipe=recipe):
            out = module(x, m_splits, is_first_microbatch=is_first)
        out.sum().backward()
        return out.detach().float(), x.grad.detach().float()

    atol, rtol = 1e-3, 1e-3
    for mb in range(microbatches):
        ref_out, ref_dgrad = grouped_step(ref, inputs[mb], None)
        opt_out, opt_dgrad = grouped_step(opt, inputs[mb], mb == 0)
        torch.testing.assert_close(
            opt_out, ref_out, atol=atol, rtol=rtol, msg=f"fprop mismatch at mb {mb}"
        )
        torch.testing.assert_close(
            opt_dgrad, ref_dgrad, atol=atol, rtol=rtol, msg=f"dgrad mismatch at mb {mb}"
        )

    workspaces = opt._fp8_workspaces
    assert len(workspaces) == num_gemms, "expected one cached workspace per expert"
    for name, ws in workspaces.items():
        assert (
            getattr(ws, "_with_gemm_swizzled_scales", False) is True
        ), f"cached weight workspace {name!r} scales were not pre-swizzled"


@pytest.mark.parametrize("kind", ["Linear", "LayerNormLinear"])
def test_lazy_path_not_swizzled(kind):
    """Without weight caching (is_first_microbatch=None) no workspace is created
    and the optimization stays off — guards against accidentally always-on."""
    torch.manual_seed(0)
    device = "cuda"
    recipe = NVFP4BlockScaling(disable_stochastic_rounding=True)
    module = _make_module(kind, 1024, 1024, device)
    x = torch.randn(512, 1024, dtype=torch.bfloat16, device=device, requires_grad=True)
    with te.autocast(enabled=True, recipe=recipe):
        out = module(x, is_first_microbatch=None)
    out.sum().backward()
    assert (
        not module._fp8_workspaces
    ), "lazy path (is_first_microbatch=None) must not populate the weight cache"
