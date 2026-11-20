# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Gradient fan-in for te.LayerNormLinear: RMSNorm backward adds the other consumers' sum."""

from typing import List, Optional, Tuple

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from transformer_engine.pytorch import is_mxfp8_available

mxfp8_available, reason_for_no_mxfp8 = is_mxfp8_available(return_reason=True)

S, B, H, F = 256, 2, 512, 384


class FanIn:
    """Minimal fan-in implementing the protocol of Linear/LayerNormLinear.supports_grad_fanin."""

    def __init__(self, num_consumers: int):
        self.num_consumers = num_consumers
        self.acc: Optional[torch.Tensor] = None
        self.finished: List[Tuple[bool, bool]] = []  # (accumulated, returned_sum)

    def accum_target(self) -> Optional[torch.Tensor]:
        return self.acc

    def partial_sum(self) -> Optional[torch.Tensor]:
        return self.acc

    def finish(self, grad: Optional[torch.Tensor], accumulated: bool) -> Optional[torch.Tensor]:
        if grad is not None:
            if accumulated or self.acc is None:
                self.acc = grad
            else:
                self.acc = self.acc + grad
        last = len(self.finished) + 1 == self.num_consumers
        self.finished.append((accumulated, last))
        if last:
            out, self.acc = self.acc, None
            return out
        return None


class Tap(torch.autograd.Function):
    """Identity whose backward reports the gradient to a fan-in."""

    @staticmethod
    def forward(ctx, inp, fanin):
        ctx.fanin = fanin
        return inp.view_as(inp)

    @staticmethod
    def backward(ctx, grad):
        # Copy so the fan-in owns the buffer the module accumulates into.
        return ctx.fanin.finish(grad.clone(), accumulated=False), None


def _inputs(seed: int, out_features: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(S, B, H, device="cuda", generator=gen).to(torch.bfloat16)
    dy_branch = torch.randn(S, B, H, device="cuda", generator=gen).to(torch.bfloat16)
    dy_module = torch.randn(S, B, out_features, device="cuda", generator=gen).to(torch.bfloat16)
    return x, dy_branch, dy_module


def _run(module, x, dy_branch, dy_module, fanin, quant_recipe):
    """``x`` feeds ``module`` and a second branch (scaled identity) whose gradient comes first."""
    x = x.detach().requires_grad_()
    with te.autocast(enabled=quant_recipe is not None, recipe=quant_recipe):
        if fanin is not None:
            module._grad_fanin = fanin
        y = module(x)
    # Created after the module, so its backward runs first.
    branch = Tap.apply(x, fanin) if fanin is not None else x
    torch.autograd.backward([branch * 2.0, y], [dy_branch, dy_module])
    return x.grad


def _rel_err(actual, expected):
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


def _recipes():
    out = [None]
    if mxfp8_available:
        out.append(recipe.MXFP8BlockScaling())
    return out


@pytest.mark.parametrize("quant_recipe", _recipes(), ids=lambda r: type(r).__name__)
@pytest.mark.parametrize("zero_centered_gamma", [False, True])
def test_layernorm_linear_rmsnorm_adds_fanin(quant_recipe, zero_centered_gamma):
    torch.manual_seed(0)
    module = te.LayerNormLinear(
        H,
        F,
        bias=False,
        normalization="RMSNorm",
        zero_centered_gamma=zero_centered_gamma,
        params_dtype=torch.bfloat16,
    )
    x, dy_branch, dy_module = _inputs(2, F)
    ref = _run(module, x, dy_branch, dy_module, None, quant_recipe)
    ref_dgamma = module.layer_norm_weight.grad.clone()
    ref_wgrad = module.weight.grad.clone()
    module.layer_norm_weight.grad = None
    module.weight.grad = None

    fanin = FanIn(2)
    got = _run(module, x, dy_branch, dy_module, fanin, quant_recipe)
    assert fanin.finished == [(False, False), (True, True)]
    torch.testing.assert_close(module.layer_norm_weight.grad, ref_dgamma, rtol=0, atol=0)
    torch.testing.assert_close(module.weight.grad, ref_wgrad, rtol=0, atol=0)
    torch.testing.assert_close(got.float(), ref.float(), rtol=1.6e-2, atol=1e-2)
    assert _rel_err(got, ref) < 4e-3


def test_module_reports_first():
    """Module contribution arrives first: no running sum yet, result is still exact."""
    module = te.LayerNormLinear(
        H, F, bias=False, normalization="RMSNorm", params_dtype=torch.bfloat16
    )
    x, dy_branch, dy_module = _inputs(3, F)
    x_ref = x.detach().requires_grad_()
    y = module(x_ref)
    torch.autograd.backward([x_ref * 2.0, y], [dy_branch, dy_module])

    fanin = FanIn(2)
    x_got = x.detach().requires_grad_()
    branch = Tap.apply(x_got, fanin)  # created first, so its backward runs last
    module._grad_fanin = fanin
    y = module(x_got)
    torch.autograd.backward([branch * 2.0, y], [dy_branch, dy_module])
    assert fanin.finished == [(False, False), (False, True)]
    # Same two bf16 values added in the same precision: bitwise equal to autograd.
    torch.testing.assert_close(x_got.grad, x_ref.grad, rtol=0, atol=0)


def test_no_grad_forward_keeps_fanin():
    """A forward call without grad, or with an input that needs no grad, keeps the fan-in."""
    module = te.LayerNormLinear(
        H, F, bias=False, normalization="RMSNorm", params_dtype=torch.bfloat16
    )
    assert module.supports_grad_fanin
    fanin = FanIn(1)
    module._grad_fanin = fanin
    x, _, _ = _inputs(4, F)
    with torch.no_grad():
        module(x)
    assert module.__dict__["_grad_fanin"] is fanin
    module(x)  # x does not require grad
    assert module.__dict__.pop("_grad_fanin") is fanin


class AccumTargetOnlyFanIn(FanIn):
    """Fan-in without partial_sum(): LayerNormLinear reads accum_target() instead."""

    partial_sum = None


class ReadOnlyFanIn(FanIn):
    """Fan-in whose running sum may not be written in place (accum_target() is None)."""

    def accum_target(self):
        return None


@pytest.mark.parametrize("fanin_cls", [AccumTargetOnlyFanIn, ReadOnlyFanIn])
def test_partial_sum_and_accum_target(fanin_cls):
    torch.manual_seed(0)
    module = te.LayerNormLinear(
        H, F, bias=False, normalization="RMSNorm", params_dtype=torch.bfloat16
    )
    x, dy_branch, dy_module = _inputs(5, F)
    ref = _run(module, x, dy_branch, dy_module, None, None)
    fanin = fanin_cls(2)
    got = _run(module, x, dy_branch, dy_module, fanin, None)
    assert fanin.finished == [(False, False), (True, True)]
    assert _rel_err(got, ref) < 4e-3


def test_layernorm_does_not_fuse():
    """With LayerNorm the input gradient is reported unchanged and autograd-equal."""
    module = te.LayerNormLinear(
        H, F, bias=False, normalization="LayerNorm", params_dtype=torch.bfloat16
    )
    x, dy_branch, dy_module = _inputs(6, F)
    ref = _run(module, x, dy_branch, dy_module, None, None)
    fanin = FanIn(2)
    got = _run(module, x, dy_branch, dy_module, fanin, None)
    assert fanin.finished == [(False, False), (False, True)]
    torch.testing.assert_close(got, ref, rtol=0, atol=0)
