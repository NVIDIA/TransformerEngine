# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Gradient fan-in for te.Linear: the dgrad GEMM accumulates into a shared gradient sum."""

from typing import List, Optional

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, MXFP8BlockScaling

mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)

S, B, H, F = 256, 2, 512, 384


class GradFanIn:
    """Reference fan-in: sums the gradients of ``num_consumers`` consumers in one buffer."""

    def __init__(self, num_consumers: int):
        self.num_consumers = num_consumers
        self.seen = 0
        self.acc: Optional[torch.Tensor] = None
        self.accumulated: List[bool] = []

    def accum_target(self) -> Optional[torch.Tensor]:
        return self.acc

    def finish(self, grad: Optional[torch.Tensor], accumulated: bool) -> Optional[torch.Tensor]:
        self.accumulated.append(accumulated)
        if grad is not None:
            if accumulated or self.acc is None:
                self.acc = grad
            else:
                self.acc.add_(grad)
        self.seen += 1
        if self.seen == self.num_consumers:
            out, self.acc = self.acc, None
            return out
        return None


class GradFanInTap(torch.autograd.Function):
    """Identity whose backward reports its incoming gradient to a GradFanIn."""

    @staticmethod
    def forward(ctx, inp, fanin):
        ctx.fanin = fanin
        return inp.view_as(inp)

    @staticmethod
    def backward(ctx, grad):
        return ctx.fanin.finish(grad, accumulated=False), None


def _inputs(seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(S, B, H, device="cuda", generator=g).bfloat16()
    gy_tap = torch.randn(S, B, H, device="cuda", generator=g).bfloat16()
    gy_lin = torch.randn(S, B, F, device="cuda", generator=g).bfloat16()
    return x, gy_tap, gy_lin


def _run(linear, x, gy_tap, gy_lin, fanin, recipe, tap_first=True):
    """x feeds ``linear`` and an elementwise branch; returns x.grad."""
    x = x.detach().requires_grad_()
    with te.autocast(enabled=recipe is not None, recipe=recipe):
        if fanin is not None:
            linear._grad_fanin = fanin
        if tap_first:
            y = linear(x)
            branch = GradFanInTap.apply(x, fanin) if fanin is not None else x
        else:
            # The tap is created first, so autograd runs it after the linear's backward.
            branch = GradFanInTap.apply(x, fanin) if fanin is not None else x
            y = linear(x)
    torch.autograd.backward([branch * 1.5, y], [gy_tap, gy_lin])
    return x.grad


@pytest.mark.parametrize(
    "quantization",
    [
        None,
        pytest.param(
            "mxfp8",
            marks=pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8),
        ),
    ],
)
@pytest.mark.parametrize("tap_first", [True, False])
def test_linear_grad_fanin_matches_autograd(quantization, tap_first):
    torch.manual_seed(0)
    recipe = MXFP8BlockScaling(fp8_format=Format.E4M3) if quantization == "mxfp8" else None
    linear = te.Linear(H, F, bias=False, params_dtype=torch.bfloat16)
    x, gy_tap, gy_lin = _inputs(1)
    ref = _run(linear, x, gy_tap, gy_lin, None, recipe, tap_first)
    ref_wgrad = linear.weight.grad.clone()
    linear.weight.grad = None
    fanin = GradFanIn(2)
    got = _run(linear, x, gy_tap, gy_lin, fanin, recipe, tap_first)

    # The tap reports first in backward when it was created after the linear; the dgrad GEMM
    # then accumulates into its gradient. Otherwise the linear reports first and its plain
    # dgrad seeds the sum.
    assert fanin.accumulated == ([False, True] if tap_first else [False, False])
    assert fanin.acc is None
    assert "_grad_fanin" not in linear.__dict__
    torch.testing.assert_close(linear.weight.grad, ref_wgrad, atol=0, rtol=0)
    if tap_first:
        # The epilogue rounds the sum once instead of rounding dgrad and the sum separately.
        torch.testing.assert_close(got.float(), ref.float(), atol=2e-2, rtol=2e-2)
        rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
        assert rel < 4e-3
    else:
        torch.testing.assert_close(got, ref, atol=0, rtol=0)


def test_linear_grad_fanin_epilogue_matches_fp32_sum():
    """The accumulated result is the correctly rounded fp32 sum of the two gradients."""
    torch.manual_seed(0)
    linear = te.Linear(H, F, bias=False, params_dtype=torch.bfloat16)
    x, gy_tap, gy_lin = _inputs(2)
    got = _run(linear, x, gy_tap, gy_lin, GradFanIn(2), None)
    dgrad = gy_lin.float().reshape(-1, F) @ linear.weight.float()
    expected = (dgrad.reshape(S, B, H) + (gy_tap * 1.5).float()).bfloat16()
    torch.testing.assert_close(got, expected, atol=2e-2, rtol=1e-2)


def test_linear_grad_fanin_not_consumed_without_backward():
    """A forward that will not run backward for its input leaves the fan-in to the caller."""
    linear = te.Linear(H, F, bias=False, params_dtype=torch.bfloat16)
    x, _, _ = _inputs(3)
    fanin = GradFanIn(2)
    linear._grad_fanin = fanin
    with torch.no_grad():
        linear(x)
    assert linear.__dict__.get("_grad_fanin") is fanin
    linear(x.detach())  # input does not require grad
    assert linear.__dict__.get("_grad_fanin") is fanin
    del linear._grad_fanin
    assert fanin.seen == 0


def test_linear_grad_fanin_cuda_graph():
    """The accumulate path captured in a CUDA graph replays to the eager result."""
    torch.manual_seed(0)
    linear = te.Linear(H, F, bias=False, params_dtype=torch.bfloat16)
    x, gy_tap, gy_lin = _inputs(4)
    static_x = x.clone()
    out = {}

    def step():
        out["dx"] = _run(linear, static_x, gy_tap, gy_lin, GradFanIn(2), None)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    new_x, _, _ = _inputs(5)
    static_x.copy_(new_x)
    graph.replay()
    torch.cuda.synchronize()
    ref = _run(linear, new_x, gy_tap, gy_lin, GradFanIn(2), None)
    torch.testing.assert_close(out["dx"], ref, atol=0, rtol=0)
