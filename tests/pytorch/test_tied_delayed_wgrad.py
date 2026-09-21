# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Tied parameters must accumulate delayed wgrads, not overwrite them (#3437).

``backward_dw()`` used to *assign* the popped delayed wgrad to
``Parameter.grad``. When two modules share one Parameter (tied weights), each
pops its own contribution and the later one silently replaced the earlier, so
the shared gradient came out too small. The bias takes the same path and hit the
same bug, with the extra twist that the second module's bgrad is often ~0 and so
wiped out a correct contribution.

These tests use only the public ``te.Linear`` API, so they cannot be satisfied by
a shared implementation detail of the fix.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import transformer_engine.pytorch as te

SIZE = 16
BATCH = 2
DTYPE = torch.bfloat16
DEVICE = "cuda"

# The delayed path sums contributions in arrival order while the regular path
# lets autograd accumulate, so with several microbatches the two orders can
# differ by a BF16 ULP. That is not something the fix promises to remove, so the
# tolerance is declared here and applies to the multi-microbatch case only;
# every single-microbatch comparison below stays bit-exact.
MULTI_MICROBATCH_RTOL = 1e-2
MULTI_MICROBATCH_ATOL = 1e-2


def _tie(model, bias):
    """Re-tie after load_state_dict, which rebinds parameters."""
    model[1].weight = model[0].weight
    if bias:
        model[1].bias = model[0].bias
    return model


def _build(delay, tied, bias=False):
    model = nn.Sequential(
        te.Linear(
            SIZE,
            SIZE,
            bias=bias,
            params_dtype=DTYPE,
            device=DEVICE,
            delay_wgrad_compute=delay,
            fuse_wgrad_accumulation=False,
        ),
        te.Linear(
            SIZE,
            SIZE,
            bias=bias,
            params_dtype=DTYPE,
            device=DEVICE,
            delay_wgrad_compute=delay,
            fuse_wgrad_accumulation=False,
        ),
    )
    return _tie(model, bias) if tied else model


def _run(model, x, microbatches=1, delay=False, order=(0, 1)):
    """Run the microbatches, then drain the delayed wgrad queues."""
    for xb in torch.chunk(x, microbatches, dim=0):
        model(xb).float().sum().backward()
    if delay:
        for _ in range(microbatches):
            for i in order:
                model[i].backward_dw()


def _x(batch=BATCH):
    return torch.randn(batch, SIZE, device=DEVICE, dtype=DTYPE, requires_grad=True)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(9)


def test_tied_delayed_wgrad_accumulates():
    """The issue's own case: a tied delayed grad must match the regular path."""
    delayed, regular = _build(True, True), _build(False, True)
    regular.load_state_dict(delayed.state_dict())
    _tie(regular, False)
    xd = _x()
    _run(delayed, xd, delay=True)
    _run(regular, xd.detach().clone().requires_grad_(True))
    torch.testing.assert_close(delayed[0].weight.grad, regular[0].weight.grad, rtol=0, atol=0)


def test_non_tied_control_still_passes():
    """Independent parameters keep working."""
    delayed, regular = _build(True, False), _build(False, False)
    regular.load_state_dict(delayed.state_dict())
    xd = _x()
    _run(delayed, xd, delay=True)
    _run(regular, xd.detach().clone().requires_grad_(True))
    for i in (0, 1):
        torch.testing.assert_close(delayed[i].weight.grad, regular[i].weight.grad, rtol=0, atol=0)


def test_both_backward_dw_orders_match():
    """The shared gradient must not depend on which module pops first."""
    a, b = _build(True, True), _build(True, True)
    b.load_state_dict(a.state_dict())
    _tie(b, False)
    xa = _x()
    _run(a, xa, delay=True, order=(0, 1))
    _run(b, xa.detach().clone().requires_grad_(True), delay=True, order=(1, 0))
    torch.testing.assert_close(a[0].weight.grad, b[0].weight.grad, rtol=0, atol=0)


def test_multiple_microbatches_accumulate():
    delayed, regular = _build(True, True), _build(False, True)
    regular.load_state_dict(delayed.state_dict())
    _tie(regular, False)
    x = torch.randn(4, SIZE, device=DEVICE, dtype=DTYPE)
    _run(delayed, x.clone().requires_grad_(True), microbatches=2, delay=True)
    _run(regular, x.clone().requires_grad_(True), microbatches=2)
    torch.testing.assert_close(
        delayed[0].weight.grad,
        regular[0].weight.grad,
        rtol=MULTI_MICROBATCH_RTOL,
        atol=MULTI_MICROBATCH_ATOL,
    )


@pytest.mark.parametrize("set_to_none", [True, False])
def test_two_optimizer_steps_do_not_pollute_each_other(set_to_none):
    """After a reset, the second step's grad must not inherit the first's."""
    model = _build(True, True)
    params = list({id(p): p for p in model.parameters()}.values())
    opt = torch.optim.SGD(params, lr=0.0)

    _run(model, torch.randn(BATCH, SIZE, device=DEVICE, dtype=DTYPE), delay=True)
    first = model[0].weight.grad.detach().clone()
    opt.zero_grad(set_to_none=set_to_none)

    _run(model, torch.randn(BATCH, SIZE, device=DEVICE, dtype=DTYPE), delay=True)
    second = model[0].weight.grad.detach().clone()

    assert not torch.equal(first, second)
    assert second.abs().sum() > 0


def test_tied_bias_delayed_wgrad_accumulates():
    """The bias shares the pop path, so it must accumulate too."""
    delayed, regular = _build(True, True, bias=True), _build(False, True, bias=True)
    regular.load_state_dict(delayed.state_dict())
    _tie(regular, True)
    xd = _x()
    _run(delayed, xd, delay=True)
    _run(regular, xd.detach().clone().requires_grad_(True))
    torch.testing.assert_close(delayed[0].bias.grad, regular[0].bias.grad, rtol=0, atol=0)


def test_matches_pure_pytorch_reference():
    """Independent oracle: plain nn.Linear over the same weight and input."""
    model = _build(True, True)
    x = _x()
    _run(model, x, delay=True)

    lin = nn.Linear(SIZE, SIZE, bias=False).to(device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        lin.weight.copy_(model[0].weight)
    lin(lin(x.detach().clone().requires_grad_(True))).float().sum().backward()
    torch.testing.assert_close(model[0].weight.grad, lin.weight.grad, rtol=0, atol=0)
