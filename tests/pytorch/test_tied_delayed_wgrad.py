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


def _tie(model, *, bias):
    """Re-tie after load_state_dict, which rebinds parameters."""
    model[1].weight = model[0].weight
    if bias:
        model[1].bias = model[0].bias
    return model


def _build(delay, *, bias=False, device="cuda"):
    """Two Linear layers over the same shapes. Tying is the caller's business."""
    model = nn.Sequential(
        te.Linear(
            SIZE,
            SIZE,
            bias=bias,
            params_dtype=DTYPE,
            device=device,
            delay_wgrad_compute=delay,
            fuse_wgrad_accumulation=False,
        ),
        te.Linear(
            SIZE,
            SIZE,
            bias=bias,
            params_dtype=DTYPE,
            device=device,
            delay_wgrad_compute=delay,
            fuse_wgrad_accumulation=False,
        ),
    )
    return model


def _run(model, x, microbatches=1, delay=False, backward_dw_order=(0, 1)):
    """Run the microbatches, then drain the delayed wgrad queues.

    ``backward_dw_order`` is the module order used for draining, which decides which tied
    module pops its delayed wgrad first.
    """
    for xb in torch.chunk(x, microbatches, dim=0):
        model(xb).float().sum().backward()
    if delay:
        for _ in range(microbatches):
            for i in backward_dw_order:
                model[i].backward_dw()


def _x(batch=BATCH, device="cuda"):
    return torch.randn(batch, SIZE, device=device, dtype=DTYPE, requires_grad=True)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(9)


def test_tied_delayed_wgrad_accumulates():
    """The issue's own case: a tied delayed grad must match the regular path."""
    delayed = _tie(_build(True), bias=False)
    regular = _build(False)
    regular.load_state_dict(delayed.state_dict())
    _tie(regular, bias=False)
    xd = _x()
    _run(delayed, xd, delay=True)
    _run(regular, xd.detach().clone().requires_grad_(True))
    torch.testing.assert_close(delayed[0].weight.grad, regular[0].weight.grad, rtol=0, atol=0)


def test_non_tied_control_still_passes():
    """Independent parameters keep working."""
    delayed, regular = _build(True), _build(False)
    regular.load_state_dict(delayed.state_dict())
    xd = _x()
    _run(delayed, xd, delay=True)
    _run(regular, xd.detach().clone().requires_grad_(True))
    for i in (0, 1):
        torch.testing.assert_close(delayed[i].weight.grad, regular[i].weight.grad, rtol=0, atol=0)


def test_both_backward_dw_orders_match():
    """The shared gradient must not depend on which module pops first."""
    a = _tie(_build(True), bias=False)
    b = _build(True)
    b.load_state_dict(a.state_dict())
    _tie(b, bias=False)
    xa = _x()
    _run(a, xa, delay=True, backward_dw_order=(0, 1))
    _run(b, xa.detach().clone().requires_grad_(True), delay=True, backward_dw_order=(1, 0))
    torch.testing.assert_close(a[0].weight.grad, b[0].weight.grad, rtol=0, atol=0)


def test_multiple_microbatches_accumulate():
    # The delayed path sums contributions in arrival order while the regular path lets
    # autograd accumulate, so with several microbatches the two orders can differ by a BF16
    # ULP. That is not something the fix promises to remove, so the tolerance belongs to this
    # test only; every single-microbatch comparison below stays bit-exact.
    rtol = atol = 1e-2

    delayed = _tie(_build(True), bias=False)
    regular = _build(False)
    regular.load_state_dict(delayed.state_dict())
    _tie(regular, bias=False)
    x = _x(4)
    _run(delayed, x.clone().requires_grad_(True), microbatches=2, delay=True)
    _run(regular, x.clone().requires_grad_(True), microbatches=2)
    torch.testing.assert_close(delayed[0].weight.grad, regular[0].weight.grad, rtol=rtol, atol=atol)


@pytest.mark.parametrize("set_to_none", [True, False])
def test_two_optimizer_steps_do_not_pollute_each_other(set_to_none):
    """After a reset, the second step's grad must be the second step's alone.

    ``zero_grad(set_to_none=False)`` leaves a zeroed tensor where ``set_to_none=True`` leaves
    ``None``, so the first contribution of the step takes the add path instead of the assign
    path. Both settings have to deliver the same value, and ``second != first`` on its own
    would not show pollution: a first-step value carried into the second still differs from
    the first.
    """
    model = _tie(_build(True), bias=False)
    params = list({id(p): p for p in model.parameters()}.values())
    opt = torch.optim.SGD(params, lr=0.0)

    _run(model, _x(), delay=True)
    first = model[0].weight.grad.detach().clone()
    opt.zero_grad(set_to_none=set_to_none)

    second_x = _x()
    _run(model, second_x, delay=True)
    second = model[0].weight.grad.detach().clone()

    # What the second step owes on its own: same weights, same input, first backward only.
    fresh = _build(True)
    fresh.load_state_dict(model.state_dict())
    _tie(fresh, bias=False)
    _run(fresh, second_x.detach().clone().requires_grad_(True), delay=True)

    torch.testing.assert_close(second, fresh[0].weight.grad, rtol=0, atol=0)
    assert not torch.equal(first, second)
    assert second.abs().sum() > 0


def test_tied_bias_delayed_wgrad_accumulates():
    """The bias shares the pop path, so it must accumulate too."""
    delayed = _tie(_build(True, bias=True), bias=True)
    regular = _build(False, bias=True)
    regular.load_state_dict(delayed.state_dict())
    _tie(regular, bias=True)
    xd = _x()
    _run(delayed, xd, delay=True)
    _run(regular, xd.detach().clone().requires_grad_(True))
    torch.testing.assert_close(delayed[0].bias.grad, regular[0].bias.grad, rtol=0, atol=0)


def test_matches_pure_pytorch_reference():
    """Independent oracle: plain nn.Linear over the same weight and input."""
    model = _tie(_build(True), bias=False)
    x = _x()
    _run(model, x, delay=True)

    weight = model[0].weight
    lin = nn.Linear(SIZE, SIZE, bias=False).to(device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        lin.weight.copy_(weight)
    lin(lin(x.detach().clone().requires_grad_(True))).float().sum().backward()
    torch.testing.assert_close(model[0].weight.grad, lin.weight.grad, rtol=0, atol=0)
