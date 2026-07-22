# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Targeted FP8 checkpoint ownership and exception-state regressions."""

from dataclasses import dataclass

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format
from transformer_engine.pytorch.distributed import (
    _ActivationRecomputeState,
    activation_recompute_forward,
    in_fp8_activation_recompute_phase,
    is_fp8_activation_recompute_enabled,
)
from transformer_engine.pytorch.quantization import FP8GlobalStateManager


fp8_available, reason_for_no_fp8 = te.is_fp8_available(return_reason=True)
pytestmark = pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)

WIDTH = 16
SEED = 20260722


@dataclass
class UpdateCounter:
    forward: int = 0
    backward: int = 0

    def snapshot(self) -> tuple[int, int]:
        return self.forward, self.backward

    def delta(self, before: tuple[int, int]) -> tuple[int, int]:
        return self.forward - before[0], self.backward - before[1]


@pytest.fixture(autouse=True)
def clean_fp8_state():
    FP8GlobalStateManager.reset()
    yield
    qstate = FP8GlobalStateManager.quantization_state
    assert qstate.autocast_depth == 0
    assert not FP8GlobalStateManager.is_fp8_enabled()
    assert not is_fp8_activation_recompute_enabled()
    assert not in_fp8_activation_recompute_phase()


@pytest.fixture
def update_counter(monkeypatch) -> UpdateCounter:
    counter = UpdateCounter()
    original = FP8GlobalStateManager.reduce_and_update_fp8_tensors

    def counted(_cls, forward=True):
        if forward:
            counter.forward += 1
        else:
            counter.backward += 1
        return original(forward=forward)

    monkeypatch.setattr(
        FP8GlobalStateManager,
        "reduce_and_update_fp8_tensors",
        classmethod(counted),
    )
    return counter


def make_linear() -> te.Linear:
    return te.Linear(
        WIDTH,
        WIDTH,
        bias=False,
        params_dtype=torch.float32,
        init_method=lambda tensor: torch.nn.init.normal_(tensor, mean=0.0, std=0.1),
    ).cuda()


def make_input() -> torch.Tensor:
    return torch.randn(
        WIDTH,
        WIDTH,
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )


def assert_finite_nonzero(loss, inp, module) -> None:
    assert torch.isfinite(loss)
    assert inp.grad is not None
    assert torch.isfinite(inp.grad).all()
    assert torch.count_nonzero(inp.grad) > 0
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def assert_global_state_restored() -> None:
    qstate = FP8GlobalStateManager.quantization_state
    assert qstate.autocast_depth == 0
    assert not is_fp8_activation_recompute_enabled()
    assert not in_fp8_activation_recompute_phase()


def run_recovery(update_counter: UpdateCounter) -> None:
    before = update_counter.snapshot()
    layer = make_linear()
    inp = make_input()
    recipe = DelayedScaling(fp8_format=Format.HYBRID)
    with torch.autocast("cuda", dtype=torch.bfloat16), te.autocast(
        enabled=True, recipe=recipe
    ):
        loss = layer(inp).float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    assert_finite_nonzero(loss, inp, layer)
    assert update_counter.delta(before) == (1, 1)
    assert_global_state_restored()


def test_nested_autocast_does_not_revive_consumed_owner(update_counter):
    """A nested exit must not make first-module ownership available again."""
    torch.manual_seed(SEED)
    layers = torch.nn.ModuleList([make_linear(), make_linear()])
    inp = make_input()
    recipe = DelayedScaling(fp8_format=Format.HYBRID)
    with torch.autocast("cuda", dtype=torch.bfloat16), te.autocast(
        enabled=True, recipe=recipe
    ):
        with te.autocast(enabled=True, recipe=recipe):
            out = layers[0](inp)
        loss = layers[1](out).float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    assert_finite_nonzero(loss, inp, layers)
    assert update_counter.snapshot() == (1, 1)


@pytest.mark.parametrize("use_reentrant", (True, False))
def test_nested_autocast_inside_checkpoint_has_one_owner(update_counter, use_reentrant):
    torch.manual_seed(SEED)
    layers = torch.nn.ModuleList([make_linear(), make_linear()])
    inp = make_input()
    recipe = DelayedScaling(fp8_format=Format.HYBRID)

    def body(value):
        with te.autocast(enabled=True, recipe=recipe):
            value = layers[0](value)
        return layers[1](value)

    with torch.autocast("cuda", dtype=torch.bfloat16), te.autocast(
        enabled=True, recipe=recipe
    ):
        loss = te.checkpoint(body, inp, use_reentrant=use_reentrant).float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    assert_finite_nonzero(loss, inp, layers)
    assert update_counter.snapshot() == (1, 1)


def test_original_forward_exception_restores_owner(update_counter):
    """A failed checkpoint frame must return its reservation to the outer scope."""
    recovery = make_linear()
    inp = make_input()
    recipe = DelayedScaling(fp8_format=Format.HYBRID)

    def fail(_value):
        raise RuntimeError("intentional original-forward failure")

    with torch.autocast("cuda", dtype=torch.bfloat16), te.autocast(
        enabled=True, recipe=recipe
    ):
        with pytest.raises(RuntimeError, match="intentional original-forward failure"):
            te.checkpoint(fail, inp, use_reentrant=True)
        loss = recovery(inp).float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    assert_finite_nonzero(loss, inp, recovery)
    assert update_counter.snapshot() == (1, 1)


def test_recompute_enter_failure_does_not_leak_state(update_counter):
    """Validation must happen before process-global recompute state is changed."""
    qstate = FP8GlobalStateManager.quantization_state
    qstate.is_first_fp8_module = True
    with pytest.raises(RuntimeError, match="was not captured"):
        with activation_recompute_forward(
            activation_recompute=True,
            recompute_phase=True,
            state=_ActivationRecomputeState(),
        ):
            pass
    assert qstate.is_first_fp8_module
    assert_global_state_restored()
    run_recovery(update_counter)


def test_recompute_body_exception_restores_state(update_counter):
    """A recompute exception must not poison a subsequent normal FP8 scope."""
    failing = make_linear()
    failed_input = make_input()
    recipe = DelayedScaling(fp8_format=Format.HYBRID)

    def fail_during_recompute(value):
        result = failing(value)
        if torch.is_grad_enabled():
            raise RuntimeError("intentional recompute failure")
        return result

    with torch.autocast("cuda", dtype=torch.bfloat16), te.autocast(
        enabled=True, recipe=recipe
    ):
        loss = te.checkpoint(
            fail_during_recompute,
            failed_input,
            use_reentrant=True,
        ).float().square().mean()
    with pytest.raises(RuntimeError, match="intentional recompute failure"):
        loss.backward()
    assert_global_state_restored()
    run_recovery(update_counter)
