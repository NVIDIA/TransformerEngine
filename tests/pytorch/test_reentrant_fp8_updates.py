# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Regression tests for FP8 update ownership during activation recompute."""

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from transformer_engine.pytorch.distributed import (
    in_fp8_activation_recompute_phase,
    is_fp8_activation_recompute_enabled,
)
from transformer_engine.pytorch.quantization import FP8GlobalStateManager


fp8_available, reason_for_no_fp8 = te.is_fp8_available(return_reason=True)


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.parametrize(
    ("checkpoint_mode", "segments", "num_layers"),
    (
        ("none", "single", 3),
        ("non-reentrant", "single", 3),
        ("non-reentrant", "per-layer", 3),
        ("reentrant", "single", 1),
        ("reentrant", "single", 3),
        ("reentrant", "per-layer", 3),
        ("nested-reentrant", "nested", 3),
    ),
)
def test_delayed_scaling_updates_once_per_autocast(
    monkeypatch, checkpoint_mode, segments, num_layers
):
    """Activation recompute must not advance global FP8 state per module/segment."""

    FP8GlobalStateManager.reset()
    counts = {"forward": 0, "backward": 0}
    original_update = FP8GlobalStateManager.reduce_and_update_fp8_tensors

    def counted_update(cls, forward=True):
        del cls
        counts["forward" if forward else "backward"] += 1
        return original_update(forward=forward)

    monkeypatch.setattr(
        FP8GlobalStateManager,
        "reduce_and_update_fp8_tensors",
        classmethod(counted_update),
    )

    torch.manual_seed(20260715)
    torch.cuda.manual_seed_all(20260715)
    layers = [
        te.Linear(16, 16, bias=False, params_dtype=torch.float32).cuda() for _ in range(num_layers)
    ]
    network = torch.nn.Sequential(*layers)
    inp = torch.randn(
        16,
        16,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    fp8_recipe = recipe.DelayedScaling(fp8_format=recipe.Format.HYBRID)

    with torch.autocast("cuda", dtype=torch.bfloat16), te.autocast(
        enabled=True,
        recipe=fp8_recipe,
    ):
        if checkpoint_mode == "none":
            out = network(inp)
        elif checkpoint_mode == "nested-reentrant":

            def inner(x):
                return layers[0](x)

            def outer(x):
                x = te.checkpoint(inner, x, use_reentrant=True)
                for layer in layers[1:]:
                    x = layer(x)
                return x

            out = te.checkpoint(outer, inp, use_reentrant=True)
        elif segments == "single":
            out = te.checkpoint(
                network,
                inp,
                use_reentrant=checkpoint_mode == "reentrant",
            )
        else:
            out = inp
            for layer in layers:
                out = te.checkpoint(
                    layer,
                    out,
                    use_reentrant=checkpoint_mode == "reentrant",
                )
        loss = out.float().sum()

    loss.backward()
    torch.cuda.synchronize()

    assert torch.isfinite(loss)
    assert inp.grad is not None
    assert torch.isfinite(inp.grad).all()
    assert inp.grad.abs().max() > 0
    for layer in layers:
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()
        assert layer.weight.grad.abs().max() > 0
    assert counts == {"forward": 1, "backward": 1}
    assert FP8GlobalStateManager.quantization_state.autocast_depth == 0
    assert not FP8GlobalStateManager.is_fp8_enabled()
    assert not is_fp8_activation_recompute_enabled()
    assert not in_fp8_activation_recompute_phase()


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
def test_reentrant_checkpoint_gradients_match_uncheckpointed():
    """Reentrant recompute should preserve the uncheckpointed FP8 numerics."""

    def run(checkpoint):
        FP8GlobalStateManager.reset()
        torch.manual_seed(20260715)
        torch.cuda.manual_seed_all(20260715)
        layers = [
            te.Linear(16, 16, bias=False, params_dtype=torch.float32).cuda() for _ in range(3)
        ]
        network = torch.nn.Sequential(*layers)
        inp = torch.randn(
            16,
            16,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        fp8_recipe = recipe.DelayedScaling(fp8_format=recipe.Format.HYBRID)
        with torch.autocast("cuda", dtype=torch.bfloat16), te.autocast(
            enabled=True,
            recipe=fp8_recipe,
        ):
            out = te.checkpoint(network, inp, use_reentrant=True) if checkpoint else network(inp)
            loss = out.float().sum()
        loss.backward()
        torch.cuda.synchronize()
        return (
            loss.detach(),
            inp.grad.detach(),
            *(layer.weight.grad.detach() for layer in layers),
        )

    reference = run(checkpoint=False)
    checkpointed = run(checkpoint=True)
    for actual, expected in zip(checkpointed, reference):
        torch.testing.assert_close(actual, expected, rtol=0.125, atol=0.0675)
