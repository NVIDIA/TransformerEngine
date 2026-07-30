# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from transformer_engine.pytorch import Linear, autocast, checkpoint
from transformer_engine.pytorch.quantization import FP8GlobalStateManager


fp8_available, reason_for_no_fp8 = te.is_fp8_available(return_reason=True)


def _make_input():
    return torch.randn(
        16,
        16,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )


def _assert_finite_loss_and_grads(loss, inp, *layers):
    assert torch.isfinite(loss)
    assert inp.grad is not None
    assert torch.isfinite(inp.grad).all()
    for layer in layers:
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.parametrize("use_reentrant", [True, False])
def test_fp8_checkpoint_with_inner_autocast(use_reentrant):
    """Delayed-scaling metadata is preserved when FP8 starts inside the checkpoint."""
    FP8GlobalStateManager.reset()
    fp8_recipe = recipe.DelayedScaling(fp8_format=recipe.Format.HYBRID)
    layer = Linear(16, 16, bias=False, params_dtype=torch.float32).cuda()
    inp = _make_input()

    def checkpointed_body(value):
        with autocast(enabled=True, recipe=fp8_recipe):
            return layer(value)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = checkpoint(checkpointed_body, inp, use_reentrant=use_reentrant)
        loss = out.float().sum()
    loss.backward()
    torch.cuda.synchronize()

    _assert_finite_loss_and_grads(loss, inp, layer)
    assert "global_fp8_buffer_pos_fwd_recompute" in layer.fp8_meta


@pytest.mark.parametrize("use_reentrant", [True, False])
def test_checkpoint_without_fp8_does_not_save_fp8_recompute_state(use_reentrant):
    """A checkpointed non-FP8 module does not save FP8 recompute metadata."""
    FP8GlobalStateManager.reset()
    layer = Linear(16, 16, bias=False, params_dtype=torch.float32).cuda()
    inp = _make_input()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = checkpoint(layer, inp, use_reentrant=use_reentrant)
        loss = out.float().sum()
    loss.backward()
    torch.cuda.synchronize()

    _assert_finite_loss_and_grads(loss, inp, layer)
    assert "global_fp8_buffer_pos_fwd_recompute" not in layer.fp8_meta


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.parametrize("use_reentrant", [True, False])
def test_checkpoint_with_mixed_fp8_regions_saves_only_fp8_recompute_state(use_reentrant):
    """Only the inner FP8 region of a mixed checkpoint saves recompute metadata."""
    FP8GlobalStateManager.reset()
    fp8_recipe = recipe.DelayedScaling(fp8_format=recipe.Format.HYBRID)
    non_fp8_layer = Linear(16, 16, bias=False, params_dtype=torch.float32).cuda()
    fp8_layer = Linear(16, 16, bias=False, params_dtype=torch.float32).cuda()
    inp = _make_input()

    def checkpointed_body(value):
        value = non_fp8_layer(value)
        with autocast(enabled=True, recipe=fp8_recipe):
            return fp8_layer(value)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = checkpoint(checkpointed_body, inp, use_reentrant=use_reentrant)
        loss = out.float().sum()
    loss.backward()
    torch.cuda.synchronize()

    _assert_finite_loss_and_grads(loss, inp, non_fp8_layer, fp8_layer)
    assert "global_fp8_buffer_pos_fwd_recompute" not in non_fp8_layer.fp8_meta
    assert "global_fp8_buffer_pos_fwd_recompute" in fp8_layer.fp8_meta
