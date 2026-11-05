# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Fused wgrads into an ``overwrite_main_grad`` weight used several times in one backward."""

import pytest
import torch

import transformer_engine.pytorch as te

HIDDEN, OUT, TOKENS = 128, 256, 64


def _build(module_cls, fused, delay):
    return module_cls(
        HIDDEN,
        OUT,
        bias=False,
        params_dtype=torch.bfloat16,
        device="cuda",
        fuse_wgrad_accumulation=fused,
        delay_wgrad_compute=delay,
    )


def _reset_main_grad(weight):
    """Hand ``main_grad`` back the way Megatron-FSDP does after consuming it."""
    weight.main_grad.fill_(1e3)  # Stale contents that must be overwritten, never accumulated.
    weight.overwrite_main_grad = True
    weight.grad_added_to_main_grad = False
    weight._fsdp_wgrad_written = False


@pytest.mark.parametrize("module_cls", [te.Linear, te.LayerNormLinear])
@pytest.mark.parametrize("uses", [1, 2, 3])
@pytest.mark.parametrize("delay", [False, True])
def test_reused_weight_accumulates_after_first_wgrad(module_cls, uses, delay):
    """The first wgrad since the reset overwrites ``main_grad``; later ones accumulate."""
    torch.manual_seed(1234)
    fused = _build(module_cls, True, delay)
    reference = _build(module_cls, False, False)
    reference.load_state_dict(fused.state_dict())
    weight = fused.weight
    weight.main_grad = torch.empty(OUT, HIDDEN, device="cuda", dtype=torch.float32)

    # Two backwards, each followed by the caller consuming main_grad.
    for _ in range(2):
        _reset_main_grad(weight)
        reference.weight.grad = None
        xs = [torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16) for _ in range(uses)]
        dys = [torch.randn(TOKENS, OUT, device="cuda", dtype=torch.bfloat16) for _ in range(uses)]
        for model in (fused, reference):
            loss = sum((model(x).float() * dy.float()).sum() for x, dy in zip(xs, dys))
            loss.backward()
        if delay:
            for _ in range(uses):
                fused.backward_dw()

        expected = reference.weight.grad.float()
        atol = 4 * uses * torch.finfo(torch.bfloat16).eps * expected.abs().max().item()
        torch.testing.assert_close(weight.main_grad, expected, rtol=1e-2, atol=atol)
        assert weight._fsdp_wgrad_written


class _TwoUses(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x1, x2):
        return self.module(x1) + self.module(x2)


@pytest.mark.parametrize("module_cls", [te.Linear, te.LayerNormLinear])
def test_reused_weight_graph_replay_overwrites_stale_main_grad(module_cls):
    """Warmup must not leave the flag set: replay overwrites first, then accumulates."""
    torch.manual_seed(1234)
    fused = _build(module_cls, True, False)
    reference = _build(module_cls, False, False)
    reference.load_state_dict(fused.state_dict())
    weight = fused.weight
    weight.main_grad = torch.empty(OUT, HIDDEN, device="cuda", dtype=torch.float32)
    _reset_main_grad(weight)
    sample = [torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    graphed = te.make_graphed_callables(_TwoUses(fused), tuple(sample))
    for _ in range(2):
        _reset_main_grad(weight)
        reference.weight.grad = None
        xs = [torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        dy = torch.randn(TOKENS, OUT, device="cuda", dtype=torch.bfloat16)
        (graphed(*xs).float() * dy.float()).sum().backward()
        (_TwoUses(reference)(*xs).float() * dy.float()).sum().backward()
        expected = reference.weight.grad.float()
        atol = 8 * torch.finfo(torch.bfloat16).eps * expected.abs().max().item()
        torch.testing.assert_close(weight.main_grad, expected, rtol=1e-2, atol=atol)


@pytest.mark.parametrize("module_cls", [te.Linear, te.LayerNormLinear])
def test_weight_without_flag_keeps_overwriting(module_cls):
    """A caller that never clears ``_fsdp_wgrad_written`` keeps getting overwrites across backwards."""
    torch.manual_seed(1234)
    fused = _build(module_cls, True, False)
    reference = _build(module_cls, False, False)
    reference.load_state_dict(fused.state_dict())
    weight = fused.weight
    weight.main_grad = torch.empty(OUT, HIDDEN, device="cuda", dtype=torch.float32)
    weight.overwrite_main_grad = True
    for _ in range(2):
        weight.main_grad.fill_(1e3)
        weight.grad_added_to_main_grad = False
        reference.weight.grad = None
        x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16)
        dy = torch.randn(TOKENS, OUT, device="cuda", dtype=torch.bfloat16)
        for model in (fused, reference):
            (model(x).float() * dy.float()).sum().backward()
        expected = reference.weight.grad.float()
        atol = 4 * torch.finfo(torch.bfloat16).eps * expected.abs().max().item()
        torch.testing.assert_close(weight.main_grad, expected, rtol=1e-2, atol=atol)
        assert not hasattr(weight, "_fsdp_wgrad_written")
