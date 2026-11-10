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


class _Uses(torch.nn.Module):
    def __init__(self, module, uses):
        super().__init__()
        self.module = module
        self.uses = uses

    def forward(self, x):
        return sum(self.module(x * (i + 1)) for i in range(self.uses))


@pytest.mark.parametrize("module_cls", [te.Linear, te.LayerNormLinear])
@pytest.mark.parametrize("uses", [1, 2])
@pytest.mark.parametrize("with_order", [False, True])
@pytest.mark.parametrize("consume", [False, True], ids=["shared_main_grad", "consumed_each"])
def test_weight_shared_by_graphed_callables(module_cls, uses, with_order, consume):
    """Captures follow the flag like eager backwards: graphs feeding one ``main_grad`` accumulate,
    and a caller that consumes ``main_grad`` between them clears the flag in a capture-time hook."""
    torch.manual_seed(1234)
    fused = _build(module_cls, True, False)
    reference = _build(module_cls, False, False)
    reference.load_state_dict(fused.state_dict())
    weight = fused.weight
    weight.main_grad = torch.empty(OUT, HIDDEN, device="cuda", dtype=torch.float32)
    _reset_main_grad(weight)
    microbatches = 2
    samples = tuple(
        (torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16),)
        for _ in range(microbatches)
    )
    num_callables = 1 if with_order else microbatches
    hooks = None
    if consume:
        # The caller consumes main_grad before each backward, so it clears the flag at capture too.
        clear = {"clear": lambda module: setattr(weight, "_fsdp_wgrad_written", False)}
        hooks = [{"backward_pre_hooks": clear} for _ in range(num_callables)]
    if with_order:
        # One layer, both microbatches forward, then both backward, as pipelined schedules capture.
        graphed = te.make_graphed_callables(
            (_Uses(fused, uses),),
            samples,
            _order=[1] * microbatches + [-1] * microbatches,
            _num_layers_per_chunk=[1],
            capture_time_hooks=hooks,
        )
        bwd_order = list(range(microbatches))
    else:
        graphed = te.make_graphed_callables(
            tuple(_Uses(fused, uses) for _ in range(microbatches)),
            samples,
            capture_time_hooks=hooks,
        )
        bwd_order = list(reversed(range(microbatches)))
    assert len(graphed) == microbatches
    assert weight._fsdp_wgrad_written is False
    for _ in range(2):
        xs = [torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16) for _ in graphed]
        dys = [torch.randn(TOKENS, OUT, device="cuda", dtype=torch.bfloat16) for _ in graphed]
        outs = [g(x) for g, x in zip(graphed, xs)]
        _reset_main_grad(weight)
        reference.weight.grad = None
        for i in bwd_order:  # Replay backwards in capture order.
            if consume:
                _reset_main_grad(weight)
                reference.weight.grad = None
            (outs[i].float() * dys[i].float()).sum().backward()
            (_Uses(reference, uses)(xs[i]).float() * dys[i].float()).sum().backward()
            if consume:
                expected = reference.weight.grad.float()
                atol = 8 * uses * torch.finfo(torch.bfloat16).eps * expected.abs().max().item()
                torch.testing.assert_close(weight.main_grad, expected, rtol=1e-2, atol=atol)
        if not consume:
            expected = reference.weight.grad.float()
            atol = 16 * uses * torch.finfo(torch.bfloat16).eps * expected.abs().max().item()
            torch.testing.assert_close(weight.main_grad, expected, rtol=1e-2, atol=atol)


@pytest.mark.parametrize("module_cls", [te.Linear, te.LayerNormLinear])
@pytest.mark.parametrize("uses", [1, 2])
def test_eager_wgrad_after_graph_replay_accumulates(module_cls, uses):
    """Replay sets the flag its capture set, so an eager wgrad of the same weight after the graphed
    backward accumulates onto the replayed one instead of overwriting it."""
    torch.manual_seed(1234)
    fused = _build(module_cls, True, False)
    reference = _build(module_cls, False, False)
    reference.load_state_dict(fused.state_dict())
    weight = fused.weight
    weight.main_grad = torch.empty(OUT, HIDDEN, device="cuda", dtype=torch.float32)
    _reset_main_grad(weight)
    sample = (torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16),)
    graphed = te.make_graphed_callables(_Uses(fused, uses), sample)
    assert weight._fsdp_wgrad_written is False
    for _ in range(2):
        _reset_main_grad(weight)
        reference.weight.grad = None
        x_eager, x_graph = (
            torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16) for _ in range(2)
        )
        dy = torch.randn(TOKENS, OUT, device="cuda", dtype=torch.bfloat16)
        # Autograd runs the later forward's backward first: the graph replays, then eager runs.
        ((fused(x_eager) + graphed(x_graph)).float() * dy.float()).sum().backward()
        out_ref = reference(x_eager) + _Uses(reference, uses)(x_graph)
        (out_ref.float() * dy.float()).sum().backward()
        expected = reference.weight.grad.float()
        atol = 8 * (uses + 1) * torch.finfo(torch.bfloat16).eps * expected.abs().max().item()
        torch.testing.assert_close(weight.main_grad, expected, rtol=1e-2, atol=atol)
        assert weight._fsdp_wgrad_written


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
