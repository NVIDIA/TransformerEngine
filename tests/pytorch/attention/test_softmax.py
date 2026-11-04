# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Numerical, allocation, and custom-op contracts for fused softmax backward."""

import pytest
import torch

from transformer_engine.pytorch.attention.dot_product_attention import softmax  # noqa: F401
from transformer_engine.pytorch.attention.dot_product_attention.backends import (
    UnfusedDotProductAttention,
)
from transformer_engine.pytorch.utils import attention_mask_func


_VARIANTS = (
    "scaled_softmax",
    "scaled_masked_softmax",
    "scaled_upper_triang_masked_softmax",
    "scaled_aligned_causal_masked_softmax",
)


def _softmax_case(variant, dtype, seq_len=128, seed=1234):
    """Return a TE operation and its independent FP32 reference."""
    torch.manual_seed(seed)
    sq = seq_len // 2 if variant == "scaled_aligned_causal_masked_softmax" else seq_len
    shape = (
        (8, sq, seq_len) if variant == "scaled_upper_triang_masked_softmax" else (2, 4, sq, seq_len)
    )
    inputs = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    reference = inputs.detach().float().requires_grad_(True)
    scale = 0.7
    scores = reference * scale
    args = (scale,)
    if variant == "scaled_masked_softmax":
        mask = torch.rand((2, 1, sq, seq_len), device="cuda") > 0.75
        mask[..., 0] = False  # Keep every row nonempty.
        scores = scores.masked_fill(mask, -10000.0)
        args = (mask, scale)
    elif "causal" in variant or "upper_triang" in variant:
        mask = torch.ones((sq, seq_len), device="cuda", dtype=torch.bool).triu(seq_len - sq + 1)
        scores = scores.masked_fill(mask, float("-inf"))
    return inputs, reference, scores.softmax(dim=-1), args


def _check_close(actual, expected, dtype):
    # BF16/FP16 outputs and the saved softmax probabilities are rounded once.
    rtol, atol = (2e-2, 2e-3) if dtype == torch.bfloat16 else (3e-3, 3e-4)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("variant", _VARIANTS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seq_len", [32, 128, 512])
@pytest.mark.parametrize("noncontiguous_grad", [False, True])
@pytest.mark.parametrize("create_graph", [False, True])
@pytest.mark.parametrize("overwrite", [False, True])
def test_softmax_backward(variant, dtype, seq_len, noncontiguous_grad, create_graph, overwrite):
    """Compare eager forward and backward with native FP32 softmax."""
    inputs, reference, expected, args = _softmax_case(variant, dtype, seq_len)
    output = getattr(torch.ops.te_softmax, variant + "_fwd")(inputs, *args, overwrite=overwrite)
    grad = torch.randn_like(output)
    if noncontiguous_grad:
        grad = grad.transpose(-1, -2).contiguous().transpose(-1, -2)
        assert not grad.is_contiguous()
    expected_grad = torch.autograd.grad(expected, reference, grad.float())[0]
    saved_grad = grad.clone()
    actual_grad = torch.autograd.grad(output, inputs, grad, create_graph=create_graph)[0]
    if overwrite and not create_graph and not noncontiguous_grad:
        assert actual_grad.data_ptr() == grad.data_ptr()
    else:
        torch.testing.assert_close(grad, saved_grad, rtol=0, atol=0)
        assert actual_grad.data_ptr() != grad.data_ptr()
    _check_close(output, expected, dtype)
    _check_close(actual_grad, expected_grad, dtype)


@pytest.mark.parametrize("variant", _VARIANTS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("noncontiguous_grad", [False, True])
def test_softmax_backward_custom_op(variant, dtype, noncontiguous_grad):
    """The compiler-facing backward must neither mutate nor alias either input."""
    inputs, reference, expected, args = _softmax_case(variant, dtype)
    output = getattr(torch.ops.te_softmax, variant + "_fwd")(inputs, *args).detach()
    grad = torch.randn_like(output)
    if noncontiguous_grad:
        grad = grad.transpose(-1, -2).contiguous().transpose(-1, -2)
    saved_grad, saved_output = grad.clone(), output.clone()
    expected_grad = torch.autograd.grad(expected, reference, grad.float())[0]
    backward = getattr(torch.ops.te_softmax, variant + "_bwd")
    actual_grad = backward(grad, output, args[-1])
    assert actual_grad.data_ptr() not in (grad.data_ptr(), output.data_ptr())
    torch.testing.assert_close(grad, saved_grad, rtol=0, atol=0)
    torch.testing.assert_close(output, saved_output, rtol=0, atol=0)
    _check_close(actual_grad, expected_grad, dtype)
    torch.library.opcheck(backward, (grad, output, args[-1]))


@pytest.mark.parametrize("variant", _VARIANTS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("overwrite", [False, True])
def test_softmax_backward_eager_memory(variant, dtype, overwrite):
    """Only explicit buffer donation may avoid allocating a gradient tensor."""
    inputs, _, _, args = _softmax_case(variant, dtype, seq_len=512)
    forward = getattr(torch.ops.te_softmax, variant + "_fwd")
    # Warm up kernel compilation and autograd before measuring live allocations.
    output = forward(inputs, *args, overwrite=overwrite)
    torch.autograd.grad(output, inputs, torch.randn_like(output))
    output = forward(inputs, *args, overwrite=overwrite)
    grad = torch.randn_like(output)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated = torch.cuda.memory_allocated()
    result = torch.autograd.grad(output, inputs, grad)[0]
    torch.cuda.synchronize()
    peak_delta = torch.cuda.max_memory_allocated() - allocated
    score_bytes = grad.numel() * grad.element_size()
    if overwrite:
        assert peak_delta < score_bytes, (peak_delta, score_bytes)
        assert result.data_ptr() == grad.data_ptr()
    else:
        assert peak_delta >= score_bytes, (peak_delta, score_bytes)
        assert result.data_ptr() != grad.data_ptr()


@pytest.mark.parametrize("variant", _VARIANTS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("overwrite", [None, False, True])
@pytest.mark.parametrize("noncontiguous_grad", [False, True])
def test_softmax_backward_binding(variant, dtype, overwrite, noncontiguous_grad):
    """Native backward defaults to a separate result and reports explicit mutations."""
    inputs, reference, expected, args = _softmax_case(variant, dtype)
    output = getattr(torch.ops.te_softmax, variant + "_fwd")(inputs, *args).detach()
    grad = torch.randn_like(output)
    if noncontiguous_grad:
        grad = grad.transpose(-1, -2).contiguous().transpose(-1, -2)
    saved_grad, saved_output = grad.clone(), output.clone()
    version = grad._version
    expected_grad = torch.autograd.grad(expected, reference, grad.float())[0]
    backward = getattr(softmax.tex, variant + "_backward")
    kwargs = {} if overwrite is None else {"overwrite": overwrite}
    actual_grad = backward(grad, output, args[-1], **kwargs)
    if overwrite and not noncontiguous_grad:
        assert actual_grad.data_ptr() == grad.data_ptr()
        assert grad._version == version + 1
    else:
        assert actual_grad.data_ptr() != grad.data_ptr()
        torch.testing.assert_close(grad, saved_grad, rtol=0, atol=0)
        assert grad._version == version
    torch.testing.assert_close(output, saved_output, rtol=0, atol=0)
    _check_close(actual_grad, expected_grad, dtype)


@pytest.mark.parametrize("variant", _VARIANTS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("reverse", [False, True])
def test_softmax_backward_shared_gradient(variant, dtype, reverse):
    """AddBackward may supply the same gradient to softmax and another branch."""
    inputs, reference, expected, args = _softmax_case(variant, dtype)
    output = getattr(torch.ops.te_softmax, variant + "_fwd")(inputs, *args)
    residual = torch.randn_like(output, requires_grad=True)
    combined = residual + output if reverse else output + residual
    grad = torch.randn_like(output)
    saved_grad = grad.clone()
    expected_grad = torch.autograd.grad(expected, reference, grad.float())[0]
    actual_grad, residual_grad = torch.autograd.grad(combined, (inputs, residual), grad)
    torch.testing.assert_close(grad, saved_grad, rtol=0, atol=0)
    torch.testing.assert_close(residual_grad, saved_grad, rtol=0, atol=0)
    _check_close(actual_grad, expected_grad, dtype)


@pytest.mark.parametrize("variant", _VARIANTS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_softmax_backward_preserves_saved_gradient(variant, dtype):
    """A supplied gradient may already be saved by an independent autograd graph."""
    inputs, _, _, args = _softmax_case(variant, dtype)
    output = getattr(torch.ops.te_softmax, variant + "_fwd")(inputs, *args)
    grad = torch.randn_like(output, requires_grad=True)
    saved_grad = grad.detach().clone()
    other_loss = grad.square().sum()
    torch.autograd.grad(output, inputs, grad)
    other_grad = torch.autograd.grad(other_loss, grad)[0]
    torch.testing.assert_close(other_grad, 2 * saved_grad, rtol=0, atol=0)


@pytest.mark.parametrize("global_hook", [False, True])
@pytest.mark.parametrize("hook", ["forward", "forward_pre", "full_backward", "full_backward_pre"])
def test_softmax_overwrite_with_hooks(global_hook, hook):
    """Local and global module observers must receive the original gradient."""
    module = softmax.FusedScaleMaskSoftmax(attention_mask_func)
    observed = []

    def observe(*args):
        if hook.startswith("full_backward"):
            observed.append(args[-1][0])
        elif hook == "forward":
            args[-1].register_hook(lambda grad: observed.append(grad))

    if global_hook:
        register = getattr(softmax.nn_module, "register_module_" + hook + "_hook")
    else:
        register = getattr(module, "register_" + hook + "_hook")
    handle = register(observe)
    try:
        inputs = torch.randn(
            (2, 4, 32, 32), dtype=torch.bfloat16, device="cuda", requires_grad=True
        )
        output = module(inputs, None, "no_mask", overwrite=True)
        grad = torch.randn_like(output)
        saved_grad = grad.clone()
        result = torch.autograd.grad(output, inputs, grad)[0]
        torch.testing.assert_close(grad, saved_grad, rtol=0, atol=0)
        assert result.data_ptr() != grad.data_ptr()
        for value in observed:
            torch.testing.assert_close(value, saved_grad, rtol=0, atol=0)
    finally:
        handle.remove()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mask", ["no_mask", "causal"])
@pytest.mark.parametrize("dropout", [0.0, 0.1])
@pytest.mark.parametrize("hook_target", [None, "scale_mask_softmax", "attention_dropout"])
def test_unfused_attention_softmax_overwrite(dtype, mask, dropout, hook_target, monkeypatch):
    """Unfused attention opts in only when its private gradient has no observers."""
    policy = softmax._softmax_bwd_in_place
    native_calls = []
    for name in ["scaled_softmax_backward", "scaled_masked_softmax_backward"]:
        original = getattr(softmax.tex, name)

        def backward(*args, _original=original, **kwargs):
            native_calls.append(kwargs.get("overwrite", False))
            return _original(*args, **kwargs)

        monkeypatch.setattr(softmax.tex, name, backward)

    def run(functional):
        torch.manual_seed(1234)
        monkeypatch.setattr(
            softmax, "_softmax_bwd_in_place", (lambda *_: False) if functional else policy
        )
        q, k, v = [
            torch.randn((32, 2, 4, 16), device="cuda", dtype=dtype, requires_grad=True)
            for _ in range(3)
        ]
        module = UnfusedDotProductAttention(16**-0.5, attention_dropout=dropout)
        observed = []
        if hook_target:
            getattr(module, hook_target).register_full_backward_hook(
                lambda _module, _inputs, outputs: observed.append(outputs[0])
            )
        output = module({}, q, k, v, qkv_layout="sbhd_sbhd_sbhd", attn_mask_type=mask)
        residual = torch.randn_like(output, requires_grad=True)
        grad = torch.randn_like(output)
        saved_grad = grad.clone()
        native_calls.clear()
        gradients = torch.autograd.grad(output + residual, (q, k, v, residual), grad)
        torch.testing.assert_close(grad, saved_grad, rtol=0, atol=0)
        torch.testing.assert_close(gradients[-1], saved_grad, rtol=0, atol=0)
        assert native_calls == [not functional and hook_target is None]
        return output, gradients, observed

    expected, expected_gradients, expected_observed = run(True)
    actual, actual_gradients, actual_observed = run(False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_grad, expected_grad in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    for actual_grad, expected_grad in zip(actual_observed, expected_observed):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("variant", _VARIANTS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("overwrite", [False, True])
def test_softmax_torch_compile(variant, dtype, overwrite):
    """Compile and replay forward/backward for every fused softmax variant."""
    forward = getattr(torch.ops.te_softmax, variant + "_fwd")
    torch._dynamo.reset()
    compiled = torch.compile(forward, fullgraph=True, mode="reduce-overhead")
    for iteration in range(3):
        torch.compiler.cudagraph_mark_step_begin()
        # Vary the data to catch replay of stale buffers.
        inputs, reference, expected, args = _softmax_case(variant, dtype, seed=1234 + iteration)
        output = compiled(inputs, *args, overwrite=overwrite)
        grad = torch.randn_like(output)
        saved_grad = grad.clone()
        expected_grad = torch.autograd.grad(expected, reference, grad.float())[0]
        actual_grad = torch.autograd.grad(output, inputs, grad)[0]
        torch.testing.assert_close(grad, saved_grad, rtol=0, atol=0)
        _check_close(output, expected, dtype)
        _check_close(actual_grad, expected_grad, dtype)
        del output, actual_grad
