# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Numerical, allocation, and custom-op contracts for fused softmax backward."""

import pytest
import torch

from transformer_engine.pytorch.attention.dot_product_attention import softmax  # noqa: F401


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
def test_softmax_backward(variant, dtype, seq_len, noncontiguous_grad, create_graph):
    """Compare eager forward and backward with native FP32 softmax."""
    inputs, reference, expected, args = _softmax_case(variant, dtype, seq_len)
    output = getattr(torch.ops.te_softmax, variant + "_fwd")(inputs, *args)
    grad = torch.randn_like(output)
    if noncontiguous_grad:
        grad = grad.transpose(-1, -2).contiguous().transpose(-1, -2)
        assert not grad.is_contiguous()
    expected_grad = torch.autograd.grad(expected, reference, grad.float())[0]
    saved_grad = grad.clone() if create_graph else None
    actual_grad = torch.autograd.grad(output, inputs, grad, create_graph=create_graph)[0]
    if create_graph:
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
def test_softmax_backward_eager_memory(variant, dtype):
    """Eager backward must not allocate another attention-score-sized tensor."""
    inputs, _, _, args = _softmax_case(variant, dtype, seq_len=512)
    forward = getattr(torch.ops.te_softmax, variant + "_fwd")
    # Warm up kernel compilation and autograd before measuring live allocations.
    output = forward(inputs, *args)
    torch.autograd.grad(output, inputs, torch.randn_like(output))
    output = forward(inputs, *args)
    grad = torch.randn_like(output)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated = torch.cuda.memory_allocated()
    result = torch.autograd.grad(output, inputs, grad)[0]
    torch.cuda.synchronize()
    peak_delta = torch.cuda.max_memory_allocated() - allocated
    score_bytes = grad.numel() * grad.element_size()
    assert peak_delta < score_bytes, (peak_delta, score_bytes)
    assert result.data_ptr() == grad.data_ptr()


@pytest.mark.parametrize("variant", _VARIANTS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_softmax_torch_compile(variant, dtype):
    """Compile and replay forward/backward for every fused softmax variant."""
    forward = getattr(torch.ops.te_softmax, variant + "_fwd")
    torch._dynamo.reset()
    compiled = torch.compile(forward, fullgraph=True, mode="reduce-overhead")
    for iteration in range(3):
        torch.compiler.cudagraph_mark_step_begin()
        # Vary the data to catch replay of stale buffers.
        inputs, reference, expected, args = _softmax_case(variant, dtype, seed=1234 + iteration)
        output = compiled(inputs, *args)
        grad = torch.randn_like(output)
        expected_grad = torch.autograd.grad(expected, reference, grad.float())[0]
        actual_grad = torch.autograd.grad(output, inputs, grad)[0]
        _check_close(output, expected, dtype)
        _check_close(actual_grad, expected_grad, dtype)
        del output, actual_grad
