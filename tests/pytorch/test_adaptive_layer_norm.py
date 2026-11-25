# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Adaptive LayerNorm numerics and composition with fusible operations.

Run with NVIDIA_TF32_OVERRIDE=0 to compare TE and PyTorch GEMMs in full FP32,
as configured in qa/L0_pytorch_unittest/test.sh.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

import transformer_engine.pytorch as te
import transformer_engine.pytorch.ops as te_ops
from transformer_engine.common.recipe import DelayedScaling, Float8CurrentScaling


_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


@pytest.fixture(autouse=True)
def _seed_rng():
    torch.manual_seed(1234)


def _tolerances(dtype):
    # The reference accumulates in FP64; the operation accumulates in FP32.
    if dtype == torch.float16:
        return {"rtol": 2e-3, "atol": 2e-3}
    if dtype == torch.bfloat16:
        return {"rtol": 1.6e-2, "atol": 2e-2}
    return {"rtol": 2e-5, "atol": 2e-5}


def _assert_close(actual, expected):
    torch.testing.assert_close(actual, expected.to(actual.dtype), **_tolerances(actual.dtype))


def _condition_view(condition, input_shape, batch_dim):
    if condition.ndim == 2 and len(input_shape) > 2:
        shape = [1] * len(input_shape)
        shape[batch_dim] = input_shape[batch_dim]
        shape[-1] = input_shape[-1]
        return condition.reshape(shape)
    return condition


def _reference(x, scale, shift, *, eps=1e-5, batch_dim=0):
    """Mathematical reference with FP64 statistics and modulation."""
    x = x.double()
    scale = _condition_view(scale.double(), x.shape, batch_dim)
    shift = _condition_view(shift.double(), x.shape, batch_dim)
    normalized = F.layer_norm(x, (x.shape[-1],), eps=eps)
    return normalized * (1 + scale) + shift


def _make_inputs(shape, batch_dim, dtype, *, cond_dtype=None, broadcast=False):
    if cond_dtype is None:
        cond_dtype = dtype
    x = torch.randn(shape, device="cuda", dtype=dtype).requires_grad_()
    batch_size, hidden_size = shape[batch_dim], shape[-1]
    scale = torch.randn(batch_size, hidden_size, device="cuda", dtype=cond_dtype)
    shift = torch.randn_like(scale)
    if broadcast and len(shape) > 2:
        cond_shape = [1] * len(shape)
        cond_shape[batch_dim] = batch_size
        cond_shape[-1] = hidden_size
        scale = scale.reshape(cond_shape)
        shift = shift.reshape(cond_shape)
    return x, scale.requires_grad_(), shift.requires_grad_()


def _check_against_reference(inputs, *, batch_dim=0, eps=1e-5, op=None):
    x, scale, shift = inputs
    reference_inputs = tuple(
        tensor.detach().double().requires_grad_(tensor.requires_grad) for tensor in inputs
    )
    if op is None:
        op = te_ops.AdaptiveLayerNorm(x.shape[-1], eps=eps, batch_dim=batch_dim)
    actual = op(x, scale, shift)
    expected = _reference(*reference_inputs, eps=eps, batch_dim=batch_dim)
    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    _assert_close(actual, expected)
    if any(tensor.requires_grad for tensor in inputs):
        grad = torch.randn_like(actual)
        actual.backward(grad)
        expected.backward(grad.double())
        for tensor, ref_tensor in zip(inputs, reference_inputs):
            if tensor.requires_grad:
                assert tensor.grad is not None
                assert tensor.grad.shape == tensor.shape
                assert tensor.grad.dtype == tensor.dtype
                _assert_close(tensor.grad, ref_tensor.grad)
            else:
                assert tensor.grad is None
    else:
        assert not actual.requires_grad
    return actual


# Cover distinct layout/reduction cases, including non-power-of-two hidden
# sizes and the kernel's smallest/largest advertised hidden dimensions.
_LAYOUTS = (
    ((3, 1), 0, False),
    ((3, 31), 0, False),
    ((2, 7, 128), 0, False),
    ((2, 7, 128), 0, True),
    ((7, 2, 128), 1, False),
    ((7, 2, 128), 1, True),
    ((2, 3, 5, 63), 0, True),
    ((3, 2, 5, 63), 1, False),
    ((3, 5, 2, 63), 2, True),
    ((2, 7, 1536), 0, False),
    ((7, 2, 5120), 1, True),
    ((2, 3, 16384), 0, False),
)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape,batch_dim,broadcast", _LAYOUTS)
def test_forward_backward(shape, batch_dim, broadcast, dtype):
    inputs = _make_inputs(shape, batch_dim, dtype, broadcast=broadcast)
    _check_against_reference(inputs, batch_dim=batch_dim)
    if shape[-1] == 1:
        assert torch.count_nonzero(inputs[0].grad).item() == 0
        assert torch.count_nonzero(inputs[1].grad).item() == 0


@pytest.mark.parametrize(
    "dtype,cond_dtype",
    (
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float16),
        (torch.float32, torch.bfloat16),
    ),
)
@pytest.mark.parametrize("batch_dim", (0, 1))
def test_mixed_condition_dtype(dtype, cond_dtype, batch_dim):
    shape = (3, 11, 256) if batch_dim == 0 else (11, 3, 256)
    inputs = _make_inputs(shape, batch_dim, dtype, cond_dtype=cond_dtype, broadcast=True)
    _check_against_reference(inputs, batch_dim=batch_dim)


@pytest.mark.parametrize(
    "requires_grad",
    (
        (True, False, False),
        (False, True, True),
        (False, True, False),
        (False, False, True),
        (False, False, False),
    ),
)
def test_gradient_requirements(requires_grad):
    inputs = _make_inputs((3, 11, 64), 0, torch.float32)
    for tensor, required in zip(inputs, requires_grad):
        tensor.requires_grad_(required)
    _check_against_reference(inputs)


@pytest.mark.parametrize("batch_dim", (0, 1))
@pytest.mark.parametrize("condition_layout", ("chunk", "strided"))
def test_noncontiguous_inputs(batch_dim, condition_layout):
    shape = (3, 11, 64) if batch_dim == 0 else (11, 3, 64)
    # Non-contiguous hidden dimension, not just a transpose of token axes.
    x = torch.randn(*shape[:-1], 128, device="cuda")[..., ::2]
    x = x.detach().requires_grad_()
    if condition_layout == "chunk":
        packed = torch.randn(3, 128, device="cuda")
        scale, shift = packed.chunk(2, dim=-1)
    else:
        packed = torch.randn(3, 256, device="cuda")
        scale, shift = packed[:, :128:2], packed[:, 128::2]
    scale = scale.detach().requires_grad_()
    shift = shift.detach().requires_grad_()
    assert not x.is_contiguous()
    assert not scale.is_contiguous()
    assert not shift.is_contiguous()
    _check_against_reference((x, scale, shift), batch_dim=batch_dim)


@pytest.mark.parametrize("eps", (1e-6, 1e-3))
def test_nearly_constant_input(eps):
    inputs = _make_inputs((2, 7, 96), 0, torch.float32)
    with torch.no_grad():
        inputs[0].mul_(1e-4).add_(1)
    dy = torch.randn_like(inputs[0])
    output = te_ops.AdaptiveLayerNorm(96, eps=eps)(*inputs)
    actual = (output, *torch.autograd.grad(output, inputs, dy))

    references = []
    for dtype in (torch.float64, torch.float32):
        x, scale, shift = (tensor.detach().to(dtype).requires_grad_() for tensor in inputs)
        y = F.layer_norm(x, (96,), eps=eps)
        y = y * (1 + scale[:, None, :]) + shift[:, None, :]
        references.append((y, *torch.autograd.grad(y, (x, scale, shift), dy.to(dtype))))

    # The small variance amplifies FP32 mean rounding around one. Compare to
    # FP64, allowing the error scale of native PyTorch's FP32 LayerNorm rather
    # than requiring FP64 statistics from either FP32 implementation.
    for result, exact, native in zip(actual, references[0], references[1]):
        native_error = (native.double() - exact).abs().max()
        rounding_floor = 32 * torch.finfo(torch.float32).eps * exact.abs().max().clamp_min(1)
        atol = torch.maximum(2 * native_error, rounding_floor).item()
        torch.testing.assert_close(result, exact.to(result.dtype), rtol=2e-5, atol=atol)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_small_zero_centered_scale(dtype):
    # Forming (1 + scale) in the input dtype would round it back to one.
    # Keep x in FP32 so this lost modulation remains visible in the output.
    x = torch.tensor([[[-1.0, 1.0, -1.0, 1.0]]], device="cuda", requires_grad=True)
    value = 2**-12 if dtype == torch.float16 else 2**-10
    scale = torch.full((1, 4), value, device="cuda", dtype=dtype, requires_grad=True)
    shift = torch.zeros_like(scale, requires_grad=True)
    actual = _check_against_reference((x, scale, shift))
    rounded_scale_result = F.layer_norm(x, (4,)) * (1 + scale).float().unsqueeze(1)
    assert (actual - rounded_scale_result).abs().max().item() > value / 2


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("batch_dim", (0, 1))
def test_sequential_upstream_and_downstream_gradients(dtype, batch_dim, monkeypatch):
    """The parameter-free op must preserve gradients through upstream Linear."""
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    first = te_ops.Linear(32, 64, device="cuda", dtype=dtype)
    last = te_ops.Linear(64, 16, device="cuda", dtype=dtype)
    adaptive = te_ops.AdaptiveLayerNorm(64, batch_dim=batch_dim)
    model = te_ops.Sequential(first, adaptive, last)
    shape = (3, 7, 32) if batch_dim == 0 else (7, 3, 32)
    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    scale = torch.randn(3, 64, device="cuda", dtype=torch.float32, requires_grad=True)
    shift = torch.randn_like(scale, requires_grad=True)
    tensors = (x, scale, shift, first.weight, first.bias, last.weight, last.bias)
    rx, rs, rb = (tensor.detach().clone().requires_grad_() for tensor in (x, scale, shift))
    ref_first, ref_last = copy.deepcopy(first), copy.deepcopy(last)
    reference_tensors = (
        rx,
        rs,
        rb,
        ref_first.weight,
        ref_first.bias,
        ref_last.weight,
        ref_last.bias,
    )
    # Keep the GEMM implementation and rounding boundaries identical. TE and
    # PyTorch BF16 GEMMs may differ by one ULP, amplified by later weight grads.
    # Normalization/modulation still use the independent FP64 reference.
    expected = ref_first(rx)
    expected = _reference(expected, rs, rb, batch_dim=batch_dim).to(dtype)
    expected = ref_last(expected)
    actual = model(x, scale, shift)
    _assert_close(actual, expected)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    for tensor, reference_tensor in zip(tensors, reference_tensors):
        assert tensor.grad is not None
        assert reference_tensor.grad is not None
        # BF16 linear outputs are rounded before normalization in both paths.
        # Dcondition is FP32 but inherits the activation precision of the MLP.
        tolerances = _tolerances(dtype)
        torch.testing.assert_close(tensor.grad, reference_tensor.grad, **tolerances)


def test_conditions_change_between_calls():
    op = te_ops.AdaptiveLayerNorm(64)
    assert list(op.parameters()) == []
    for batch_size, tokens in ((2, 7), (3, 11)):
        inputs = _make_inputs((batch_size, tokens, 64), 0, torch.float32)
        _check_against_reference(inputs, op=op)
    assert list(op.parameters()) == []


@pytest.mark.parametrize(
    "shape,batch_dim,broadcast",
    (
        ((0, 7, 64), 0, False),
        ((3, 0, 64), 0, True),
        ((0, 3, 64), 1, False),
        ((7, 0, 64), 1, True),
    ),
)
def test_empty_input(shape, batch_dim, broadcast):
    inputs = _make_inputs(shape, batch_dim, torch.float32, broadcast=broadcast)
    actual = te_ops.AdaptiveLayerNorm(64, batch_dim=batch_dim)(*inputs)
    assert actual.shape == shape
    actual.sum().backward()
    for tensor in inputs:
        assert tensor.grad is not None
        assert tensor.grad.shape == tensor.shape
        assert torch.count_nonzero(tensor.grad).item() == 0


@pytest.mark.parametrize("hidden_size", (0, -1, 16385))
def test_invalid_hidden_size(hidden_size):
    with pytest.raises((TypeError, ValueError)):
        te_ops.AdaptiveLayerNorm(hidden_size)


@pytest.mark.parametrize("eps", (-1e-5, float("nan"), float("inf")))
def test_invalid_eps(eps):
    with pytest.raises((TypeError, ValueError)):
        te_ops.AdaptiveLayerNorm(64, eps=eps)


@pytest.mark.parametrize("batch_dim", (2, 3, -4))
def test_invalid_batch_axis(batch_dim):
    inputs = _make_inputs((3, 7, 64), 0, torch.float32)
    with pytest.raises((TypeError, ValueError)):
        te_ops.AdaptiveLayerNorm(64, batch_dim=batch_dim)(*inputs)


@pytest.mark.parametrize(
    "input_shape,scale_shape,shift_shape",
    (
        ((64,), (1, 64), (1, 64)),
        ((3, 7, 32), (3, 64), (3, 64)),
        ((3, 7, 64), (2, 64), (3, 64)),
        ((3, 7, 64), (3, 64), (2, 64)),
        ((3, 7, 64), (3, 32), (3, 64)),
        ((3, 7, 64), (3, 7, 64), (3, 1, 64)),
        ((3, 7, 64), (64,), (3, 64)),
        ((3, 7, 64), (3, 1, 1, 64), (3, 64)),
    ),
)
def test_invalid_tensor_shapes(input_shape, scale_shape, shift_shape):
    tensors = tuple(
        torch.randn(shape, device="cuda") for shape in (input_shape, scale_shape, shift_shape)
    )
    with pytest.raises((TypeError, ValueError)):
        te_ops.AdaptiveLayerNorm(64)(*tensors)


@pytest.mark.parametrize("argument", (0, 1, 2))
def test_invalid_tensor_dtype(argument):
    inputs = list(_make_inputs((3, 7, 64), 0, torch.float32))
    inputs[argument] = inputs[argument].detach().to(torch.int32)
    with pytest.raises((TypeError, ValueError)):
        te_ops.AdaptiveLayerNorm(64)(*inputs)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_autocast_preserves_input_dtype(dtype):
    inputs = _make_inputs((2, 7, 64), 0, dtype, cond_dtype=torch.float32)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _check_against_reference(inputs)


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_condition_projection_gradients(dtype, monkeypatch):
    """Condition gradients must reach the trainable timestep projection."""
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    hidden_size = 64
    projection = te_ops.Linear(16, 2 * hidden_size, device="cuda", dtype=dtype)
    conditioner = te_ops.Sequential(projection)
    op = te_ops.AdaptiveLayerNorm(hidden_size, batch_dim=1)
    # Neither data input requires a gradient: projection parameters are the
    # only reason the adaptive operation participates in backward.
    x = torch.randn(7, 3, hidden_size, device="cuda", dtype=dtype)
    timestep = torch.randn(3, 16, device="cuda", dtype=dtype)
    scale, shift = conditioner(timestep).chunk(2, dim=-1)
    actual = op(x, scale, shift)
    ref_weight = projection.weight.detach().clone().requires_grad_()
    ref_bias = projection.bias.detach().clone().requires_grad_()
    ref_scale, ref_shift = F.linear(timestep, ref_weight, ref_bias).chunk(2, dim=-1)
    expected = _reference(x, ref_scale, ref_shift, batch_dim=1).to(dtype)
    _assert_close(actual, expected)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    assert x.grad is None
    assert timestep.grad is None
    for param, ref_param in (
        (projection.weight, ref_weight),
        (projection.bias, ref_bias),
    ):
        assert param.grad is not None
        _assert_close(param.grad, ref_param.grad)


@pytest.mark.parametrize("batch_dim", (0, 1))
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_long_sequence_reduction_is_repeatable(batch_dim, dtype):
    """Exercise split condition-gradient reduction across multiple token tiles."""
    shape = (2, 257, 65) if batch_dim == 0 else (257, 2, 65)
    inputs = _make_inputs(shape, batch_dim, dtype, cond_dtype=torch.float32)
    reference_inputs = tuple(tensor.detach().double().requires_grad_() for tensor in inputs)
    expected = _reference(*reference_inputs, batch_dim=batch_dim)
    grad = torch.randn(shape, device="cuda", dtype=dtype)
    expected.backward(grad.double())
    op = te_ops.AdaptiveLayerNorm(65, batch_dim=batch_dim)
    previous = None
    for _ in range(2):
        actual = op(*inputs)
        actual.backward(grad)
        _assert_close(actual, expected)
        for tensor, ref_tensor in zip(inputs, reference_inputs):
            _assert_close(tensor.grad, ref_tensor.grad)
        results = (actual.detach().clone(), *(tensor.grad.clone() for tensor in inputs))
        if previous is not None:
            for result, old_result in zip(results, previous):
                assert torch.equal(result, old_result)
        previous = results
        for tensor in inputs:
            tensor.grad = None


@pytest.mark.parametrize("recipe_type", (DelayedScaling, Float8CurrentScaling))
def test_fp8_sequential_matches_separate_operations(recipe_type):
    """Preserve all gradients when FP8 linears surround the unquantized op."""
    available, reason = te.is_fp8_available(return_reason=True)
    if not available:
        pytest.skip(reason)
    dtype = torch.bfloat16
    operations = (
        te_ops.Linear(64, 64, device="cuda", dtype=dtype),
        te_ops.AdaptiveLayerNorm(64, batch_dim=1),
        te_ops.Linear(64, 64, device="cuda", dtype=dtype),
    )
    separate = tuple(copy.deepcopy(op) for op in operations)
    model = te_ops.Sequential(*operations)
    inputs = _make_inputs((8, 2, 64), 1, dtype, cond_dtype=torch.float32)
    reference_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in inputs)
    with te.autocast(recipe=recipe_type()):
        actual = model(*inputs)
    with te.autocast(recipe=recipe_type()):
        x, scale, shift = reference_inputs
        expected = separate[2](separate[1](separate[0](x), scale, shift))
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    assert actual.dtype == dtype
    _assert_close(actual, expected)
    parameters = tuple(model.parameters())
    reference_parameters = tuple(param for op in separate for param in op.parameters())
    for tensor, reference_tensor in zip(
        inputs + parameters, reference_inputs + reference_parameters
    ):
        assert tensor.grad is not None
        assert reference_tensor.grad is not None
        torch.testing.assert_close(tensor.grad, reference_tensor.grad, **_tolerances(dtype))
