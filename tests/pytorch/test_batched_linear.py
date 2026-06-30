# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for the PyTorch BatchedLinear module."""

from __future__ import annotations

import contextlib

import pytest
import torch

import transformer_engine.common.recipe
import transformer_engine.pytorch as te
import transformer_engine.pytorch.module.batched_linear as batched_linear_module
from transformer_engine.pytorch.quantization import FP8GlobalStateManager
from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage
from utils import assert_close, dtype_tols, make_recipe, quantization_tols

mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)


@pytest.fixture(autouse=True)
def reset_global_fp8_state():
    """Keep FP8 global state isolated between tests."""
    yield
    FP8GlobalStateManager.reset()


def _to_reference(tensor: torch.Tensor, *, requires_grad: bool) -> torch.Tensor:
    """Make an FP64 CPU reference tensor."""
    if isinstance(tensor, QuantizedTensorStorage):
        tensor = tensor.dequantize()
    return tensor.detach().to(dtype=torch.float64, device="cpu").requires_grad_(requires_grad)


def _reference_linear(
    inp: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    batch_dim: int,
) -> torch.Tensor:
    """Reference batched linear with the two supported input layouts."""
    if batch_dim == 0:
        out = torch.einsum("g...d,grd->g...r", inp, weight)
        bias_shape = [bias.size(0)] + [1] * (out.ndim - 2) + [bias.size(1)]
        return out + bias.view(bias_shape)
    return torch.einsum("...gd,grd->...gr", inp, weight) + bias


@pytest.mark.parametrize("batch_dim", (0, -2))
@pytest.mark.parametrize("quantized_compute", (False, True))
@pytest.mark.parametrize("quantized_weight", (False, True))
@pytest.mark.parametrize(
    "accumulate_into_main_grad,overwrite_main_grad",
    ((False, False), (True, False), (True, True)),
)
def test_forward_backward(
    monkeypatch: pytest.MonkeyPatch,
    *,
    batch_dim: int,
    quantized_compute: bool,
    quantized_weight: bool,
    accumulate_into_main_grad: bool,
    overwrite_main_grad: bool,
) -> None:
    """Check numerics, layouts, bias, and main-grad accumulation."""
    if (quantized_compute or quantized_weight) and not mxfp8_available:
        pytest.skip(reason_for_no_mxfp8)

    dtype = torch.bfloat16
    device = torch.device("cuda")
    num_gemms, rows0, rows1 = 3, 3, 32
    in_features, out_features = 160, 96
    if batch_dim == 0:
        input_shape = (num_gemms, rows0, rows1, in_features)
        output_shape = (num_gemms, rows0, rows1, out_features)
    else:
        input_shape = (rows0, rows1, num_gemms, in_features)
        output_shape = (rows0, rows1, num_gemms, out_features)

    inp = torch.rand(input_shape, dtype=dtype, device=device, requires_grad=True)
    weight = torch.rand(
        num_gemms,
        out_features,
        in_features,
        dtype=dtype,
        device=device,
    )
    bias = torch.rand(num_gemms, out_features, dtype=dtype, device=device)
    grad_output = torch.rand(output_shape, dtype=dtype, device=device)

    recipe = make_recipe("mxfp8")
    with te.quantized_model_init(enabled=quantized_weight, recipe=recipe):
        module = te.BatchedLinear(
            num_gemms,
            in_features,
            out_features,
            batch_dim=batch_dim,
            accumulate_into_main_grad=accumulate_into_main_grad,
            params_dtype=dtype,
            device=device,
        )
    with torch.no_grad():
        module.weight.copy_(weight)
        module.bias.copy_(bias)
        module.weight.main_grad = torch.full(
            module.weight.shape,
            0.5,
            dtype=torch.float32,
            device=device,
        )
        module.weight.overwrite_main_grad = overwrite_main_grad

    inp_ref = _to_reference(inp, requires_grad=True)
    weight_ref = _to_reference(module.weight, requires_grad=True)
    bias_ref = _to_reference(module.bias, requires_grad=True)
    grad_output_ref = _to_reference(grad_output, requires_grad=False)
    output_ref = _reference_linear(inp_ref, weight_ref, bias_ref, batch_dim)
    output_ref.backward(grad_output_ref)

    gemm_calls = []
    original_strided_batched_gemm = batched_linear_module.strided_batched_gemm

    def capture_strided_batched_gemm(*args, **kwargs):
        gemm_calls.append((kwargs["layout"], args[0], args[1], kwargs.get("accumulate", False)))
        return original_strided_batched_gemm(*args, **kwargs)

    monkeypatch.setattr(
        batched_linear_module,
        "strided_batched_gemm",
        capture_strided_batched_gemm,
    )
    with te.autocast(enabled=quantized_compute, recipe=recipe):
        output = module(inp)
    output.backward(grad_output)

    assert tuple(output.shape) == output_shape
    assert output.is_contiguous()
    tols = (
        quantization_tols("mxfp8") if quantized_compute or quantized_weight else dtype_tols(dtype)
    )
    assert_close(output, output_ref, **tols)
    assert_close(inp.grad, inp_ref.grad, **tols)
    assert_close(module.bias.grad, bias_ref.grad, **tols)

    if accumulate_into_main_grad:
        assert module.weight.grad is None
        grad_weight = module.weight.main_grad
        if not overwrite_main_grad:
            grad_weight = grad_weight - 0.5
        assert_close(grad_weight, weight_ref.grad, **tols)
    else:
        assert_close(module.weight.grad, weight_ref.grad, **tols)
        torch.testing.assert_close(
            module.weight.main_grad,
            torch.full_like(module.weight.main_grad, 0.5),
            rtol=0,
            atol=0,
        )

    assert [layout for layout, _, _, _ in gemm_calls] == ["TN", "NN", "NT"]
    assert gemm_calls[-1][3] == (accumulate_into_main_grad and not overwrite_main_grad)
    if quantized_compute:

        def assert_mxfp8_usage(tensor, *, rowwise: bool, columnwise: bool) -> None:
            assert (tensor._rowwise_data is not None) == rowwise
            assert (tensor._columnwise_data is not None) == columnwise

        _, fprop_weight, fprop_input, _ = gemm_calls[0]
        _, dgrad_weight, dgrad_grad_output, _ = gemm_calls[1]
        _, wgrad_input, wgrad_grad_output, _ = gemm_calls[2]
        assert_mxfp8_usage(fprop_weight, rowwise=True, columnwise=True)
        assert_mxfp8_usage(fprop_input, rowwise=True, columnwise=True)
        assert_mxfp8_usage(dgrad_weight, rowwise=True, columnwise=True)
        assert_mxfp8_usage(dgrad_grad_output, rowwise=True, columnwise=True)
        assert_mxfp8_usage(wgrad_input, rowwise=True, columnwise=True)
        assert_mxfp8_usage(wgrad_grad_output, rowwise=True, columnwise=True)
        assert dgrad_weight is fprop_weight
        assert wgrad_input is fprop_input
        assert dgrad_grad_output is wgrad_grad_output
        if quantized_weight:
            assert fprop_weight is module.weight
            assert not module.weight._with_gemm_swizzled_scales


@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
@pytest.mark.parametrize("batch_dim", (0, -2))
@pytest.mark.parametrize(
    "input_requires_grad,weight_requires_grad",
    ((True, False), (False, True)),
)
def test_mxfp8_selective_gradients(
    *,
    batch_dim: int,
    input_requires_grad: bool,
    weight_requires_grad: bool,
) -> None:
    """MXFP8 backward supports independently frozen inputs and weights."""
    recipe = make_recipe("mxfp8")
    module = te.BatchedLinear(
        2,
        64,
        32,
        batch_dim=batch_dim,
        bias=False,
        params_dtype=torch.bfloat16,
        device="cuda",
    )
    module.weight.requires_grad_(weight_requires_grad)
    input_shape = (2, 32, 64) if batch_dim == 0 else (32, 2, 64)
    inp = torch.rand(
        input_shape,
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=input_requires_grad,
    )

    with te.autocast(enabled=True, recipe=recipe):
        output = module(inp)
    output.sum().backward()

    assert (inp.grad is not None) == input_requires_grad
    assert (module.weight.grad is not None) == weight_requires_grad


@pytest.mark.parametrize("deferred_init", (False, True))
def test_init_method_and_rng_tracker(deferred_init: bool) -> None:
    """Initialization uses the requested RNG tracker, including after meta init."""
    events = []

    class TestRNGTracker:
        @contextlib.contextmanager
        def fork(self):
            events.append("enter")
            try:
                yield
            finally:
                events.append("exit")

    tracker = TestRNGTracker()

    def get_rng_tracker():
        events.append("tracker")
        return tracker

    def init_method(weight):
        assert events[-1] == "enter"
        events.append("init")
        torch.nn.init.constant_(weight, 0.25)

    module = te.BatchedLinear(
        2,
        32,
        64,
        device="meta" if deferred_init else "cuda",
        params_dtype=torch.bfloat16,
        rng_state_tracker_function=get_rng_tracker,
        init_method=init_method,
    )
    if deferred_init:
        assert events == []
        module.reset_parameters()

    assert events == ["tracker", "enter", "init", "exit"]
    torch.testing.assert_close(
        module.weight,
        torch.full_like(module.weight, 0.25),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(module.bias, torch.zeros_like(module.bias), rtol=0, atol=0)


@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
@pytest.mark.parametrize("deferred_init", (False, True))
def test_preserve_high_precision_init_val(deferred_init: bool) -> None:
    """MXFP8 weights retain the pre-quantization CPU value on request."""
    recipe = make_recipe("mxfp8")

    def init_method(weight):
        torch.nn.init.constant_(weight, 0.25)

    with te.quantized_model_init(
        enabled=True,
        recipe=recipe,
        preserve_high_precision_init_val=True,
    ):
        module = te.BatchedLinear(
            2,
            32,
            64,
            device="meta" if deferred_init else "cuda",
            params_dtype=torch.bfloat16,
            init_method=init_method,
        )
    if deferred_init:
        module.reset_parameters()

    weight = module.weight
    assert isinstance(weight, te.QuantizedTensor)
    high_precision = weight.get_high_precision_init_val()
    assert high_precision.device.type == "cpu"
    assert tuple(high_precision.shape) == (2, 64, 32)
    torch.testing.assert_close(
        high_precision,
        torch.full_like(high_precision, 0.25),
        rtol=0,
        atol=0,
    )

    quantizer = weight._get_quantizer()
    new_weight = quantizer.make_empty(
        shape=weight.shape,
        dtype=weight.dtype,
        device=weight.device,
    )
    quantizer.update_quantized(high_precision.to(weight.device), new_weight)
    torch.testing.assert_close(
        new_weight.dequantize(dtype=weight.dtype),
        weight.dequantize(dtype=weight.dtype),
        rtol=0,
        atol=0,
    )

    weight.clear_high_precision_init_val()
    assert weight.get_high_precision_init_val() is None
    assert not hasattr(weight, "_high_precision_init_val")

    inp = torch.randn(32, 2, 32, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    with te.autocast(enabled=True, recipe=recipe):
        output = module(inp)
    output.sum().backward()
    assert module.weight.grad is not None


@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
def test_quantized_weight_inference() -> None:
    """Inference primary weight needs only row-wise compact MXFP8 data."""
    recipe = make_recipe("mxfp8")
    weight = torch.rand(2, 96, 160, device="cuda", dtype=torch.bfloat16)
    inp = torch.rand(96, 2, 160, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad(), te.quantized_model_init(enabled=True, recipe=recipe):
        module = te.BatchedLinear(
            2,
            160,
            96,
            bias=False,
            params_dtype=torch.bfloat16,
            device="cuda",
        )
    module.requires_grad_(False)
    with torch.no_grad():
        module.weight.copy_(weight)
    reference = torch.einsum("...gd,grd->...gr", inp, module.weight.dequantize())

    assert module.weight._rowwise_data is not None
    assert module.weight._columnwise_data is None
    with torch.no_grad(), te.autocast(enabled=True, recipe=recipe):
        output = module(inp)
    assert_close(output, reference, **quantization_tols("mxfp8"))
    assert not module.weight._with_gemm_swizzled_scales


@pytest.mark.parametrize("batch_dim", (0, -2))
@pytest.mark.parametrize("bias", (False, True))
def test_return_bias(batch_dim: int, bias: bool) -> None:
    """return_bias leaves the bias unapplied and returns it separately."""
    module = te.BatchedLinear(
        2,
        32,
        32,
        batch_dim=batch_dim,
        bias=bias,
        return_bias=True,
        params_dtype=torch.bfloat16,
        device="cuda",
    )
    input_shape = (2, 32, 32) if batch_dim == 0 else (32, 2, 32)
    inp = torch.randn(input_shape, dtype=torch.bfloat16, device="cuda")
    output, returned_bias = module(inp)
    weight_ref = module.weight.detach()
    if batch_dim == 0:
        output_ref = torch.einsum("g...d,grd->g...r", inp, weight_ref)
    else:
        output_ref = torch.einsum("...gd,grd->...gr", inp, weight_ref)
    assert_close(output, output_ref, **dtype_tols(torch.bfloat16))
    if bias:
        assert returned_bias is module.bias
    else:
        assert returned_bias is None


@pytest.mark.parametrize(
    "recipe",
    (
        transformer_engine.common.recipe.DelayedScaling(),
        transformer_engine.common.recipe.Float8CurrentScaling(),
        transformer_engine.common.recipe.Float8BlockScaling(),
        transformer_engine.common.recipe.NVFP4BlockScaling(),
    ),
)
def test_rejects_non_mxfp8_recipe(recipe) -> None:
    """Only high precision and MXFP8 compute are accepted."""
    with pytest.raises(ValueError, match="only high-precision compute or the MXFP8 recipe"):
        with te.quantized_model_init(enabled=True, recipe=recipe):
            te.BatchedLinear(
                2,
                32,
                32,
                bias=False,
                params_dtype=torch.bfloat16,
                device="cuda",
            )


@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
@pytest.mark.parametrize("backward_override", ("high_precision", "dequantized"))
def test_rejects_backward_override(backward_override: str) -> None:
    """MXFP8 backward overrides are not implemented."""
    module = te.BatchedLinear(
        2,
        32,
        32,
        bias=False,
        params_dtype=torch.bfloat16,
        device="cuda",
    )
    inp = torch.randn(32, 2, 32, dtype=torch.bfloat16, device="cuda")
    recipe = transformer_engine.common.recipe.MXFP8BlockScaling(backward_override=backward_override)
    with pytest.raises(ValueError, match="does not support MXFP8 backward_override"):
        with te.autocast(enabled=True, recipe=recipe):
            module(inp)


def test_rejects_unsupported_layout() -> None:
    """Reject ambiguous batch dimensions and non-contiguous input storage."""
    with pytest.raises(ValueError, match="batch_dim=0 or batch_dim=-2"):
        te.BatchedLinear(2, 32, 32, batch_dim=1, device="cuda")

    module = te.BatchedLinear(2, 32, 32, batch_dim=-2, device="cuda")
    inp = torch.empty(32, 2, 64, device="cuda")[:, :, ::2]
    assert inp.shape == (32, 2, 32) and not inp.is_contiguous()
    with pytest.raises(ValueError, match="contiguous input"):
        module(inp)
