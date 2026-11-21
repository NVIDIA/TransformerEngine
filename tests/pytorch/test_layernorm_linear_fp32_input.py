# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FP32 residual inputs must not be rounded before normalization.

Run this file in separate processes with NVTE_NORM_FWD_USE_CUDNN and
NVTE_NORM_BWD_USE_CUDNN set to 0 and 1 to exercise both normalization backends.
These flags are cached by the native library, so changing os.environ within a
parametrized test would not reliably select a different backend.
"""

import importlib
import os
from contextlib import nullcontext

import pytest
import torch
import torch.nn.functional as F

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, MXFP8BlockScaling


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _check_dtype(dtype):
    if dtype == torch.bfloat16 and not te.is_bf16_available():
        pytest.skip("BF16 is not supported on this GPU")


def _make_module(normalization, dtype, zero_centered_gamma, return_norm, hidden_size=512):
    torch.manual_seed(1234)
    module = te.LayerNormLinear(
        hidden_size,
        64,
        eps=1e-5,
        normalization=normalization,
        params_dtype=dtype,
        zero_centered_gamma=zero_centered_gamma,
        return_layernorm_output=return_norm,
        device="cuda",
    )
    with torch.no_grad():
        module.layer_norm_weight.copy_(
            torch.empty_like(module.layer_norm_weight).uniform_(-0.25, 0.25)
            + (0.0 if zero_centered_gamma else 1.0)
        )
        if normalization == "LayerNorm":
            module.layer_norm_bias.uniform_(-0.125, 0.125)
        module.weight.normal_(0.0, hidden_size**-0.5)
        module.bias.uniform_(-0.125, 0.125)
    return module


def _reference(
    module,
    x,
    activation_dtype,
    normalization,
    zero_centered_gamma,
    gamma_in_weight_dtype=False,
):
    """Independent FP32 norm followed by an explicit low-precision GEMM."""
    params = {
        name: param.detach().clone().requires_grad_() for name, param in module.named_parameters()
    }
    # Match AMP's existing parameter casts, while retaining x in FP32. Parameter
    # gradients pass through these casts just as in LayerNormLinear.
    gamma = params["layer_norm_weight"].to(activation_dtype)
    if not gamma_in_weight_dtype:
        gamma = gamma.float()
    if zero_centered_gamma:
        gamma = gamma + 1.0
    gamma = gamma.float()
    if normalization == "LayerNorm":
        centered = x - x.mean(dim=-1, keepdim=True)
        normalized = centered * torch.rsqrt(
            centered.square().mean(dim=-1, keepdim=True) + module.eps
        )
        beta = params["layer_norm_bias"].to(activation_dtype).float()
        norm_out = (normalized * gamma + beta).to(activation_dtype)
    else:
        normalized = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + module.eps)
        norm_out = (normalized * gamma).to(activation_dtype)
    out = F.linear(
        norm_out,
        params["weight"].to(activation_dtype),
        params["bias"].to(activation_dtype),
    )
    return out, norm_out, params


def _compare_to_reference(
    module,
    x,
    activation_dtype,
    normalization,
    zero_centered_gamma,
    return_norm,
    autocast=False,
):
    x_ref = x.detach().clone().requires_grad_()
    original_input = x.detach().clone()
    context = torch.autocast("cuda", dtype=activation_dtype) if autocast else nullcontext()
    with context:
        result = module(x)
    if return_norm:
        out, norm_out = result
        assert norm_out.dtype == activation_dtype
    else:
        out, norm_out = result, None
    assert out.dtype == activation_dtype
    # The FP32 residual accumulator must also remain untouched by the fused op.
    torch.testing.assert_close(x, original_input, rtol=0, atol=0)

    out_ref, norm_ref, params_ref = _reference(
        module, x_ref, activation_dtype, normalization, zero_centered_gamma
    )
    grad_out = torch.randn_like(out)
    outputs, references, grad_outputs = [out], [out_ref], [grad_out]
    if return_norm:
        outputs.append(norm_out)
        references.append(norm_ref)
        # Exercise the extra backward path when the normalized activation is
        # consumed independently, rather than merely inspecting its dtype.
        grad_outputs.append(torch.randn_like(norm_out))
    torch.autograd.backward(outputs, grad_outputs)
    torch.autograd.backward(references, grad_outputs)

    if activation_dtype == torch.bfloat16:
        output_tolerance = dict(rtol=2e-2, atol=2e-2)
        dx_tolerance = dict(rtol=2e-2, atol=5e-3)
        param_tolerance = dict(rtol=3e-2, atol=5e-2)
    else:
        output_tolerance = dict(rtol=3e-3, atol=3e-3)
        dx_tolerance = dict(rtol=3e-3, atol=5e-4)
        param_tolerance = dict(rtol=3e-3, atol=1e-2)
    torch.testing.assert_close(out, out_ref, **output_tolerance)
    if return_norm:
        torch.testing.assert_close(norm_out, norm_ref, **output_tolerance)
    assert x.grad.dtype == torch.float32
    torch.testing.assert_close(x.grad, x_ref.grad, **dx_tolerance)
    for name, param in module.named_parameters():
        assert param.grad.dtype == param.dtype
        torch.testing.assert_close(
            param.grad,
            params_ref[name].grad,
            msg=lambda msg: f"{name}: {msg}",
            **param_tolerance,
        )


@pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("zero_centered_gamma", [False, True])
@pytest.mark.parametrize("return_norm", [False, True])
def test_fp32_input_with_low_precision_parameters(
    normalization, dtype, zero_centered_gamma, return_norm
):
    _check_dtype(dtype)
    module = _make_module(normalization, dtype, zero_centered_gamma, return_norm)
    x = torch.randn(17, 512, device="cuda", dtype=torch.float32, requires_grad=True)
    _compare_to_reference(module, x, dtype, normalization, zero_centered_gamma, return_norm)


@pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
@pytest.mark.parametrize("zero_centered_gamma", [False, True])
@pytest.mark.parametrize("return_norm", [False, True])
def test_fp32_input_and_parameters_under_autocast(normalization, zero_centered_gamma, return_norm):
    _check_dtype(torch.bfloat16)
    module = _make_module(normalization, torch.float32, zero_centered_gamma, return_norm)
    x = torch.randn(17, 512, device="cuda", dtype=torch.float32, requires_grad=True)
    _compare_to_reference(
        module,
        x,
        torch.bfloat16,
        normalization,
        zero_centered_gamma,
        return_norm,
        autocast=True,
    )


@pytest.mark.parametrize("hidden_size", [1792, 10240])
def test_fp32_residual_model_hidden_sizes(hidden_size):
    """Cover non-power-of-two and wide RMSNorm kernel dispatch sizes."""
    _check_dtype(torch.bfloat16)
    module = _make_module("RMSNorm", torch.bfloat16, False, True, hidden_size)
    x = torch.randn(13, hidden_size, device="cuda", dtype=torch.float32, requires_grad=True)
    _compare_to_reference(module, x, torch.bfloat16, "RMSNorm", False, True)


@pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sub_ulp_input_survives_normalization(normalization, dtype):
    """A dtype-only test would miss pre-rounding x and widening dx afterward."""
    _check_dtype(dtype)
    module = _make_module(normalization, dtype, False, True)
    with torch.no_grad():
        module.layer_norm_weight.fill_(1)
        if normalization == "LayerNorm":
            module.layer_norm_bias.zero_()

    # All coordinates collapse to one when cast to the parameter dtype, but
    # FP32 normalization must preserve their differences. Stay below the
    # smaller ULP immediately below 1, including for FP16.
    amplitude = 0.2 * torch.finfo(dtype).eps
    offsets = torch.linspace(-amplitude, amplitude, 512, device="cuda")
    x = (1 + offsets).repeat(17, 1).requires_grad_()
    assert torch.equal(x.to(dtype), torch.ones_like(x, dtype=dtype))
    x_ref = x.detach().clone().requires_grad_()
    x_rounded = x.detach().to(dtype).float().requires_grad_()
    out, norm_out = module(x)
    _, norm_ref, _ = _reference(module, x_ref, dtype, normalization, False)
    _, norm_rounded, _ = _reference(module, x_rounded, dtype, normalization, False)

    # With RMSNorm, rounding can leave the returned values unchanged while
    # erasing the coordinate-dependent dx. LN also exposes a forward error.
    grad_norm = (
        torch.linspace(-1, 1, 512, device="cuda", dtype=dtype).expand_as(norm_out)
        if normalization == "LayerNorm"
        else torch.ones_like(norm_out)
    )
    torch.autograd.backward([out, norm_out], [torch.zeros_like(out), grad_norm])
    norm_ref.backward(grad_norm)
    norm_rounded.backward(grad_norm)
    assert x.grad.dtype == torch.float32
    torch.testing.assert_close(norm_out, norm_ref, rtol=3e-3, atol=3e-4)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=3e-3, atol=2e-6)
    # Prove that this case rejects the old FP32 -> low precision -> FP32 path,
    # even if it were changed to advertise an FP32 input-gradient dtype.
    rounding_error = (x_rounded.grad - x_ref.grad).abs().max()
    actual_error = (x.grad - x_ref.grad).abs().max()
    assert rounding_error > 1e-5
    assert actual_error < rounding_error * 0.1


@pytest.mark.skipif(
    any(
        os.environ.get(name) != "1"
        for name in (
            "NVTE_NORM_FWD_USE_CUDNN",
            "NVTE_NORM_BWD_USE_CUDNN",
            "NVTE_ZERO_CENTERED_GAMMA_IN_WTYPE",
        )
    ),
    reason="Run separately with cuDNN normalization and weight-dtype gamma addition enabled",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fp32_input_zero_centered_gamma_in_weight_dtype(dtype):
    """Retain cuDNN's opt-in low-precision gamma + 1 arithmetic."""
    _check_dtype(dtype)
    module = _make_module("LayerNorm", dtype, True, True)
    with torch.no_grad():
        module.layer_norm_weight.uniform_(0.01, 0.5)
        module.layer_norm_bias.zero_()
    x = torch.randn(17, 512, device="cuda", dtype=torch.float32, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    x_full_precision_gamma = x.detach().clone().requires_grad_()
    out, norm_out = module(x)
    _, norm_ref, _ = _reference(module, x_ref, dtype, "LayerNorm", True, gamma_in_weight_dtype=True)
    _, norm_full_precision_gamma, _ = _reference(
        module, x_full_precision_gamma, dtype, "LayerNorm", True
    )
    grad_norm = torch.randn_like(norm_out)
    torch.autograd.backward([out, norm_out], [torch.zeros_like(out), grad_norm])
    norm_ref.backward(grad_norm)
    norm_full_precision_gamma.backward(grad_norm)

    assert out.dtype == norm_out.dtype == dtype
    assert x.grad.dtype == torch.float32
    torch.testing.assert_close(norm_out, norm_ref, rtol=torch.finfo(dtype).eps, atol=2e-5)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-4, atol=1e-5)
    # A broad low-precision tolerance could hide a switch to FP32 gamma + 1.
    # Require the requested arithmetic to explain the output and dx better.
    for actual, reference, alternative in (
        (norm_out, norm_ref, norm_full_precision_gamma),
        (x.grad, x_ref.grad, x_full_precision_gamma.grad),
    ):
        reference_error = torch.linalg.vector_norm(actual.float() - reference.float())
        alternative_error = torch.linalg.vector_norm(actual.float() - alternative.float())
        assert alternative_error > 0
        assert reference_error < alternative_error * 0.1


@pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
@pytest.mark.parametrize("return_norm", [False, True])
@pytest.mark.parametrize("recipe_name", ["delayed", "mxfp8"])
def test_fp32_input_fp8(normalization, return_norm, recipe_name, monkeypatch):
    if recipe_name == "mxfp8":
        available, reason = te.is_mxfp8_available(return_reason=True)
        recipe = MXFP8BlockScaling()
    else:
        available, reason = te.is_fp8_available(return_reason=True)
        recipe = DelayedScaling()
    if not available:
        pytest.skip(reason)
    torch.manual_seed(1234)
    module = te.LayerNormLinear(
        512,
        128,
        normalization=normalization,
        params_dtype=torch.bfloat16,
        return_layernorm_output=return_norm,
        device="cuda",
    )
    x = torch.randn(128, 512, device="cuda", dtype=torch.float32, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    if normalization == "LayerNorm":
        # Direct cuDNN LayerNorm statistics lose small FP32 variations. The
        # module must normalize first and then apply the FP8 quantizer.
        implementation = importlib.import_module(te.LayerNormLinear.__module__)
        original = implementation.apply_normalization

        def check_unquantized_norm(*args, **kwargs):
            assert args[0].dtype == torch.float32
            assert args[5] is None
            return original(*args, **kwargs)

        monkeypatch.setattr(implementation, "apply_normalization", check_unquantized_norm)
    with te.autocast(enabled=True, recipe=recipe):
        result = module(x)
    out, norm_out = result if return_norm else (result, None)
    out_ref, norm_ref, _ = _reference(module, x_ref, torch.bfloat16, normalization, False)
    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    out_ref.backward(grad_out)
    assert out.dtype == torch.bfloat16
    assert x.grad.dtype == torch.float32
    assert module.layer_norm_weight.grad.dtype == torch.bfloat16
    for value, reference in [(out, out_ref), (x.grad, x_ref.grad)]:
        assert torch.isfinite(value).all()
        # This comparison includes FP8 quantization of activations, weights,
        # and output gradients; the reference uses an unquantized BF16 GEMM.
        relative_rms_error = torch.linalg.vector_norm(value.float() - reference.float())
        relative_rms_error /= torch.linalg.vector_norm(reference.float())
        assert relative_rms_error < 0.15
    if return_norm:
        assert norm_out.dtype == torch.bfloat16
        torch.testing.assert_close(norm_out, norm_ref, rtol=2e-2, atol=2e-2)
