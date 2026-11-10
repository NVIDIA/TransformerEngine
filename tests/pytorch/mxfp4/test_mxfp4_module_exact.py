# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.common.recipe import CustomRecipe
from transformer_engine.pytorch.custom_recipes.quantizer_factory_zoo import (
    mxfp4_weight_mxfp8_activation_factory,
)
from transformer_engine.pytorch.tensor.mxfp4_tensor import MXFP4Quantizer
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
pytestmark = pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)

RECIPE = CustomRecipe(qfactory=mxfp4_weight_mxfp8_activation_factory)


def _dequantized(quantizer, x):
    return quantizer(x).dequantize(dtype=torch.float32)


def _mxfp8(x, columnwise):
    return _dequantized(
        MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=not columnwise, columnwise=columnwise), x
    )


def _mxfp4(w, columnwise):
    return _dequantized(MXFP4Quantizer(rowwise=not columnwise, columnwise=columnwise), w)


def _reference_linear(x, w, b, dy):
    """W4A8 forward and backward computed from independently quantized operands."""
    y = _mxfp8(x, False) @ _mxfp4(w, False).t()
    if b is not None:
        y = y + b.float()
    dx = _mxfp8(dy, False) @ _mxfp4(w, True)
    dw = _mxfp8(dy, True).t() @ _mxfp8(x, True)
    db = dy.float().sum(dim=0) if b is not None else None
    return y, dx, dw, db


def _assert_close_bf16(actual, expected):
    torch.testing.assert_close(actual, expected.to(actual.dtype), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("tokens,in_features,out_features", [(128, 128, 256), (512, 1024, 384)])
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_linear_mxfp4_weight_mxfp8_activation(tokens, in_features, out_features, use_bias, dtype):
    torch.manual_seed(0)
    layer = te.Linear(in_features, out_features, bias=use_bias, params_dtype=dtype, device="cuda")
    x = torch.randn((tokens, in_features), dtype=dtype, device="cuda", requires_grad=True)
    dy = torch.randn((tokens, out_features), dtype=dtype, device="cuda")

    with te.autocast(enabled=True, recipe=RECIPE):
        y = layer(x)
    y.backward(dy)

    y_ref, dx_ref, dw_ref, db_ref = _reference_linear(
        x.detach(), layer.weight.detach(), layer.bias.detach() if use_bias else None, dy
    )
    _assert_close_bf16(y, y_ref)
    _assert_close_bf16(x.grad, dx_ref)
    _assert_close_bf16(layer.weight.grad, dw_ref)
    if use_bias:
        _assert_close_bf16(layer.bias.grad, db_ref)


def test_linear_mxfp4_cached_weight_survives_param_overwrite():
    """With ``is_first_microbatch``, later microbatches only read the cached MXFP4 weight."""
    torch.manual_seed(1)
    in_features, out_features, tokens = 256, 256, 128
    layer = te.Linear(in_features, out_features, params_dtype=torch.bfloat16, device="cuda")
    x = torch.randn((tokens, in_features), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    dy = torch.randn((tokens, out_features), dtype=torch.bfloat16, device="cuda")
    w = layer.weight.detach().clone()
    _, dx_ref, _, _ = _reference_linear(x.detach(), w, None, dy)

    with te.autocast(enabled=True, recipe=RECIPE):
        y_first = layer(x, is_first_microbatch=True)
    y_first.backward(dy)

    layer.weight.data.zero_()
    x.grad = None
    with te.autocast(enabled=True, recipe=RECIPE):
        y_cached = layer(x, is_first_microbatch=False)
    y_cached.backward(dy)

    torch.testing.assert_close(y_cached, y_first, rtol=0, atol=0)
    _assert_close_bf16(x.grad, dx_ref)


@pytest.mark.parametrize("m_splits", [[128, 0, 256, 64], [32, 32, 32, 32]])
def test_grouped_linear_mxfp4_weight_mxfp8_activation(m_splits):
    torch.manual_seed(2)
    num_gemms = len(m_splits)
    in_features, out_features = 256, 384
    layer = te.GroupedLinear(
        num_gemms, in_features, out_features, bias=True, params_dtype=torch.bfloat16, device="cuda"
    )
    tokens = sum(m_splits)
    x = torch.randn((tokens, in_features), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    dy = torch.randn((tokens, out_features), dtype=torch.bfloat16, device="cuda")

    with te.autocast(enabled=True, recipe=RECIPE):
        y = layer(x, m_splits)
    y.backward(dy)

    x_chunks = torch.split(x.detach(), m_splits)
    dy_chunks = torch.split(dy, m_splits)
    for i in range(num_gemms):
        if m_splits[i] == 0:
            continue
        w = getattr(layer, f"weight{i}")
        b = getattr(layer, f"bias{i}")
        y_ref, dx_ref, dw_ref, db_ref = _reference_linear(
            x_chunks[i], w.detach(), b.detach(), dy_chunks[i]
        )
        _assert_close_bf16(torch.split(y, m_splits)[i], y_ref)
        _assert_close_bf16(torch.split(x.grad, m_splits)[i], dx_ref)
        _assert_close_bf16(w.grad, dw_ref)
        _assert_close_bf16(b.grad, db_ref)


def test_layernorm_linear_mxfp4_weight_mxfp8_activation():
    torch.manual_seed(3)
    in_features, out_features, tokens = 256, 256, 128
    layer = te.LayerNormLinear(
        in_features, out_features, params_dtype=torch.bfloat16, device="cuda"
    )
    x = torch.randn((tokens, in_features), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    with te.autocast(enabled=True, recipe=RECIPE):
        y = layer(x)
    y.float().square().sum().backward()
    assert torch.isfinite(x.grad).all()

    # The fused kernel quantizes the FP32 normalized values, so a few elements
    # may round differently than the BF16 reference input.
    ln_out = torch.nn.functional.layer_norm(x.detach().float(), (in_features,), eps=layer.eps)
    y_ref = _mxfp8(ln_out.to(torch.bfloat16), False) @ _mxfp4(layer.weight.detach(), False).t()
    y_ref = y_ref + layer.bias.detach().float()
    rel_err = (y.float() - y_ref).norm() / y_ref.norm()
    assert rel_err < 1e-2


@pytest.mark.parametrize("backward_override", ["dequantized", "high_precision"])
def test_linear_mxfp4_backward_override(backward_override):
    torch.manual_seed(4)
    in_features, out_features, tokens = 256, 384, 128
    layer = te.Linear(in_features, out_features, params_dtype=torch.bfloat16, device="cuda")
    x = torch.randn((tokens, in_features), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    dy = torch.randn((tokens, out_features), dtype=torch.bfloat16, device="cuda")
    recipe = CustomRecipe(
        qfactory=mxfp4_weight_mxfp8_activation_factory, backward_override=backward_override
    )

    with te.autocast(enabled=True, recipe=recipe):
        y = layer(x)
    y.backward(dy)

    w = layer.weight.detach()
    if backward_override == "dequantized":
        w_bwd, x_bwd = _mxfp4(w, False), _mxfp8(x.detach(), False)
    else:
        w_bwd, x_bwd = w.float(), x.detach().float()
    y_ref, _, _, _ = _reference_linear(x.detach(), w, None, dy)
    _assert_close_bf16(y, y_ref)
    _assert_close_bf16(x.grad, dy.float() @ w_bwd)
    _assert_close_bf16(layer.weight.grad, dy.float().t() @ x_bwd)


def _make_layer(module_type, in_features, out_features, num_gemms):
    if module_type == "linear":
        return te.Linear(in_features, out_features, params_dtype=torch.bfloat16, device="cuda")
    if module_type == "layernorm_linear":
        return te.LayerNormLinear(
            in_features, out_features, params_dtype=torch.bfloat16, device="cuda"
        )
    return te.GroupedLinear(
        num_gemms, in_features, out_features, params_dtype=torch.bfloat16, device="cuda"
    )


def _gemm_weights(layer, module_type, num_gemms):
    if module_type == "grouped_linear":
        return [getattr(layer, f"weight{i}") for i in range(num_gemms)]
    return [layer.weight]


@pytest.mark.parametrize("module_type", ["linear", "layernorm_linear", "grouped_linear"])
@pytest.mark.parametrize("backward_override", [None, "dequantized"])
def test_backward_does_not_read_bf16_weights(module_type, backward_override):
    """The BF16 GEMM weights may be offloaded between forward and backward: poisoning
    them after forward must leave every output and gradient bitwise unchanged."""
    torch.manual_seed(5)
    in_features, out_features, tokens = 256, 384, 256
    m_splits = [128, 0, 64, 64]
    layer = _make_layer(module_type, in_features, out_features, len(m_splits))
    x = torch.randn((tokens, in_features), dtype=torch.bfloat16, device="cuda")
    dy = torch.randn((tokens, out_features), dtype=torch.bfloat16, device="cuda")
    recipe = CustomRecipe(
        qfactory=mxfp4_weight_mxfp8_activation_factory, backward_override=backward_override
    )
    weights = _gemm_weights(layer, module_type, len(m_splits))
    saved_weights = [w.detach().clone() for w in weights]

    def run(poison):
        for w, saved in zip(weights, saved_weights):
            w.data.copy_(saved)
        layer.zero_grad(set_to_none=True)
        inp = x.clone().requires_grad_(True)
        with te.autocast(enabled=True, recipe=recipe):
            y = layer(inp, m_splits) if module_type == "grouped_linear" else layer(inp)
        if poison:
            for w in weights:
                w.data.fill_(float("nan"))
        y.backward(dy)
        grads = [inp.grad] + [p.grad for p in layer.parameters()]
        return [y.detach().clone()] + [g.detach().clone() for g in grads if g is not None]

    expected = run(poison=False)
    actual = run(poison=True)
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=0, atol=0)


@pytest.mark.parametrize("use_bias", [False, True])
def test_linear_mxfp4_zero_tokens(use_bias):
    in_features, out_features = 256, 384
    layer = te.Linear(
        in_features, out_features, bias=use_bias, params_dtype=torch.bfloat16, device="cuda"
    )
    x = torch.randn((0, in_features), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    with te.autocast(enabled=True, recipe=RECIPE):
        y = layer(x)
    y.sum().backward()
    assert y.shape == (0, out_features)
    assert x.grad.shape == x.shape
    assert torch.count_nonzero(layer.weight.grad) == 0
