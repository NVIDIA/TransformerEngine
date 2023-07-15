# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Test TE Paddle Layer-level APIs"""

import pytest
from utils import assert_allclose

import paddle

import transformer_engine.paddle as te

LINEAR_CASES = [(16, 16, 32), (32, 32, 64), (64, 128, 256)]
NORM_CASES = [(16, 32), (256, 1024)]
MLP_CASES = [(32, 32, 32), (64, 256, 512)]


def calc_output_and_grad(layer, x, dy):
    """
    Calculate forward and backward pass
    """
    inp = paddle.to_tensor(x)
    inp.stop_gradient = x.stop_gradient
    y = layer(inp)
    y.backward(dy)

    return y, inp.grad if not inp.stop_gradient else None


@pytest.mark.skipif(paddle.device.cuda.get_device_capability() < (8, 0),
                    reason="BF16 Linear requires Ampere+ GPU")
@pytest.mark.parametrize('bs,in_features,out_features', LINEAR_CASES)
def test_linear_bf16(bs, in_features, out_features):
    """
    Test BF16 Linear
    """
    rtol = 1e-2
    atol = 1e-2

    input_tensor = paddle.uniform(shape=(bs, in_features), dtype='bfloat16')
    input_tensor.stop_gradient = False
    grad_out = paddle.uniform(shape=(bs, out_features), dtype='bfloat16')

    paddle.set_default_dtype("bfloat16")
    layer_te = te.Linear(in_features, out_features)
    layer_pd = te.Linear(in_features, out_features, backend='paddle')
    layer_pd.weight.copy_(layer_te.weight.T, True)
    layer_pd.bias.copy_(layer_te.bias, True)

    out_ref, grad_input_ref = calc_output_and_grad(layer_pd, input_tensor, grad_out)
    out, grad_input = calc_output_and_grad(layer_te, input_tensor, grad_out)

    assert_allclose(out, out_ref, rtol=rtol, atol=atol)
    assert_allclose(grad_input, grad_input_ref, rtol=rtol, atol=atol)
    assert_allclose(layer_te.weight.grad, layer_pd.weight.grad.T, rtol=rtol, atol=atol)
    assert_allclose(layer_te.bias.grad, layer_pd.bias.grad, rtol=rtol, atol=atol)


@pytest.mark.parametrize('bs,hidden_size', NORM_CASES)
def test_layernorm_bf16(bs, hidden_size):
    """
    Test BF16 LayerNorm
    """
    eps = 1e-3
    rtol = 1e-2
    atol = 1e-2

    x = paddle.uniform(shape=(bs, hidden_size), dtype='bfloat16')
    x.stop_gradient = False
    grad_out = paddle.uniform(shape=(bs, hidden_size), dtype='bfloat16')

    paddle.set_default_dtype("bfloat16")
    layer_te = te.LayerNorm(hidden_size=hidden_size, eps=eps)
    layer_pd = te.LayerNorm(hidden_size=hidden_size, eps=eps, backend='paddle')
    layer_pd.weight.copy_(layer_te.weight, True)
    layer_pd.bias.copy_(layer_te.bias, True)

    out_ref, grad_input_ref = calc_output_and_grad(layer_pd, x, grad_out)
    out, grad_input = calc_output_and_grad(layer_te, x, grad_out)

    assert_allclose(out, out_ref, rtol=rtol, atol=atol)
    assert_allclose(grad_input, grad_input_ref, rtol=rtol, atol=atol)
    assert_allclose(layer_te.weight.grad, layer_pd.weight.grad, rtol=rtol, atol=atol)
    assert_allclose(layer_te.bias.grad, layer_pd.bias.grad, rtol=rtol, atol=atol)


@pytest.mark.skipif(paddle.device.cuda.get_device_capability() < (8, 0),
                    reason="BF16 Linear requires Ampere+ GPU")
@pytest.mark.parametrize('bs,in_features,out_features', LINEAR_CASES)
def test_layernorm_linear_bf16(bs, in_features, out_features):
    """
    Test BF16 LayerNormLinear Layer
    """
    paddle.set_default_dtype("bfloat16")
    rtol = 1e-2
    atol = 1e-2

    input_tensor = paddle.uniform(shape=(bs, in_features), dtype='bfloat16')
    input_tensor.stop_gradient = False
    grad_out = paddle.uniform(shape=(bs, out_features), dtype='bfloat16')
    eps = 1e-3

    layer_te = te.LayerNormLinear(
        in_features=in_features,
        out_features=out_features,
        eps=eps,
    )

    layer_pd = te.LayerNormLinear(in_features=in_features,
                                  out_features=out_features,
                                  eps=eps,
                                  backend='paddle')
    layer_pd.ln_weight.copy_(layer_te.ln_weight, True)
    layer_pd.ln_bias.copy_(layer_te.ln_bias, True)
    layer_pd.weight.copy_(layer_te.weight.T, True)
    layer_pd.bias.copy_(layer_te.bias, True)

    out_ref, grad_input_ref = calc_output_and_grad(layer_pd, input_tensor, grad_out)
    out, grad_input = calc_output_and_grad(layer_te, input_tensor, grad_out)

    assert_allclose(out, out_ref, rtol=rtol, atol=atol)
    assert_allclose(grad_input, grad_input_ref, rtol=rtol, atol=atol)
    assert_allclose(layer_te.weight.grad, layer_pd.weight.grad.T, rtol=rtol, atol=atol)
    assert_allclose(layer_te.bias.grad, layer_pd.bias.grad, rtol=rtol, atol=atol)
    assert_allclose(layer_te.ln_weight.grad, layer_pd.ln_weight.grad, rtol=rtol, atol=atol)
    assert_allclose(layer_te.ln_bias.grad, layer_pd.ln_bias.grad, rtol=rtol, atol=atol)


@pytest.mark.skipif(paddle.device.cuda.get_device_capability() < (8, 0),
                    reason="BF16 Linear requires Ampere+ GPU")
@pytest.mark.parametrize('bs,hidden_size,ffn_hidden_size', MLP_CASES)
def test_layernorm_mlp_bf16(bs, hidden_size, ffn_hidden_size):
    """
    Test BF16 LayerNormMLP Layer
    """
    paddle.set_default_dtype("bfloat16")
    rtol = 5e-2
    atol = 5e-2

    input_tensor = paddle.uniform(shape=(bs, hidden_size), dtype='bfloat16')
    input_tensor.stop_gradient = False
    grad_out = paddle.uniform(shape=(bs, hidden_size), dtype='bfloat16')
    eps = 1e-3

    layer_te = te.LayerNormMLP(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        eps=eps,
    )
    layer_pd = te.LayerNormMLP(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        eps=eps,
        backend='paddle',
    )
    layer_pd.ln_weight.copy_(layer_te.ln_weight, True)
    layer_pd.ln_bias.copy_(layer_te.ln_bias, True)
    layer_pd.fc1_weight.copy_(layer_te.fc1_weight.T, True)
    layer_pd.fc1_bias.copy_(layer_te.fc1_bias, True)
    layer_pd.fc2_weight.copy_(layer_te.fc2_weight.T, True)
    layer_pd.fc2_bias.copy_(layer_te.fc2_bias, True)

    out_ref, grad_input_ref = calc_output_and_grad(layer_pd, input_tensor, grad_out)
    out, grad_input = calc_output_and_grad(layer_te, input_tensor, grad_out)

    assert_allclose(out, out_ref, rtol=rtol, atol=atol)
    assert_allclose(grad_input, grad_input_ref, rtol=rtol, atol=atol)
    assert_allclose(layer_te.fc1_weight.grad, layer_pd.fc1_weight.grad.T, rtol=rtol, atol=atol)
    assert_allclose(layer_te.fc1_bias.grad, layer_pd.fc1_bias.grad, rtol=rtol, atol=atol)
    assert_allclose(layer_te.fc2_weight.grad, layer_pd.fc2_weight.grad.T, rtol=rtol, atol=atol)
    assert_allclose(layer_te.fc2_bias.grad, layer_pd.fc2_bias.grad, rtol=rtol, atol=atol)
    assert_allclose(layer_te.ln_weight.grad, layer_pd.ln_weight.grad, rtol=rtol, atol=atol)
    assert_allclose(layer_te.ln_bias.grad, layer_pd.ln_bias.grad, rtol=rtol, atol=atol)
