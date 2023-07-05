# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Test TE Paddle Layer-level APIs"""

import pytest
from utils import assert_allclose

import paddle
import paddle.nn.functional as F

import transformer_engine.paddle as te

LINEAR_CASES = [(256, 256, 512), (32, 32, 32), (16384, 1024, 2816), (16384, 2816, 1024),
                (16384, 1024, 1024)]
NORM_CASES = [(16, 32), (256, 1024)]
MLP_CASES = [(32, 32, 32), (64, 256, 512)]


class TestLinear:
    """
    Tests for Linear layer
    """

    @staticmethod
    def calc_ref(x, weight, bias, dy):
        """
        Calculate reference using paddle linear op
        """
        # Create a copy of input tensor to enable grad calculation
        x = paddle.to_tensor(x)
        x.stop_gradient = False
        weight = paddle.to_tensor(weight)
        weight.stop_gradient = False
        bias = paddle.to_tensor(bias)
        bias.stop_gradient = False

        y = F.linear(x, weight, bias)
        y.backward(dy)

        return y, x.grad, weight.grad.T, bias.grad

    @pytest.mark.skipif(paddle.device.cuda.get_device_capability() < (8, 0),
                        reason="BF16 Linear requires Ampere+ GPU")
    @pytest.mark.parametrize('bs,in_features,out_features', LINEAR_CASES)
    def test_bf16(self, bs, in_features, out_features):
        """
        Test BF16 Linear
        """
        rtol = 1e-2
        atol = 1e-2

        input_tensor = paddle.rand(shape=(bs, in_features), dtype='bfloat16')
        input_tensor.stop_gradient = False
        grad_out = paddle.rand(shape=(bs, out_features), dtype='bfloat16')

        paddle.set_default_dtype("bfloat16")
        layer = te.Linear(
            in_features=in_features,
            out_features=out_features,
        )

        (
            out_ref,
            grad_input_ref,
            grad_weight_ref,
            grad_bias_ref,
        ) = self.calc_ref(input_tensor.clone()._to(dtype=paddle.float32),
                          layer.weight.T.clone()._to(dtype=paddle.float32),
                          layer.bias.clone()._to(dtype=paddle.float32),
                          grad_out.clone()._to(dtype=paddle.float32))

        out = layer(input_tensor)
        out.backward(grad_out)

        assert_allclose(out, out_ref, rtol=rtol, atol=atol)
        assert_allclose(input_tensor.grad, grad_input_ref, rtol=rtol, atol=atol)
        assert_allclose(layer.weight.grad, grad_weight_ref, rtol=rtol, atol=atol)
        assert_allclose(layer.bias.grad, grad_bias_ref, rtol=rtol, atol=atol)


class TestLayerNorm:
    """
    Tests for LayerNorm layer
    """

    @staticmethod
    def calc_ref(x, eps, gamma, beta, dy):
        """
        Calculate reference using paddle layer_norm op
        """
        x = paddle.to_tensor(x)
        x.stop_gradient = False
        gamma = paddle.to_tensor(gamma)
        gamma.stop_gradient = False
        beta = paddle.to_tensor(beta)
        beta.stop_gradient = False

        y = paddle.nn.functional.layer_norm(x=x,
                                            normalized_shape=x.shape[1:],
                                            weight=gamma,
                                            bias=beta,
                                            epsilon=eps)

        y.backward(dy)

        return y, x.grad, gamma.grad, beta.grad

    @pytest.mark.parametrize('bs,hidden_size', NORM_CASES)
    def test_bf16(self, bs, hidden_size):
        """
        Test BF16 LayerNorm
        """
        eps = 1e-3
        rtol = 1e-2
        atol = 1e-2

        x = paddle.uniform(shape=(bs, hidden_size), dtype='bfloat16')
        x.stop_gradient = False
        grad_out = paddle.rand(shape=(bs, hidden_size), dtype='bfloat16')

        paddle.set_default_dtype("bfloat16")
        layer = te.LayerNorm(hidden_size=hidden_size, eps=eps)

        (
            out_ref,
            grad_x_ref,
            grad_weight_ref,
            grad_bias_ref,
        ) = self.calc_ref(x, eps, layer.weight, layer.bias, grad_out)

        out = layer(x)
        out.backward(grad_out)

        assert_allclose(out, out_ref, rtol=rtol, atol=atol)
        assert_allclose(x.grad, grad_x_ref, rtol=rtol, atol=atol)
        assert_allclose(layer.weight.grad, grad_weight_ref, rtol=rtol, atol=atol)
        assert_allclose(layer.bias.grad, grad_bias_ref, rtol=rtol, atol=atol)


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

    input_tensor = paddle.rand(shape=(bs, in_features), dtype='bfloat16')
    input_tensor.stop_gradient = False
    grad_out = paddle.rand(shape=(bs, out_features), dtype='bfloat16')
    eps = 1e-3

    layernorm_linear = te.LayerNormLinear(
        in_features=in_features,
        out_features=out_features,
        eps=eps,
    )

    linear = te.Linear(in_features=in_features, out_features=out_features)
    linear.weight.copy_(layernorm_linear.weight, True)
    linear.bias.copy_(layernorm_linear.bias, True)

    layernorm = te.LayerNorm(hidden_size=in_features, eps=eps)
    layernorm.weight.copy_(layernorm_linear.ln_weight, True)
    layernorm.bias.copy_(layernorm_linear.ln_bias, True)

    # Calculate ref
    input_tensor_ref = paddle.to_tensor(input_tensor)
    input_tensor_ref.stop_gradient = False

    y_ref = linear(layernorm(input_tensor_ref))
    y_ref.backward(grad_out)

    # Calculate actual
    y = layernorm_linear(input_tensor)
    y.backward(grad_out)

    assert_allclose(y, y_ref, rtol=rtol, atol=atol)
    assert_allclose(input_tensor.grad, input_tensor_ref.grad, rtol=rtol, atol=atol)
    assert_allclose(layernorm_linear.weight.grad, linear.weight.grad, rtol=rtol, atol=atol)
    assert_allclose(layernorm_linear.bias.grad, linear.bias.grad, rtol=rtol, atol=atol)
    assert_allclose(layernorm_linear.ln_weight.grad, layernorm.weight.grad, rtol=rtol, atol=atol)
    assert_allclose(layernorm_linear.ln_bias.grad, layernorm.bias.grad, rtol=rtol, atol=atol)


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

    input_tensor = paddle.rand(shape=(bs, hidden_size), dtype='bfloat16')
    input_tensor.stop_gradient = False
    grad_out = paddle.rand(shape=(bs, hidden_size), dtype='bfloat16')
    eps = 1e-3

    layernorm_mlp = te.LayerNormMLP(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        eps=eps,
    )

    layernorm = te.LayerNorm(hidden_size=hidden_size, eps=eps)
    layernorm.weight.copy_(layernorm_mlp.ln_weight, True)
    layernorm.bias.copy_(layernorm_mlp.ln_bias, True)

    fc1 = te.Linear(in_features=hidden_size, out_features=ffn_hidden_size)
    fc1.weight.copy_(layernorm_mlp.fc1_weight, True)
    fc1.bias.copy_(layernorm_mlp.fc1_bias, True)

    fc2 = te.Linear(in_features=ffn_hidden_size, out_features=hidden_size)
    fc2.weight.copy_(layernorm_mlp.fc2_weight, True)
    fc2.bias.copy_(layernorm_mlp.fc2_bias, True)

    act = paddle.nn.GELU()

    # Calculate ref
    input_tensor_ref = paddle.to_tensor(input_tensor)
    input_tensor_ref.stop_gradient = False

    y_ref = fc2(act(fc1(layernorm(input_tensor_ref))))
    y_ref.backward(grad_out)

    # Calculate actual
    y = layernorm_mlp(input_tensor)
    y.backward(grad_out)

    assert_allclose(y, y_ref, rtol=5e-2, atol=5e-2)
    assert_allclose(input_tensor.grad, input_tensor_ref.grad, rtol=rtol, atol=atol)
    assert_allclose(layernorm_mlp.fc1_weight.grad, fc1.weight.grad, rtol=rtol, atol=atol)
    assert_allclose(layernorm_mlp.fc1_bias.grad, fc1.bias.grad, rtol=rtol, atol=atol)
    assert_allclose(layernorm_mlp.fc2_weight.grad, fc2.weight.grad, rtol=rtol, atol=atol)
    assert_allclose(layernorm_mlp.fc2_bias.grad, fc2.bias.grad, rtol=rtol, atol=atol)
    assert_allclose(layernorm_mlp.ln_weight.grad, layernorm.weight.grad, rtol=rtol, atol=atol)
    assert_allclose(layernorm_mlp.ln_bias.grad, layernorm.bias.grad, rtol=rtol, atol=atol)
