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

        assert_allclose(out, out_ref, rtol=1e-2, atol=1e-2)
        assert_allclose(input_tensor.grad, grad_input_ref, rtol=1e-2, atol=1e-2)
        assert_allclose(layer.weight.grad, grad_weight_ref, rtol=1e-2, atol=1e-2)
        assert_allclose(layer.bias.grad, grad_bias_ref, rtol=1e-2, atol=1e-2)


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

        assert_allclose(out, out_ref, rtol=1e-2, atol=1e-2)
        assert_allclose(x.grad, grad_x_ref, rtol=1e-2, atol=1e-2)
        assert_allclose(layer.weight.grad, grad_weight_ref, rtol=1e-2, atol=1e-2)
        assert_allclose(layer.bias.grad, grad_bias_ref, rtol=1e-2, atol=1e-2)
