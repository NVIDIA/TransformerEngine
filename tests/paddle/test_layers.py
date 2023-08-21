# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Test TE Paddle Layer-level APIs"""

import math
import os
import pytest
from utils import assert_allclose

import paddle

import transformer_engine.paddle as te
from transformer_engine.paddle.fp8 import is_fp8_available, fp8_autocast
from transformer_engine.common.recipe import DelayedScaling

paddle.seed(10)
is_fp8_supported, reason = is_fp8_available()
LINEAR_CASES = [(16, 16, 32), (32, 32, 64)]
NORM_CASES = [(16, 32), (256, 1024)]


@pytest.mark.skipif(not is_fp8_supported, reason=reason)
@pytest.mark.parametrize('use_fp8', [True, False])
def test_checkpoint(use_fp8):
    """Test checkpoint save / load"""
    bs = 16
    in_features = 16
    out_features = 32
    file_name = "model.pdparams"
    input_tensor = paddle.uniform(shape=(bs, in_features), dtype='float32')
    model = te.Linear(in_features, out_features)
    model_loaded = te.Linear(in_features, out_features)
    # Populate amax_history
    with fp8_autocast(enabled=False, calibrating=True):
        _ = model(input_tensor)
    # Save model
    paddle.save(model.state_dict(), file_name)
    # Get ref output
    with fp8_autocast(enabled=use_fp8):
        out_ref = model(input_tensor)
    # Load model
    model_loaded.set_state_dict(paddle.load(file_name))
    if os.path.exists(file_name):
        os.remove(file_name)
    # Get actual output
    with fp8_autocast(enabled=use_fp8):
        out = model_loaded(input_tensor)

    assert_allclose(out, out_ref)


def calc_output_and_grad(layer, x, dy):
    """
    Calculate forward and backward pass
    """
    inp = paddle.to_tensor(x)
    inp.stop_gradient = x.stop_gradient
    y = layer(inp)
    y.backward(dy)

    return y, inp.grad if not inp.stop_gradient else None


@staticmethod
def calc_output_and_grad_ln_out(layer, x, dy, return_ln_out=False):
    """
    Calculate forward and backward pass for layernorm
    """
    inp = paddle.to_tensor(x)
    inp.stop_gradient = x.stop_gradient
    outputs = layer(inp)
    ln_out = None
    if return_ln_out:
        y, ln_out = outputs
    else:
        y = outputs
    y.backward(dy)

    return y, ln_out, inp.grad if not inp.stop_gradient else None


class TestLinear:
    """
    Tests for Linear layer
    """

    @staticmethod
    @pytest.mark.skipif(paddle.device.cuda.get_device_capability() < (8, 0),
                        reason="BF16 Linear requires Ampere+ GPU")
    @pytest.mark.parametrize('bs,in_features,out_features', LINEAR_CASES)
    @pytest.mark.parametrize('has_bias,no_dbias', [[True, False], [True, True], [False, False]])
    @pytest.mark.parametrize('no_dgrad', [True, False])
    @pytest.mark.parametrize('no_wgrad', [True, False])
    @pytest.mark.parametrize('activation_dtype', ['bfloat16', 'float32'])
    def test_linear_bf16(bs, in_features, out_features, has_bias, no_dbias, no_dgrad, no_wgrad,
                         activation_dtype):
        """
        Test BF16 Linear
        """
        rtol = 1e-2
        atol = 1e-2

        input_tensor = paddle.uniform(shape=(bs, in_features), dtype=activation_dtype)
        input_tensor.stop_gradient = no_dgrad
        grad_out = paddle.uniform(shape=(bs, out_features), dtype=activation_dtype)

        paddle.set_default_dtype(activation_dtype)
        layer_te = te.Linear(in_features, out_features, bias_attr=None if has_bias else False)
        layer_pd = te.Linear(in_features,
                             out_features,
                             bias_attr=None if has_bias else False,
                             backend='paddle')
        layer_pd.weight.copy_(layer_te.weight.T, True)
        if has_bias:
            layer_pd.bias.copy_(layer_te.bias, True)

        layer_te.weight.stop_gradient = no_wgrad
        layer_pd.weight.stop_gradient = no_wgrad
        if has_bias:
            layer_te.bias.stop_gradient = no_dbias
            layer_pd.bias.stop_gradient = no_dbias

        out_ref, grad_input_ref = calc_output_and_grad(layer_pd, input_tensor, grad_out)
        out, grad_input = calc_output_and_grad(layer_te, input_tensor, grad_out)

        assert_allclose(out, out_ref, rtol=rtol, atol=atol)
        if not no_dgrad:
            assert_allclose(grad_input, grad_input_ref, rtol=rtol, atol=atol)
        if not no_wgrad:
            assert_allclose(layer_te.weight.grad, layer_pd.weight.grad.T, rtol=rtol, atol=atol)
        if has_bias and not no_dbias:
            assert_allclose(layer_te.bias.grad, layer_pd.bias.grad, rtol=rtol, atol=atol)

    @staticmethod
    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize('bs,in_features,out_features', LINEAR_CASES)
    @pytest.mark.parametrize('has_bias,no_dbias', [[True, False], [True, True], [False, False]])
    @pytest.mark.parametrize('no_dgrad', [True, False])
    @pytest.mark.parametrize('no_wgrad', [True, False])
    @pytest.mark.parametrize('fp8_wgrad', [True, False])
    @pytest.mark.parametrize('do_calibration', [True, False])
    @pytest.mark.parametrize('activation_dtype', ['bfloat16', 'float32'])
    def test_linear_fp8(bs, in_features, out_features, has_bias, no_dbias, no_dgrad, no_wgrad,
                        fp8_wgrad, do_calibration, activation_dtype):
        """
        Test FP8 Linear
        """
        rtol = 0.1
        atol = 0.5

        input_tensor = paddle.uniform(shape=(bs, in_features), dtype=activation_dtype)
        input_tensor.stop_gradient = no_dgrad
        grad_out = paddle.uniform(shape=(bs, out_features), dtype=activation_dtype)

        recipe = DelayedScaling(override_linear_precision=(False, False, not fp8_wgrad))

        paddle.set_default_dtype(activation_dtype)
        layer_te = te.Linear(
            in_features=in_features,
            out_features=out_features,
            bias_attr=None if has_bias else False,
        )
        layer_pd = te.Linear(
            in_features=in_features,
            out_features=out_features,
            bias_attr=None if has_bias else False,
            backend='paddle',
        )
        layer_pd.weight.copy_(layer_te.weight.T, True)
        if has_bias:
            layer_pd.bias.copy_(layer_te.bias, True)

        layer_te.weight.stop_gradient = no_wgrad
        layer_pd.weight.stop_gradient = no_wgrad
        if has_bias:
            layer_te.bias.stop_gradient = no_dbias
            layer_pd.bias.stop_gradient = no_dbias

        with fp8_autocast(enabled=not do_calibration, calibrating=do_calibration,
                          fp8_recipe=recipe):
            out_ref, grad_input_ref = calc_output_and_grad(layer_pd, input_tensor, grad_out)
            out, grad_input = calc_output_and_grad(layer_te, input_tensor, grad_out)

        assert_allclose(out, out_ref, rtol=rtol, atol=atol)
        if not no_dgrad:
            assert_allclose(grad_input, grad_input_ref, rtol=rtol, atol=atol)
        if not no_wgrad:
            assert_allclose(layer_te.weight.grad, layer_pd.weight.grad.T, rtol=rtol, atol=atol)
        if has_bias and not no_dbias:
            assert_allclose(layer_te.bias.grad, layer_pd.bias.grad, rtol=rtol, atol=atol)
        if do_calibration:
            assert paddle.count_nonzero(layer_te.fp8_meta["scaling_fwd"].amax_history).item() > 0


@pytest.mark.parametrize('bs,hidden_size', NORM_CASES)
@pytest.mark.parametrize('has_bias,no_dbias', [[True, False], [True, True], [False, False]])
@pytest.mark.parametrize('no_dgrad', [True, False])
@pytest.mark.parametrize('no_wgrad', [True, False])
@pytest.mark.parametrize('activation_dtype', ['bfloat16', 'float32'])
def test_layernorm_bf16(bs, hidden_size, has_bias, no_dbias, no_dgrad, no_wgrad, activation_dtype):
    """
    Test BF16 LayerNorm
    """
    eps = 1e-3
    rtol = 1e-2
    atol = 1e-2

    x = paddle.uniform(shape=(bs, hidden_size), dtype=activation_dtype)
    x.stop_gradient = no_dgrad
    grad_out = paddle.uniform(shape=(bs, hidden_size), dtype=activation_dtype)

    paddle.set_default_dtype(activation_dtype)
    layer_te = te.LayerNorm(hidden_size=hidden_size, eps=eps, bias_attr=None if has_bias else False)
    layer_pd = te.LayerNorm(hidden_size=hidden_size,
                            eps=eps,
                            bias_attr=None if has_bias else False,
                            backend='paddle')
    layer_pd.weight.copy_(layer_te.weight, True)
    if has_bias:
        layer_pd.bias.copy_(layer_te.bias, True)

    layer_te.weight.stop_gradient = no_wgrad
    layer_pd.weight.stop_gradient = no_wgrad
    if has_bias:
        layer_te.bias.stop_gradient = no_dbias
        layer_pd.bias.stop_gradient = no_dbias

    out_ref, grad_input_ref = calc_output_and_grad(layer_pd, x, grad_out)
    out, grad_input = calc_output_and_grad(layer_te, x, grad_out)

    assert_allclose(out, out_ref, rtol=rtol, atol=atol)
    if not no_dgrad:
        assert_allclose(grad_input, grad_input_ref, rtol=rtol, atol=atol)
    if not no_wgrad:
        assert_allclose(layer_te.weight.grad, layer_pd.weight.grad, rtol=rtol, atol=atol)
    if has_bias and not no_dbias:
        assert_allclose(layer_te.bias.grad, layer_pd.bias.grad, rtol=rtol, atol=atol)


class TestLayerNormLinear:
    """
    Tests for LayerNormLinear layer
    """

    @staticmethod
    @pytest.mark.skipif(paddle.device.cuda.get_device_capability() < (8, 0),
                        reason="BF16 Linear requires Ampere+ GPU")
    @pytest.mark.parametrize('bs,in_features,out_features', LINEAR_CASES)
    @pytest.mark.parametrize('has_bias,no_dbias', [[True, False], [True, True], [False, False]])
    @pytest.mark.parametrize('no_dgrad', [True, False])
    @pytest.mark.parametrize('no_wgrad', [True, False])
    @pytest.mark.parametrize('return_ln_out', [True, False])
    @pytest.mark.parametrize('activation_dtype', ['bfloat16', 'float32'])
    def test_layernorm_linear_bf16(bs, in_features, out_features, has_bias, no_dbias, no_dgrad,
                                   no_wgrad, return_ln_out, activation_dtype):
        """
        Test BF16 LayerNormLinear Layer
        """
        paddle.set_default_dtype(activation_dtype)
        rtol = 1e-2
        atol = 1e-2

        input_tensor = paddle.uniform(shape=(bs, in_features), dtype=activation_dtype)
        input_tensor.stop_gradient = no_dgrad
        grad_out = paddle.uniform(shape=(bs, out_features), dtype=activation_dtype)
        eps = 1e-3

        layer_te = te.LayerNormLinear(
            in_features=in_features,
            out_features=out_features,
            eps=eps,
            bias_attr=None if has_bias else False,
            return_layernorm_output=return_ln_out,
        )

        layer_pd = te.LayerNormLinear(
            in_features=in_features,
            out_features=out_features,
            eps=eps,
            bias_attr=None if has_bias else False,
            return_layernorm_output=return_ln_out,
            backend='paddle',
        )

        layer_pd.ln_weight.copy_(layer_te.ln_weight, True)
        layer_pd.ln_bias.copy_(layer_te.ln_bias, True)
        layer_pd.weight.copy_(layer_te.weight.T, True)
        if has_bias:
            layer_pd.bias.copy_(layer_te.bias, True)

        layer_te.weight.stop_gradient = no_wgrad
        layer_te.ln_weight.stop_gradient = no_wgrad
        layer_te.ln_bias.stop_gradient = no_dbias
        layer_pd.weight.stop_gradient = no_wgrad
        layer_pd.ln_weight.stop_gradient = no_wgrad
        layer_pd.ln_bias.stop_gradient = no_dbias
        if has_bias:
            layer_te.bias.stop_gradient = no_dbias
            layer_pd.bias.stop_gradient = no_dbias

        out_ref, ln_out_ref, grad_input_ref = calc_output_and_grad_ln_out(
            layer_pd, input_tensor, grad_out, return_ln_out=return_ln_out)
        out, ln_out, grad_input = calc_output_and_grad_ln_out(layer_te,
                                                              input_tensor,
                                                              grad_out,
                                                              return_ln_out=return_ln_out)

        assert_allclose(out, out_ref, rtol=rtol, atol=atol)
        if not no_dgrad:
            assert_allclose(grad_input, grad_input_ref, rtol=rtol, atol=atol)
        if not no_wgrad:
            assert_allclose(layer_te.weight.grad, layer_pd.weight.grad.T, rtol=rtol, atol=atol)
            assert_allclose(layer_te.ln_weight.grad, layer_pd.ln_weight.grad, rtol=rtol, atol=atol)
        if not no_dbias:
            assert_allclose(layer_te.ln_bias.grad, layer_pd.ln_bias.grad, rtol=rtol, atol=atol)
            if has_bias:
                assert_allclose(layer_te.bias.grad, layer_pd.bias.grad, rtol=rtol, atol=atol)
        if return_ln_out:
            assert_allclose(ln_out, ln_out_ref, rtol=rtol, atol=atol)

    @staticmethod
    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize('bs,in_features,out_features', LINEAR_CASES)
    @pytest.mark.parametrize('has_bias,no_dbias', [[True, False], [True, True], [False, False]])
    @pytest.mark.parametrize('no_dgrad', [True, False])
    @pytest.mark.parametrize('no_wgrad', [True, False])
    @pytest.mark.parametrize('fp8_wgrad', [True, False])
    @pytest.mark.parametrize('do_calibration', [True, False])
    @pytest.mark.parametrize('return_ln_out', [True, False])
    @pytest.mark.parametrize('activation_dtype', ['bfloat16', 'float32'])
    def test_layernorm_linear_fp8(bs, in_features, out_features, has_bias, no_dbias, no_dgrad,
                                  no_wgrad, fp8_wgrad, do_calibration, return_ln_out,
                                  activation_dtype):
        """
        Test FP8 LayerNormLinear Layer
        """
        paddle.set_default_dtype(activation_dtype)
        rtol = 0.1
        atol = 0.75

        input_tensor = paddle.uniform(shape=(bs, in_features), dtype=activation_dtype)
        input_tensor.stop_gradient = no_dgrad
        grad_out = paddle.uniform(shape=(bs, out_features), dtype=activation_dtype)
        eps = 1e-3

        recipe = DelayedScaling(override_linear_precision=(False, False, not fp8_wgrad))

        layer_te = te.LayerNormLinear(
            in_features=in_features,
            out_features=out_features,
            eps=eps,
            bias_attr=None if has_bias else False,
            return_layernorm_output=return_ln_out,
        )

        layer_pd = te.LayerNormLinear(
            in_features=in_features,
            out_features=out_features,
            eps=eps,
            bias_attr=None if has_bias else False,
            return_layernorm_output=return_ln_out,
            backend='paddle',
        )

        layer_pd.ln_weight.copy_(layer_te.ln_weight, True)
        layer_pd.ln_bias.copy_(layer_te.ln_bias, True)
        layer_pd.weight.copy_(layer_te.weight.T, True)
        if has_bias:
            layer_pd.bias.copy_(layer_te.bias, True)

        layer_te.weight.stop_gradient = no_wgrad
        layer_te.ln_weight.stop_gradient = no_wgrad
        layer_te.ln_bias.stop_gradient = no_dbias
        layer_pd.weight.stop_gradient = no_wgrad
        layer_pd.ln_weight.stop_gradient = no_wgrad
        layer_pd.ln_bias.stop_gradient = no_dbias
        if has_bias:
            layer_te.bias.stop_gradient = no_dbias
            layer_pd.bias.stop_gradient = no_dbias

        with fp8_autocast(enabled=not do_calibration, calibrating=do_calibration,
                          fp8_recipe=recipe):
            out_ref, ln_out_ref, grad_input_ref = calc_output_and_grad_ln_out(
                layer_pd, input_tensor, grad_out, return_ln_out=return_ln_out)
            out, ln_out, grad_input = calc_output_and_grad_ln_out(layer_te,
                                                                  input_tensor,
                                                                  grad_out,
                                                                  return_ln_out=return_ln_out)

        assert_allclose(out, out_ref, rtol=rtol, atol=atol)
        if not no_dgrad:
            assert_allclose(grad_input, grad_input_ref, rtol=rtol, atol=atol)
        if not no_wgrad:
            assert_allclose(layer_te.weight.grad, layer_pd.weight.grad.T, rtol=rtol, atol=atol)
            assert_allclose(layer_te.ln_weight.grad, layer_pd.ln_weight.grad, rtol=rtol, atol=atol)
        if not no_dbias:
            assert_allclose(layer_te.ln_bias.grad, layer_pd.ln_bias.grad, rtol=rtol, atol=atol)
            if has_bias:
                assert_allclose(layer_te.bias.grad, layer_pd.bias.grad, rtol=rtol, atol=atol)
        if return_ln_out:
            assert_allclose(ln_out, ln_out_ref, rtol=rtol, atol=atol)
        if do_calibration:
            assert paddle.count_nonzero(layer_te.fp8_meta["scaling_fwd"].amax_history).item() > 0


class TestLayerNormMLP:
    """
    Test LayerNormMLP Layer
    """

    @staticmethod
    @pytest.mark.skipif(paddle.device.cuda.get_device_capability() < (8, 0),
                        reason="BF16 Linear requires Ampere+ GPU")
    @pytest.mark.parametrize('bs,hidden_size,ffn_hidden_size', LINEAR_CASES)
    @pytest.mark.parametrize('has_bias,no_dbias', [[True, False], [True, True], [False, False]])
    @pytest.mark.parametrize('no_dgrad', [True, False])
    @pytest.mark.parametrize('no_wgrad', [True, False])
    @pytest.mark.parametrize('return_ln_out', [True, False])
    @pytest.mark.parametrize('activation_dtype', ['bfloat16', 'float32'])
    def test_layernorm_mlp_bf16(bs, hidden_size, ffn_hidden_size, has_bias, no_dbias, no_dgrad,
                                no_wgrad, return_ln_out, activation_dtype):
        """
        Tests for TestLayerNormMLP layer
        """
        paddle.set_default_dtype(activation_dtype)
        rtol = 5e-2
        atol = 5e-2

        input_tensor = paddle.uniform(shape=(bs, hidden_size), dtype=activation_dtype)
        input_tensor.stop_gradient = no_dgrad
        grad_out = paddle.uniform(shape=(bs, hidden_size), dtype=activation_dtype)
        eps = 1e-3

        layer_te = te.LayerNormMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            eps=eps,
            bias_attr=None if has_bias else False,
            return_layernorm_output=return_ln_out,
        )
        layer_pd = te.LayerNormMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            eps=eps,
            bias_attr=None if has_bias else False,
            return_layernorm_output=return_ln_out,
            backend='paddle',
        )
        layer_pd.ln_weight.copy_(layer_te.ln_weight, True)
        layer_pd.ln_bias.copy_(layer_te.ln_bias, True)
        layer_pd.fc1_weight.copy_(layer_te.fc1_weight.T, True)
        layer_pd.fc2_weight.copy_(layer_te.fc2_weight.T, True)
        if has_bias:
            layer_pd.fc1_bias.copy_(layer_te.fc1_bias, True)
            layer_pd.fc2_bias.copy_(layer_te.fc2_bias, True)

        layer_te.fc1_weight.stop_gradient = no_wgrad
        layer_te.fc2_weight.stop_gradient = no_wgrad
        layer_te.ln_weight.stop_gradient = no_wgrad
        layer_te.ln_bias.stop_gradient = no_dbias
        layer_pd.fc1_weight.stop_gradient = no_wgrad
        layer_pd.fc2_weight.stop_gradient = no_wgrad
        layer_pd.ln_weight.stop_gradient = no_wgrad
        layer_pd.ln_bias.stop_gradient = no_dbias
        if has_bias:
            layer_te.fc1_bias.stop_gradient = no_dbias
            layer_te.fc2_bias.stop_gradient = no_dbias
            layer_pd.fc1_bias.stop_gradient = no_dbias
            layer_pd.fc2_bias.stop_gradient = no_dbias

        out_ref, ln_out_ref, grad_input_ref = calc_output_and_grad_ln_out(
            layer_pd, input_tensor, grad_out, return_ln_out=return_ln_out)
        out, ln_out, grad_input = calc_output_and_grad_ln_out(layer_te,
                                                              input_tensor,
                                                              grad_out,
                                                              return_ln_out=return_ln_out)

        assert_allclose(out, out_ref, rtol=rtol, atol=atol)
        if not no_dgrad:
            assert_allclose(grad_input, grad_input_ref, rtol=rtol, atol=atol)
        if not no_wgrad:
            assert_allclose(layer_te.ln_weight.grad, layer_pd.ln_weight.grad, rtol=rtol, atol=atol)
            assert_allclose(layer_te.fc1_weight.grad,
                            layer_pd.fc1_weight.grad.T,
                            rtol=rtol,
                            atol=atol)
            assert_allclose(layer_te.fc2_weight.grad,
                            layer_pd.fc2_weight.grad.T,
                            rtol=rtol,
                            atol=atol)
        if not no_dbias:
            assert_allclose(layer_te.ln_bias.grad, layer_pd.ln_bias.grad, rtol=rtol, atol=atol)
            if has_bias:
                assert_allclose(layer_te.fc1_bias.grad,
                                layer_pd.fc1_bias.grad,
                                rtol=rtol,
                                atol=atol)
                assert_allclose(layer_te.fc2_bias.grad,
                                layer_pd.fc2_bias.grad,
                                rtol=rtol,
                                atol=atol)
        if return_ln_out:
            assert_allclose(ln_out, ln_out_ref, rtol=rtol, atol=atol)

    @staticmethod
    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize('bs,hidden_size,ffn_hidden_size', LINEAR_CASES)
    @pytest.mark.parametrize('has_bias,no_dbias', [[True, False], [True, True], [False, False]])
    @pytest.mark.parametrize('no_dgrad', [True, False])
    @pytest.mark.parametrize('no_wgrad', [True, False])
    @pytest.mark.parametrize('fp8_wgrad', [True, False])
    @pytest.mark.parametrize('do_calibration', [True, False])
    @pytest.mark.parametrize('return_ln_out', [True, False])
    @pytest.mark.parametrize('activation_dtype', ['bfloat16', 'float32'])
    def test_layernorm_mlp_fp8(bs, hidden_size, ffn_hidden_size, has_bias, no_dbias, no_dgrad,
                               no_wgrad, fp8_wgrad, do_calibration, return_ln_out,
                               activation_dtype):
        """
        Test FP8 LayerNormMLP Layer
        """
        paddle.set_default_dtype(activation_dtype)
        rtol = 0.1
        atol = 0.75

        input_tensor = paddle.uniform(shape=(bs, hidden_size), dtype=activation_dtype)
        input_tensor.stop_gradient = no_dgrad
        grad_out = paddle.uniform(shape=(bs, hidden_size), dtype=activation_dtype)
        eps = 1e-3

        recipe = DelayedScaling(override_linear_precision=(False, False, not fp8_wgrad))

        layer_te = te.LayerNormMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            eps=eps,
            bias_attr=None if has_bias else False,
            return_layernorm_output=return_ln_out,
        )

        layer_pd = te.LayerNormMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            eps=eps,
            bias_attr=None if has_bias else False,
            return_layernorm_output=return_ln_out,
            backend='paddle',
        )
        layer_pd.ln_weight.copy_(layer_te.ln_weight, True)
        layer_pd.ln_bias.copy_(layer_te.ln_bias, True)
        layer_pd.fc1_weight.copy_(layer_te.fc1_weight.T, True)
        layer_pd.fc2_weight.copy_(layer_te.fc2_weight.T, True)
        if has_bias:
            layer_pd.fc1_bias.copy_(layer_te.fc1_bias, True)
            layer_pd.fc2_bias.copy_(layer_te.fc2_bias, True)

        layer_te.fc1_weight.stop_gradient = no_wgrad
        layer_te.fc2_weight.stop_gradient = no_wgrad
        layer_te.ln_weight.stop_gradient = no_wgrad
        layer_te.ln_bias.stop_gradient = no_dbias
        layer_pd.fc1_weight.stop_gradient = no_wgrad
        layer_pd.fc2_weight.stop_gradient = no_wgrad
        layer_pd.ln_weight.stop_gradient = no_wgrad
        layer_pd.ln_bias.stop_gradient = no_dbias
        if has_bias:
            layer_te.fc1_bias.stop_gradient = no_dbias
            layer_te.fc2_bias.stop_gradient = no_dbias
            layer_pd.fc1_bias.stop_gradient = no_dbias
            layer_pd.fc2_bias.stop_gradient = no_dbias

        with fp8_autocast(enabled=not do_calibration, calibrating=do_calibration,
                          fp8_recipe=recipe):
            out_ref, ln_out_ref, grad_input_ref = calc_output_and_grad_ln_out(
                layer_pd, input_tensor, grad_out, return_ln_out=return_ln_out)
            out, ln_out, grad_input = calc_output_and_grad_ln_out(layer_te,
                                                                  input_tensor,
                                                                  grad_out,
                                                                  return_ln_out=return_ln_out)

        assert_allclose(out, out_ref, rtol=rtol, atol=atol)
        if not no_dgrad:
            assert_allclose(grad_input, grad_input_ref, rtol=rtol, atol=atol)
        if not no_wgrad:
            assert_allclose(layer_te.ln_weight.grad, layer_pd.ln_weight.grad, rtol=rtol, atol=atol)
            assert_allclose(layer_te.fc1_weight.grad,
                            layer_pd.fc1_weight.grad.T,
                            rtol=rtol,
                            atol=atol)
            assert_allclose(layer_te.fc2_weight.grad,
                            layer_pd.fc2_weight.grad.T,
                            rtol=rtol,
                            atol=atol)
        if not no_dbias:
            assert_allclose(layer_te.ln_bias.grad, layer_pd.ln_bias.grad, rtol=rtol, atol=atol)
            if has_bias:
                assert_allclose(layer_te.fc1_bias.grad,
                                layer_pd.fc1_bias.grad,
                                rtol=rtol,
                                atol=atol)
                assert_allclose(layer_te.fc2_bias.grad,
                                layer_pd.fc2_bias.grad,
                                rtol=rtol,
                                atol=atol)
        if return_ln_out:
            assert_allclose(ln_out, ln_out_ref, rtol=rtol, atol=atol)

        if do_calibration:
            assert paddle.count_nonzero(layer_te.fp8_meta["scaling_fwd"].amax_history).item() > 0


@pytest.mark.skipif(paddle.device.cuda.get_device_capability() < (8, 0),
                    reason="cuDNN fMHA requires Ampere+ GPU")
@pytest.mark.parametrize('bs', [1, 2, 8])
@pytest.mark.parametrize('hidden_size, num_heads', [[1024, 16], [768, 12]])
@pytest.mark.parametrize('q_seqlen, kv_seqlen', [[128, 128], [512, 512]])
@pytest.mark.parametrize('attn_type', ['self', 'cross'])
@pytest.mark.parametrize('mask_type', ['causal', 'padding'])
@pytest.mark.parametrize('math_dtype', ['bfloat16', 'float16'])
def test_dot_product_attention(bs, hidden_size, num_heads, q_seqlen, kv_seqlen, attn_type,
                               mask_type, math_dtype):
    """
    Test DotProductAttention Layer
    """
    paddle.set_default_dtype(math_dtype)
    rtol = 1e-4
    atol = 2e-2

    head_size = hidden_size // num_heads
    self_attn_qkv_input = paddle.normal(mean=0.0,
                                        std=0.02,
                                        shape=(bs, q_seqlen, 3, num_heads,
                                               head_size)).astype(math_dtype)
    cross_attn_q_input = paddle.normal(mean=0.0,
                                       std=0.02,
                                       shape=(bs, q_seqlen, num_heads,
                                              head_size)).astype(math_dtype)
    cross_attn_kv_input = paddle.normal(mean=0.0,
                                        std=0.02,
                                        shape=(bs, kv_seqlen, 2, num_heads,
                                               head_size)).astype(math_dtype)

    q_actual_seqlen = paddle.randint(low=20, high=q_seqlen, shape=(bs,), dtype='int32')
    kv_actual_seqlen = paddle.randint(low=20, high=kv_seqlen, shape=(bs,),
                                      dtype='int32') if attn_type == 'cross' else q_actual_seqlen
    attn_mask = paddle.ones(shape=(bs, 1, q_seqlen, kv_seqlen), dtype='bool')

    grad_out = paddle.normal(mean=0.0, std=0.02,
                             shape=(bs, q_seqlen, num_heads, head_size)).astype('float32')
    for i in range(0, bs):
        grad_out[i, q_actual_seqlen[i]:, :, :] = 0
    grad_out = grad_out.astype(math_dtype)

    for i in range(0, bs):
        attn_mask[i, 0, 0:q_actual_seqlen[i], 0:kv_actual_seqlen[i]] = False

    norm_factor = math.sqrt(hidden_size // num_heads)
    layer_te = te.DotProductAttention(norm_factor,
                                      attention_dropout=0.0,
                                      attn_mask_type=mask_type,
                                      attention_type=attn_type,
                                      backend='transformer_engine')
    layer_pd = te.DotProductAttention(norm_factor,
                                      attention_dropout=0.0,
                                      attn_mask_type=mask_type,
                                      attention_type=attn_type,
                                      backend='paddle')

    def calc_attn_output_and_grad(layer, q, kv, mask, dout):
        _q = paddle.to_tensor(q, stop_gradient=False)
        _kv = paddle.to_tensor(kv, stop_gradient=False) if kv is not None else None

        out = layer(_q, _kv, mask)
        out.backward(dout)
        return out, _q.grad, _kv.grad if _kv is not None else None

    if attn_type == 'self':
        out, qkv_grad, _ = calc_attn_output_and_grad(layer_te, self_attn_qkv_input, None, attn_mask,
                                                     grad_out)
        out_ref, qkv_grad_ref, _ = calc_attn_output_and_grad(layer_pd, self_attn_qkv_input, None,
                                                             attn_mask, grad_out)
        valid_out_ref = paddle.full_like(out_ref, 0)
        for i in range(0, bs):
            valid_out_ref[i, 0:q_actual_seqlen[i], :, :] = out_ref[i, 0:q_actual_seqlen[i], :, :]

        q_grad = qkv_grad[:, :, 0]
        k_grad = qkv_grad[:, :, 1]
        v_grad = qkv_grad[:, :, 2]
        q_grad_ref = qkv_grad_ref[:, :, 0]
        k_grad_ref = qkv_grad_ref[:, :, 1]
        v_grad_ref = qkv_grad_ref[:, :, 2]

    else:
        out, q_grad, kv_grad = calc_attn_output_and_grad(layer_te, cross_attn_q_input,
                                                         cross_attn_kv_input, attn_mask, grad_out)
        out_ref, q_grad_ref, kv_grad_ref = calc_attn_output_and_grad(layer_pd, cross_attn_q_input,
                                                                     cross_attn_kv_input, attn_mask,
                                                                     grad_out)

        valid_out_ref = paddle.full_like(out_ref, 0)
        for i in range(0, bs):
            valid_out_ref[i, 0:q_actual_seqlen[i], :, :] = out_ref[i, 0:q_actual_seqlen[i], :, :]

        k_grad = kv_grad[:, :, 0]
        v_grad = kv_grad[:, :, 1]
        k_grad_ref = kv_grad_ref[:, :, 0]
        v_grad_ref = kv_grad_ref[:, :, 1]

    valid_q_grad_ref = paddle.full_like(q_grad_ref, 0)
    valid_k_grad_ref = paddle.full_like(k_grad_ref, 0)
    valid_v_grad_ref = paddle.full_like(v_grad_ref, 0)
    for i in range(0, bs):
        valid_q_grad_ref[i, 0:q_actual_seqlen[i], :, :] = q_grad_ref[i, 0:q_actual_seqlen[i], :, :]
        valid_k_grad_ref[i, 0:kv_actual_seqlen[i], :, :] = k_grad_ref[i,
                                                                      0:kv_actual_seqlen[i], :, :]
        valid_v_grad_ref[i, 0:kv_actual_seqlen[i], :, :] = v_grad_ref[i,
                                                                      0:kv_actual_seqlen[i], :, :]

    assert_allclose(out, valid_out_ref, rtol=rtol, atol=atol)
    assert_allclose(q_grad, valid_q_grad_ref, rtol=rtol, atol=atol)
    assert_allclose(k_grad, valid_k_grad_ref, rtol=rtol, atol=atol)
    assert_allclose(v_grad, valid_v_grad_ref, rtol=rtol, atol=atol)


@pytest.mark.skipif(paddle.device.cuda.get_device_capability() < (8, 0),
                    reason="cuDNN fMHA requires Ampere+ GPU")
@pytest.mark.parametrize('bs', [1, 2, 8])
@pytest.mark.parametrize('hidden_size, num_heads, ffn_hidden_size', [[1024, 16, 4096]])
@pytest.mark.parametrize('q_seqlen, kv_seqlen', [[128, 128], [512, 512]])
@pytest.mark.parametrize('has_bias, no_dbias', [[False, True], [True, True], [True, False]])
@pytest.mark.parametrize('no_wgrad', [True, False])
@pytest.mark.parametrize('mask_type', ['causal', 'padding'])
@pytest.mark.parametrize('math_dtype', ['bfloat16', 'float16'])
@pytest.mark.parametrize('output_layernorm', [True, False])
@pytest.mark.parametrize('return_layernorm_output', [True, False])
def test_transformer_encoder_layer(bs, hidden_size, num_heads, ffn_hidden_size, has_bias, no_dbias,
                                   no_wgrad, q_seqlen, kv_seqlen, mask_type, math_dtype,
                                   output_layernorm, return_layernorm_output):
    """
    Test Transformer Encoder Layer
    """
    paddle.set_default_dtype(math_dtype)
    rtol = 5e-2
    atol = 5e-2
    eps = 1e-3

    encoder_input = paddle.uniform(shape=(bs, q_seqlen, hidden_size), dtype=math_dtype)

    q_actual_seqlen = paddle.ones(shape=(bs,), dtype='int32') * q_seqlen
    kv_actual_seqlen = q_actual_seqlen
    attn_mask = paddle.ones(shape=(bs, 1, q_seqlen, kv_seqlen), dtype='bool')

    grad_out = paddle.normal(mean=0.0, std=0.02,
                             shape=(bs, q_seqlen, hidden_size)).astype('float32')
    for i in range(0, bs):
        grad_out[i, q_actual_seqlen[i]:, :] = 0
    grad_out = grad_out.astype(math_dtype)

    for i in range(0, bs):
        attn_mask[i, 0, 0:q_actual_seqlen[i], 0:kv_actual_seqlen[i]] = False

    layer_te = te.TransformerLayer(hidden_size,
                                   ffn_hidden_size,
                                   num_heads,
                                   layernorm_epsilon=eps,
                                   hidden_dropout=0.0,
                                   attention_dropout=0.0,
                                   weight_attr=None,
                                   bias_attr=None if has_bias else False,
                                   self_attn_mask_type=mask_type,
                                   apply_residual_connection_post_layernorm=return_layernorm_output,
                                   output_layernorm=output_layernorm,
                                   layer_type='encoder',
                                   backend='transformer_engine')
    layer_pd = te.TransformerLayer(hidden_size,
                                   ffn_hidden_size,
                                   num_heads,
                                   layernorm_epsilon=eps,
                                   hidden_dropout=0.0,
                                   attention_dropout=0.0,
                                   weight_attr=None,
                                   bias_attr=None if has_bias else False,
                                   self_attn_mask_type=mask_type,
                                   apply_residual_connection_post_layernorm=return_layernorm_output,
                                   output_layernorm=output_layernorm,
                                   layer_type='encoder',
                                   backend='paddle')

    # MultiHeadAttention params
    if output_layernorm:
        layer_pd.self_attention.qkv.weight.copy_(layer_te.self_attention.qkv.weight.T, True)
        layer_pd.self_attention.qkv.weight.stop_gradient = no_wgrad
        layer_te.self_attention.qkv.weight.stop_gradient = no_wgrad
        if has_bias:
            layer_pd.self_attention.qkv.bias.copy_(layer_te.self_attention.qkv.bias, True)
            layer_pd.self_attention.qkv.bias.stop_gradient = no_dbias
            layer_te.self_attention.qkv.bias.stop_gradient = no_dbias
    else:
        layer_pd.self_attention.layernorm_qkv.ln_weight.copy_(
            layer_te.self_attention.layernorm_qkv.ln_weight, True)
        layer_pd.self_attention.layernorm_qkv.ln_bias.copy_(
            layer_te.self_attention.layernorm_qkv.ln_bias, True)
        layer_pd.self_attention.layernorm_qkv.weight.copy_(
            layer_te.self_attention.layernorm_qkv.weight.T, True)
        layer_pd.self_attention.layernorm_qkv.ln_weight.stop_gradient = no_wgrad
        layer_pd.self_attention.layernorm_qkv.ln_bias.stop_gradient = no_dbias
        layer_pd.self_attention.layernorm_qkv.weight.stop_gradient = no_wgrad
        layer_te.self_attention.layernorm_qkv.ln_weight.stop_gradient = no_wgrad
        layer_te.self_attention.layernorm_qkv.ln_bias.stop_gradient = no_dbias
        layer_te.self_attention.layernorm_qkv.weight.stop_gradient = no_wgrad
        if has_bias:
            layer_pd.self_attention.layernorm_qkv.bias.copy_(
                layer_te.self_attention.layernorm_qkv.bias, True)
            layer_pd.self_attention.layernorm_qkv.bias.stop_gradient = no_dbias
            layer_te.self_attention.layernorm_qkv.bias.stop_gradient = no_dbias

    layer_pd.self_attention.proj.weight.copy_(layer_te.self_attention.proj.weight.T, True)
    layer_pd.self_attention.proj.weight.stop_gradient = no_wgrad
    layer_te.self_attention.proj.weight.stop_gradient = no_wgrad
    if has_bias:
        layer_pd.self_attention.proj.bias.copy_(layer_te.self_attention.proj.bias, True)
        layer_pd.self_attention.proj.bias.stop_gradient = no_dbias
        layer_te.self_attention.proj.bias.stop_gradient = no_dbias

    # LayerNorm MLP params
    layer_pd.layernorm_mlp.ln_weight.copy_(layer_te.layernorm_mlp.ln_weight, True)
    layer_pd.layernorm_mlp.ln_bias.copy_(layer_te.layernorm_mlp.ln_bias, True)
    layer_pd.layernorm_mlp.fc1_weight.copy_(layer_te.layernorm_mlp.fc1_weight.T, True)
    layer_pd.layernorm_mlp.fc2_weight.copy_(layer_te.layernorm_mlp.fc2_weight.T, True)
    layer_pd.layernorm_mlp.ln_weight.stop_gradient = no_wgrad
    layer_pd.layernorm_mlp.ln_bias.stop_gradient = no_dbias
    layer_pd.layernorm_mlp.fc1_weight.stop_gradient = no_wgrad
    layer_pd.layernorm_mlp.fc2_weight.stop_gradient = no_wgrad
    layer_te.layernorm_mlp.ln_weight.stop_gradient = no_wgrad
    layer_te.layernorm_mlp.ln_bias.stop_gradient = no_dbias
    layer_te.layernorm_mlp.fc1_weight.stop_gradient = no_wgrad
    layer_te.layernorm_mlp.fc2_weight.stop_gradient = no_wgrad
    if has_bias:
        layer_pd.layernorm_mlp.fc1_bias.copy_(layer_te.layernorm_mlp.fc1_bias, True)
        layer_pd.layernorm_mlp.fc2_bias.copy_(layer_te.layernorm_mlp.fc2_bias, True)
        layer_pd.layernorm_mlp.fc1_bias.stop_gradient = no_dbias
        layer_pd.layernorm_mlp.fc2_bias.stop_gradient = no_dbias
        layer_te.layernorm_mlp.fc1_bias.stop_gradient = no_dbias
        layer_te.layernorm_mlp.fc2_bias.stop_gradient = no_dbias

    if output_layernorm:
        layer_pd.layernorm.weight.copy_(layer_te.layernorm.weight, True)
        layer_pd.layernorm.bias.copy_(layer_te.layernorm.bias, True)
        layer_pd.layernorm.weight.stop_gradient = no_wgrad
        layer_pd.layernorm.bias.stop_gradient = no_dbias
        layer_te.layernorm.weight.stop_gradient = no_wgrad
        layer_te.layernorm.bias.stop_gradient = no_dbias

    def calc_transformer_output_and_grad(layer, encoder_input, mask, dout):
        _encoder_input = paddle.to_tensor(encoder_input, stop_gradient=False)
        out = layer(_encoder_input, mask)
        out.backward(dout)
        return out, _encoder_input.grad

    out_ref, grad_input_ref = calc_transformer_output_and_grad(layer_pd, encoder_input, attn_mask,
                                                               grad_out)
    out, grad_input = calc_transformer_output_and_grad(layer_te, encoder_input, attn_mask, grad_out)

    assert_allclose(out, out_ref, rtol=rtol, atol=atol)
    assert_allclose(grad_input, grad_input_ref, rtol=rtol, atol=atol)
    if not no_wgrad:
        if output_layernorm:
            assert_allclose(layer_te.self_attention.qkv.weight.grad,
                            layer_pd.self_attention.qkv.weight.grad.T,
                            rtol=rtol,
                            atol=atol)
        else:
            assert_allclose(layer_te.self_attention.layernorm_qkv.weight.grad,
                            layer_pd.self_attention.layernorm_qkv.weight.grad.T,
                            rtol=rtol,
                            atol=atol)
    if not no_dbias:
        if output_layernorm:
            assert_allclose(layer_te.self_attention.qkv.bias.grad,
                            layer_pd.self_attention.qkv.bias.grad,
                            rtol=0.01,
                            atol=0.5)
        else:
            assert_allclose(layer_te.self_attention.layernorm_qkv.bias.grad,
                            layer_pd.self_attention.layernorm_qkv.bias.grad,
                            rtol=0.01,
                            atol=0.5)


@pytest.mark.skipif(paddle.device.cuda.get_device_capability() < (8, 0),
                    reason="cuDNN fMHA requires Ampere+ GPU")
@pytest.mark.parametrize('bs', [1, 2, 8])
@pytest.mark.parametrize('hidden_size, num_heads, ffn_hidden_size', [[1024, 16, 4096]])
@pytest.mark.parametrize('q_seqlen, kv_seqlen', [[128, 128], [512, 512]])
@pytest.mark.parametrize('has_bias, no_dbias', [[False, True], [True, True], [True, False]])
@pytest.mark.parametrize('no_wgrad', [True, False])
@pytest.mark.parametrize('mask_type', ['causal', 'padding'])
@pytest.mark.parametrize('math_dtype', ['bfloat16', 'float16'])
@pytest.mark.parametrize('output_layernorm', [True, False])
@pytest.mark.parametrize('return_layernorm_output', [True, False])
def test_transformer_decoder_layer(bs, hidden_size, num_heads, ffn_hidden_size, has_bias, no_dbias,
                                   no_wgrad, q_seqlen, kv_seqlen, mask_type, math_dtype,
                                   output_layernorm, return_layernorm_output):
    """
    Test Transformer Decoder Layer
    """
    paddle.set_default_dtype(math_dtype)
    rtol = 5e-2
    atol = 5e-2
    eps = 1e-3

    encoder_input = paddle.uniform(shape=(bs, q_seqlen, hidden_size), dtype=math_dtype)
    encoder_output = paddle.uniform(shape=(bs, kv_seqlen, hidden_size), dtype=math_dtype)

    q_actual_seqlen = paddle.ones(shape=(bs,), dtype='int32') * q_seqlen
    kv_actual_seqlen = q_actual_seqlen
    attn_mask = paddle.ones(shape=(bs, 1, q_seqlen, kv_seqlen), dtype='bool')

    grad_out = paddle.normal(mean=0.0, std=0.2, shape=(bs, q_seqlen, hidden_size)).astype('float32')
    for i in range(0, bs):
        grad_out[i, q_actual_seqlen[i]:, :] = 0
    grad_out = grad_out.astype(math_dtype)

    for i in range(0, bs):
        attn_mask[i, 0, 0:q_actual_seqlen[i], 0:kv_actual_seqlen[i]] = False

    layer_te = te.TransformerLayer(hidden_size,
                                   ffn_hidden_size,
                                   num_heads,
                                   layernorm_epsilon=eps,
                                   hidden_dropout=0.0,
                                   attention_dropout=0.0,
                                   weight_attr=None,
                                   bias_attr=None if has_bias else False,
                                   self_attn_mask_type=mask_type,
                                   apply_residual_connection_post_layernorm=return_layernorm_output,
                                   output_layernorm=output_layernorm,
                                   layer_type='decoder',
                                   backend='transformer_engine')
    layer_pd = te.TransformerLayer(hidden_size,
                                   ffn_hidden_size,
                                   num_heads,
                                   layernorm_epsilon=eps,
                                   hidden_dropout=0.0,
                                   attention_dropout=0.0,
                                   weight_attr=None,
                                   bias_attr=None if has_bias else False,
                                   self_attn_mask_type=mask_type,
                                   apply_residual_connection_post_layernorm=return_layernorm_output,
                                   output_layernorm=output_layernorm,
                                   layer_type='decoder',
                                   backend='paddle')

    # MultiHeadAttention params - self attn
    if output_layernorm:
        layer_pd.self_attention.qkv.weight.copy_(layer_te.self_attention.qkv.weight.T, True)
        layer_pd.self_attention.qkv.weight.stop_gradient = no_wgrad
        layer_te.self_attention.qkv.weight.stop_gradient = no_wgrad
        if has_bias:
            layer_pd.self_attention.qkv.bias.copy_(layer_te.self_attention.qkv.bias, True)
            layer_pd.self_attention.qkv.bias.stop_gradient = no_dbias
            layer_te.self_attention.qkv.bias.stop_gradient = no_dbias
    else:
        layer_pd.self_attention.layernorm_qkv.ln_weight.copy_(
            layer_te.self_attention.layernorm_qkv.ln_weight, True)
        layer_pd.self_attention.layernorm_qkv.ln_bias.copy_(
            layer_te.self_attention.layernorm_qkv.ln_bias, True)
        layer_pd.self_attention.layernorm_qkv.weight.copy_(
            layer_te.self_attention.layernorm_qkv.weight.T, True)
        layer_pd.self_attention.layernorm_qkv.ln_weight.stop_gradient = no_wgrad
        layer_pd.self_attention.layernorm_qkv.ln_bias.stop_gradient = no_dbias
        layer_pd.self_attention.layernorm_qkv.weight.stop_gradient = no_wgrad
        layer_te.self_attention.layernorm_qkv.ln_weight.stop_gradient = no_wgrad
        layer_te.self_attention.layernorm_qkv.ln_bias.stop_gradient = no_dbias
        layer_te.self_attention.layernorm_qkv.weight.stop_gradient = no_wgrad
        if has_bias:
            layer_pd.self_attention.layernorm_qkv.bias.copy_(
                layer_te.self_attention.layernorm_qkv.bias, True)
            layer_pd.self_attention.layernorm_qkv.bias.stop_gradient = no_dbias
            layer_te.self_attention.layernorm_qkv.bias.stop_gradient = no_dbias

    layer_pd.self_attention.proj.weight.copy_(layer_te.self_attention.proj.weight.T, True)
    layer_pd.self_attention.proj.weight.stop_gradient = no_wgrad
    layer_te.self_attention.proj.weight.stop_gradient = no_wgrad
    if has_bias:
        layer_pd.self_attention.proj.bias.copy_(layer_te.self_attention.proj.bias, True)
        layer_pd.self_attention.proj.bias.stop_gradient = no_dbias
        layer_te.self_attention.proj.bias.stop_gradient = no_dbias

    # MultiHeadAttention params - cross attn
    layer_pd.inter_attention.layernorm_query.ln_weight.copy_(
        layer_te.inter_attention.layernorm_query.ln_weight, True)
    layer_pd.inter_attention.layernorm_query.ln_bias.copy_(
        layer_te.inter_attention.layernorm_query.ln_bias, True)
    layer_pd.inter_attention.layernorm_query.weight.copy_(
        layer_te.inter_attention.layernorm_query.weight.T, True)
    layer_pd.inter_attention.layernorm_query.ln_weight.stop_gradient = no_wgrad
    layer_pd.inter_attention.layernorm_query.ln_bias.stop_gradient = no_dbias
    layer_pd.inter_attention.layernorm_query.weight.stop_gradient = no_wgrad
    layer_te.inter_attention.layernorm_query.ln_weight.stop_gradient = no_wgrad
    layer_te.inter_attention.layernorm_query.ln_bias.stop_gradient = no_dbias
    layer_te.inter_attention.layernorm_query.weight.stop_gradient = no_wgrad
    if has_bias:
        layer_pd.inter_attention.layernorm_query.bias.copy_(
            layer_te.inter_attention.layernorm_query.bias, True)
        layer_pd.inter_attention.layernorm_query.bias.stop_gradient = no_dbias
        layer_te.inter_attention.layernorm_query.bias.stop_gradient = no_dbias

    layer_pd.inter_attention.key_value.weight.copy_(layer_te.inter_attention.key_value.weight.T,
                                                    True)
    layer_pd.inter_attention.key_value.weight.stop_gradient = no_wgrad
    layer_te.inter_attention.key_value.weight.stop_gradient = no_wgrad
    layer_pd.inter_attention.proj.weight.copy_(layer_te.inter_attention.proj.weight.T, True)
    layer_pd.inter_attention.proj.weight.stop_gradient = no_wgrad
    layer_te.inter_attention.proj.weight.stop_gradient = no_wgrad
    if has_bias:
        layer_pd.inter_attention.key_value.bias.copy_(layer_te.inter_attention.key_value.bias, True)
        layer_pd.inter_attention.key_value.bias.stop_gradient = no_dbias
        layer_te.inter_attention.key_value.bias.stop_gradient = no_dbias
        layer_pd.inter_attention.proj.bias.copy_(layer_te.inter_attention.proj.bias, True)
        layer_pd.inter_attention.proj.bias.stop_gradient = no_dbias
        layer_te.inter_attention.proj.bias.stop_gradient = no_dbias

    # LayerNorm MLP params
    layer_pd.layernorm_mlp.ln_weight.copy_(layer_te.layernorm_mlp.ln_weight, True)
    layer_pd.layernorm_mlp.ln_bias.copy_(layer_te.layernorm_mlp.ln_bias, True)
    layer_pd.layernorm_mlp.fc1_weight.copy_(layer_te.layernorm_mlp.fc1_weight.T, True)
    layer_pd.layernorm_mlp.fc2_weight.copy_(layer_te.layernorm_mlp.fc2_weight.T, True)
    layer_pd.layernorm_mlp.ln_weight.stop_gradient = no_wgrad
    layer_pd.layernorm_mlp.ln_bias.stop_gradient = no_dbias
    layer_pd.layernorm_mlp.fc1_weight.stop_gradient = no_wgrad
    layer_pd.layernorm_mlp.fc2_weight.stop_gradient = no_wgrad
    layer_te.layernorm_mlp.ln_weight.stop_gradient = no_wgrad
    layer_te.layernorm_mlp.ln_bias.stop_gradient = no_dbias
    layer_te.layernorm_mlp.fc1_weight.stop_gradient = no_wgrad
    layer_te.layernorm_mlp.fc2_weight.stop_gradient = no_wgrad
    if has_bias:
        layer_pd.layernorm_mlp.fc1_bias.copy_(layer_te.layernorm_mlp.fc1_bias, True)
        layer_pd.layernorm_mlp.fc2_bias.copy_(layer_te.layernorm_mlp.fc2_bias, True)
        layer_pd.layernorm_mlp.fc1_bias.stop_gradient = no_dbias
        layer_pd.layernorm_mlp.fc2_bias.stop_gradient = no_dbias
        layer_te.layernorm_mlp.fc1_bias.stop_gradient = no_dbias
        layer_te.layernorm_mlp.fc2_bias.stop_gradient = no_dbias

    if output_layernorm:
        layer_pd.layernorm.weight.copy_(layer_te.layernorm.weight, True)
        layer_pd.layernorm.bias.copy_(layer_te.layernorm.bias, True)
        layer_pd.layernorm.weight.stop_gradient = no_wgrad
        layer_pd.layernorm.bias.stop_gradient = no_dbias
        layer_te.layernorm.weight.stop_gradient = no_wgrad
        layer_te.layernorm.bias.stop_gradient = no_dbias

    def calc_transformer_output_and_grad(layer, encoder_input, mask, encoder_output,
                                         enc_dec_attn_mask, dout):
        _encoder_input = paddle.to_tensor(encoder_input, stop_gradient=False)
        _encoder_output = paddle.to_tensor(encoder_output, stop_gradient=False)
        out = layer(_encoder_input, mask, _encoder_output, enc_dec_attn_mask)
        out.backward(dout)
        return out, _encoder_input.grad, _encoder_output.grad

    out_ref, grad_encoder_input_ref, grad_encoder_output_ref = calc_transformer_output_and_grad(
        layer_pd, encoder_input, attn_mask, encoder_output, attn_mask, grad_out)
    out, grad_encoder_input, grad_encoder_output = calc_transformer_output_and_grad(
        layer_te, encoder_input, attn_mask, encoder_output, attn_mask, grad_out)

    assert_allclose(out, out_ref, rtol=rtol, atol=atol)
    assert_allclose(grad_encoder_input, grad_encoder_input_ref, rtol=rtol, atol=atol)
    assert_allclose(grad_encoder_output, grad_encoder_output_ref, rtol=rtol, atol=atol)
    if not no_wgrad:
        if output_layernorm:
            assert_allclose(layer_te.self_attention.qkv.weight.grad,
                            layer_pd.self_attention.qkv.weight.grad.T,
                            rtol=rtol,
                            atol=atol)
        else:
            assert_allclose(layer_te.self_attention.layernorm_qkv.weight.grad,
                            layer_pd.self_attention.layernorm_qkv.weight.grad.T,
                            rtol=rtol,
                            atol=0.1)
            assert_allclose(layer_te.inter_attention.layernorm_query.weight.grad,
                            layer_pd.inter_attention.layernorm_query.weight.grad.T,
                            rtol=rtol,
                            atol=atol)
    if not no_dbias:
        if output_layernorm:
            assert_allclose(layer_te.self_attention.qkv.bias.grad,
                            layer_pd.self_attention.qkv.bias.grad,
                            rtol=0.01,
                            atol=0.5)
        else:
            assert_allclose(layer_te.self_attention.layernorm_qkv.bias.grad,
                            layer_pd.self_attention.layernorm_qkv.bias.grad,
                            rtol=0.01,
                            atol=0.5)
            assert_allclose(layer_te.inter_attention.layernorm_query.bias.grad,
                            layer_pd.inter_attention.layernorm_query.bias.grad,
                            rtol=rtol,
                            atol=atol)
