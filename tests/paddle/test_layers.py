# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Test TE Paddle Layer-level APIs"""

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
