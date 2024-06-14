# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Test TE Paddle Layer-level APIs"""

import os
from utils import assert_allclose, is_fused_attention_supported

import paddle
import pytest

from transformer_engine.common.recipe import DelayedScaling
import transformer_engine.paddle as te
from transformer_engine.paddle.fp8 import is_fp8_available, fp8_autocast

is_fp8_supported, reason = is_fp8_available()
LINEAR_CASES = [(16, 16, 32), (32, 32, 64)]
NORM_CASES = [(16, 32), (256, 1024)]


@pytest.fixture(autouse=True)
def setup():
    """Setup random seed before each test"""
    paddle.seed(10)
    yield


@pytest.mark.skipif(not is_fp8_supported, reason=reason)
@pytest.mark.parametrize("use_fp8", [True, False])
def test_checkpoint(use_fp8):
    """Test checkpoint save / load"""
    bs = 16
    in_features = 16
    out_features = 32
    file_name = "model.pdparams"
    input_tensor = paddle.uniform(shape=(bs, in_features), dtype="float32")
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
    @pytest.mark.skipif(
        paddle.device.cuda.get_device_capability() < (8, 0),
        reason="BF16 Linear requires Ampere+ GPU",
    )
    @pytest.mark.parametrize("bs,in_features,out_features", LINEAR_CASES)
    @pytest.mark.parametrize("has_bias,no_dbias", [[True, False], [True, True], [False, False]])
    @pytest.mark.parametrize("no_dgrad", [True, False])
    @pytest.mark.parametrize("no_wgrad", [True, False])
    @pytest.mark.parametrize("activation_dtype", ["bfloat16", "float32"])
    def test_linear_bf16(
        bs, in_features, out_features, has_bias, no_dbias, no_dgrad, no_wgrad, activation_dtype
    ):
        """
        Test BF16 Linear
        """
        rtol = 5e-2
        atol = 5e-2

        input_tensor = paddle.uniform(shape=(bs, in_features), dtype=activation_dtype)
        input_tensor.stop_gradient = no_dgrad
        grad_out = paddle.uniform(shape=(bs, out_features), dtype=activation_dtype)

        paddle.set_default_dtype(activation_dtype)
        layer_te = te.Linear(in_features, out_features, bias_attr=None if has_bias else False)
        layer_pd = te.Linear(
            in_features, out_features, bias_attr=None if has_bias else False, backend="paddle"
        )
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
    @pytest.mark.parametrize("bs,in_features,out_features", LINEAR_CASES)
    @pytest.mark.parametrize("has_bias,no_dbias", [[True, False], [True, True], [False, False]])
    @pytest.mark.parametrize("no_dgrad", [True, False])
    @pytest.mark.parametrize("no_wgrad", [True, False])
    @pytest.mark.parametrize("fp8_wgrad", [True, False])
    @pytest.mark.parametrize("do_calibration", [True, False])
    @pytest.mark.parametrize("activation_dtype", ["bfloat16", "float32"])
    def test_linear_fp8(
        bs,
        in_features,
        out_features,
        has_bias,
        no_dbias,
        no_dgrad,
        no_wgrad,
        fp8_wgrad,
        do_calibration,
        activation_dtype,
    ):
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
            backend="paddle",
        )
        layer_pd.weight.copy_(layer_te.weight.T, True)
        if has_bias:
            layer_pd.bias.copy_(layer_te.bias, True)

        layer_te.weight.stop_gradient = no_wgrad
        layer_pd.weight.stop_gradient = no_wgrad
        if has_bias:
            layer_te.bias.stop_gradient = no_dbias
            layer_pd.bias.stop_gradient = no_dbias

        with fp8_autocast(
            enabled=not do_calibration, calibrating=do_calibration, fp8_recipe=recipe
        ):
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

    @staticmethod
    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize("bs,in_features,out_features", LINEAR_CASES)
    @pytest.mark.parametrize("activation_dtype", ["bfloat16"])
    @pytest.mark.parametrize("num_microbatch", [8])
    def test_linear_fp8_microbatch(bs, in_features, out_features, activation_dtype, num_microbatch):
        """
        Test FP8 Linear
        """
        rtol = 0.1
        atol = 0.1

        recipe = DelayedScaling()

        paddle.set_default_dtype(activation_dtype)
        layer_cached = te.Linear(
            in_features=in_features,
            out_features=out_features,
        )
        layer_normal = te.Linear(
            in_features=in_features,
            out_features=out_features,
        )
        layer_cached.weight.copy_(layer_normal.weight, True)
        layer_cached.bias.copy_(layer_normal.bias, True)

        for iteration in range(num_microbatch):
            input_tensor = paddle.uniform(shape=(bs, in_features), dtype=activation_dtype)
            grad_out = paddle.uniform(shape=(bs, out_features), dtype=activation_dtype)

            with fp8_autocast(enabled=True, fp8_recipe=recipe):
                out = layer_cached(input_tensor, is_first_microbatch=(iteration == 0))
                out.backward(grad_out)

            with fp8_autocast(enabled=True, fp8_recipe=recipe):
                out_ref = layer_normal(input_tensor)
                out_ref.backward(grad_out)

            assert_allclose(out, out_ref, rtol=rtol, atol=atol)
            assert_allclose(
                layer_cached.weight.grad, layer_normal.weight.grad, rtol=rtol, atol=atol
            )


@pytest.mark.parametrize("bs,hidden_size", NORM_CASES)
@pytest.mark.parametrize("has_bias,no_dbias", [[True, False], [True, True], [False, False]])
@pytest.mark.parametrize("no_dgrad", [True, False])
@pytest.mark.parametrize("no_wgrad", [True, False])
@pytest.mark.parametrize("activation_dtype", ["bfloat16", "float32"])
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
    layer_pd = te.LayerNorm(
        hidden_size=hidden_size, eps=eps, bias_attr=None if has_bias else False, backend="paddle"
    )
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
    @pytest.mark.skipif(
        paddle.device.cuda.get_device_capability() < (8, 0),
        reason="BF16 Linear requires Ampere+ GPU",
    )
    @pytest.mark.parametrize("bs,in_features,out_features", LINEAR_CASES)
    @pytest.mark.parametrize("has_bias,no_dbias", [[True, False], [True, True], [False, False]])
    @pytest.mark.parametrize("no_dgrad", [True, False])
    @pytest.mark.parametrize("no_wgrad", [True, False])
    @pytest.mark.parametrize("return_ln_out", [True, False])
    @pytest.mark.parametrize("activation_dtype", ["bfloat16", "float32"])
    @pytest.mark.parametrize("normalization", ["RMSNorm", "LayerNorm"])
    def test_layernorm_linear_bf16(
        bs,
        in_features,
        out_features,
        has_bias,
        no_dbias,
        no_dgrad,
        no_wgrad,
        return_ln_out,
        activation_dtype,
        normalization,
    ):
        """
        Test BF16 LayerNormLinear Layer
        """
        paddle.set_default_dtype(activation_dtype)
        rtol = 5e-2
        atol = 5e-2

        input_tensor = paddle.uniform(shape=(bs, in_features), dtype=activation_dtype)
        input_tensor.stop_gradient = no_dgrad
        grad_out = paddle.uniform(shape=(bs, out_features), dtype=activation_dtype)
        eps = 1e-3
        has_ln_bias = normalization == "LayerNorm"

        layer_te = te.LayerNormLinear(
            in_features=in_features,
            out_features=out_features,
            eps=eps,
            normalization=normalization,
            bias_attr=None if has_bias else False,
            return_layernorm_output=return_ln_out,
        )

        layer_pd = te.LayerNormLinear(
            in_features=in_features,
            out_features=out_features,
            eps=eps,
            normalization=normalization,
            bias_attr=None if has_bias else False,
            return_layernorm_output=return_ln_out,
            backend="paddle",
        )

        layer_pd.ln_weight.copy_(layer_te.ln_weight, True)
        if has_ln_bias:
            layer_pd.ln_bias.copy_(layer_te.ln_bias, True)
        layer_pd.weight.copy_(layer_te.weight.T, True)
        if has_bias:
            layer_pd.bias.copy_(layer_te.bias, True)

        layer_te.weight.stop_gradient = no_wgrad
        layer_te.ln_weight.stop_gradient = no_wgrad
        layer_pd.weight.stop_gradient = no_wgrad
        layer_pd.ln_weight.stop_gradient = no_wgrad
        if has_ln_bias:
            layer_te.ln_bias.stop_gradient = no_dbias
            layer_pd.ln_bias.stop_gradient = no_dbias
        if has_bias:
            layer_te.bias.stop_gradient = no_dbias
            layer_pd.bias.stop_gradient = no_dbias

        out_ref, ln_out_ref, grad_input_ref = calc_output_and_grad_ln_out(
            layer_pd, input_tensor, grad_out, return_ln_out=return_ln_out
        )
        out, ln_out, grad_input = calc_output_and_grad_ln_out(
            layer_te, input_tensor, grad_out, return_ln_out=return_ln_out
        )

        assert_allclose(out, out_ref, rtol=rtol, atol=atol)
        if not no_dgrad:
            assert_allclose(grad_input, grad_input_ref, rtol=rtol, atol=atol)
        if not no_wgrad:
            assert_allclose(layer_te.weight.grad, layer_pd.weight.grad.T, rtol=rtol, atol=atol)
            assert_allclose(layer_te.ln_weight.grad, layer_pd.ln_weight.grad, rtol=rtol, atol=atol)
        if not no_dbias:
            if has_ln_bias:
                assert_allclose(layer_te.ln_bias.grad, layer_pd.ln_bias.grad, rtol=rtol, atol=atol)
            if has_bias:
                assert_allclose(layer_te.bias.grad, layer_pd.bias.grad, rtol=rtol, atol=atol)
        if return_ln_out:
            assert_allclose(ln_out, ln_out_ref, rtol=rtol, atol=atol)

    @staticmethod
    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize("bs,in_features,out_features", LINEAR_CASES)
    @pytest.mark.parametrize("has_bias,no_dbias", [[True, False], [True, True], [False, False]])
    @pytest.mark.parametrize("no_dgrad", [True, False])
    @pytest.mark.parametrize("no_wgrad", [True, False])
    @pytest.mark.parametrize("fp8_wgrad", [True, False])
    @pytest.mark.parametrize("do_calibration", [True, False])
    @pytest.mark.parametrize("return_ln_out", [True, False])
    @pytest.mark.parametrize("activation_dtype", ["bfloat16", "float32"])
    @pytest.mark.parametrize("normalization", ["RMSNorm", "LayerNorm"])
    def test_layernorm_linear_fp8(
        bs,
        in_features,
        out_features,
        has_bias,
        no_dbias,
        no_dgrad,
        no_wgrad,
        fp8_wgrad,
        do_calibration,
        return_ln_out,
        activation_dtype,
        normalization,
    ):
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
        has_ln_bias = normalization == "LayerNorm"

        recipe = DelayedScaling(override_linear_precision=(False, False, not fp8_wgrad))

        layer_te = te.LayerNormLinear(
            in_features=in_features,
            out_features=out_features,
            eps=eps,
            normalization=normalization,
            bias_attr=None if has_bias else False,
            return_layernorm_output=return_ln_out,
        )

        layer_pd = te.LayerNormLinear(
            in_features=in_features,
            out_features=out_features,
            eps=eps,
            normalization=normalization,
            bias_attr=None if has_bias else False,
            return_layernorm_output=return_ln_out,
            backend="paddle",
        )

        layer_pd.ln_weight.copy_(layer_te.ln_weight, True)
        if has_ln_bias:
            layer_pd.ln_bias.copy_(layer_te.ln_bias, True)
        layer_pd.weight.copy_(layer_te.weight.T, True)
        if has_bias:
            layer_pd.bias.copy_(layer_te.bias, True)

        layer_te.weight.stop_gradient = no_wgrad
        layer_te.ln_weight.stop_gradient = no_wgrad
        layer_pd.weight.stop_gradient = no_wgrad
        layer_pd.ln_weight.stop_gradient = no_wgrad
        if has_ln_bias:
            layer_te.ln_bias.stop_gradient = no_dbias
            layer_pd.ln_bias.stop_gradient = no_dbias
        if has_bias:
            layer_te.bias.stop_gradient = no_dbias
            layer_pd.bias.stop_gradient = no_dbias

        with fp8_autocast(
            enabled=not do_calibration, calibrating=do_calibration, fp8_recipe=recipe
        ):
            out_ref, ln_out_ref, grad_input_ref = calc_output_and_grad_ln_out(
                layer_pd, input_tensor, grad_out, return_ln_out=return_ln_out
            )
            out, ln_out, grad_input = calc_output_and_grad_ln_out(
                layer_te, input_tensor, grad_out, return_ln_out=return_ln_out
            )

        assert_allclose(out, out_ref, rtol=rtol, atol=atol)
        if not no_dgrad:
            assert_allclose(grad_input, grad_input_ref, rtol=rtol, atol=atol)
        if not no_wgrad:
            assert_allclose(layer_te.weight.grad, layer_pd.weight.grad.T, rtol=rtol, atol=atol)
            assert_allclose(layer_te.ln_weight.grad, layer_pd.ln_weight.grad, rtol=rtol, atol=atol)
        if not no_dbias:
            if has_ln_bias:
                assert_allclose(layer_te.ln_bias.grad, layer_pd.ln_bias.grad, rtol=rtol, atol=atol)
            if has_bias:
                assert_allclose(layer_te.bias.grad, layer_pd.bias.grad, rtol=rtol, atol=atol)
        if return_ln_out:
            assert_allclose(ln_out, ln_out_ref, rtol=rtol, atol=atol)
        if do_calibration:
            assert paddle.count_nonzero(layer_te.fp8_meta["scaling_fwd"].amax_history).item() > 0

    @staticmethod
    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize("bs,in_features,out_features", LINEAR_CASES)
    @pytest.mark.parametrize("activation_dtype", ["bfloat16"])
    @pytest.mark.parametrize("num_microbatch", [8])
    def test_layernorm_linear_fp8_microbatch(
        bs, in_features, out_features, activation_dtype, num_microbatch
    ):
        """
        Test FP8 LayerNormLinear Layer
        """
        paddle.set_default_dtype(activation_dtype)
        eps = 1e-3
        rtol = 0.5
        atol = 0.5

        recipe = DelayedScaling()

        layer_cached = te.LayerNormLinear(
            in_features=in_features,
            out_features=out_features,
            eps=eps,
        )

        layer_normal = te.LayerNormLinear(
            in_features=in_features,
            out_features=out_features,
            eps=eps,
        )

        layer_cached.ln_weight.copy_(layer_normal.ln_weight, True)
        layer_cached.ln_bias.copy_(layer_normal.ln_bias, True)
        layer_cached.weight.copy_(layer_normal.weight, True)
        layer_cached.bias.copy_(layer_normal.bias, True)

        for iteration in range(num_microbatch):
            input_tensor = paddle.uniform(shape=(bs, in_features), dtype=activation_dtype)
            grad_out = paddle.uniform(shape=(bs, out_features), dtype=activation_dtype)

            with fp8_autocast(enabled=True, fp8_recipe=recipe):
                out = layer_cached(input_tensor, is_first_microbatch=(iteration == 0))
                out.backward(grad_out)

            with fp8_autocast(enabled=True, fp8_recipe=recipe):
                out_ref = layer_normal(input_tensor)
                out_ref.backward(grad_out)

            assert_allclose(out, out_ref, rtol=rtol, atol=atol)
            assert_allclose(
                layer_cached.weight.grad, layer_normal.weight.grad, rtol=rtol, atol=atol
            )
            assert_allclose(
                layer_cached.ln_weight.grad, layer_normal.ln_weight.grad, rtol=rtol, atol=atol
            )


class TestLayerNormMLP:
    """
    Test LayerNormMLP Layer
    """

    @staticmethod
    @pytest.mark.skipif(
        paddle.device.cuda.get_device_capability() < (8, 0),
        reason="BF16 Linear requires Ampere+ GPU",
    )
    @pytest.mark.parametrize("bs,hidden_size,ffn_hidden_size", LINEAR_CASES)
    @pytest.mark.parametrize("has_bias,no_dbias", [[True, False], [True, True], [False, False]])
    @pytest.mark.parametrize("no_dgrad", [True, False])
    @pytest.mark.parametrize("no_wgrad", [True, False])
    @pytest.mark.parametrize("return_ln_out", [True, False])
    @pytest.mark.parametrize("activation_dtype", ["bfloat16", "float32"])
    @pytest.mark.parametrize("normalization", ["RMSNorm", "LayerNorm"])
    @pytest.mark.parametrize("activation", ["gelu", "swiglu"])
    def test_layernorm_mlp_bf16(
        bs,
        hidden_size,
        ffn_hidden_size,
        has_bias,
        no_dbias,
        no_dgrad,
        no_wgrad,
        return_ln_out,
        activation_dtype,
        normalization,
        activation,
    ):
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
        has_ln_bias = normalization == "LayerNorm"

        layer_te = te.LayerNormMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            eps=eps,
            normalization=normalization,
            activation=activation,
            bias_attr=None if has_bias else False,
            return_layernorm_output=return_ln_out,
        )
        layer_pd = te.LayerNormMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            eps=eps,
            normalization=normalization,
            activation=activation,
            bias_attr=None if has_bias else False,
            return_layernorm_output=return_ln_out,
            backend="paddle",
        )
        layer_pd.ln_weight.copy_(layer_te.ln_weight, True)
        if has_ln_bias:
            layer_pd.ln_bias.copy_(layer_te.ln_bias, True)
        layer_pd.fc1_weight.copy_(layer_te.fc1_weight.T, True)
        layer_pd.fc2_weight.copy_(layer_te.fc2_weight.T, True)
        if has_bias:
            layer_pd.fc1_bias.copy_(layer_te.fc1_bias, True)
            layer_pd.fc2_bias.copy_(layer_te.fc2_bias, True)

        layer_te.fc1_weight.stop_gradient = no_wgrad
        layer_te.fc2_weight.stop_gradient = no_wgrad
        layer_te.ln_weight.stop_gradient = no_wgrad
        layer_pd.fc1_weight.stop_gradient = no_wgrad
        layer_pd.fc2_weight.stop_gradient = no_wgrad
        layer_pd.ln_weight.stop_gradient = no_wgrad
        if has_ln_bias:
            layer_te.ln_bias.stop_gradient = no_dbias
            layer_pd.ln_bias.stop_gradient = no_dbias
        if has_bias:
            layer_te.fc1_bias.stop_gradient = no_dbias
            layer_te.fc2_bias.stop_gradient = no_dbias
            layer_pd.fc1_bias.stop_gradient = no_dbias
            layer_pd.fc2_bias.stop_gradient = no_dbias

        out_ref, ln_out_ref, grad_input_ref = calc_output_and_grad_ln_out(
            layer_pd, input_tensor, grad_out, return_ln_out=return_ln_out
        )
        out, ln_out, grad_input = calc_output_and_grad_ln_out(
            layer_te, input_tensor, grad_out, return_ln_out=return_ln_out
        )

        assert_allclose(out, out_ref, rtol=rtol, atol=atol)
        if not no_dgrad:
            assert_allclose(grad_input, grad_input_ref, rtol=rtol, atol=atol)
        if not no_wgrad:
            assert_allclose(layer_te.ln_weight.grad, layer_pd.ln_weight.grad, rtol=rtol, atol=atol)
            assert_allclose(
                layer_te.fc1_weight.grad, layer_pd.fc1_weight.grad.T, rtol=rtol, atol=atol
            )
            assert_allclose(
                layer_te.fc2_weight.grad, layer_pd.fc2_weight.grad.T, rtol=rtol, atol=atol
            )
        if not no_dbias:
            if has_ln_bias:
                assert_allclose(layer_te.ln_bias.grad, layer_pd.ln_bias.grad, rtol=rtol, atol=atol)
            if has_bias:
                assert_allclose(
                    layer_te.fc1_bias.grad, layer_pd.fc1_bias.grad, rtol=rtol, atol=atol
                )
                assert_allclose(
                    layer_te.fc2_bias.grad, layer_pd.fc2_bias.grad, rtol=rtol, atol=atol
                )
        if return_ln_out:
            assert_allclose(ln_out, ln_out_ref, rtol=rtol, atol=atol)

    @staticmethod
    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize("bs,hidden_size,ffn_hidden_size", LINEAR_CASES)
    @pytest.mark.parametrize("has_bias,no_dbias", [[True, False], [True, True], [False, False]])
    @pytest.mark.parametrize("no_dgrad", [True, False])
    @pytest.mark.parametrize("no_wgrad", [True, False])
    @pytest.mark.parametrize("fp8_wgrad", [True, False])
    @pytest.mark.parametrize("do_calibration", [True, False])
    @pytest.mark.parametrize("return_ln_out", [True, False])
    @pytest.mark.parametrize("activation_dtype", ["bfloat16", "float32"])
    @pytest.mark.parametrize("normalization", ["RMSNorm", "LayerNorm"])
    @pytest.mark.parametrize("activation", ["gelu", "swiglu"])
    def test_layernorm_mlp_fp8(
        bs,
        hidden_size,
        ffn_hidden_size,
        has_bias,
        no_dbias,
        no_dgrad,
        no_wgrad,
        fp8_wgrad,
        do_calibration,
        return_ln_out,
        activation_dtype,
        normalization,
        activation,
    ):
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
        has_ln_bias = normalization == "LayerNorm"

        recipe = DelayedScaling(override_linear_precision=(False, False, not fp8_wgrad))

        layer_te = te.LayerNormMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            eps=eps,
            normalization=normalization,
            activation=activation,
            bias_attr=None if has_bias else False,
            return_layernorm_output=return_ln_out,
        )

        layer_pd = te.LayerNormMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            eps=eps,
            normalization=normalization,
            activation=activation,
            bias_attr=None if has_bias else False,
            return_layernorm_output=return_ln_out,
            backend="paddle",
        )
        layer_pd.ln_weight.copy_(layer_te.ln_weight, True)
        if has_ln_bias:
            layer_pd.ln_bias.copy_(layer_te.ln_bias, True)
        layer_pd.fc1_weight.copy_(layer_te.fc1_weight.T, True)
        layer_pd.fc2_weight.copy_(layer_te.fc2_weight.T, True)
        if has_bias:
            layer_pd.fc1_bias.copy_(layer_te.fc1_bias, True)
            layer_pd.fc2_bias.copy_(layer_te.fc2_bias, True)

        layer_te.fc1_weight.stop_gradient = no_wgrad
        layer_te.fc2_weight.stop_gradient = no_wgrad
        layer_te.ln_weight.stop_gradient = no_wgrad
        layer_pd.fc1_weight.stop_gradient = no_wgrad
        layer_pd.fc2_weight.stop_gradient = no_wgrad
        layer_pd.ln_weight.stop_gradient = no_wgrad
        if has_ln_bias:
            layer_te.ln_bias.stop_gradient = no_dbias
            layer_pd.ln_bias.stop_gradient = no_dbias
        if has_bias:
            layer_te.fc1_bias.stop_gradient = no_dbias
            layer_te.fc2_bias.stop_gradient = no_dbias
            layer_pd.fc1_bias.stop_gradient = no_dbias
            layer_pd.fc2_bias.stop_gradient = no_dbias

        with fp8_autocast(
            enabled=not do_calibration, calibrating=do_calibration, fp8_recipe=recipe
        ):
            out_ref, ln_out_ref, grad_input_ref = calc_output_and_grad_ln_out(
                layer_pd, input_tensor, grad_out, return_ln_out=return_ln_out
            )
            out, ln_out, grad_input = calc_output_and_grad_ln_out(
                layer_te, input_tensor, grad_out, return_ln_out=return_ln_out
            )

        assert_allclose(out, out_ref, rtol=rtol, atol=atol)
        if not no_dgrad:
            assert_allclose(grad_input, grad_input_ref, rtol=rtol, atol=atol)
        if not no_wgrad:
            assert_allclose(layer_te.ln_weight.grad, layer_pd.ln_weight.grad, rtol=rtol, atol=atol)
            assert_allclose(
                layer_te.fc1_weight.grad, layer_pd.fc1_weight.grad.T, rtol=rtol, atol=atol
            )
            assert_allclose(
                layer_te.fc2_weight.grad, layer_pd.fc2_weight.grad.T, rtol=rtol, atol=atol
            )
        if not no_dbias:
            if has_ln_bias:
                assert_allclose(layer_te.ln_bias.grad, layer_pd.ln_bias.grad, rtol=rtol, atol=atol)
            if has_bias:
                assert_allclose(
                    layer_te.fc1_bias.grad, layer_pd.fc1_bias.grad, rtol=rtol, atol=atol
                )
                assert_allclose(
                    layer_te.fc2_bias.grad, layer_pd.fc2_bias.grad, rtol=rtol, atol=atol
                )
        if return_ln_out:
            assert_allclose(ln_out, ln_out_ref, rtol=rtol, atol=atol)

        if do_calibration:
            assert paddle.count_nonzero(layer_te.fp8_meta["scaling_fwd"].amax_history).item() > 0

    @staticmethod
    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize("bs,hidden_size,ffn_hidden_size", LINEAR_CASES)
    @pytest.mark.parametrize("activation_dtype", ["bfloat16"])
    @pytest.mark.parametrize("num_microbatch", [8])
    def test_layernorm_mlp_fp8_microbatch(
        bs, hidden_size, ffn_hidden_size, activation_dtype, num_microbatch
    ):
        """
        Test FP8 LayerNormMLP Layer
        """
        paddle.set_default_dtype(activation_dtype)
        rtol = 1e-5
        atol = 1e-5
        eps = 1e-3

        recipe = DelayedScaling()

        layer_cached = te.LayerNormMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            eps=eps,
        )

        layer_normal = te.LayerNormMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            eps=eps,
        )
        layer_normal.ln_weight.copy_(layer_cached.ln_weight, True)
        layer_normal.ln_bias.copy_(layer_cached.ln_bias, True)
        layer_normal.fc1_weight.copy_(layer_cached.fc1_weight, True)
        layer_normal.fc2_weight.copy_(layer_cached.fc2_weight, True)
        layer_normal.fc1_bias.copy_(layer_cached.fc1_bias, True)
        layer_normal.fc2_bias.copy_(layer_cached.fc2_bias, True)

        # Calibration to make sure weight scale is the same
        input_tensor = paddle.uniform(shape=(bs, hidden_size), dtype=activation_dtype)
        with fp8_autocast(enabled=False, calibrating=True, fp8_recipe=recipe):
            _ = layer_cached(input_tensor)

        with fp8_autocast(enabled=False, calibrating=True, fp8_recipe=recipe):
            _ = layer_normal(input_tensor)

        for iteration in range(num_microbatch):
            input_tensor = paddle.uniform(shape=(bs, hidden_size), dtype=activation_dtype)
            grad_out = paddle.uniform(shape=(bs, hidden_size), dtype=activation_dtype)

            with fp8_autocast(enabled=True, fp8_recipe=recipe):
                out = layer_cached(input_tensor, is_first_microbatch=(iteration == 0))
                out.backward(grad_out)

            with fp8_autocast(enabled=True, fp8_recipe=recipe):
                out_ref = layer_normal(input_tensor)
                out_ref.backward(grad_out)

            assert_allclose(out, out_ref, rtol=rtol, atol=atol)
            assert_allclose(
                layer_cached.ln_weight.grad, layer_normal.ln_weight.grad, rtol=rtol, atol=atol
            )
            assert_allclose(
                layer_cached.fc1_weight.grad, layer_normal.fc1_weight.grad, rtol=rtol, atol=atol
            )
            assert_allclose(
                layer_cached.fc2_weight.grad, layer_normal.fc2_weight.grad, rtol=rtol, atol=atol
            )


@pytest.mark.parametrize("bs", [1, 2])
@pytest.mark.parametrize("hidden_size, num_heads", [[1024, 16]])
@pytest.mark.parametrize("q_seqlen, kv_seqlen", [[1024, 1024]])
@pytest.mark.parametrize("attn_type", ["self", "cross"])
@pytest.mark.parametrize("mask_type", ["causal", "padding"])
@pytest.mark.parametrize("math_dtype", ["bfloat16", "float16"])
def test_dot_product_attention(
    bs, hidden_size, num_heads, q_seqlen, kv_seqlen, attn_type, mask_type, math_dtype
):
    """
    Test DotProductAttention Layer
    """
    paddle.set_default_dtype(math_dtype)
    rtol = 1e-4
    atol = 2e-2
    head_size = hidden_size // num_heads

    # Skip if cuDNN fused attention is not supported
    if not is_fused_attention_supported(
        num_heads=num_heads,
        num_gqa_groups=num_heads,
        q_seqlen=q_seqlen,
        kv_seqlen=kv_seqlen,
        head_size=head_size,
        dtype=math_dtype,
        dropout=0.0,
        qkv_layout="bshd_bshd_bshd",
        bias_type="no_bias",
        mask_type=mask_type,
    ):
        pytest.skip("cuDNN fused attention is not supported")

    attn_q_input = paddle.normal(
        mean=0.0, std=0.02, shape=(bs, q_seqlen, num_heads, head_size)
    ).astype(math_dtype)
    attn_k_input = paddle.normal(
        mean=0.0, std=0.02, shape=(bs, kv_seqlen, num_heads, head_size)
    ).astype(math_dtype)
    attn_v_input = paddle.normal(
        mean=0.0, std=0.02, shape=(bs, kv_seqlen, num_heads, head_size)
    ).astype(math_dtype)

    q_actual_seqlen = paddle.randint(low=20, high=q_seqlen, shape=(bs,), dtype="int32")
    kv_actual_seqlen = (
        paddle.randint(low=20, high=kv_seqlen, shape=(bs,), dtype="int32")
        if attn_type == "cross"
        else q_actual_seqlen
    )
    attn_mask = paddle.ones(shape=(bs, 1, q_seqlen, kv_seqlen), dtype="bool")

    grad_out = paddle.normal(mean=0.0, std=0.02, shape=(bs, q_seqlen, num_heads, head_size)).astype(
        "float32"
    )
    for i in range(0, bs):
        grad_out[i, q_actual_seqlen[i] :, :, :] = 0
    grad_out = grad_out.astype(math_dtype)

    for i in range(0, bs):
        attn_mask[i, 0, 0 : q_actual_seqlen[i], 0 : kv_actual_seqlen[i]] = False

    head_size = hidden_size // num_heads
    layer_te = te.DotProductAttention(
        num_heads,
        head_size,
        attention_dropout=0.0,
        attn_mask_type=mask_type,
        attention_type=attn_type,
        backend="transformer_engine",
    )
    layer_pd = te.DotProductAttention(
        num_heads,
        head_size,
        attention_dropout=0.0,
        attn_mask_type=mask_type,
        attention_type=attn_type,
        backend="paddle",
    )

    def calc_attn_output_and_grad(layer, q, k, v, mask, dout):
        _q = paddle.to_tensor(q, stop_gradient=False)
        _k = paddle.to_tensor(k, stop_gradient=False)
        _v = paddle.to_tensor(v, stop_gradient=False)

        out = layer(_q, _k, _v, mask)
        out.backward(dout)
        return out, _q.grad, _k.grad, _v.grad

    out, q_grad, k_grad, v_grad = calc_attn_output_and_grad(
        layer_te, attn_q_input, attn_k_input, attn_v_input, attn_mask, grad_out
    )
    out_ref, q_grad_ref, k_grad_ref, v_grad_ref = calc_attn_output_and_grad(
        layer_pd, attn_q_input, attn_k_input, attn_v_input, attn_mask, grad_out
    )
    valid_out_ref = paddle.full_like(out_ref, 0)
    for i in range(0, bs):
        valid_out_ref[i, 0 : q_actual_seqlen[i], :, :] = out_ref[i, 0 : q_actual_seqlen[i], :, :]

    valid_q_grad_ref = paddle.full_like(q_grad_ref, 0)
    valid_k_grad_ref = paddle.full_like(k_grad_ref, 0)
    valid_v_grad_ref = paddle.full_like(v_grad_ref, 0)
    for i in range(0, bs):
        valid_q_grad_ref[i, 0 : q_actual_seqlen[i], :, :] = q_grad_ref[
            i, 0 : q_actual_seqlen[i], :, :
        ]
        valid_k_grad_ref[i, 0 : kv_actual_seqlen[i], :, :] = k_grad_ref[
            i, 0 : kv_actual_seqlen[i], :, :
        ]
        valid_v_grad_ref[i, 0 : kv_actual_seqlen[i], :, :] = v_grad_ref[
            i, 0 : kv_actual_seqlen[i], :, :
        ]

    assert_allclose(out, valid_out_ref, rtol=rtol, atol=atol)
    assert_allclose(q_grad, valid_q_grad_ref, rtol=rtol, atol=atol)
    assert_allclose(k_grad, valid_k_grad_ref, rtol=rtol, atol=atol)
    assert_allclose(v_grad, valid_v_grad_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("bs", [1, 2])
@pytest.mark.parametrize("num_gqa_groups", [1, 2, 4])
@pytest.mark.parametrize("hidden_size, num_heads, ffn_hidden_size", [[256, 4, 1024]])
@pytest.mark.parametrize("q_seqlen, kv_seqlen", [[1024, 1024]])
@pytest.mark.parametrize("has_bias, no_dbias", [[False, True], [True, True], [True, False]])
@pytest.mark.parametrize("no_wgrad", [True, False])
@pytest.mark.parametrize("mask_type", ["causal", "padding"])
@pytest.mark.parametrize("math_dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("output_layernorm", [True, False])
@pytest.mark.parametrize("return_layernorm_output", [True, False])
@pytest.mark.parametrize("normalization", ["RMSNorm", "LayerNorm"])
def test_transformer_encoder_layer(
    bs,
    hidden_size,
    num_heads,
    num_gqa_groups,
    ffn_hidden_size,
    has_bias,
    no_dbias,
    no_wgrad,
    q_seqlen,
    kv_seqlen,
    mask_type,
    math_dtype,
    output_layernorm,
    return_layernorm_output,
    normalization,
):
    """
    Test Transformer Encoder Layer
    """
    paddle.set_default_dtype(math_dtype)
    rtol = 5e-2
    atol = 5e-2
    eps = 1e-3
    has_ln_bias = normalization == "LayerNorm"

    # Skip if cuDNN fused attention is not supported
    if not is_fused_attention_supported(
        num_heads=num_heads,
        num_gqa_groups=num_gqa_groups,
        q_seqlen=q_seqlen,
        kv_seqlen=kv_seqlen,
        head_size=hidden_size // num_heads,
        dtype=math_dtype,
        dropout=0.0,
        qkv_layout="bshd_bshd_bshd",
        bias_type="no_bias",
        mask_type=mask_type,
    ):
        pytest.skip("cuDNN fused attention is not supported")

    encoder_input = paddle.uniform(shape=(bs, q_seqlen, hidden_size), dtype=math_dtype)

    q_actual_seqlen = paddle.ones(shape=(bs,), dtype="int32") * q_seqlen
    kv_actual_seqlen = q_actual_seqlen
    attn_mask = paddle.ones(shape=(bs, 1, q_seqlen, kv_seqlen), dtype="bool")

    grad_out = paddle.normal(mean=0.0, std=0.02, shape=(bs, q_seqlen, hidden_size)).astype(
        "float32"
    )
    for i in range(0, bs):
        grad_out[i, q_actual_seqlen[i] :, :] = 0
    grad_out = grad_out.astype(math_dtype)

    for i in range(0, bs):
        attn_mask[i, 0, 0 : q_actual_seqlen[i], 0 : kv_actual_seqlen[i]] = False

    layer_te = te.TransformerLayer(
        hidden_size,
        ffn_hidden_size,
        num_heads,
        num_gqa_groups=num_gqa_groups,
        layernorm_epsilon=eps,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        weight_attr=None,
        bias_attr=None if has_bias else False,
        self_attn_mask_type=mask_type,
        apply_residual_connection_post_layernorm=return_layernorm_output,
        output_layernorm=output_layernorm,
        layer_type="encoder",
        normalization=normalization,
        backend="transformer_engine",
    )
    layer_pd = te.TransformerLayer(
        hidden_size,
        ffn_hidden_size,
        num_heads,
        num_gqa_groups=num_gqa_groups,
        layernorm_epsilon=eps,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        weight_attr=None,
        bias_attr=None if has_bias else False,
        self_attn_mask_type=mask_type,
        apply_residual_connection_post_layernorm=return_layernorm_output,
        output_layernorm=output_layernorm,
        layer_type="encoder",
        normalization=normalization,
        backend="paddle",
    )

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
            layer_te.self_attention.layernorm_qkv.ln_weight, True
        )
        layer_pd.self_attention.layernorm_qkv.weight.copy_(
            layer_te.self_attention.layernorm_qkv.weight.T, True
        )
        layer_pd.self_attention.layernorm_qkv.ln_weight.stop_gradient = no_wgrad
        layer_pd.self_attention.layernorm_qkv.weight.stop_gradient = no_wgrad
        layer_te.self_attention.layernorm_qkv.ln_weight.stop_gradient = no_wgrad
        layer_te.self_attention.layernorm_qkv.weight.stop_gradient = no_wgrad
        if has_ln_bias:
            layer_pd.self_attention.layernorm_qkv.ln_bias.copy_(
                layer_te.self_attention.layernorm_qkv.ln_bias, True
            )
            layer_pd.self_attention.layernorm_qkv.ln_bias.stop_gradient = no_dbias
            layer_te.self_attention.layernorm_qkv.ln_bias.stop_gradient = no_dbias
        if has_bias:
            layer_pd.self_attention.layernorm_qkv.bias.copy_(
                layer_te.self_attention.layernorm_qkv.bias, True
            )
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
    layer_pd.layernorm_mlp.fc1_weight.copy_(layer_te.layernorm_mlp.fc1_weight.T, True)
    layer_pd.layernorm_mlp.fc2_weight.copy_(layer_te.layernorm_mlp.fc2_weight.T, True)
    layer_pd.layernorm_mlp.ln_weight.stop_gradient = no_wgrad
    layer_pd.layernorm_mlp.fc1_weight.stop_gradient = no_wgrad
    layer_pd.layernorm_mlp.fc2_weight.stop_gradient = no_wgrad
    layer_te.layernorm_mlp.ln_weight.stop_gradient = no_wgrad
    layer_te.layernorm_mlp.fc1_weight.stop_gradient = no_wgrad
    layer_te.layernorm_mlp.fc2_weight.stop_gradient = no_wgrad
    if has_ln_bias:
        layer_pd.layernorm_mlp.ln_bias.copy_(layer_te.layernorm_mlp.ln_bias, True)
        layer_pd.layernorm_mlp.ln_bias.stop_gradient = no_dbias
        layer_te.layernorm_mlp.ln_bias.stop_gradient = no_dbias
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

    out_ref, grad_input_ref = calc_transformer_output_and_grad(
        layer_pd, encoder_input, attn_mask, grad_out
    )
    out, grad_input = calc_transformer_output_and_grad(layer_te, encoder_input, attn_mask, grad_out)

    assert_allclose(out, out_ref, rtol=rtol, atol=atol)
    assert_allclose(grad_input, grad_input_ref, rtol=rtol, atol=atol)
    if not no_wgrad:
        if output_layernorm:
            assert_allclose(
                layer_te.self_attention.qkv.weight.grad,
                layer_pd.self_attention.qkv.weight.grad.T,
                rtol=rtol,
                atol=atol,
            )
        else:
            assert_allclose(
                layer_te.self_attention.layernorm_qkv.weight.grad,
                layer_pd.self_attention.layernorm_qkv.weight.grad.T,
                rtol=rtol,
                atol=atol,
            )
    if not no_dbias:
        if output_layernorm:
            assert_allclose(
                layer_te.self_attention.qkv.bias.grad,
                layer_pd.self_attention.qkv.bias.grad,
                rtol=0.01,
                atol=0.5,
            )
        else:
            assert_allclose(
                layer_te.self_attention.layernorm_qkv.bias.grad,
                layer_pd.self_attention.layernorm_qkv.bias.grad,
                rtol=0.01,
                atol=0.5,
            )


@pytest.mark.parametrize("bs", [1, 2])
@pytest.mark.parametrize("num_gqa_groups", [1, 2, 4])
@pytest.mark.parametrize("hidden_size, num_heads, ffn_hidden_size", [[256, 4, 1024]])
@pytest.mark.parametrize("q_seqlen, kv_seqlen", [[1024, 1024]])
@pytest.mark.parametrize("has_bias, no_dbias", [[False, True], [True, True], [True, False]])
@pytest.mark.parametrize("no_wgrad", [True, False])
@pytest.mark.parametrize("mask_type", ["causal", "padding"])
@pytest.mark.parametrize("math_dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("output_layernorm", [True, False])
@pytest.mark.parametrize("return_layernorm_output", [True, False])
@pytest.mark.parametrize("recompute_core_attention", [True, False])
@pytest.mark.parametrize("normalization", ["RMSNorm", "LayerNorm"])
def test_transformer_decoder_layer(
    bs,
    hidden_size,
    num_heads,
    num_gqa_groups,
    ffn_hidden_size,
    has_bias,
    no_dbias,
    no_wgrad,
    q_seqlen,
    kv_seqlen,
    mask_type,
    math_dtype,
    output_layernorm,
    return_layernorm_output,
    recompute_core_attention,
    normalization,
):
    """
    Test Transformer Decoder Layer
    """
    paddle.set_default_dtype(math_dtype)
    rtol = 5e-2
    atol = 6e-2
    eps = 1e-3
    has_ln_bias = normalization == "LayerNorm"

    # Skip if cuDNN fused attention is not supported
    if not is_fused_attention_supported(
        num_heads=num_heads,
        num_gqa_groups=num_gqa_groups,
        q_seqlen=q_seqlen,
        kv_seqlen=kv_seqlen,
        head_size=hidden_size // num_heads,
        dtype=math_dtype,
        dropout=0.0,
        qkv_layout="bshd_bshd_bshd",
        bias_type="no_bias",
        mask_type=mask_type,
    ):
        pytest.skip("cuDNN fused attention is not supported")

    encoder_input = paddle.normal(mean=0.0, std=0.1, shape=(bs, q_seqlen, hidden_size)).astype(
        math_dtype
    )
    encoder_output = paddle.normal(mean=0.0, std=0.1, shape=(bs, kv_seqlen, hidden_size)).astype(
        math_dtype
    )

    q_actual_seqlen = paddle.ones(shape=(bs,), dtype="int32") * q_seqlen
    kv_actual_seqlen = q_actual_seqlen
    attn_mask = paddle.ones(shape=(bs, 1, q_seqlen, kv_seqlen), dtype="bool")

    grad_out = paddle.normal(mean=0.0, std=0.01, shape=(bs, q_seqlen, hidden_size)).astype(
        "float32"
    )

    # rounding to avoid numerical issues
    encoder_input = paddle.round(encoder_input * 1000) / 1000
    encoder_output = paddle.round(encoder_output * 1000) / 1000
    grad_out = paddle.round(grad_out * 1000) / 1000

    for i in range(0, bs):
        grad_out[i, q_actual_seqlen[i] :, :] = 0
    grad_out = grad_out.astype(math_dtype)

    for i in range(0, bs):
        attn_mask[i, 0, 0 : q_actual_seqlen[i], 0 : kv_actual_seqlen[i]] = False

    layer_te = te.TransformerLayer(
        hidden_size,
        ffn_hidden_size,
        num_heads,
        num_gqa_groups=num_gqa_groups,
        layernorm_epsilon=eps,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        weight_attr=None,
        bias_attr=None if has_bias else False,
        self_attn_mask_type=mask_type,
        apply_residual_connection_post_layernorm=return_layernorm_output,
        output_layernorm=output_layernorm,
        layer_type="decoder",
        normalization=normalization,
        backend="transformer_engine",
    )
    layer_pd = te.TransformerLayer(
        hidden_size,
        ffn_hidden_size,
        num_heads,
        num_gqa_groups=num_gqa_groups,
        layernorm_epsilon=eps,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        weight_attr=None,
        bias_attr=None if has_bias else False,
        self_attn_mask_type=mask_type,
        apply_residual_connection_post_layernorm=return_layernorm_output,
        output_layernorm=output_layernorm,
        layer_type="decoder",
        normalization=normalization,
        backend="paddle",
    )

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
            layer_te.self_attention.layernorm_qkv.ln_weight, True
        )
        layer_pd.self_attention.layernorm_qkv.weight.copy_(
            layer_te.self_attention.layernorm_qkv.weight.T, True
        )
        layer_pd.self_attention.layernorm_qkv.ln_weight.stop_gradient = no_wgrad
        layer_pd.self_attention.layernorm_qkv.weight.stop_gradient = no_wgrad
        layer_te.self_attention.layernorm_qkv.ln_weight.stop_gradient = no_wgrad
        layer_te.self_attention.layernorm_qkv.weight.stop_gradient = no_wgrad
        if has_ln_bias:
            layer_pd.self_attention.layernorm_qkv.ln_bias.copy_(
                layer_te.self_attention.layernorm_qkv.ln_bias, True
            )
            layer_pd.self_attention.layernorm_qkv.ln_bias.stop_gradient = no_dbias
            layer_te.self_attention.layernorm_qkv.ln_bias.stop_gradient = no_dbias
        if has_bias:
            layer_pd.self_attention.layernorm_qkv.bias.copy_(
                layer_te.self_attention.layernorm_qkv.bias, True
            )
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
        layer_te.inter_attention.layernorm_query.ln_weight, True
    )
    layer_pd.inter_attention.layernorm_query.weight.copy_(
        layer_te.inter_attention.layernorm_query.weight.T, True
    )
    layer_pd.inter_attention.layernorm_query.ln_weight.stop_gradient = no_wgrad
    layer_pd.inter_attention.layernorm_query.weight.stop_gradient = no_wgrad
    layer_te.inter_attention.layernorm_query.ln_weight.stop_gradient = no_wgrad
    layer_te.inter_attention.layernorm_query.weight.stop_gradient = no_wgrad
    if has_ln_bias:
        layer_pd.inter_attention.layernorm_query.ln_bias.copy_(
            layer_te.inter_attention.layernorm_query.ln_bias, True
        )
        layer_pd.inter_attention.layernorm_query.ln_bias.stop_gradient = no_dbias
        layer_te.inter_attention.layernorm_query.ln_bias.stop_gradient = no_dbias
    if has_bias:
        layer_pd.inter_attention.layernorm_query.bias.copy_(
            layer_te.inter_attention.layernorm_query.bias, True
        )
        layer_pd.inter_attention.layernorm_query.bias.stop_gradient = no_dbias
        layer_te.inter_attention.layernorm_query.bias.stop_gradient = no_dbias

    layer_pd.inter_attention.key_value.weight.copy_(
        layer_te.inter_attention.key_value.weight.T, True
    )
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
    layer_pd.layernorm_mlp.fc1_weight.copy_(layer_te.layernorm_mlp.fc1_weight.T, True)
    layer_pd.layernorm_mlp.fc2_weight.copy_(layer_te.layernorm_mlp.fc2_weight.T, True)
    layer_pd.layernorm_mlp.ln_weight.stop_gradient = no_wgrad
    layer_pd.layernorm_mlp.fc1_weight.stop_gradient = no_wgrad
    layer_pd.layernorm_mlp.fc2_weight.stop_gradient = no_wgrad
    layer_te.layernorm_mlp.ln_weight.stop_gradient = no_wgrad
    layer_te.layernorm_mlp.fc1_weight.stop_gradient = no_wgrad
    layer_te.layernorm_mlp.fc2_weight.stop_gradient = no_wgrad
    if has_ln_bias:
        layer_pd.layernorm_mlp.ln_bias.copy_(layer_te.layernorm_mlp.ln_bias, True)
        layer_pd.layernorm_mlp.ln_bias.stop_gradient = no_dbias
        layer_te.layernorm_mlp.ln_bias.stop_gradient = no_dbias
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

    def calc_transformer_output_and_grad(
        layer,
        encoder_input,
        mask,
        encoder_output,
        enc_dec_attn_mask,
        dout,
        recompute_core_attention=False,
    ):
        _encoder_input = paddle.to_tensor(encoder_input, stop_gradient=False)
        _encoder_output = paddle.to_tensor(encoder_output, stop_gradient=False)
        out = layer(
            _encoder_input,
            mask,
            _encoder_output,
            enc_dec_attn_mask,
            recompute_core_attention=recompute_core_attention,
        )
        out.backward(dout)
        return out, _encoder_input.grad, _encoder_output.grad

    out_ref, grad_encoder_input_ref, grad_encoder_output_ref = calc_transformer_output_and_grad(
        layer_pd, encoder_input, attn_mask, encoder_output, attn_mask, grad_out
    )
    out, grad_encoder_input, grad_encoder_output = calc_transformer_output_and_grad(
        layer_te,
        encoder_input,
        attn_mask,
        encoder_output,
        attn_mask,
        grad_out,
        recompute_core_attention=recompute_core_attention,
    )

    assert_allclose(out, out_ref, rtol=rtol, atol=atol)
    assert_allclose(grad_encoder_input, grad_encoder_input_ref, rtol=rtol, atol=atol)
    assert_allclose(grad_encoder_output, grad_encoder_output_ref, rtol=rtol, atol=atol)
    if not no_wgrad:
        if output_layernorm:
            assert_allclose(
                layer_te.self_attention.qkv.weight.grad,
                layer_pd.self_attention.qkv.weight.grad.T,
                rtol=rtol,
                atol=atol,
            )
        else:
            assert_allclose(
                layer_te.self_attention.layernorm_qkv.weight.grad,
                layer_pd.self_attention.layernorm_qkv.weight.grad.T,
                rtol=rtol,
                atol=atol,
            )
            assert_allclose(
                layer_te.inter_attention.layernorm_query.weight.grad,
                layer_pd.inter_attention.layernorm_query.weight.grad.T,
                rtol=rtol,
                atol=atol,
            )
    if not no_dbias:
        if output_layernorm:
            assert_allclose(
                layer_te.self_attention.qkv.bias.grad,
                layer_pd.self_attention.qkv.bias.grad,
                rtol=0.5,
                atol=0.6,
            )
        else:
            assert_allclose(
                layer_te.self_attention.layernorm_qkv.bias.grad,
                layer_pd.self_attention.layernorm_qkv.bias.grad,
                rtol=0.01,
                atol=0.5,
            )
            assert_allclose(
                layer_te.inter_attention.layernorm_query.bias.grad,
                layer_pd.inter_attention.layernorm_query.bias.grad,
                rtol=rtol,
                atol=atol,
            )


@pytest.mark.skipif(not is_fp8_supported, reason=reason)
@pytest.mark.parametrize("bs", [8])
@pytest.mark.parametrize("hidden_size, num_heads, ffn_hidden_size", [[1024, 16, 4096]])
@pytest.mark.parametrize("q_seqlen, kv_seqlen", [[128, 128]])
@pytest.mark.parametrize("mask_type", ["causal"])
@pytest.mark.parametrize("math_dtype", ["bfloat16"])
@pytest.mark.parametrize("num_microbatch", [8])
def test_transformer_encoder_layer_microbatch(
    bs,
    hidden_size,
    num_heads,
    ffn_hidden_size,
    q_seqlen,
    kv_seqlen,
    mask_type,
    math_dtype,
    num_microbatch,
):
    """
    Test Transformer Encoder Layer with FP8 weight caching
    """
    paddle.set_default_dtype(math_dtype)
    rtol = 1e-5
    atol = 1e-5
    eps = 1e-3

    # Skip if cuDNN fused attention is not supported
    if not is_fused_attention_supported(
        num_heads=num_heads,
        num_gqa_groups=num_heads,
        q_seqlen=q_seqlen,
        kv_seqlen=kv_seqlen,
        head_size=hidden_size // num_heads,
        dtype=math_dtype,
        dropout=0.0,
        qkv_layout="bs3hd",
        bias_type="no_bias",
        mask_type=mask_type,
    ):
        pytest.skip("cuDNN fused attention is not supported")

    layer_cached = te.TransformerLayer(
        hidden_size,
        ffn_hidden_size,
        num_heads,
        layernorm_epsilon=eps,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        weight_attr=None,
        bias_attr=None,
        self_attn_mask_type=mask_type,
        layer_type="encoder",
    )
    layer_normal = te.TransformerLayer(
        hidden_size,
        ffn_hidden_size,
        num_heads,
        layernorm_epsilon=eps,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        weight_attr=None,
        bias_attr=None,
        self_attn_mask_type=mask_type,
        layer_type="encoder",
    )

    layer_normal.self_attention.layernorm_qkv.ln_weight.copy_(
        layer_cached.self_attention.layernorm_qkv.ln_weight, True
    )
    layer_normal.self_attention.layernorm_qkv.ln_bias.copy_(
        layer_cached.self_attention.layernorm_qkv.ln_bias, True
    )
    layer_normal.self_attention.layernorm_qkv.weight.copy_(
        layer_cached.self_attention.layernorm_qkv.weight, True
    )
    layer_normal.self_attention.layernorm_qkv.bias.copy_(
        layer_cached.self_attention.layernorm_qkv.bias, True
    )

    layer_normal.self_attention.proj.weight.copy_(layer_cached.self_attention.proj.weight, True)
    layer_normal.self_attention.proj.bias.copy_(layer_cached.self_attention.proj.bias, True)

    # LayerNorm MLP params
    layer_normal.layernorm_mlp.ln_weight.copy_(layer_cached.layernorm_mlp.ln_weight, True)
    layer_normal.layernorm_mlp.ln_bias.copy_(layer_cached.layernorm_mlp.ln_bias, True)
    layer_normal.layernorm_mlp.fc1_weight.copy_(layer_cached.layernorm_mlp.fc1_weight, True)
    layer_normal.layernorm_mlp.fc2_weight.copy_(layer_cached.layernorm_mlp.fc2_weight, True)
    layer_normal.layernorm_mlp.fc1_bias.copy_(layer_cached.layernorm_mlp.fc1_bias, True)
    layer_normal.layernorm_mlp.fc2_bias.copy_(layer_cached.layernorm_mlp.fc2_bias, True)

    recipe = DelayedScaling()

    def generate_input():
        encoder_input = paddle.uniform(shape=(bs, q_seqlen, hidden_size), dtype=math_dtype)

        q_actual_seqlen = paddle.ones(shape=(bs,), dtype="int32") * q_seqlen
        kv_actual_seqlen = q_actual_seqlen
        attn_mask = paddle.ones(shape=(bs, 1, q_seqlen, kv_seqlen), dtype="bool")

        grad_out = paddle.normal(mean=0.0, std=0.02, shape=(bs, q_seqlen, hidden_size)).astype(
            "float32"
        )
        for i in range(0, bs):
            grad_out[i, q_actual_seqlen[i] :, :] = 0
        grad_out = grad_out.astype(math_dtype)

        for i in range(0, bs):
            attn_mask[i, 0, 0 : q_actual_seqlen[i], 0 : kv_actual_seqlen[i]] = False

        return encoder_input, attn_mask, grad_out

    # Calibration to make sure weight scale is the same
    encoder_input, mask, _ = generate_input()
    with fp8_autocast(enabled=False, calibrating=True, fp8_recipe=recipe):
        _ = layer_cached(encoder_input, mask)

    with fp8_autocast(enabled=False, calibrating=True, fp8_recipe=recipe):
        _ = layer_normal(encoder_input, mask)

    for iteration in range(num_microbatch):
        encoder_input, mask, grad_out = generate_input()

        with fp8_autocast(enabled=True, fp8_recipe=recipe):
            out = layer_cached(encoder_input, mask, is_first_microbatch=(iteration == 0))
            out.backward(grad_out)

        with fp8_autocast(enabled=True, fp8_recipe=recipe):
            out_ref = layer_normal(encoder_input, mask)
            out_ref.backward(grad_out)

        assert_allclose(out, out_ref, rtol=rtol, atol=atol)
        assert_allclose(
            layer_cached.self_attention.layernorm_qkv.weight.grad,
            layer_normal.self_attention.layernorm_qkv.weight.grad,
            rtol=rtol,
            atol=atol,
        )
