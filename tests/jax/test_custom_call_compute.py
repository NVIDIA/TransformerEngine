# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import jit, value_and_grad
from functools import reduce
from typing import Union
import operator

from utils import (
    assert_allclose,
    assert_tree_like_allclose,
    pytest_parametrize_wrapper,
)
from transformer_engine.jax.layernorm import layernorm
from transformer_engine.jax.layernorm_mlp import layernorm_mlp

from transformer_engine.jax.cpp_extensions.activation import _jax_act_lu, _jax_quantize_dact_dbias
from transformer_engine.jax.cpp_extensions.normalization import (
    _jax_layernorm,
    _jax_rmsnorm,
    is_norm_zero_centered_gamma_in_weight_dtype,
)
from transformer_engine.jax.cpp_extensions.quantization import (
    _jax_quantize,
    _jax_quantize_dbias,
)
from transformer_engine.jax.cpp_extensions.misc import get_cudnn_version
from transformer_engine.jax import cpp_extensions as tex
from transformer_engine.jax.quantize import (
    DelayedScaleQuantizer,
    ScaledTensor,
    ScaledTensor1x,
    ScaledTensor2x,
    GroupedScaledTensor1x,
    ScalingMode,
    QuantizerFactory,
    QuantizeLayout,
)
from transformer_engine.jax.quantize import helper
from transformer_engine.jax.activation import activation
from transformer_engine.jax.dense import dense
from transformer_engine.jax.layernorm_dense import layernorm_dense

GEMM_CASES = [
    (256, 256, 512),
    (32, 32, 32),
    (2048, 1024, 2048),
    (2048, 2048, 1024),
    (2048, 1024, 1024),
]
FP8_COMPUTE_TYPE = [jnp.float8_e4m3fn, jnp.float8_e5m2]
LN_CASES = [(256, 128), (128, 256)]
DTYPES = [jnp.bfloat16, jnp.float32]
is_fp8_supported, reason = helper.is_fp8_available()
is_mxfp8_supported, reason = helper.is_fp8_available(ScalingMode.MXFP8_1D_SCALING)

supported_scaling_modes = []
""" Find supported scaling modes"""
if is_fp8_supported:
    supported_scaling_modes.append(ScalingMode.DELAYED_TENSOR_SCALING)
    supported_scaling_modes.append(ScalingMode.CURRENT_TENSOR_SCALING)
if is_mxfp8_supported:
    supported_scaling_modes.append(ScalingMode.MXFP8_1D_SCALING)


def is_shape_supported_by_mxfp8(input_shape):
    try:
        if isinstance(input_shape, type(pytest.param(0))):
            input_shape = input_shape.values[0]
        ScalingMode.MXFP8_1D_SCALING.get_scale_shape_2x(input_shape)
        return True
    except:
        # get_scale_shapes will raise an exception if the shape is not supported
        return False


def assert_bitwise_scaled_tensors(a: ScaledTensor, b: ScaledTensor):
    if isinstance(a, ScaledTensor1x) and isinstance(b, ScaledTensor1x):
        assert a.scaling_mode == b.scaling_mode
        assert a.scale_inv.dtype == b.scale_inv.dtype
        if a.scaling_mode.is_tensor_scaling():
            # Assert in dq_dtype as some unfused codepaths have an intermediate cast
            # to an input dtype which reduces precision compared to everything in fp32
            assert_allclose(a.scale_inv, b.scale_inv, dtype=a.dq_dtype)
        elif a.scaling_mode == ScalingMode.MXFP8_1D_SCALING:
            # Compare MXFP8 scales as uint8
            assert_allclose(a.scale_inv.astype(jnp.uint8), b.scale_inv.astype(jnp.uint8))
        else:
            raise ValueError(f"Unsupported scaling mode {a.scaling_mode}")
        assert_allclose(a.data, b.data)

    elif isinstance(a, ScaledTensor2x) and isinstance(b, ScaledTensor2x):
        assert_bitwise_scaled_tensors(a.rowwise_tensor, b.rowwise_tensor)
        assert_bitwise_scaled_tensors(a.colwise_tensor, b.colwise_tensor)
    else:
        pytest.fail("Unsupported input types")


def assert_dequantized_scaled_tensor(a: ScaledTensor, b: jnp.ndarray):
    if isinstance(a, ScaledTensor1x):
        if a.data_layout == "T":
            flatten_axis = a.data.ndim - a.flatten_axis
            b_transpose = jnp.transpose(b, (*range(flatten_axis, b.ndim), *range(flatten_axis)))
            assert_allclose(a.dequantize(), b_transpose, dtype=a.data.dtype)
        else:
            assert_allclose(a.dequantize(), b, dtype=a.data.dtype)
    elif isinstance(a, ScaledTensor2x):
        assert_dequantized_scaled_tensor(a.get_rowwise_tensor(), b)
        assert_dequantized_scaled_tensor(a.get_colwise_tensor(), b)
    else:
        pytest.fail("a must be a ScaledTensor object")


def assert_dequantized_grouped_scaled_tensor(
    a: Union[GroupedScaledTensor1x, ScaledTensor2x], b: jnp.ndarray
):
    if isinstance(a, GroupedScaledTensor1x):
        assert a.group_sizes.sum() == b.shape[0]
        b = jnp.split(b, jnp.cumulative_sum(a.group_sizes)[:-1], axis=0)
        dq_a = a.dequantize()
        for dq_a_i, b_i in zip(dq_a, b):
            if a.data_layout == "T":
                data_ndim = len(a.original_shape)
                flatten_axis = a.flatten_axis
                if b_i.shape[0] == 1:
                    b_i = jnp.transpose(b_i, (0, *range(flatten_axis, data_ndim), *range(1, flatten_axis)))
                else:
                    b_i = jnp.transpose(b_i, (*range(flatten_axis, data_ndim), *range(flatten_axis)))
            dq_a_i = dq_a_i.reshape(b_i.shape)
            assert_allclose(dq_a_i, b_i, dtype=a.data.dtype)
    elif isinstance(a, ScaledTensor2x):
        assert isinstance(a.get_rowwise_tensor(), GroupedScaledTensor1x)
        assert isinstance(a.get_colwise_tensor(), GroupedScaledTensor1x)
        assert_dequantized_grouped_scaled_tensor(a.get_rowwise_tensor(), b)
        assert_dequantized_grouped_scaled_tensor(a.get_colwise_tensor(), b)
    else:
        pytest.fail("a must be a GroupedScaledTensor object")


ALL_ACTIVATION_SHAPES = [(32, 64), (16, 128, 256)]
ALL_ACTIVATION_TYPES = [
    ("gelu",),
    ("gelu", "linear"),
    ("silu",),
    ("silu", "linear"),
    ("relu",),
    ("relu", "linear"),
    ("quick_gelu",),
    ("quick_gelu", "linear"),
    ("squared_relu",),
    ("squared_relu", "linear"),
]

ACTIVATION_TYPES = {
    "L0": [
        ("gelu",),
        ("gelu", "linear"),
    ],
    "L2": ALL_ACTIVATION_TYPES,
}


class TestActivation:
    def ref_act(self, x, activation_type):
        return _jax_act_lu(x, activation_type)

    def value_n_grad_ref_func(self, x, activation_type):
        jitted_reference = jit(
            value_and_grad(lambda out: jnp.mean(self.ref_act(out, activation_type)), (0,))
        )
        return jitted_reference(x)

    def primitive_func(self, inputs, activation_type, quantizer):
        out = activation(inputs, activation_type=activation_type, quantizer=quantizer)
        return jnp.mean(out)

    @pytest_parametrize_wrapper("shape", ALL_ACTIVATION_SHAPES)
    @pytest_parametrize_wrapper(
        "activation_type",
        (
            ALL_ACTIVATION_TYPES  # Test all activation types for this test to ensure all are functional, then just test a subset for the other tests to verify other functionality
        ),
    )
    def test_act_grad(self, shape, activation_type):
        key = jax.random.PRNGKey(0)
        x = jax.random.uniform(key, shape, jnp.float32)
        x = jnp.expand_dims(x, axis=-2)
        x = jnp.repeat(x, len(activation_type), axis=-2)

        value_n_grad_primitive_func = jit(
            value_and_grad(self.primitive_func, (0,)), static_argnums=(1,)
        )

        prim_out, (prim_grad,) = value_n_grad_primitive_func(x, activation_type, None)
        ref_out, (ref_grad,) = self.value_n_grad_ref_func(x, activation_type)

        assert_allclose(prim_out, ref_out, dtype=x.dtype)
        assert_allclose(prim_grad, ref_grad, dtype=x.dtype)

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest_parametrize_wrapper("shape", ALL_ACTIVATION_SHAPES)
    @pytest_parametrize_wrapper("activation_type", ACTIVATION_TYPES)
    @pytest_parametrize_wrapper("output_type", [jnp.float8_e4m3fn, jnp.float8_e5m2])
    @pytest_parametrize_wrapper(
        "scaling_mode", [ScalingMode.DELAYED_TENSOR_SCALING, ScalingMode.CURRENT_TENSOR_SCALING]
    )
    def test_act_grad_with_tensor_scaling_fp8(
        self, random_inputs, activation_type, output_type, scaling_mode
    ):
        x = random_inputs
        x = jnp.expand_dims(x, axis=-2)
        x = jnp.repeat(x, len(activation_type), axis=-2)
        self.activation_type = activation_type

        value_n_grad_primitive_func = jit(
            value_and_grad(self.primitive_func, (0,)), static_argnums=(1,)
        )

        quantizer = QuantizerFactory.create(
            scaling_mode=scaling_mode,
            q_dtype=output_type,
            q_layout=QuantizeLayout.ROWWISE,
        )

        prim_out, (prim_grad,) = value_n_grad_primitive_func(x, activation_type, quantizer)
        ref_out, (ref_grad,) = self.value_n_grad_ref_func(x, activation_type)

        assert_allclose(prim_out, ref_out, dtype=output_type)
        assert_allclose(prim_grad, ref_grad, dtype=output_type)

    @pytest.mark.skipif(not is_mxfp8_supported, reason=reason)
    @pytest_parametrize_wrapper("shape", ALL_ACTIVATION_SHAPES)
    @pytest_parametrize_wrapper("activation_type", ACTIVATION_TYPES)
    @pytest_parametrize_wrapper("output_type", [jnp.float8_e4m3fn, jnp.float8_e5m2])
    @pytest_parametrize_wrapper(
        "q_layout", [QuantizeLayout.ROWWISE, QuantizeLayout.ROWWISE_COLWISE]
    )
    @pytest_parametrize_wrapper(
        "scaling_mode", [ScalingMode.DELAYED_TENSOR_SCALING, ScalingMode.CURRENT_TENSOR_SCALING]
    )
    def test_act_forward_with_tensor_scaling_fp8(
        self, random_inputs, activation_type, output_type, q_layout, scaling_mode
    ):
        x = random_inputs
        x = jnp.expand_dims(x, axis=-2)
        x = jnp.repeat(x, len(activation_type), axis=-2)
        self.activation_type = activation_type

        te_quantizer, jax_quantizer = QuantizerFactory.create(
            n_quantizers=2,
            scaling_mode=scaling_mode,
            q_dtype=output_type,
            q_layout=q_layout,
        )

        te_output = tex.act_lu(x, activation_type, te_quantizer)
        jax_output = _jax_act_lu(x, activation_type, jax_quantizer)

        assert_bitwise_scaled_tensors(te_output, jax_output)

    @pytest.mark.skipif(not is_mxfp8_supported, reason=reason)
    @pytest_parametrize_wrapper("shape", [(2, 64, 1, 256)])
    @pytest_parametrize_wrapper("activation_type", ACTIVATION_TYPES)
    @pytest_parametrize_wrapper("output_type", [jnp.float8_e4m3fn, jnp.float8_e5m2])
    @pytest_parametrize_wrapper(
        "q_layout", [QuantizeLayout.ROWWISE, QuantizeLayout.ROWWISE_COLWISE]
    )
    def test_act_forward_with_block_scaling_fp8(
        self, random_inputs, activation_type, output_type, q_layout
    ):
        x = random_inputs
        x = jnp.repeat(x, len(activation_type), axis=-2)
        self.activation_type = activation_type

        quantizer = QuantizerFactory.create(
            scaling_mode=ScalingMode.MXFP8_1D_SCALING, q_dtype=output_type, q_layout=q_layout
        )

        output = tex.act_lu(x, activation_type, quantizer)
        ref_out = self.ref_act(x, activation_type)

        assert_dequantized_scaled_tensor(output, ref_out)


NORM_OUTPUT_DTYPES = {
    "L0": [jnp.float8_e4m3fn],
    "L2": [jnp.float8_e4m3fn, jnp.float8_e5m2],
}


@pytest_parametrize_wrapper("n, hidden", LN_CASES)
@pytest_parametrize_wrapper("inp_dtype", DTYPES)
@pytest_parametrize_wrapper("norm_type", ["layernorm", "rmsnorm"])
@pytest_parametrize_wrapper(
    "zero_centered_gamma",
    [
        pytest.param(True, id="zero_centered"),
        pytest.param(False, id="no_zero_centered"),
    ],
)
@pytest_parametrize_wrapper("epsilon", [1e-2, 1e-6])
class TestNorm:
    """
    Test transformer_engine.jax.layernorm APIs
    """

    def _test_norm_grad(
        self, n, hidden, norm_type, zero_centered_gamma, epsilon, inp_dtype, quantizer
    ):
        def compute_loss(x):
            # Higher precision to compute the loss
            x_ = x.astype(jnp.float32)
            return jnp.mean(jnp.square(x_)).astype(x.dtype)

        def reference_func(x, gamma, beta, norm_type, zero_centered_gamma, eps, quantizer):
            if norm_type == "rmsnorm":
                ln_out, _ = _jax_rmsnorm(x, gamma, zero_centered_gamma, eps, quantizer)
            else:
                ln_out, _, _ = _jax_layernorm(x, gamma, beta, zero_centered_gamma, eps, quantizer)
            # if isinstance(ln_out, ScaledTensor):
            #     ln_out = ln_out.dequantize()
            return ln_out

        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 3)

        x = jax.random.uniform(subkeys[0], (n, hidden), jnp.float32, -1, 1)
        x = x.astype(inp_dtype)
        gamma_range = (-1, 1) if zero_centered_gamma else (0, 2)
        gamma = jax.random.uniform(subkeys[1], (hidden,), jnp.float32, *gamma_range)
        gamma = jnp.asarray(gamma, inp_dtype)
        if norm_type == "layernorm":
            beta = jax.random.uniform(subkeys[2], (hidden,), jnp.float32, -1, 1)
            beta = jnp.asarray(beta, inp_dtype)
        else:
            beta = None

        jitted_reference = jit(
            value_and_grad(
                lambda x, gamma, beta: compute_loss(
                    reference_func(
                        x, gamma, beta, norm_type, zero_centered_gamma, epsilon, quantizer=None
                    )
                ),
                (0, 1, 2),
            )
        )
        jitted_primitive = jit(
            value_and_grad(
                lambda x, gamma, beta: compute_loss(
                    layernorm(x, gamma, beta, norm_type, zero_centered_gamma, epsilon, quantizer)
                ),
                (0, 1, 2),
            )
        )

        reference_out, (reference_dx, reference_dgamma, reference_dbeta) = jitted_reference(
            x, gamma, beta
        )
        primitive_out, (primitive_dx, primitive_dgamma, primitive_dbeta) = jitted_primitive(
            x, gamma, beta
        )

        out_dtype = inp_dtype if quantizer is None else quantizer.q_dtype
        assert_allclose(primitive_out, reference_out, dtype=out_dtype)
        assert_allclose(primitive_dx, reference_dx, dtype=out_dtype)
        assert_allclose(primitive_dgamma, reference_dgamma, dtype=out_dtype)
        if beta is not None:
            assert_allclose(primitive_dbeta, reference_dbeta, dtype=out_dtype)

    def test_norm_grad(self, n, hidden, norm_type, zero_centered_gamma, epsilon, inp_dtype):
        """
        Test transformer_engine.jax.layernorm.layernorm
        """
        if norm_type == "rmsnorm" and zero_centered_gamma is True:
            pytest.skip("RMSNorm and zero_centered_gamma is not supported!")

        self._test_norm_grad(
            n, hidden, norm_type, zero_centered_gamma, epsilon, inp_dtype, quantizer=None
        )

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    # No Norm FWD E5M2 in TE backend
    @pytest_parametrize_wrapper("out_dtype", [jnp.float8_e4m3fn])
    @pytest_parametrize_wrapper(
        "q_layout", [QuantizeLayout.ROWWISE, QuantizeLayout.ROWWISE_COLWISE]
    )
    @pytest_parametrize_wrapper(
        "scaling_mode", [ScalingMode.DELAYED_TENSOR_SCALING, ScalingMode.CURRENT_TENSOR_SCALING]
    )
    def test_norm_grad_with_tensor_scaling_fp8(
        self,
        n,
        hidden,
        norm_type,
        zero_centered_gamma,
        epsilon,
        inp_dtype,
        out_dtype,
        q_layout,
        scaling_mode,
    ):
        """
        Test transformer_engine.jax.layernorm.layernorm
        """
        if norm_type == "rmsnorm" and zero_centered_gamma is True:
            pytest.skip("RMSNorm and zero_centered_gamma is not supported!")

        quantizer = QuantizerFactory.create(
            scaling_mode=scaling_mode, q_dtype=out_dtype, q_layout=q_layout
        )
        self._test_norm_grad(
            n, hidden, norm_type, zero_centered_gamma, epsilon, inp_dtype, quantizer
        )

    def _test_norm_forward(
        self,
        n,
        hidden,
        norm_type,
        zero_centered_gamma,
        epsilon,
        inp_dtype,
        out_dtype,
        scaling_mode,
        q_layout,
    ):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 3)

        x = jax.random.uniform(subkeys[0], (n, hidden), inp_dtype, -1, 1)
        x = jnp.asarray(x, inp_dtype)
        gamma_range = (-1, 1) if zero_centered_gamma else (0, 2)
        gamma = jax.random.uniform(subkeys[1], (hidden,), jnp.float32, *gamma_range)
        gamma = jnp.asarray(gamma, inp_dtype)

        quantizer, ref_quantizer = QuantizerFactory.create(
            n_quantizers=2, scaling_mode=scaling_mode, q_dtype=out_dtype, q_layout=q_layout
        )
        if norm_type == "layernorm":
            beta = jax.random.uniform(subkeys[2], (hidden,), jnp.float32, -1, 1)
            beta = jnp.asarray(beta, inp_dtype)
            output, mu, rsigma = tex.layernorm_fwd(
                x, gamma, beta, zero_centered_gamma, epsilon, quantizer=quantizer
            )
            ref_out, ref_mu, ref_rsigma = _jax_layernorm(
                x, gamma, beta, zero_centered_gamma, epsilon, quantizer=ref_quantizer
            )
        else:
            output, rsigma = tex.rmsnorm_fwd(
                x, gamma, zero_centered_gamma, epsilon, quantizer=quantizer
            )
            ref_out, ref_rsigma = _jax_rmsnorm(
                x, gamma, zero_centered_gamma, epsilon, quantizer=ref_quantizer
            )
            ref_mu = None

        precise_comparison = True

        if get_cudnn_version() < (9, 10, 0) and scaling_mode == ScalingMode.MXFP8_1D_SCALING:
            # Reduce precision of test as we don't use fused norm below this version CuDNN for MXFP8 and instead
            # do an unfused norm and quantize with an intermediate cast into in_dtype which can reduce precision
            precise_comparison = False
        elif is_norm_zero_centered_gamma_in_weight_dtype(scaling_mode):
            # Larger tolerances as our JAX implementation _jax_*norm uses the compute dtype float32
            # for zero-centered gamma always
            precise_comparison = False
        elif scaling_mode == ScalingMode.CURRENT_TENSOR_SCALING and inp_dtype != jnp.float32:
            # Current implementation of Current Tensor Scaling performs unfused layernorm and quantization
            # and writes intermediate results into the input dtype, which will slightly reduce precision
            # if the input dtype is not float32
            precise_comparison = False

        if precise_comparison:
            assert_bitwise_scaled_tensors(output, ref_out)
        else:
            if isinstance(ref_out, ScaledTensor1x):
                assert_allclose(output.dequantize(), ref_out.dequantize(), dtype=out_dtype)
            elif isinstance(ref_out, ScaledTensor2x):
                assert_allclose(
                    output.rowwise_tensor.dequantize(),
                    ref_out.rowwise_tensor.dequantize(),
                    dtype=out_dtype,
                )
                assert_allclose(
                    output.colwise_tensor.dequantize(),
                    ref_out.colwise_tensor.dequantize(),
                    dtype=out_dtype,
                )
            else:
                pytest.fail("Unsupported output type")

        assert_allclose(rsigma, ref_rsigma, dtype=inp_dtype)
        if norm_type == "layernorm":
            assert_allclose(mu, ref_mu, dtype=inp_dtype)

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    # No Norm FWD E5M2 in TE backend
    @pytest_parametrize_wrapper("out_dtype", [jnp.float8_e4m3fn])
    @pytest_parametrize_wrapper(
        "q_layout", [QuantizeLayout.ROWWISE, QuantizeLayout.ROWWISE_COLWISE]
    )
    @pytest_parametrize_wrapper(
        "scaling_mode", [ScalingMode.DELAYED_TENSOR_SCALING, ScalingMode.CURRENT_TENSOR_SCALING]
    )
    def test_norm_forward_with_tensor_scaling_fp8(
        self,
        n,
        hidden,
        norm_type,
        zero_centered_gamma,
        epsilon,
        inp_dtype,
        out_dtype,
        q_layout,
        scaling_mode,
    ):
        if norm_type == "rmsnorm" and zero_centered_gamma is True:
            pytest.skip("RMSNorm and zero_centered_gamma is not supported!")

        self._test_norm_forward(
            n=n,
            hidden=hidden,
            norm_type=norm_type,
            zero_centered_gamma=zero_centered_gamma,
            epsilon=epsilon,
            inp_dtype=inp_dtype,
            out_dtype=out_dtype,
            scaling_mode=scaling_mode,
            q_layout=q_layout,
        )

    @pytest.mark.skipif(not is_mxfp8_supported, reason=reason)
    @pytest.mark.parametrize("out_dtype", [jnp.float8_e4m3fn, jnp.float8_e5m2])
    def test_norm_forward_with_block_scaling_fp8(
        self, n, hidden, norm_type, zero_centered_gamma, epsilon, inp_dtype, out_dtype
    ):
        self._test_norm_forward(
            n=n,
            hidden=hidden,
            norm_type=norm_type,
            zero_centered_gamma=zero_centered_gamma,
            epsilon=epsilon,
            inp_dtype=inp_dtype,
            out_dtype=out_dtype,
            scaling_mode=ScalingMode.MXFP8_1D_SCALING,
            q_layout=QuantizeLayout.ROWWISE_COLWISE,
        )


QUANTIZE_OUTPUT_DTYPES = {
    "L0": [jnp.float8_e4m3fn],
    "L2": [jnp.float8_e4m3fn, jnp.float8_e5m2],
}

ALL_QUANTIZE_TEST_SHAPES_AND_FLATTEN_AXES = [
    ((32, 64), -1),
    ((2, 64, 32), -1),
    ((2, 64, 32), -2),
    ((32, 256, 128), -1),
    ((32, 256, 128), -2),
    ((64, 32, 32, 256), -1),
    ((64, 32, 32, 256), -2),
    ((64, 32, 32, 256), -3),
]

QUANTIZE_TEST_SHAPES_AND_FLATTEN_AXES = {
    "L0": [
        ((32, 64), -1),
        ((2, 64, 32), -1),
        ((2, 64, 32), -2),
    ],
    "L2": ALL_QUANTIZE_TEST_SHAPES_AND_FLATTEN_AXES,
}

QUANTIZATION_INPUT_DTYPE = {
    "L0": [jnp.bfloat16],
    "L2": [jnp.float32, jnp.float16, jnp.bfloat16],
}


@pytest.mark.skipif(not is_fp8_supported, reason=reason)
@pytest_parametrize_wrapper("in_dtype", QUANTIZATION_INPUT_DTYPE)
@pytest_parametrize_wrapper("q_dtype", [jnp.float8_e4m3fn, jnp.float8_e5m2])
@pytest_parametrize_wrapper("input_shape,flatten_axis", ALL_QUANTIZE_TEST_SHAPES_AND_FLATTEN_AXES)
@pytest_parametrize_wrapper("scaling_mode", supported_scaling_modes)
@pytest_parametrize_wrapper(
    "q_layout", [QuantizeLayout.ROWWISE, QuantizeLayout.COLWISE, QuantizeLayout.ROWWISE_COLWISE]
)
class TestQuantize:
    """
    Purely quantization related tests that will always test on a wider set of types and shapes
    """

    def test_qdq(self, in_dtype, input_shape, q_dtype, scaling_mode, q_layout, flatten_axis):
        key = jax.random.PRNGKey(0)

        # Quantizer is created once as some quantization approaches use state from previous iterations (e.g. delayed scaling)
        quantizer = QuantizerFactory.create(
            scaling_mode=scaling_mode,
            q_dtype=q_dtype,
            q_layout=q_layout,
        )
        # Adding dimension to test if padding is done correctly when flatten 3D to 2D
        if flatten_axis == -2:
            input_shape = input_shape[:-1] + (2,) + input_shape[-1:]

        n_iterations = 3 if scaling_mode == ScalingMode.DELAYED_TENSOR_SCALING else 1
        for _ in range(n_iterations):
            x = jax.random.uniform(key, input_shape, in_dtype)

            scaled_tensor = quantizer.quantize(x, flatten_axis=flatten_axis)
            assert_dequantized_scaled_tensor(scaled_tensor, x)

    def test_quantize_bitwise(
        self, in_dtype, input_shape, q_dtype, scaling_mode, q_layout, flatten_axis
    ):

        key = jax.random.PRNGKey(0)
        if flatten_axis == -2:
            input_shape = input_shape[:-1] + (2,) + input_shape[-1:]
        input = jax.random.uniform(key, input_shape, in_dtype)

        te_quantizer, jax_quantizer = QuantizerFactory.create(
            n_quantizers=2, q_dtype=q_dtype, scaling_mode=scaling_mode, q_layout=q_layout
        )

        jax_output = _jax_quantize(input, quantizer=jax_quantizer, flatten_axis=flatten_axis)

        te_output = tex.quantize(input, quantizer=te_quantizer, flatten_axis=flatten_axis)
        assert_bitwise_scaled_tensors(te_output, jax_output)


@pytest.mark.skipif(not is_fp8_supported, reason=reason)
@pytest_parametrize_wrapper("in_dtype", QUANTIZATION_INPUT_DTYPE)
@pytest_parametrize_wrapper("input_shape", [(8, 64, 32)])
@pytest_parametrize_wrapper("q_dtype", [jnp.float8_e4m3fn])
@pytest_parametrize_wrapper("scaling_mode", supported_scaling_modes)
@pytest_parametrize_wrapper("flatten_axis", [-1])
@pytest_parametrize_wrapper("with_group_sizes", [True, False])
@pytest_parametrize_wrapper(
    "q_layout", [QuantizeLayout.ROWWISE, QuantizeLayout.ROWWISE_COLWISE, QuantizeLayout.COLWISE]
)
class TestGroupedQuantize:
    def test_grouped_qdq(
        self, in_dtype, input_shape, q_dtype, scaling_mode, q_layout, flatten_axis, with_group_sizes
    ):
        n_groups, m, n = input_shape
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 2)

        # *32 so that the input shapes works for MXFP8
        input_shape = (m * 32, n)

        if with_group_sizes:
            group_sizes = jnp.sort(jax.random.randint(subkeys[0], (n_groups - 1,), 0, m))
            group_sizes = jnp.concatenate([jnp.array([0]), group_sizes, jnp.array([m])])
            group_sizes = jnp.diff(group_sizes)
            assert group_sizes.sum() == m
            group_sizes = group_sizes * 32
        else:
            group_sizes = None
            input_shape = (n_groups, input_shape[0] // n_groups, input_shape[1])

        if flatten_axis == -2:
            input_shape = input_shape[:-1] + (2,) + input_shape[-1:]

        x = jax.random.uniform(subkeys[1], input_shape, in_dtype)

        grouped_quantizer = QuantizerFactory.create(
            scaling_mode=scaling_mode,
            q_dtype=q_dtype,
            q_layout=q_layout,
            n_groups=n_groups,
        )

        scaled_tensor = tex.grouped_quantize(
            x, group_sizes=group_sizes, flatten_axis=flatten_axis, quantizer=grouped_quantizer
        )

        assert_dequantized_grouped_scaled_tensor(scaled_tensor, x)


"""
    def test_grouped_qdq(
        self, in_dtype, input_shape, q_dtype, scaling_mode, q_layout, flatten_axis, with_group_sizes
    ):
        if with_group_sizes and q_layout in (QuantizeLayout.COLWISE, QuantizeLayout.ROWWISE_COLWISE):
            pytest.skip("Skip as NotImplemented!")
        n_groups, m, n = input_shape
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 2)

        group_sizes = jnp.sort(jax.random.randint(subkeys[0], (n_groups - 1,), 0, m))
        group_sizes = jnp.concatenate([jnp.array([0]), group_sizes, jnp.array([m])])
        group_sizes = jnp.diff(group_sizes)

        assert group_sizes.sum() == m

        # post-process so that the input shapes works for MXFP8
        input_shape = (m * 32, n)
        group_sizes = group_sizes * 32

        if flatten_axis == -2:
            input_shape = input_shape[:-1] + (2,) + input_shape[-1:]

        x = jax.random.uniform(subkeys[1], input_shape, in_dtype)

        grouped_quantizer = QuantizerFactory.create(
            scaling_mode=scaling_mode,
            q_dtype=q_dtype,
            q_layout=q_layout,
            n_groups=n_groups,
        )

        scaled_tensor = grouped_quantizer.quantize(
            x, group_sizes=group_sizes, flatten_axis=flatten_axis
        )

        # Dequantize and compare with original
        assert_dequantized_grouped_scaled_tensor(scaled_tensor, x)
"""


@pytest_parametrize_wrapper("in_dtype", QUANTIZATION_INPUT_DTYPE)
class TestFusedQuantize:

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest_parametrize_wrapper("scaling_mode", supported_scaling_modes)
    @pytest_parametrize_wrapper("input_shape,flatten_axis", QUANTIZE_TEST_SHAPES_AND_FLATTEN_AXES)
    @pytest_parametrize_wrapper("out_dtype", QUANTIZE_OUTPUT_DTYPES)
    @pytest_parametrize_wrapper(
        "q_layout", [QuantizeLayout.ROWWISE, QuantizeLayout.ROWWISE_COLWISE]
    )
    def test_quantize_dbias(
        self, in_dtype, input_shape, out_dtype, scaling_mode, q_layout, flatten_axis
    ):
        if scaling_mode == ScalingMode.MXFP8_1D_SCALING and not is_shape_supported_by_mxfp8(
            input_shape
        ):
            pytest.skip(f"Input shape {input_shape} is not supported by MXFP8")

        if (flatten_axis < 0 and flatten_axis + len(input_shape) <= 0) or flatten_axis <= 0:
            pytest.skip(
                f"Flatten axis {flatten_axis} is not supported for input shape {input_shape}. There"
                " must be at least one axis on either side of the flatten_axis split."
            )

        key = jax.random.PRNGKey(0)
        input = jax.random.uniform(key, input_shape, in_dtype)

        jax_quantizer, te_quantizer = QuantizerFactory.create(
            n_quantizers=2, q_dtype=out_dtype, scaling_mode=scaling_mode, q_layout=q_layout
        )

        te_output, te_dbias = jit(
            lambda input: tex.quantize_dbias(
                input, quantizer=te_quantizer, flatten_axis=flatten_axis
            )
        )(input)

        jax_output, jax_dbias = jit(
            lambda input: _jax_quantize_dbias(
                input, quantizer=jax_quantizer, flatten_axis=flatten_axis
            )
        )(input)

        assert_bitwise_scaled_tensors(te_output, jax_output)

        assert_allclose(te_dbias, jax_dbias)

    def _test_quantize_dact_dbias(
        self, in_dtype, input_shape, out_dtype, scaling_mode, activation_type, is_dbias, q_layout
    ):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 2)
        x = jax.random.uniform(subkeys[0], input_shape, in_dtype, -1, 1)
        x = jnp.expand_dims(x, axis=-2)
        x = jnp.repeat(x, len(activation_type), axis=-2)
        dz = jax.random.uniform(subkeys[1], input_shape, in_dtype, -1, 1)

        jax_quantizer, te_quantizer = QuantizerFactory.create(
            n_quantizers=2, q_dtype=out_dtype, scaling_mode=scaling_mode, q_layout=q_layout
        )
        is_casted_output = te_quantizer is not None

        te_output, te_dbias = jit(
            lambda dz, x: tex.quantize_dact_dbias(
                dz,
                x,
                activation_type=activation_type,
                is_dbias=is_dbias,
                quantizer=te_quantizer,
            )
        )(dz, x)

        jax_output, jax_dbias = jit(
            lambda dz, x: _jax_quantize_dact_dbias(
                dz,
                x,
                activation_type=activation_type,
                is_dbias=is_dbias,
                quantizer=jax_quantizer,
            )
        )(dz, x)

        if is_casted_output:
            assert_bitwise_scaled_tensors(te_output, jax_output)
        else:
            assert_allclose(te_output, jax_output)

        if is_dbias:
            assert_allclose(te_dbias, jax_dbias)

    @pytest_parametrize_wrapper("activation_type", ACTIVATION_TYPES)
    @pytest_parametrize_wrapper("input_shape", ALL_ACTIVATION_SHAPES)
    @pytest_parametrize_wrapper("is_dbias", [True, False])
    def test_quantize_dact_dbias_no_quantization(
        self,
        in_dtype,
        input_shape,
        activation_type,
        is_dbias,
    ):
        self._test_quantize_dact_dbias(
            in_dtype=in_dtype,
            input_shape=input_shape,
            out_dtype=in_dtype,
            scaling_mode=ScalingMode.NO_SCALING,
            activation_type=activation_type,
            is_dbias=is_dbias,
            q_layout=QuantizeLayout.ROWWISE,
        )

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest_parametrize_wrapper("activation_type", ACTIVATION_TYPES)
    @pytest_parametrize_wrapper("input_shape", ALL_ACTIVATION_SHAPES)
    @pytest_parametrize_wrapper("out_dtype", QUANTIZE_OUTPUT_DTYPES)
    @pytest_parametrize_wrapper("is_dbias", [True, False])
    @pytest_parametrize_wrapper(
        "q_layout", [QuantizeLayout.ROWWISE, QuantizeLayout.ROWWISE_COLWISE]
    )
    @pytest_parametrize_wrapper(
        "scaling_mode", [ScalingMode.DELAYED_TENSOR_SCALING, ScalingMode.CURRENT_TENSOR_SCALING]
    )
    def test_quantize_dact_dbias_tensor_scaling(
        self, in_dtype, input_shape, out_dtype, activation_type, is_dbias, q_layout, scaling_mode
    ):
        self._test_quantize_dact_dbias(
            in_dtype=in_dtype,
            input_shape=input_shape,
            out_dtype=out_dtype,
            scaling_mode=scaling_mode,
            activation_type=activation_type,
            is_dbias=is_dbias,
            q_layout=q_layout,
        )

    @pytest.mark.skipif(not is_mxfp8_supported, reason=reason)
    @pytest_parametrize_wrapper("activation_type", ACTIVATION_TYPES)
    @pytest_parametrize_wrapper(
        "input_shape", [s for s in ALL_ACTIVATION_SHAPES if is_shape_supported_by_mxfp8(s)]
    )
    @pytest_parametrize_wrapper("out_dtype", QUANTIZE_OUTPUT_DTYPES)
    @pytest_parametrize_wrapper("is_dbias", [True, False])
    @pytest_parametrize_wrapper(
        "q_layout", [QuantizeLayout.COLWISE, QuantizeLayout.ROWWISE_COLWISE]
    )
    def test_quantize_dact_dbias_mxfp8_scaling(
        self, in_dtype, input_shape, out_dtype, activation_type, is_dbias, q_layout
    ):
        if reduce(operator.mul, input_shape[:-1]) % 128 != 0 or input_shape[-1] % 128 != 0:
            # TODO(Jeremy): Remove this if pulling in newer TE branch supports non-full-tile shapes.
            # If it doesn't, move this check into the quantize_dact_dbias function and revert to JAX
            # implementation in the unsupported cases
            pytest.skip(
                f"Input shape {input_shape} is not supported by dact MXFP8 kernel in TE currently"
            )

        self._test_quantize_dact_dbias(
            in_dtype=in_dtype,
            input_shape=input_shape,
            out_dtype=out_dtype,
            scaling_mode=ScalingMode.MXFP8_1D_SCALING,
            activation_type=activation_type,
            is_dbias=is_dbias,
            q_layout=q_layout,
        )


class TestDense:
    def _ref_gemm_with_jnp_dot(self, a, b, data_layout):
        if data_layout[0] == "T":
            a = jnp.swapaxes(a, -1, -2)
        if data_layout[1] == "T":
            b = jnp.swapaxes(b, -1, -2)
        return jnp.dot(a, b)

    def _generate_gemm_input(self, m, n, k, data_layout):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 2)
        x = jax.random.uniform(
            subkeys[0],
            (m if data_layout[0] == "N" else k, k if data_layout[0] == "N" else m),
            dtype=jnp.bfloat16,
        ) / jnp.sqrt(k)
        w = jax.random.uniform(
            subkeys[1],
            (k if data_layout[1] == "N" else n, n if data_layout[1] == "N" else k),
            dtype=jnp.bfloat16,
        ) / jnp.sqrt(n)
        lhs_contracting_dim = (1,) if data_layout[0] == "N" else (0,)
        rhs_contracting_dim = (0,) if data_layout[1] == "N" else (1,)
        contracting_dims = (lhs_contracting_dim, rhs_contracting_dim)

        return (x, w, contracting_dims)

    @pytest_parametrize_wrapper("m,n,k", [(64, 32, 64)])
    @pytest_parametrize_wrapper("data_layout", ["TN", "NT", "NN", "TT"])
    def test_gemm_bf16(self, m, n, k, data_layout):
        x, w, contracting_dims = self._generate_gemm_input(m, n, k, data_layout)

        primitive_out = tex.gemm(x, w, contracting_dims)
        ref_out = self._ref_gemm_with_jnp_dot(x, w, data_layout)

        assert_allclose(primitive_out, ref_out, dtype=jnp.bfloat16)

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest_parametrize_wrapper("m,n,k", [(64, 32, 64)])
    @pytest_parametrize_wrapper("q_dtype", [jnp.float8_e4m3fn, jnp.float8_e5m2])
    @pytest_parametrize_wrapper("scaling_mode", supported_scaling_modes)
    @pytest_parametrize_wrapper("data_layout", ["TN", "NT", "NN", "TT"])
    def test_gemm_fp8(self, m, n, k, q_dtype, scaling_mode, data_layout):
        x, w, contracting_dims = self._generate_gemm_input(m, n, k, data_layout)
        quantizer_set = QuantizerFactory.create_set(
            scaling_mode=scaling_mode, fwd_dtype=q_dtype, bwd_dtype=q_dtype, is_2x2x=False
        )
        primitive_out = tex.gemm(
            x, w, contracting_dims=contracting_dims, quantizer_set=quantizer_set
        )
        ref_out = self._ref_gemm_with_jnp_dot(x, w, data_layout)

        assert_allclose(primitive_out, ref_out, dtype=q_dtype)

    @pytest_parametrize_wrapper("m,n,k", [(64, 32, 64)])
    def test_dense_grad_bf16(self, m, n, k):
        data_layout = "NN"
        x, w, contracting_dims = self._generate_gemm_input(m, n, k, data_layout)

        def primitive_func(x, w, contracting_dims):
            primitive_out = dense(x, w, contracting_dims=contracting_dims)
            return jnp.mean(primitive_out)

        def ref_func(x, w, data_layout):
            return jnp.mean(self._ref_gemm_with_jnp_dot(x, w, data_layout))

        value_n_grad_primitive_func = value_and_grad(primitive_func, (0, 1))

        value_n_grad_ref_func = value_and_grad(ref_func, (0, 1))

        primitive_out, (primitive_x_grad, primitive_w_grad) = value_n_grad_primitive_func(
            x, w, contracting_dims
        )
        ref_out, (ref_x_grad, ref_w_grad) = value_n_grad_ref_func(x, w, data_layout)

        assert_allclose(primitive_out, ref_out, dtype=jnp.bfloat16)
        assert_allclose(primitive_x_grad, ref_x_grad, dtype=jnp.bfloat16)
        assert_allclose(primitive_w_grad, ref_w_grad, dtype=jnp.bfloat16)

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest_parametrize_wrapper("m,n,k", [(64, 32, 64)])
    @pytest_parametrize_wrapper("q_dtype", [jnp.float8_e4m3fn, jnp.float8_e5m2])
    @pytest_parametrize_wrapper("scaling_mode", supported_scaling_modes)
    def test_dense_grad_fp8(self, m, n, k, q_dtype, scaling_mode):
        data_layout = "NN"
        x, w, contracting_dims = self._generate_gemm_input(m, n, k, data_layout)

        key = jax.random.PRNGKey(1)
        bias = jax.random.uniform(key, n, dtype=jnp.bfloat16)

        def primitive_func(x, w, bias, contracting_dims, quantizer_set):
            primitive_out = dense(
                x, w, bias, contracting_dims=contracting_dims, quantizer_set=quantizer_set
            )
            return jnp.mean(primitive_out)

        def ref_func(x, w, bias, data_layout):
            return jnp.mean(
                self._ref_gemm_with_jnp_dot(x, w, data_layout) + jnp.expand_dims(bias, axis=0)
            )

        value_n_grad_primitive_func = value_and_grad(primitive_func, (0, 1, 2))
        value_n_grad_ref_func = value_and_grad(ref_func, (0, 1, 2))

        quantizer_set = QuantizerFactory.create_set(
            scaling_mode=scaling_mode, fwd_dtype=q_dtype, bwd_dtype=q_dtype, is_2x2x=True
        )

        n_iterations = 3 if scaling_mode == ScalingMode.DELAYED_TENSOR_SCALING else 1
        for _ in range(n_iterations):
            primitive_out, (primitive_x_grad, primitive_w_grad, primitive_bias_grad) = (
                value_n_grad_primitive_func(x, w, bias, contracting_dims, quantizer_set)
            )

        ref_out, (ref_x_grad, ref_w_grad, ref_bias_grad) = value_n_grad_ref_func(
            x, w, bias, data_layout
        )

        assert_allclose(primitive_out, ref_out, dtype=q_dtype)
        assert_allclose(primitive_x_grad, ref_x_grad, dtype=q_dtype)
        assert_allclose(primitive_w_grad, ref_w_grad, dtype=q_dtype)
        assert_allclose(primitive_bias_grad, ref_bias_grad, dtype=q_dtype)


@pytest.fixture(name="random_inputs")
def random_inputs_fixture(shape):
    key = jax.random.PRNGKey(0)
    subkeys = jax.random.split(key, 4)
    out = jax.random.uniform(subkeys[0], shape, jnp.bfloat16, 5, 8)
    return out


def _ref_jax_norm_impl(x, gamma, beta, norm_type, zero_centered_gamma, eps, quantizer):
    if norm_type == "rmsnorm":
        ln_out, _ = _jax_rmsnorm(x, gamma, zero_centered_gamma, eps, quantizer)
    else:
        ln_out, _, _ = _jax_layernorm(x, gamma, beta, zero_centered_gamma, eps, quantizer)
    if isinstance(ln_out, ScaledTensor):
        ln_out = ln_out.dequantize()
    return ln_out


class TestFusedDense:
    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize("m,n,k", [(64, 32, 64)])
    @pytest.mark.parametrize("q_dtype", [jnp.float8_e4m3fn, jnp.float8_e5m2])
    @pytest.mark.parametrize("scaling_mode", supported_scaling_modes)
    @pytest.mark.parametrize("norm_type", ["layernorm", "rmsnorm"])
    def test_layernorm_dense_grad(self, m, n, k, q_dtype, scaling_mode, norm_type):
        """
        Test layernorm_dense VJP Rule
        """
        # No Norm FWD E5M2 in TE backend
        if q_dtype == jnp.float8_e5m2 and scaling_mode in (
            ScalingMode.DELAYED_TENSOR_SCALING,
            ScalingMode.CURRENT_TENSOR_SCALING,
        ):
            pytest.skip("E5M2 is not supported in normalization with TE Backend!")

        # zero_centered_gamma is already tested in TestNorm
        zero_centered_gamma = False
        eps = 1e-6

        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 4)

        # NN in FWD
        x = jax.random.normal(subkeys[0], (m, k)).astype(jnp.bfloat16) / jnp.sqrt(k)
        w = jax.random.normal(subkeys[1], (k, n)).astype(jnp.bfloat16) / jnp.sqrt(n)

        gamma = jax.random.normal(subkeys[2], (k,)).astype(jnp.bfloat16)

        quantizer_set = QuantizerFactory.create_set(
            scaling_mode=scaling_mode,
            fwd_dtype=q_dtype,
            bwd_dtype=q_dtype,
            is_2x2x=True,
        )

        if norm_type == "layernorm":
            beta = jax.random.normal(subkeys[3], (k,)).astype(jnp.bfloat16)
        else:
            beta = None

        def prim_func(x, w, gamma, beta):
            # bias = None as quantize_dbias is already tested in test_dense_grad_fp8
            prim_out = layernorm_dense(
                x,
                w,
                gamma,
                beta,
                None,
                norm_type,
                zero_centered_gamma,
                eps,
                quantizer_set=quantizer_set,
            )
            return jnp.mean(prim_out)

        def ref_func(x, w, gamma, beta):
            x = _ref_jax_norm_impl(
                x, gamma, beta, norm_type, zero_centered_gamma, eps, quantizer=None
            )
            return jnp.mean(jnp.dot(x, w))

        value_n_grad_prim_func = value_and_grad(prim_func, (0, 1, 2, 3))
        value_n_grad_ref_func = value_and_grad(ref_func, (0, 1, 2, 3))

        ref_out, (ref_x_grad, ref_w_grad, ref_gamma_grad, ref_beta_grad) = value_n_grad_ref_func(
            x, w, gamma, beta
        )

        n_iterations = 3 if scaling_mode == ScalingMode.DELAYED_TENSOR_SCALING else 1
        for _ in range(n_iterations):
            prim_out, (
                prim_x_grad,
                prim_w_grad,
                prim_gamma_grad,
                prim_beta_grad,
            ) = value_n_grad_prim_func(x, w, gamma, beta)

        assert_allclose(prim_out, ref_out, dtype=q_dtype)
        assert_allclose(prim_x_grad, ref_x_grad, dtype=q_dtype)
        assert_allclose(prim_w_grad, ref_w_grad, dtype=q_dtype)
        assert_allclose(prim_gamma_grad, ref_gamma_grad, dtype=q_dtype)
        if beta is not None:
            assert_allclose(prim_beta_grad, ref_beta_grad, dtype=q_dtype)

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize("m,n,k", [(64, 32, 64)])
    @pytest.mark.parametrize("activation_type", [("gelu",), ("gelu", "linear")])
    @pytest.mark.parametrize("q_dtype", [jnp.float8_e4m3fn, jnp.float8_e5m2])
    @pytest.mark.parametrize("scaling_mode", supported_scaling_modes)
    @pytest.mark.parametrize("norm_type", ["layernorm", "rmsnorm"])
    @pytest.mark.parametrize("use_bias", [True, False])
    def test_layernorm_mlp_grad(
        self, m, n, k, activation_type, q_dtype, scaling_mode, norm_type, use_bias
    ):
        """
        Test layernorm_mlp VJP Rule
        """
        # No Norm FWD E5M2 in TE backend
        if q_dtype == jnp.float8_e5m2 and scaling_mode in (
            ScalingMode.DELAYED_TENSOR_SCALING,
            ScalingMode.CURRENT_TENSOR_SCALING,
        ):
            pytest.skip("E5M2 is not supported in normalization with TE Backend!")

        # zero_centered_gamma is already tested in TestNorm
        zero_centered_gamma = False
        eps = 1e-6

        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 6)

        x = jax.random.normal(subkeys[0], (m, k), jnp.bfloat16)
        kernel_1 = jax.random.normal(
            subkeys[1], (k, len(activation_type), n), jnp.bfloat16
        ) / jnp.sqrt(k)
        kernel_2 = jax.random.normal(subkeys[2], (n, k), jnp.bfloat16) / jnp.sqrt(n)
        gamma = jax.random.normal(subkeys[5], (k,), jnp.bfloat16)
        beta = None  # was tested in TestNorm
        if use_bias:
            bias_1 = jax.random.normal(subkeys[3], (len(activation_type), n), jnp.bfloat16)
            bias_2 = jax.random.normal(subkeys[4], (k,), jnp.bfloat16)
        else:
            bias_1 = None
            bias_2 = None

        quantizer_sets = QuantizerFactory.create_set(
            n_quantizer_sets=2,
            scaling_mode=scaling_mode,
            fwd_dtype=q_dtype,
            bwd_dtype=q_dtype,
            is_2x2x=True,
        )

        if norm_type == "layernorm":
            beta = jax.random.normal(subkeys[3], (k,)).astype(jnp.bfloat16)
        else:
            beta = None

        def prim_func(x, gamma, kernel_1, kernel_2, bias_1, bias_2):
            return jnp.mean(
                layernorm_mlp(
                    x,
                    gamma,
                    beta,
                    [kernel_1, kernel_2],
                    [bias_1, bias_2],
                    norm_type,
                    zero_centered_gamma=zero_centered_gamma,
                    epsilon=eps,
                    activation_type=activation_type,
                    quantizer_sets=quantizer_sets,
                )
            )

        def _ref_func_impl(x, gamma, kernel_1, kernel_2, bias_1, bias_2):
            ln_out = _ref_jax_norm_impl(
                x, gamma, beta, norm_type, zero_centered_gamma, eps, quantizer=None
            )
            # TODO: replace gemm with jnp.dot
            linear_1_out = tex.gemm(ln_out, kernel_1, ((1,), (0,)))
            if use_bias:
                bias_1_shape = (1,) * (linear_1_out.ndim - bias_1.ndim) + bias_1.shape
                linear_1_out += jnp.reshape(bias_1, bias_1_shape)

            x = _jax_act_lu(linear_1_out, activation_type)
            linear_2_out = tex.gemm(x, kernel_2, ((1,), (0,)))
            if use_bias:
                bias_2_shape = (1,) * (linear_2_out.ndim - bias_2.ndim) + bias_2.shape
                linear_2_out += jnp.reshape(bias_2, bias_2_shape)

            return linear_2_out

        def ref_func(x, gamma, kernel_1, kernel_2, bias_1, bias_2):
            return jnp.mean(_ref_func_impl(x, gamma, kernel_1, kernel_2, bias_1, bias_2))

        value_n_grad_prim_func = value_and_grad(prim_func, range(6))
        value_n_grad_ref_func = value_and_grad(ref_func, range(6))

        n_iterations = 3 if scaling_mode == ScalingMode.DELAYED_TENSOR_SCALING else 1
        for _ in range(n_iterations):
            prim_out, (
                prim_x_grad,
                prim_gamma_grad,
                prim_kernel_1_grad,
                prim_kernel_2_grad,
                prim_bias_1_grad,
                prim_bias_2_grad,
            ) = value_n_grad_prim_func(x, gamma, kernel_1, kernel_2, bias_1, bias_2)

        ref_out, (
            ref_x_grad,
            ref_gamma_grad,
            ref_kernel_1_grad,
            ref_kernel_2_grad,
            ref_bias_1_grad,
            ref_bias_2_grad,
        ) = value_n_grad_ref_func(x, gamma, kernel_1, kernel_2, bias_1, bias_2)

        assert_allclose(prim_out, ref_out, dtype=q_dtype)

        assert_allclose(prim_kernel_2_grad, ref_kernel_2_grad, dtype=q_dtype)
        if use_bias:
            assert_allclose(prim_bias_2_grad, ref_bias_2_grad, dtype=q_dtype)

        assert_allclose(prim_kernel_1_grad, ref_kernel_1_grad, dtype=q_dtype)
        if use_bias:
            assert_allclose(prim_bias_1_grad, ref_bias_1_grad, dtype=q_dtype)

        assert_allclose(prim_gamma_grad, ref_gamma_grad, dtype=q_dtype)
        assert_allclose(prim_x_grad, ref_x_grad, dtype=q_dtype)


# This function is modified from transformer_engine/jax/cpp_extensions/gemm.py::_jax_gemm()
def _quantize_gemm_pair(lhs, rhs, contracting_dims, lhs_quantizer, rhs_quantizer):
    ((lhs_contract_dim,), (rhs_contract_dim,)) = contracting_dims
    lhs_is_rowwise = lhs_contract_dim == lhs.ndim - 1
    rhs_is_rowwise = rhs_contract_dim == rhs.ndim - 1
    lhs_q = lhs_quantizer.quantize(
        lhs,
        is_rowwise=lhs_is_rowwise,
        is_colwise=not lhs_is_rowwise,
    )
    rhs_q = rhs_quantizer.quantize(
        rhs,
        is_rowwise=rhs_is_rowwise,
        is_colwise=not rhs_is_rowwise,
    )
    return lhs_q, rhs_q


# E5M2 * E5M2 is not supported
fwd_bwd_dtypes = [
    [jnp.float8_e4m3fn, jnp.float8_e4m3fn],
    [jnp.float8_e4m3fn, jnp.float8_e5m2],
    [jnp.float8_e5m2, jnp.float8_e4m3fn],
]

"""
@pytest_parametrize_wrapper(
    "shape_list", [[(512, 128, 256), (256, 128, 256), (256, 128, 128), (512, 256, 128)]]
)
class TestGroupedDense:
    def _ref_grouped_gemm_with_jnp_dot(self, lhs_list, rhs_list, contracting_dims_list):
        ref_out_list = []
        for lhs, rhs, contracting_dims in zip(lhs_list, rhs_list, contracting_dims_list):
            dim_nums = (contracting_dims, ((), ()))
            ref_out_list.append(jax.lax.dot_general(lhs, rhs, dim_nums))
        return ref_out_list

    def _generate_grouped_gemm_input(self, dtype, shape_list, layout_list):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, len(shape_list) * 2)

        lhs_list, rhs_list, contracting_dims_list = [], [], []
        for i, ((m, n, k), data_layout) in enumerate(zip(shape_list, layout_list)):
            lhs = jax.random.uniform(
                subkeys[2 * i],
                (m if data_layout[0] == "N" else k, k if data_layout[0] == "N" else m),
                dtype=dtype,
            )
            rhs = jax.random.uniform(
                subkeys[2 * i + 1],
                (k if data_layout[1] == "N" else n, n if data_layout[1] == "N" else k),
                dtype=dtype,
            )
            lhs_contracting_dim = (1,) if data_layout[0] == "N" else (0,)
            rhs_contracting_dim = (0,) if data_layout[1] == "N" else (1,)
            contracting_dims = (lhs_contracting_dim, rhs_contracting_dim)

            lhs_list.append(lhs)
            rhs_list.append(rhs)
            contracting_dims_list.append(contracting_dims)

        return lhs_list, rhs_list, contracting_dims_list

    @pytest_parametrize_wrapper("dtype", [jnp.bfloat16, jnp.float16])
    @pytest_parametrize_wrapper("layout_list", [["NN", "TN", "NT", "TT"]])
    def test_grouped_gemm_fp16(self, dtype, shape_list, layout_list):
        lhs_list, rhs_list, contracting_dims_list = self._generate_grouped_gemm_input(
            dtype, shape_list, layout_list
        )
        ref_out = self._ref_grouped_gemm_with_jnp_dot(lhs_list, rhs_list, contracting_dims_list)
        primitive_out = tex.grouped_gemm(lhs_list, rhs_list, contracting_dims_list)
        for i in range(len(shape_list)):
            assert_allclose(primitive_out[i], ref_out[i], dtype=dtype)

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize("fwd_bwd_dtype", fwd_bwd_dtypes)
    @pytest_parametrize_wrapper("scaling_mode", supported_scaling_modes)
    @pytest_parametrize_wrapper("layout_list", [["NN", "TN", "NT", "TT"]])
    def test_grouped_gemm_fp8(self, fwd_bwd_dtype, scaling_mode, shape_list, layout_list):
        fwd_dtype, bwd_dtype = fwd_bwd_dtype
        quantizer_set = QuantizerFactory.create_set(
            scaling_mode=scaling_mode, fwd_dtype=fwd_dtype, bwd_dtype=bwd_dtype, is_2x2x=False
        )

        out_dtype = jnp.bfloat16
        lhs_list, rhs_list, contracting_dims_list = self._generate_grouped_gemm_input(
            out_dtype, shape_list, layout_list
        )
        q_lhs_list = []
        q_rhs_list = []
        for lhs, rhs, contracting_dims in zip(lhs_list, rhs_list, contracting_dims_list):
            # quantizer_set.x and quantizer_set.kernel have the same q_dtype, we want to
            # test the case where lhs and rhs have different q_dtypes
            q_lhs, q_rhs = _quantize_gemm_pair(
                lhs, rhs, contracting_dims, quantizer_set.x, quantizer_set.dgrad
            )
            q_lhs_list.append(q_lhs)
            q_rhs_list.append(q_rhs)

        ref_out = self._ref_grouped_gemm_with_jnp_dot(lhs_list, rhs_list, contracting_dims_list)
        primitive_out = tex.grouped_gemm(q_lhs_list, q_rhs_list, contracting_dims_list)

        allclose_dtype = jnp.float8_e4m3fn
        if fwd_dtype == jnp.float8_e5m2 or bwd_dtype == jnp.float8_e5m2:
            allclose_dtype = jnp.float8_e5m2
        for i in range(len(shape_list)):
            assert_allclose(primitive_out[i], ref_out[i], dtype=allclose_dtype)

    @pytest_parametrize_wrapper("dtype", [jnp.bfloat16, jnp.float16])
    def test_grouped_dense_grad_fp16(self, dtype, shape_list):
        group_size = len(shape_list)
        layout_list = ["NN" for _ in range(group_size)]

        x_list, kernel_list, contracting_dims_list = self._generate_grouped_gemm_input(
            dtype, shape_list, layout_list
        )
        bias_list = []
        key = jax.random.PRNGKey(1)
        for shape in shape_list:
            n = shape[1]
            bias = jax.random.uniform(key, n, dtype=dtype)
            bias_list.append(bias)

        def ref_func(x_list, kernel_list, bias_list, contracting_dims_list):
            out_list = []
            for i in range(len(x_list)):
                out_list.append(
                    dense(
                        x_list[i],
                        kernel_list[i],
                        bias_list[i],
                        contracting_dims=contracting_dims_list[i],
                    )
                )
            # Note: we use jnp.sum instead of jnp.mean to make the gradient larger
            # and prevent them from being clamp to zero
            out_sum_list = [jnp.sum(out) for out in out_list]
            return jnp.sum(jnp.asarray(out_sum_list))

        def primitive_func(x_list, kernel_list, bias_list, contracting_dims_list):
            out_list = grouped_dense(x_list, kernel_list, bias_list, contracting_dims_list)
            out_sum_list = [jnp.sum(out) for out in out_list]
            return jnp.sum(jnp.asarray(out_sum_list))

        value_n_grad_ref_func = value_and_grad(ref_func, (0, 1, 2))
        value_n_grad_primitive_func = value_and_grad(primitive_func, (0, 1, 2))

        ref_out_mean, (ref_dgrad_list, ref_wgrad_list, ref_dbias_list) = value_n_grad_ref_func(
            x_list, kernel_list, bias_list, contracting_dims_list
        )
        primitive_out_mean, (primitive_dgrad_list, primitive_wgrad_list, primitive_dbias_list) = (
            value_n_grad_primitive_func(x_list, kernel_list, bias_list, contracting_dims_list)
        )

        assert_allclose(primitive_out_mean, ref_out_mean, dtype=dtype)
        for i in range(group_size):
            assert_allclose(primitive_dgrad_list[i], ref_dgrad_list[i], dtype=dtype)
            assert_allclose(primitive_wgrad_list[i], ref_wgrad_list[i], dtype=dtype)
            assert_allclose(primitive_dbias_list[i], ref_dbias_list[i], dtype=dtype)

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize("fwd_bwd_dtype", fwd_bwd_dtypes)
    @pytest_parametrize_wrapper("scaling_mode", supported_scaling_modes)
    def test_grouped_dense_grad_fp8(self, fwd_bwd_dtype, scaling_mode, shape_list):
        group_size = len(shape_list)
        layout_list = ["NN" for _ in range(group_size)]
        fwd_dtype, bwd_dtype = fwd_bwd_dtype
        if fwd_dtype == jnp.float8_e5m2:
            pytest.skip("We never use E5M2 for fwd_dtype in training")

        # Question: should we use different quantizers for different groups?
        ref_quantizer_set_list = []
        quantizer_set_list = []
        for _ in range(group_size):
            ref_quantizer_set = QuantizerFactory.create_set(
                scaling_mode=scaling_mode, fwd_dtype=fwd_dtype, bwd_dtype=bwd_dtype, is_2x2x=True
            )
            ref_quantizer_set_list.append(ref_quantizer_set)
            quantizer_set = QuantizerFactory.create_set(
                scaling_mode=scaling_mode, fwd_dtype=fwd_dtype, bwd_dtype=bwd_dtype, is_2x2x=True
            )
            quantizer_set_list.append(quantizer_set)

        out_dtype = jnp.bfloat16
        x_list, kernel_list, contracting_dims_list = self._generate_grouped_gemm_input(
            out_dtype, shape_list, layout_list
        )
        bias_list = []
        key = jax.random.PRNGKey(1)
        for shape in shape_list:
            n = shape[1]
            bias = jax.random.uniform(key, n, dtype=out_dtype)
            bias_list.append(bias)

        def ref_func(x_list, kernel_list, bias_list, contracting_dims_list, quantizer_set_list):
            out_list = []
            for i in range(len(x_list)):
                out_list.append(
                    dense(
                        x_list[i],
                        kernel_list[i],
                        bias_list[i],
                        contracting_dims=contracting_dims_list[i],
                        quantizer_set=quantizer_set_list[i],
                    )
                )
            # Note: we use jnp.sum instead of jnp.mean to make the gradient larger
            # and prevent them from being clamp to zero
            out_sum_list = [jnp.sum(out) for out in out_list]
            return jnp.sum(jnp.asarray(out_sum_list))

        def primitive_func(
            x_list, kernel_list, bias_list, contracting_dims_list, quantizer_set_list
        ):
            out_list = grouped_dense(
                x_list, kernel_list, bias_list, contracting_dims_list, quantizer_set_list
            )
            out_sum_list = [jnp.sum(out) for out in out_list]
            return jnp.sum(jnp.asarray(out_sum_list))

        value_n_grad_ref_func = value_and_grad(ref_func, (0, 1, 2))
        value_n_grad_primitive_func = value_and_grad(primitive_func, (0, 1, 2))

        ref_out_mean, (ref_dgrad_list, ref_wgrad_list, ref_dbias_list) = value_n_grad_ref_func(
            x_list, kernel_list, bias_list, contracting_dims_list, ref_quantizer_set_list
        )
        primitive_out_mean, (primitive_dgrad_list, primitive_wgrad_list, primitive_dbias_list) = (
            value_n_grad_primitive_func(
                x_list, kernel_list, bias_list, contracting_dims_list, quantizer_set_list
            )
        )

        allclose_dtype = jnp.float8_e4m3fn
        if fwd_dtype == jnp.float8_e5m2 or bwd_dtype == jnp.float8_e5m2:
            allclose_dtype = jnp.float8_e5m2
        assert_allclose(primitive_out_mean, ref_out_mean, dtype=allclose_dtype)
        for i in range(group_size):
            assert_allclose(primitive_dgrad_list[i], ref_dgrad_list[i], dtype=allclose_dtype)
            assert_allclose(primitive_wgrad_list[i], ref_wgrad_list[i], dtype=allclose_dtype)
            assert_allclose(primitive_dbias_list[i], ref_dbias_list[i], dtype=allclose_dtype)
"""
