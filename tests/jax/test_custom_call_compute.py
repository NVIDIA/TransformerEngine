# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from contextlib import nullcontext
import functools
import operator
from typing import Callable, List, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import jit, value_and_grad
from flax import linen as nn

from utils import assert_allclose
from transformer_engine.jax.dot import type_safe_dot_general, dequantize, quantize
from transformer_engine.jax.fp8 import FP8MetaPackage, FP8Helper, is_fp8_available
from transformer_engine.jax.layernorm import layernorm, layernorm_fp8_dot
from transformer_engine.jax.layernorm_mlp import activation_lu, fused_layernorm_fp8_mlp
from transformer_engine.jax import cpp_extensions as tex

GEMM_CASES = [
    (256, 256, 512),
    (32, 32, 32),
    (2048, 1024, 2048),
    (2048, 2048, 1024),
    (2048, 1024, 1024),
]
FP8_COMPUTE_TYPE = [jnp.float8_e4m3fn, jnp.float8_e5m2]
LN_CASES = [(512, 1024)]
DTYPES = [jnp.bfloat16, jnp.float32]
is_fp8_supported, reason = is_fp8_available()


def _convert_to_activation_function(fn_or_string):
    """Convert a string to an activation function."""
    if fn_or_string == "linear":
        return lambda x: x
    if fn_or_string == "quick_gelu":
        return lambda x: nn.gelu(x, approximate=True)
    if fn_or_string == "squared_relu":
        return lambda x: functools.reduce(operator.mul, [nn.relu(x), nn.relu(x)])
    if isinstance(fn_or_string, str):
        return getattr(nn, fn_or_string)
    if callable(fn_or_string):
        return fn_or_string
    raise ValueError(f"don't know how to convert {fn_or_string} to an activation function")


class TestFP8Dot:

    @staticmethod
    def _generate_fp8_meta():
        fp8_dtype_list = [FP8Helper.FWD_DTYPE, FP8Helper.FWD_DTYPE, FP8Helper.BWD_DTYPE]
        amax_list = [
            jnp.zeros((FP8Helper.AMAX_HISTORY_LEN,), jnp.float32),
            jnp.zeros((FP8Helper.AMAX_HISTORY_LEN,), jnp.float32),
            jnp.zeros((FP8Helper.AMAX_HISTORY_LEN,), jnp.float32),
        ]
        scale_list = [
            jnp.ones((1,), jnp.float32),
            jnp.ones((1,), jnp.float32),
            jnp.ones((1,), jnp.float32),
        ]
        return fp8_dtype_list, amax_list, scale_list

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    def test_qdq(self):
        FP8_E4M3_MAX = (jnp.finfo(jnp.float8_e4m3fn).max).astype(jnp.float32)
        x = jnp.asarray([[-1, 0.1], [2, 3]], jnp.float32)
        amax = jnp.max(jnp.abs(x)).reshape(1)
        scale = jnp.asarray(FP8_E4M3_MAX / amax, jnp.float32).reshape(1)
        scale_inv = (1 / scale).reshape(1)

        y, _ = quantize(x, q_dtype=jnp.float8_e4m3fn, scale=scale)
        z = dequantize(y, dq_dtype=jnp.float32, scale_inv=scale_inv)

        assert_allclose(z, x, dtype=jnp.float8_e4m3fn)

    @pytest.mark.parametrize("m,n,k", GEMM_CASES)
    def test_forward_bf16(self, m, n, k):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 2)
        a = jax.random.normal(subkeys[0], (m, k), jnp.bfloat16)
        b = jax.random.normal(subkeys[1], (k, n), jnp.bfloat16)

        primitive_out = type_safe_dot_general(a, b)
        ref_out = jnp.dot(a, b)

        assert_allclose(primitive_out, ref_out, dtype=jnp.bfloat16)

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize("m,n,k", GEMM_CASES)
    def test_forward_fp8_randint(self, m, n, k):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 2)

        dtype = jnp.bfloat16

        # TODO(rewang): add float random test
        min_val, max_val = -8, 8
        a = jax.random.randint(subkeys[0], (m, k), min_val, max_val).astype(dtype)
        b = jax.random.randint(subkeys[1], (k, n), min_val, max_val).astype(dtype)

        _, amax_list, scale_list = TestFP8Dot._generate_fp8_meta()
        fp8_meta_pkg = FP8MetaPackage(
            amax_list[0],
            scale_list[0],
            amax_list[1],
            scale_list[1],
            amax_list[2],
            scale_list[2],
        )
        primitive_out = type_safe_dot_general(a, b, fp8_meta_pkg)
        ref_out = jnp.dot(a, b)

        ref_out = ref_out.astype(jnp.float32)
        primitive_out = primitive_out.astype(jnp.float32)

        assert_allclose(primitive_out, ref_out, dtype=FP8Helper.FWD_DTYPE)

    @pytest.mark.parametrize("m,n,k", GEMM_CASES)
    def test_grad_bf16(self, m, n, k):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 2)
        a = jax.random.normal(subkeys[0], (m, k), jnp.bfloat16)
        b = jax.random.normal(subkeys[1], (k, n), jnp.bfloat16)

        def primitive_func(x, y):
            primitive_out = type_safe_dot_general(x, y)
            return jnp.mean(primitive_out)

        def ref_func(x, y):
            return jnp.mean(jnp.dot(x, y))

        value_n_grad_primitive_func = value_and_grad(primitive_func, (0, 1))

        value_n_grad_ref_func = value_and_grad(ref_func, (0, 1))

        primitive_out, (primitive_a_grad, primitive_b_grad) = value_n_grad_primitive_func(a, b)
        ref_out, (ref_a_grad, ref_b_grad) = value_n_grad_ref_func(a, b)

        assert_allclose(primitive_out, ref_out, dtype=jnp.bfloat16)
        assert_allclose(primitive_a_grad, ref_a_grad, dtype=jnp.bfloat16)
        assert_allclose(primitive_b_grad, ref_b_grad, dtype=jnp.bfloat16)

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize("m,n,k", GEMM_CASES)
    def test_grad_fp8_dot(self, m, n, k):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 2)

        a = jax.random.normal(subkeys[0], (m, k)).astype(jnp.bfloat16)
        b = jax.random.normal(subkeys[1], (k, n)).astype(jnp.bfloat16)

        _, amax_list, scale_list = TestFP8Dot._generate_fp8_meta()

        def primitive_func(x, y, amax_list, scale_list):
            fp8_meta_pkg = FP8MetaPackage(
                amax_list[0],
                scale_list[0],
                amax_list[1],
                scale_list[1],
                amax_list[2],
                scale_list[2],
            )
            primitive_out = type_safe_dot_general(x, y, fp8_meta_pkg)
            return jnp.mean(primitive_out)

        def ref_func(x, y):
            return jnp.mean(jnp.dot(x, y))

        value_n_grad_primitive_func = value_and_grad(primitive_func, (0, 1, 2, 3))
        value_n_grad_ref_func = value_and_grad(ref_func, (0, 1))

        ref_out, (ref_a_grad, ref_b_grad) = value_n_grad_ref_func(a, b)

        for _ in range(3):
            primitive_out, (primitive_a_grad, primitive_b_grad, amax_list, scale_list) = (
                value_n_grad_primitive_func(a, b, amax_list, scale_list)
            )

        assert_allclose(primitive_out, ref_out, dtype=FP8Helper.FWD_DTYPE)
        assert_allclose(primitive_a_grad, ref_a_grad, dtype=FP8Helper.BWD_DTYPE)
        assert_allclose(primitive_b_grad, ref_b_grad, dtype=FP8Helper.BWD_DTYPE)

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize(
        "m,n,k", [(256, 128, 512), (16384, 1024, 2816), (16384, 2816, 1024), (16384, 1024, 1024)]
    )
    @pytest.mark.parametrize(
        "activation_type",
        [
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
        ],
    )
    @pytest.mark.parametrize("use_bias", [True, False])
    def test_grad_fused_layernorm_fp8_mlp(
        self, m, n, k, activation_type: Sequence[Union[str, Callable]], use_bias: bool
    ):
        """N/a"""
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 6)

        a = jax.random.normal(subkeys[0], (m, k), jnp.bfloat16)
        k1 = jax.random.normal(subkeys[1], (k, len(activation_type), n), jnp.bfloat16) / jnp.sqrt(k)
        k2 = jax.random.normal(subkeys[2], (n, k), jnp.bfloat16) / jnp.sqrt(n)
        s = jax.random.normal(subkeys[5], (k,), jnp.bfloat16)
        if use_bias:
            b1 = jax.random.normal(subkeys[3], (len(activation_type), n), jnp.bfloat16)
            b2 = jax.random.normal(subkeys[4], (k,), jnp.bfloat16)
        else:
            b1 = None
            b2 = None

        def primitive_func(
            x, ln_s, y, z, w, v, amax_list_1, amax_list_2, scale_list_1, scale_list_2
        ):
            # x is input tensor, matrix 2d
            # y, z are weights, matrix 2d
            # out = ((x * y) + w) * z + v
            fp8_meta_pkg_1 = FP8MetaPackage(
                amax_list_1[0],
                scale_list_1[0],
                amax_list_1[1],
                scale_list_1[1],
                amax_list_1[2],
                scale_list_1[2],
            )
            fp8_meta_pkg_2 = FP8MetaPackage(
                amax_list_2[0],
                scale_list_2[0],
                amax_list_2[1],
                scale_list_2[1],
                amax_list_2[2],
                scale_list_2[2],
            )
            return jnp.mean(
                fused_layernorm_fp8_mlp(
                    x,
                    ln_s,
                    None,
                    [y, z],
                    [w, v],
                    [fp8_meta_pkg_1, fp8_meta_pkg_2],
                    "rmsnorm",
                    activation_type=activation_type,
                    use_bias=use_bias,
                )
            )

        def layernorm_fp8_mlp_ref(
            x: jnp.ndarray,
            ln_scale: jnp.ndarray,
            kernel_1: jnp.ndarray,
            kernel_2: jnp.ndarray,
            bias_1: jnp.ndarray,
            bias_2: jnp.ndarray,
            amax_list_1: List[jnp.ndarray],
            amax_list_2: List[jnp.ndarray],
            scale_list_1: List[jnp.ndarray],
            scale_list_2: List[jnp.ndarray],
        ) -> jnp.ndarray:

            x = jnp.asarray(x, jnp.float32)
            mean2 = jnp.mean(jax.lax.square(x), axis=-1, keepdims=True)
            y = jnp.asarray(x * jax.lax.rsqrt(mean2 + 1e-6), jnp.bfloat16)
            ln_out = y * ln_scale
            ln_out = jnp.asarray(ln_out, jnp.bfloat16)

            fp8_meta_pkg_1 = FP8MetaPackage(
                amax_list_1[0],
                scale_list_1[0],
                amax_list_1[1],
                scale_list_1[1],
                amax_list_1[2],
                scale_list_1[2],
            )
            linear_1_out = type_safe_dot_general(ln_out, kernel_1, fp8_meta_pkg_1, ((1,), (0,)))

            if use_bias:
                bias_1_shape = (1,) * (linear_1_out.ndim - bias_1.ndim) + bias_1.shape
                linear_1_out += jnp.reshape(bias_1, bias_1_shape)

            x = jnp.split(linear_1_out, len(activation_type), axis=-2)
            acts = []
            for idx, act_fn in enumerate(activation_type):
                x_i = _convert_to_activation_function(act_fn)(x[idx])
                acts.append(x_i)
            x = functools.reduce(operator.mul, acts)

            x = jnp.asarray(jnp.squeeze(x, axis=-2), jnp.bfloat16)

            fp8_meta_pkg_2 = FP8MetaPackage(
                amax_list_2[0],
                scale_list_2[0],
                amax_list_2[1],
                scale_list_2[1],
                amax_list_2[2],
                scale_list_2[2],
            )
            output = type_safe_dot_general(x, kernel_2, fp8_meta_pkg_2, ((1,), (0,)))

            if use_bias:
                bias_2_shape = (1,) * (output.ndim - bias_2.ndim) + bias_2.shape
                output += jnp.reshape(bias_2, bias_2_shape)

            return output

        def ref_func(x, ln_s, y, z, w, v, amax_list_1, amax_list_2, scale_list_1, scale_list_2):
            return jnp.mean(
                layernorm_fp8_mlp_ref(
                    x, ln_s, y, z, w, v, amax_list_1, amax_list_2, scale_list_1, scale_list_2
                )
            )

        value_n_grad_primitive_func = jit(
            value_and_grad(primitive_func, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
        )
        value_n_grad_ref_func = jit(value_and_grad(ref_func, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)))

        _, amax_list_1, scale_list_1 = TestFP8Dot._generate_fp8_meta()
        _, amax_list_2, scale_list_2 = TestFP8Dot._generate_fp8_meta()

        ref_amax_list_1 = amax_list_1
        ref_scale_list_1 = scale_list_1
        ref_amax_list_2 = amax_list_2
        ref_scale_list_2 = scale_list_2

        primitive_amax_list_1 = amax_list_1
        primitive_scale_list_1 = scale_list_1
        primitive_amax_list_2 = amax_list_2
        primitive_scale_list_2 = scale_list_2

        primitive_amax_list_1, primitive_scale_list_1, primitive_amax_list_2, primitive_scale_list_2

        # Convert str to index as str is not a valid type for JAX JIT
        for _ in range(3):
            ref_out, (
                ref_a_grad,
                ref_s_grad,
                ref_k1_grad,
                ref_k2_grad,
                ref_b1_grad,
                ref_b2_grad,
                ref_amax_list_1,
                ref_amax_list_2,
                ref_scale_list_1,
                ref_scale_list_2,
            ) = value_n_grad_ref_func(
                a,
                s,
                k1,
                k2,
                b1,
                b2,
                ref_amax_list_1,
                ref_amax_list_2,
                ref_scale_list_1,
                ref_scale_list_2,
            )

        for _ in range(3):
            primitive_out, (
                primitive_a_grad,
                primitive_s_grad,
                primitive_k1_grad,
                primitive_k2_grad,
                primitive_b1_grad,
                primitive_b2_grad,
                primitive_amax_list_1,
                primitive_amax_list_2,
                primitive_scale_list_1,
                primitive_scale_list_2,
            ) = value_n_grad_primitive_func(
                a,
                s,
                k1,
                k2,
                b1,
                b2,
                primitive_amax_list_1,
                primitive_amax_list_2,
                primitive_scale_list_1,
                primitive_scale_list_2,
            )

        assert_allclose(primitive_out, ref_out, dtype=FP8Helper.FWD_DTYPE)
        assert_allclose(
            jnp.asarray(primitive_a_grad, np.float32),
            jnp.asarray(ref_a_grad, np.float32),
            dtype=FP8Helper.BWD_DTYPE,
        )
        assert_allclose(
            jnp.asarray(primitive_k1_grad, np.float32),
            jnp.asarray(ref_k1_grad, np.float32),
            dtype=FP8Helper.BWD_DTYPE,
        )
        assert_allclose(
            jnp.asarray(primitive_s_grad, np.float32),
            jnp.asarray(ref_s_grad, np.float32),
            dtype=FP8Helper.BWD_DTYPE,
        )
        assert_allclose(
            jnp.asarray(primitive_k2_grad, np.float32),
            jnp.asarray(ref_k2_grad, np.float32),
            dtype=FP8Helper.BWD_DTYPE,
        )
        if use_bias:
            assert_allclose(
                jnp.asarray(primitive_b2_grad, np.float32),
                jnp.asarray(ref_b2_grad, np.float32),
                dtype=FP8Helper.BWD_DTYPE,
            )
            assert_allclose(
                jnp.asarray(primitive_b1_grad, np.float32),
                jnp.asarray(ref_b1_grad, np.float32),
                dtype=FP8Helper.BWD_DTYPE,
            )


@pytest.fixture(name="random_inputs")
def random_inputs_fixture(shape):
    key = jax.random.PRNGKey(0)
    subkeys = jax.random.split(key, 4)
    out = jax.random.uniform(subkeys[0], shape, jnp.bfloat16, 5, 8)
    return out


class TestActivationLu:

    def ref_func(self, x, activation_type):

        def ref_act_lu(inputs):
            x = jnp.split(inputs, len(activation_type), axis=-2)
            acts = []
            for idx, act_fn in enumerate(activation_type):
                x_i = _convert_to_activation_function(act_fn)(x[idx])
                acts.append(x_i)
            x = functools.reduce(operator.mul, acts)
            return jnp.mean(x)

        ref_act_func = jit(value_and_grad(ref_act_lu, (0,)))
        return ref_act_func(x)

    def primitive_func(self, inputs):
        return jnp.mean(activation_lu(inputs, activation_type=self.activation_type))

    @pytest.mark.parametrize("shape", [(32, 1, 64), (64, 1, 256)])
    @pytest.mark.parametrize(
        "activation_type",
        [
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
        ],
    )
    def test_activation_lu(self, random_inputs, activation_type):
        x = random_inputs
        x = jnp.repeat(x, len(activation_type), axis=1)
        self.activation_type = activation_type

        value_n_grad_primitive_func = jit(value_and_grad(self.primitive_func, (0,)))

        prim_out, (prim_grad,) = value_n_grad_primitive_func(x)
        ref_out, (ref_grad,) = self.ref_func(x, activation_type)

        assert_allclose(prim_out, ref_out, dtype=x.dtype)
        assert_allclose(prim_grad, ref_grad, dtype=x.dtype)


class TestActivationLuFP8(TestActivationLu):

    def prim_func(self, x):
        amax = self.amax
        scale = self.scale
        scale_inv = self.scale_inv
        activation_type = self.activation_type

        @jax.custom_vjp
        def _prim_func(x, _x_t, _dbias, _amax):
            output = _prim_func_fwd(x, _x_t, _dbias, _amax)
            return output

        def _prim_func_fwd(x, _x_t, _dbias, _amax):
            activation_lu_out, _ = tex.act_lu_fp8(
                x, amax, scale, scale_inv, FP8Helper.FWD_DTYPE, activation_type
            )
            activation_lu_out = dequantize(activation_lu_out, x.dtype, scale_inv)
            ctx = x
            return activation_lu_out, ctx

        def _prim_func_bwd(ctx, g):
            x = ctx
            if len(self.activation_type) > 1:  # gated, no bias
                dactivation_lu, dactivation_lu_trans, amax_out = tex.dgated_act_lu_cast_transpose(
                    g, x, amax, scale, scale_inv, FP8Helper.BWD_DTYPE, -1, activation_type
                )
                dbias = jnp.empty(x.shape[-1], x.dtype)
            else:  # not gated, with bias
                dactivation_lu, dactivation_lu_trans, dbias, amax_out = (
                    tex.dact_lu_dbias_cast_transpose(
                        g,
                        x,
                        amax,
                        scale,
                        scale_inv,
                        FP8Helper.BWD_DTYPE,
                        -1,
                        -2,
                        self.activation_type,
                    )
                )
            dactivation_lu = dequantize(dactivation_lu, x.dtype, scale_inv)
            dactivation_lu_trans = dequantize(dactivation_lu_trans, x.dtype, scale_inv)
            ctx = (dactivation_lu, dactivation_lu_trans, dbias, amax_out)
            return ctx

        _prim_func.defvjp(_prim_func_fwd, _prim_func_bwd)

        dx_trans_no_use = jnp.empty([x.shape[i] for i in self.transpose_indices], dtype=x.dtype)
        dbias_no_use = jnp.empty(x.shape[-1], dtype=x.dtype)
        amax_no_use = jnp.zeros(1, jnp.float32)
        value_n_grad_primitive_func = value_and_grad(
            lambda a, b, c, d: jnp.mean(_prim_func(a, b, c, d)), (0, 1, 2, 3)
        )
        return value_n_grad_primitive_func(x, dx_trans_no_use, dbias_no_use, amax_no_use)

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize("shape", [(32, 1, 64), (64, 1, 256)])
    @pytest.mark.parametrize(
        "activation_type",
        [
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
        ],
    )
    def test_activation_lu(self, random_inputs, activation_type):
        self.amax = jnp.zeros(1, jnp.float32)
        self.scale = jnp.ones(1, jnp.float32)
        self.scale_inv = jnp.ones(1, jnp.float32)
        self.activation_type = activation_type
        self.transpose_indices = (1, 2, 0)

        x = random_inputs
        x = jnp.repeat(x, len(activation_type), axis=1)

        prim_out, (prim_grad, prim_grad_trans, dbias, amax) = self.prim_func(x)
        ref_out, (ref_grad,) = self.ref_func(x, activation_type)

        assert_allclose(prim_out, ref_out, dtype=FP8Helper.FWD_DTYPE)
        assert_allclose(amax, jnp.amax(jnp.abs(ref_grad)), rtol=1e-2)
        if "linear" not in activation_type:
            assert_allclose(dbias, jnp.sum(ref_grad, axis=(i for i in range(x.ndim - 1))))
        assert_allclose(prim_grad, ref_grad, dtype=FP8Helper.BWD_DTYPE)
        assert_allclose(
            prim_grad_trans,
            jnp.transpose(ref_grad, self.transpose_indices),
            dtype=FP8Helper.BWD_DTYPE,
        )


class TestNorm:
    """
    Test transformer_engine.jax.layernorm APIs
    """

    @staticmethod
    def _generate_fp8_meta():
        fp8_dtype_list = [FP8Helper.FWD_DTYPE, FP8Helper.FWD_DTYPE, FP8Helper.BWD_DTYPE]
        amax_list = [
            jnp.zeros((FP8Helper.AMAX_HISTORY_LEN,), jnp.float32),
            jnp.zeros((FP8Helper.AMAX_HISTORY_LEN,), jnp.float32),
            jnp.zeros((FP8Helper.AMAX_HISTORY_LEN,), jnp.float32),
        ]
        scale_list = [
            jnp.ones((1,), jnp.float32),
            jnp.ones((1,), jnp.float32),
            jnp.ones((1,), jnp.float32),
        ]
        return fp8_dtype_list, amax_list, scale_list

    def reference_layernorm(self, x, scale, bias, zero_centered_gamma, eps):
        """
        JAX native layernorm implementations
        - bias is not None: layernorm
        - bias is None: rmsnorm
        """
        x_ = jnp.asarray(x, jnp.float32)
        if bias is None:
            mean = 0.0
        else:
            mean = jnp.mean(x_, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x_ - mean), axis=-1, keepdims=True)
        normed_input = (x_ - mean) * jax.lax.rsqrt(var + eps)
        if zero_centered_gamma:
            scale += 1.0
        if bias is None:
            bias = 0.0
        return jnp.asarray(normed_input * scale + bias).astype(x.dtype)

    @pytest.mark.parametrize("n, hidden", LN_CASES)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("ln_type", ["layernorm", "rmsnorm"])
    @pytest.mark.parametrize("zero_centered_gamma", [False, True])
    @pytest.mark.parametrize("epsilon", [1e-2, 1e-6])
    def test_layernorm_forward_backward(
        self, n, hidden, ln_type, zero_centered_gamma, epsilon, dtype
    ):
        """
        Test transformer_engine.jax.layernorm.layernorm
        """
        expect_assert = False
        if ln_type == "rmsnorm" and zero_centered_gamma:
            # zero_centered_gamma is not supported for rmsnorm, expect an assertion.
            expect_assert = True

        with (
            pytest.raises(AssertionError, match=r".*zero_centered_gamma is not supported.*")
            if expect_assert
            else nullcontext()
        ):
            key = jax.random.PRNGKey(0)
            subkeys = jax.random.split(key, 3)

            x = jax.random.uniform(subkeys[0], (n, hidden), dtype, -1, 1)
            gamma_range = (-1, 1) if zero_centered_gamma else (0, 2)
            gamma = jax.random.uniform(subkeys[1], (hidden,), jnp.float32, *gamma_range)
            gamma = jnp.asarray(gamma, dtype)
            if ln_type == "layernorm":
                beta = jax.random.uniform(subkeys[2], (hidden,), jnp.float32, -1, 1)
                beta = jnp.asarray(beta, dtype)
            else:
                beta = None

            def compute_loss(x):
                # Higher precision to compute the loss
                x_ = x.astype(jnp.float32)
                return jnp.mean(jnp.square(x_)).astype(x.dtype)

            jitted_primitive = jit(
                value_and_grad(
                    lambda x, gamma, beta: compute_loss(
                        layernorm(x, gamma, beta, ln_type, zero_centered_gamma, epsilon)
                    ),
                    (0, 1, 2),
                )
            )

            jitted_reference = jit(
                value_and_grad(
                    lambda x, gamma, beta: compute_loss(
                        self.reference_layernorm(x, gamma, beta, zero_centered_gamma, epsilon)
                    ),
                    (0, 1, 2),
                )
            )

            primitive_out, (primitive_dx, primitive_dgamma, primitive_dbeta) = jitted_primitive(
                x, gamma, beta
            )
            reference_out, (reference_dx, reference_dgamma, reference_dbeta) = jitted_reference(
                x, gamma, beta
            )

            assert_allclose(primitive_out, reference_out, dtype=dtype)
            assert_allclose(primitive_dx, reference_dx, dtype=dtype)
            assert_allclose(primitive_dgamma, reference_dgamma, dtype=dtype)
            if beta is not None:
                assert_allclose(primitive_dbeta, reference_dbeta, dtype=dtype)

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize("m,n,k", GEMM_CASES)
    @pytest.mark.parametrize("ln_type", ["layernorm", "rmsnorm"])
    @pytest.mark.parametrize("zero_centered_gamma", [True, False])
    @pytest.mark.parametrize("epsilon", [1e-2, 1e-6])
    def test_ln_fp8_dot_forward_backward(self, m, n, k, ln_type, zero_centered_gamma, epsilon):
        """
        Test transformer_engine.jax.layernorm.layernorm_fp8_dot
        """
        expect_assert = False
        if ln_type == "rmsnorm" and zero_centered_gamma:
            # zero_centered_gamma is not supported for rmsnorm, expect an assertion.
            expect_assert = True

        with (
            pytest.raises(AssertionError, match=r".*zero_centered_gamma is not supported.*")
            if expect_assert
            else nullcontext()
        ):
            key = jax.random.PRNGKey(0)
            subkeys = jax.random.split(key, 4)

            a = jax.random.normal(subkeys[0], (m, k)).astype(jnp.bfloat16)
            b = jax.random.normal(subkeys[1], (k, n)).astype(jnp.bfloat16)

            gamma = jax.random.normal(subkeys[2], (k,)).astype(jnp.bfloat16)
            if ln_type == "layernorm":
                beta = jax.random.normal(subkeys[3], (k,)).astype(jnp.bfloat16)
            else:
                beta = None

            _, amax_list_1, scale_list_1 = TestNorm._generate_fp8_meta()

            def primitive_func(x, y, gamma, beta, amax_list_1, scale_list_1):
                fp8_meta_pkg = FP8MetaPackage(
                    amax_list_1[0],
                    scale_list_1[0],
                    amax_list_1[1],
                    scale_list_1[1],
                    amax_list_1[2],
                    scale_list_1[2],
                )
                primitive_out = layernorm_fp8_dot(
                    x, y, gamma, beta, fp8_meta_pkg, ln_type, zero_centered_gamma
                )
                return jnp.mean(primitive_out)

            def ref_func(x, y, gamma, beta, zero_centered_gamma):
                x = self.reference_layernorm(x, gamma, beta, zero_centered_gamma, epsilon)
                return jnp.mean(jnp.dot(x, y))

            value_n_grad_primitive_func = value_and_grad(primitive_func, range(6))
            value_n_grad_ref_func = value_and_grad(ref_func, (0, 1, 2, 3))

            ref_out, (ref_a_grad, ref_b_grad, ref_gamma_grad, ref_beta_grad) = (
                value_n_grad_ref_func(a, b, gamma, beta, zero_centered_gamma)
            )

            for _ in range(3):
                primitive_out, (
                    primitive_a_grad,
                    primitive_b_grad,
                    primitive_gamma_grad,
                    primitive_beta_grad,
                    amax_list_1,
                    scale_list_1,
                ) = value_n_grad_primitive_func(a, b, gamma, beta, amax_list_1, scale_list_1)

            assert_allclose(primitive_out, ref_out, dtype=FP8Helper.FWD_DTYPE)
            assert_allclose(primitive_a_grad, ref_a_grad, dtype=FP8Helper.BWD_DTYPE)
            assert_allclose(primitive_b_grad, ref_b_grad, dtype=FP8Helper.BWD_DTYPE)
            assert_allclose(primitive_gamma_grad, ref_gamma_grad, dtype=FP8Helper.BWD_DTYPE)
            if beta is not None:
                assert_allclose(primitive_beta_grad, ref_beta_grad, dtype=FP8Helper.BWD_DTYPE)
