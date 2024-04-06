# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import functools
import operator

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax import jit, value_and_grad
from flax import linen as nn

from utils import assert_allclose
from transformer_engine.jax.cpp_extensions import dgelu, dgelu_dbias_cast_transpose
from transformer_engine.jax.cpp_extensions import gelu, gelu_fp8
from transformer_engine.jax.cpp_extensions import dgated_gelu, gated_gelu
from transformer_engine.jax.cpp_extensions import dgated_gelu_cast_transpose, gated_gelu_fp8
from transformer_engine.jax.dot import type_safe_dot_general, dequantize, quantize
from transformer_engine.jax.fp8 import FP8MetaPackage, FP8Helper
from transformer_engine.jax.fp8 import is_fp8_available
from transformer_engine.jax.layernorm import layernorm
from transformer_engine.jax.mlp import layernorm_geglu_fp8_mlp
from transformer_engine.jax.mlp import layernorm_gelu_fp8_mlp

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


@pytest.fixture(autouse=True, scope='function')
def clear_live_arrays():
    """
    Clear all live arrays to keep the resource clean
    """
    yield
    for arr in jax.live_arrays():
        arr.delete()


class TestFP8Dot:

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

    @pytest.mark.parametrize('m,n,k', GEMM_CASES)
    def test_forward_bf16(self, m, n, k):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 2)
        a = jax.random.normal(subkeys[0], (m, k), jnp.bfloat16)
        b = jax.random.normal(subkeys[1], (k, n), jnp.bfloat16)

        primitive_out = type_safe_dot_general(a, b)
        ref_out = jnp.dot(a, b)

        assert_allclose(primitive_out, ref_out, dtype=jnp.bfloat16)

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize('m,n,k', GEMM_CASES)
    def test_forward_fp8_randint(self, m, n, k):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 2)

        # TODO(rewang): add float random test
        min_val, max_val = -8, 8
        a = jax.random.randint(subkeys[0], (m, k), min_val, max_val).astype(jnp.bfloat16)
        b = jax.random.randint(subkeys[1], (k, n), min_val, max_val).astype(jnp.bfloat16)

        fp8_max = FP8Helper.generate_fp8_max_array(FP8Helper.NUM_META_PER_GEMM)
        fp8_metas_amax = jnp.zeros((FP8Helper.NUM_META_PER_GEMM, FP8Helper.AMAX_HISTORY_LEN),
                                   jnp.float32)
        fp8_metas_scale = jnp.ones((FP8Helper.NUM_META_PER_GEMM, 1), jnp.float32)
        fp8_metas_scale_inv = jnp.ones((FP8Helper.NUM_META_PER_GEMM, 1), jnp.float32)
        fp8_meta_pkg = FP8MetaPackage(1, fp8_max, fp8_metas_amax, fp8_metas_scale,
                                      fp8_metas_scale_inv)

        primitive_out = type_safe_dot_general(a, b, fp8_meta_pkg)

        # calculate scale by amax
        fp8_metas_scale, fp8_metas_scale_inv = FP8Helper.update_fp8_scale(
            fp8_max, fp8_metas_amax, fp8_metas_scale)
        fp8_meta_pkg = FP8MetaPackage(1, fp8_max, fp8_metas_amax, fp8_metas_scale,
                                      fp8_metas_scale_inv)

        primitive_out = type_safe_dot_general(a, b, fp8_meta_pkg)
        ref_out = jnp.dot(a, b)

        ref_out = ref_out.astype(jnp.float32)
        primitive_out = primitive_out.astype(jnp.float32)

        assert_allclose(primitive_out, ref_out, dtype=FP8Helper.FWD_DTYPE)

    @pytest.mark.parametrize('m,n,k', GEMM_CASES)
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
    @pytest.mark.parametrize('m,n,k', GEMM_CASES)
    def test_grad_fp8_dot(self, m, n, k):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 2)

        a = jax.random.normal(subkeys[0], (m, k)).astype(jnp.bfloat16)
        b = jax.random.normal(subkeys[1], (k, n)).astype(jnp.bfloat16)

        fp8_max = FP8Helper.generate_fp8_max_array(FP8Helper.NUM_META_PER_GEMM)
        fp8_metas_amax = jnp.zeros((FP8Helper.NUM_META_PER_GEMM, FP8Helper.AMAX_HISTORY_LEN),
                                   jnp.float32)
        fp8_metas_scale = jnp.ones((FP8Helper.NUM_META_PER_GEMM, 1), jnp.float32)
        fp8_metas_scale_inv = jnp.ones((FP8Helper.NUM_META_PER_GEMM, 1), jnp.float32)

        def primitive_func(x, y, fp8_max, fp8_metas_amax, fp8_metas_scale, fp8_metas_scale_inv):
            fp8_meta_pkg = FP8MetaPackage(1, fp8_max, fp8_metas_amax, fp8_metas_scale,
                                          fp8_metas_scale_inv)
            primitive_out = type_safe_dot_general(x, y, fp8_meta_pkg)
            return jnp.mean(primitive_out)

        def ref_func(x, y):
            return jnp.mean(jnp.dot(x, y))

        value_n_grad_primitive_func = value_and_grad(primitive_func, (0, 1, 2, 3, 4, 5))
        value_n_grad_ref_func = value_and_grad(ref_func, (0, 1))

        ref_out, (ref_a_grad, ref_b_grad) = value_n_grad_ref_func(a, b)

        for _ in range(3):
            primitive_out, (primitive_a_grad, primitive_b_grad, fp8_max, fp8_metas_amax,
                            fp8_metas_scale, fp8_metas_scale_inv) = value_n_grad_primitive_func(
                                a, b, fp8_max, fp8_metas_amax, fp8_metas_scale, fp8_metas_scale_inv)

        assert_allclose(primitive_out, ref_out, dtype=FP8Helper.FWD_DTYPE)
        assert_allclose(primitive_a_grad, ref_a_grad, dtype=FP8Helper.BWD_DTYPE)
        assert_allclose(primitive_b_grad, ref_b_grad, dtype=FP8Helper.BWD_DTYPE)

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize('m,n,k', [(256, 256, 512), (16384, 1024, 2816), (16384, 2816, 1024),
                                       (16384, 1024, 1024)])
    def test_grad_ln_geglu_fp8_mlp(self, m, n, k):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 4)
        activations = ('gelu', 'linear')

        a = jax.random.normal(subkeys[0], (m, k), jnp.bfloat16)
        k1 = jax.random.normal(subkeys[1], (k, len(activations), n), jnp.bfloat16)
        k2 = jax.random.normal(subkeys[2], (n, k), jnp.bfloat16)
        s = jax.random.normal(subkeys[3], (k,), jnp.bfloat16)

        init_fp8_max = FP8Helper.generate_fp8_max_array(FP8Helper.NUM_META_PER_GEMM * 2)
        init_fp8_metas_amax = jnp.zeros(
            (FP8Helper.NUM_META_PER_GEMM * 2, FP8Helper.AMAX_HISTORY_LEN), jnp.float32)
        init_fp8_metas_scale = jnp.ones((FP8Helper.NUM_META_PER_GEMM * 2, 1), jnp.float32)
        init_fp8_metas_scale_inv = jnp.ones((FP8Helper.NUM_META_PER_GEMM * 2, 1), jnp.float32)

        def primitive_func(x, ln_s, y, z, fp8_max, fp8_metas_amax, fp8_metas_scale,
                           fp8_metas_scale_inv):
            # x is input tensor, matrix 2d
            # y, z are weights, matrix 2d
            # out = (x * y) * z
            fp8_meta_pkg = FP8MetaPackage(2, fp8_max, fp8_metas_amax, fp8_metas_scale,
                                          fp8_metas_scale_inv)
            return jnp.mean(layernorm_geglu_fp8_mlp(x, ln_s, None, [y, z], fp8_meta_pkg, "rmsnorm"))

        def _convert_to_activation_function(fn_or_string):
            """Convert a string to an activation function."""
            if fn_or_string == 'linear':
                return lambda x: x
            if isinstance(fn_or_string, str):
                return getattr(nn, fn_or_string)
            if callable(fn_or_string):
                return fn_or_string
            raise ValueError(f"don't know how to convert {fn_or_string} to an activation function")

        def ln_geglu_fp8_mlp_ref(x: jnp.ndarray, ln_scale: jnp.ndarray, kernel_1: jnp.ndarray,
                                 kernel_2: jnp.ndarray, fp8_maxs: jnp.ndarray, amax: jnp.ndarray,
                                 scale: jnp.ndarray, scale_inv: jnp.ndarray) -> jnp.ndarray:

            x = jnp.asarray(x, jnp.float32)
            mean2 = jnp.mean(jax.lax.square(x), axis=-1, keepdims=True)
            y = jnp.asarray(x * jax.lax.rsqrt(mean2 + 1e-6), jnp.bfloat16)
            ln_out = y * ln_scale
            ln_out = jnp.asarray(ln_out, jnp.bfloat16)

            fp8_gemm_1_pkg = FP8MetaPackage(1, fp8_maxs[:FP8Helper.NUM_META_PER_GEMM],
                                            amax[:FP8Helper.NUM_META_PER_GEMM],
                                            scale[:FP8Helper.NUM_META_PER_GEMM],
                                            scale_inv[:FP8Helper.NUM_META_PER_GEMM])
            linear_1_out = type_safe_dot_general(ln_out, kernel_1, fp8_gemm_1_pkg, ((1,), (0,)))

            x = jnp.split(linear_1_out, len(activations), axis=-2)
            acts = []
            for idx, act_fn in enumerate(activations):
                x_i = _convert_to_activation_function(act_fn)(x[idx])
                acts.append(x_i)
            x = functools.reduce(operator.mul, acts)
            x = jnp.asarray(jnp.squeeze(x, axis=-2), jnp.bfloat16)

            fp8_gemm_2_pkg = FP8MetaPackage(1, fp8_maxs[FP8Helper.NUM_META_PER_GEMM:],
                                            amax[FP8Helper.NUM_META_PER_GEMM:],
                                            scale[FP8Helper.NUM_META_PER_GEMM:],
                                            scale_inv[FP8Helper.NUM_META_PER_GEMM:])
            output = type_safe_dot_general(x, kernel_2, fp8_gemm_2_pkg, ((1,), (0,)))
            return output

        def ref_func(x, ln_s, y, z, fp8_max, fp8_metas_amax, fp8_metas_scale, fp8_metas_scale_inv):
            return jnp.mean(
                ln_geglu_fp8_mlp_ref(x, ln_s, y, z, fp8_max, fp8_metas_amax, fp8_metas_scale,
                                     fp8_metas_scale_inv))

        value_n_grad_primitive_func = jit(value_and_grad(primitive_func, (0, 1, 2, 3, 4, 5, 6, 7)))
        value_n_grad_ref_func = jit(value_and_grad(ref_func, (0, 1, 2, 3, 4, 5, 6, 7)))

        ref_fp8_max = init_fp8_max
        ref_fp8_metas_amax = init_fp8_metas_amax
        ref_fp8_metas_scale = init_fp8_metas_scale
        ref_fp8_metas_scale_inv = init_fp8_metas_scale_inv

        pri_fp8_max = init_fp8_max
        pri_fp8_metas_amax = init_fp8_metas_amax
        pri_fp8_metas_scale = init_fp8_metas_scale
        pri_fp8_metas_scale_inv = init_fp8_metas_scale_inv

        for _ in range(3):
            ref_out, (ref_a_grad, ref_s_grad, ref_k1_grad, ref_k2_grad, ref_fp8_max,
                      ref_fp8_metas_amax, ref_fp8_metas_scale,
                      ref_fp8_metas_scale_inv) = value_n_grad_ref_func(
                          a, s, k1, k2, ref_fp8_max, ref_fp8_metas_amax, ref_fp8_metas_scale,
                          ref_fp8_metas_scale_inv)

        for _ in range(3):
            primitive_out, (primitive_a_grad, primitive_s_grad, primitive_k1_grad,
                            primitive_k2_grad, pri_fp8_max, pri_fp8_metas_amax, pri_fp8_metas_scale,
                            pri_fp8_metas_scale_inv) = value_n_grad_primitive_func(
                                a, s, k1, k2, pri_fp8_max, pri_fp8_metas_amax, pri_fp8_metas_scale,
                                pri_fp8_metas_scale_inv)

        assert_allclose(primitive_out, ref_out, dtype=FP8Helper.FWD_DTYPE)
        assert_allclose(jnp.asarray(primitive_a_grad, np.float32),
                        jnp.asarray(ref_a_grad, np.float32),
                        dtype=FP8Helper.BWD_DTYPE)
        assert_allclose(jnp.asarray(primitive_k1_grad, np.float32),
                        jnp.asarray(ref_k1_grad, np.float32),
                        dtype=FP8Helper.BWD_DTYPE)
        assert_allclose(jnp.asarray(primitive_k2_grad, np.float32),
                        jnp.asarray(ref_k2_grad, np.float32),
                        dtype=FP8Helper.BWD_DTYPE)
        assert_allclose(jnp.asarray(primitive_s_grad, np.float32),
                        jnp.asarray(ref_s_grad, np.float32),
                        dtype=FP8Helper.BWD_DTYPE)

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize('m,n,k', [(256, 256, 512), (16384, 1024, 2816), (16384, 2816, 1024),
                                       (16384, 1024, 1024)])
    def test_grad_ln_gelu_fp8_mlp(self, m, n, k):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 6)
        activations = ('gelu',)

        a = jax.random.normal(subkeys[0], (m, k), jnp.bfloat16)
        k1 = jax.random.normal(subkeys[1], (k, len(activations), n), jnp.bfloat16)
        k2 = jax.random.normal(subkeys[2], (n, k), jnp.bfloat16)
        b1 = jax.random.normal(subkeys[3], (len(activations), n), jnp.bfloat16)
        b2 = jax.random.normal(subkeys[4], (k,), jnp.bfloat16)
        s = jax.random.normal(subkeys[5], (k,), jnp.bfloat16)

        init_fp8_max = FP8Helper.generate_fp8_max_array(FP8Helper.NUM_META_PER_GEMM * 2)
        init_fp8_metas_amax = jnp.zeros(
            (FP8Helper.NUM_META_PER_GEMM * 2, FP8Helper.AMAX_HISTORY_LEN), jnp.float32)
        init_fp8_metas_scale = jnp.ones((FP8Helper.NUM_META_PER_GEMM * 2, 1), jnp.float32)
        init_fp8_metas_scale_inv = jnp.ones((FP8Helper.NUM_META_PER_GEMM * 2, 1), jnp.float32)

        def primitive_func(x, ln_s, y, z, w, v, fp8_max, fp8_metas_amax, fp8_metas_scale,
                           fp8_metas_scale_inv):
            # x is input tensor, matrix 2d
            # y, z are weights, matrix 2d
            # out = ((x * y) + w) * z + v
            fp8_meta_pkg = FP8MetaPackage(2, fp8_max, fp8_metas_amax, fp8_metas_scale,
                                          fp8_metas_scale_inv)
            return jnp.mean(
                layernorm_gelu_fp8_mlp(x, ln_s, None, [y, z], [w, v], fp8_meta_pkg, "rmsnorm"))

        def ln_gelu_fp8_mlp_ref(x: jnp.ndarray, ln_scale: jnp.ndarray, kernel_1: jnp.ndarray,
                                kernel_2: jnp.ndarray, bias_1: jnp.ndarray, bias_2: jnp.ndarray,
                                fp8_maxs: jnp.ndarray, amax: jnp.ndarray, scale: jnp.ndarray,
                                scale_inv: jnp.ndarray) -> jnp.ndarray:

            x = jnp.asarray(x, jnp.float32)
            mean2 = jnp.mean(jax.lax.square(x), axis=-1, keepdims=True)
            y = jnp.asarray(x * jax.lax.rsqrt(mean2 + 1e-6), jnp.bfloat16)
            ln_out = y * ln_scale
            ln_out = jnp.asarray(ln_out, jnp.bfloat16)

            fp8_gemm_1_pkg = FP8MetaPackage(1, fp8_maxs[:FP8Helper.NUM_META_PER_GEMM],
                                            amax[:FP8Helper.NUM_META_PER_GEMM],
                                            scale[:FP8Helper.NUM_META_PER_GEMM],
                                            scale_inv[:FP8Helper.NUM_META_PER_GEMM])
            linear_1_out = type_safe_dot_general(ln_out, kernel_1, fp8_gemm_1_pkg, ((1,), (0,)))

            bias_1_shape = (1,) * (linear_1_out.ndim - bias_1.ndim) + bias_1.shape
            linear_1_out += jnp.reshape(bias_1, bias_1_shape)

            x = jax.nn.gelu(linear_1_out)
            x = jnp.asarray(jnp.squeeze(x, axis=-2), jnp.bfloat16)

            fp8_gemm_2_pkg = FP8MetaPackage(1, fp8_maxs[FP8Helper.NUM_META_PER_GEMM:],
                                            amax[FP8Helper.NUM_META_PER_GEMM:],
                                            scale[FP8Helper.NUM_META_PER_GEMM:],
                                            scale_inv[FP8Helper.NUM_META_PER_GEMM:])
            output = type_safe_dot_general(x, kernel_2, fp8_gemm_2_pkg, ((1,), (0,)))

            bias_2_shape = (1,) * (output.ndim - bias_2.ndim) + bias_2.shape
            output += jnp.reshape(bias_2, bias_2_shape)

            return output

        def ref_func(x, ln_s, y, z, w, v, fp8_max, fp8_metas_amax, fp8_metas_scale,
                     fp8_metas_scale_inv):
            return jnp.mean(
                ln_gelu_fp8_mlp_ref(x, ln_s, y, z, w, v, fp8_max, fp8_metas_amax, fp8_metas_scale,
                                    fp8_metas_scale_inv))

        value_n_grad_primitive_func = jit(
            value_and_grad(primitive_func, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)))
        value_n_grad_ref_func = jit(value_and_grad(ref_func, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)))

        ref_fp8_max = init_fp8_max
        ref_fp8_metas_amax = init_fp8_metas_amax
        ref_fp8_metas_scale = init_fp8_metas_scale
        ref_fp8_metas_scale_inv = init_fp8_metas_scale_inv

        pri_fp8_max = init_fp8_max
        pri_fp8_metas_amax = init_fp8_metas_amax
        pri_fp8_metas_scale = init_fp8_metas_scale
        pri_fp8_metas_scale_inv = init_fp8_metas_scale_inv

        for _ in range(3):
            ref_out, (ref_a_grad, ref_s_grad, ref_k1_grad, ref_k2_grad, ref_b1_grad, ref_b2_grad,
                      ref_fp8_max, ref_fp8_metas_amax, ref_fp8_metas_scale,
                      ref_fp8_metas_scale_inv) = value_n_grad_ref_func(
                          a, s, k1, k2, b1, b2, ref_fp8_max, ref_fp8_metas_amax,
                          ref_fp8_metas_scale, ref_fp8_metas_scale_inv)

        for _ in range(3):
            primitive_out, (primitive_a_grad, primitive_s_grad, primitive_k1_grad,
                            primitive_k2_grad, primitive_b1_grad, primitive_b2_grad, pri_fp8_max,
                            pri_fp8_metas_amax, pri_fp8_metas_scale,
                            pri_fp8_metas_scale_inv) = value_n_grad_primitive_func(
                                a, s, k1, k2, b1, b2, pri_fp8_max, pri_fp8_metas_amax,
                                pri_fp8_metas_scale, pri_fp8_metas_scale_inv)

        assert_allclose(primitive_out, ref_out, dtype=FP8Helper.FWD_DTYPE)
        assert_allclose(jnp.asarray(primitive_a_grad, np.float32),
                        jnp.asarray(ref_a_grad, np.float32),
                        dtype=FP8Helper.BWD_DTYPE)
        assert_allclose(jnp.asarray(primitive_k1_grad, np.float32),
                        jnp.asarray(ref_k1_grad, np.float32),
                        dtype=FP8Helper.BWD_DTYPE)
        assert_allclose(jnp.asarray(primitive_k2_grad, np.float32),
                        jnp.asarray(ref_k2_grad, np.float32),
                        dtype=FP8Helper.BWD_DTYPE)
        assert_allclose(jnp.asarray(primitive_s_grad, np.float32),
                        jnp.asarray(ref_s_grad, np.float32),
                        dtype=FP8Helper.BWD_DTYPE)
        assert_allclose(jnp.asarray(primitive_b1_grad, np.float32),
                        jnp.asarray(ref_b1_grad, np.float32),
                        dtype=jnp.bfloat16)
        assert_allclose(jnp.asarray(primitive_b2_grad, np.float32),
                        jnp.asarray(ref_b2_grad, np.float32),
                        dtype=jnp.bfloat16)


@pytest.fixture(name="random_inputs")
def random_inputs_fixture(shape):
    key = jax.random.PRNGKey(0)
    subkeys = jax.random.split(key, 4)
    out = jax.random.uniform(subkeys[0], shape, jnp.bfloat16, 5, 8)
    return out


class TestGeLu:

    def ref_func(self, inputs):

        func = jit(value_and_grad(lambda x: jnp.mean(jax.nn.gelu(x))))
        return func(inputs)

    def prim_func(self, inputs):

        @jax.custom_vjp
        def primitive(x):
            out, _ = primitive_fwd(x)
            return out

        def primitive_fwd(x):
            out = gelu(x)
            ctx = x
            return out, ctx

        def primitive_bwd(ctx, g):
            x = ctx
            out = dgelu(g, x)
            return (out,)

        primitive.defvjp(primitive_fwd, primitive_bwd)
        func = value_and_grad(lambda x: jnp.mean(primitive(x)))
        return func(inputs)

    @pytest.mark.parametrize('shape', [(32, 2, 64), (64, 2, 256)])
    def test_gelu(self, random_inputs):
        x = random_inputs
        prim_out, prim_grad = self.prim_func(x)
        ref_out, ref_grad = self.ref_func(x)

        assert_allclose(prim_out, ref_out, dtype=x.dtype)
        assert_allclose(prim_grad, ref_grad, dtype=x.dtype)


class TestGeLuFP8(TestGeLu):

    def prim_func(self, inputs):
        amax = self.amax
        scale = self.scale
        scale_inv = self.scale_inv
        no_use = jnp.zeros(1, jnp.float32)

        @jax.custom_vjp
        def primitive(x, y, z, w):
            out = primitive_fwd(x)
            return out

        def primitive_fwd(x, y, z, w):
            out, _ = gelu_fp8(x, amax, scale, scale_inv, jnp.float8_e4m3fn)
            out = dequantize(out, x.dtype, scale_inv)
            ctx = x
            return out, ctx

        def primitive_bwd(ctx, g):
            x = ctx
            dgelu, dgelu_trans, dbias, amax_out = dgelu_dbias_cast_transpose(
                g, x, amax, scale, scale_inv, jnp.float8_e5m2, -1)
            dgelu = dequantize(dgelu, x.dtype, scale_inv)
            dgelu_trans = dequantize(dgelu_trans, x.dtype, scale_inv)
            return dgelu, dgelu_trans, dbias, amax_out

        primitive.defvjp(primitive_fwd, primitive_bwd)
        func = value_and_grad(lambda x, y, z, w: jnp.mean(primitive(x, y, z, w)), (0, 1, 2, 3))

        return func(inputs, jnp.transpose(inputs, (2, 0, 1)),
                    jnp.zeros(inputs.shape[-1], dtype=inputs.dtype), no_use)

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize('shape', [(32, 2, 64), (64, 2, 256)])
    def test_gelu(self, random_inputs):
        self.amax = jnp.zeros(1, jnp.float32)
        self.scale = jnp.ones(1, jnp.float32)
        self.scale_inv = jnp.ones(1, jnp.float32)

        x = random_inputs
        prim_out, (prim_grad, prim_grad_trans, dbias, amax) = self.prim_func(x)
        ref_out, ref_grad = self.ref_func(x)

        assert_allclose(prim_out, ref_out, dtype=FP8Helper.FWD_DTYPE)
        assert_allclose(amax, jnp.amax(jnp.abs(ref_grad)), rtol=1e-2)
        assert_allclose(dbias, jnp.sum(ref_grad, axis=(i for i in range(x.ndim - 1))))
        assert_allclose(prim_grad, ref_grad, dtype=FP8Helper.BWD_DTYPE)
        assert_allclose(prim_grad_trans,
                        jnp.transpose(ref_grad, (2, 0, 1)),
                        dtype=FP8Helper.BWD_DTYPE)


class TestGatedGeLu:

    def ref_func(self, inputs):

        def jax_gated_gelu(x):
            x = jnp.split(x, 2, axis=-2)
            acts = [jax.nn.gelu(x[0]), x[1]]
            x = functools.reduce(operator.mul, acts)
            x = jnp.asarray(jnp.squeeze(x, -2), jnp.bfloat16)
            return x

        func = jit(value_and_grad(lambda x: jnp.mean(jax_gated_gelu(x))))
        return func(inputs)

    def prim_func(self, inputs):

        @jax.custom_vjp
        def primitive(x):
            out, _ = primitive_fwd(x)
            return out

        def primitive_fwd(x):
            out = gated_gelu(x)
            ctx = x
            return out, ctx

        def primitive_bwd(ctx, g):
            x = ctx
            out = dgated_gelu(g, x)
            return (out,)

        primitive.defvjp(primitive_fwd, primitive_bwd)
        func = value_and_grad(lambda x: jnp.mean(primitive(x)))
        return func(inputs)

    @pytest.mark.parametrize('shape', [(32, 2, 64), (64, 2, 256)])
    def test_gated_gelu(self, random_inputs):
        x = random_inputs
        prim_out, prim_grad = self.prim_func(x)
        ref_out, ref_grad = self.ref_func(x)

        assert_allclose(prim_out, ref_out, dtype=x.dtype)
        assert_allclose(prim_grad, ref_grad, dtype=x.dtype)


class TestGatedGeLuFP8(TestGatedGeLu):

    def prim_func(self, inputs):
        amax = self.amax
        scale = self.scale
        scale_inv = self.scale_inv
        no_use = jnp.zeros(1, jnp.float32)

        @jax.custom_vjp
        def primitive(x, y, z):
            out = primitive_fwd(x)
            return out

        def primitive_fwd(x, y, z):
            out, _ = gated_gelu_fp8(x, amax, scale, scale_inv, jnp.float8_e4m3fn)
            out = dequantize(out, x.dtype, scale_inv)
            ctx = x
            return out, ctx

        def primitive_bwd(ctx, g):
            x = ctx
            dgelu, dgelu_trans, amax_out = dgated_gelu_cast_transpose(g, x, amax, scale, scale_inv,
                                                                      jnp.float8_e5m2, -1)
            dgelu = dequantize(dgelu, x.dtype, scale_inv)
            dgelu_trans = dequantize(dgelu_trans, x.dtype, scale_inv)
            return dgelu, dgelu_trans, amax_out

        primitive.defvjp(primitive_fwd, primitive_bwd)
        func = value_and_grad(lambda x, y, z: jnp.mean(primitive(x, y, z)), (0, 1, 2))

        return func(inputs, jnp.transpose(inputs, (1, 2, 0)), no_use)

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize('shape', [(32, 2, 64), (64, 2, 256)])
    def test_gated_gelu(self, random_inputs):
        self.amax = jnp.zeros(1, jnp.float32)
        self.scale = jnp.ones(1, jnp.float32)
        self.scale_inv = jnp.ones(1, jnp.float32)

        x = random_inputs
        prim_out, (prim_grad, prim_grad_trans, amax) = self.prim_func(x)
        ref_out, ref_grad = self.ref_func(x)

        assert_allclose(prim_out, ref_out, dtype=FP8Helper.FWD_DTYPE)
        assert_allclose(amax, jnp.amax(jnp.abs(ref_grad)), rtol=1e-2)
        assert_allclose(prim_grad, ref_grad, dtype=FP8Helper.BWD_DTYPE)
        assert_allclose(prim_grad_trans,
                        jnp.transpose(ref_grad, (1, 2, 0)),
                        dtype=FP8Helper.BWD_DTYPE)


class TestRMSNorm:

    @pytest.mark.parametrize('n, hidden', LN_CASES)
    @pytest.mark.parametrize('dtype', DTYPES)
    def test_forward_backward(self, n, hidden, dtype):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 2)

        x = jax.random.uniform(subkeys[0], (n, hidden), dtype, -2, 1)
        scale = jax.random.uniform(subkeys[1], (hidden,), jnp.float32, -2, 1)
        scale = jnp.asarray(scale, dtype)
        epsilon = 1e-6

        def reference_rmsnorm(x, scale):
            x = jnp.asarray(x, jnp.float32)
            mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
            y = jnp.asarray(x * lax.rsqrt(mean2 + epsilon), dtype)
            return y * scale

        jitted_primitive = jit(
            value_and_grad(lambda x, scale: jnp.mean(layernorm(x, scale, None, "rmsnorm")), (0, 1)))

        jitted_reference = jit(
            value_and_grad(lambda x, scale: jnp.mean(reference_rmsnorm(x, scale)), (0, 1)))

        primitive_out, (primitive_dx, primitive_dgamma) = jitted_primitive(x, scale)
        reference_out, (reference_dx, reference_dgamma) = jitted_reference(x, scale)

        assert_allclose(primitive_out, reference_out, dtype=dtype)
        assert_allclose(primitive_dx, reference_dx, dtype=dtype)
        assert_allclose(primitive_dgamma, reference_dgamma, dtype=dtype)


class TestLayerNorm:

    @pytest.mark.parametrize('n, hidden', LN_CASES)
    @pytest.mark.parametrize('dtype', DTYPES)
    @pytest.mark.parametrize('zero_centered_gamma', [False, True])
    def test_forward_backward(self, n, hidden, zero_centered_gamma, dtype):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 3)

        x = jax.random.uniform(subkeys[0], (n, hidden), dtype, -1, 1)
        scale_range = (-1, 1) if zero_centered_gamma else (0, 2)
        scale = jax.random.uniform(subkeys[1], (hidden,), jnp.float32, *scale_range)
        scale = jnp.asarray(scale, dtype)
        bias = jax.random.uniform(subkeys[2], (hidden,), jnp.float32, -1, 1)
        bias = jnp.asarray(bias, dtype)
        epsilon = 1e-6

        def reference_layernorm(x, scale, bias, zero_centered_gamma, eps):
            x_ = jnp.asarray(x, jnp.float32)
            mean = jnp.mean(x_, axis=-1, keepdims=True)
            var = jnp.mean(jnp.square(x_ - mean), axis=-1, keepdims=True)
            normed_input = (x_ - mean) * jax.lax.rsqrt(var + eps)
            # Align TE implementation
            if zero_centered_gamma:
                return jnp.asarray(normed_input * (scale + 1) + bias).astype(x.dtype)
            return jnp.asarray(normed_input * scale + bias).astype(x.dtype)

        def compute_loss(x):
            # Higher precision to compute the loss
            x_ = x.astype(jnp.float32)
            return jnp.mean(jnp.square(x_)).astype(x.dtype)

        jitted_primitive = jit(
            value_and_grad(
                lambda x, scale, bias: compute_loss(
                    layernorm(x, scale, bias, "layernorm", zero_centered_gamma, epsilon)),
                (0, 1, 2)))

        jitted_reference = jit(
            value_and_grad(
                lambda x, scale, bias: compute_loss(
                    reference_layernorm(x, scale, bias, zero_centered_gamma, epsilon)), (0, 1, 2)))

        primitive_out, (primitive_dx, primitive_dgamma,
                        primitive_dbeta) = jitted_primitive(x, scale, bias)
        reference_out, (reference_dx, reference_dgamma,
                        reference_dbeta) = jitted_reference(x, scale, bias)

        assert_allclose(primitive_out, reference_out, dtype=dtype)
        assert_allclose(primitive_dx, reference_dx, dtype=dtype)
        assert_allclose(primitive_dgamma, reference_dgamma, dtype=dtype)
        assert_allclose(primitive_dbeta, reference_dbeta, dtype=dtype)
