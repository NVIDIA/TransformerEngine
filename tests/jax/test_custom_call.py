# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring

import functools
import operator
import pytest
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax import value_and_grad, jit
from flax import linen as nn

from utils import is_fp8_supported

from transformer_engine.common.recipe import Format
from transformer_engine.jax.cpp_extensions import te_gated_gelu, te_cast_transpose_dgated_gelu
from transformer_engine.jax.fp8 import FP8Helper, _get_ctypes, DType
from transformer_engine.jax.dot import fp8_dot
from transformer_engine.jax.mlp import fp8_ln_mlp

GEMM_CASES = [(256, 256, 512), (32, 32, 32), (16384, 1024, 2816),
              (16384, 2816, 1024), (16384, 1024, 1024)]
FP8_COMPUTE_TYPE = [_get_ctypes(Format.E4M3), _get_ctypes(Format.HYBRID)]


class TestFP8Dot():

    def test_compile_bf16(self):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 2)
        a = jax.random.normal(subkeys[0], (256, 512), jnp.bfloat16)
        b = jax.random.normal(subkeys[1], (512, 256), jnp.bfloat16)

        def func(x, y):
            fp8_max = FP8Helper.generate_fp8_max_array(
                FP8Helper.NUM_META_PER_GEMM)
            fp8_metas_amax = jnp.zeros(
                (FP8Helper.NUM_META_PER_GEMM, FP8Helper.AMAX_HISTORY_SIZE),
                jnp.float32)
            fp8_metas_scale = jnp.ones((FP8Helper.NUM_META_PER_GEMM, 1),
                                       jnp.float32)
            fp8_metas_scale_inv = jnp.ones((FP8Helper.NUM_META_PER_GEMM, 1),
                                           jnp.float32)
            return jnp.sum(
                fp8_dot(x, y, fp8_max, fp8_metas_amax, fp8_metas_scale,
                        fp8_metas_scale_inv, 0, *_get_ctypes(None)))

        value_n_grad_func = value_and_grad(func, (0, 1))
        value_n_grad_func_compiled = jit(value_n_grad_func).lower(a,
                                                                  b).compile()
        value_n_grad_func_compiled(a, b)

    @pytest.mark.skipif(not is_fp8_supported(),
                        reason='GPU capability is not enough to run FP8')
    @pytest.mark.parametrize('compute_type', FP8_COMPUTE_TYPE)
    def test_compile_fp8(self, compute_type):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 2)
        a = jax.random.normal(subkeys[0], (256, 512), jnp.bfloat16)
        b = jax.random.normal(subkeys[1], (512, 256), jnp.bfloat16)

        def func(x, y):
            fp8_max = FP8Helper.generate_fp8_max_array(
                FP8Helper.NUM_META_PER_GEMM)
            fp8_metas_amax = jnp.zeros(
                (FP8Helper.NUM_META_PER_GEMM, FP8Helper.AMAX_HISTORY_SIZE),
                jnp.float32)
            fp8_metas_scale = jnp.ones((FP8Helper.NUM_META_PER_GEMM, 1),
                                       jnp.float32)
            fp8_metas_scale_inv = jnp.ones((FP8Helper.NUM_META_PER_GEMM, 1),
                                           jnp.float32)
            return jnp.sum(
                fp8_dot(x, y, fp8_max, fp8_metas_amax, fp8_metas_scale,
                        fp8_metas_scale_inv, 0, *compute_type))

        value_n_grad_func = value_and_grad(func, (0, 1))
        value_n_grad_func_compiled = jit(value_n_grad_func).lower(a,
                                                                  b).compile()
        value_n_grad_func_compiled(a, b)

    @pytest.mark.parametrize('m,n,k', GEMM_CASES)
    def test_forward_bf16(self, m, n, k):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 2)
        a = jax.random.normal(subkeys[0], (m, k), jnp.bfloat16)
        b = jax.random.normal(subkeys[1], (k, n), jnp.bfloat16)

        fp8_max = FP8Helper.generate_fp8_max_array(FP8Helper.NUM_META_PER_GEMM)
        fp8_metas_amax = jnp.zeros(
            (FP8Helper.NUM_META_PER_GEMM, FP8Helper.AMAX_HISTORY_SIZE),
            jnp.float32)
        fp8_metas_scale = jnp.ones((FP8Helper.NUM_META_PER_GEMM, 1),
                                   jnp.float32)
        fp8_metas_scale_inv = jnp.ones((FP8Helper.NUM_META_PER_GEMM, 1),
                                       jnp.float32)
        primitive_out = fp8_dot(a, b, fp8_max, fp8_metas_amax, fp8_metas_scale,
                                fp8_metas_scale_inv, 0, *_get_ctypes(None))
        ref_out = jnp.dot(a, b)

        assert jnp.allclose(primitive_out, ref_out)

    @pytest.mark.skipif(not is_fp8_supported(),
                        reason='GPU capability is not enough to run FP8')
    @pytest.mark.parametrize('m,n,k', GEMM_CASES)
    @pytest.mark.parametrize('compute_type', FP8_COMPUTE_TYPE)
    def test_forward_fp8_randint(self, m, n, k, compute_type):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 2)

        # TODO(rewang): add float random test
        min_val, max_val = -8, 8
        a = jax.random.randint(subkeys[0], (m, k), min_val,
                               max_val).astype(jnp.bfloat16)
        b = jax.random.randint(subkeys[1], (k, n), min_val,
                               max_val).astype(jnp.bfloat16)

        fp8_max = FP8Helper.generate_fp8_max_array(FP8Helper.NUM_META_PER_GEMM)
        fp8_metas_amax = jnp.zeros(
            (FP8Helper.NUM_META_PER_GEMM, FP8Helper.AMAX_HISTORY_SIZE),
            jnp.float32)
        fp8_metas_scale = jnp.ones((FP8Helper.NUM_META_PER_GEMM, 1),
                                   jnp.float32)
        fp8_metas_scale_inv = jnp.ones((FP8Helper.NUM_META_PER_GEMM, 1),
                                       jnp.float32)
        fp8_meta = [
            fp8_max, fp8_metas_amax, fp8_metas_scale, fp8_metas_scale_inv
        ]

        # calculate amax
        primitive_out = fp8_dot(a, b, *fp8_meta, 0, *compute_type)
        # calculate scale by amax
        fp8_meta = FP8Helper._update_fp8_metas_impl(fp8_meta)

        primitive_out = fp8_dot(a, b, *fp8_meta, 0, *compute_type)
        ref_out = jnp.dot(a, b)

        ref_out = ref_out.astype(jnp.float32)
        primitive_out = primitive_out.astype(jnp.float32)

        assert jnp.allclose(primitive_out, ref_out)

    @pytest.mark.parametrize('m,n,k', GEMM_CASES)
    def test_grad_bf16(self, m, n, k):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 2)
        a = jax.random.normal(subkeys[0], (m, k), jnp.bfloat16)
        b = jax.random.normal(subkeys[1], (k, n), jnp.bfloat16)

        def primitive_func(x, y):
            fp8_max = FP8Helper.generate_fp8_max_array(
                FP8Helper.NUM_META_PER_GEMM)
            fp8_metas_amax = jnp.zeros(
                (FP8Helper.NUM_META_PER_GEMM, FP8Helper.AMAX_HISTORY_SIZE),
                jnp.float32)
            fp8_metas_scale = jnp.ones((FP8Helper.NUM_META_PER_GEMM, 1),
                                       jnp.float32)
            fp8_metas_scale_inv = jnp.ones((FP8Helper.NUM_META_PER_GEMM, 1),
                                           jnp.float32)
            return jnp.mean(
                fp8_dot(x, y, fp8_max, fp8_metas_amax, fp8_metas_scale,
                        fp8_metas_scale_inv, 0, *_get_ctypes(None)))

        def ref_func(x, y):
            return jnp.mean(jnp.dot(x, y))

        value_n_grad_primitive_func = jit(
            value_and_grad(primitive_func, (0, 1)))
        value_n_grad_ref_func = jit(value_and_grad(ref_func, (0, 1)))
        primitive_out, (primitive_a_grad,
                        primitive_b_grad) = value_n_grad_primitive_func(a, b)
        ref_out, (ref_a_grad, ref_b_grad) = value_n_grad_ref_func(a, b)

        assert jnp.allclose(primitive_out, ref_out)
        assert jnp.allclose(primitive_a_grad, ref_a_grad)
        assert jnp.allclose(primitive_b_grad, ref_b_grad, atol=1e-5)

    @pytest.mark.skipif(not is_fp8_supported(),
                        reason='GPU capability is not enough to run FP8')
    @pytest.mark.parametrize('m,n,k', GEMM_CASES)
    @pytest.mark.parametrize('compute_type', FP8_COMPUTE_TYPE)
    def test_grad_fp8_randint(self, m, n, k, compute_type):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 2)

        # TODO(rewang): add float random test
        min_val, max_val = -8, 8
        a = jax.random.randint(subkeys[0], (m, k), min_val,
                               max_val).astype(jnp.bfloat16)
        b = jax.random.randint(subkeys[1], (k, n), min_val,
                               max_val).astype(jnp.bfloat16)

        fp8_max = FP8Helper.generate_fp8_max_array(FP8Helper.NUM_META_PER_GEMM)
        fp8_metas_amax = jnp.zeros(
            (FP8Helper.NUM_META_PER_GEMM, FP8Helper.AMAX_HISTORY_SIZE),
            jnp.float32)
        fp8_metas_scale = jnp.ones((FP8Helper.NUM_META_PER_GEMM, 1),
                                   jnp.float32)
        fp8_metas_scale_inv = jnp.ones((FP8Helper.NUM_META_PER_GEMM, 1),
                                       jnp.float32)
        fp8_meta = [
            fp8_max, fp8_metas_amax, fp8_metas_scale, fp8_metas_scale_inv
        ]

        def primitive_func(x, y, metas):
            return jnp.sum(fp8_dot(x, y, *metas, 0, *compute_type))

        def ref_func(x, y):
            return jnp.sum(jnp.dot(x, y))

        value_n_grad_primitive_func = jit(
            value_and_grad(primitive_func, (0, 1)))
        value_n_grad_ref_func = jit(value_and_grad(ref_func, (0, 1)))

        ref_out, (ref_a_grad, ref_b_grad) = value_n_grad_ref_func(a, b)

        # calculate amax
        primitive_out, (primitive_a_grad,
                        primitive_b_grad) = value_n_grad_primitive_func(
                            a, b, fp8_meta)

        # calculate scale by amax
        fp8_meta = FP8Helper._update_fp8_metas_impl(fp8_meta)
        primitive_out, (primitive_a_grad,
                        primitive_b_grad) = value_n_grad_primitive_func(
                            a, b, fp8_meta)

        assert jnp.allclose(primitive_out, ref_out)
        assert jnp.allclose(primitive_a_grad, ref_a_grad)
        assert jnp.allclose(primitive_b_grad, ref_b_grad)

    def test_contracting_dims_bf16(self):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 2)
        a = jax.random.normal(subkeys[0], (32, 8, 16, 64), jnp.bfloat16)
        b = jax.random.normal(subkeys[1], (16, 64, 128), jnp.bfloat16)

        def primitive_func(x, y):
            fp8_max = FP8Helper.generate_fp8_max_array(
                FP8Helper.NUM_META_PER_GEMM)
            fp8_metas_amax = jnp.zeros(
                (FP8Helper.NUM_META_PER_GEMM, FP8Helper.AMAX_HISTORY_SIZE),
                jnp.float32)
            fp8_metas_scale = jnp.ones((FP8Helper.NUM_META_PER_GEMM, 1),
                                       jnp.float32)
            fp8_metas_scale_inv = jnp.ones((FP8Helper.NUM_META_PER_GEMM, 1),
                                           jnp.float32)
            return jnp.sum(
                fp8_dot(x, y, fp8_max, fp8_metas_amax,
                        fp8_metas_scale, fp8_metas_scale_inv, 0,
                        *_get_ctypes(None), ((2, 3), (0, 1))))

        def ref_func(x, y):
            return jnp.sum(
                lax.dot_general(x,
                                y,
                                dimension_numbers=(((2, 3), (0, 1)), ((),
                                                                      ()))))

        value_n_grad_primitive_func = jit(
            value_and_grad(primitive_func, (0, 1)))
        value_n_grad_ref_func = jit(value_and_grad(ref_func, (0, 1)))
        primitive_out, (primitive_a_grad,
                        primitive_b_grad) = value_n_grad_primitive_func(a, b)
        ref_out, (ref_a_grad, ref_b_grad) = value_n_grad_ref_func(a, b)

        assert jnp.allclose(primitive_out, ref_out)
        assert jnp.allclose(primitive_a_grad, ref_a_grad)
        assert jnp.allclose(primitive_b_grad, ref_b_grad)

    @pytest.mark.skipif(not is_fp8_supported(),
                        reason='GPU capability is not enough to run FP8')
    @pytest.mark.parametrize('m,k', [(32, 64), (64, 256)])
    @pytest.mark.parametrize('dtype', [DType.kBFloat16, DType.kFloat8E5M2])
    def test_gated_gelu(self, m, k, dtype):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 4)

        a = jax.random.uniform(subkeys[0], (m, k * 2), jnp.bfloat16, 5, 8)

        fp8_metas_amax = jnp.zeros(
            (FP8Helper.NUM_META_PER_GEMM, FP8Helper.AMAX_HISTORY_SIZE),
            jnp.float32)
        fp8_metas_scale = jnp.ones((FP8Helper.NUM_META_PER_GEMM, 1),
                                   jnp.float32)
        fp8_metas_scale_inv = jnp.ones((FP8Helper.NUM_META_PER_GEMM, 1),
                                       jnp.float32)

        @jax.custom_vjp
        def primitive_func(x, y, z):
            r, _ = primitive_func_fwd(x, y, z)
            return r

        def primitive_func_fwd(x, y, z):  # pylint: disable=unused-argument
            r, _ = te_gated_gelu(x, fp8_metas_amax[0], fp8_metas_scale[0],
                                 fp8_metas_scale_inv[0], DType.kBFloat16)
            return r, x

        def primitive_func_bwd(x, g):
            dgelu, dgelu_trans, amax = te_cast_transpose_dgated_gelu(
                g, x, fp8_metas_amax[0], fp8_metas_scale[0],
                fp8_metas_scale_inv[0], dtype)
            return dgelu, dgelu_trans, amax

        primitive_func.defvjp(primitive_func_fwd, primitive_func_bwd)

        def ref_func(x):
            x = jnp.split(x, 2, axis=-1)
            acts = [jax.nn.gelu(x[0]), x[1]]
            x = functools.reduce(operator.mul, acts)
            x = jnp.asarray(x, jnp.bfloat16)
            return x

        value_n_grad_primitive_func = jit(
            value_and_grad(lambda x, y, z: jnp.mean(primitive_func(x, y, z)),
                           (0, 1, 2)))
        value_n_grad_ref_func = jit(
            value_and_grad(lambda x: jnp.mean(ref_func(x))))

        ref_out, ref_grad = value_n_grad_ref_func(a)

        primitive_out, (primitive_grad, primitive_grad_trans,
                        amax) = value_n_grad_primitive_func(a, a, a)

        assert jnp.allclose(primitive_out, ref_out, rtol=1e-2)
        assert jnp.allclose(amax, jnp.amax(jnp.abs(ref_grad)), rtol=1e-2)
        if dtype == DType.kBFloat16:
            np.testing.assert_allclose(jnp.asarray(ref_grad, np.float32),
                                       jnp.asarray(primitive_grad, np.float32),
                                       rtol=1e-2)
            np.testing.assert_allclose(np.transpose(
                jnp.asarray(ref_grad, np.float32)),
                                       jnp.asarray(primitive_grad_trans,
                                                   np.float32),
                                       rtol=1e-2)

    @pytest.mark.skipif(not is_fp8_supported(),
                        reason='GPU capability is not enough to run FP8')
    @pytest.mark.parametrize('m,n,k', [(256, 256, 512), (16384, 1024, 2816),
              (16384, 2816, 1024), (16384, 1024, 1024)])
    def test_grad_fp8_mlp_randint(self, m, n, k):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 4)
        activations = ('gelu', 'linear')

        a = jax.random.uniform(subkeys[0], (m, k), jnp.bfloat16, 5, 8)
        k1 = jax.random.uniform(subkeys[1], (k, n * len(activations)),
                                jnp.bfloat16, 5, 8)
        k2 = jax.random.uniform(subkeys[2], (n, k), jnp.bfloat16, 5, 8)
        s = jax.random.uniform(subkeys[3], (k, ), jnp.bfloat16, 5, 8)

        fp8_max = FP8Helper.generate_fp8_max_array(
            FP8Helper.NUM_META_PER_GEMM * 2)
        fp8_metas_amax = jnp.zeros(
            (FP8Helper.NUM_META_PER_GEMM * 2, FP8Helper.AMAX_HISTORY_SIZE),
            jnp.float32)
        fp8_metas_scale = jnp.ones((FP8Helper.NUM_META_PER_GEMM * 2, 1),
                                   jnp.float32)
        fp8_metas_scale_inv = jnp.ones((FP8Helper.NUM_META_PER_GEMM * 2, 1),
                                       jnp.float32)
        fp8_meta = [
            fp8_max, fp8_metas_amax, fp8_metas_scale, fp8_metas_scale_inv
        ]
        compute_type = _get_ctypes(Format.HYBRID)

        def primitive_func(x, ln_s, y, z, metas):
            return jnp.mean(
                fp8_ln_mlp(x,
                           ln_s,
                           y,
                           z,
                           *metas,
                           0,
                           *compute_type,
                           activations=activations))

        def _convert_to_activation_function(fn_or_string):
            """Convert a string to an activation function."""
            if fn_or_string == 'linear':
                return lambda x: x
            if isinstance(fn_or_string, str):
                return getattr(nn, fn_or_string)
            if callable(fn_or_string):
                return fn_or_string
            raise ValueError(
                f"don't know how to convert {fn_or_string} to an activation function"
            )

        def fp8_ln_mlp_py(
            inputs: jnp.ndarray,
            ln_scale: jnp.ndarray,
            kernel_1: jnp.ndarray,
            kernel_2: jnp.ndarray,
            fp8_maxs: jnp.ndarray,
            amax: jnp.ndarray,
            scale: jnp.ndarray,
            scale_inv: jnp.ndarray,
            amax_history_idx: int,
            fwd_ctype,
            bwd_ctype,
            epsilon=1e-6,
            contracting_dims=((-1, ), (0, )),
            batch_axis_resource='data',
            batch_dim_index=0,
            activations=('gelu', 'linear')) -> jnp.ndarray:
            x = jnp.asarray(inputs, jnp.float32)
            mean2 = jnp.mean(jax.lax.square(x), axis=-1, keepdims=True)
            y = jnp.asarray(x * jax.lax.rsqrt(mean2 + epsilon), jnp.bfloat16)
            ln_out = y * ln_scale
            ln_out = jnp.asarray(ln_out, jnp.bfloat16)
            linear_1_out = fp8_dot(ln_out, kernel_1,
                                   fp8_maxs[:FP8Helper.NUM_META_PER_GEMM],
                                   amax[:FP8Helper.NUM_META_PER_GEMM],
                                   scale[:FP8Helper.NUM_META_PER_GEMM],
                                   scale_inv[:FP8Helper.NUM_META_PER_GEMM],
                                   amax_history_idx, fwd_ctype, bwd_ctype,
                                   contracting_dims, batch_axis_resource,
                                   batch_dim_index)
            x = jnp.split(linear_1_out, len(activations), axis=-1)
            acts = []
            for idx, act_fn in enumerate(activations):
                x_i = _convert_to_activation_function(act_fn)(x[idx])
                acts.append(x_i)
            x = functools.reduce(operator.mul, acts)
            x = jnp.asarray(x, jnp.bfloat16)
            output = fp8_dot(x, kernel_2,
                             fp8_maxs[FP8Helper.NUM_META_PER_GEMM:],
                             amax[FP8Helper.NUM_META_PER_GEMM:],
                             scale[FP8Helper.NUM_META_PER_GEMM:],
                             scale_inv[FP8Helper.NUM_META_PER_GEMM:],
                             amax_history_idx, fwd_ctype, bwd_ctype,
                             contracting_dims, batch_axis_resource,
                             batch_dim_index)
            return output

        def ref_func(x, ln_s, y, z, metas):
            return jnp.mean(
                fp8_ln_mlp_py(x,
                              ln_s,
                              y,
                              z,
                              *metas,
                              0,
                              *compute_type,
                              activations=activations))

        value_n_grad_primitive_func = jit(
            value_and_grad(primitive_func, (0, 1, 2, 3)))
        value_n_grad_ref_func = jit(value_and_grad(ref_func, (0, 1, 2, 3)))

        ref_out, (ref_a_grad, ref_s_grad, ref_k1_grad,
                  ref_k2_grad) = value_n_grad_ref_func(a, s, k1, k2, fp8_meta)

        # calculate amax
        primitive_out, (primitive_a_grad, primitive_s_grad, primitive_k1_grad,
                        primitive_k2_grad) = value_n_grad_primitive_func(
                            a, s, k1, k2, fp8_meta)

        # calculate scale by amax
        fp8_meta = FP8Helper._update_fp8_metas_impl(fp8_meta)
        primitive_out, (primitive_a_grad, primitive_s_grad, primitive_k1_grad,
                        primitive_k2_grad) = value_n_grad_primitive_func(
                            a, s, k1, k2, fp8_meta)

        assert jnp.allclose(primitive_out, ref_out, rtol=1e-2)
        np.testing.assert_allclose(jnp.asarray(primitive_a_grad, np.float32),
                                   jnp.asarray(ref_a_grad, np.float32),
                                   rtol=1e-2)
        np.testing.assert_allclose(jnp.asarray(primitive_k1_grad, np.float32),
                                   jnp.asarray(ref_k1_grad, np.float32),
                                   rtol=1e-2)
        np.testing.assert_allclose(jnp.asarray(primitive_k2_grad, np.float32),
                                   jnp.asarray(ref_k2_grad, np.float32),
                                   rtol=1e-2)
        np.testing.assert_allclose(jnp.asarray(primitive_s_grad, np.float32),
                                   jnp.asarray(ref_s_grad, np.float32),
                                   rtol=1e-2)
