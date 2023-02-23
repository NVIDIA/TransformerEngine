# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from utils import assert_allclose, is_fp8_supported
from transformer_engine.common.recipe import Format
from transformer_engine.jax.cpp_extensions import dgated_gelu, gated_gelu
from transformer_engine.jax.cpp_extensions import dgated_gelu_cast_transpose, gated_gelu_fp8
from transformer_engine.jax.cpp_extensions import dequantize, quantize
from transformer_engine.jax.dot import fp8_dot
from transformer_engine.jax.fp8 import DType, FP8GemmPackage, FP8Helper, _format2dtypes
from transformer_engine.jax.layernorm import layernorm
from transformer_engine.jax.mlp import fp8_ln_mlp

GEMM_CASES = [(256, 256, 512), (32, 32, 32), (16384, 1024, 2816), (16384, 2816, 1024),
              (16384, 1024, 1024)]
FP8_COMPUTE_TYPE = [_format2dtypes(Format.E4M3), _format2dtypes(Format.HYBRID)]
LN_CASES = [(512, 1024)]
DTYPES = [jnp.bfloat16, jnp.float32]


class TestFP8Dot:

    @pytest.mark.skipif(not is_fp8_supported(), reason='GPU capability is not enough to run FP8')
    def test_qdq(self):
        FP8_E4M3_MAX = 448
        x = jnp.asarray([[-1, 0.1], [2, 3]], jnp.float32)
        amax = jnp.max(jnp.abs(x)).reshape(1)
        scale = jnp.asarray(FP8_E4M3_MAX / amax, jnp.float32).reshape(1)
        scale_inv = (1 / scale).reshape(1)

        y, new_amax = quantize(x, amax, scale, scale_inv, out_dtype=DType.kFloat8E4M3)
        assert_allclose(new_amax, 3.0)

        no_use = jnp.zeros(1, jnp.float32)
        z = dequantize(y,
                       no_use,
                       no_use,
                       scale_inv,
                       fp8_dtype=DType.kFloat8E4M3,
                       out_dtype=DType.kFloat32)
        assert_allclose(z, x, rtol=5e-2, atol=5e-2)

    def test_compile_bf16(self):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 2)
        a = jax.random.normal(subkeys[0], (256, 512), jnp.bfloat16)
        b = jax.random.normal(subkeys[1], (512, 256), jnp.bfloat16)

        def func(x, y):
            fp8_max = FP8Helper.generate_fp8_max_array(FP8Helper.NUM_META_PER_GEMM)
            fp8_metas_amax = jnp.zeros((FP8Helper.NUM_META_PER_GEMM, FP8Helper.AMAX_HISTORY_SIZE),
                                       jnp.float32)
            fp8_metas_scale = jnp.ones((FP8Helper.NUM_META_PER_GEMM, 1), jnp.float32)
            fp8_metas_scale_inv = jnp.ones((FP8Helper.NUM_META_PER_GEMM, 1), jnp.float32)
            # x = input, matrix 2d
            # y = input, matrix 2d (weight)
            fp8_gemm_pkg = FP8GemmPackage(1, x, [y], fp8_max, fp8_metas_amax, fp8_metas_scale,
                                          fp8_metas_scale_inv)
            return jnp.sum(fp8_dot(fp8_gemm_pkg, 0, *_format2dtypes(None)))

        value_n_grad_func = value_and_grad(func, (0, 1))
        value_n_grad_func_compiled = jit(value_n_grad_func).lower(a, b).compile()
        value_n_grad_func_compiled(a, b)

    @pytest.mark.skipif(not is_fp8_supported(), reason='GPU capability is not enough to run FP8')
    @pytest.mark.parametrize('compute_type', FP8_COMPUTE_TYPE)
    def test_compile_fp8(self, compute_type):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 2)
        a = jax.random.normal(subkeys[0], (256, 512), jnp.bfloat16)
        b = jax.random.normal(subkeys[1], (512, 256), jnp.bfloat16)

        def func(x, y):
            fp8_max = FP8Helper.generate_fp8_max_array(FP8Helper.NUM_META_PER_GEMM)
            fp8_metas_amax = jnp.zeros((FP8Helper.NUM_META_PER_GEMM, FP8Helper.AMAX_HISTORY_SIZE),
                                       jnp.float32)
            fp8_metas_scale = jnp.ones((FP8Helper.NUM_META_PER_GEMM, 1), jnp.float32)
            fp8_metas_scale_inv = jnp.ones((FP8Helper.NUM_META_PER_GEMM, 1), jnp.float32)
            fp8_gemm_pkg = FP8GemmPackage(1, x, [y], fp8_max, fp8_metas_amax, fp8_metas_scale,
                                          fp8_metas_scale_inv)
            return jnp.sum(fp8_dot(fp8_gemm_pkg, 0, *compute_type))

        value_n_grad_func = value_and_grad(func, (0, 1))
        value_n_grad_func_compiled = jit(value_n_grad_func).lower(a, b).compile()
        value_n_grad_func_compiled(a, b)

    @pytest.mark.parametrize('m,n,k', GEMM_CASES)
    def test_forward_bf16(self, m, n, k):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 2)
        a = jax.random.normal(subkeys[0], (m, k), jnp.bfloat16)
        b = jax.random.normal(subkeys[1], (k, n), jnp.bfloat16)

        fp8_max = FP8Helper.generate_fp8_max_array(FP8Helper.NUM_META_PER_GEMM)
        fp8_metas_amax = jnp.zeros((FP8Helper.NUM_META_PER_GEMM, FP8Helper.AMAX_HISTORY_SIZE),
                                   jnp.float32)
        fp8_metas_scale = jnp.ones((FP8Helper.NUM_META_PER_GEMM, 1), jnp.float32)
        fp8_metas_scale_inv = jnp.ones((FP8Helper.NUM_META_PER_GEMM, 1), jnp.float32)
        fp8_gemm_pkg = FP8GemmPackage(1, a, [b], fp8_max, fp8_metas_amax, fp8_metas_scale,
                                      fp8_metas_scale_inv)
        primitive_out = fp8_dot(fp8_gemm_pkg, 0, *_format2dtypes(None))
        ref_out = jnp.dot(a, b)

        assert_allclose(primitive_out, ref_out)

    @pytest.mark.skipif(not is_fp8_supported(), reason='GPU capability is not enough to run FP8')
    @pytest.mark.parametrize('m,n,k', GEMM_CASES)
    @pytest.mark.parametrize('compute_type', FP8_COMPUTE_TYPE)
    def test_forward_fp8_randint(self, m, n, k, compute_type):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 2)

        # TODO(rewang): add float random test
        min_val, max_val = -8, 8
        a = jax.random.randint(subkeys[0], (m, k), min_val, max_val).astype(jnp.bfloat16)
        b = jax.random.randint(subkeys[1], (k, n), min_val, max_val).astype(jnp.bfloat16)

        fp8_max = FP8Helper.generate_fp8_max_array(FP8Helper.NUM_META_PER_GEMM)
        fp8_metas_amax = jnp.zeros((FP8Helper.NUM_META_PER_GEMM, FP8Helper.AMAX_HISTORY_SIZE),
                                   jnp.float32)
        fp8_metas_scale = jnp.ones((FP8Helper.NUM_META_PER_GEMM, 1), jnp.float32)
        fp8_metas_scale_inv = jnp.ones((FP8Helper.NUM_META_PER_GEMM, 1), jnp.float32)
        fp8_meta = [fp8_max, fp8_metas_amax, fp8_metas_scale, fp8_metas_scale_inv]

        # calculate amax
        fp8_gemm_pkg = FP8GemmPackage(1, a, [b], *fp8_meta)
        primitive_out = fp8_dot(fp8_gemm_pkg, 0, *compute_type)
        # calculate scale by amax
        fp8_meta = FP8Helper._update_fp8_metas_impl(fp8_meta)

        fp8_gemm_pkg = FP8GemmPackage(1, a, [b], *fp8_meta)
        primitive_out = fp8_dot(fp8_gemm_pkg, 0, *compute_type)
        ref_out = jnp.dot(a, b)

        ref_out = ref_out.astype(jnp.float32)
        primitive_out = primitive_out.astype(jnp.float32)

        assert_allclose(primitive_out, ref_out)

    @pytest.mark.parametrize('m,n,k', GEMM_CASES)
    def test_grad_bf16(self, m, n, k):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 2)
        a = jax.random.normal(subkeys[0], (m, k), jnp.bfloat16)
        b = jax.random.normal(subkeys[1], (k, n), jnp.bfloat16)

        def primitive_func(x, y):
            fp8_max = FP8Helper.generate_fp8_max_array(FP8Helper.NUM_META_PER_GEMM)
            fp8_metas_amax = jnp.zeros((FP8Helper.NUM_META_PER_GEMM, FP8Helper.AMAX_HISTORY_SIZE),
                                       jnp.float32)
            fp8_metas_scale = jnp.ones((FP8Helper.NUM_META_PER_GEMM, 1), jnp.float32)
            fp8_metas_scale_inv = jnp.ones((FP8Helper.NUM_META_PER_GEMM, 1), jnp.float32)
            fp8_gemm_pkg = FP8GemmPackage(1, x, [y], fp8_max, fp8_metas_amax, fp8_metas_scale,
                                          fp8_metas_scale_inv)
            return jnp.mean(fp8_dot(fp8_gemm_pkg, 0, *_format2dtypes(None)))

        def ref_func(x, y):
            return jnp.mean(jnp.dot(x, y))

        value_n_grad_primitive_func = value_and_grad(primitive_func, (0, 1))

        value_n_grad_ref_func = value_and_grad(ref_func, (0, 1))

        primitive_out, (primitive_a_grad, primitive_b_grad) = value_n_grad_primitive_func(a, b)
        ref_out, (ref_a_grad, ref_b_grad) = value_n_grad_ref_func(a, b)

        assert_allclose(primitive_out, ref_out)
        assert_allclose(primitive_a_grad, ref_a_grad)
        assert_allclose(primitive_b_grad, ref_b_grad, atol=1e-5)

    @pytest.mark.skipif(not is_fp8_supported(), reason='GPU capability is not enough to run FP8')
    @pytest.mark.parametrize('m,n,k', GEMM_CASES)
    @pytest.mark.parametrize('compute_type', FP8_COMPUTE_TYPE)
    def test_grad_fp8_randint(self, m, n, k, compute_type):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 2)

        # TODO(rewang): add float random test
        min_val, max_val = -8, 8
        a = jax.random.randint(subkeys[0], (m, k), min_val, max_val).astype(jnp.bfloat16)
        b = jax.random.randint(subkeys[1], (k, n), min_val, max_val).astype(jnp.bfloat16)

        fp8_max = FP8Helper.generate_fp8_max_array(FP8Helper.NUM_META_PER_GEMM)
        fp8_metas_amax = jnp.zeros((FP8Helper.NUM_META_PER_GEMM, FP8Helper.AMAX_HISTORY_SIZE),
                                   jnp.float32)
        fp8_metas_scale = jnp.ones((FP8Helper.NUM_META_PER_GEMM, 1), jnp.float32)
        fp8_metas_scale_inv = jnp.ones((FP8Helper.NUM_META_PER_GEMM, 1), jnp.float32)
        fp8_meta = [fp8_max, fp8_metas_amax, fp8_metas_scale, fp8_metas_scale_inv]

        def primitive_func(x, y, metas):
            fp8_gemm_pkg = FP8GemmPackage(1, x, [y], *metas)
            return jnp.sum(fp8_dot(fp8_gemm_pkg, 0, *compute_type))

        def ref_func(x, y):
            return jnp.sum(jnp.dot(x, y))

        value_n_grad_primitive_func = value_and_grad(primitive_func, (0, 1))
        value_n_grad_ref_func = value_and_grad(ref_func, (0, 1))

        ref_out, (ref_a_grad, ref_b_grad) = value_n_grad_ref_func(a, b)

        # calculate amax
        primitive_out, (primitive_a_grad,
                        primitive_b_grad) = value_n_grad_primitive_func(a, b, fp8_meta)

        # calculate scale by amax
        fp8_meta = FP8Helper._update_fp8_metas_impl(fp8_meta)
        primitive_out, (primitive_a_grad,
                        primitive_b_grad) = value_n_grad_primitive_func(a, b, fp8_meta)

        assert_allclose(primitive_out, ref_out)
        assert_allclose(primitive_a_grad, ref_a_grad)
        assert_allclose(primitive_b_grad, ref_b_grad)

    def test_contracting_dims_bf16(self):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 2)
        a = jax.random.normal(subkeys[0], (32, 8, 16, 64), jnp.bfloat16)
        b = jax.random.normal(subkeys[1], (16, 64, 128), jnp.bfloat16)

        def primitive_func(x, y):
            fp8_max = FP8Helper.generate_fp8_max_array(FP8Helper.NUM_META_PER_GEMM)
            fp8_metas_amax = jnp.zeros((FP8Helper.NUM_META_PER_GEMM, FP8Helper.AMAX_HISTORY_SIZE),
                                       jnp.float32)
            fp8_metas_scale = jnp.ones((FP8Helper.NUM_META_PER_GEMM, 1), jnp.float32)
            fp8_metas_scale_inv = jnp.ones((FP8Helper.NUM_META_PER_GEMM, 1), jnp.float32)
            fp8_gemm_pkg = FP8GemmPackage(1, x, [y], fp8_max, fp8_metas_amax, fp8_metas_scale,
                                          fp8_metas_scale_inv)
            return jnp.sum(fp8_dot(fp8_gemm_pkg, 0, *_format2dtypes(None), ((2, 3), (0, 1))))

        def ref_func(x, y):
            return jnp.sum(lax.dot_general(x, y, dimension_numbers=(((2, 3), (0, 1)), ((), ()))))

        value_n_grad_primitive_func = value_and_grad(primitive_func, (0, 1))
        value_n_grad_ref_func = value_and_grad(ref_func, (0, 1))
        primitive_out, (primitive_a_grad, primitive_b_grad) = value_n_grad_primitive_func(a, b)
        ref_out, (ref_a_grad, ref_b_grad) = value_n_grad_ref_func(a, b)

        assert_allclose(primitive_out, ref_out)
        assert_allclose(primitive_a_grad, ref_a_grad)
        assert_allclose(primitive_b_grad, ref_b_grad)

    @pytest.mark.skipif(not is_fp8_supported(), reason='GPU capability is not enough to run FP8')
    @pytest.mark.parametrize('m,n,k', [(256, 256, 512), (16384, 1024, 2816), (16384, 2816, 1024),
                                       (16384, 1024, 1024)])
    def test_grad_fp8_mlp_randint(self, m, n, k):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 4)
        activations = ('gelu', 'linear')

        a = jax.random.uniform(subkeys[0], (m, k), jnp.bfloat16, 5, 8)
        k1 = jax.random.uniform(subkeys[1], (k, n * len(activations)), jnp.bfloat16, 5, 8)
        k2 = jax.random.uniform(subkeys[2], (n, k), jnp.bfloat16, 5, 8)
        s = jax.random.uniform(subkeys[3], (k,), jnp.bfloat16, 5, 8)

        fp8_max = FP8Helper.generate_fp8_max_array(FP8Helper.NUM_META_PER_GEMM * 2)
        fp8_metas_amax = jnp.zeros((FP8Helper.NUM_META_PER_GEMM * 2, FP8Helper.AMAX_HISTORY_SIZE),
                                   jnp.float32)
        fp8_metas_scale = jnp.ones((FP8Helper.NUM_META_PER_GEMM * 2, 1), jnp.float32)
        fp8_metas_scale_inv = jnp.ones((FP8Helper.NUM_META_PER_GEMM * 2, 1), jnp.float32)
        fp8_meta = [fp8_max, fp8_metas_amax, fp8_metas_scale, fp8_metas_scale_inv]
        compute_type = _format2dtypes(Format.HYBRID)

        def primitive_func(x, ln_s, y, z, metas):
            # x is input tensor, matrix 2d
            # y, z are weights, matrix 2d
            # out = (x * y) * z
            fp8_gemm_pkg = FP8GemmPackage(2, x, [y, z], *metas)
            return jnp.mean(
                fp8_ln_mlp(fp8_gemm_pkg,
                           ln_s,
                           None,
                           "rmsnorm",
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
            raise ValueError(f"don't know how to convert {fn_or_string} to an activation function")

        def fp8_ln_mlp_py(inputs: jnp.ndarray,
                          ln_scale: jnp.ndarray,
                          kernel_1: jnp.ndarray,
                          kernel_2: jnp.ndarray,
                          fp8_maxs: jnp.ndarray,
                          amax: jnp.ndarray,
                          scale: jnp.ndarray,
                          scale_inv: jnp.ndarray,
                          amax_history_idx: int,
                          fwd_dtype,
                          bwd_dtype,
                          epsilon=1e-6,
                          contracting_dims=((-1,), (0,)),
                          dp_dim_index=0,
                          activations=('gelu', 'linear')) -> jnp.ndarray:
            x = jnp.asarray(inputs, jnp.float32)
            mean2 = jnp.mean(jax.lax.square(x), axis=-1, keepdims=True)
            y = jnp.asarray(x * jax.lax.rsqrt(mean2 + epsilon), jnp.bfloat16)
            ln_out = y * ln_scale
            ln_out = jnp.asarray(ln_out, jnp.bfloat16)
            fp8_gemm_1_pkg = FP8GemmPackage(1, ln_out, [kernel_1],
                                            fp8_maxs[:FP8Helper.NUM_META_PER_GEMM],
                                            amax[:FP8Helper.NUM_META_PER_GEMM],
                                            scale[:FP8Helper.NUM_META_PER_GEMM],
                                            scale_inv[:FP8Helper.NUM_META_PER_GEMM])
            linear_1_out = fp8_dot(fp8_gemm_1_pkg,
                                   amax_history_idx,
                                   fwd_dtype,
                                   bwd_dtype,
                                   contracting_dims,
                                   dp_dim_index=dp_dim_index)
            x = jnp.split(linear_1_out, len(activations), axis=-1)
            acts = []
            for idx, act_fn in enumerate(activations):
                x_i = _convert_to_activation_function(act_fn)(x[idx])
                acts.append(x_i)
            x = functools.reduce(operator.mul, acts)
            x = jnp.asarray(x, jnp.bfloat16)
            fp8_gemm_2_pkg = FP8GemmPackage(1, x, [kernel_2],
                                            fp8_maxs[FP8Helper.NUM_META_PER_GEMM:],
                                            amax[FP8Helper.NUM_META_PER_GEMM:],
                                            scale[FP8Helper.NUM_META_PER_GEMM:],
                                            scale_inv[FP8Helper.NUM_META_PER_GEMM:])
            output = fp8_dot(fp8_gemm_2_pkg,
                             amax_history_idx,
                             fwd_dtype,
                             bwd_dtype,
                             contracting_dims,
                             dp_dim_index=dp_dim_index)
            return output

        def ref_func(x, ln_s, y, z, metas):
            return jnp.mean(
                fp8_ln_mlp_py(x, ln_s, y, z, *metas, 0, *compute_type, activations=activations))

        value_n_grad_primitive_func = jit(value_and_grad(primitive_func, (0, 1, 2, 3)))
        value_n_grad_ref_func = jit(value_and_grad(ref_func, (0, 1, 2, 3)))

        ref_out, (ref_a_grad, ref_s_grad, ref_k1_grad,
                  ref_k2_grad) = value_n_grad_ref_func(a, s, k1, k2, fp8_meta)

        # calculate amax
        primitive_out, (primitive_a_grad, primitive_s_grad, primitive_k1_grad,
                        primitive_k2_grad) = value_n_grad_primitive_func(a, s, k1, k2, fp8_meta)

        # calculate scale by amax
        fp8_meta = FP8Helper._update_fp8_metas_impl(fp8_meta)
        primitive_out, (primitive_a_grad, primitive_s_grad, primitive_k1_grad,
                        primitive_k2_grad) = value_n_grad_primitive_func(a, s, k1, k2, fp8_meta)

        assert_allclose(primitive_out, ref_out, rtol=1e-2)
        assert_allclose(jnp.asarray(primitive_a_grad, np.float32),
                        jnp.asarray(ref_a_grad, np.float32),
                        rtol=1e-2)
        assert_allclose(jnp.asarray(primitive_k1_grad, np.float32),
                        jnp.asarray(ref_k1_grad, np.float32),
                        rtol=1e-2)
        assert_allclose(jnp.asarray(primitive_k2_grad, np.float32),
                        jnp.asarray(ref_k2_grad, np.float32),
                        rtol=1e-2)
        assert_allclose(jnp.asarray(primitive_s_grad, np.float32),
                        jnp.asarray(ref_s_grad, np.float32),
                        rtol=1e-2)


@pytest.fixture(name="random_inputs")
def random_inputs_fixture(shape):
    key = jax.random.PRNGKey(0)
    subkeys = jax.random.split(key, 4)
    out = jax.random.uniform(subkeys[0], shape, jnp.bfloat16, 5, 8)
    return out


class TestGatedGeLu:

    def ref_func(self, inputs):

        def jax_gated_gelu(x):
            x = jnp.split(x, 2, axis=-1)
            acts = [jax.nn.gelu(x[0]), x[1]]
            x = functools.reduce(operator.mul, acts)
            x = jnp.asarray(x, jnp.bfloat16)
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
        func = jit(value_and_grad(lambda x: jnp.mean(primitive(x))))
        return func(inputs)

    @pytest.mark.parametrize('shape', [(32, 64), (64, 256)])
    def test_gated_gelu(self, random_inputs):
        x = random_inputs
        prim_out, prim_grad = self.prim_func(x)
        ref_out, ref_grad = self.ref_func(x)

        assert_allclose(prim_out, ref_out, rtol=1e-2)
        assert_allclose(prim_grad, ref_grad, rtol=1e-1, atol=1e-3)


class TestGatedGeLuFP8(TestGatedGeLu):

    def prim_func(self, inputs):
        amax = self.amax
        scale = self.scale
        scale_inv = self.scale_inv
        no_use = jnp.zeros(1, jnp.float32)

        @jax.custom_vjp
        def primitive(x, y, z):
            out = primitive_fwd(x, y, z)
            return out

        def primitive_fwd(x, y, z):    # pylint: disable=unused-argument
            out, _ = gated_gelu_fp8(x, amax, scale, scale_inv, DType.kFloat8E5M2)
            out = dequantize(out, no_use, no_use, scale_inv, DType.kFloat8E5M2, DType.kBFloat16)
            ctx = x
            return out, ctx

        def primitive_bwd(ctx, g):
            x = ctx
            dgelu, dgelu_trans, amax_out = dgated_gelu_cast_transpose(g, x, amax, scale, scale_inv,
                                                                      DType.kFloat8E5M2)
            dgelu = dequantize(dgelu, no_use, no_use, scale_inv, DType.kFloat8E5M2, DType.kFloat32)
            dgelu_trans = dequantize(dgelu_trans, no_use, no_use, scale_inv, DType.kFloat8E5M2,
                                     DType.kFloat32)
            return dgelu, dgelu_trans, amax_out

        primitive.defvjp(primitive_fwd, primitive_bwd)
        func = jit(value_and_grad(lambda x, y, z: jnp.mean(primitive(x, y, z)), (0, 1, 2)))

        return func(inputs, no_use, no_use)

    @pytest.mark.skipif(not is_fp8_supported(), reason='GPU capability is not enough to run FP8')
    @pytest.mark.parametrize('shape', [(32, 64), (64, 256)])
    def test_gated_gelu(self, random_inputs):
        self.amax = jnp.zeros(1, jnp.float32)
        self.scale = jnp.ones(1, jnp.float32)
        self.scale_inv = jnp.ones(1, jnp.float32)

        x = random_inputs
        prim_out, (prim_grad, prim_grad_trans, amax) = self.prim_func(x)
        ref_out, ref_grad = self.ref_func(x)

        assert_allclose(prim_out, ref_out, rtol=1e-2)
        assert_allclose(amax, jnp.amax(jnp.abs(ref_grad)), rtol=1e-2)
        assert_allclose(prim_grad, ref_grad, rtol=1e-1, atol=1e-3)
        assert_allclose(prim_grad_trans, jnp.transpose(ref_grad), rtol=1e-1, atol=1e-3)


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

        if dtype == jnp.float32:
            assert_allclose(primitive_out, reference_out, rtol=1e-7)
            assert_allclose(primitive_dx, reference_dx, rtol=1e-7)
            assert_allclose(primitive_dgamma, reference_dgamma, rtol=1e-7)
        else:
            assert_allclose(primitive_out, reference_out, rtol=1e-3)
            assert_allclose(primitive_dx, reference_dx, rtol=1e-4, atol=5e-8)
            assert_allclose(primitive_dgamma, reference_dgamma, rtol=1e-4, atol=5e-8)


class TestLayerNorm:

    @pytest.mark.parametrize('n, hidden', LN_CASES)
    @pytest.mark.parametrize('dtype', DTYPES)
    def test_forward_backward(self, n, hidden, dtype):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 3)

        x = jax.random.uniform(subkeys[0], (n, hidden), dtype, -2, 1)
        scale = jax.random.uniform(subkeys[1], (hidden,), jnp.float32, -2, 1)
        scale = jnp.asarray(scale, dtype)
        bias = jax.random.uniform(subkeys[2], (hidden,), jnp.float32, -2, 1)
        bias = jnp.asarray(bias, dtype)
        epsilon = 1e-6

        def reference_layernorm(x, scale, bias):
            x = jnp.asarray(x, jnp.float32)
            mean = jnp.mean(x, axis=-1, keepdims=True)
            var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
            normed_input = (x - mean) * jax.lax.rsqrt(var + epsilon)
            # Align TE implementation
            return jnp.asarray(normed_input * scale + bias)

        jitted_primitive = jit(
            value_and_grad(lambda x, scale, bias: jnp.mean(layernorm(x, scale, bias, "layernorm")),
                           (0, 1, 2)))

        jitted_reference = jit(
            value_and_grad(lambda x, scale, bias: jnp.mean(reference_layernorm(x, scale, bias)),
                           (0, 1, 2)))

        primitive_out, (primitive_dx, primitive_dgamma,
                        primitive_dbeta) = jitted_primitive(x, scale, bias)
        reference_out, (reference_dx, reference_dgamma,
                        reference_dbeta) = jitted_reference(x, scale, bias)

        if dtype == jnp.float32:
            assert_allclose(primitive_out, reference_out, rtol=1e-7)
            assert_allclose(primitive_dx, reference_dx, rtol=1e-7)
            assert_allclose(primitive_dgamma, reference_dgamma, rtol=1e-7)
            assert_allclose(primitive_dbeta, reference_dbeta, rtol=1e-7)
        else:
            assert_allclose(primitive_out, reference_out, rtol=1e-3)
            assert_allclose(primitive_dx, reference_dx, rtol=1e-4, atol=5e-8)
            assert_allclose(primitive_dgamma, reference_dgamma, rtol=1e-4, atol=5e-8)
            assert_allclose(primitive_dbeta, reference_dbeta, rtol=1e-4, atol=5e-8)
