# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import jax.numpy as jnp
from jax.core import ShapedArray

from transformer_engine_jax import DType
from transformer_engine.jax.cpp_extensions import te_dtype_to_jax_dtype
from transformer_engine.jax.cpp_extensions import GemmPrimitive

SHAPES = [(256, 256, 512), (32, 32, 32), (16384, 1024, 2816), (16384, 2816, 1024),
          (16384, 1024, 1024)]
NAMED_SHAPES = [{}, {
    "data": 4
}, {
    "data": 2
}, {
    "model": 4
}, {
    "model": 2
}, {
    "data": 4,
    "model": 2
}, {
    "model": 4,
    "data": 2
}]
DTYPE = [DType.kFloat32, DType.kFloat16, DType.kBFloat16]
TRANSPOSE = [True, False]


class TestGEMMShapeInfer:

    @staticmethod
    def _joint_named_shape(ns1, ns2):
        output_named_shape = {**ns1}
        need_assert = False
        for key in ns2:
            if key in output_named_shape and output_named_shape[key] != ns2[key]:
                need_assert = True
            else:
                output_named_shape[key] = ns2[key]
        return output_named_shape, need_assert

    @staticmethod
    def _get_shapes(m, n, k, transa, transb):
        # te_gemm only support TN and col-major, then we have to reorder a, b shape
        # to compute row-major matrices calculate in col-major algos.
        a = (m, k) if transa else (k, m)
        b = (k, n) if transb else (n, k)
        out = (n, m)
        return a, b, out

    @pytest.mark.parametrize('shapes', SHAPES)
    @pytest.mark.parametrize('named_shape1', NAMED_SHAPES)
    @pytest.mark.parametrize('named_shape2', NAMED_SHAPES)
    @pytest.mark.parametrize('te_dtype', DTYPE)
    @pytest.mark.parametrize('transa', TRANSPOSE)
    @pytest.mark.parametrize('transb', TRANSPOSE)
    def test_shape_infer(self, shapes, named_shape1, named_shape2, te_dtype, transa, transb):
        a_shape, b_shape, out_shape = TestGEMMShapeInfer._get_shapes(*shapes, transa, transb)
        dtype = te_dtype_to_jax_dtype(te_dtype)
        mat_a = ShapedArray(a_shape, dtype, named_shape=named_shape1)
        mat_b = ShapedArray(b_shape, dtype, named_shape=named_shape2)

        scale_inv_a = ShapedArray((3, 1), jnp.float32)
        scale_inv_b = ShapedArray((3, 1), jnp.float32)

        ref_out_named_shape, need_assert = TestGEMMShapeInfer._joint_named_shape(
            named_shape1, named_shape2)
        ref_out = ShapedArray(out_shape, dtype, named_shape=ref_out_named_shape)

        try:
            test_out = GemmPrimitive.abstract(mat_a,
                                              mat_b,
                                              scale_inv_a,
                                              scale_inv_b,
                                              A_dtype=te_dtype,
                                              B_dtype=te_dtype,
                                              D_dtype=te_dtype,
                                              transa=transa,
                                              transb=transb,
                                              use_split_accumulator=False)
            assert not need_assert
            assert ref_out == test_out
        except AssertionError as ae:
            assert need_assert, f"{ae.args}"
