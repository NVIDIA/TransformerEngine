# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX/TE miscellaneous for custom ops"""

import numpy as np
import jax.numpy as jnp
from jax import dtypes
from jax.interpreters.mlir import dtype_to_ir_type

from transformer_engine.transformer_engine_jax import DType as TEDType

from ..sharding import get_padded_spec as te_get_padded_spec


def te_dtype_to_jax_dtype(te_dtype):
    """
    convert TE dtype to jax dtype
    """
    assert isinstance(te_dtype, TEDType)

    converter = {
        TEDType.kFloat32: jnp.float32,
        TEDType.kFloat16: jnp.float16,
        TEDType.kBFloat16: jnp.bfloat16,
        TEDType.kInt32: jnp.int32,
        TEDType.kInt64: jnp.int64,
        TEDType.kFloat8E4M3: jnp.float8_e4m3fn,
        TEDType.kFloat8E5M2: jnp.float8_e5m2,
        TEDType.kByte: jnp.uint8,
    }

    if te_dtype not in converter:
        raise ValueError(f"Unsupported {te_dtype=}")

    return converter.get(te_dtype)


def te_dtype_to_ir_dtype(te_dtype):
    """
    convert TE dtype to MLIR dtype
    """
    return dtype_to_ir_type(np.dtype(te_dtype_to_jax_dtype(te_dtype)))


def jax_dtype_to_ir_dtype(jax_dtype):
    """
    convert Jax dtype to MLIR dtype
    """
    return dtype_to_ir_type(np.dtype(jax_dtype))


def jax_dtype_to_te_dtype(jax_dtype):
    """
    convert jax dtype to TE dtype
    """
    jax_dtype = dtypes.canonicalize_dtype(jax_dtype)

    converter = {
        jnp.float32.dtype: TEDType.kFloat32,
        jnp.float16.dtype: TEDType.kFloat16,
        jnp.bfloat16.dtype: TEDType.kBFloat16,
        jnp.int32.dtype: TEDType.kInt32,
        jnp.int64.dtype: TEDType.kInt64,
        jnp.float8_e4m3fn.dtype: TEDType.kFloat8E4M3,
        jnp.float8_e5m2.dtype: TEDType.kFloat8E5M2,
        jnp.uint8.dtype: TEDType.kByte,
    }

    if jax_dtype not in converter:
        raise ValueError(f"Unsupported {jax_dtype=}")

    return converter.get(jax_dtype)


def get_padded_spec(arg_info):
    """
    Get padded spec for partitioning from arguments' information
    """
    if arg_info.sharding is None:
        return te_get_padded_spec(None, arg_info.ndim)
    ndim, spec = arg_info.ndim, arg_info.sharding.spec
    return te_get_padded_spec(spec, ndim)


def check_valid_batch_dims(bdims):
    """
    Assert out non-supported bath dims
    """
    for dim in bdims:
        assert dim in [0, None], f"Currently only support batch_dim in [0, None], but got {dim=}"


def normalize_axis_boundary(axis, ndim):
    """NA"""
    return axis if axis >= 0 else ndim + axis


def multidim_transpose(shape, static_axis_boundary, transpose_axis_boundary):
    """
    te_cast_transpose_p multi-dims transpose

    static_axis_boundary: int, Indicate those axes <= static_axis_boundary would not be
        involved into transpose, -1 means all axes involve into transpose.
    transpose_axis_boundary: int, Indicate how to split multi-dimensions tensors to 2D matrix for
        transpose. Note, transpose_axis_boundary should be greater than static_axis_boundary

    examples:
        X in shape (dim0, dim1, dim2, dim3, dim4)

        static_axis_boundary == -1, transpose_axis_boundary == 2
            Xt = (dim2, dim3, dim4, dim0, dim1)

        static_axis_boundary == 0, transpose_axis_boundary == 2
            Xt = (dim0, dim2, dim3, dim4, dim1)

        static_axis_boundary == 0, transpose_axis_boundary == 3
            Xt = (dim0, dim3, dim4, dim1. dim2)
    """
    if static_axis_boundary < 0:
        static_axis_boundary = -1  # means no static axes
    assert static_axis_boundary < len(shape) - 2  # at least 2 remaining for transpose.
    transpose_start_idx = static_axis_boundary + 1
    transpose_axis_boundary = normalize_axis_boundary(transpose_axis_boundary, len(shape))
    assert transpose_start_idx < transpose_axis_boundary
    return (
        *shape[:transpose_start_idx],
        *shape[transpose_axis_boundary:],
        *shape[transpose_start_idx:transpose_axis_boundary],
    )
