# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX/TE miscellaneous for custom ops"""

import os
import functools
from typing import Tuple
from importlib.metadata import version as get_pkg_version
from packaging.version import Version as PkgVersion

import numpy as np

import jax
from jax import dtypes
import jax.numpy as jnp
from jax.interpreters.mlir import dtype_to_ir_type

import transformer_engine_jax

from ..sharding import get_padded_spec as te_get_padded_spec
from ..quantize import ScaledTensorFactory, QuantizeLayout

TEDType = transformer_engine_jax.DType


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


def multidim_transpose(shape, static_axis_boundary=-1, transpose_axis=-1):
    """
    te_cast_transpose_p multi-dims transpose

    static_axis_boundary: int, Indicate those axes <= static_axis_boundary would not be
        involved into transpose, -1 means all axes involve into transpose.
    transpose_axis: int, Indicate how to split multi-dimensions tensors to 2D matrix for
        transpose. Note, transpose_axis should be greater than static_axis_boundary

    examples:
        X in shape (dim0, dim1, dim2, dim3, dim4)

        static_axis_boundary == -1, transpose_axis == 2
            Xt = (dim2, dim3, dim4, dim0, dim1)

        static_axis_boundary == 0, transpose_axis == 2
            Xt = (dim0, dim2, dim3, dim4, dim1)

        static_axis_boundary == 0, transpose_axis == 3
            Xt = (dim0, dim3, dim4, dim1. dim2)
    """
    if static_axis_boundary < 0:
        static_axis_boundary = -1  # means no static axes
    assert static_axis_boundary < len(shape) - 2  # at least 2 remaining for transpose.
    transpose_start_idx = static_axis_boundary + 1
    transpose_axis = normalize_axis_boundary(transpose_axis, len(shape))
    assert transpose_start_idx < transpose_axis
    return (
        *shape[:transpose_start_idx],
        *shape[transpose_axis:],
        *shape[transpose_start_idx:transpose_axis],
    )


@functools.lru_cache(maxsize=None)
def get_cudnn_version() -> Tuple[int, int, int]:
    """Runtime cuDNN version (major, minor, patch)"""
    encoded_version = transformer_engine_jax.get_cudnn_version()
    major_version_magnitude = 1000 if encoded_version < 90000 else 10000
    major, encoded_version = divmod(encoded_version, major_version_magnitude)
    minor, patch = divmod(encoded_version, 100)
    return (major, minor, patch)


@functools.lru_cache(maxsize=None)
def jax_version_meet_requirement(version: str):
    """
    Helper function checking if required JAX version is available
    """
    jax_version = PkgVersion(get_pkg_version("jax"))
    jax_version_required = PkgVersion(version)
    return jax_version >= jax_version_required


def get_xla_flag(flag: str, default=None, cast=str):
    """
    Returns the value of a flag/option in XLA_FLAGS environment variable if present or returns the default value.
    """
    xla_flags = []
    if xla_flags_env := os.getenv("XLA_FLAGS"):
        xla_flags.extend(xla_flags_env.split())

    for flag_i in sorted(xla_flags):
        if "=" in flag_i:
            # option like --xla_abc=foo
            name, val = flag_i.split("=", 2)
            if name == flag:
                return val if cast is None else cast(val)
        else:
            # flag like --xla_enable_foo
            name, val = flag_i, None
            if name == flag:
                return True
    return default


def get_min_device_compute_capability():
    """
    Returns the minimum compute capability of all local devices.
    """
    return min(
        transformer_engine_jax.get_device_compute_capability(local_gpu_id)
        for local_gpu_id in range(len(jax.local_devices()))
    )


def get_all_device_compute_capability():
    """
    Returns a list of compute capability of all local devices.
    """
    return tuple(
        transformer_engine_jax.get_device_compute_capability(local_gpu_id)
        for local_gpu_id in range(len(jax.local_devices()))
    )


def should_apply_1x_fused_dbias_war_for_arch_l_100(is_dbias: bool = False, quantizer=None):
    """
    Fused dbias is not supported for arch < 100 for 1x quantization, so we need to apply a workaround to
    calculate dbias separately. This function checks if the workaround should be applied.
    """
    if quantizer is None:
        return False

    arch_l_100 = False
    for local_gpu_id in range(len(jax.local_devices())):
        if transformer_engine_jax.get_device_compute_capability(local_gpu_id) < 100:
            arch_l_100 = True
            break
    # _quantize_dbias_impl forcing 1x quantization for tensor scaling switches q_layout to ROWWISE,
    # but this fails when bias fusion is turned on with arch < 100.
    force_1x_quantization = quantizer.scaling_mode.is_tensor_scaling() and quantizer.is_2x2x()
    return (
        (force_1x_quantization or quantizer.q_layout == QuantizeLayout.ROWWISE)
        and arch_l_100
        and is_dbias
    )


def try_apply_delayed_scaling_2x_war(f, *args, quantizer=None, flatten_axis=-1, **kwargs):
    """
    Applies a workaround for delayed scaling 2x and can be used when the TE common kernels do not yet support 2x delayed scaling.
    It will call the given function 'f' with the given arguments and quantizer as 1x and calculate the colwise output by transposing result.

    If 'f' returns a tuple, the first output must be the only ScaledTensor output.

    @param f: function to call
    @param args: positional arguments to pass to 'f'
    @param quantizer: quantizer to use
    @param kwargs: keyword arguments to pass to 'f'
    @return: the output of 'f' with the colwise output calculated
    """
    should_apply_war = (
        quantizer is not None and quantizer.scaling_mode.is_tensor_scaling() and quantizer.is_2x2x()
    )
    if not should_apply_war:
        return None

    # 2x is not supported by TE kernels for delayed scaling
    # so revert to 1x and transpose in JAX
    quantizer.q_layout = QuantizeLayout.ROWWISE
    rowwise = f(*args, **kwargs, quantizer=quantizer)
    other_outputs = None
    if isinstance(rowwise, tuple):
        other_outputs = rowwise[1:]
        rowwise = rowwise[0]
    quantizer.q_layout = QuantizeLayout.ROWWISE_COLWISE
    if flatten_axis < 0:
        flatten_axis += rowwise.data.ndim
    assert 0 < flatten_axis < rowwise.data.ndim, "flatten_axis is out of bounds"
    colwise_data = jnp.transpose(
        rowwise.data, (*range(flatten_axis, rowwise.data.ndim), *range(flatten_axis))
    )
    output_2x = ScaledTensorFactory.create(
        data=rowwise.data,
        scale_inv=rowwise.scale_inv,
        colwise_data=colwise_data,
        colwise_scale_inv=rowwise.scale_inv,
        scaling_mode=quantizer.scaling_mode,
        dq_dtype=rowwise.dq_dtype,
        q_layout=QuantizeLayout.ROWWISE_COLWISE,
        data_layout=quantizer.get_data_layout(),
        flatten_axis=flatten_axis,
    )
    if other_outputs is not None:
        return (output_2x,) + other_outputs
    return output_2x


class NamedSharding(jax.sharding.NamedSharding):
    """
    Wrapper around jax.sharding.NamedSharding that adds a string description field as metadata for easier debugging.
    """

    def __init__(self, *args, desc: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.desc = desc

    def __repr__(self):
        return f"NamedSharding({self.mesh}, {self.spec}, desc={self.desc})"

    def duplicate_with_new_description(self, desc: str):
        """
        Create a new NamedSharding with the same mesh and spec but with a new description.
        """
        return NamedSharding(self.mesh, self.spec, desc=desc)


@functools.lru_cache(maxsize=1)
def is_all_reduce_in_float32():
    """
    Check if all-reduce is in float32
    """
    return os.getenv("NVTE_JAX_ALL_REDUCE_IN_FP32", "0") == "1"
