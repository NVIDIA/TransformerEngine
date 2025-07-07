# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX te modules"""

from typing import Tuple, Sequence, Union, Dict
from functools import partial, reduce
import operator
import math
import jax
import jax.numpy as jnp
from transformer_engine_jax import get_device_compute_capability, get_num_compute_streams

from .base import BasePrimitive, register_primitive
from .quantization import grouped_quantize

from ..quantize import (
    ScaledTensor,
    GroupedScaledTensor1x,
    ScalingMode,
    Quantizer,
    GroupedQuantizer,
    QuantizeConfig,
    QuantizerSet,
    QuantizeLayout,
    noop_quantizer_set,
    is_fp8_gemm_with_all_layouts_supported,
)

__all__ = ["gemm", "grouped_gemm"]


num_cublas_streams = get_num_compute_streams()


def get_cublas_workspace_size_bytes() -> None:
    """Return 32 MiB if using hopper, 4 MiB for all other architectures."""
    if get_device_compute_capability(0) >= 90:
        return 33_554_432
    return 4_194_304


class GroupedGemmPrimitive(BasePrimitive):
    """
    Primitive for grouped GEMM
    """

    name = "te_grouped_gemm_ffi"
    multiple_results = True
    impl_static_args = (7, 8, 9, 10, 11, 12, 13, 14, 15)
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        lhs_data_aval,
        lhs_scale_inv_aval,
        rhs_data_aval,
        rhs_scale_inv_aval,
        bias_aval,
        group_sizes_aval,
        group_offset_aval,
        *,
        M,
        N,
        K,
        lhs_is_trans,
        rhs_is_trans,
        scaling_mode,
        out_dtype,
        has_bias,
        is_grouped_dense_wgrad,
    ):
        """
        Grouped GEMM operation.

        Args:
            lhs_data: Left-hand side input matrix data, 1D flattened array
            lhs_scale_inv: Left-hand side input scale_inv matrix, 1D flattened array
            rhs_data: Right-hand side input matrix data, 1D flattened array
            rhs_scale_inv: Right-hand side input scale_inv matrix, 1D flattened array
            bias: Bias matrix of shape (G, N)
            group_sizes: 1D array containing the sizes of each group
            group_offset: 1D array containing offsets for each group (not yet implemented)
            M: Number of rows in the output matrix
            N: Number of columns in the output matrix
            K: Number of columns in the left-hand side matrix
            lhs_is_trans: Boolean indicating if the left-hand side matrix is transposed
            rhs_is_trans: Boolean indicating if the right-hand side matrix is transposed
            scaling_mode: Scaling mode for the GEMM operations
            out_dtype: Data type of the output tensors
            has_bias: Boolean indicating if bias tensors are provided
            is_grouped_dense_wgrad: Boolean indicating if this is a grouped dense wgrad operation
                                    where both lhs and rhs are 2D matrices and output is (G, M, N)

        Returns:
            A jnp.ndarray containing the result of the grouped GEMM operation
        """
        del lhs_data_aval, rhs_data_aval, bias_aval, group_offset_aval
        del K, lhs_is_trans, rhs_is_trans, scaling_mode, has_bias
        # TODO(Phuong): move some shape checks from Cpp to here
        workspace_size = get_cublas_workspace_size_bytes() * num_cublas_streams
        # cuBLAS workspace ptr must be 256 bytes aligned but JAX buffers are not
        # necessarily 256 bytes aligned, we add some padding to ensure alignment.
        # We also pad scale_inv swizzle buffers size for 256 bytes alignment.
        workspace_size += 256
        workspace_size += lhs_scale_inv_aval.size + 256
        workspace_size += rhs_scale_inv_aval.size + 256
        workspace_aval = jax.core.ShapedArray(shape=(workspace_size,), dtype=jnp.uint8)

        out_shape = (M, N)
        if is_grouped_dense_wgrad:
            out_shape = (group_sizes_aval.size, M, N)
        out_aval = jax.core.ShapedArray(shape=out_shape, dtype=out_dtype)
        return (out_aval, workspace_aval)

    @staticmethod
    def outer_abstract(*args, **kwargs):
        (out_aval, _) = GroupedGemmPrimitive.abstract(*args, **kwargs)
        return (out_aval,)

    @staticmethod
    def lowering(
        ctx,
        *args,
        M,
        N,
        K,
        lhs_is_trans,
        rhs_is_trans,
        scaling_mode,
        out_dtype,
        has_bias,
        is_grouped_dense_wgrad,
    ):
        del out_dtype
        return jax.ffi.ffi_lowering(GroupedGemmPrimitive.name)(
            ctx,
            *args,
            M=M,
            N=N,
            K=K,
            lhs_is_trans=lhs_is_trans,
            rhs_is_trans=rhs_is_trans,
            scaling_mode=scaling_mode.value,
            has_bias=has_bias,
            is_grouped_dense_wgrad=is_grouped_dense_wgrad,
        )

    @staticmethod
    def impl(
        lhs_data,
        lhs_scale_inv,
        rhs_data,
        rhs_scale_inv,
        bias,
        group_sizes,
        group_offset,
        M,
        N,
        K,
        lhs_is_trans,
        rhs_is_trans,
        scaling_mode,
        out_dtype,
        has_bias,
        is_grouped_dense_wgrad,
    ):
        assert GroupedGemmPrimitive.inner_primitive is not None
        (out, _) = GroupedGemmPrimitive.inner_primitive.bind(
            lhs_data,
            lhs_scale_inv,
            rhs_data,
            rhs_scale_inv,
            bias,
            group_sizes,
            group_offset,
            M=M,
            N=N,
            K=K,
            lhs_is_trans=lhs_is_trans,
            rhs_is_trans=rhs_is_trans,
            scaling_mode=scaling_mode,
            out_dtype=out_dtype,
            has_bias=has_bias,
            is_grouped_dense_wgrad=is_grouped_dense_wgrad,
        )
        return (out,)


register_primitive(GroupedGemmPrimitive)


def _shape_normalization(x, dimension_numbers, already_transposed: bool = False):
    orig_order = list(range(x.ndim))
    contracting_dims, batch_dims = dimension_numbers
    contracting_order = [d for d in orig_order if d in contracting_dims]
    batch_order = [d for d in orig_order if d in batch_dims]
    non_contracting_order = [
        d for d in orig_order if d not in contracting_dims and d not in batch_dims
    ]
    batch_shape = [x.shape[d] for d in batch_order]
    rows_shape = [x.shape[d] for d in non_contracting_order]
    cols_shape = [x.shape[d] for d in contracting_order]
    new_order = batch_order + non_contracting_order + contracting_order
    rows, cols, batches = (
        reduce(operator.mul, rows_shape, 1),
        reduce(operator.mul, cols_shape, 1),
        reduce(operator.mul, batch_shape, 1),
    )
    # Remove this transpose when non-TN dot is supported
    if not already_transposed:
        t = jnp.transpose(x, new_order)
    else:
        t = x
    return jnp.reshape(t, (batches, rows, cols))


def _calculate_remaining_shape(shape, contracting_dims):
    return tuple(shape[dim] for dim in range(len(shape)) if dim not in contracting_dims)


def _transpose_contract_dims(ndim, contracting_dims):
    return tuple(ndim - i - 1 for i in contracting_dims)[::-1]


# Apply jit to guarantee correctness of FP8 GEMM.
@partial(jax.jit, static_argnums=(2, 3))
def _jax_gemm_tensor_scaling_fp8(lhs, rhs, dim_nums, precision):
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dim_nums
    if lhs.data_layout == "T":
        lhs_contract = _transpose_contract_dims(lhs.data.ndim, lhs_contract)
    if rhs.data_layout == "T":
        rhs_contract = _transpose_contract_dims(rhs.data.ndim, rhs_contract)

    dim_nums = (lhs_contract, rhs_contract), (lhs_batch, rhs_batch)

    out_fp8 = jax.lax.dot_general(
        lhs.data, rhs.data, dim_nums, precision=precision, preferred_element_type=lhs.dq_dtype
    )
    scale_inv = lhs.scale_inv * rhs.scale_inv
    out = (out_fp8 * scale_inv).astype(lhs.dq_dtype)

    return out


@partial(jax.jit, static_argnums=(2,))
def _jax_gemm_mxfp8_1d(
    lhs: ScaledTensor, rhs: ScaledTensor, dim_nums: Tuple[Tuple[Sequence[int], Sequence[int]]]
):
    """
    JAX GEMM for MXFP8 via scaled_matmul
    """
    assert (
        rhs.scaling_mode == ScalingMode.MXFP8_1D_SCALING
    ), "rhs does not have MXFP8 1D scaling mode"

    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dim_nums

    expected_lhs_is_colwise = lhs_contract[-1] != lhs.data.ndim - 1
    expected_rhs_is_colwise = rhs_contract[-1] != rhs.data.ndim - 1
    assert lhs.is_colwise is expected_lhs_is_colwise, (
        f"LHS with unexpected quantize dimension.\nExpect is_colwise={expected_lhs_is_colwise}, got"
        f" {lhs.is_colwise}"
    )
    assert rhs.is_colwise is expected_rhs_is_colwise, (
        f"RHS with unexpected quantize dimension.\nExpect is_colwise={expected_rhs_is_colwise}, got"
        f" {rhs.is_colwise}"
    )

    # Reshape + Transpose (if needed)
    # [..., M, K] -> [1, reduce(..., M), K]
    # [..., K, M] -> [1, reduce(..., M), K]
    lhs_3d = _shape_normalization(lhs.data, (lhs_contract, lhs_batch))
    rhs_3d = _shape_normalization(rhs.data, (rhs_contract, rhs_batch))
    lhs_scale_3d = _shape_normalization(lhs.scale_inv, (lhs_contract, lhs_batch))
    rhs_scale_3d = _shape_normalization(rhs.scale_inv, (rhs_contract, rhs_batch))

    # Slice out the padding as scaled_matmul does not support padded scales yet
    lhs_scale_3d = jnp.asarray(lhs_scale_3d[:, : lhs_3d.shape[1], : int(lhs_3d.shape[2] / 32)])
    rhs_scale_3d = jnp.asarray(rhs_scale_3d[:, : rhs_3d.shape[1], : int(rhs_3d.shape[2] / 32)])

    # JAX scaled_matmul only supports NT now (TN-gemm)
    # * Expected shape:
    # * lhs_data  (B, M, K)           * rhs_data  (B, N, K)
    # * lhs_scale (B, M, K_block)     * rhs_scale (B, N, K_block)
    out_3d = jax.nn.scaled_matmul(
        lhs_3d, rhs_3d, lhs_scale_3d, rhs_scale_3d, preferred_element_type=lhs.dq_dtype
    )
    # Reshape [1, reduce(..., M), N] -> [..., M, N]
    lhs_remain_shape = tuple(
        lhs.data.shape[dim] for dim in range(len(lhs.data.shape)) if dim not in lhs_contract
    )
    rhs_remain_shape = tuple(
        rhs.data.shape[dim] for dim in range(len(rhs.data.shape)) if dim not in rhs_contract
    )
    out = out_3d.reshape(*lhs_remain_shape, *rhs_remain_shape)
    return out


def _jax_gemm(
    lhs: Union[jnp.ndarray, ScaledTensor],
    rhs: Union[jnp.ndarray, ScaledTensor],
    contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((1,), (0,)),
    quantizer_set: Dict["str", Quantizer] = noop_quantizer_set,
) -> jnp.ndarray:
    """
    FP8 GEMM via JAX
    """

    dim_nums = (contracting_dims, ((), ()))

    def _jax_gemm_fp8_impl(lhs, rhs):
        if lhs.scaling_mode.is_tensor_scaling():
            assert (
                rhs.scaling_mode == lhs.scaling_mode
            ), f"rhs.scaling_mode={rhs.scaling_mode} != lhs.scaling_mode={lhs.scaling_mode}"
            precision = (
                jax.lax.Precision.HIGHEST
                if QuantizeConfig.FP8_2X_ACC_FPROP
                else jax.lax.Precision.DEFAULT
            )
            return _jax_gemm_tensor_scaling_fp8(lhs, rhs, dim_nums, precision)

        if lhs.scaling_mode == ScalingMode.MXFP8_1D_SCALING:
            return _jax_gemm_mxfp8_1d(lhs, rhs, dim_nums)

        raise NotImplementedError("Unsupported ScalingMode: {lhs.scaling_mode}")

    if isinstance(lhs, ScaledTensor) and isinstance(rhs, ScaledTensor):
        return _jax_gemm_fp8_impl(lhs, rhs)

    if not isinstance(lhs, ScaledTensor) and not isinstance(rhs, ScaledTensor):
        if quantizer_set != noop_quantizer_set:
            assert type(quantizer_set.x) is type(quantizer_set.kernel)
            if (
                quantizer_set.x.scaling_mode.is_tensor_scaling()
                and is_fp8_gemm_with_all_layouts_supported()
            ):
                lhs_is_rowwise = rhs_is_rowwise = True
            else:
                (((lhs_contract_dim,), (rhs_contract_dim,)), _) = dim_nums
                lhs_is_rowwise = lhs_contract_dim == lhs.ndim - 1
                rhs_is_rowwise = rhs_contract_dim == rhs.ndim - 1
            lhs_q = quantizer_set.x.quantize(
                lhs,
                is_rowwise=lhs_is_rowwise,
                is_colwise=not lhs_is_rowwise,
            )
            rhs_q = quantizer_set.kernel.quantize(
                rhs,
                is_rowwise=rhs_is_rowwise,
                is_colwise=not rhs_is_rowwise,
            )
            return _jax_gemm_fp8_impl(lhs_q, rhs_q)

    if (
        isinstance(lhs, jnp.ndarray)
        and isinstance(rhs, jnp.ndarray)
        and quantizer_set == noop_quantizer_set
    ):
        return jax.lax.dot_general(lhs, rhs, dim_nums, preferred_element_type=lhs.dtype)

    raise NotImplementedError("Not supporting multiplication of ScaledTensor and jnp.array")


def gemm(
    lhs: Union[jnp.ndarray, ScaledTensor],
    rhs: Union[jnp.ndarray, ScaledTensor],
    contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((1,), (0,)),
    quantizer_set: QuantizerSet = noop_quantizer_set,
) -> jnp.ndarray:
    """General matrix multiplication with optional quantization.

    Args:
        lhs: First input matrix.
        rhs: Second input matrix.
        contracting_dims: Tuple of two sequences representing the contracting dimensions.
            The first sequence represents the contracting dimensions of the first matrix,
            and the second sequence represents the contracting dimensions of the second matrix.
        quantizer_set: Set of quantizers for FP8 quantization of the output.
            If None, no quantization is applied and the output has the same dtype as the inputs.

    Returns:
        If quantizer_set is None:
            The matrix multiplication result.
            Shape: (M, N)
            Dtype: Same as input dtype
          If quantizer_set is provided:
            A ScaledTensor containing the quantized matrix multiplication result.
    """

    return _jax_gemm(lhs, rhs, contracting_dims, quantizer_set)


def grouped_gemm(
    lhs: Union[jnp.ndarray, GroupedScaledTensor1x],
    rhs: Union[jnp.ndarray, GroupedScaledTensor1x],
    group_sizes: jnp.ndarray,
    contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((1,), (2,)),
    bias: jnp.ndarray = None,
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT,
    preferred_element_type: jnp.dtype = None,
    group_offset: jnp.array = None,
    quantizer_set: QuantizerSet = noop_quantizer_set,
) -> jnp.ndarray:
    """
    Grouped GEMM operation.

    Args:
        lhs: Left-hand side input matrix, can be a jnp.ndarray or GroupedScaledTensor1x
        rhs: Right-hand side input matrix, can be a jnp.ndarray or GroupedScaledTensor1x
        group_sizes: 1D array containing the sizes of each group
        contracting_dims: Tuple of two sequences representing the contracting dimensions
        bias: Bias tensor of shape (G, N)
        precision: JAX precision for the GEMM operation
        preferred_element_type: Preferred data type for the output tensor
        group_offset: 1D array containing offsets for each group (not yet implemented)
        quantizer_set: Set of quantizers for FP8 quantization of the input and output

    Returns:
        A jnp.ndarray containing the result of the grouped GEMM operation

    Note:
        Tested shapes:
        lhs: [M, K] or [K, N]
        rhs: [G, N, K] or [G, K, N] or [G * K, N] or [N, G * K]
    """
    # TODO(Phuong): implement the group_offset
    group_offset = group_offset or jnp.zeros((1,), jnp.int32)

    # TODO(Phuong): implement the precision
    del precision

    if isinstance(lhs, jnp.ndarray):
        assert isinstance(rhs, jnp.ndarray)
        out_dtype = lhs.dtype
        lhs_shape = lhs.shape
        rhs_shape = rhs.shape
        lhs_data = lhs
        rhs_data = rhs
        lhs_scale_inv = rhs_scale_inv = jnp.empty((0,), jnp.float32)
        scaling_mode = ScalingMode.NO_SCALING
    elif isinstance(lhs, GroupedScaledTensor1x):
        assert isinstance(rhs, GroupedScaledTensor1x)
        out_dtype = lhs.dq_dtype
        lhs_shape = lhs.original_shape
        rhs_shape = rhs.original_shape
        lhs_data = lhs.data
        rhs_data = rhs.data
        lhs_scale_inv = lhs.scale_inv
        rhs_scale_inv = rhs.scale_inv
        assert lhs.scaling_mode == rhs.scaling_mode
        scaling_mode = lhs.scaling_mode
    else:
        raise TypeError("Unsupported lhs type object!")

    out_dtype = preferred_element_type or out_dtype

    lhs_contract_dim, rhs_contract_dim = contracting_dims

    lhs_is_trans = lhs_contract_dim[-1] != len(lhs_shape) - 1
    lhs_flatten_axis = len(lhs_contract_dim) * (1 if lhs_is_trans else -1)

    # rhs_shape [G, K, N]
    rhs_is_trans = rhs_contract_dim[0] != 1
    rhs_flatten_axis = -len(rhs_contract_dim) if rhs_is_trans else 1 + len(rhs_contract_dim)

    is_grouped_dense_wgrad = False
    if len(rhs_shape) == 2:
        rhs_is_trans = rhs_contract_dim[0] != 0
        is_grouped_dense_wgrad = True

    # TODO(Hua): thses are for fp16 dense wgrad, any better way to handle this?
    if (
        is_grouped_dense_wgrad
        and not isinstance(lhs, ScaledTensor)
        and not isinstance(rhs, ScaledTensor)
    ):
        lhs_is_trans = True
        rhs_is_trans = False
        lhs_flatten_axis = 1
        rhs_flatten_axis = 1

    if (
        not isinstance(lhs, ScaledTensor)
        and not isinstance(rhs, ScaledTensor)
        and quantizer_set != noop_quantizer_set
    ):
        assert isinstance(quantizer_set.x, GroupedQuantizer)
        assert type(quantizer_set.x) is type(quantizer_set.kernel)
        scaling_mode = quantizer_set.x.scaling_mode
        if (
            quantizer_set.x.scaling_mode.is_tensor_scaling()
            and is_fp8_gemm_with_all_layouts_supported()
        ):
            lhs_is_rowwise = rhs_is_rowwise = True
        else:
            lhs_is_rowwise = not lhs_is_trans
            rhs_is_rowwise = rhs_is_trans
        quantizer_set.x.q_layout = (
            QuantizeLayout.ROWWISE if lhs_is_rowwise else QuantizeLayout.COLWISE
        )
        quantizer_set.kernel.q_layout = (
            QuantizeLayout.ROWWISE if rhs_is_rowwise else QuantizeLayout.COLWISE
        )
        lhs_q = grouped_quantize(lhs, quantizer_set.x, group_sizes, lhs_flatten_axis)
        rhs_q = grouped_quantize(
            rhs, quantizer_set.kernel, group_sizes=None, flatten_axis=rhs_flatten_axis
        )
        lhs_data = lhs_q.data
        rhs_data = rhs_q.data
        lhs_scale_inv = lhs_q.scale_inv
        rhs_scale_inv = rhs_q.scale_inv
        lhs_shape = lhs_q.original_shape
        rhs_shape = rhs_q.original_shape

    assert not (
        lhs_data.dtype == jnp.float8_e5m2 and rhs_data.dtype == jnp.float8_e5m2
    ), "FP8 GEMM does not support E5M2 * E5M2"

    # Only support FP8 GEMM with NT layout on Hopper and other earlier GPUs
    # thus additional transpose is required
    if scaling_mode.is_tensor_scaling() and not is_fp8_gemm_with_all_layouts_supported():
        if isinstance(lhs, ScaledTensor) and isinstance(rhs, ScaledTensor):
            lhs_layout_is_T = lhs.data_layout == "T"
            rhs_layout_is_T = rhs.data_layout == "T"
        else:
            lhs_layout_is_T = lhs_q.data_layout == "T"
            rhs_layout_is_T = rhs_q.data_layout == "T"
        # we can't apply _shape_normalization on the grouped input
        # thus we need to ensure that lhs is in N and rhs is in T
        assert (
            lhs_is_trans == lhs_layout_is_T
        ), "lhs input must be transposed before calling grouped_gemm"
        assert (
            not rhs_is_trans == rhs_layout_is_T
        ), "rhs input must be transposed before calling grouped_gemm"
        lhs_is_trans = False
        rhs_is_trans = True
        lhs_ndim = len(lhs_shape)
        rhs_ndim = len(rhs_shape)
        if lhs_layout_is_T:
            lhs_contract_dim = tuple((lhs_ndim - 1 - i) % lhs_ndim for i in lhs_contract_dim)
        if rhs_layout_is_T:
            # For rhs [G, K, N], need to exclude the G dim from contract_dim
            if group_sizes.size == rhs_shape[0]:
                rhs_contract_dim = tuple(
                    (rhs_ndim - 1 - i) % (rhs_ndim - 1) + 1 for i in rhs_contract_dim
                )
            else:
                rhs_contract_dim = tuple((rhs_ndim - 1 - i) % rhs_ndim for i in rhs_contract_dim)

    # Calling GroupedGEMM Custom Call
    K_lhs = math.prod(lhs_shape[i] for i in lhs_contract_dim)
    K_rhs = math.prod(rhs_shape[i] for i in rhs_contract_dim)
    assert K_lhs == K_rhs
    M = math.prod(_calculate_remaining_shape(lhs_shape, lhs_contract_dim))
    N = math.prod(_calculate_remaining_shape(rhs_shape, rhs_contract_dim)[1:])  # Exclude G

    if is_grouped_dense_wgrad:
        N = math.prod(_calculate_remaining_shape(rhs_shape, rhs_contract_dim))
    else:
        assert group_sizes.size == rhs_shape[0]

    assert group_offset.size == 1

    has_bias = bias is not None
    assert not has_bias or bias.shape == (group_sizes.size, N)
    bias = jnp.empty((), jnp.float32) if bias is None else bias

    (out,) = GroupedGemmPrimitive.outer_primitive.bind(
        lhs_data,
        lhs_scale_inv,
        rhs_data,
        rhs_scale_inv,
        bias,
        group_sizes,
        group_offset,
        M=M,
        N=N,
        K=K_lhs,
        lhs_is_trans=lhs_is_trans,
        rhs_is_trans=rhs_is_trans,
        scaling_mode=scaling_mode.value,
        out_dtype=out_dtype,
        has_bias=has_bias,
        is_grouped_dense_wgrad=is_grouped_dense_wgrad,
    )
    return out
