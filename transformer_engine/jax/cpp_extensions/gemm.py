# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX te modules"""

import operator
from collections.abc import Iterable
from typing import Tuple, Sequence, Union, Dict
from functools import partial, reduce

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

from transformer_engine_jax import get_device_compute_capability

from .base import BasePrimitive, register_primitive

from ..quantize import (
    ScaledTensor,
    ScalingMode,
    Quantizer,
    QuantizerSet,
    QuantizeConfig,
    noop_quantizer_set,
)

from ..sharding import get_padded_spec


__all__ = [
    "gemm",
    "te_gemm_impl"
]


num_cublas_streams = 4


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
    impl_static_args = ()
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(*args, num_gemms, scaling_mode, out_dtype, has_bias):
        """
        Args:
            *args: Size num_gemms * 4 or num_gemms * 5 depending on has_bias:
                args[  0         :   num_gemms] are the lhs tensors,
                args[  num_gemms : 2*num_gemms] are the rhs tensors,
                args[2*num_gemms : 3*num_gemms] are the lhs scale_inv tensors,
                args[3*num_gemms : 4*num_gemms] are the rhs scale_inv tensors,
                args[4*num_gemms : 5*num_gemms] are the bias tensors if has_bias is True.
            num_gemms: Number of GEMM operations to perform.
            scaling_mode: Scaling mode for the GEMM operations.
            out_dtype: Data type of the output tensors.
            has_bias: Boolean indicating if bias tensors are provided.

        Returns:
           A tuple of ShapedArray objects of size num_gemms+1:
               ret[0 : num_gemms]: GEMM output tensors,
               ret[num_gemms]:workspace tensor.
        """
        del scaling_mode
        expected_num_args = 5 * num_gemms if has_bias else 4 * num_gemms
        assert (
            len(args) == expected_num_args
        ), f"Expected {expected_num_args} input arguments, but got {len(args)}"
        A_list = args[0:num_gemms]
        B_list = args[num_gemms : 2 * num_gemms]
        # A and B have shapes [1, m, k] and [1, n, k]
        out_list_aval = tuple(
            jax.core.ShapedArray((A.shape[1], B.shape[1]), dtype=out_dtype)
            for A, B in zip(A_list, B_list)
        )
        workspace_size = get_cublas_workspace_size_bytes() * num_cublas_streams
        workspace_aval = jax.core.ShapedArray(shape=(workspace_size,), dtype=jnp.uint8)
        return (*out_list_aval, workspace_aval)

    @staticmethod
    def outer_abstract(*args, **kwargs):
        (out_aval, _) = GroupedGemmPrimitive.abstract(*args, **kwargs)
        return out_aval

    @staticmethod
    def lowering(ctx, *args, num_gemms, scaling_mode, out_dtype, has_bias):
        del out_dtype
        return jax.ffi.ffi_lowering(GroupedGemmPrimitive.name)(
            ctx,
            *args,
            num_gemms=num_gemms,
            scaling_mode=int(scaling_mode),
            has_bias=has_bias,
        )

    @staticmethod
    def impl(*args, num_gemms, scaling_mode, out_dtype, has_bias):
        assert GroupedGemmPrimitive.inner_primitive is not None
        out = GroupedGemmPrimitive.inner_primitive.bind(
            *args,
            num_gemms=num_gemms,
            scaling_mode=scaling_mode.value,
            out_dtype=out_dtype,
            has_bias=has_bias,
        )
        return out[:-1]  # out is [out_list, wkspace], only return out_list


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
        lhs.data, rhs.data, dim_nums, precision=precision, preferred_element_type=jnp.float32
    )
    scale_inv = (lhs.scale_inv * rhs.scale_inv).astype(jnp.float32)

    return (out_fp8 * scale_inv).astype(lhs.dq_dtype)


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
            (((lhs_contract_dim,), (rhs_contract_dim,)), _) = dim_nums
            lhs_is_rowwise = lhs_contract_dim == lhs.ndim - 1
            rhs_is_rowwise = rhs_contract_dim == rhs.ndim - 1
            # Call JAX quantization so that XLA can do pattern matching (QDQ --> FP8 gemm)
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
    quantizer_set: Dict["str", Quantizer] = noop_quantizer_set,
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


def sanitize_dims(dims: Sequence[int], ndim: int) -> Sequence[int]:
    """Convert relative (negative) dimension indexes to absolute (positive) dimensions."""
    dims_ = dims if isinstance(dims, Iterable) else (dims, )
    if len(dims_) == 0:
        return dims_
    return tuple([ ndim + dim if dim < 0 else dim for dim in dims_ ])


def get_gemm_layout(contracting_dims: Tuple[Sequence[int], Sequence[int]],
                    operand_ndims: Tuple[int, int]) -> Tuple[bool, bool]:
    """Convert JAX-style contracting dimensions into cuBLAS-style transpose flags."""
    lhs_contracting, rhs_contracting = map(sanitize_dims, contracting_dims, operand_ndims)
    transpose_lhs = operand_ndims[0] - 1 not in lhs_contracting
    transpose_rhs = operand_ndims[1] - 1 in rhs_contracting
    return transpose_lhs, transpose_rhs


def is_non_nt_fp8_gemm_supported():
    """Check if the device compute capability supports cuBLAS FP8 GEMM with different layouts."""
    arch = get_device_compute_capability(0)
    return (100 <= arch < 120) or (arch >= 130)


class GemmPrimitive(BasePrimitive):
    """
    Primitive for cuBLAS GEMM
    """

    name = "te_gemm_ffi"
    multiple_results = True
    impl_static_args = (6, 7, 8, 9, 10, 11, 12, 13, 14)
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(lhs, lhs_scale_inv, rhs, rhs_scale_inv, bias, gelu_input, contracting_dims,
                 scaling_mode, lhs_scaled_colwise, rhs_scaled_colwise, fuse_bias, fuse_gelu,
                 grad, accumulate, use_split_accumulator):
        del lhs_scaled_colwise, rhs_scaled_colwise, accumulate, use_split_accumulator
        operand_ndims = (lhs.ndim, rhs.ndim)
        (
            lhs_contracting_dims,
            rhs_contracting_dims,
        ) = map(sanitize_dims, contracting_dims, operand_ndims)
        lhs_is_transposed, rhs_is_transposed = get_gemm_layout(contracting_dims, operand_ndims)

        # Sanity-check operand types and layouts
        assert lhs.dtype == rhs.dtype, (
            f"cuBLAS GEMM operands have incompatible dtypes: {lhs.dtype} X {rhs.dtype}"
        )
        assert not (lhs_is_transposed and rhs_is_transposed), (
            "cuBLAS GEMM operands cannot both be transposed."
        )
        if scaling_mode != ScalingMode.NO_SCALING:
            assert lhs_scale_inv.size > 0 and rhs_scale_inv.size > 0, (
                "Quantized cuBLAS GEMM requires inverse scaling factors for both operands."
            )
            if not is_non_nt_fp8_gemm_supported():
                assert not lhs_is_transposed and rhs_is_transposed, (
                    "Quantized cuBLAS GEMM on devices with compute capability < 10.0 (Hopper) "
                    "require non-transposed LHS and transposed RHS operands "
                    "(`contracting_dims=((-1, ), (-1, ))`)."
                )

        lhs_contracting_size, rhs_contracting_size = map(
            lambda shape, dims: reduce(operator.mul, [shape[dim] for dim in dims]),
            (lhs.shape, rhs.shape),
            (lhs_contracting_dims, rhs_contracting_dims)
        )
        assert lhs_contracting_size == rhs_contracting_size, (
            "cuBLAS GEMM operands have incompatible contracting dimensions: "
            f"{lhs.shape} @ idx {lhs_contracting_dims} X {rhs.shape} @ idx {rhs_contracting_dims}."
        )

        # Determine output shape and dtype
        lhs_leading_shape, rhs_leading_shape = map(
            lambda shape, dims: [ shape[dim] for dim in range(len(shape)) if dim not in dims ],
            (lhs.shape, rhs.shape),
            (lhs_contracting_dims, rhs_contracting_dims)
        )
        output_shape = (reduce(operator.mul, lhs_leading_shape),
                        reduce(operator.mul, rhs_leading_shape))
        output_dtype = jnp.bfloat16 if scaling_mode != ScalingMode.NO_SCALING else lhs.dtype
        if lhs.ndim > 2:
            if rhs.ndim > 2:
                assert lhs_is_transposed and not rhs_is_transposed, (
                    "cuBLAS GEMM operands can both be batched only if LHS is transposed and RHS "
                    "is not, i.e. batched dimensions are contracting."
                )
            else:
                # Restore batched dimensions in the output
                output_shape = (*lhs_leading_shape, output_shape[-1])
        output = jax.core.ShapedArray(shape=output_shape, dtype=output_dtype)

        # Validate bias
        bias_size = 0
        bias_dtype = output_dtype
        if fuse_bias:
            bias_size = output_shape[-1]
            if not grad:
                assert bias.ndim == 1 and bias.size == bias_size, (
                    "cuBLAS GEMM bias tensor has incorrect shape, "
                    f"expected ({bias_size}, ) but found {bias.shape}."
                )
                assert bias.dtype == output_dtype, (
                    "cuBLAS GEMM bias tensor has incorrect data type, "
                    f"expected {bias_dtype} but found {bias.dtype}."
                )
        bias_grad = jax.core.ShapedArray(shape=(bias_size, ), dtype=bias_dtype)

        # Validate pre-GeLU
        pre_gelu_shape = (0, )
        pre_gelu_dtype = output_dtype
        if fuse_gelu:
            pre_gelu_shape = output_shape
            if grad:
                pre_gelu_ndim = len(pre_gelu_shape)
                assert (
                    gelu_input.ndim == pre_gelu_shape
                    and all(gelu_input.shape[i] == pre_gelu_shape[i] for i in range(pre_gelu_ndim))
                ), (
                    "cuBLAS GEMM pre-GeLU tensor has incorrect shape, "
                    f"expected {pre_gelu_shape} but found {gelu_input.shape}."
                )
                assert gelu_input.dtype == output_dtype, (
                    "cuBLAS GEMM pre-GeLU tensor has incorrect data type, "
                    f"expected {pre_gelu_dtype} but found {gelu_input.dtype}."
                )
        pre_gelu_out = jax.core.ShapedArray(shape=pre_gelu_shape, dtype=pre_gelu_dtype)

        # Declare cuBLAS workspace
        workspace = jax.core.ShapedArray(shape=(get_cublas_workspace_size_bytes(), ),
                                         dtype=jnp.uint8)

        return output, bias_grad, pre_gelu_out, workspace

    @staticmethod
    def outer_abstract(*args, **kwargs):
        outputs = GemmPrimitive.abstract(*args, **kwargs)
        return outputs[:-1]  # discard cuBLAS workspace

    @staticmethod
    def lowering(ctx, lhs, lhs_scale_inv, rhs, rhs_scale_inv, bias, gelu_input, contracting_dims,
                 scaling_mode, lhs_scaled_colwise, rhs_scaled_colwise, fuse_bias, fuse_gelu, grad,
                 accumulate, use_split_accumulator):
        lhs_aval, _, rhs_aval, *_ = ctx.avals_in
        lhs_cdims, rhs_cdims = map(sanitize_dims, contracting_dims, (lhs_aval.ndim, rhs_aval.ndim))
        lhs_transposed, rhs_transposed = get_gemm_layout((lhs_cdims, rhs_cdims),
                                                         (lhs_aval.ndim, rhs_aval.ndim))

        args = (lhs, lhs_scale_inv, rhs, rhs_scale_inv, bias, gelu_input)
        kwargs = {
            "lhs_axis_boundary" : max(lhs_cdims) + 1 if lhs_transposed else min(lhs_cdims),
            "rhs_axis_boundary" : min(rhs_cdims) if rhs_transposed else max(rhs_cdims) + 1,
            "scaling_mode" : int(scaling_mode.value),
            "lhs_scaled_colwise" : lhs_scaled_colwise,
            "rhs_scaled_colwise" : rhs_scaled_colwise,
            "lhs_transposed" : lhs_transposed,
            "rhs_transposed" : rhs_transposed,
            "fuse_bias" : fuse_bias,
            "fuse_gelu" : fuse_gelu,
            "grad" : grad,
            "accumulate" : accumulate,
            "use_split_accumulator" : use_split_accumulator,
        }

        operand_output_aliases = {}
        if fuse_bias and not grad:
            operand_output_aliases.update({ 4 : 1 })  # bias <-> bias_grad
        if fuse_gelu and grad:
            operand_output_aliases.update({ 5 : 2 })  # gelu_input <-> pre_gelu_out

        return jax.ffi.ffi_lowering(
            GemmPrimitive.name,
            operand_output_aliases=operand_output_aliases
        )(ctx, *args, **kwargs)

    @staticmethod
    def impl(lhs, lhs_scale_inv, rhs, rhs_scale_inv, bias, gelu_input, contracting_dims,
             scaling_mode, lhs_scaled_colwise, rhs_scaled_colwise, fuse_bias, fuse_gelu, grad,
             accumulate, use_split_accumulator):
        outputs = GemmPrimitive.inner_primitive.bind(
            lhs,
            lhs_scale_inv,
            rhs,
            rhs_scale_inv,
            bias,
            gelu_input,
            contracting_dims=contracting_dims,
            scaling_mode=scaling_mode,
            lhs_scaled_colwise=lhs_scaled_colwise,
            rhs_scaled_colwise=rhs_scaled_colwise,
            fuse_bias=fuse_bias,
            fuse_gelu=fuse_gelu,
            grad=grad,
            accumulate=accumulate,
            use_split_accumulator=use_split_accumulator,
        )
        return outputs[:-1]  # discard cuBLAS workspace

    @staticmethod
    def batcher(batched_args, batch_dims, contracting_dims, scaling_mode, lhs_scaled_colwise,
                rhs_scaled_colwise, fuse_bias, fuse_gelu, grad, accumulate, use_split_accumulator):
        assert GemmPrimitive.outer_primitive is not None
        lhs, _, rhs, *_ = batched_args
        lhs_bdims, *_ = batch_dims

        # Output is batched like LHS only if LHS is batched and RHS is not
        out_bdims = lhs_bdims if lhs.ndim > 2 and rhs.ndim == 2 else (None, )
        bias_bdims = (None, )  # Bias is never batched
        pre_gelu_bdims = (None, )  # Pre-GeLU output, if exists, is batched like GEMM output
        if fuse_gelu and not grad:
            pre_gelu_bdims = out_bdims

        return (
            GemmPrimitive.outer_primitive.bind(
                *batched_args,
                contracting_dims=contracting_dims,
                scaling_mode=scaling_mode,
                lhs_scaled_colwise=lhs_scaled_colwise,
                rhs_scaled_colwise=rhs_scaled_colwise,
                fuse_bias=fuse_bias,
                fuse_gelu=fuse_gelu,
                grad=grad,
                accumulate=accumulate,
                use_split_accumulator=use_split_accumulator,
            ),
            (out_bdims, bias_bdims, pre_gelu_bdims)
        )

    @staticmethod
    def infer_sharding_from_operands(contracting_dims, scaling_mode, lhs_scaled_colwise,
                                     rhs_scaled_colwise, fuse_bias, fuse_gelu, grad, accumulate,
                                     use_split_accumulator, mesh, arg_infos, result_infos):
        del lhs_scaled_colwise, rhs_scaled_colwise, scaling_mode, accumulate, use_split_accumulator, result_infos

        # Check contracting dimensions
        lhs_spec, _, rhs_spec, *_ = map(get_padded_spec, arg_infos)
        operand_ndims = (len(lhs_spec), len(rhs_spec))
        lhs_contracting_dims, rhs_contracting_dims = map(
            sanitize_dims, contracting_dims, operand_ndims
        )
        lhs_contracting_specs, rhs_contracting_specs = map(
            lambda specs, dims: [ specs[dim] for dim in dims if specs[dim] is not None],
            (lhs_spec, rhs_spec),
            (lhs_contracting_dims, rhs_contracting_dims)
        )
        assert len(lhs_contracting_specs) <= 1 and len(rhs_contracting_specs) <= 1, (
            "cuBLAS GEMM operands can have only one sharded contracting dimension."
        )
        lhs_contracting_spec, rhs_contracting_spec = map(
            lambda spec: None if len(spec) == 0 else spec[0],
            (lhs_contracting_specs, rhs_contracting_specs)
        )
        assert lhs_contracting_spec == rhs_contracting_spec, (
            "cuBLAS GEMM operands must have the same sharding in contracting dimensions."
        )

        # Sanity check leading dimensions, allow for simultaneous batch and sequence sharding
        lhs_leading_dims, rhs_leading_dims = map(
            lambda ndim, excludes: [ dim for dim in range(ndim) if dim not in excludes ],
            operand_ndims,
            (lhs_contracting_dims, rhs_contracting_dims)
        )
        lhs_leading_specs, rhs_leading_specs = map(
            lambda specs, dims: [ specs[dim] for dim in dims if specs[dim] is not None ],
            (lhs_spec, rhs_spec),
            (lhs_leading_dims, rhs_leading_dims)
        )
        assert len(lhs_leading_specs) <= 1 and len(rhs_leading_specs) <= 1, (
            "cuBLAS GEMM operands cannot have more than one sharded leading dimensions. This error "
            "usually means a sequence-parallel operand was not all-gathered before the GEMM op."
        )

        # Determine output sharding
        lhs_leading_spec, rhs_leading_spec = map(
            lambda spec: None if len(spec) == 0 else spec[0],
            (lhs_leading_specs, rhs_leading_specs)
        )
        out_spec = (lhs_leading_spec, rhs_leading_spec)
        if operand_ndims[0] > 2 and operand_ndims[1] == 2:
            # Restore batch dimensions/sharding to the output
            out_spec = (*lhs_leading_specs, rhs_leading_spec)
        out_sharding = NamedSharding(mesh, PartitionSpec(*out_spec))

        # Bias gradient sharding inherits the RHS contracting spec
        bias_spec = (None, )
        if fuse_bias and grad:
            bias_spec = (rhs_contracting_spec, )
        bias_sharding = NamedSharding(mesh, PartitionSpec(*bias_spec))

        # Pre-GeLU sharding matches output sharding
        pre_gelu_spec = (None, )
        if fuse_gelu and not grad:
            pre_gelu_spec = out_spec
        pre_gelu_sharding = NamedSharding(mesh, PartitionSpec(*pre_gelu_spec))

        return (out_sharding, bias_sharding, pre_gelu_sharding)

    @staticmethod
    def partition(contracting_dims, scaling_mode, lhs_scaled_colwise, rhs_scaled_colwise,
                  fuse_bias,fuse_gelu, grad, accumulate, use_split_accumulator, mesh, arg_infos,
                  result_infos):
        out_shardings = GemmPrimitive.infer_sharding_from_operands(
            contracting_dims, scaling_mode, lhs_scaled_colwise, rhs_scaled_colwise, fuse_bias,
            fuse_gelu, grad, accumulate, use_split_accumulator, mesh, arg_infos, result_infos
        )
        output_spec = out_shardings[0].spec

        # Operand shardings are already guarded with asserts so leave them unchanged here
        lhs_spec, _, rhs_spec, *_ = map(get_padded_spec, arg_infos)
        lhs_sharding = NamedSharding(mesh, PartitionSpec(*lhs_spec))
        rhs_sharding = NamedSharding(mesh, PartitionSpec(*rhs_spec))

        # Any distributed scales (e.g. MXFP8) need to be gathered
        scale_sharding = NamedSharding(mesh, PartitionSpec(None))

        # Bias has to be sharded same as the trailing dimension of the GEMM output
        bias_spec = (None, )
        if fuse_bias and not grad:
            bias_spec = (output_spec[-1], )
        bias_sharding = NamedSharding(mesh, PartitionSpec(*bias_spec))

        # Pre-GeLU output has to be sharded same as the GEMM output
        pre_gelu_spec = (None, )
        if fuse_gelu and grad:
            pre_gelu_spec = output_spec
        pre_gelu_sharding = NamedSharding(mesh, PartitionSpec(*pre_gelu_spec))

        arg_shardings = (
            lhs_sharding,
            scale_sharding,
            rhs_sharding,
            scale_sharding,
            bias_sharding,
            pre_gelu_sharding,
        )

        return mesh, GemmPrimitive.impl, out_shardings, arg_shardings


register_primitive(GemmPrimitive)


@partial(jax.jit, static_argnums=(6, 7, 8, 9, 10, 11, 12, 13, 14))
def _te_gemm_impl(lhs, lhs_scale_inv, rhs, rhs_scale_inv, bias, gelu_input, contracting_dims,
                  scaling_mode, lhs_scaled_colwise, rhs_scaled_colwise, fuse_bias, fuse_gelu,
                  grad, accumulate, use_split_accumulator):
    return GemmPrimitive.outer_primitive.bind(
        lhs,
        lhs_scale_inv,
        rhs,
        rhs_scale_inv,
        bias,
        gelu_input,
        contracting_dims=contracting_dims,
        scaling_mode=scaling_mode,
        lhs_scaled_colwise=lhs_scaled_colwise,
        rhs_scaled_colwise=rhs_scaled_colwise,
        fuse_bias=fuse_bias,
        fuse_gelu=fuse_gelu,
        grad=grad,
        accumulate=accumulate,
        use_split_accumulator=use_split_accumulator,
    )


def te_gemm_impl(
    lhs: jax.Array,
    rhs: jax.Array,
    bias: jax.Array = None,
    gelu_input: jax.Array = None,
    contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((-1, ), (0, )),
    scaling_mode: ScalingMode = ScalingMode.NO_SCALING,
    lhs_scaled_colwise: bool = False,
    lhs_scale_inv: jax.Array = None,
    rhs_scaled_colwise: bool = False,
    rhs_scale_inv: jax.Array = None,
    fuse_bias: bool = False,
    fuse_gelu: bool = False,
    grad: bool = False,
    accumulate: bool = False,
    use_split_accumulator: bool = False,
):
    r"""
    cuBLAS GEMM custom op.

    Parameters
    ----------
    lhs: jax.Array
        Left-hand side operand in GEMM.
    rhs: jax.Array
        Right-hand side operand in GEMM.
    bias: jax.Array, default = None
        Optional additive bias term, required for forward GEMM with bias fusion.
    gelu_input: jax.Array, default = None
        Pre-GeLU output from forward GEMM, required for backward/grad GEMM with dGeLU fusion.
    contracting_dims: Tuple[Sequence[int], Sequence[int]], default = ((-1, ), (0, ))
        Tuple of contracting dimensions for both operands.
    scaling_mode: ScalingMode, default = ScalingMode.NO_SCALING
        FP8 scaling mode for quantized GEMM.
    lhs_scale_inv: jax.Array, default = None
        Inverse scale factor for quantized LHS operand.
    lhs_scaled_colwise: bool, default = False
        Direction of LHS operand quantization, used for 1D block scaling modes.
    rhs_scale_inv: jax.Array, default = None
        Inverse scale factor for quantized RHS operand.
    rhs_scaled_colwise: bool, deafult = False
        Direction of RHS operand quantiazation,used for 1D block scaling modes.
    fuse_bias: bool, default = False
        Enable bias addition in forward GEMM or bias gradient in backward GEMM.
    fuse_gelu: bool, default = False
        Enable GeLU activation in forward GEMM or GeLU gradient in backward GEMM.
    grad: bool, default = False
        Flag for switching bias and GeLU fusions from forward to backward mode.

    Returns
    -------
    output: jax.Array
        GEMM output. When `fuse_bias=True` and `grad=False`, this result includes the additive
        bias. When `fuse_gelu=True`, the forward output with `grad=False` includes a GeLU
        application, while the backward output with `grad=True` includes the GeLU gradient
        contribution.
    bias_grad: jax.Array
        Bias gradient when `fuse_bias=True` and `grad=True`. Otherwise empty.
    pre_gelu_out: jax.Array
        Pure forward GEMM output before GeLU activation when `fuse_gelu=True` and `grad=False`.
        Otherwise emtpy. This is required as an input to the backward GEMM with `grad=True` in
        order to compute the GeLU gradient contribution.
    """
    out_dtype = lhs.dtype if scaling_mode == ScalingMode.NO_SCALING else jnp.bfloat16
    if bias is None or not (fuse_bias and not grad):
        bias = jnp.empty(0, dtype=out_dtype)
    if gelu_input is None or not (fuse_gelu and grad):
        gelu_input = jnp.empty(0, dtype=out_dtype)
    if lhs_scale_inv is None or scaling_mode == ScalingMode.NO_SCALING:
        lhs_scale_inv = jnp.empty(0, dtype=jnp.float32)
    if rhs_scale_inv is None or scaling_mode == ScalingMode.NO_SCALING:
        rhs_scale_inv = jnp.empty(0, dtype=jnp.float32)

    return _te_gemm_impl(lhs, lhs_scale_inv, rhs, rhs_scale_inv, bias, gelu_input,
                         contracting_dims, scaling_mode, lhs_scaled_colwise, rhs_scaled_colwise,
                         fuse_bias, fuse_gelu, grad, accumulate, use_split_accumulator)


"""
def swizzled_scale(scales):
    # Swizzle the scale tensor for FP8 GEMM
    assert scales.ndim == 2
    rows, cols = scales.shape
    scales = scales.reshape(rows // 128, 4, 32, cols // 4, 4)
    scales = jnp.transpose(scales, (0, 3, 2, 1, 4))
    scales = scales.reshape(rows, cols)
    return scales


def grouped_gemm(
    lhs_list: List[Union[jnp.ndarray, ScaledTensor]],
    rhs_list: List[Union[jnp.ndarray, ScaledTensor]],
    contracting_dims_list: List[Tuple[Sequence[int], Sequence[int]]],
    bias_list: List[jnp.ndarray] = None,
) -> List[jnp.ndarray]:
    # Grouped GEMM for multiple pairs of tensors.
    assert (
        len(lhs_list) == len(rhs_list) == len(contracting_dims_list)
    ), "lhs_list, rhs_list, contracting_dims_list must have the same length"

    num_gemms = len(lhs_list)
    lhs_list_ = []
    rhs_list_ = []
    lhs_sinv_list_ = []
    rhs_sinv_list_ = []
    bias_list_ = []
    for i in range(num_gemms):
        lhs = lhs_list[i]
        rhs = rhs_list[i]
        contracting_dims = contracting_dims_list[i]
        dim_nums = (contracting_dims, ((), ()))
        if isinstance(lhs, ScaledTensor) and isinstance(rhs, ScaledTensor):
            scaling_mode = lhs.scaling_mode
            lhs_shape = lhs.data.shape
            rhs_shape = rhs.data.shape
            out_dtype = lhs.dq_dtype
            # For ScaledTensors and DELAYED_TENSOR_SCALING, need to handle internal data_layout
            if lhs.scaling_mode.is_tensor_scaling():
                assert not (
                    lhs.data.dtype == jnp.float8_e5m2 and rhs.data.dtype == jnp.float8_e5m2
                ), "FP8 GEMM does not support E5M2 * E5M2"
                ((lhs_contract_dim,), (rhs_contract_dim,)) = contracting_dims
                if lhs.data_layout == "T":
                    lhs_contract_dim = (lhs_contract_dim - 1) % lhs.data.ndim
                if rhs.data_layout == "T":
                    rhs_contract_dim = (rhs_contract_dim - 1) % rhs.data.ndim
                dim_nums = ((lhs_contract_dim,), (rhs_contract_dim,)), ((), ())
        else:
            # For jnp.ndarray, only consider contracting_dims, data_layout is always NN
            scaling_mode = ScalingMode.NO_SCALING
            lhs_shape = lhs.shape
            rhs_shape = rhs.shape
            out_dtype = lhs.dtype

        (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dim_nums
        lhs_dn = (lhs_contract, lhs_batch)
        rhs_dn = (rhs_contract, rhs_batch)

        lhs_remain_shape = _calculate_remaining_shape(lhs_shape, lhs_contract)
        rhs_remain_shape = _calculate_remaining_shape(rhs_shape, rhs_contract)

        # Note: do not squeeze() for {lhs, rhs}_3d, it will trigger a D2D memcpy
        if scaling_mode == ScalingMode.NO_SCALING:
            lhs_3d = _shape_normalization(lhs, lhs_dn)
            rhs_3d = _shape_normalization(rhs, rhs_dn)
        elif scaling_mode.is_tensor_scaling():
            lhs_3d = _shape_normalization(lhs.data, lhs_dn, lhs.data_layout == "N")
            rhs_3d = _shape_normalization(rhs.data, rhs_dn, rhs.data_layout == "T")
        elif scaling_mode == ScalingMode.MXFP8_1D_SCALING:
            lhs_3d = _shape_normalization(lhs.data, lhs_dn)
            rhs_3d = _shape_normalization(rhs.data, rhs_dn)
            lhs_scale_inv = _shape_normalization(lhs.scale_inv, lhs_dn)
            rhs_scale_inv = _shape_normalization(rhs.scale_inv, rhs_dn)
            # swizzled_scale requires a matrix
            lhs_scale_inv = swizzled_scale(lhs_scale_inv.squeeze())
            rhs_scale_inv = swizzled_scale(rhs_scale_inv.squeeze())
        else:
            raise NotImplementedError("Unsupported ScalingMode: {scaling_mode}")

        # Note: already_transposed doesn't matter for the output shape
        # x.shape = [B, D1, D2]
        # contracting_dims = (2, )    --> output.shape = [1, B * D1, D2]
        # contracting_dims = (0, 1, ) --> output.shape = [1, D2, B * D1]
        # x.shape = [D1, D2]
        # contracting_dims = (1, )    --> output.shape = [1, D1, D2]
        # contracting_dims = (0, )    --> output.shape = [1, D2, D1]
        bm = lhs_remain_shape[0]
        bn = rhs_remain_shape[0]
        kl = lhs_3d.shape[-1]
        kr = rhs_3d.shape[-1]
        assert kl == kr, f"After shape normalization, contracting dim size mismatch: {kl} != {kr}"
        if (bm % 16 != 0) or (bn % 16 != 0) or (kl % 16 != 0):
            print("grouped_gemm input pair {i} has invalid problem shape for lowering: ")
            print(f"m = {bm}, n = {bn}, k = {kl}; ")
            print("cuBLAS requires the problem shapes being multiples of 16")
            assert (bm % 16 == 0) and (bn % 16 == 0) and (kl % 16 == 0)

        lhs_list_.append(lhs_3d)
        rhs_list_.append(rhs_3d)
        if scaling_mode == ScalingMode.NO_SCALING:
            lhs_sinv_list_.append(jnp.ones(1, dtype=jnp.float32))
            rhs_sinv_list_.append(jnp.ones(1, dtype=jnp.float32))
        if scaling_mode.is_tensor_scaling():
            lhs_sinv_list_.append(lhs.scale_inv)
            rhs_sinv_list_.append(rhs.scale_inv)
        if scaling_mode == ScalingMode.MXFP8_1D_SCALING:
            lhs_sinv_list_.append(lhs_scale_inv)
            rhs_sinv_list_.append(rhs_scale_inv)
        if bias_list is not None:
            bias_list_.append(bias_list[i])

    out_list = GroupedGemmPrimitive.outer_primitive.bind(
        *lhs_list_,
        *rhs_list_,
        *lhs_sinv_list_,
        *rhs_sinv_list_,
        *bias_list_,
        num_gemms=num_gemms,
        scaling_mode=scaling_mode,
        out_dtype=out_dtype,
        has_bias=1 if bias_list is not None else 0,
    )

    return out_list
"""
