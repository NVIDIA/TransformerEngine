# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX te modules"""

from typing import Tuple, Sequence, Union, Dict, List
from functools import partial, reduce
import operator
import jax
import jax.numpy as jnp
from transformer_engine_jax import get_device_compute_capability

from .base import BasePrimitive, register_primitive

from ..quantize import (
    ScaledTensor,
    ScalingMode,
    Quantizer,
    QuantizeConfig,
    noop_quantizer_set,
)

__all__ = ["gemm", "grouped_gemm"]


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
    impl_static_args = (6, 7, 8, 9)
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        lhs_contig_aval,
        lhs_scale_contig_aval,
        rhs_contig_aval,
        rhs_scale_contig_aval,
        bias_contig_aval,
        dim_list_aval,
        *,
        num_gemms,
        scaling_mode,
        out_dtype,
        out_flat_size,
    ):
        del lhs_contig_aval, lhs_scale_contig_aval
        del rhs_contig_aval, rhs_scale_contig_aval
        del bias_contig_aval, dim_list_aval
        del num_gemms, scaling_mode
        out_flat_aval = jax.core.ShapedArray(shape=(out_flat_size,), dtype=out_dtype)
        wkspace_size = get_cublas_workspace_size_bytes() * num_cublas_streams
        wkspace_aval = jax.core.ShapedArray(shape=(wkspace_size,), dtype=jnp.uint8)
        return (out_flat_aval, wkspace_aval)

    @staticmethod
    def outer_abstract(*args, **kwargs):
        (out_aval, _) = GroupedGemmPrimitive.abstract(*args, **kwargs)
        return out_aval

    @staticmethod
    def lowering(
        ctx,
        lhs_contig,
        lhs_scale_inv_contig,
        rhs_contig,
        rhs_scale_inv_contig,
        bias_contig,
        dim_list,
        *,
        num_gemms,
        scaling_mode,
        out_dtype,
        out_flat_size,
    ) -> jnp.ndarray:
        del out_dtype, out_flat_size
        return jax.ffi.ffi_lowering(GroupedGemmPrimitive.name)(
            ctx,
            lhs_contig,
            lhs_scale_inv_contig,
            rhs_contig,
            rhs_scale_inv_contig,
            bias_contig,
            dim_list,
            num_gemms=num_gemms,
            scaling_mode=int(scaling_mode),
        )

    @staticmethod
    def impl(
        lhs_contig,
        lhs_scale_inv_contig,
        rhs_contig,
        rhs_scale_inv_contig,
        bias_contig,
        dim_list,
        num_gemms,
        scaling_mode,
        out_dtype,
        out_flat_size,
    ) -> jnp.ndarray:
        assert GroupedGemmPrimitive.inner_primitive is not None
        out = GroupedGemmPrimitive.inner_primitive.bind(
            lhs_contig,
            lhs_scale_inv_contig,
            rhs_contig,
            rhs_scale_inv_contig,
            bias_contig,
            dim_list,
            num_gemms=num_gemms,
            scaling_mode=scaling_mode.value,
            out_dtype=out_dtype,
            out_flat_size=out_flat_size,
        )
        return out[0]  # out is [out_flat, wkspace], only return out_flat


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


def _dequantize(x, scale_inv, dq_dtype):
    return x.astype(dq_dtype) * scale_inv.astype(dq_dtype)


# Apply jit to guarantee correctness of FP8 GEMM.
@partial(
    jax.jit,
    static_argnums=(
        2,
        3,
        4,
    ),
)
def __jitted_jax_gemm_delayed_scaling_fp8(lhs, rhs, lhs_dn, rhs_dn, precision):
    # Need to hard-code the dequantize here instead of calling lhs.dequantize() for pattern matching
    lhs_dq = _dequantize(lhs.data, lhs.scale_inv, lhs.dq_dtype)
    rhs_dq = _dequantize(rhs.data, rhs.scale_inv, rhs.dq_dtype)

    # Reshape + Transpose
    # [..., M, K] -> [B, M, K]
    # [..., K, M] -> [B, M, K]
    lhs_3d = _shape_normalization(lhs_dq, lhs_dn, lhs.data_layout == "N")
    rhs_3d = _shape_normalization(rhs_dq, rhs_dn, rhs.data_layout == "T")

    dim_nums = (((2,), (2,)), ((0,), (0,)))
    out_3d = jax.lax.dot_general(
        lhs_3d, rhs_3d, dim_nums, precision=precision, preferred_element_type=lhs.dq_dtype
    )
    return out_3d


def _jax_gemm_delayed_scaling_fp8(
    lhs: ScaledTensor, rhs: ScaledTensor, dim_nums: Tuple[Tuple[Sequence[int], Sequence[int]]]
):
    """FP8 GEMM for XLA pattern match"""
    assert (
        rhs.scaling_mode == ScalingMode.NVTE_DELAYED_TENSOR_SCALING
    ), "rhs does not have delayed tensor scaling mode"

    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dim_nums
    if lhs.data_layout == "T":
        lhs_contract = tuple((lhs.data.ndim - 1 - i) % lhs.data.ndim for i in lhs_contract)
    if rhs.data_layout == "T":
        rhs_contract = tuple((rhs.data.ndim - 1 - i) % rhs.data.ndim for i in rhs_contract)

    lhs_dn = (lhs_contract, lhs_batch)
    rhs_dn = (rhs_contract, rhs_batch)

    lhs_remain_shape = _calculate_remaining_shape(lhs.data.shape, lhs_contract)
    rhs_remain_shape = _calculate_remaining_shape(rhs.data.shape, rhs_contract)

    precision = (
        jax.lax.Precision.HIGHEST if QuantizeConfig.FP8_2X_ACC_FPROP else jax.lax.Precision.DEFAULT
    )
    out_3d = __jitted_jax_gemm_delayed_scaling_fp8(lhs, rhs, lhs_dn, rhs_dn, precision)

    # Reshape [B, M, N] -> [..., M, N]
    out = out_3d.reshape(*lhs_remain_shape, *rhs_remain_shape)
    return out


def _jax_gemm_mxfp8_1d(
    lhs: ScaledTensor, rhs: ScaledTensor, dim_nums: Tuple[Tuple[Sequence[int], Sequence[int]]]
):
    """
    JAX GEMM for MXFP8 via scaled_matmul
    """
    assert (
        rhs.scaling_mode == ScalingMode.NVTE_MXFP8_1D_SCALING
    ), "rhs does not have MXFP8 1D scaling mode"
    from jax._src.cudnn.scaled_matmul_stablehlo import scaled_matmul_wrapper

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
    out_3d = scaled_matmul_wrapper(
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

        if lhs.scaling_mode == ScalingMode.NVTE_DELAYED_TENSOR_SCALING:
            return _jax_gemm_delayed_scaling_fp8(lhs, rhs, dim_nums)

        if lhs.scaling_mode == ScalingMode.NVTE_MXFP8_1D_SCALING:
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


def swizzled_scale(scales):
    """Swizzle the scale tensor for FP8 GEMM"""
    assert scales.ndim == 2
    rows, cols = scales.shape
    scales = scales.reshape(rows // 128, 4, 32, cols // 4, 4)
    scales = jnp.transpose(scales, (0, 3, 2, 1, 4))
    return scales


def grouped_gemm(
    lhs_list: List[Union[jnp.ndarray, ScaledTensor]],
    rhs_list: List[Union[jnp.ndarray, ScaledTensor]],
    contracting_dims_list: List[Tuple[Sequence[int], Sequence[int]]],
    bias_list: List[jnp.ndarray] = None,
) -> List[jnp.ndarray]:
    """Grouped GEMM for multiple pairs of tensors."""
    assert (
        len(lhs_list) == len(rhs_list) == len(contracting_dims_list)
    ), "lhs_list, rhs_list, contracting_dims_list must have the same length"

    # Flatten inputs and save their shapes
    num_gemms = len(lhs_list)
    out_flat_size = 0
    dims = []
    lhs_contig_ = []
    rhs_contig_ = []
    lhs_scale_inv_contig_ = []
    rhs_scale_inv_contig_ = []
    bias_contig_ = []
    out_offsets = []
    remain_shape_list = []
    num_gemms = len(lhs_list)
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
            # For ScaledTensors and NVTE_DELAYED_TENSOR_SCALING, need to handle internal data_layout
            if lhs.scaling_mode == ScalingMode.NVTE_DELAYED_TENSOR_SCALING:
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
            scaling_mode = ScalingMode.NVTE_NO_SCALING
            lhs_shape = lhs.shape
            rhs_shape = rhs.shape
            out_dtype = lhs.dtype

        (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dim_nums
        lhs_dn = (lhs_contract, lhs_batch)
        rhs_dn = (rhs_contract, rhs_batch)

        lhs_remain_shape = _calculate_remaining_shape(lhs_shape, lhs_contract)
        rhs_remain_shape = _calculate_remaining_shape(rhs_shape, rhs_contract)

        if scaling_mode == ScalingMode.NVTE_NO_SCALING:
            lhs_3d = _shape_normalization(lhs, lhs_dn)
            rhs_3d = _shape_normalization(rhs, rhs_dn)
        elif scaling_mode == ScalingMode.NVTE_DELAYED_TENSOR_SCALING:
            lhs_3d = _shape_normalization(lhs.data, lhs_dn, lhs.data_layout == "N")
            rhs_3d = _shape_normalization(rhs.data, rhs_dn, rhs.data_layout == "T")
        elif scaling_mode == ScalingMode.NVTE_MXFP8_1D_SCALING:
            lhs_3d = _shape_normalization(lhs.data, lhs_dn)
            rhs_3d = _shape_normalization(rhs.data, rhs_dn)
            lhs_scale_inv = _shape_normalization(lhs.scale_inv, lhs_dn)
            rhs_scale_inv = _shape_normalization(rhs.scale_inv, rhs_dn)
            lhs_scale_inv = swizzled_scale(lhs_scale_inv.squeeze())
            rhs_scale_inv = swizzled_scale(rhs_scale_inv.squeeze())
        else:
            raise NotImplementedError("Unsupported ScalingMode: {scaling_mode}")

        # Note: if _shape_normalization() is updated to support non-TN, need to update here
        # already_transposed doesn't matter for the output shape
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
        remain_shape_list.append(((bm,), (bn,)))
        assert kl == kr, f"lhs_3d.shape[-1] ({kl}) != rhs_3d.shape[-1] ({kr})"
        k = kl

        if (bm % 16 != 0) or (bn % 16 != 0) or (k % 16 != 0):
            print(f"grouped_gemm input pair {i} has invalid problem shape for lowering: ")
            print(
                f"m = {bm}, n = {bn}, k = {k}; cuBLAS requires the problem shapes being multiples"
                " of 16"
            )
            assert bm % 16 == 0 and bn % 16 == 0 and k % 16 == 0

        dims.append((bm, bn, k))
        lhs_contig_.append(lhs_3d.reshape(-1))
        rhs_contig_.append(rhs_3d.reshape(-1))
        if scaling_mode == ScalingMode.NVTE_NO_SCALING:
            lhs_scale_inv_contig_.append(jnp.ones(1, dtype=jnp.float32))
            rhs_scale_inv_contig_.append(jnp.ones(1, dtype=jnp.float32))
        if scaling_mode == ScalingMode.NVTE_DELAYED_TENSOR_SCALING:
            lhs_scale_inv_contig_.append(lhs.scale_inv.reshape(-1))
            rhs_scale_inv_contig_.append(rhs.scale_inv.reshape(-1))
        if scaling_mode == ScalingMode.NVTE_MXFP8_1D_SCALING:
            lhs_scale_inv_contig_.append(lhs_scale_inv.reshape(-1))
            rhs_scale_inv_contig_.append(rhs_scale_inv.reshape(-1))
        if bias_list is not None:
            bias_contig_.append(bias_list[i].reshape(-1))
        out_flat_size += bm * bn
        out_offsets.append(out_flat_size)

    lhs_contig = jnp.concatenate(lhs_contig_)
    rhs_contig = jnp.concatenate(rhs_contig_)
    lhs_scale_inv_contig = jnp.concatenate(lhs_scale_inv_contig_)
    rhs_scale_inv_contig = jnp.concatenate(rhs_scale_inv_contig_)
    bias_contig = jnp.empty(0) if bias_list is None else jnp.concatenate(bias_contig_)
    dim_list = jnp.array(dims, dtype=jnp.int32)

    # TE/common does not support NVTE_NO_SCALING yet
    # It expects NVTE_DELAYED_TENSOR_SCALING as default for FP32, BF16, FP16
    if scaling_mode == ScalingMode.NVTE_NO_SCALING:
        scaling_mode = ScalingMode.NVTE_DELAYED_TENSOR_SCALING

    # Perform batched GEMM on flattened inputs
    out_contig = GroupedGemmPrimitive.outer_primitive.bind(
        lhs_contig,
        lhs_scale_inv_contig,
        rhs_contig,
        rhs_scale_inv_contig,
        bias_contig,
        dim_list,
        num_gemms=num_gemms,
        scaling_mode=scaling_mode,
        out_dtype=out_dtype,
        out_flat_size=out_flat_size,
    )

    # Split the output back into tensors
    out_offsets = jnp.array(out_offsets)
    out_flat_list = jnp.split(out_contig, out_offsets[:-1])
    out_tensors = []
    for out_flat, (lhs_remain_shape, rhs_remain_shape) in zip(out_flat_list, remain_shape_list):
        out_tensors.append(out_flat.reshape(*lhs_remain_shape, *rhs_remain_shape))

    return out_tensors
