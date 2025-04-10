# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX te modules"""

import warnings
import operator
from functools import partial, reduce
from typing import Optional, Tuple, Sequence, Union, Dict, List
from packaging import version

from transformer_engine_jax import get_device_compute_capability
import jax
import jax.numpy as jnp
from jax import dtypes
from jax.sharding import PartitionSpec, NamedSharding
from jax.typing import ArrayLike

from .base import BasePrimitive, register_primitive

from .misc import (
    jax_dtype_is_fp8,
    get_padded_spec,
    check_valid_batch_dims,
)
from ..sharding import (
    global_mesh_resource,
    all_reduce_max_along_all_axes_except_PP,
)

from ..quantize import (
    ScaledTensor,
    ScalingMode,
    Quantizer,
    QuantizeConfig,
    noop_quantizer_set,
)

if version.parse(jax.__version__) >= version.parse("0.5.0"):
    from jax import ffi  # pylint: disable=ungrouped-imports
else:
    from jax.extend import ffi  # pylint: disable=ungrouped-imports


__all__ = ["gemm",
           "grouped_gemm",
           "collective_fp8_gemm_impl",
           "collective_gemm_impl"]


num_cublas_streams = 4

def sanitize_dims(dim, ndims):
    return (ndims + dim) if dim < 0 else dim


def mirror_dim(dim, ndims):
    return ndims - 2 if dim == ndims - 1 else ndims - 1


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
    lhs_3d = _shape_normalization(lhs_dq, lhs_dn, lhs.layout == "N")
    rhs_3d = _shape_normalization(rhs_dq, rhs_dn, rhs.layout == "T")

    # _shape_normalization ensures contracting_dims=2 and batch_dims=0
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
    if lhs.layout == "T":
        lhs_contract = tuple((lhs.data.ndim - 1 - i) % lhs.data.ndim for i in lhs_contract)
    if rhs.layout == "T":
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
            # For ScaledTensors and NVTE_DELAYED_TENSOR_SCALING, need to handle internal layout
            if lhs.scaling_mode == ScalingMode.NVTE_DELAYED_TENSOR_SCALING:
                assert not (
                    lhs.data.dtype == jnp.float8_e5m2 and rhs.data.dtype == jnp.float8_e5m2
                ), "FP8 GEMM does not support E5M2 * E5M2"
                ((lhs_contract_dim,), (rhs_contract_dim,)) = contracting_dims
                if lhs.layout == "T":
                    lhs_contract_dim = (lhs_contract_dim - 1) % lhs.data.ndim
                if rhs.layout == "T":
                    rhs_contract_dim = (rhs_contract_dim - 1) % rhs.data.ndim
                dim_nums = ((lhs_contract_dim,), (rhs_contract_dim,)), ((), ())
        else:
            # For jnp.ndarray, only consider contracting_dims, layout is always NN
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
            lhs_3d = _shape_normalization(lhs.data, lhs_dn, lhs.layout == "N")
            rhs_3d = _shape_normalization(rhs.data, rhs_dn, rhs.layout == "T")
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
class CollectiveGemmPrimitive(BasePrimitive):
    """
    cuBlasLt GEMM Primitive w/ support for distributed inputs
    """

    name = "te_gemm"
    impl_static_args = (8, 9, 10, 11, 12, 13, 14, 15)
    multiple_results = True
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        lhs_aval,
        lhs_scale_inv_aval,
        rhs_aval,
        rhs_scale_inv_aval,
        bias_aval,
        gelu_input_aval,
        out_amax_aval,
        out_scale_aval,
        out_dtype,
        batched_output,
        contracting_dims,
        fuse_gelu,
        fuse_bias,
        grad,
        accumulate,
        use_split_accumulator,
    ):
        """
        cuBlasLt GEMM abstract
        """
        del grad, accumulate, use_split_accumulator

        # Validate operand dtypes
        lhs_dtype = dtypes.canonicalize_dtype(lhs_aval.dtype)
        rhs_dtype = dtypes.canonicalize_dtype(rhs_aval.dtype)
        assert lhs_dtype == rhs_dtype, "Mismatched matrix dtypes for GEMM."
        is_fp8 = False
        if jax_dtype_is_fp8(lhs_dtype):
            assert (
                lhs_scale_inv_aval.size == 1
                and dtypes.canonicalize_dtype(lhs_scale_inv_aval.dtype) == jnp.float32
            ), "Missing LHS operand scale inverse in FP8 GEMM."
            is_fp8 = True
        if jax_dtype_is_fp8(rhs_dtype):
            assert (
                rhs_scale_inv_aval.size == 1
                and dtypes.canonicalize_dtype(rhs_scale_inv_aval.dtype) == jnp.float32
            ), "Missing RHS operand scale inverse in FP8 GEMM."

        # Validate operand layouts
        lhs_inner_dim, rhs_inner_dim = map(
            sanitize_dims, contracting_dims, (lhs_aval.ndim, rhs_aval.ndim)
        )
        assert (
            lhs_aval.shape[lhs_inner_dim] == rhs_aval.shape[rhs_inner_dim]
        ), f"Incompatible operand sizes: {lhs_aval.shape} x {rhs_aval.shape}."

        lhs_trans = lhs_inner_dim != lhs_aval.ndim - 1
        rhs_trans = rhs_inner_dim == rhs_aval.ndim - 1
        assert not (
            lhs_trans and rhs_trans
        ), "GEMM does not support transposed LHS and transposed RHS at the same time."
        if is_fp8:
            assert not lhs_trans, "FP8 GEMM does not support transposed LHS."
            assert rhs_trans, "FP8 GEMM requires transposed RHS."

        # Validate output dtype
        if jax_dtype_is_fp8(out_dtype):
            assert jax_dtype_is_fp8(lhs_dtype) and jax_dtype_is_fp8(
                rhs_dtype
            ), "FP8 GEMM output requires FP8 inputs."
            assert (
                out_amax_aval.size == out_scale_aval.size == 1
            ), "Invalid/missing output amax and scale."
            out_amax_updated_dtype = dtypes.canonicalize_dtype(out_amax_aval.dtype)
            out_scale_updated_dtype = dtypes.canonicalize_dtype(out_scale_aval.dtype)
            assert (
                out_amax_updated_dtype == out_scale_updated_dtype == jnp.float32
            ), "Invalid output amax or scale dtype."
        else:
            out_dtype = lhs_dtype
            out_amax_updated_dtype = jnp.float32
            out_scale_updated_dtype = jnp.float32

        # Make sure leading dimensions of RHS is broadcast-compatible with LHS
        lhs_outer_dim, rhs_outer_dim = map(
            mirror_dim,
            (lhs_inner_dim, rhs_inner_dim),
            (lhs_aval.ndim, rhs_aval.ndim),
        )
        lhs_bdims = [
            dim for dim in range(lhs_aval.ndim) if dim not in [lhs_outer_dim, lhs_inner_dim]
        ]
        lhs_batch_shape = [lhs_aval.shape[dim] for dim in lhs_bdims]
        lhs_batch_size = reduce(operator.mul, lhs_batch_shape, 1)

        # Infer output shape
        if batched_output:
            assert (
                lhs_aval.ndim > 2 and rhs_aval.ndim == 2
            ), "Batched output requires batched LHS and non-batched RHS operands."
            out_shape = (
                *lhs_batch_shape,
                lhs_aval.shape[lhs_outer_dim],
                rhs_aval.shape[rhs_outer_dim],
            )
        else:
            assert (
                lhs_aval.ndim == rhs_aval.ndim
            ), "Non-batched output requires LHS and RHS operands with same number of dimensions."
            if lhs_aval.ndim > 2:
                rhs_bdims = [
                    dim for dim in range(rhs_aval.ndim) if dim not in [rhs_outer_dim, rhs_inner_dim]
                ]
                rhs_batch_shape = [rhs_aval.shape[dim] for dim in rhs_bdims]
                rhs_batch_size = reduce(operator.mul, rhs_batch_shape, 1)
                assert lhs_batch_size == rhs_batch_size, (
                    f"Leading dimensins of RHS ({rhs_aval.shape=}) is not broadcast-compatible "
                    + f"with the leading dimensions of LHS ({lhs_aval.shape=})."
                )
            out_shape = (lhs_aval.shape[lhs_outer_dim], rhs_aval.shape[rhs_outer_dim])

        # Validate bias/bias_grad shape against inferred output
        bias_dtype = jnp.bfloat16 if jax_dtype_is_fp8(out_dtype) else out_dtype
        if fuse_bias:
            assert (
                bias_aval.size > 0 and bias_aval.ndim == 1 and bias_aval.shape[0] == out_shape[-1]
            ), "Incorrect bias shape."
            bias_dtype = dtypes.canonicalize_dtype(bias_aval.dtype)
        else:
            assert bias_aval.size == 0, "Internal TE error."

        # Validate GELU input/output
        gelu_shape = (0,)
        if fuse_gelu:
            gelu_shape = (
                (reduce(operator.mul, out_shape[:-1], 1), out_shape[-1])
                if len(out_shape) > 2
                else out_shape
            )
            assert gelu_input_aval.ndim == 2 and all(
                gelu_input_aval.shape[i] == gelu_shape[i] for i in range(len(gelu_shape))
            ), "Invalid GELU input shape."
            assert gelu_input_aval.dtype == bias_dtype, "Invalid GELU dtype."
        else:
            assert gelu_input_aval.size == 0, "Internal TE error."

        # Create abstract arrays for all outputs
        out_aval = lhs_aval.update(shape=out_shape, dtype=out_dtype)
        out_amax_updated_aval = out_amax_aval.update(
            shape=out_amax_aval.shape, dtype=out_amax_updated_dtype
        )
        out_scale_updated_aval = out_scale_aval.update(
            shape=out_scale_aval.shape, dtype=out_scale_updated_dtype
        )
        pre_gelu_out_aval = gelu_input_aval.update(shape=gelu_shape, dtype=bias_dtype)
        bias_grad_aval = bias_aval.update(shape=bias_aval.shape, dtype=bias_dtype)
        workspace_aval = jax.core.ShapedArray(
            shape=(get_cublas_workspace_size_bytes(),), dtype=jnp.uint8
        )

        return (
            out_aval,
            out_amax_updated_aval,
            out_scale_updated_aval,
            pre_gelu_out_aval,
            bias_grad_aval,
            workspace_aval,
        )

    @staticmethod
    def outer_abstract(*args, **kwargs):
        """
        cuBlasLt GEMM outer abstract
        """
        (out_aval, out_amax_aval, out_scale_aval, pre_gelu_out_aval, bias_grad_aval, _) = (
            CollectiveGemmPrimitive.abstract(*args, **kwargs)
        )
        return out_aval, out_amax_aval, out_scale_aval, pre_gelu_out_aval, bias_grad_aval
    @staticmethod
    def lowering(
        ctx,
        lhs,
        lhs_scale_inv,
        rhs,
        rhs_scale_inv,
        bias,
        gelu_input,
        out_amax,
        out_scale,
        *,
        out_dtype,
        batched_output,
        contracting_dims,
        fuse_gelu,
        fuse_bias,
        grad,
        accumulate,
        use_split_accumulator,
    ):
        """
        Fused attention fwd lowering rules
        """
        del batched_output
        lhs_aval, _, rhs_aval, _, _, *_ = ctx.avals_in
        lhs_inner_dim, rhs_inner_dim = map(
            sanitize_dims, contracting_dims, (lhs_aval.ndim, rhs_aval.ndim)
        )
        lhs_trans = lhs_inner_dim != lhs_aval.ndim - 1
        rhs_trans = rhs_inner_dim == rhs_aval.ndim - 1

        operand_output_aliases = {
            4: 4,  # bias        <-->  bias_grad
            5: 3,  # gelu_input  <-->  pre_gelu_out
            6: 1,  # out_amax    <-->  out_amax_updated
            7: 2,  # out_scale   <-->  out_scale_updated
        }

        name = "te_gemm_ffi"
        return ffi.ffi_lowering(name, operand_output_aliases=operand_output_aliases)(
            ctx,
            lhs,
            lhs_scale_inv,
            rhs,
            rhs_scale_inv,
            bias,
            gelu_input,
            out_amax,
            out_scale,
            lhs_trans=lhs_trans,
            rhs_trans=rhs_trans,
            fuse_gelu=fuse_gelu,
            fuse_bias=fuse_bias,
            grad=grad,
            accumulate=accumulate,
            use_split_accumulator=use_split_accumulator,
        )


    @staticmethod
    def impl(
        lhs,
        lhs_scale_inv,
        rhs,
        rhs_scale_inv,
        bias,
        gelu_input,
        out_amax,
        out_scale,
        out_dtype,
        batched_output,
        contracting_dims,
        fuse_gelu,
        fuse_bias,
        grad,
        accumulate,
        use_split_accumulator,
    ):
        assert CollectiveGemmPrimitive.inner_primitive is not None

        lhs_inner_dim, rhs_inner_dim = map(sanitize_dims, contracting_dims, (lhs.ndim, rhs.ndim))
        lhs_outer_dim, rhs_outer_dim = map(
            mirror_dim, (lhs_inner_dim, rhs_inner_dim), (lhs.ndim, rhs.ndim)
        )

        # Infer output shape and collapse batch dimensions
        lhs_2d_shape = rhs_2d_shape = None
        lhs_layout = rhs_layout = None
        lhs_batch_dims = [
            dim for dim in range(lhs.ndim) if dim not in [lhs_inner_dim, lhs_outer_dim]
        ]
        lhs_batch_shape = [lhs.shape[dim] for dim in lhs_batch_dims]
        lhs_batch_size = reduce(operator.mul, lhs_batch_shape, 1)
        contracting_dims_2d = list(contracting_dims).copy()
        if lhs.ndim > 2 and rhs.ndim > 2:
            # If both LHS and RHS are batched, the batch dimensions collapse into the
            # contracting dimensions for both operands
            lhs_2d_shape = (lhs_batch_size * lhs.shape[lhs_inner_dim], lhs.shape[lhs_outer_dim])
            lhs_layout = (*lhs_batch_dims, lhs_inner_dim, lhs_outer_dim)
            contracting_dims_2d[0] = 0

            rhs_batch_dims = [
                dim for dim in range(rhs.ndim) if dim not in [rhs_inner_dim, rhs_outer_dim]
            ]
            rhs_batch_shape = [rhs.shape[dim] for dim in rhs_batch_dims]
            rhs_batch_size = reduce(operator.mul, rhs_batch_shape, 1)
            rhs_2d_shape = (rhs_batch_size * rhs.shape[rhs_inner_dim], rhs.shape[rhs_outer_dim])
            rhs_layout = (*rhs_batch_dims, rhs_inner_dim, rhs_outer_dim)
            contracting_dims_2d[1] = 0
        elif lhs.ndim > 2:
            # If only the LHS is batched,the batch dimension collapses into the outer dimension
            lhs_2d_shape = (lhs_batch_size * lhs.shape[lhs_outer_dim], lhs.shape[lhs_inner_dim])
            lhs_layout = (*lhs_batch_dims, lhs_outer_dim, lhs_inner_dim)
            contracting_dims_2d[0] = 1

        # Reshape LHS and RHS into 2D and fix layouts for FP8 GEMM
        if lhs_2d_shape is not None and lhs.ndim > 2:
            lhs = jax.lax.reshape(lhs, lhs_2d_shape, dimensions=lhs_layout)
            if jax_dtype_is_fp8(lhs.dtype):
                lhs = jax.lax.transpose(lhs, (1, 0))
                contracting_dims_2d[0] = 1
        else:
            contracting_dims_2d[0] = contracting_dims[0]

        if rhs_2d_shape is not None and rhs.ndim > 2:
            rhs = jax.lax.reshape(rhs, rhs_2d_shape, dimensions=rhs_layout)
            if jax_dtype_is_fp8(rhs.dtype):
                rhs = jax.lax.transpose(rhs, (1, 0))
                contracting_dims_2d[1] = 1
        else:
            contracting_dims_2d[1] = contracting_dims[1]

        # Invoke GEMM with guaranteed 2D inputs, so batched_output=False
        (
            out,
            out_amax_updated,
            out_scale_updated,
            pre_gelu_out,
            bias_grad,
            _,
        ) = CollectiveGemmPrimitive.inner_primitive.bind(
            lhs,
            lhs_scale_inv,
            rhs,
            rhs_scale_inv,
            bias,
            gelu_input,
            out_amax,
            out_scale,
            out_dtype=out_dtype,
            batched_output=False,
            contracting_dims=contracting_dims_2d,
            fuse_gelu=fuse_gelu,
            fuse_bias=fuse_bias,
            grad=grad,
            accumulate=accumulate,
            use_split_accumulator=use_split_accumulator,
        )

        # Recover batched dimensions in the output
        if batched_output:
            out_shape = (*lhs_batch_shape, out.shape[-2] // lhs_batch_size, out.shape[-1])
            out = jax.lax.reshape(out, out_shape)

        return out, out_amax_updated, out_scale_updated, pre_gelu_out, bias_grad

    @staticmethod
    def batcher(
        batched_args,
        batch_dims,
        *,
        out_dtype,
        batched_output,
        contracting_dims,
        fuse_gelu,
        fuse_bias,
        grad,
        accumulate,
        use_split_accumulator,
    ):
        assert CollectiveGemmPrimitive.outer_primitive is not None
        check_valid_batch_dims(batch_dims)
        lhs_bdims, *_, bias_bdims, gelu_input_bdims, out_amax_bdims, out_scale_bdims = batch_dims

        return (
            CollectiveGemmPrimitive.outer_primitive.bind(
                *batched_args,
                out_dtype=out_dtype,
                batched_output=batched_output,
                contracting_dims=contracting_dims,
                fuse_gelu=fuse_gelu,
                fuse_bias=fuse_bias,
                grad=grad,
                accumulate=accumulate,
                use_split_accumulator=use_split_accumulator,
            ),
            (lhs_bdims, out_amax_bdims, out_scale_bdims, gelu_input_bdims, bias_bdims),
        )

    @staticmethod
    def infer_sharding_from_operands(
        out_dtype,
        batched_output,
        contracting_dims,
        fuse_gelu,
        fuse_bias,
        grad,
        accumulate,
        use_split_accumulator,
        mesh,
        arg_infos,
        result_infos,
    ):
        del out_dtype, accumulate, use_split_accumulator, result_infos
        lhs, _, rhs, *_ = arg_infos
        lhs_spec, rhs_spec = map(get_padded_spec, [lhs, rhs])

        lhs_inner_dim, rhs_inner_dim = map(sanitize_dims, contracting_dims, (lhs.ndim, rhs.ndim))
        lhs_outer_dim, rhs_outer_dim = map(
            mirror_dim,
            (lhs_inner_dim, rhs_inner_dim),
            (lhs.ndim, rhs.ndim),
        )

        # Modify operand specs:
        # - If contracting dimensions of both operands are sharded, force them to match.
        # - If contracting dimensions of both operands are sharded, all-gather outer dimensions.
        # - If contracting dimension of only one operand is sharded, all-gather the sharded
        #   operand.
        # - Never scatter any operand.
        lhs_spec_new = list(lhs_spec).copy()
        rhs_spec_new = list(rhs_spec).copy()
        lhs_spec_new[lhs_outer_dim] = None
        if lhs_spec_new[lhs_inner_dim] is not None and rhs_spec_new[rhs_inner_dim] is not None:
            assert (
                lhs_spec_new[lhs_inner_dim] == rhs_spec_new[rhs_inner_dim]
            ), "Contracting dimensions of LHS and RHS operands must have the same sharding."
            if lhs_spec_new[lhs_outer_dim] is not None:
                warnings.warn(
                    "Outer dimension of the LHS operand must be all-gathered when both contracting "
                    + "dimensions are sharded. This will cause additional communication overhead."
                )

            if rhs_spec_new[rhs_outer_dim] is not None:
                warnings.warn(
                    "Outer dimension of the RHS operand must be all-gathered when both contracting "
                    + "dimensions are sharded. This will cause additional communication overhead."
                )
            rhs_spec_new[rhs_outer_dim] = None
        else:
            if lhs_spec_new[lhs_inner_dim] is None and rhs_spec_new[rhs_inner_dim] is not None:
                warnings.warn(
                    "Contracting dimension of the RHS operand must be all-gathered when the "
                    + "contracting dimension of the LHS operand is unsharded. This will cause "
                    + "additional communication overhead."
                )
            if lhs_spec_new[lhs_inner_dim] is not None and rhs_spec_new[rhs_inner_dim] is None:
                if not grad:
                    # This is expected for sequence/context-parallel gradient in BWD (DGRAD) GEMM.
                    warnings.warn(
                        "Contracting dimension of the LHS operand must be all-gathered when the "
                        + "contracting dimension of the RHS operand is unsharded. This will cause "
                        + "additional communication overhead."
                    )
            lhs_spec_new[lhs_inner_dim] = None
            rhs_spec_new[rhs_inner_dim] = None
        out_col_spec = rhs_spec_new[rhs_outer_dim]

        # Output sharding is conditional on output shape
        lhs_bdims = [dim for dim in range(lhs.ndim) if dim not in [lhs_inner_dim, lhs_outer_dim]]
        batch_spec = [lhs_spec_new[dim] for dim in lhs_bdims]
        out_spec = [None, out_col_spec]
        if batched_output:
            out_spec = batch_spec + out_spec
        out_sharding = NamedSharding(mesh, PartitionSpec(*out_spec))

        # FP8 metas are always unsharded
        fp8_meta_sharding = NamedSharding(mesh, PartitionSpec(None))

        # Pre-GELU output is always 2D if GELU fusion is turned on, otherwise unsharded
        gelu_spec = [None, out_col_spec] if fuse_gelu else [None]
        gelu_sharding = NamedSharding(mesh, PartitionSpec(*gelu_spec))

        # Bias gradient spec matches outer dimension of output if bias fusion is turned on
        bias_sharding = NamedSharding(mesh, PartitionSpec(out_col_spec if fuse_bias else None))

        return (out_sharding, fp8_meta_sharding, fp8_meta_sharding, gelu_sharding, bias_sharding)

    @staticmethod
    def partition(
        out_dtype,
        batched_output,
        contracting_dims,
        fuse_gelu,
        fuse_bias,
        grad,
        accumulate,
        use_split_accumulator,
        mesh,
        arg_infos,
        result_infos,
    ):
        del result_infos
        lhs, _, rhs, *_ = arg_infos
        lhs_spec, rhs_spec = map(get_padded_spec, [lhs, rhs])

        lhs_inner_dim, rhs_inner_dim = map(sanitize_dims, contracting_dims, (lhs.ndim, rhs.ndim))
        lhs_outer_dim, rhs_outer_dim = map(
            mirror_dim,
            (lhs_inner_dim, rhs_inner_dim),
            (lhs.ndim, rhs.ndim),
        )

        # Modify operand specs:
        # - Always all-gather the outer dimension of LHS.
        # - If contracting dimensions of both operands are sharded, all-gather RHS outer dimension.
        # - If contracting dimension of only one operand is sharded, all-gather the sharded
        #   operand.
        # - Never scatter any operand.
        lhs_spec_new = list(lhs_spec).copy()
        rhs_spec_new = list(rhs_spec).copy()
        reduce_output = False
        lhs_spec_new[lhs_outer_dim] = None
        if lhs_spec_new[lhs_inner_dim] is not None and rhs_spec_new[rhs_inner_dim] is not None:
            rhs_spec_new[rhs_outer_dim] = None
            reduce_output = True
        else:
            lhs_spec_new[lhs_inner_dim] = None
            rhs_spec_new[rhs_inner_dim] = None
        out_col_spec = rhs_spec_new[rhs_outer_dim]
        lhs_sharding = NamedSharding(mesh, PartitionSpec(*lhs_spec_new))
        rhs_sharding = NamedSharding(mesh, PartitionSpec(*rhs_spec_new))

        # Bias is sharded to match outer dimension spec of the RHS operand (also the output)
        bias_sharding = NamedSharding(mesh, PartitionSpec(out_col_spec if fuse_bias else None))

        # FP8 metas are always unsharded
        fp8_meta_sharding = NamedSharding(mesh, PartitionSpec(None))

        # Output sharding is conditional on output shape
        lhs_bdims = [dim for dim in range(lhs.ndim) if dim not in [lhs_inner_dim, lhs_outer_dim]]
        batch_spec = [lhs_spec_new[dim] for dim in lhs_bdims]
        out_spec = [None, out_col_spec]
        if batched_output:
            out_spec = batch_spec + out_spec
        out_sharding = NamedSharding(mesh, PartitionSpec(*out_spec))

        # Pre-GELU output is always 2D if GELU fusion is turned on, otherwise unsharded
        gelu_spec = [None, out_col_spec] if fuse_gelu else [None]
        gelu_sharding = NamedSharding(mesh, PartitionSpec(*gelu_spec))

        arg_shardings = (
            lhs_sharding,
            fp8_meta_sharding,
            rhs_sharding,
            fp8_meta_sharding,
            bias_sharding,
            gelu_sharding,
            fp8_meta_sharding,
            fp8_meta_sharding,
        )
        out_shardings = (
            out_sharding,
            fp8_meta_sharding,
            fp8_meta_sharding,
            gelu_sharding,
            bias_sharding,
        )

        def sharded_impl(
            lhs, lhs_scale_inv, rhs, rhs_scale_inv, bias, gelu_input, out_amax, out_scale
        ):
            (
                out,
                out_amax_updated,
                out_scale_updated,
                pre_gelu_out,
                bias_grad,
            ) = CollectiveGemmPrimitive.impl(
                lhs,
                lhs_scale_inv,
                rhs,
                rhs_scale_inv,
                bias,
                gelu_input,
                out_amax,
                out_scale,
                out_dtype=out_dtype,
                batched_output=batched_output,
                contracting_dims=contracting_dims,
                fuse_gelu=fuse_gelu,
                fuse_bias=fuse_bias,
                grad=grad,
                accumulate=accumulate,
                use_split_accumulator=use_split_accumulator,
            )

            # FP8 amax reduction
            if jax_dtype_is_fp8(lhs.dtype):
                out_amax_updated = all_reduce_max_along_all_axes_except_PP(out_amax_updated, mesh)

            # All-reduce sum GEMM output when contracting dimensions are sharded
            if reduce_output:
                out = jax.lax.psum(out, global_mesh_resource().tp_resource)
                if fuse_gelu:
                    pre_gelu_out = jax.lax.psum(pre_gelu_out, global_mesh_resource().tp_resource)

            return out, out_amax_updated, out_scale_updated, pre_gelu_out, bias_grad

        return mesh, sharded_impl, out_shardings, arg_shardings


register_primitive(CollectiveGemmPrimitive)


def collective_fp8_gemm_impl(
    lhs: ArrayLike,
    lhs_scale_inv: ArrayLike,
    rhs_t: ArrayLike,
    rhs_scale_inv: ArrayLike,
    bias: Optional[ArrayLike] = None,
    gelu_input: Optional[ArrayLike] = None,
    out_amax: Optional[ArrayLike] = None,
    out_scale: Optional[ArrayLike] = None,
    out_dtype: jnp.dtype = jnp.bfloat16,
    batched_output: bool = False,
    fuse_gelu: bool = False,
    fuse_bias: bool = False,
    accumulate: bool = False,
    use_split_accumulator: bool = False,
) -> Tuple[ArrayLike, ...]:
    """FP8 mat-mul with `nvte_cublas_gemm()` custom op."""
    if out_dtype is not None and jax_dtype_is_fp8(out_dtype):
        assert out_amax is not None and out_scale is not None, "Missing output amax and scale."
    else:
        out_amax = jnp.zeros(0, dtype=jnp.float32)
        out_scale = jnp.zeros(0, dtype=jnp.float32)

    if not fuse_bias:
        bias = jnp.zeros(0, dtype=jnp.bfloat16)
    else:
        assert bias is not None, "Missing bias in forward GEMM when bias epilogue is enabled."

    if not fuse_gelu:
        gelu_input = jnp.zeros(0, dtype=bias.dtype)
    elif gelu_input is None:
        gelu_shape = (reduce(operator.mul, lhs.shape[:-1]), rhs_t.shape[-1])
        gelu_input = jnp.zeros(gelu_shape, dtype=bias.dtype)

    out, out_amax, out_scale, pre_gelu_out, _ = CollectiveGemmPrimitive.outer_primitive.bind(
        lhs,
        lhs_scale_inv,
        rhs_t,
        rhs_scale_inv,
        bias,
        gelu_input,
        out_amax,
        out_scale,
        out_dtype=out_dtype,
        batched_output=batched_output,
        contracting_dims=(-1, -1),
        fuse_gelu=fuse_gelu,
        fuse_bias=fuse_bias,
        grad=False,
        accumulate=accumulate,
        use_split_accumulator=use_split_accumulator,
    )

    return out, out_amax, out_scale, pre_gelu_out


def collective_gemm_impl(
    lhs: ArrayLike,
    rhs: ArrayLike,
    bias: Optional[ArrayLike] = None,
    gelu_input: Optional[ArrayLike] = None,
    batched_output: bool = False,
    contracting_dims: Tuple[int, int] = (-1, -2),
    fuse_gelu: bool = False,
    fuse_bias: bool = False,
    grad: bool = False,
    accumulate: bool = False,
    use_split_accumulator: bool = False,
) -> Tuple[ArrayLike, ...]:
    """Non-FP8 mat-mul with `nvte_cublas_gemm()` custom op."""
    lhs_inner_dim, rhs_inner_dim = map(sanitize_dims, contracting_dims, (lhs.ndim, rhs.ndim))
    lhs_outer_dim, rhs_outer_dim = map(
        mirror_dim,
        (lhs_inner_dim, rhs_inner_dim),
        (lhs.ndim, rhs.ndim),
    )

    if not fuse_bias:
        bias = jnp.zeros(0, dtype=lhs.dtype)
    elif grad:
        bias = jnp.zeros(rhs.shape[rhs_outer_dim], dtype=lhs.dtype)
    else:
        assert bias is not None, "Missing bias in forward GEMM when bias epilogue is enabled."

    if not fuse_gelu:
        gelu_input = jnp.zeros(0, dtype=lhs.dtype)
    elif grad:
        assert (
            gelu_input is not None
        ), "Backward GEMM with dGELU epilogue requires pre-GELU output from forward GEMM."
    elif gelu_input is None:
        bdims = [dim for dim in range(lhs.ndim) if dim not in [lhs_inner_dim, lhs_outer_dim]]
        batch_size = reduce(operator.mul, [lhs.shape[dim] for dim in bdims], 1)
        gelu_shape = (batch_size * lhs.shape[lhs_outer_dim], rhs.shape[rhs_outer_dim])
        gelu_input = jnp.zeros(gelu_shape, dtype=lhs.dtypes)

    dummy_fp8_meta = jnp.zeros(0, dtype=jnp.float32)
    out, _, _, pre_gelu_out, bias_grad = CollectiveGemmPrimitive.outer_primitive.bind(
        lhs,
        dummy_fp8_meta,
        rhs,
        dummy_fp8_meta,
        bias,
        gelu_input,
        dummy_fp8_meta,
        dummy_fp8_meta,
        out_dtype=lhs.dtype,
        batched_output=batched_output,
        contracting_dims=contracting_dims,
        fuse_gelu=fuse_gelu,
        fuse_bias=fuse_bias,
        grad=grad,
        accumulate=accumulate,
        use_split_accumulator=use_split_accumulator,
    )

    if grad:
        return out, pre_gelu_out, bias_grad
    return out, pre_gelu_out, None
