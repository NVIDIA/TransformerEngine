# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX te modules"""

import math
import operator
from collections.abc import Iterable
from typing import Tuple, Sequence, Union
from functools import partial, reduce

import jax
import jax.numpy as jnp
from jax import dtypes
from jax.sharding import NamedSharding, PartitionSpec

import transformer_engine_jax as tex
from transformer_engine_jax import get_device_compute_capability, get_num_compute_streams

from .base import BasePrimitive, register_primitive
from .quantization import grouped_quantize
from ..quantize import (
    ScaledTensor,
    ScaledTensor2x,
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
from .misc import get_padded_spec


__all__ = [
    "gemm",
    "grouped_gemm",
    "gemm_uses_jax_dot",
    "sanitize_dims",
    "get_non_contracting_dims",
    "transpose_contracting_dims",
]


num_cublas_streams = get_num_compute_streams()


def get_cublas_workspace_size_bytes() -> None:
    """Return 32 MiB if using hopper, 4 MiB for all other architectures."""
    if tex.get_device_compute_capability(0) >= 90:
        return 33_554_432
    return 4_194_304


def sanitize_dims(ndim: int, dims: Union[int, Sequence[int]]) -> Sequence[int]:
    """Convert relative (negative) indexes to absolute dimension numbers."""
    dims_ = dims if isinstance(dims, Iterable) else (dims,)
    if len(dims_) == 0:
        return dims_
    return tuple(ndim + dim if dim < 0 else dim for dim in dims_)


def get_non_contracting_dims(ndim, contracting_dims):
    """Return a tuple of dimensions not included in the contracting dimensions."""
    contracting_dims = sanitize_dims(ndim, contracting_dims)
    return tuple(dim for dim in range(ndim) if dim not in contracting_dims)


def transpose_contracting_dims(ndim, contracting_dims):
    """Compute the new dimension numbers for contracting dimensions after a transpose."""
    contracting_dims = sanitize_dims(ndim, contracting_dims)
    return tuple(ndim - i - 1 for i in contracting_dims)[::-1]


def _compatible_fp8_gemm_dtypes(lhs_dtype, rhs_dtype) -> bool:
    lhs, rhs, e4m3, e5m2 = map(
        dtypes.canonicalize_dtype,
        (
            lhs_dtype,
            rhs_dtype,
            jnp.float8_e4m3fn,
            jnp.float8_e5m2,
        ),
    )

    # FP8 GEMM supports (e4m3 x e4m3), (e4m3 x e5m2) and (e5m2 x e4m3)
    if (lhs is e4m3 and rhs in (e4m3, e5m2)) or (lhs in (e4m3, e5m2) and rhs is e4m3):
        return True

    # Any other combination of data types is not supported
    return False


def _get_gemm_layout(
    operand_ndims: Tuple[int, int], contracting_dims: Tuple[Sequence[int], Sequence[int]]
) -> Tuple[bool, bool]:
    lhs_contracting, rhs_contracting = map(sanitize_dims, operand_ndims, contracting_dims)
    lhs_is_transposed = operand_ndims[0] - 1 not in lhs_contracting
    rhs_is_transposed = operand_ndims[1] - 1 in rhs_contracting
    return lhs_is_transposed, rhs_is_transposed


def _quantize_gemm_operands(lhs, rhs, lhs_quantizer, rhs_quantizer, contracting_dims):
    lhs_q = lhs
    rhs_q = rhs
    if not isinstance(lhs, ScaledTensor) and lhs_quantizer is not None:
        lhs_cdims = sanitize_dims(lhs.ndim, contracting_dims[0])
        lhs_is_transposed = lhs.ndim - 1 not in lhs_cdims
        need_lhs_colwise = lhs_is_transposed and (
            lhs_quantizer.scaling_mode.is_1d_block_scaling()
            or not is_fp8_gemm_with_all_layouts_supported()
        )
        flatten_axis = max(lhs_cdims) + 1 if lhs_is_transposed else min(lhs_cdims)
        lhs_q = lhs_quantizer.quantize(
            lhs,
            is_rowwise=not need_lhs_colwise,
            is_colwise=need_lhs_colwise,
            flatten_axis=flatten_axis,
        )
        # TODO: remove
        # if isinstance(lhs_q, ScaledTensor2x):
        #     lhs_q = lhs_q.get_colwise_tensor() if lhs_is_transposed else lhs_q.get_rowwise_tensor()

    if not isinstance(rhs, ScaledTensor) and rhs_quantizer is not None:
        rhs_cdims = sanitize_dims(rhs.ndim, contracting_dims[1])
        rhs_is_transposed = rhs.ndim - 1 in rhs_cdims
        need_rhs_colwise = not rhs_is_transposed and (
            rhs_quantizer.scaling_mode.is_1d_block_scaling()
            or not is_fp8_gemm_with_all_layouts_supported()
        )
        flatten_axis = min(rhs_cdims) if rhs_is_transposed else max(rhs_cdims) + 1
        rhs_q = rhs_quantizer.quantize(
            rhs,
            is_rowwise=not need_rhs_colwise,
            is_colwise=need_rhs_colwise,
            flatten_axis=flatten_axis,
        )
        # if isinstance(rhs_q, ScaledTensor2x):
        #     rhs_q = rhs_q.get_rowwise_tensor() if rhs_is_transposed else rhs_q.get_colwise_tensor()
    assert not isinstance(lhs_q, ScaledTensor2x)
    assert not isinstance(rhs_q, ScaledTensor2x)

    return lhs_q, rhs_q


class GemmPrimitive(BasePrimitive):
    """
    Primitive for cuBLAS GEMM
    """

    name = "te_gemm_ffi"
    multiple_results = True
    impl_static_args = (6, 7, 8, 9, 10, 11, 12)
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        lhs,
        lhs_scale_inv,
        rhs,
        rhs_scale_inv,
        bias,
        gelu_input,
        out_dtype,
        contracting_dims,
        scaling_mode,
        fuse_bias,
        fuse_gelu,
        grad,
        use_split_accumulator,
    ):
        del use_split_accumulator

        # Sanity-check operand layouts and types
        operand_ndims = (lhs.ndim, rhs.ndim)
        (
            lhs_contracting_dims,
            rhs_contracting_dims,
        ) = map(sanitize_dims, operand_ndims, contracting_dims)
        lhs_contracting_size, rhs_contracting_size = map(
            lambda shape, dims: reduce(operator.mul, [shape[dim] for dim in dims]),
            (lhs.shape, rhs.shape),
            (lhs_contracting_dims, rhs_contracting_dims),
        )
        assert lhs_contracting_size == rhs_contracting_size, (
            "cuBLAS GEMM operands have incompatible contracting dimensions: "
            f"{lhs.shape} @ idx {lhs_contracting_dims} X {rhs.shape} @ idx {rhs_contracting_dims}."
        )

        lhs_is_transposed, rhs_is_transposed = _get_gemm_layout(operand_ndims, contracting_dims)
        if scaling_mode != ScalingMode.NO_SCALING:
            assert _compatible_fp8_gemm_dtypes(lhs.dtype, rhs.dtype), (
                "cuBLAS GEMM quantized operands have incompatible data types: "
                f"{lhs.dtype} x {rhs.dtype}."
            )
            assert (
                lhs_scale_inv.size > 0 and rhs_scale_inv.size > 0
            ), "Quantized cuBLAS GEMM requires inverse scaling factors for both operands."
            if (
                scaling_mode != ScalingMode.MXFP8_1D_SCALING
                and not tex.is_non_nt_fp8_gemm_supported()
            ):
                assert not lhs_is_transposed and rhs_is_transposed, (
                    "cuBLAS FP8 GEMM on devices with compute capability < 10.0 (Hopper) "
                    "require non-transposed LHS and transposed RHS operands "
                    "(`contracting_dims=((-1, ), (-1, ))`)."
                )

        # Determine output shape and dtype
        assert (
            dtypes.canonicalize_dtype(out_dtype).itemsize > 1
        ), "cuBLAS GEMM custom op does not support 8-bit quantized output types."
        lhs_non_contracting_shape, rhs_non_contracting_shape = map(
            lambda shape, dims: [shape[dim] for dim in range(len(shape)) if dim not in dims],
            (lhs.shape, rhs.shape),
            (lhs_contracting_dims, rhs_contracting_dims),
        )
        out_shape = (*lhs_non_contracting_shape, *rhs_non_contracting_shape)
        output = jax.core.ShapedArray(shape=out_shape, dtype=out_dtype)

        # Validate bias
        bias_shape = (0,)
        bias_dtype = out_dtype
        if fuse_bias:
            expected_bias_size = reduce(operator.mul, rhs_non_contracting_shape)
            if not grad:
                assert bias.size == expected_bias_size, (
                    "cuBLAS GEMM bias tensor has incorrect shape, "
                    f"expected ({expected_bias_size}, ) but found {bias.shape}."
                )
                assert bias.dtype == out_dtype, (
                    "cuBLAS GEMM bias tensor has incorrect data type, "
                    f"expected {bias_dtype} but found {bias.dtype}."
                )
                bias_shape = bias.shape
            else:
                bias_shape = rhs_non_contracting_shape
        bias_grad = jax.core.ShapedArray(shape=bias_shape, dtype=bias_dtype)

        # Validate pre-GeLU
        pre_gelu_shape = (0,)
        pre_gelu_dtype = out_dtype
        if fuse_gelu:
            pre_gelu_shape = out_shape
            if grad:
                pre_gelu_ndim = len(pre_gelu_shape)
                assert gelu_input.ndim == pre_gelu_shape and all(
                    gelu_input.shape[i] == pre_gelu_shape[i] for i in range(pre_gelu_ndim)
                ), (
                    "cuBLAS GEMM pre-GeLU tensor has incorrect shape, "
                    f"expected {pre_gelu_shape} but found {gelu_input.shape}."
                )
                assert gelu_input.dtype == out_dtype, (
                    "cuBLAS GEMM pre-GeLU tensor has incorrect data type, "
                    f"expected {pre_gelu_dtype} but found {gelu_input.dtype}."
                )
        pre_gelu_out = jax.core.ShapedArray(shape=pre_gelu_shape, dtype=pre_gelu_dtype)

        # Need extra workspace for swizzled scale factors
        lhs_swizzle_size = 0
        rhs_swizzle_size = 0
        swizzle_dtype = jnp.uint8
        if scaling_mode == ScalingMode.MXFP8_1D_SCALING:
            lhs_swizzle_size = lhs_scale_inv.size
            rhs_swizzle_size = rhs_scale_inv.size
        lhs_swizzle = jax.core.ShapedArray(shape=(lhs_swizzle_size,), dtype=swizzle_dtype)
        rhs_swizzle = jax.core.ShapedArray(shape=(rhs_swizzle_size,), dtype=swizzle_dtype)

        # Declare cuBLAS workspace
        # cuBLAS workspace ptr must be 256 bytes aligned but JAX buffers are not
        # necessarily 256 bytes aligned, we add some padding to ensure alignment.
        workspace_size = get_cublas_workspace_size_bytes() + 256
        workspace = jax.core.ShapedArray(shape=(workspace_size,), dtype=jnp.uint8)

        return output, bias_grad, pre_gelu_out, lhs_swizzle, rhs_swizzle, workspace

    @staticmethod
    def outer_abstract(*args, **kwargs):
        outputs = GemmPrimitive.abstract(*args, **kwargs)
        return outputs[:-3]  # discard workspace arrays

    @staticmethod
    def lowering(
        ctx,
        lhs,
        lhs_scale_inv,
        rhs,
        rhs_scale_inv,
        bias,
        gelu_input,
        out_dtype,
        contracting_dims,
        scaling_mode,
        fuse_bias,
        fuse_gelu,
        grad,
        use_split_accumulator,
    ):
        del out_dtype
        lhs_aval, _, rhs_aval, *_ = ctx.avals_in
        lhs_cdims, rhs_cdims = map(sanitize_dims, (lhs_aval.ndim, rhs_aval.ndim), contracting_dims)
        lhs_transposed, rhs_transposed = _get_gemm_layout(
            (lhs_aval.ndim, rhs_aval.ndim), (lhs_cdims, rhs_cdims)
        )

        args = (lhs, lhs_scale_inv, rhs, rhs_scale_inv, bias, gelu_input)
        kwargs = {
            "scaling_mode": int(scaling_mode.value),
            "lhs_axis_boundary": max(lhs_cdims) + 1 if lhs_transposed else min(lhs_cdims),
            "rhs_axis_boundary": min(rhs_cdims) if rhs_transposed else max(rhs_cdims) + 1,
            "lhs_transposed": lhs_transposed,
            "rhs_transposed": rhs_transposed,
            "fuse_bias": fuse_bias,
            "fuse_gelu": fuse_gelu,
            "grad": grad,
            "use_split_accumulator": use_split_accumulator,
        }

        operand_output_aliases = {}
        if fuse_bias and not grad:
            operand_output_aliases.update({4: 1})  # bias <-> bias_grad
        if fuse_gelu and grad:
            operand_output_aliases.update({5: 2})  # gelu_input <-> pre_gelu_out

        return jax.ffi.ffi_lowering(
            GemmPrimitive.name,
            operand_output_aliases=operand_output_aliases,
        )(ctx, *args, **kwargs)

    @staticmethod
    def impl(
        lhs,
        lhs_scale_inv,
        rhs,
        rhs_scale_inv,
        bias,
        gelu_input,
        out_dtype,
        contracting_dims,
        scaling_mode,
        fuse_bias,
        fuse_gelu,
        grad,
        use_split_accumulator,
    ):
        outputs = GemmPrimitive.inner_primitive.bind(
            lhs,
            lhs_scale_inv,
            rhs,
            rhs_scale_inv,
            bias,
            gelu_input,
            out_dtype=out_dtype,
            contracting_dims=contracting_dims,
            scaling_mode=scaling_mode,
            fuse_bias=fuse_bias,
            fuse_gelu=fuse_gelu,
            grad=grad,
            use_split_accumulator=use_split_accumulator,
        )
        return outputs[:-3]  # discard workspace arrays

    @staticmethod
    def batcher(
        batched_args,
        batch_dims,
        out_dtype,
        contracting_dims,
        scaling_mode,
        fuse_bias,
        fuse_gelu,
        grad,
        use_split_accumulator,
    ):
        assert GemmPrimitive.outer_primitive is not None
        lhs, _, rhs, *_ = batched_args
        lhs_bdims, *_ = batch_dims

        # Output is batched like LHS only if LHS is batched and RHS is not
        out_bdims = lhs_bdims if lhs.ndim > 2 and rhs.ndim == 2 else (None,)
        bias_bdims = (None,)  # Bias is never batched
        pre_gelu_bdims = (None,)  # Pre-GeLU output, if exists, is batched like GEMM output
        if fuse_gelu and not grad:
            pre_gelu_bdims = out_bdims

        return (
            GemmPrimitive.outer_primitive.bind(
                *batched_args,
                out_dtype=out_dtype,
                contracting_dims=contracting_dims,
                scaling_mode=scaling_mode,
                fuse_bias=fuse_bias,
                fuse_gelu=fuse_gelu,
                grad=grad,
                use_split_accumulator=use_split_accumulator,
            ),
            (out_bdims, bias_bdims, pre_gelu_bdims),
        )

    @staticmethod
    def infer_sharding_from_operands(
        out_dtype,
        contracting_dims,
        scaling_mode,
        fuse_bias,
        fuse_gelu,
        grad,
        use_split_accumulator,
        mesh,
        arg_infos,
        result_infos,
    ):
        del out_dtype, scaling_mode, use_split_accumulator, result_infos

        # Check contracting dimensions
        lhs_spec, _, rhs_spec, *_ = map(get_padded_spec, arg_infos)
        operand_ndims = (len(lhs_spec), len(rhs_spec))
        lhs_contracting_dims, rhs_contracting_dims = map(
            sanitize_dims, operand_ndims, contracting_dims
        )
        lhs_contracting_specs, rhs_contracting_specs = map(
            lambda specs, dims: [specs[dim] for dim in dims if specs[dim] is not None],
            (lhs_spec, rhs_spec),
            (lhs_contracting_dims, rhs_contracting_dims),
        )
        assert (
            len(lhs_contracting_specs) <= 1 and len(rhs_contracting_specs) <= 1
        ), "cuBLAS GEMM operands can have only one sharded contracting dimension."
        lhs_contracting_spec, rhs_contracting_spec = map(
            lambda spec: None if len(spec) == 0 else spec[0],
            (lhs_contracting_specs, rhs_contracting_specs),
        )
        assert (
            lhs_contracting_spec == rhs_contracting_spec
        ), "cuBLAS GEMM operands must have the same sharding in contracting dimensions."

        # Sanity check leading dimensions, allow for simultaneous batch and sequence sharding
        lhs_leading_dims, rhs_leading_dims = map(
            get_non_contracting_dims, operand_ndims, (lhs_contracting_dims, rhs_contracting_dims)
        )
        lhs_leading_specs, rhs_leading_specs = map(
            lambda specs, dims: [specs[dim] for dim in dims if specs[dim] is not None],
            (lhs_spec, rhs_spec),
            (lhs_leading_dims, rhs_leading_dims),
        )
        assert len(lhs_leading_specs) <= 1 and len(rhs_leading_specs) <= 1, (
            "cuBLAS GEMM operands cannot have more than one sharded leading dimensions. This error "
            "usually means a sequence-parallel operand was not all-gathered before the GEMM op."
        )

        # Determine output sharding
        lhs_leading_spec, rhs_leading_spec = map(
            lambda spec: None if len(spec) == 0 else spec[0], (lhs_leading_specs, rhs_leading_specs)
        )
        out_spec = (lhs_leading_spec, rhs_leading_spec)
        if operand_ndims[0] > 2 and operand_ndims[1] == 2:
            # Restore batch dimensions/sharding to the output
            out_spec = (*lhs_leading_specs, rhs_leading_spec)
        out_sharding = NamedSharding(mesh, PartitionSpec(*out_spec))

        # Bias gradient sharding inherits the RHS contracting spec
        bias_spec = (None,)
        if fuse_bias and grad:
            bias_spec = (rhs_contracting_spec,)
        bias_sharding = NamedSharding(mesh, PartitionSpec(*bias_spec))

        # Pre-GeLU sharding matches output sharding
        pre_gelu_spec = (None,)
        if fuse_gelu and not grad:
            pre_gelu_spec = out_spec
        pre_gelu_sharding = NamedSharding(mesh, PartitionSpec(*pre_gelu_spec))

        return (out_sharding, bias_sharding, pre_gelu_sharding)

    @staticmethod
    def partition(
        out_dtype,
        contracting_dims,
        scaling_mode,
        fuse_bias,
        fuse_gelu,
        grad,
        use_split_accumulator,
        mesh,
        arg_infos,
        result_infos,
    ):
        out_shardings = GemmPrimitive.infer_sharding_from_operands(
            out_dtype,
            contracting_dims,
            scaling_mode,
            fuse_bias,
            fuse_gelu,
            grad,
            use_split_accumulator,
            mesh,
            arg_infos,
            result_infos,
        )
        output_spec = out_shardings[0].spec

        # Operand shardings are already guarded with asserts so leave them unchanged here
        lhs_spec, _, rhs_spec, *_ = map(get_padded_spec, arg_infos)
        lhs_sharding = NamedSharding(mesh, PartitionSpec(*lhs_spec))
        rhs_sharding = NamedSharding(mesh, PartitionSpec(*rhs_spec))

        # Any distributed scales (e.g. MXFP8) need to be gathered
        scale_sharding = NamedSharding(mesh, PartitionSpec(None))

        # Bias has to be sharded same as the trailing dimension of the GEMM output
        bias_spec = (None,)
        if fuse_bias and not grad:
            bias_spec = (output_spec[-1],)
        bias_sharding = NamedSharding(mesh, PartitionSpec(*bias_spec))

        # Pre-GeLU output has to be sharded same as the GEMM output
        pre_gelu_spec = (None,)
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


def gemm_uses_jax_dot() -> bool:
    """Check if the GEMM call directs to the TE custom cuBLAS call or native JAX dot."""
    return not GemmPrimitive.enabled()


def _te_gemm(
    lhs: Union[jax.Array, ScaledTensor],
    rhs: Union[jax.Array, ScaledTensor],
    bias: jax.Array = None,
    gelu_input: jax.Array = None,
    lhs_quantizer: Quantizer = None,
    rhs_quantizer: Quantizer = None,
    contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((-1,), (-2,)),
    fuse_bias: bool = False,
    fuse_gelu: bool = False,
    grad: bool = False,
    use_split_accumulator: bool = QuantizeConfig.FP8_2X_ACC_FPROP,
) -> Tuple[jax.Array, ...]:
    # Prepare non-quantized GEMM operands
    lhs_data = lhs
    rhs_data = rhs
    lhs_scale_inv = jnp.empty(0, dtype=jnp.float32)
    rhs_scale_inv = jnp.empty(0, dtype=jnp.float32)
    scaling_mode = ScalingMode.NO_SCALING
    lhs_is_transposed, rhs_is_transposed = _get_gemm_layout((lhs.ndim, rhs.ndim), contracting_dims)
    lhs_cdims, rhs_cdims = map(sanitize_dims, (lhs.ndim, rhs.ndim), contracting_dims)

    # Quantize operands (if necessary)
    lhs_q, rhs_q = _quantize_gemm_operands(lhs, rhs, lhs_quantizer, rhs_quantizer, contracting_dims)

    # Extract GEMM custom op inputs from quantized operands
    if isinstance(lhs_q, ScaledTensor):
        assert isinstance(rhs_q, ScaledTensor) or rhs_quantizer is not None, (
            "cuBLAS GEMM with quantized LHS and non-quantized RHS operands requires a valid "
            "`Quantizer` object to quantize the RHS operand."
        )
        if isinstance(lhs_q, ScaledTensor2x):
            # Choose the quantization of the contracting dimension(s)
            lhs_q = lhs_q.get_colwise_tensor() if lhs_is_transposed else lhs_q.get_rowwise_tensor()
        scaling_mode = lhs_q.scaling_mode
        lhs_data = lhs_q.data
        lhs_scale_inv = lhs_q.scale_inv
        if lhs_q.data_layout == "T":
            lhs_cdims = transpose_contracting_dims(lhs_q.ndim, lhs_cdims)

    if isinstance(rhs_q, ScaledTensor):
        assert isinstance(lhs_q, ScaledTensor) or lhs_quantizer is not None, (
            "cuBLAS GEMM with non-quantized LHS and quantized RHS operands requires a valid "
            "`Quantizer` object to quantize the LHS operand."
        )
        if isinstance(rhs_q, ScaledTensor2x):
            # Choose the quantization of the contracting dimension(s)
            rhs_q = rhs_q.get_rowwise_tensor() if rhs_is_transposed else rhs_q.get_colwise_tensor()
        assert rhs_q.scaling_mode == lhs_q.scaling_mode, (
            "cuBLAS GEMM quantized operands have mismatched scaling types, "
            f"LHS:{lhs_q.scaling_mode} x RHS:{rhs_q.scaling_mode}."
        )
        rhs_data = rhs_q.data
        rhs_scale_inv = rhs_q.scale_inv
        if rhs_q.data_layout == "T":
            rhs_cdims = transpose_contracting_dims(rhs_q.ndim, rhs_cdims)

    # Dummy empties for bias and gelu
    out_dtype = lhs_q.dq_dtype if isinstance(lhs_q, ScaledTensor) else lhs_data.dtype
    if bias is None or not (fuse_bias and not grad):
        bias = jnp.empty(0, dtype=out_dtype)
    if gelu_input is None or not (fuse_gelu and grad):
        gelu_input = jnp.empty(0, dtype=out_dtype)

    return GemmPrimitive.outer_primitive.bind(
        lhs_data,
        lhs_scale_inv,
        rhs_data,
        rhs_scale_inv,
        bias,
        gelu_input,
        out_dtype=out_dtype,
        contracting_dims=(lhs_cdims, rhs_cdims),
        scaling_mode=scaling_mode,
        fuse_bias=fuse_bias,
        fuse_gelu=fuse_gelu,
        grad=grad,
        use_split_accumulator=use_split_accumulator,
    )


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
    contracting_dims_ = sanitize_dims(len(shape), contracting_dims)
    return tuple(shape[dim] for dim in range(len(shape)) if dim not in contracting_dims_)


# Apply jit to guarantee correctness of FP8 GEMM.
@partial(jax.jit, static_argnums=(2, 3))
def _jax_gemm_tensor_scaling_fp8(lhs, rhs, dim_nums, precision):
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dim_nums
    if lhs.data_layout == "T":
        lhs_contract = transpose_contracting_dims(lhs.data.ndim, lhs_contract)
    if rhs.data_layout == "T":
        rhs_contract = transpose_contracting_dims(rhs.data.ndim, rhs_contract)

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
    lhs_quantizer: Quantizer = None,
    rhs_quantizer: Quantizer = None,
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

    lhs_q, rhs_q = _quantize_gemm_operands(lhs, rhs, lhs_quantizer, rhs_quantizer, contracting_dims)

    if isinstance(lhs_q, ScaledTensor) and isinstance(rhs_q, ScaledTensor):
        # if isinstance(lhs_q, ScaledTensor2x):
        #     lhs_is_transposed = lhs.ndim - 1 not in sanitize_dims(lhs.ndim, contracting_dims[0])
        #     lhs_q = lhs_q.get_colwise_tensor() if lhs_is_transposed else lhs_q.get_rowwise_tensor()
        # if isinstance(rhs_q, ScaledTensor2x):
        #     rhs_is_transposed = rhs.ndim - 1 in sanitize_dims(rhs.ndim, contracting_dims[1])
        #     rhs_q = rhs_q.get_rowwise_tensor() if rhs_is_transposed else rhs_q.get_colwise_tensor()
        return _jax_gemm_fp8_impl(lhs_q, rhs_q)

    if (
        isinstance(lhs, jnp.ndarray)
        and isinstance(rhs, jnp.ndarray)
        and lhs_quantizer is None
        and rhs_quantizer is None
    ):
        return jax.lax.dot_general(lhs, rhs, dim_nums, preferred_element_type=lhs.dtype)

    raise NotImplementedError("Not supporting multiplication of ScaledTensor and jnp.array")


def gemm(
    lhs: Union[jnp.ndarray, ScaledTensor],
    rhs: Union[jnp.ndarray, ScaledTensor],
    contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((-1,), (-2,)),
    lhs_quantizer: Quantizer = None,
    rhs_quantizer: Quantizer = None,
    **kwargs,
) -> Tuple[jnp.ndarray, ...]:
    r"""General matrix multiplication with optional quantization.

    Parameters
    ----------
    lhs: Union[jax.Array, ScaledTensor]
        Left-hand side operand in the matrix multiplication.
    rhs: Union[jax.Array, ScaledTensor]
        Right-hand side operand in the matrix multiplication.
    lhs_quantizer: Quantizer, default = None
        Object for down-casting the LHS operand for quantized GEMM.
    rhs_quantizer: Quantizer, default = None
        Object for down-casting the RHS operand for quantized GEMM.
    contracting_dims: Tuple[Sequence[int], Sequence[int]], default = ((-1, ), (-2, ))
        Tuple of two sequences representing the contracting dimensions. The first sequence
        represents the contracting dimensions of the LHS operand, and the second sequence
        represents the contracting dimensions of the RHS operand.
    bias: jax.Array, default = None
        Optional additive bias term, required for forward GEMM with bias fusion. Only supported
        with TE's custom call to cuBLAS GEMM.
    gelu_input: jax.Array, default = None
        Pre-GeLU output from forward GEMM, required for backward/grad GEMM with dGeLU fusion. Only
        supported with TE's custom call to cuBLAS GEMM.
    fuse_bias: bool, default = False
        Enable bias addition in forward GEMM or bias gradient in backward GEMM. Only supported with
        TE's custom call to cuBLAS GEMM.
    fuse_gelu: bool, default = False
        Enable GeLU activation in forward GEMM or GeLU gradient in backward GEMM. Only supported
        with TE's custom call to cuBLAS GEMM.
    grad: bool, default = False
        Flag for switching bias and GeLU fusions from forward to backward mode. Only supported with
        TE's custom call to cuBLAS GEMM.
    use_split_accumulator: bool, default = True
        Enable promoting some intermediate sums to higher precision when accumulating the result in
        the cuBLAS GEMM kernel. Disabling this trades off numerical accuracy for speed.

    Returns
    -------
    jax.Array:
        Result of the operation. For TE's custom call to cuBLAS GEMM, this result can include the
        GeLU application when `fuse_gelu=True` and `grad=False`, the GeLU gradient contribution
        when `fuse_gelu=True` and `grad=True`, and the additive bias when `fuse_bias=True` and
        `grad=False`.
    Optional[jax.Array]:
        Bias gradient when `fuse_bias=True` and `grad=True`. Only supported with TE's custom call
        to cuBLAS GEMM.
    Optional[jax.Array]:
        Pre-GeLU GEMM output when `fuse_gelu=True` and `grad=False`. This is required as an input
        to `_te_gemm()` with `fuse_gelu=True` and `grad=True` in the backward pass in order to
        compute the GeLU contribution to the gradient. Only supported with TE's custom call to
        cuBLAS GEMM.
    """
    # Try to get LHS and RHS quantizers from a quantizer set for backward compatibility
    if lhs_quantizer is None or rhs_quantizer is None:
        quantizer_set = kwargs.get("quantizer_set", None)
        if quantizer_set is not None:
            lhs_quantizer = quantizer_set.x
            rhs_quantizer = quantizer_set.kernel

    # Fall back on a native JAX implementation when the custom call to cuBLAS GEMM is disabled
    fuse_bias = kwargs.get("fuse_bias", False)
    fuse_gelu = kwargs.get("fuse_gelu", False)
    if not GemmPrimitive.enabled():
        assert kwargs.get("bias", None) is None and not fuse_gelu, (
            "TE GEMM was invoked with bias fusion options that are not supported by the "
            "`jax.lax.dot_general` and `jnp.scaled_matmul` backends used when the custom cuBLAS "
            "GEMM primitive is disabled."
        )
        assert kwargs.get("gelu_input", None) is None and not fuse_bias, (
            "TE GEMM was invoked with GeLU fusion options that are not supported by the "
            "`jax.lax.dot_general` and `jnp.scaled_matmul` backends used when the custom cuBLAS "
            "GEMM primitive is disabled."
        )
        return _jax_gemm(lhs, rhs, contracting_dims, lhs_quantizer, rhs_quantizer)

    outputs = _te_gemm(
        lhs,
        rhs,
        lhs_quantizer=lhs_quantizer,
        rhs_quantizer=rhs_quantizer,
        contracting_dims=contracting_dims,
        **kwargs,
    )

    # Discard empty outputs
    grad = kwargs.get("grad", False)
    clean_outputs = outputs[0]  # first output is the final result and is never empty
    if (fuse_bias and grad) or (fuse_gelu and not grad):
        clean_outputs = (outputs[0],)
        if fuse_bias and grad:  # only return bias gradient if it exists
            clean_outputs += (outputs[1],)
        if fuse_gelu and not grad:  # only return pre-GeLU output if it exists
            clean_outputs += (outputs[2],)
    return clean_outputs


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
