# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX te modules"""

import math
import operator
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial, reduce
from typing import Tuple, Sequence, Union
from enum import Enum
import warnings

import jax
import jax.numpy as jnp
from jax import dtypes
from jax.sharding import NamedSharding, PartitionSpec
from jax.experimental.custom_partitioning import SdyShardingRule

from transformer_engine_jax import (
    get_num_compute_streams,
    JAXX_Collective_Op,
    get_device_compute_capability,
)

from .base import BasePrimitive, register_primitive
from .quantization import grouped_quantize
from ..quantize import (
    AbstractBaseTensor,
    NoScaleTensor,
    ScaledTensor,
    ScaledTensor2x,
    GroupedScaledTensor1x,
    ScalingMode,
    Quantizer,
    GroupedQuantizer,
    get_quantize_config,
    QuantizerSet,
    QuantizeLayout,
    noop_quantizer_set,
    is_fp8_gemm_with_all_layouts_supported,
    apply_padding_to_scale_inv,
)
from .misc import get_padded_spec, is_all_reduce_in_float32
from ..sharding import (
    global_mesh_resource,
    tpsp_axis_size,
    dp_or_fsdp_axis_size,
)


__all__ = [
    "CollectiveGemmConfig",
    "CollectiveGemmConfigSet",
    "CollectiveOp",
    "noop_cgemm_config_set",
    "gemm",
    "grouped_gemm",
    "gemm_uses_jax_dot",
    "sanitize_dims",
    "get_non_contracting_dims",
    "transpose_dims",
]


num_cublas_streams = get_num_compute_streams()


def get_cublas_workspace_size_bytes() -> None:
    """Return 32 MiB if using hopper, 4 MiB for all other architectures."""
    if get_device_compute_capability(0) >= 90:
        return 33_554_432
    return 4_194_304


def sanitize_dims(ndim: int, dims: Union[int, Sequence[int]]) -> Sequence[int]:
    """Convert relative (negative) indexes to absolute dimension numbers."""
    dims_ = dims if isinstance(dims, Iterable) else (dims,)
    if len(dims_) == 0:
        return dims_
    return tuple(ndim + dim if dim < 0 else dim for dim in dims_ if dim is not None)


def get_non_contracting_dims(ndim, contracting_dims):
    """Return a tuple of dimensions not included in the contracting dimensions."""
    contracting_dims = sanitize_dims(ndim, contracting_dims)
    return tuple(dim for dim in range(ndim) if dim not in contracting_dims)


def transpose_dims(ndim, dims_to_transpose, flatten_axis=-1):
    """Compute the new dimension numbers after transpose."""
    if len(dims_to_transpose) == 0:
        return dims_to_transpose
    flatten_axis = ndim - flatten_axis if flatten_axis > 0 else flatten_axis
    transposed_dims = (*range(flatten_axis, ndim), *range(flatten_axis))
    return tuple(transposed_dims.index(dim) for dim in dims_to_transpose)


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

    assert not isinstance(lhs_q, ScaledTensor2x)
    assert not isinstance(rhs_q, ScaledTensor2x)

    return lhs_q, rhs_q


class CollectiveOp(Enum):
    "Enum for Collective Type in Collective GEMM"

    NONE = JAXX_Collective_Op.NONE
    ALL_GATHER = JAXX_Collective_Op.ALL_GATHER
    REDUCE_SCATTER = JAXX_Collective_Op.REDUCE_SCATTER

    @property
    def is_all_gather(self) -> bool:
        return self == CollectiveOp.ALL_GATHER

    @property
    def is_reduce_scatter(self) -> bool:
        return self == CollectiveOp.REDUCE_SCATTER

    @property
    def is_none(self) -> bool:
        return self == CollectiveOp.NONE


@dataclass(frozen=True)
class CollectiveGemmConfig:
    """
    Configuration object that carries collective GEMM configuration
    """

    collective_op: CollectiveOp
    num_max_streams: int
    lowering_cgemm_attrs: dict

    def __hash__(self):
        return hash(tuple(self.lowering_cgemm_attrs.items()))

    @staticmethod
    def create(
        collective_op: CollectiveOp,
        num_max_streams: int = 3,  # Why 3?
        gemm_priority: int = 0,
        comm_priority: int = 0,
        num_comm_sm: int = None,
        use_ce: bool = True,
        aggregate_ag: bool = False,  # TODO: do we need this?
    ):
        """Create a CollectiveGemmConfig with all values properly set and initialize the Userbuffer"""

        tp_size = tpsp_axis_size() if not collective_op.is_none else 1
        num_comm_sm = num_comm_sm or 2

        # Create the communication plan
        lowering_cgemm_attrs = {
            "collective_op": int(collective_op.value),
            "tp_size": int(tp_size),
            "num_max_streams": int(num_max_streams),
            "gemm_priority": int(gemm_priority),
            "comm_priority": int(comm_priority),
            "num_comm_sm": int(num_comm_sm),
            "use_ce": use_ce,
            "aggregate_ag": aggregate_ag,
        }

        # Create the instance
        instance = CollectiveGemmConfig(
            collective_op=collective_op,
            num_max_streams=num_max_streams,
            lowering_cgemm_attrs=lowering_cgemm_attrs,
        )

        return instance

    def __post_init__(self):
        """Validate the configuration after initialization."""
        if not self.collective_op.is_none:
            assert tpsp_axis_size() > 1, (
                "Communication + GEMM overlap requires a valid TP axis size. Got TP axis size"
                f" {tpsp_axis_size()}."
            )
            assert (
                tpsp_axis_size() % 2 == 0
            ), f"Tensor-parallel axis of {tpsp_axis_size()} is not divisible by 2."


@dataclass(frozen=True)
class CollectiveGemmConfigSet:
    """
    A set of CollectiveGemmConfig objects that provide complementary collective GEMM configurations for the Forward and Backward passes through Dense-layers.
    """

    forward: CollectiveGemmConfig
    backward: CollectiveGemmConfig

    @staticmethod
    def create(
        forward_collective_op: CollectiveOp,
        num_max_streams: int = 3,
        gemm_priority: int = 0,
        comm_priority: int = 0,
        num_comm_sm: int = None,
        use_ce: bool = True,
        aggregate_ag: bool = False,
    ):
        if forward_collective_op.is_all_gather:
            backward_collective_op = CollectiveOp.REDUCE_SCATTER
        elif forward_collective_op.is_reduce_scatter:
            backward_collective_op = CollectiveOp.ALL_GATHER
        else:
            backward_collective_op = CollectiveOp.NONE

        forward = CollectiveGemmConfig.create(
            forward_collective_op,
            num_max_streams,
            gemm_priority,
            comm_priority,
            num_comm_sm,
            use_ce,
            aggregate_ag,
        )

        backward = CollectiveGemmConfig.create(
            backward_collective_op,
            num_max_streams,
            gemm_priority,
            comm_priority,
            num_comm_sm,
            use_ce,
            aggregate_ag,
        )
        return CollectiveGemmConfigSet(forward=forward, backward=backward)


noop_cgemm_config = CollectiveGemmConfig.create(collective_op=CollectiveOp.NONE)

noop_cgemm_config_set = CollectiveGemmConfigSet.create(forward_collective_op=CollectiveOp.NONE)


@partial(jax.jit, static_argnums=(1,))
def _cgemm_output_reorder(output, sequence_dim):
    assert sequence_dim >= 0, f"Invalid sequence_dim. Got sequence_dim={sequence_dim}"
    original_shape = output.shape
    reshaped = output.reshape(
        -1,
        tpsp_axis_size(),
        int(original_shape[sequence_dim] / tpsp_axis_size()),
        *original_shape[sequence_dim + 1 :],
    )
    reordered = reshaped.transpose(1, 0, 2, *range(3, reshaped.ndim))
    output = reordered.reshape(original_shape)
    return output


class GemmPrimitive(BasePrimitive):
    """
    Primitive for cuBLAS GEMM
    """

    name = "te_gemm_ffi"
    multiple_results = True
    impl_static_args = 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
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
        transpose_batch_sequence,
        sequence_dim,
        is_outer,
        cgemm_config,
    ):
        del use_split_accumulator, transpose_batch_sequence

        def _dims_are_consecutive(dims):
            if len(dims) <= 1:
                return True
            return sorted(dims) == list(range(min(dims), max(dims) + 1))

        # Sanity-check operand layouts and types
        operand_ndims = (lhs.ndim, rhs.ndim)

        (
            lhs_contracting_dims,
            rhs_contracting_dims,
        ) = map(sanitize_dims, operand_ndims, contracting_dims)

        assert _dims_are_consecutive(lhs_contracting_dims), (
            "cuBLAS GEMM expected consecutive contracting dimensions for LHS operand, but got "
            f"{lhs_contracting_dims}."
        )
        assert _dims_are_consecutive(rhs_contracting_dims), (
            "cuBLAS GEMM expected consecutive contracting dimensions for RHS operand, but got "
            f"{rhs_contracting_dims}."
        )

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
                and not is_fp8_gemm_with_all_layouts_supported()
            ):
                assert not lhs_is_transposed and rhs_is_transposed, (
                    "cuBLAS FP8 GEMM on devices with compute capability < 10.0 (Hopper) "
                    "require non-transposed LHS and transposed RHS operands "
                    "(`contracting_dims=((-1, ), (-1, ))`)."
                )
        else:
            assert lhs.dtype == rhs.dtype, (
                "For TE cuBLAS GEMM for non-quantized inputs, the operand dtypes must be equal."
                f" LHS dtype != RHS dtype, lhs.dtype={lhs.dtype}, rhs.dtype={rhs.dtype}"
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

        # Adjust output shape for comm+GEMM overlap
        if not cgemm_config.collective_op.is_none and not is_outer:  # Inner abstract
            assert sequence_dim == 1, f"Invalid sequence_dim. Got sequence_dim={sequence_dim}"
            overlap_out_shape = list(out_shape).copy()
            if cgemm_config.collective_op.is_all_gather:
                overlap_out_shape[1] *= tpsp_axis_size()
            else:  # RS
                overlap_out_shape[sequence_dim] = (
                    overlap_out_shape[sequence_dim] // tpsp_axis_size()
                )
            assert out_dtype == jnp.bfloat16, f"Unsupported out_dtype={out_dtype}"
            output = jax.core.ShapedArray(shape=overlap_out_shape, dtype=out_dtype)

        # Validate bias -- shape always depends on pure GEMM output
        # (Phuong) This is not true for TP + Hidden
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

        # Validate pre-GeLU -- shape always depends on pure GEMM output
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

        # Size cuBLAS workspace -- multiplied by number of comm+GEMM overlap compute streams
        workspace_size = get_cublas_workspace_size_bytes()
        if cgemm_config is not None:
            workspace_size *= cgemm_config.num_max_streams

        # cuBLAS requires workspace pointers aligned to 256 bytes but XLA does not guarantee that
        # so we add to the size here and align the pointer in the C++ custom call.
        workspace_size += 256
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
        transpose_batch_sequence,
        sequence_dim,
        is_outer,
        cgemm_config,
    ):
        del out_dtype, transpose_batch_sequence, sequence_dim, is_outer

        lhs_aval, _, rhs_aval, *_ = ctx.avals_in
        lhs_cdims, rhs_cdims = map(sanitize_dims, (lhs_aval.ndim, rhs_aval.ndim), contracting_dims)
        lhs_transposed, rhs_transposed = _get_gemm_layout(
            (lhs_aval.ndim, rhs_aval.ndim), (lhs_cdims, rhs_cdims)
        )

        args = (lhs, lhs_scale_inv, rhs, rhs_scale_inv, bias, gelu_input)
        kwargs = {
            "scaling_mode": int(scaling_mode.value),
            "lhs_axis_boundary": int(max(lhs_cdims) + 1 if lhs_transposed else min(lhs_cdims)),
            "rhs_axis_boundary": int(min(rhs_cdims) if rhs_transposed else max(rhs_cdims) + 1),
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
        )(ctx, *args, **kwargs, cgemm_config=cgemm_config.lowering_cgemm_attrs)

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
        transpose_batch_sequence,
        sequence_dim,
        is_outer,
        cgemm_config,
    ):
        lhs_cdims, rhs_cdims = map(sanitize_dims, (lhs.ndim, rhs.ndim), contracting_dims)
        lhs_transposed, rhs_transposed = _get_gemm_layout(
            (lhs.ndim, rhs.ndim), (lhs_cdims, rhs_cdims)
        )
        lhs_scale_inv = apply_padding_to_scale_inv(
            lhs_scale_inv,
            scaling_mode,
            lhs.shape,
            is_colwise=lhs_transposed,
            flatten_axis=max(lhs_cdims) + 1 if lhs_transposed else min(lhs_cdims),
        )
        rhs_scale_inv = apply_padding_to_scale_inv(
            rhs_scale_inv,
            scaling_mode,
            rhs.shape,
            is_colwise=not rhs_transposed,
            flatten_axis=min(rhs_cdims) if rhs_transposed else max(rhs_cdims) + 1,
        )
        # Alter lhs blocks so that CGEMM RS outputs correctly
        if (
            cgemm_config.collective_op.is_reduce_scatter
            and not transpose_batch_sequence
            and not is_outer
        ):
            assert sequence_dim == 1, f"Invalid sequence_dim. Got sequence_dim={sequence_dim}"
            original_shape = lhs.shape
            reshaped = lhs.reshape(
                dp_or_fsdp_axis_size(),
                int(original_shape[0] / dp_or_fsdp_axis_size()),
                tpsp_axis_size(),
                int(original_shape[1] / tpsp_axis_size()),
                *original_shape[2:],
            )
            reordered = reshaped.transpose(2, 0, 1, 3, *range(4, reshaped.ndim))
            lhs = reordered.reshape(original_shape)

        (output, bias_grad, pre_gelu_out, _, _, _) = GemmPrimitive.inner_primitive.bind(
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
            cgemm_config=cgemm_config,
            transpose_batch_sequence=transpose_batch_sequence,
            sequence_dim=sequence_dim,
            is_outer=is_outer,
        )
        # Alter output blocks for CGEMM AG
        if (
            cgemm_config.collective_op.is_all_gather
            and not transpose_batch_sequence
            and not is_outer
        ):
            assert sequence_dim == 1, f"Invalid sequence_dim. Got sequence_dim={sequence_dim}"
            original_shape = output.shape
            reshaped = output.reshape(
                tpsp_axis_size(),
                dp_or_fsdp_axis_size(),
                int(original_shape[0] / dp_or_fsdp_axis_size()),
                int(original_shape[1] / tpsp_axis_size()),
                *original_shape[2:],
            )
            reordered = reshaped.transpose(1, 2, 0, 3, *range(4, reshaped.ndim))
            output = reordered.reshape(original_shape)

        return [output, bias_grad, pre_gelu_out]

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
        cgemm_config,
        transpose_batch_sequence,
        sequence_dim,
        is_outer,
    ):
        del transpose_batch_sequence, sequence_dim, is_outer
        assert GemmPrimitive.outer_primitive is not None
        lhs_bdims, _, rhs_bdims = batch_dims

        # Batched GEMM is not supported
        assert (
            lhs_bdims is None and rhs_bdims is None
        ), f"(Batching is not supported, got lhs_bdims={lhs_bdims}, rhs_bdims={rhs_bdims})"
        out_bdims = (None,)

        # Batched GEMM is not supported
        assert (
            lhs_bdims is None and rhs_bdims is None
        ), f"(Batching is not supported, got lhs_bdims={lhs_bdims}, rhs_bdims={rhs_bdims})"
        out_bdims = (None,)

        # Bias gradient is never batched
        bias_bdims = (None,)

        # Pre-GeLU output, if exists, is batched like GEMM output
        pre_gelu_bdims = (None,)
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
                cgemm_config=cgemm_config,
            ),
            (out_bdims, bias_bdims, pre_gelu_bdims),
        )

    @staticmethod
    def _parse_operand_output_specs(
        arg_infos,
        contracting_dims,
        transpose_batch_sequence,
        cgemm_config,
    ):
        lhs_specs, _, rhs_specs, *_ = map(get_padded_spec, arg_infos)

        gsr = global_mesh_resource()

        lhs_ndim, rhs_ndim = map(len, (lhs_specs, rhs_specs))
        lhs_cdims, rhs_cdims = map(sanitize_dims, (lhs_ndim, rhs_ndim), contracting_dims)
        lhs_non_cdims, rhs_non_cdims = map(
            lambda ndim, cdims: tuple(i for i in range(ndim) if i not in cdims),
            (lhs_ndim, rhs_ndim),
            (lhs_cdims, rhs_cdims),
        )
        lhs_non_cspecs, lhs_cspecs, rhs_non_cspecs, rhs_cspecs = map(
            lambda specs, dims: tuple(specs[i] for i in dims),
            (lhs_specs, lhs_specs, rhs_specs, rhs_specs),
            (lhs_non_cdims, lhs_cdims, rhs_non_cdims, rhs_cdims),
        )

        reduce_spec = None
        for l in lhs_cspecs:
            for r in rhs_cspecs:
                if l is not None and l == r:
                    assert reduce_spec is None, "Multiple reduce dimension is detected!"
                    reduce_spec = l

        sequence_dim = None

        # Find sequence dimension in lhs_specs if tensor sequence parallel is enabled
        # We only do CollectiveGemm AG on the x or dY thus they always the LHS and have sequence dim
        if cgemm_config.collective_op.is_all_gather:
            try:
                tpsp_idx = lhs_specs.index(gsr.tpsp_resource)
            except ValueError as exc:
                raise ValueError(
                    f"tpsp_resource '{gsr.tpsp_resource}' is not found in lhs_specs: {lhs_specs}."
                    " Please check your sharding configuration."
                ) from exc
            sequence_dim = tpsp_idx
            assert (sequence_dim == 1) ^ transpose_batch_sequence, (
                "CollectiveGEMM supports only (sequence_dim=1 and transpose_batch_sequence=False)"
                " or (sequence_dim=0 and transpose_batch_sequence=True). Received:"
                f" sequence_dim={sequence_dim},"
                f" transpose_batch_sequence={transpose_batch_sequence}."
            )

        elif cgemm_config.collective_op.is_reduce_scatter:
            assert reduce_spec == gsr.tpsp_resource, (
                "Only CollectiveGemm RS with the Reduction over the TPSP axis is supported! Got"
                f" reduce_spec={reduce_spec}, tpsp_resource={gsr.tpsp_resource}"
            )
            sequence_dim = int(not transpose_batch_sequence)

        if reduce_spec is not None:
            # Other non-reduce cdims (if exists) need to be unsharded
            lhs_cspecs = tuple(s if s == reduce_spec else None for s in lhs_cspecs)
            # Only do AG Sequence dim if not Overlap
            if cgemm_config.collective_op.is_all_gather:
                rhs_cspecs = tuple(
                    s if s in (reduce_spec, gsr.tpsp_resource) else None for s in rhs_cspecs
                )
            else:
                rhs_cspecs = tuple(s if s == reduce_spec else None for s in rhs_cspecs)

            # Non-contracting dims of RHS always needs to be gathered, i.e. for TP + activation_hidden
            # No batch-dim check needed as `rhs_non_cspecs` never contains batch-dim.
            # In `rhs_specs`, the batch dim appears only in Wgrad GEMM under `rhs_cspecs`.
            rhs_non_cspecs = tuple(
                None if spec in lhs_non_cspecs else spec for spec in rhs_non_cspecs
            )

        else:
            # Otherwise, require contracting dims of both operands to be unsharded
            lhs_cspecs = (None,) * len(lhs_cspecs)
            rhs_cspecs = (None,) * len(rhs_cspecs)

            # Non-contracting dims of RHS always needs to be gathered along the FSDP axis
            rhs_non_cspecs = tuple(
                None if spec is not None and spec == gsr.fsdp_resource else spec
                for spec in rhs_non_cspecs
            )

        # Only do AG Sequence dim if not Overlap
        if not cgemm_config.collective_op.is_all_gather:
            # Non-contracting dims of LHS to be gathered along the SP axis.
            # Minor note: This causes MaxText TP (= Megatron TP + activation_hidden sharding) gathering x for
            # dW1 = x^T * dY1 which is unexpected. This is a known issue and no solution has found yet.
            lhs_non_cspecs = tuple(
                None if spec in rhs_non_cspecs else spec for spec in lhs_non_cspecs
            )

        out_specs = lhs_non_cspecs + rhs_non_cspecs

        # Only do AG Sequence dim if not Overlap RS
        if cgemm_config.collective_op.is_all_gather:
            assert sequence_dim <= len(
                lhs_non_cspecs
            ), f"Sequence dim {sequence_dim} is out of bounds for lhs_non_cspecs: {lhs_non_cspecs}"
            out_specs = out_specs[:sequence_dim] + (None,) + out_specs[sequence_dim + 1 :]
        elif cgemm_config.collective_op.is_reduce_scatter:
            assert sequence_dim <= len(
                lhs_non_cspecs
            ), f"Sequence dim {sequence_dim} is out of bounds for lhs_non_cspecs: {lhs_non_cspecs}"
            out_specs = (
                out_specs[:sequence_dim] + (gsr.tpsp_resource,) + out_specs[sequence_dim + 1 :]
            )

        # specs = merge(cspecs, non_cspecs)
        lhs_specs, rhs_specs = map(
            lambda cdims, cspecs, non_cspecs: (
                cspecs + non_cspecs if cdims[0] == 0 else non_cspecs + cspecs
            ),
            (lhs_cdims, rhs_cdims),
            (lhs_cspecs, rhs_cspecs),
            (lhs_non_cspecs, rhs_non_cspecs),
        )

        # Bias and Pre-GeLU sharding is based on GEMM output before any scatter
        bias_specs = tuple(list(rhs_non_cspecs).copy())
        gelu_specs = tuple(list(out_specs).copy())

        if not cgemm_config.collective_op.is_none:
            assert sequence_dim >= 0, f"Invalid sequence_dim. Got sequence_dim={sequence_dim}"

        return (
            (lhs_specs, rhs_specs, bias_specs, gelu_specs),
            (out_specs, bias_specs, gelu_specs),
            reduce_spec,
            sequence_dim,
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
        transpose_batch_sequence,
        sequence_dim,
        is_outer,
        cgemm_config,
        mesh,
        arg_infos,
        result_infos,
    ):
        del (
            out_dtype,
            scaling_mode,
            grad,
            use_split_accumulator,
            result_infos,
            is_outer,
            sequence_dim,
        )

        (_, (out_specs, dbias_specs, pre_gelu_specs), *_) = (
            GemmPrimitive._parse_operand_output_specs(
                arg_infos, contracting_dims, transpose_batch_sequence, cgemm_config
            )
        )
        out_sharding = NamedSharding(mesh, PartitionSpec(*out_specs))

        # Discard bias gradient spec if there is no bias fusion
        if not fuse_bias:
            dbias_specs = (None,)
        dbias_sharding = NamedSharding(mesh, PartitionSpec(*dbias_specs))

        # Discard pre-GeLU output spec if there is no GeLU fusion
        if not fuse_gelu:
            pre_gelu_specs = (None,)
        pre_gelu_sharding = NamedSharding(mesh, PartitionSpec(*pre_gelu_specs))

        return [out_sharding, dbias_sharding, pre_gelu_sharding]

    @staticmethod
    def partition(
        out_dtype,
        contracting_dims,
        scaling_mode,
        fuse_bias,
        fuse_gelu,
        grad,
        use_split_accumulator,
        transpose_batch_sequence,
        sequence_dim,
        is_outer,
        cgemm_config,
        mesh,
        arg_infos,
        result_infos,
    ):
        del result_infos, is_outer, sequence_dim

        (
            (lhs_specs, rhs_specs, bias_input_specs, gelu_input_specs),
            (out_specs, dbias_specs, pre_gelu_specs),
            reduce_spec,
            inferred_sequence_dim,
        ) = GemmPrimitive._parse_operand_output_specs(
            arg_infos, contracting_dims, transpose_batch_sequence, cgemm_config
        )

        # Block scale inverses match their operands, but tensor scale inverses are unsharded.
        none_sharding = NamedSharding(mesh, PartitionSpec(None))
        lhs_sharding = NamedSharding(mesh, PartitionSpec(*lhs_specs))
        rhs_sharding = NamedSharding(mesh, PartitionSpec(*rhs_specs))
        arg_shardings = (
            lhs_sharding,
            lhs_sharding if scaling_mode.is_1d_block_scaling() else none_sharding,
            rhs_sharding,
            rhs_sharding if scaling_mode.is_1d_block_scaling() else none_sharding,
        )

        # Discard bias input spec if there is no bias fusion
        if not fuse_bias:
            bias_input_specs = (None,)
        arg_shardings += (NamedSharding(mesh, PartitionSpec(*bias_input_specs)),)

        # Discard pre-GeLU input spec if there is no GeLU fusion
        if not fuse_gelu:
            gelu_input_specs = (None,)
        arg_shardings += (NamedSharding(mesh, PartitionSpec(*gelu_input_specs)),)

        # Assemble output shardings
        out_shardings = [NamedSharding(mesh, PartitionSpec(*out_specs))]

        # Discard bias gradient spec if there is no bias fusion
        if not fuse_bias:
            dbias_specs = (None,)
        out_shardings.append(NamedSharding(mesh, PartitionSpec(*dbias_specs)))

        # Discard pre-GeLU output spec if there is no GeLU fusion
        if not fuse_gelu:
            pre_gelu_specs = (None,)
        out_shardings.append(NamedSharding(mesh, PartitionSpec(*pre_gelu_specs)))

        def _sharded_impl(lhs, lhs_scale_inv, rhs, rhs_scale_inv, bias, gelu_input):
            outputs = GemmPrimitive.impl(
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
                transpose_batch_sequence=transpose_batch_sequence,
                sequence_dim=inferred_sequence_dim,
                is_outer=False,
                cgemm_config=cgemm_config,
            )

            if reduce_spec is not None and not cgemm_config.collective_op.is_reduce_scatter:
                if is_all_reduce_in_float32():  # For unittest only
                    outputs[0] = jax.lax.psum(outputs[0].astype(jnp.float32), reduce_spec).astype(
                        out_dtype
                    )
                else:
                    outputs[0] = jax.lax.psum(outputs[0], reduce_spec)

            return outputs

        return mesh, _sharded_impl, out_shardings, arg_shardings

    @staticmethod
    def shardy_sharding_rule(
        out_dtype,
        contracting_dims,
        scaling_mode,
        fuse_bias,
        fuse_gelu,
        grad,
        use_split_accumulator,
        transpose_batch_sequence,
        sequence_dim,
        is_outer,
        cgemm_config,
        mesh,
        operand_types,
        result_types,
    ):
        del out_dtype, grad, use_split_accumulator
        del mesh, result_types, transpose_batch_sequence, sequence_dim, is_outer, cgemm_config

        if cgemm_config is not None:
            raise NotImplementedError(
                "CollectiveGEMM with Shardy propagation is not supported yet! Please turn off"
                " Shardy by exporting env var JAX_USE_SHARDY_PARTITIONER=false"
            )

        prefix = "GemmPrimitive_"

        warnings.warn(
            "Known issues with TE GemmPrimitives when Shardy propagation is enabled. For now,"
            " please turn off Shardy by exporting the environment variable"
            " 'JAX_USE_SHARDY_PARTITIONER=0' if you experience any problems."
        )

        def _generate_operand_rules(name, ndim, cdims):
            specs = []
            ldims = tuple(i for i in range(ndim) if i not in cdims)
            for i in range(ndim):
                dim_name = None
                if i in cdims:
                    dim_idx = cdims.index(i)
                    dim_name = f"k{dim_idx}"
                else:
                    dim_idx = ldims.index(i)
                    dim_name = f"{name}_l{dim_idx}"
                specs.append(prefix + dim_name)
            return specs

        lhs, _, rhs, *_ = operand_types
        operand_ndims = (len(lhs.shape), len(rhs.shape))
        (lhs_cdims, rhs_cdims) = map(sanitize_dims, operand_ndims, contracting_dims)
        lhs_specs, rhs_specs = map(
            _generate_operand_rules,
            ("lhs", "rhs"),
            operand_ndims,
            (lhs_cdims, rhs_cdims),
        )
        lhs_scale_specs = ("…1",)
        rhs_scale_specs = ("…2",)
        if scaling_mode.is_1d_block_scaling():
            # Shardy rules for MXFP8 scales cannot be related to the operands because of the
            # global-unpadding and local-padding workflow. This can potentially insert expensive
            # re-shards in the partition call later if the scales are not already sharded correctly.
            lhs_scale_specs, rhs_scale_specs = map(
                lambda specs: tuple(spec.replace(prefix, prefix + "scale_inv_") for spec in specs),
                (lhs_specs, rhs_specs),
            )

        lhs_non_cspec = tuple(lhs_specs[i] for i in range(operand_ndims[0]) if i not in lhs_cdims)
        rhs_non_cspec = tuple(rhs_specs[i] for i in range(operand_ndims[1]) if i not in rhs_cdims)
        out_spec = (*lhs_non_cspec, *rhs_non_cspec)
        bias_spec = rhs_non_cspec if fuse_bias else ("…4",)
        gelu_spec = out_spec if fuse_gelu else ("…5",)

        return SdyShardingRule(
            operand_mappings=(
                lhs_specs,
                lhs_scale_specs,
                rhs_specs,
                rhs_scale_specs,
                bias_spec,
                gelu_spec,
            ),
            result_mappings=(
                out_spec,
                bias_spec,
                gelu_spec,
            ),
        )


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
    contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((-1,), (0,)),
    fuse_bias: bool = False,
    fuse_gelu: bool = False,
    grad: bool = False,
    use_split_accumulator: bool = get_quantize_config().FP8_2X_ACC_FPROP,
    transpose_batch_sequence: bool = False,
    cgemm_config: CollectiveGemmConfig = noop_cgemm_config,
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
            lhs_cdims = transpose_dims(lhs_q.ndim, lhs_cdims, flatten_axis=lhs_q.flatten_axis)

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
            rhs_cdims = transpose_dims(rhs_q.ndim, rhs_cdims, flatten_axis=rhs_q.flatten_axis)

    # Dummy empties for bias, gelu and aux_in
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
        transpose_batch_sequence=transpose_batch_sequence,
        sequence_dim=-1,
        is_outer=True,
        cgemm_config=cgemm_config,
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
        del K, lhs_is_trans, rhs_is_trans, has_bias
        # TODO(Phuong): move some shape checks from Cpp to here
        workspace_size = get_cublas_workspace_size_bytes() * num_cublas_streams
        workspace_alignment_padding = 256
        tensor_scaling_sinv_aligment = 16
        mxfp8_scaling_sinv_alignment_padding = 256
        # cuBLAS workspace ptr must be 256 bytes aligned but JAX buffers are not
        # necessarily 256 bytes aligned, we add some padding to ensure alignment.
        workspace_size += workspace_alignment_padding
        if scaling_mode in (
            ScalingMode.DELAYED_TENSOR_SCALING.value,
            ScalingMode.CURRENT_TENSOR_SCALING.value,
        ):
            # For tensor scaling, each matrix has a single scale value, but it
            # needs to be aligned to 16 bytes for CUDA 12.9.1 and later.
            workspace_size += lhs_scale_inv_aval.size * tensor_scaling_sinv_aligment
            workspace_size += rhs_scale_inv_aval.size * tensor_scaling_sinv_aligment
        elif scaling_mode == ScalingMode.MXFP8_1D_SCALING.value:
            # We also pad scale_inv swizzle buffers size for 256 bytes alignment.
            workspace_size += lhs_scale_inv_aval.size + mxfp8_scaling_sinv_alignment_padding
            workspace_size += rhs_scale_inv_aval.size + mxfp8_scaling_sinv_alignment_padding
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
        lhs_contract = transpose_dims(lhs.data.ndim, lhs_contract, flatten_axis=lhs.flatten_axis)
    if rhs.data_layout == "T":
        rhs_contract = transpose_dims(rhs.data.ndim, rhs_contract, flatten_axis=rhs.flatten_axis)

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
                if get_quantize_config().FP8_2X_ACC_FPROP
                else jax.lax.Precision.DEFAULT
            )
            return _jax_gemm_tensor_scaling_fp8(lhs, rhs, dim_nums, precision)

        if lhs.scaling_mode == ScalingMode.MXFP8_1D_SCALING:
            return _jax_gemm_mxfp8_1d(lhs, rhs, dim_nums)

        raise NotImplementedError(f"Unsupported ScalingMode: {lhs.scaling_mode}")

    lhs_q, rhs_q = _quantize_gemm_operands(lhs, rhs, lhs_quantizer, rhs_quantizer, contracting_dims)

    if isinstance(lhs_q, ScaledTensor) and isinstance(rhs_q, ScaledTensor):
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
    lhs: Union[jnp.ndarray, AbstractBaseTensor],
    rhs: Union[jnp.ndarray, AbstractBaseTensor],
    contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((-1,), (0,)),
    lhs_quantizer: Quantizer = None,
    rhs_quantizer: Quantizer = None,
    transpose_batch_sequence: bool = False,
    cgemm_config: CollectiveGemmConfig = noop_cgemm_config,
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
    contracting_dims: Tuple[Sequence[int], Sequence[int]], default = ((-1, ), (0, ))
        Tuple of sequences representing the contracting dimensions of the operands.
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
    transpose_batch_sequence: bool, default = False
        Transpose the batch and sequence dimensions of the input tensor.
    cgemm_config: CollectiveGemmConfig, default = noop_cgemm_config
        Helper object that manages comm+GEMM overlap options.

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
    if isinstance(lhs, NoScaleTensor):
        lhs = lhs.data
    if isinstance(rhs, NoScaleTensor):
        rhs = rhs.data

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
            "`jax.lax.dot_general` and `jax.nn.scaled_matmul` backends used when the custom cuBLAS "
            "GEMM primitive is disabled."
        )
        assert kwargs.get("gelu_input", None) is None and not fuse_bias, (
            "TE GEMM was invoked with GeLU fusion options that are not supported by the "
            "`jax.lax.dot_general` and `jax.nn.scaled_matmul` backends used when the custom cuBLAS "
            "GEMM primitive is disabled."
        )
        assert cgemm_config.collective_op.is_none, "JAX GEMM does not support collective GEMM"
        return _jax_gemm(lhs, rhs, contracting_dims, lhs_quantizer, rhs_quantizer)

    outputs = _te_gemm(
        lhs,
        rhs,
        lhs_quantizer=lhs_quantizer,
        rhs_quantizer=rhs_quantizer,
        contracting_dims=contracting_dims,
        transpose_batch_sequence=transpose_batch_sequence,
        cgemm_config=cgemm_config,
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
