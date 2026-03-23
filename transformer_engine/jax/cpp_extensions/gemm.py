# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX te modules"""

import math
import operator
import os
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial, reduce, cache
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
    initialize_cgemm_communicator,
    get_cgemm_num_max_streams,
    get_grouped_gemm_setup_workspace_size,
)

from .base import BasePrimitive, register_primitive
from .quantization import grouped_quantize
from ..quantize import (
    AbstractBaseTensor,
    NoScaleTensor,
    ScaledTensor,
    ScaledTensor1x,
    ScaledTensor2x,
    GroupedScaledTensor1x,
    ScalingMode,
    Quantizer,
    GroupedQuantizer,
    QuantizerSet,
    noop_quantizer_set,
    is_fp8_gemm_with_all_layouts_supported,
    apply_padding_to_scale_inv,
    QuantizeLayout,
)
from .misc import get_padded_spec, is_all_reduce_in_float32
from ..sharding import (
    global_mesh_resource,
    tpsp_axis_size,
    dp_or_fsdp_axis_size,
)


__all__ = [
    "CollectiveOp",
    "CollectiveOpSet",
    "collective_gemm_bootstrap",
    "noop_collective_op_set",
    "gemm",
    "grouped_gemm_copy_group_sizes",
    "grouped_gemm",
    "sanitize_dims",
    "get_non_contracting_dims",
    "transpose_dims",
]


num_cublas_streams = get_num_compute_streams()

# Cache whether the CUDA-graphable grouped GEMM implementation is available at import time.
# Calling get_grouped_gemm_setup_workspace_size raises a RuntimeError mentioning "cublas" when
# compiled against cuBLAS < 13.2, in which case the cuda-graphable path is unavailable.
try:
    get_grouped_gemm_setup_workspace_size(1)
    _v2_grouped_gemm_available = True
except RuntimeError as e:
    if "cublas" in str(e).lower():
        _v2_grouped_gemm_available = False
    else:
        raise


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
            or lhs_quantizer.scaling_mode.is_nvfp4_scaling
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
            or rhs_quantizer.scaling_mode.is_nvfp4_scaling
        )
        flatten_axis = min(rhs_cdims) if rhs_is_transposed else max(rhs_cdims) + 1
        rhs_q = rhs_quantizer.quantize(
            rhs,
            is_rowwise=not need_rhs_colwise,
            is_colwise=need_rhs_colwise,
            flatten_axis=flatten_axis,
        )

    if isinstance(lhs_q, ScaledTensor2x):
        raise TypeError(
            "Expected lhs_q to not be ScaledTensor2x after quantization, but got"
            f" type={type(lhs_q)}"
        )
    if isinstance(rhs_q, ScaledTensor2x):
        raise TypeError(
            "Expected rhs_q to not be ScaledTensor2x after quantization, but got"
            f" type={type(rhs_q)}"
        )

    def has_rht_applied(q: AbstractBaseTensor) -> bool:
        return isinstance(q, ScaledTensor1x) and q.has_rht_applied

    if has_rht_applied(lhs_q) != has_rht_applied(rhs_q):
        raise ValueError(
            "With NVFP4_1D_SCALING, if one operand is quantized with RHT, the other must be"
            " quantized with RHT as well. This is to ensure the RHT is applied to both and will"
            " cancel out in the GEMM."
        )

    return lhs_q, rhs_q


def _get_nvfp4_tensor_scale_inv(amax):
    DATA_DTYPE_MAX = jnp.finfo(jnp.float4_e2m1fn.dtype).max.astype(jnp.float32)
    SCALE_DTYPE_MAX = jnp.finfo(jnp.float8_e4m3fn.dtype).max.astype(jnp.float32)
    return amax / (DATA_DTYPE_MAX * SCALE_DTYPE_MAX)


def collective_gemm_bootstrap(
    num_total_devices,
    num_devices_per_process,
    process_id,
    tensor_parallel_size,
    num_max_streams=3,
    compute_stream_priority=0,
    communication_stream_priority=0,
    num_sm_for_communication=2,
    use_ce=True,
    aggregate_all_gather=False,
):
    """Initialize NCCL communicators for Collective GEMM operations.

    This function sets up the distributed communication infrastructure needed for
    tensor parallel collective GEMM operations. It supports two main scenarios:

    1. **Multi-device per process**: TP domain = single process
       - Each process manages multiple GPUs (num_devices_per_process > 1)
       - TP group consists of GPUs within the same process
       - Example: 2 processes × 4 GPUs each = 8 total ranks, tp_size=4

    2. **Single device per process**: TP domain spans multiple processes
       - Each process manages one GPU (num_devices_per_process = 1)
       - TP group spans across multiple processes
       - Example: 8 processes × 1 GPU each = 8 total ranks, tp_size=4

    Args:
        num_total_devices (int): Total number of ranks across all processes.
            Must be divisible by num_devices_per_process.
        num_devices_per_process (int): Number of GPUs per process.
            - For multi-device: equals tp_size (e.g., 4 GPUs per process)
            - For single-device: equals 1 (1 GPU per process)
        process_id (int): Process identifier (0-based).
            Must be in range [0, num_total_devices // num_devices_per_process).
        tensor_parallel_size (int): Size of tensor parallel groups.
            Must divide num_total_devices evenly.
        num_max_streams (int, optional): Maximum number of CUDA streams for overlap.
            Higher values enable more parallelism but use more GPU resources. Default: 3.
        compute_stream_priority (int, optional): Priority for GEMM computation streams.
            Lower values = higher priority. Range: 0 (highest) to 3 (lowest). Default: 0.
        communication_stream_priority (int, optional): Priority for NCCL communication streams.
            Lower values = higher priority. Range: 0 (highest) to 3 (lowest). Default: 0.
        num_sm_for_communication (int, optional): Number of streaming multiprocessors
            reserved for communication operations. Default: 2.
        use_ce (bool, optional): Enable CUDA copy engines for memory transfers.
            Can improve performance by offloading memory operations. Default: True.
        aggregate_all_gather (bool, optional): Aggregate multiple small all-gather operations
            into larger ones for better efficiency. Default: False.

    Raises:
        AssertionError: If num_total_devices is not divisible by num_devices_per_process,
            or if process_id is out of valid range.
        AssertionError: If num_devices_per_process is not 1 (Temporary: only single device per process is supported for now)
        RuntimeError: If NCCL initialization fails or if configuration
            is invalid (e.g., insufficient GPUs).

    Example:
        # Basic initialization (single device per process)
        collective_gemm_bootstrap(
            num_total_devices=8,
            num_devices_per_process=1,
            process_id=0,
            tensor_parallel_size=4
        )

        # Advanced configuration with custom performance settings
        collective_gemm_bootstrap(
            num_total_devices=8,
            num_devices_per_process=1,
            process_id=0,
            tensor_parallel_size=4,
            num_max_streams=5,                    # More parallelism
            compute_stream_priority=1,            # Lower compute priority
            communication_stream_priority=0,      # Higher comm priority
            num_sm_for_communication=4,           # More SMs for communication
            use_ce=True,                         # Enable copy engines
            aggregate_all_gather=True            # Aggregate small operations
        )

    Note:
        This function must be called after JAX distributed initialization
        and before any collective GEMM operations. Each process should call
        this function with its own unique process_id.
    """

    if not (num_devices_per_process == 1 and jax.local_device_count() == 1):
        raise RuntimeError("Only single device per process is supported at the moment!")
    if num_total_devices % num_devices_per_process != 0:
        raise ValueError(
            f"Invalid num_total_devices={num_total_devices},"
            f" num_devices_per_process={num_devices_per_process}"
        )
    if not 0 <= process_id < num_total_devices:
        raise ValueError(f"Invalid process_id={process_id}")
    initialize_cgemm_communicator(
        num_total_devices,
        num_devices_per_process,
        process_id,
        tensor_parallel_size,
        num_max_streams,
        compute_stream_priority,
        communication_stream_priority,
        num_sm_for_communication,
        use_ce,
        aggregate_all_gather,
    )


class CollectiveOp(Enum):
    "Enum for Collective Type in Collective GEMM"

    NONE = JAXX_Collective_Op.NONE
    ALL_GATHER = JAXX_Collective_Op.ALL_GATHER
    REDUCE_SCATTER = JAXX_Collective_Op.REDUCE_SCATTER

    @property
    def is_all_gather(self) -> bool:
        """Check if AllGather"""
        return self == CollectiveOp.ALL_GATHER

    @property
    def is_reduce_scatter(self) -> bool:
        """Check if ReduceScatter"""
        return self == CollectiveOp.REDUCE_SCATTER

    @property
    def is_none(self) -> bool:
        """Check if None"""
        return self == CollectiveOp.NONE


@dataclass(frozen=True)
class CollectiveOpSet:
    """
    A set of CollectiveOp objects that provide complementary collective GEMM configurations for the Forward and Backward passes through Dense-layers.
    """

    forward: CollectiveOp
    backward: CollectiveOp

    @staticmethod
    def create(forward_collective_op: CollectiveOp):
        """Create a set of CollectiveOp for forward and backward passes"""
        if forward_collective_op.is_all_gather:
            backward_collective_op = CollectiveOp.REDUCE_SCATTER
        elif forward_collective_op.is_reduce_scatter:
            backward_collective_op = CollectiveOp.ALL_GATHER
        else:
            backward_collective_op = CollectiveOp.NONE
        return CollectiveOpSet(forward=forward_collective_op, backward=backward_collective_op)


noop_collective_op_set = CollectiveOpSet.create(forward_collective_op=CollectiveOp.NONE)


@partial(jax.jit, static_argnums=(1, 2))
def swizzled_scale(scale_inv, flatten_axis, is_colwise):
    "Swizzle scale_inv via JAX transpose ops"
    original_shape = scale_inv.shape
    shape_2d = (math.prod(original_shape[:flatten_axis]), math.prod(original_shape[flatten_axis:]))
    if is_colwise:
        scale_inv = jnp.transpose(scale_inv.reshape(shape_2d))
        cols, rows = shape_2d
    else:
        rows, cols = shape_2d
    reshape = scale_inv.reshape(rows // 128, 4, 32, cols // 4, 4)
    swizzled = jnp.transpose(reshape, (0, 3, 2, 1, 4))
    return swizzled.reshape(original_shape)


def get_lhs_axis_boundary(lhs_cdims, is_transposed):
    """Get the axis boundary for the LHS operand."""
    return max(lhs_cdims) + 1 if is_transposed else min(lhs_cdims)


def get_rhs_axis_boundary(rhs_cdims, is_transposed):
    """Get the axis boundary for the RHS operand."""
    return min(rhs_cdims) if is_transposed else max(rhs_cdims) + 1


@cache
def _get_high_precision_accumulation_from_env() -> bool:
    """Read NVTE_FP8_GEMM_HIGH_PRECISION_ACCUMULATION once per process (cached)."""
    return os.getenv("NVTE_FP8_GEMM_HIGH_PRECISION_ACCUMULATION", "0") == "1"


def assert_cublas_requirements(scaling_mode, contracting_size, tensor_name):
    """Assert that the given tensor shape and layout meet the requirements for cuBLAS GEMM."""
    if scaling_mode != ScalingMode.NO_SCALING:
        # Requirements from https://docs.nvidia.com/cuda/cublas/#tensor-core-usage
        alignment = 32 if scaling_mode.is_nvfp4_scaling else 16

        if contracting_size % alignment != 0:
            raise ValueError(
                f"cuBLAS GEMM {tensor_name} tensor's contracting dimension must be a multiple of"
                f" {alignment} when using quantized inputs. Got contracting_size={contracting_size}"
            )


def _reorder_tpsp_leading(tensor, original_shape):
    """Reorder tensor so the tpsp axis is leading: reshape (dp, n, tpsp, m, ...), transpose (2, 0, 1, 3, ...)."""
    assert original_shape[0] % dp_or_fsdp_axis_size() == 0 or original_shape[0] == 1, (
        f"Original_shape[0]={original_shape[0]} is not divisible by"
        f" dp_or_fsdp_axis_size()={dp_or_fsdp_axis_size()}"
    )
    assert original_shape[1] % tpsp_axis_size() == 0 or original_shape[1] == 1, (
        f"Original_shape[1]={original_shape[1]} is not divisible by"
        f" tpsp_axis_size()={tpsp_axis_size()}"
    )
    reshaped = tensor.reshape(
        dp_or_fsdp_axis_size(),
        int(original_shape[0] / dp_or_fsdp_axis_size()),
        tpsp_axis_size(),
        int(original_shape[1] / tpsp_axis_size()),
        *original_shape[2:],
    )
    reordered = reshaped.transpose(2, 0, 1, 3, *range(4, reshaped.ndim))
    return reordered.reshape(original_shape)


def _reorder_dp_leading(tensor, original_shape):
    """Reorder tensor so the dp axis is leading: reshape (tpsp, dp, n, m, ...), transpose (1, 2, 0, 3, ...)."""
    assert original_shape[0] % dp_or_fsdp_axis_size() == 0 or original_shape[0] == 1, (
        f"Original_shape[0]={original_shape[0]} is not divisible by"
        f" dp_or_fsdp_axis_size()={dp_or_fsdp_axis_size()}"
    )
    assert original_shape[1] % tpsp_axis_size() == 0 or original_shape[1] == 1, (
        f"Original_shape[1]={original_shape[1]} is not divisible by"
        f" tpsp_axis_size()={tpsp_axis_size()}"
    )
    reshaped = tensor.reshape(
        tpsp_axis_size(),
        dp_or_fsdp_axis_size(),
        int(original_shape[0] / dp_or_fsdp_axis_size()),
        int(original_shape[1] / tpsp_axis_size()),
        *original_shape[2:],
    )
    reordered = reshaped.transpose(1, 2, 0, 3, *range(4, reshaped.ndim))
    return reordered.reshape(original_shape)


class GemmPrimitive(BasePrimitive):
    """
    Primitive for cuBLAS GEMM
    """

    name = "te_gemm_v2_ffi"
    multiple_results = True
    impl_static_args = (7, 8, 9, 10, 11, 12, 13, 14)
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        lhs,
        lhs_scale_inv,
        rhs,
        rhs_scale_inv,
        bias,
        alpha,
        beta,
        out_dtype,
        contracting_dims,
        scaling_mode,
        use_split_accumulator,
        transpose_batch_sequence,
        sequence_dim,
        is_outer,
        collective_op,
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
        if not _dims_are_consecutive(lhs_contracting_dims):
            raise ValueError(
                "cuBLAS GEMM expected consecutive contracting dimensions for LHS operand, but got "
                f"{lhs_contracting_dims}."
            )
        if not _dims_are_consecutive(rhs_contracting_dims):
            raise ValueError(
                "cuBLAS GEMM expected consecutive contracting dimensions for RHS operand, but got "
                f"{rhs_contracting_dims}."
            )

        lhs_contracting_size, rhs_contracting_size = map(
            lambda shape, dims: reduce(operator.mul, [shape[dim] for dim in dims]),
            (lhs.shape, rhs.shape),
            (lhs_contracting_dims, rhs_contracting_dims),
        )
        if lhs_contracting_size != rhs_contracting_size:
            raise ValueError(
                f"cuBLAS GEMM operands have incompatible contracting dimensions: {lhs.shape} @ idx"
                f" {lhs_contracting_dims} X {rhs.shape} @ idx {rhs_contracting_dims}."
            )
        assert_cublas_requirements(scaling_mode, lhs_contracting_size, "LHS")
        assert_cublas_requirements(scaling_mode, rhs_contracting_size, "RHS")

        lhs_is_transposed, rhs_is_transposed = _get_gemm_layout(operand_ndims, contracting_dims)
        if scaling_mode != ScalingMode.NO_SCALING:
            if not (
                scaling_mode.is_nvfp4_scaling or _compatible_fp8_gemm_dtypes(lhs.dtype, rhs.dtype)
            ):
                raise ValueError(
                    "cuBLAS GEMM quantized operands have incompatible data types: "
                    f"{lhs.dtype} x {rhs.dtype}."
                )
            if not (lhs_scale_inv.size > 0 and rhs_scale_inv.size > 0):
                raise ValueError(
                    "Quantized cuBLAS GEMM requires inverse scaling factors for both operands."
                )
            if (
                scaling_mode != ScalingMode.MXFP8_1D_SCALING
                and not is_fp8_gemm_with_all_layouts_supported()
            ):
                if lhs_is_transposed or not rhs_is_transposed:
                    raise ValueError(
                        "cuBLAS FP8 GEMM on devices with compute capability < 10.0 (Hopper) "
                        "require non-transposed LHS and transposed RHS operands "
                        "(`contracting_dims=((-1, ), (-1, ))`)."
                    )
        else:
            if lhs.dtype != rhs.dtype:
                raise ValueError(
                    "For TE cuBLAS GEMM for non-quantized inputs, the operand dtypes must be equal."
                    f" LHS dtype != RHS dtype, lhs.dtype={lhs.dtype}, rhs.dtype={rhs.dtype}"
                )

        # Determine output shape and dtype
        if not dtypes.canonicalize_dtype(out_dtype).itemsize > 1:
            raise ValueError("cuBLAS GEMM custom op does not support 8-bit quantized output types.")
        lhs_non_contracting_shape, rhs_non_contracting_shape = map(
            lambda shape, dims: [shape[dim] for dim in range(len(shape)) if dim not in dims],
            (lhs.shape, rhs.shape),
            (lhs_contracting_dims, rhs_contracting_dims),
        )
        out_shape = (*lhs_non_contracting_shape, *rhs_non_contracting_shape)
        output = jax.core.ShapedArray(shape=out_shape, dtype=out_dtype)

        # Adjust output shape for comm+GEMM overlap
        if not collective_op.is_none and not is_outer:  # Inner abstract
            if sequence_dim != 1:
                raise ValueError(f"Invalid sequence_dim. Got sequence_dim={sequence_dim}")
            overlap_out_shape = list(out_shape).copy()
            if collective_op.is_all_gather:
                overlap_out_shape[1] *= tpsp_axis_size()
            else:  # RS
                overlap_out_shape[sequence_dim] = (
                    overlap_out_shape[sequence_dim] // tpsp_axis_size()
                )
            if out_dtype != jnp.bfloat16:
                raise ValueError(f"Unsupported out_dtype={out_dtype}")
            output = jax.core.ShapedArray(shape=overlap_out_shape, dtype=out_dtype)

        # Validate bias when present (bias.size > 0 means fuse bias)
        if bias.size > 0:
            if bias.shape != tuple(rhs_non_contracting_shape):
                raise ValueError(
                    "cuBLAS GEMM bias tensor has incorrect shape, "
                    f"expected ({tuple(rhs_non_contracting_shape)}, ) but found {bias.shape}."
                )
            if bias.dtype != out_dtype:
                raise ValueError(
                    "cuBLAS GEMM bias tensor has incorrect data type, "
                    f"expected {out_dtype} but found {bias.dtype}."
                )

        if alpha.size != 1 or alpha.dtype != jnp.float32:
            raise ValueError(
                f"Expected alpha to be a single float32 scalar, but got alpha.size={alpha.size},"
                f" alpha.dtype={alpha.dtype}"
            )
        if beta.size != 1 or beta.dtype != jnp.float32:
            raise ValueError(
                f"Expected beta to be a single float32 scalar, but got beta.size={beta.size},"
                f" beta.dtype={beta.dtype}"
            )

        # Declare cuBLAS workspace
        workspace_size = get_cublas_workspace_size_bytes()
        # NVFP4 swizzling happen in via nvte kernel instead of JAX transposes
        if scaling_mode.is_nvfp4_scaling:
            workspace_size += lhs_scale_inv.size + rhs_scale_inv.size
        if not collective_op.is_none:
            workspace_size *= get_cgemm_num_max_streams()
        # cuBLAS workspace ptr must be 256 bytes aligned but JAX buffers are not
        # necessarily 256 bytes aligned, we add some padding to ensure alignment.
        workspace_size += 256
        workspace = jax.core.ShapedArray(shape=(workspace_size,), dtype=jnp.uint8)

        return output, workspace

    @staticmethod
    def outer_abstract(*args, **kwargs):
        output, _ = GemmPrimitive.abstract(*args, **kwargs)
        return (output,)

    @staticmethod
    def lowering(
        ctx,
        lhs,
        lhs_scale_inv,
        rhs,
        rhs_scale_inv,
        bias,
        alpha,
        beta,
        out_dtype,
        contracting_dims,
        scaling_mode,
        use_split_accumulator,
        transpose_batch_sequence,
        sequence_dim,
        is_outer,
        collective_op,
    ):
        del out_dtype, transpose_batch_sequence, sequence_dim, is_outer

        lhs_aval, _, rhs_aval, *_ = ctx.avals_in
        lhs_cdims, rhs_cdims = map(sanitize_dims, (lhs_aval.ndim, rhs_aval.ndim), contracting_dims)
        lhs_transposed, rhs_transposed = _get_gemm_layout(
            (lhs_aval.ndim, rhs_aval.ndim), (lhs_cdims, rhs_cdims)
        )

        args = (lhs, lhs_scale_inv, rhs, rhs_scale_inv, bias, alpha, beta)
        kwargs = {
            "scaling_mode": int(scaling_mode.value),
            "collective_op": int(collective_op.value),
            "lhs_axis_boundary": get_lhs_axis_boundary(lhs_cdims, lhs_transposed),
            "rhs_axis_boundary": get_rhs_axis_boundary(rhs_cdims, rhs_transposed),
            "lhs_transposed": lhs_transposed,
            "rhs_transposed": rhs_transposed,
            "use_split_accumulator": use_split_accumulator,
        }

        return jax.ffi.ffi_lowering(GemmPrimitive.name)(ctx, *args, config=kwargs)

    @staticmethod
    def impl(
        lhs,
        lhs_scale_inv,
        rhs,
        rhs_scale_inv,
        bias,
        alpha,
        beta,
        out_dtype,
        contracting_dims,
        scaling_mode,
        use_split_accumulator,
        transpose_batch_sequence,
        sequence_dim,
        is_outer,
        collective_op,
    ):
        if scaling_mode.is_1d_block_scaling():
            lhs_cdims, rhs_cdims = map(sanitize_dims, (lhs.ndim, rhs.ndim), contracting_dims)
            lhs_transposed, rhs_transposed = _get_gemm_layout(
                (lhs.ndim, rhs.ndim), (lhs_cdims, rhs_cdims)
            )
            lhs_flatten_axis = max(lhs_cdims) + 1 if lhs_transposed else min(lhs_cdims)
            rhs_flatten_axis = min(rhs_cdims) if rhs_transposed else max(rhs_cdims) + 1

            if not collective_op.is_none and not is_outer:
                # MXFP8 + Collective AG/RS: both sides of flatten_axis must be multiples of 128.
                # No padding is needed in this case
                lhs_first, lhs_last = math.prod(lhs.shape[:lhs_flatten_axis]), math.prod(
                    lhs.shape[lhs_flatten_axis:]
                )
                assert lhs_first % 128 == 0 and lhs_last % 128 == 0, (
                    "MXFP8 + Collective AG/RS requires LHS dimensions before and after the flatten"
                    f" axis to be multiples of 128. Got lhs.shape={lhs.shape},"
                    f" lhs_flatten_axis={lhs_flatten_axis}"
                )
                rhs_first, rhs_last = math.prod(rhs.shape[:rhs_flatten_axis]), math.prod(
                    rhs.shape[rhs_flatten_axis:]
                )
                assert rhs_first % 128 == 0 and rhs_last % 128 == 0, (
                    "MXFP8 + Collective AG/RS requires LHS dimensions before and after the flatten"
                    f" axis to be multiples of 128. Got rhs.shape={rhs.shape},"
                    f" rhs_flatten_axis={rhs_flatten_axis}"
                )
                # The scale needs to be in good shape for reordering
                assert lhs_scale_inv.shape[sequence_dim] % tpsp_axis_size() == 0, (
                    "MXFP8 + Collective AG/RS requires RHS scale inv sequence dimension to be"
                    f" multiples of tpsp_axis_size. Got lhs_scale_inv.shape={lhs_scale_inv.shape},"
                    f" tpsp_axis_size={tpsp_axis_size()}, sequence_dim={sequence_dim}"
                )
            else:
                lhs_scale_inv = apply_padding_to_scale_inv(
                    lhs_scale_inv,
                    scaling_mode,
                    lhs.shape,
                    lhs_transposed,
                    lhs_flatten_axis,
                )
                rhs_scale_inv = apply_padding_to_scale_inv(
                    rhs_scale_inv, scaling_mode, rhs.shape, not rhs_transposed, rhs_flatten_axis
                )

        # Only perform JAX-based swizzle for MXFP8, NVFP4 swizzle will go though nvte kernel
        if scaling_mode.is_mxfp8_scaling:
            lhs_scale_inv = swizzled_scale(lhs_scale_inv, lhs_flatten_axis, lhs_transposed)
            rhs_scale_inv = swizzled_scale(rhs_scale_inv, rhs_flatten_axis, not rhs_transposed)

        # Determine if we need to reorder the tensor so that the input/output are in the correct layout for the collective operation
        need_reorder = not transpose_batch_sequence and not is_outer and not collective_op.is_none

        # Alter lhs blocks so that CGEMM RS outputs correctly
        if need_reorder and collective_op.is_reduce_scatter and lhs.shape[0] != 1:
            assert sequence_dim == 1, f"Invalid sequence_dim. Got sequence_dim={sequence_dim}"
            lhs = _reorder_tpsp_leading(lhs, lhs.shape)

        if (
            need_reorder
            and (collective_op.is_reduce_scatter or collective_op.is_all_gather)
            and lhs_scale_inv.shape[0] != 1
            and scaling_mode.is_1d_block_scaling()
        ):
            assert sequence_dim == 1, f"Invalid sequence_dim. Got sequence_dim={sequence_dim}"
            lhs_scale_inv = _reorder_tpsp_leading(lhs_scale_inv, lhs_scale_inv.shape)

        (output, _) = GemmPrimitive.inner_primitive.bind(
            lhs,
            lhs_scale_inv,
            rhs,
            rhs_scale_inv,
            bias,
            alpha,
            beta,
            out_dtype=out_dtype,
            contracting_dims=contracting_dims,
            scaling_mode=scaling_mode,
            use_split_accumulator=use_split_accumulator,
            transpose_batch_sequence=transpose_batch_sequence,
            sequence_dim=sequence_dim,
            is_outer=is_outer,
            collective_op=collective_op,
        )
        # Alter output blocks for CGEMM AG
        if need_reorder and collective_op.is_all_gather and output.shape[0] != 1:
            assert sequence_dim == 1, f"Invalid sequence_dim. Got sequence_dim={sequence_dim}"
            output = _reorder_dp_leading(output, output.shape)

        return (output,)

    @staticmethod
    def outer_impl(
        lhs,
        lhs_scale_inv,
        rhs,
        rhs_scale_inv,
        bias,
        alpha,
        beta,
        out_dtype,
        contracting_dims,
        scaling_mode,
        use_split_accumulator,
        transpose_batch_sequence,
        sequence_dim,
        is_outer,
        collective_op,
    ):
        return GemmPrimitive.impl(
            lhs,
            lhs_scale_inv,
            rhs,
            rhs_scale_inv,
            bias,
            alpha,
            beta,
            out_dtype,
            contracting_dims,
            scaling_mode,
            use_split_accumulator,
            transpose_batch_sequence,
            sequence_dim,
            is_outer,
            collective_op,
        )

    @staticmethod
    def batcher(
        batched_args,
        batch_dims,
        out_dtype,
        contracting_dims,
        scaling_mode,
        use_split_accumulator,
        collective_op,
        transpose_batch_sequence,
        sequence_dim,
        is_outer,
    ):
        del transpose_batch_sequence, sequence_dim, is_outer
        if GemmPrimitive.outer_primitive is None:
            raise RuntimeError("GemmPrimitive.outer_primitive has not been registered")
        lhs_bdims, _, rhs_bdims, *_ = batch_dims

        # Batched GEMM is not supported
        if not (lhs_bdims is None and rhs_bdims is None):
            raise RuntimeError(
                f"Batching is not supported, got lhs_bdims={lhs_bdims}, rhs_bdims={rhs_bdims}"
            )
        out_bdims = (None,)

        return (
            GemmPrimitive.outer_primitive.bind(
                *batched_args,
                out_dtype=out_dtype,
                contracting_dims=contracting_dims,
                scaling_mode=scaling_mode,
                use_split_accumulator=use_split_accumulator,
                collective_op=collective_op,
                transpose_batch_sequence=transpose_batch_sequence,
                sequence_dim=sequence_dim,
                is_outer=is_outer,
            ),
            (out_bdims,),
        )

    @staticmethod
    def _parse_operand_output_specs(
        arg_infos,
        contracting_dims,
        transpose_batch_sequence,
        collective_op,
        scaling_mode,
    ):
        lhs_specs, _, rhs_specs, *_ = map(get_padded_spec, arg_infos)

        gsr = global_mesh_resource()

        # Ensure that tensor sequence parallelism is not used via setting tp_resource
        if gsr.tp_resource is not None:
            if gsr.tp_resource in lhs_specs:
                warnings.warn(
                    "Tensor sequence parallelism is detected as tp_resource='{gsr.tp_resource}'"
                    " appears in lhs_specs: {lhs_specs}. Please setting MeshResource.tpsp_resource"
                    " for tensor sequence parallelism to avoid potential issues."
                )

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
                    if reduce_spec is not None:
                        raise RuntimeError("Multiple reduce dimension is detected!")
                    reduce_spec = l

        sequence_dim = None

        # Find sequence dimension in lhs_specs if tensor sequence parallel is enabled
        # We only do CollectiveGemm AG on the x or dY thus they always the LHS and have sequence dim
        if collective_op.is_all_gather:
            try:
                tpsp_idx = lhs_specs.index(gsr.tpsp_resource)
            except ValueError as exc:
                raise ValueError(
                    f"tpsp_resource '{gsr.tpsp_resource}' is not found in lhs_specs: {lhs_specs}."
                    " Please check your sharding configuration."
                ) from exc
            sequence_dim = tpsp_idx
            if not (sequence_dim == 1) ^ transpose_batch_sequence:
                raise ValueError(
                    "CollectiveGEMM supports only (sequence_dim=1 and"
                    " transpose_batch_sequence=False) or (sequence_dim=0 and"
                    f" transpose_batch_sequence=True). Received: sequence_dim={sequence_dim},"
                    f" transpose_batch_sequence={transpose_batch_sequence}."
                )

        elif collective_op.is_reduce_scatter:
            if reduce_spec != gsr.tpsp_resource:
                raise ValueError(
                    "Only CollectiveGemm RS with the Reduction over the TPSP axis is supported! Got"
                    f" reduce_spec={reduce_spec}, tpsp_resource={gsr.tpsp_resource}"
                )
            sequence_dim = int(not transpose_batch_sequence)

        if reduce_spec is not None:
            # Other non-reduce cdims (if exists) need to be unsharded
            lhs_cspecs = tuple(s if s == reduce_spec else None for s in lhs_cspecs)
            # Only do AG Sequence dim if not Overlap
            if collective_op.is_all_gather:
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
                (
                    None
                    if spec is not None
                    and (
                        spec == gsr.fsdp_resource
                        or (isinstance(spec, tuple) and gsr.fsdp_resource in spec)
                    )
                    else spec
                )
                for spec in rhs_non_cspecs
            )

        # Only do AG Sequence dim if not Overlap
        if not collective_op.is_all_gather:
            # Non-contracting dims of LHS to be gathered along the SP axis.
            # Minor note: This causes MaxText TP (= Megatron TP + activation_hidden sharding) gathering x for
            # dW1 = x^T * dY1 which is unexpected. This is a known issue and no solution has found yet.
            lhs_non_cspecs = tuple(
                None if spec in rhs_non_cspecs else spec for spec in lhs_non_cspecs
            )

        out_specs = lhs_non_cspecs + rhs_non_cspecs

        # Only do AG Sequence dim if not Overlap RS
        if collective_op.is_all_gather:
            if sequence_dim > len(lhs_non_cspecs):
                raise ValueError(
                    f"Sequence dim {sequence_dim} is out of bounds for lhs_non_cspecs:"
                    f" {lhs_non_cspecs}"
                )
            out_specs = out_specs[:sequence_dim] + (None,) + out_specs[sequence_dim + 1 :]
        elif collective_op.is_reduce_scatter:
            if sequence_dim > len(lhs_non_cspecs):
                raise ValueError(
                    f"Sequence dim {sequence_dim} is out of bounds for lhs_non_cspecs:"
                    f" {lhs_non_cspecs}"
                )
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

        # Bias sharding is based on GEMM output before any scatter
        bias_specs = rhs_non_cspecs if arg_infos[4].size > 0 else (None,)  # bias is operand index 4

        # Scale shardings are based on the scaling_mode and collective_op
        lhs_scale_specs = rhs_scale_specs = (None,)
        if scaling_mode.is_1d_block_scaling():
            rhs_scale_specs = rhs_specs
            # Set the seq spec to None to trigger AG the scales as TE/Common CGEMM does not handle
            # scale collecting yet
            if collective_op.is_all_gather:
                lhs_scale_specs = tuple(
                    None if i == sequence_dim else s for i, s in enumerate(lhs_specs)
                )
            else:
                lhs_scale_specs = lhs_specs

        if not collective_op.is_none:
            if sequence_dim < 0:
                raise ValueError(f"Invalid sequence_dim. Got sequence_dim={sequence_dim}")

        return (
            (lhs_specs, lhs_scale_specs, rhs_specs, rhs_scale_specs, bias_specs),
            out_specs,
            reduce_spec,
            sequence_dim,
        )

    @staticmethod
    def infer_sharding_from_operands(
        out_dtype,
        contracting_dims,
        scaling_mode,
        use_split_accumulator,
        transpose_batch_sequence,
        sequence_dim,
        is_outer,
        collective_op,
        mesh,
        arg_infos,
        result_infos,
    ):
        del (
            out_dtype,
            use_split_accumulator,
            result_infos,
            is_outer,
            sequence_dim,
        )

        (_, out_specs, *_) = GemmPrimitive._parse_operand_output_specs(
            arg_infos,
            contracting_dims,
            transpose_batch_sequence,
            collective_op,
            scaling_mode,
        )
        out_sharding = NamedSharding(mesh, PartitionSpec(*out_specs))

        return (out_sharding,)

    @staticmethod
    def partition(
        out_dtype,
        contracting_dims,
        scaling_mode,
        use_split_accumulator,
        transpose_batch_sequence,
        sequence_dim,
        is_outer,
        collective_op,
        mesh,
        arg_infos,
        result_infos,
    ):
        del result_infos, is_outer, sequence_dim

        (
            (lhs_specs, lhs_scale_specs, rhs_specs, rhs_scale_specs, bias_input_specs),
            out_specs,
            reduce_spec,
            inferred_sequence_dim,
        ) = GemmPrimitive._parse_operand_output_specs(
            arg_infos,
            contracting_dims,
            transpose_batch_sequence,
            collective_op,
            scaling_mode,
        )

        # Block scale inverses match their operands, but tensor scale inverses are unsharded.
        none_sharding = NamedSharding(mesh, PartitionSpec(None))
        lhs_sharding = NamedSharding(mesh, PartitionSpec(*lhs_specs))
        lhs_scale_sharding = NamedSharding(mesh, PartitionSpec(*lhs_scale_specs))
        rhs_sharding = NamedSharding(mesh, PartitionSpec(*rhs_specs))
        rhs_scale_sharding = NamedSharding(mesh, PartitionSpec(*rhs_scale_specs))

        arg_shardings = (
            lhs_sharding,
            lhs_scale_sharding,
            rhs_sharding,
            rhs_scale_sharding,
        )

        # Bias
        arg_shardings += (NamedSharding(mesh, PartitionSpec(*bias_input_specs)),)

        # Alpha, beta
        arg_shardings += (none_sharding, none_sharding)

        # Assemble output shardings
        out_sharding = (NamedSharding(mesh, PartitionSpec(*out_specs)),)

        def _sharded_impl(lhs, lhs_scale_inv, rhs, rhs_scale_inv, bias, alpha, beta):
            # We should not fuse bias in the output reduction case
            has_bias = bias.size > 0
            fuse_bias = has_bias and reduce_spec is None
            bias_for_impl = bias if fuse_bias else jnp.empty(0, dtype=bias.dtype)
            (output,) = GemmPrimitive.impl(
                lhs,
                lhs_scale_inv,
                rhs,
                rhs_scale_inv,
                bias_for_impl,
                alpha,
                beta,
                out_dtype=out_dtype,
                contracting_dims=contracting_dims,
                scaling_mode=scaling_mode,
                use_split_accumulator=use_split_accumulator,
                transpose_batch_sequence=transpose_batch_sequence,
                sequence_dim=inferred_sequence_dim,
                is_outer=False,
                collective_op=collective_op,
            )

            if reduce_spec is not None:
                if not collective_op.is_reduce_scatter:
                    if is_all_reduce_in_float32():  # For unittest only
                        output = jax.lax.psum(output.astype(jnp.float32), reduce_spec).astype(
                            out_dtype
                        )
                    else:
                        output = jax.lax.psum(output, reduce_spec)

                if has_bias:
                    output += bias

            return (output,)

        return mesh, _sharded_impl, out_sharding, arg_shardings

    @staticmethod
    def shardy_sharding_rule(
        out_dtype,
        contracting_dims,
        scaling_mode,
        use_split_accumulator,
        transpose_batch_sequence,
        sequence_dim,
        is_outer,
        collective_op,
        mesh,
        operand_types,
        result_types,
    ):
        del out_dtype, use_split_accumulator
        del mesh, result_types, transpose_batch_sequence, sequence_dim, is_outer

        if not collective_op.is_none:
            warnings.warn(
                "CollectiveGEMM with Shardy propagation may produce an incorrect sharding pattern"
                " for the output.\n To resolve this, apply a sharding constraint on the output"
                " using one of the following options:\n"
                "  - TE `dense` vjp: set `output_axes`.\n"
                "  - TE `layernorm_mlp` vjp: set `dot_2_input_axes`.\n"
                "  - TE `transformer_engine.jax.cpp_extensions.gemm`: apply"
                " `jax.lax.with_sharding_constraint` on the output.\n"
                "  - TE via MaxText: no action needed.",
                UserWarning,
            )

        prefix = "Gemm_"

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
            lhs_scale_specs = lhs_specs
            rhs_scale_specs = rhs_specs

        lhs_non_cspec = tuple(lhs_specs[i] for i in range(operand_ndims[0]) if i not in lhs_cdims)
        rhs_non_cspec = tuple(rhs_specs[i] for i in range(operand_ndims[1]) if i not in rhs_cdims)
        out_spec = (*lhs_non_cspec, *rhs_non_cspec)
        bias_aval = operand_types[4]
        bias_spec = rhs_non_cspec if math.prod(bias_aval.shape) > 0 else ("…4",)
        alpha_spec = ("_5",)
        beta_spec = ("_6",)

        return SdyShardingRule(
            operand_mappings=(
                lhs_specs,
                lhs_scale_specs,
                rhs_specs,
                rhs_scale_specs,
                bias_spec,
                alpha_spec,
                beta_spec,
            ),
            result_mappings=(out_spec,),
        )


register_primitive(GemmPrimitive)


# TODO(Phuong): move this function down after GroupedGemmPrimitive after initial review. Keep it
# here for now to minimize line changes.
def _te_gemm(
    lhs: Union[jax.Array, ScaledTensor],
    rhs: Union[jax.Array, ScaledTensor],
    bias: jax.Array = None,
    lhs_quantizer: Quantizer = None,
    rhs_quantizer: Quantizer = None,
    contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((-1,), (0,)),
    use_split_accumulator: bool = False,
    transpose_batch_sequence: bool = False,
    collective_op: CollectiveOp = CollectiveOp.NONE,
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

    lhs_amax = rhs_amax = None
    # Extract GEMM custom op inputs from quantized operands
    if isinstance(lhs_q, ScaledTensor):
        if not isinstance(rhs_q, ScaledTensor) and rhs_quantizer is None:
            raise ValueError(
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
        lhs_amax = lhs_q.amax

    if isinstance(rhs_q, ScaledTensor):
        if not isinstance(lhs_q, ScaledTensor) and lhs_quantizer is None:
            raise ValueError(
                "cuBLAS GEMM with non-quantized LHS and quantized RHS operands requires a valid "
                "`Quantizer` object to quantize the LHS operand."
            )
        if isinstance(rhs_q, ScaledTensor2x):
            # Choose the quantization of the contracting dimension(s)
            rhs_q = rhs_q.get_rowwise_tensor() if rhs_is_transposed else rhs_q.get_colwise_tensor()
        if not (
            rhs_q.scaling_mode == lhs_q.scaling_mode
            or rhs_q.scaling_mode.is_nvfp4_scaling
            and lhs_q.scaling_mode.is_nvfp4_scaling
        ):
            raise ValueError(
                "cuBLAS GEMM quantized operands have mismatched scaling types, "
                f"LHS:{lhs_q.scaling_mode} x RHS:{rhs_q.scaling_mode}."
            )
        rhs_data = rhs_q.data
        rhs_scale_inv = rhs_q.scale_inv
        if rhs_q.data_layout == "T":
            rhs_cdims = transpose_dims(rhs_q.ndim, rhs_cdims, flatten_axis=rhs_q.flatten_axis)
        rhs_amax = rhs_q.amax

    alpha = jnp.ones((1,), jnp.float32)
    beta = jnp.zeros((1,), jnp.float32)
    if scaling_mode.is_nvfp4_scaling:
        if lhs_amax is None or rhs_amax is None:
            raise ValueError("NVFP4 scaling requires non-None amax for both LHS and RHS operands")
        lhs_tensor_scale_inv = _get_nvfp4_tensor_scale_inv(lhs_amax)
        rhs_tensor_scale_inv = _get_nvfp4_tensor_scale_inv(rhs_amax)
        alpha = lhs_tensor_scale_inv * rhs_tensor_scale_inv

    if not collective_op.is_none:
        assert not scaling_mode.is_nvfp4_scaling, (
            f"Collective GEMM is not yet supported with {scaling_mode} quantization. Only"
            " DELAYED_TENSOR_SCALING, CURRENT_TENSOR_SCALING, and MXFP8_1D_SCALING are supported."
        )

    out_dtype = lhs_q.dq_dtype if isinstance(lhs_q, ScaledTensor) else lhs_data.dtype
    if bias is None:
        bias = jnp.empty(0, dtype=out_dtype)

    (output,) = GemmPrimitive.outer_primitive.bind(
        lhs_data,
        lhs_scale_inv,
        rhs_data,
        rhs_scale_inv,
        bias,
        alpha,
        beta,
        out_dtype=out_dtype,
        contracting_dims=(lhs_cdims, rhs_cdims),
        scaling_mode=scaling_mode,
        use_split_accumulator=use_split_accumulator,
        transpose_batch_sequence=transpose_batch_sequence,
        sequence_dim=-1,  #  Dummy value and will be set in the primitive
        is_outer=True,
        collective_op=collective_op,
    )
    return output


class GroupedGemmCopySizesPrimitive(BasePrimitive):
    """
    Primitive for async copying group sizes from device to host
    """

    name = "te_grouped_gemm_d2h_group_sizes_ffi"
    multiple_results = False
    impl_static_args = (1,)
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        group_sizes_aval,
        *,
        num_gemms,
    ):
        del num_gemms
        out_aval = group_sizes_aval
        return out_aval

    @staticmethod
    def outer_abstract(*args, **kwargs):
        out = GroupedGemmCopySizesPrimitive.abstract(*args, **kwargs)
        return out

    @staticmethod
    def lowering(
        ctx,
        group_sizes,
        num_gemms,
    ):
        return jax.ffi.ffi_lowering(
            GroupedGemmCopySizesPrimitive.name,
            operand_output_aliases={0: 0},  # Mark num_gemms as the output
        )(
            ctx,
            group_sizes,
            num_gemms=num_gemms,
        )

    @staticmethod
    def impl(
        group_sizes,
        num_gemms,
    ):
        if GroupedGemmCopySizesPrimitive.inner_primitive is None:
            raise RuntimeError(
                "GroupedGemmCopySizesPrimitive.inner_primitive has not been registered"
            )
        out = GroupedGemmCopySizesPrimitive.inner_primitive.bind(
            group_sizes,
            num_gemms=num_gemms,
        )
        return out


register_primitive(GroupedGemmCopySizesPrimitive)


class GroupedGemmPrimitive(BasePrimitive):
    """
    Primitive for grouped GEMM using nvte_multi_tensor_gemm (supports all scaling modes) or nvte_grouped_gemm (supporting BF16).
    """

    # args = lhs_data, lhs_scale_inv, rhs_data, rhs_scale_inv, bias, group_sizes, group_offset, unused_placeholder
    name = "te_grouped_gemm_ffi"
    # args = lhs_data, lhs_scale_inv, rhs_data, rhs_scale_inv, bias, group_sizes, alpha, beta
    name_graph_safe = "te_grouped_gemm_v2_ffi"
    multiple_results = True
    impl_static_args = (8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)
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
        *additional_args,  # group_offset_aval, unused_placeholder OR alpha_aval, beta_aval
        M,
        N,
        K,
        lhs_is_trans,
        rhs_is_trans,
        scaling_mode,
        out_dtype,
        has_bias,
        is_grouped_dense_wgrad,
        use_async_d2h_group_sizes,
        use_v2_ffi,
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
            additional_args: Either
                * group_offsets: 1D array containing offsets for each group (not yet implemented)
                OR
                * alpha: 1D array of shape (G,) containing alpha values for each group
                * beta: 1D array of shape (G,) containing beta values for each group
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
        del lhs_data_aval, rhs_data_aval, bias_aval
        del K, lhs_is_trans, rhs_is_trans, has_bias, use_async_d2h_group_sizes

        num_groups = group_sizes_aval.size

        cublas_workspace_aval = jax.core.ShapedArray(
            shape=(
                GroupedGemmPrimitive._compute_cublas_workspace_size(
                    scaling_mode, lhs_scale_inv_aval, rhs_scale_inv_aval, use_v2_ffi
                ),
            ),
            dtype=jnp.uint8,
        )

        out_shape = (M, N)
        if is_grouped_dense_wgrad:
            out_shape = (num_groups, M, N)
        out_aval = jax.core.ShapedArray(shape=out_shape, dtype=out_dtype)

        if use_v2_ffi:
            setup_workspace_aval = jax.core.ShapedArray(
                shape=(get_grouped_gemm_setup_workspace_size(num_groups),), dtype=jnp.uint8
            )
            # Temporary buffer for int32 -> int64 conversion of group_sizes on device.
            int64_workspace_size = num_groups * jnp.dtype(jnp.int64).itemsize
            int64_workspace_aval = jax.core.ShapedArray(
                shape=(int64_workspace_size,), dtype=jnp.uint8
            )

            if len(additional_args) != 2:
                raise ValueError(
                    "Expected additional_args to contain alpha, beta for the graph-safe grouped"
                    f" GEMM primitive, but got {len(additional_args)} arguments."
                )
            alpha_aval, beta_aval = additional_args
            if alpha_aval.shape != (num_groups,):
                raise ValueError(f"Expected alpha shape {(num_groups,)}, got {alpha_aval.shape}")
            if alpha_aval.dtype != jnp.float32:
                raise ValueError(f"Expected alpha dtype float32, got {alpha_aval.dtype}")
            if beta_aval.shape != (num_groups,):
                raise ValueError(f"Expected beta shape {(num_groups,)}, got {beta_aval.shape}")
            if beta_aval.dtype != jnp.float32:
                raise ValueError(f"Expected beta dtype float32, got {beta_aval.dtype}")

            return (out_aval, cublas_workspace_aval, setup_workspace_aval, int64_workspace_aval)

        return (out_aval, cublas_workspace_aval)

    @staticmethod
    def _compute_cublas_workspace_size(
        scaling_mode: ScalingMode,
        lhs_scale_inv_aval,
        rhs_scale_inv_aval,
        use_v2_ffi: bool,
    ):
        """Compute the required cuBLAS workspace size based on the scaling mode and alignment requirements."""
        stream_count = 1 if use_v2_ffi else num_cublas_streams

        # TODO(Phuong): move some shape checks from Cpp to here
        workspace_size = get_cublas_workspace_size_bytes() * stream_count
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
        return workspace_size

    @staticmethod
    def outer_abstract(*args, **kwargs):
        (out, *_) = GroupedGemmPrimitive.abstract(*args, **kwargs)
        return (out,)

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
        use_async_d2h_group_sizes,
        use_v2_ffi,
    ):
        del out_dtype
        if use_v2_ffi:
            ffi_name = GroupedGemmPrimitive.name_graph_safe
            return jax.ffi.ffi_lowering(ffi_name)(
                ctx,
                *args,
                M=M,
                N=N,
                K=K,
                lhs_is_trans=lhs_is_trans,
                rhs_is_trans=rhs_is_trans,
                scaling_mode=scaling_mode.value,
                is_grouped_dense_wgrad=is_grouped_dense_wgrad,
            )
        ffi_name = GroupedGemmPrimitive.name
        return jax.ffi.ffi_lowering(ffi_name)(
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
            use_async_d2h_group_sizes=use_async_d2h_group_sizes,
        )

    @staticmethod
    def impl(
        lhs_data,
        lhs_scale_inv,
        rhs_data,
        rhs_scale_inv,
        bias,
        group_sizes,
        additional_arg_0,  # group_offset (non-graph-safe) OR alpha (graph-safe)
        additional_arg_1,  # unused placeholder (non-graph-safe) OR beta (graph-safe)
        M,
        N,
        K,
        lhs_is_trans,
        rhs_is_trans,
        scaling_mode,
        out_dtype,
        has_bias,
        is_grouped_dense_wgrad,
        use_async_d2h_group_sizes,
        use_v2_ffi,
    ):
        if GroupedGemmPrimitive.inner_primitive is None:
            raise RuntimeError("GroupedGemmPrimitive.inner_primitive has not been registered")
        if use_v2_ffi:
            additional_args = (additional_arg_0, additional_arg_1)
        else:
            additional_args = (additional_arg_0,)
        (out, *_) = GroupedGemmPrimitive.inner_primitive.bind(
            lhs_data,
            lhs_scale_inv,
            rhs_data,
            rhs_scale_inv,
            bias,
            group_sizes,
            *additional_args,
            M=M,
            N=N,
            K=K,
            lhs_is_trans=lhs_is_trans,
            rhs_is_trans=rhs_is_trans,
            scaling_mode=scaling_mode,
            out_dtype=out_dtype,
            has_bias=has_bias,
            is_grouped_dense_wgrad=is_grouped_dense_wgrad,
            use_async_d2h_group_sizes=use_async_d2h_group_sizes,
            use_v2_ffi=use_v2_ffi,
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
def _jax_scaled_matmul(
    lhs: ScaledTensor, rhs: ScaledTensor, dim_nums: Tuple[Tuple[Sequence[int], Sequence[int]]]
):
    """
    JAX GEMM for MXFP8 via scaled_matmul
    """
    if rhs.scaling_mode not in (
        ScalingMode.MXFP8_1D_SCALING,
        ScalingMode.NVFP4_1D_SCALING,
        ScalingMode.NVFP4_2D_SCALING,
    ):
        raise ValueError(
            "rhs does not have MXFP8 or NVFP4 scaling mode, got"
            f" rhs.scaling_mode={rhs.scaling_mode}"
        )

    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dim_nums

    expected_lhs_is_colwise = lhs_contract[-1] != lhs.data.ndim - 1
    expected_rhs_is_colwise = rhs_contract[-1] != rhs.data.ndim - 1
    if lhs.is_colwise is not expected_lhs_is_colwise:
        raise ValueError(
            f"LHS with unexpected quantize dimension.\nExpect is_colwise={expected_lhs_is_colwise},"
            f" got {lhs.is_colwise}"
        )
    if rhs.is_colwise is not expected_rhs_is_colwise:
        raise ValueError(
            f"RHS with unexpected quantize dimension.\nExpect is_colwise={expected_rhs_is_colwise},"
            f" got {rhs.is_colwise}"
        )

    if lhs.scaling_mode == ScalingMode.MXFP8_1D_SCALING:
        out_dtype = lhs.dq_dtype
        if not (lhs.data_layout == "N" and rhs.data_layout == "N"):
            raise ValueError(
                f"Got lhs.data_layout={lhs.data_layout}, rhs.data_layout={rhs.data_layout}"
            )
    else:
        if lhs.data_layout == "T":
            lhs_contract = transpose_dims(
                lhs.data.ndim, lhs_contract, flatten_axis=lhs.flatten_axis
            )
        if rhs.data_layout == "T":
            rhs_contract = transpose_dims(
                rhs.data.ndim, rhs_contract, flatten_axis=rhs.flatten_axis
            )
        out_dtype = jnp.float32

    # Reshape + Transpose (if needed)
    # [..., M, K] -> [1, reduce(..., M), K]
    # [..., K, M] -> [1, reduce(..., M), K]
    lhs_3d = _shape_normalization(lhs.data, (lhs_contract, lhs_batch), lhs.data_layout == "T")
    rhs_3d = _shape_normalization(rhs.data, (rhs_contract, rhs_batch), rhs.data_layout == "T")
    lhs_scale_3d = _shape_normalization(
        lhs.scale_inv, (lhs_contract, lhs_batch), lhs.data_layout == "T"
    )
    rhs_scale_3d = _shape_normalization(
        rhs.scale_inv, (rhs_contract, rhs_batch), rhs.data_layout == "T"
    )

    # JAX scaled_matmul only supports NT now (TN-gemm)
    # * Expected shape:
    # * lhs_data  (B, M, K)           * rhs_data  (B, N, K)
    # * lhs_scale (B, M, K_block)     * rhs_scale (B, N, K_block)
    out_3d = jax.nn.scaled_matmul(
        lhs_3d, rhs_3d, lhs_scale_3d, rhs_scale_3d, preferred_element_type=out_dtype
    )
    if lhs.scaling_mode.is_nvfp4_scaling:
        if lhs.amax is None or rhs.amax is None:
            raise ValueError("NVFP4 scaling requires non-None amax for both LHS and RHS operands")
        lhs_tensor_scale_inv = _get_nvfp4_tensor_scale_inv(lhs.amax)
        rhs_tensor_scale_inv = _get_nvfp4_tensor_scale_inv(rhs.amax)
        alpha = lhs_tensor_scale_inv * rhs_tensor_scale_inv
        out_3d = (out_3d * alpha).astype(lhs.dq_dtype)

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
    use_split_accumulator: bool = False,
) -> jnp.ndarray:
    """
    FP8 GEMM via JAX
    """
    dim_nums = (contracting_dims, ((), ()))

    def _jax_gemm_impl(lhs, rhs):
        if lhs.scaling_mode.is_tensor_scaling():
            if rhs.scaling_mode != lhs.scaling_mode:
                raise ValueError(
                    f"rhs.scaling_mode={rhs.scaling_mode} != lhs.scaling_mode={lhs.scaling_mode}"
                )

            precision = (
                jax.lax.Precision.HIGHEST if use_split_accumulator else jax.lax.Precision.DEFAULT
            )
            return _jax_gemm_tensor_scaling_fp8(lhs, rhs, dim_nums, precision)

        if lhs.scaling_mode.is_1d_block_scaling:
            return _jax_scaled_matmul(lhs, rhs, dim_nums)

        raise NotImplementedError(f"Unsupported ScalingMode: {lhs.scaling_mode}")

    lhs_q, rhs_q = _quantize_gemm_operands(lhs, rhs, lhs_quantizer, rhs_quantizer, contracting_dims)

    if isinstance(lhs_q, ScaledTensor) and isinstance(rhs_q, ScaledTensor):
        return _jax_gemm_impl(lhs_q, rhs_q)

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
    bias: jnp.ndarray = None,
    contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((-1,), (0,)),
    lhs_quantizer: Quantizer = None,
    rhs_quantizer: Quantizer = None,
    transpose_batch_sequence: bool = False,
    collective_op: CollectiveOp = CollectiveOp.NONE,
    **kwargs,
) -> Tuple[jnp.ndarray, ...]:
    r"""General matrix multiplication with optional quantization.

    Parameters
    ----------
    lhs: Union[jax.Array, ScaledTensor]
        Left-hand side operand in the matrix multiplication.
    rhs: Union[jax.Array, ScaledTensor]
        Right-hand side operand in the matrix multiplication.
    bias: jax.Array, default = None
        Optional additive bias term. When provided (non-empty), bias is added to the result of the Matrix Multiplication operation.
        This bias addition is fused when using the TE's custom call to cuBLAS GEMM.
    contracting_dims: Tuple[Sequence[int], Sequence[int]], default = ((-1, ), (0, ))
        Tuple of sequences representing the contracting dimensions of the operands.
    lhs_quantizer: Quantizer, default = None
        Object for down-casting the LHS operand for quantized GEMM.
    rhs_quantizer: Quantizer, default = None
        Object for down-casting the RHS operand for quantized GEMM.
    transpose_batch_sequence: bool, default = False
        Transpose the batch and sequence dimensions of the input tensor.
    collective_op: CollectiveOp, default = CollectiveOp.NONE
        Collective operation type for collective GEMM.

    Returns
    -------
    jax.Array:
        Result of the operation lhs * rhs + bias.
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

    # This option enable promoting some intermediate sums to higher precision when accumulating the result in
    # the cuBLAS GEMM kernel. Disabling this trades off numerical accuracy for speed.
    use_split_accumulator = _get_high_precision_accumulation_from_env()

    # Fall back on a native JAX implementation when the custom call to cuBLAS GEMM is disabled
    if not GemmPrimitive.enabled():
        if not collective_op.is_none:
            raise RuntimeError("JAX GEMM does not support collective GEMM")
        output = _jax_gemm(
            lhs, rhs, contracting_dims, lhs_quantizer, rhs_quantizer, use_split_accumulator
        )
        if bias is not None:
            output += bias  # Unfused
        return output

    output = _te_gemm(
        lhs,
        rhs,
        bias,
        lhs_quantizer=lhs_quantizer,
        rhs_quantizer=rhs_quantizer,
        contracting_dims=contracting_dims,
        use_split_accumulator=use_split_accumulator,
        transpose_batch_sequence=transpose_batch_sequence,
        collective_op=collective_op,
    )

    return output


def grouped_gemm_copy_group_sizes(
    group_sizes: jnp.ndarray,
    num_gemms: int,
) -> jnp.ndarray:
    """
    Async copy group sizes from device to host

    Args:
        group_sizes: 1D array containing the sizes of each group
        num_gemms: number of grouped gemm calls to be made
    """
    out = GroupedGemmCopySizesPrimitive.outer_primitive.bind(
        group_sizes,
        num_gemms=num_gemms,
    )
    return out


def _can_use_v2_grouped_gemm(
    scaling_mode: ScalingMode,
    dtype: jnp.dtype,
    has_bias: bool,
) -> bool:
    """Determine whether the cuda-graphable grouped GEMM implementation can be used based on the input parameters."""
    # Use the cuda-graphable path for plain BF16 non-quantized inputs; fall back to the legacy
    # nvte_multi_tensor_gemm path for all other cases (FP8, MXFP8, etc.) to stay
    # feature-compatible with the main branch.
    # Bias can be supported in a kernel or in pure-JAX in the future.

    if not _v2_grouped_gemm_available:
        return False

    # nvte_grouped_gemm (the v2 kernel) requires SM100+ (Blackwell or newer).
    # Fall back to the v1 path on SM90 (Hopper) and older architectures.
    if get_device_compute_capability(0) < 100:
        return False

    return scaling_mode == ScalingMode.NO_SCALING and dtype == jnp.bfloat16 and not has_bias


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
    use_async_d2h_group_sizes: bool = False,
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

    # TODO(Phuong): implement the precision
    del precision

    if isinstance(lhs, jnp.ndarray):
        if not isinstance(rhs, jnp.ndarray):
            raise TypeError(
                f"Expected rhs to be jnp.ndarray when lhs is jnp.ndarray, but got type={type(rhs)}"
            )
        out_dtype = lhs.dtype
        lhs_shape = lhs.shape
        rhs_shape = rhs.shape
        lhs_data = lhs
        rhs_data = rhs
        lhs_scale_inv = rhs_scale_inv = jnp.empty((0,), jnp.float32)
        scaling_mode = ScalingMode.NO_SCALING
    elif isinstance(lhs, GroupedScaledTensor1x):
        if not isinstance(rhs, GroupedScaledTensor1x):
            raise TypeError(
                "Expected rhs to be GroupedScaledTensor1x when lhs is GroupedScaledTensor1x, but"
                f" got type={type(rhs)}"
            )
        out_dtype = lhs.dq_dtype
        lhs_shape = lhs.original_shape
        rhs_shape = rhs.original_shape
        lhs_data = lhs.data
        rhs_data = rhs.data
        lhs_scale_inv = lhs.scale_inv
        rhs_scale_inv = rhs.scale_inv
        if lhs.scaling_mode != rhs.scaling_mode:
            raise ValueError(
                f"Mismatched scaling modes: lhs.scaling_mode={lhs.scaling_mode},"
                f" rhs.scaling_mode={rhs.scaling_mode}"
            )
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
        if not isinstance(quantizer_set.x, GroupedQuantizer):
            raise TypeError(
                "Expected quantizer_set.x to be GroupedQuantizer, but got"
                f" type={type(quantizer_set.x)}"
            )
        if type(quantizer_set.x) is not type(quantizer_set.kernel):
            raise TypeError(
                "Expected quantizer_set.x and quantizer_set.kernel to have the same type, but got"
                f" {type(quantizer_set.x)} and {type(quantizer_set.kernel)}"
            )
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

    if lhs_data.dtype == jnp.float8_e5m2 and rhs_data.dtype == jnp.float8_e5m2:
        raise ValueError("FP8 GEMM does not support E5M2 * E5M2")

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
        if lhs_is_trans != lhs_layout_is_T:
            raise RuntimeError("lhs input must be transposed before calling grouped_gemm")
        if (not rhs_is_trans) != rhs_layout_is_T:
            raise RuntimeError("rhs input must be transposed before calling grouped_gemm")
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
    if K_lhs != K_rhs:
        raise ValueError(
            f"Mismatched contracting dimensions: K_lhs={K_lhs}, K_rhs={K_rhs} (from"
            f" lhs_shape={lhs_shape}, rhs_shape={rhs_shape})"
        )
    M = math.prod(_calculate_remaining_shape(lhs_shape, lhs_contract_dim))
    N = math.prod(_calculate_remaining_shape(rhs_shape, rhs_contract_dim)[1:])  # Exclude G

    if is_grouped_dense_wgrad:
        N = math.prod(_calculate_remaining_shape(rhs_shape, rhs_contract_dim))
    else:
        if group_sizes.size != rhs_shape[0]:
            raise ValueError(
                "Expected group_sizes.size == rhs_shape[0], but got"
                f" group_sizes.size={group_sizes.size}, rhs_shape[0]={rhs_shape[0]}"
            )

    has_bias = bias is not None
    if has_bias and bias.shape != (group_sizes.size, N):
        raise ValueError(
            f"Expected bias.shape=({group_sizes.size}, {N}), but got bias.shape={bias.shape}"
        )
    bias = jnp.empty((), jnp.float32) if bias is None else bias

    if group_offset is not None:
        raise RuntimeError(
            "group_offset is not supported yet and is instead computed"
            " internally assuming contiguous grouping. Any padding is included in the group_sizes"
            " and padded with zeros to not affect the result of the MoE block."
        )

    use_v2_ffi = _can_use_v2_grouped_gemm(scaling_mode, lhs_data.dtype, has_bias)
    if use_v2_ffi:
        num_gemms = group_sizes.shape[0]
        additional_arg_0 = jnp.ones((num_gemms,), jnp.float32)  # alpha
        additional_arg_1 = jnp.zeros((num_gemms,), jnp.float32)  # beta
    else:
        additional_arg_0 = jnp.zeros((1,), jnp.int32)  # group_offset
        additional_arg_1 = jnp.zeros((0,), jnp.int32)  # unused placeholder

    (out,) = GroupedGemmPrimitive.outer_primitive.bind(
        lhs_data,
        lhs_scale_inv,
        rhs_data,
        rhs_scale_inv,
        bias,
        group_sizes,
        additional_arg_0,
        additional_arg_1,
        M=M,
        N=N,
        K=K_lhs,
        lhs_is_trans=lhs_is_trans,
        rhs_is_trans=rhs_is_trans,
        scaling_mode=scaling_mode.value,
        out_dtype=out_dtype,
        has_bias=has_bias,
        is_grouped_dense_wgrad=is_grouped_dense_wgrad,
        use_async_d2h_group_sizes=use_async_d2h_group_sizes,
        use_v2_ffi=use_v2_ffi,
    )
    return out
