# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX te modules"""

import math
import operator
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import partial, reduce
from typing import Tuple, Sequence, Union

import jax
import jax.numpy as jnp
from jax import dtypes
from jax.sharding import NamedSharding, PartitionSpec

import transformer_engine_jax as tex

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
    apply_padding_to_scale_inv,
    remove_padding_from_scale_inv,
)
from .misc import get_padded_spec, jax_dtype_to_te_dtype
from ..sharding import global_mesh_resource


__all__ = [
    "CommOverlapHelper",
    "CommOverlapHelperSet",
    "gemm",
    "grouped_gemm",
    "gemm_uses_jax_dot",
    "sanitize_dims",
    "get_non_contracting_dims",
    "transpose_dims",
]


num_cublas_streams = tex.get_num_compute_streams()

CUDA_STREAM_PRIORITY_LOWEST = None
CUDA_STREAM_PRIORITY_HIGHEST = None


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


@dataclass(frozen=True)
class CommOverlapHelper:
    """
    Helper object that carries comm+GEMM overlap configuration, initializes the internal
    communication buffer, and generates lowering arguments and partitioning rules for
    the GemmPrimitive.
    """
    # Core init arguments
    method: tex.CommOverlapMethod = field(default=tex.CommOverlapMethod.NONE)
    comm_type: tex.CommOverlapType = field(default=tex.CommOverlapType.NONE)
    buffer_shape: Sequence[int] = field(default=None)
    buffer_dtype: jnp.dtype = field(default=jnp.bfloat16)
    tp_size: int = field(default=None)

    # Userbuffers bootstrap kwargs
    num_splits: int = field(default=None, kw_only=True)
    num_max_streams: int = field(default=3, kw_only=True)
    comm_cga_size: int = field(default=None, kw_only=True)
    gemm_priority: int = field(default=None, kw_only=True)
    comm_priority: int = field(default=None, kw_only=True)
    num_comm_sm: int = field(default=None, kw_only=True)
    set_sm_margin: bool = field(default=None, kw_only=True)
    use_ce: bool = field(default=None, kw_only=True)
    atomic_gemm: bool = field(default=False, kw_only=True)
    rs_overlap_first_gemm: bool = field(default=False, kw_only=True)
    aggregate_ag: bool = field(default=False, kw_only=True)

    # Other kwargs not passed to Userbuffers
    tp_resource: str = field(default=None, kw_only=True)
    sp_resource: str = field(default=None, kw_only=True)
    output_all_gathered_lhs: bool = field(default=False, kw_only=True)
    flatten_axis: int = field(default=-1, kw_only=True)

    # Internal attributes
    is_enabled: bool = field(default=False, init=False, compare=True)
    unique_id: int = field(default=None, init=False, compare=False)
    sharded_impl: bool = field(default=False, init=False, compare=False)
    gather_dim: int = field(default=-2, init=False, compare=False)
    scatter_dim: int = field(default=-2, init=False, compare=False)

    def __post_init__(self):
        # Update global min/max CUDA stream priority values if not already done
        global CUDA_STREAM_PRIORITY_LOWEST, CUDA_STREAM_PRIORITY_HIGHEST
        if CUDA_STREAM_PRIORITY_LOWEST is None or CUDA_STREAM_PRIORITY_HIGHEST is None:
            (
                CUDA_STREAM_PRIORITY_LOWEST,
                CUDA_STREAM_PRIORITY_HIGHEST,
            ) = tex.get_stream_priority_range()

        object.__setattr__(self, "is_enabled", self.method != tex.CommOverlapMethod.NONE)
        if self.is_enabled:
            assert self.buffer_shape is not None, (
                f"CommOverlapHelper: {self.buffer_shape} is not a valid buffer shape."
            )
            assert self.comm_type != tex.CommOverlapType.NONE, (
                f"CommOverlapHelper: {self.comm_type} is not a valid collective type for "
                f"{self.method}."
            )
            assert self.tp_size % 2 == 0, (
                "CommOverlapHelper: Tensor-parallel axis size must be divisible by 2, got "
                f"{self.tp_size}."
            )
            if not self.is_bulk() and not self.is_p2p():
                # Pipelined overlap is only for reduce-scatter
                assert self.comm_type != tex.CommOverlapType.AG, (
                    f"CommOverlapHelper: {self.comm_type} is not a valid collective type for "
                    f"{self.method}."
                )

            # Collapse buffer shape to 2D
            if len(self.buffer_shape) > 2:
                if self.flatten_axis < 0:
                    object.__setattr__(self, "flatten_axis", self.flatten_axis + len(self.buffer_shape))
                object.__setattr__(
                    self,
                    "buffer_shape",
                    (
                        reduce(operator.mul, self.buffer_shape[ : self.flatten_axis]),
                        reduce(operator.mul, self.buffer_shape[self.flatten_axis : ])
                    )
                )

            # Num splits for P2P overlap is always fixed to TP size
            if self.is_p2p():
                object.__setattr__(self, "num_splits", self.tp_size)
            elif self.num_splits is None:
                object.__setattr__(self, "num_splits", self.tp_size)

            # Set conditional defaults for config options not specified at init time
            if self.comm_cga_size is None:
                object.__setattr__(self, "comm_cga_size", 1 if self.is_p2p() else 2)
            if self.num_comm_sm is None:
                object.__setattr__(self, "num_comm_sm", 1 if self.is_p2p() else 16)
            if self.set_sm_margin is None:
                object.__setattr__(self, "set_sm_margin", not self.is_p2p())
            if self.use_ce is None:
                object.__setattr__(self, "use_ce", self.is_p2p())
            if self.gemm_priority is None:
                object.__setattr__(self, "gemm_priority", CUDA_STREAM_PRIORITY_LOWEST)
            if self.comm_priority is None:
                object.__setattr__(self, "comm_priority", CUDA_STREAM_PRIORITY_HIGHEST)

            # Update mesh resources for tensor- and sequence-parallel dimensions
            if self.tp_resource is None:
                object.__setattr__(self, "tp_resource", global_mesh_resource().tp_resource)
            if self.sp_resource is None:
                object.__setattr__(self, "sp_resource", global_mesh_resource().cp_resource)

            # Allocate the communication buffer
            args, kwargs = self.get_bootstrap_args_kwargs()
            object.__setattr__(self, "unique_id", tex.create_comm_overlap_buffer(*args, **kwargs))

    def _set_sharded_impl(self, value):
        assert isinstance(value, bool)
        object.__setattr__(self, "sharded_impl", value)

    def _set_gather_dim(self, value):
        assert isinstance(value, int)
        object.__setattr__(self, "gather_dim", value)

    def _set_scatter_dim(self, value):
        assert isinstance(value, int)
        object.__setattr__(self, "scatter_dim", value)

    def is_bulk(self):
        """Check if this is a bulk overlap."""
        return self.method == tex.CommOverlapMethod.BULK

    def is_p2p(self):
        """Check if this is a peer-to-peer (ring-exchange) overlap."""
        return self.method == tex.CommOverlapMethod.RING_EXCHANGE

    def is_all_gather(self):
        """Check if the overlapped collective is an all-gather."""
        return self.comm_type == tex.CommOverlapType.AG

    def is_reduce_scatter(self):
        """Check if the overlapped collective is a reduce-scatter."""
        return self.comm_type == tex.CommOverlapType.RS

    def has_aux_output(self):
        """Check if the comm+GEMM overlap has an auxiliary output."""
        return (
            self.is_enabled
            and (self.is_bulk() or (self.is_all_gather() and self.output_all_gathered_lhs))
        )

    def get_bootstrap_args_kwargs(self):
        """Generate positional and keyword arguments to bootstrap Userbuffers."""
        args = (
            self.comm_type,
            self.method,
            self.buffer_shape,
            jax_dtype_to_te_dtype(self.buffer_dtype),
            self.tp_size,
        )
        kwargs = {
            "num_splits" : self.num_splits,
            "num_max_streams" : self.num_max_streams,
            "comm_cga_size" : self.comm_cga_size,
            "gemm_priority" : self.gemm_priority,
            "comm_priority" : self.comm_priority,
            "num_comm_sm" : self.num_comm_sm,
            "set_sm_margin" : self.set_sm_margin,
            "use_ce" : self.use_ce,
            "atomic_gemm" : self.atomic_gemm,
            "rs_overlap_first_gemm" : self.rs_overlap_first_gemm,
            "aggregate_ag" : self.aggregate_ag
        }
        return args, kwargs

    def get_lowering_kwargs(self):
        """Generate a dictionary of keyword arguments used in GemmPrimitive.lowering()."""
        aux_axis_boundary = -1
        if self.is_enabled and self.sharded_impl:
            if self.is_all_gather():
                assert self.gather_dim >= 0, (
                    "Internal TE error: CommOverlapHelper.gather_dim is not set correctly in "
                    "GemmPrimitive."
                )
                aux_axis_boundary = self.gather_dim + 1
            elif self.is_reduce_scatter():
                assert self.scatter_dim >= 0, (
                    "Internal TE error: CommOverlapHelper.scatter_dim is not set correctly in "
                    "GemmPrimitive."
                )
                aux_axis_boundary = self.scatter_dim + 1

        return {
            "comm_overlap_id" : self.unique_id,
            "comm_overlap_method" : int(self.method.value),
            "comm_type" : int(self.comm_type.value),
            "aux_axis_boundary" : aux_axis_boundary,
        }

    @staticmethod
    def _check_operand_specs(lhs_specs, rhs_specs, dimension_numbers):
        (lhs_cdims, rhs_cdims), (lhs_bdims, rhs_bdims) = dimension_numbers
        lhs_ndim, rhs_ndim = map(len, (lhs_specs, rhs_specs))

        def _split_specs(specs, contracting_dims, batch_dims):
            ndims = len(specs)
            cdims, bdims = map(sanitize_dims, (ndims, ndims), (contracting_dims, batch_dims))

            # Batch specs
            bspecs = tuple(specs[i] for i in bdims)

            # Non-batch leading dimension specs
            lspecs = tuple(specs[i] for i in range(ndims) if i not in cdims + bdims)

            # Non-batch contracting dimension specs
            cspecs = tuple(specs[i] for i in range(ndims) if i in cdims and i not in bdims)

            return bspecs, lspecs, cspecs

        (
            (lhs_bspecs, lhs_lspecs, lhs_cspecs),
            (rhs_bspecs, rhs_lspecs, rhs_cspecs),
        ) = map(
            _split_specs,
            (lhs_specs, rhs_specs),
            (lhs_cdims, rhs_cdims),
            (lhs_bdims, rhs_bdims),
        )

        # Batched dimensions must have the same sharding
        if len(lhs_bdims) > 0 and len(rhs_bdims) > 0:
            assert all(
                lhs_bspec == rhs_bspec for lhs_bspec, rhs_bspec in zip(lhs_bspecs, rhs_bspecs)
            ), (
                "cuBLAS GEMM operand batch dimensions must have the same sharding: "
                f"{lhs_specs} @ idx {lhs_bdims} x {rhs_specs} @ idx {rhs_bdims}."
            )

        # Only one each of the non-batched leading dimensions and non-batched contracting
        # dimensions can be sharded
        lhs_ldims, rhs_ldims = map(
            lambda ndim, exclude: tuple(dim for dim in range(ndim) if dim not in exclude),
            (lhs_ndim, rhs_ndim),
            (lhs_bdims + lhs_cdims, rhs_bdims + rhs_cdims),
        )
        (lhs_lspec_not_none, rhs_lspec_not_none, lhs_cspec_not_none, rhs_cspec_not_none) = map(
            lambda specs: tuple(spec for spec in specs if spec is not None),
            (lhs_lspecs, rhs_lspecs, lhs_cspecs, rhs_cspecs),
        )
        assert len(lhs_lspec_not_none) <= 1 and len(rhs_lspec_not_none) <= 1, (
            "cuBLAS GEMM operands can have only one sharded non-batched leading dimension: "
            f"{lhs_specs} @ idx {lhs_ldims} x {rhs_specs} @ idx {rhs_ldims}."
        )
        assert len(lhs_cspec_not_none) <= 1 and len(rhs_cspec_not_none) <= 1, (
            "cuBLAS GEMM operands can have only one sharded non-batched contracting dimension: "
            f"{lhs_specs} @ idx {lhs_cdims} x {rhs_specs} @ idx {rhs_cdims}."
        )

        # Extract single leading and contracting dimension specs
        (lhs_lspec, rhs_lspec, lhs_cspec, rhs_cspec) = map(
            lambda specs: None if len(specs) == 0 else specs[0],
            (lhs_lspec_not_none, rhs_lspec_not_none, lhs_cspec_not_none, rhs_cspec_not_none),
        )
        return (lhs_lspec, lhs_cspec), (rhs_lspec, rhs_cspec)

    def _get_no_overlap_rules(self, lhs_specs, rhs_specs, aux_in_specs, dimension_numbers):
        (lhs_cdims, rhs_cdims), (lhs_bdims, rhs_bdims) = dimension_numbers
        lhs_ndim, rhs_ndim = map(len, (lhs_specs, rhs_specs))

        (lhs_lspec, lhs_cspec), (rhs_lspec, rhs_cspec) = self._check_operand_specs(
            lhs_specs, rhs_specs, ((lhs_cdims, rhs_cdims), (lhs_bdims, rhs_bdims))
        )

        # Reproducing jax.nn.scaled_matmul() custom partitioning for arbitrary GEMM layouts
        # with row-wise LHS:(B, M, K1) and row-wise RHS:(B, N, K2) operands.
        # 1. K1 == K2 != None and N == None
        #    LHS: (B, M, K)
        #    RHS: (B, None, K)
        #    OUT: (B, M, None) --(AR)-> (B, M, None)
        # 2. K1 == K2 != None and M == N != None
        #    LHS: (B, M, K)
        #    RHS: (B, N, K)--(AG)->(B, None, K)
        #    OUT: (B, M, None) --(RS)--> (B, M, N)
        # 3. M == N
        #    LHS: (B, M, K)--(AG)->(B, M, None)
        #    RHS: (B, M, K)--(AG)->(B, None, None)
        #    OUT: (B, M, None)
        # 4. M != N
        #    LHS: (B, M, K)--(AG)->(B, M, None)
        #    RHS: (B, N, K)--(AG)->(B, N, None)
        #    OUT: (B, M, N)
        reduce_flag = lhs_cspec is not None and lhs_cspec == rhs_cspec
        all_reduce_output = reduce_flag and rhs_lspec is None
        reduce_scatter_output = reduce_flag and lhs_lspec is not None and lhs_lspec == rhs_lspec
        all_reduce_spec = reduce_scatter_spec = scatter_dim = None

        lhs_non_contracting_specs, rhs_non_contracting_specs = map(
            lambda specs, cdims: tuple(specs[i] for i in range(len(specs)) if i not in cdims),
            (lhs_specs, rhs_specs),
            (lhs_cdims, rhs_cdims),
        )
        out_specs = (*lhs_non_contracting_specs, *rhs_non_contracting_specs)
        if reduce_scatter_output:
            # All-gather (if necessary) the non-batch non-contracting dimension of RHS
            # LHS: (B, M, K)
            # RHS: (B, N, K) --(AG)-> (B, None, K)
            # OUT: (B, M, K) x (B, None, K)^T = (B, M, None) --(RS)-> (B, M, N)
            rhs_spec = tuple(
                rhs_spec[i] if i in set(rhs_bdims + rhs_cdims) else None for i in range(rhs_ndim)
            )
            reduce_scatter_spec = lhs_cspec
            scatter_dim = out_specs.index(rhs_lspec)

        elif all_reduce_output:
            # Set all output trailing dimensions to zero
            out_specs = (
                *lhs_non_contracting_specs,
                *[None for _ in range(len(rhs_non_contracting_specs))],
            )
            all_reduce_spec = lhs_cspec
        else:
            # All-gather (if necessary) the non-batch contracting dimensions
            # LHS: (B, M, K) --(AG)-> (B, M, None)
            # RHS: (B, N, K) --(AG)-> (B, N, None)
            # OUT: (B, M, None) x (B, N, None)^T = (B, M, N)
            lhs_specs = tuple(
                None if i in lhs_cdims and i not in lhs_bdims else lhs_specs[i]
                for i in range(lhs_ndim)
            )
            rhs_specs = tuple(
                None if i in rhs_cdims and i not in rhs_bdims else rhs_specs[i]
                for i in range(rhs_ndim)
            )
            # Check if RHS non-contracting spec also appears in the LHS non-contracting specs
            if rhs_lspec is not None and rhs_lspec in tuple(
                lhs_specs[i] for i in range(lhs_ndim) if i not in lhs_cdims
            ):
                # All-gather (if necessary) the non-batch non-contracting dimensions of RHS
                # LHS: (B, M, None)
                # RHS: (B, N, None) --(AG)-> (B, None, None)
                # OUT: (B, M, None) x (B, None, None)^T = (B, M, None)
                rhs_specs = tuple(
                    None if i not in set(rhs_bdims + rhs_cdims) else rhs_specs[i]
                    for i in range(rhs_ndim)
                )
                # Set all output trailing dimensions to zero
                out_specs = (
                    *lhs_non_contracting_specs,
                    *[None for _ in range(len(rhs_non_contracting_specs))],
                )

        # Bias and Pre-GeLU sharding is based on GEMM output
        bias_specs = out_specs[len(lhs_non_contracting_specs) :]
        gelu_specs = out_specs

        return (
            (lhs_specs, rhs_specs, bias_specs, gelu_specs, aux_in_specs),
            (out_specs, bias_specs, gelu_specs, (None, )),
            (all_reduce_spec, reduce_scatter_spec, scatter_dim),
        )

    def _get_bulk_overlap_rules(self, lhs_specs, rhs_specs, aux_in_specs, dimension_numbers):
        assert self.sp_resource in aux_in_specs, (
            "CommOverlapHelper: Auxiliary input for bulk all-gather overlap is not sharded "
            f"over the sequence-parallel mesh resource {self.sp_resource} in any dimension."
        )

        aux_out_specs = (None, )
        bulk_comm_dim = aux_in_specs.index(self.sp_resource)
        aux_in_specs_batch = aux_in_specs[ : bulk_comm_dim]
        aux_in_specs_tail = aux_in_specs[bulk_comm_dim + 1: ]
        if self.is_all_gather():
            assert all(spec is None for spec in aux_in_specs_tail), (
                "CommOverlapHelper: Trailing dimensions of the auxiliary input for bulk all-gather "
                "overlap cannot be sharded."
            )
            self._set_gather_dim(bulk_comm_dim)
            aux_out_specs = (
                *aux_in_specs_batch,
                None,  # all-gathered dimension
                *[None for _ in range(len(aux_in_specs_tail))]
            )
        else:
            assert all(spec is None for spec in aux_in_specs[bulk_comm_dim : ]), (
                "CommOverlapHelper: Non-batch dimensions of the auxiliary input for bulk "
                "reduce-scatter overlap cannot be sharded."
            )
            self._set_scatter_dim(bulk_comm_dim)
            aux_out_specs = (
                *aux_in_specs_batch,
                self.sp_resource,
                *[None for _ in range(len(aux_in_specs_tail))],
            )

        # GEMM is independent of communication so specs are as if there is no overlap
        operand_specs, output_specs, xla_reduce_info = self._get_specs_no_overlap(
            lhs_specs, rhs_specs, aux_in_specs, dimension_numbers
        )

        return (
            operand_specs,
            (*output_specs[:-1], aux_out_specs),
            xla_reduce_info,
        )


    def _get_all_gather_rules(self, lhs_specs, rhs_specs, aux_in_specs, dimension_numbers):
        contracting_dims, batch_dims = dimension_numbers
        lhs_ndim, rhs_ndim = map(len, (lhs_specs, rhs_specs))
        lhs_cdims, rhs_cdims, lhs_bdims, rhs_bdims = map(
            sanitize_dims, 2 * [lhs_ndim, rhs_ndim], contracting_dims + batch_dims
        )

        (lhs_lspec, _), _ = self._check_operand_specs(
            lhs_specs, rhs_specs, ((lhs_cdims, rhs_cdims), (lhs_bdims, rhs_bdims))
        )
        assert lhs_lspec == self.sp_resource, (
            "CommOverlapHelper: Non-batch leading dimension of the LHS operand for AG->GEMM "
            f"overlap must be sharded over the sequence-parallel mesh resource {self.sp_resource}, "
            f"but got {lhs_lspec} sharding instead."
        )

        # AG->GEMM overlap: Require non-batched contracting dimensions to be unsharded (e.g. FSDP)
        # LHS: (B, M, None)
        # RHS: (N, None)
        # OUT: (B, M, None) --(UB-AG)-> (B, None, None) x (N, None)^T = (B, None, N)
        lhs_specs = tuple(
            None if i in lhs_cdims and i not in lhs_bdims else lhs_specs[i] for i in range(lhs_ndim)
        )
        rhs_specs = tuple(
            None if i in rhs_cdims and i not in rhs_bdims else rhs_specs[i] for i in range(rhs_ndim)
        )

        # GEMM output spec keeps LHS batch spec and RHS non-contracting specs, but is None
        # in the non-batched leading dimensions.
        lhs_non_cspecs_gathered = list(
            lhs_specs[i] if i in lhs_bdims else None for i in range(lhs_ndim) if i not in lhs_cdims
        )
        rhs_non_cspecs = tuple(
            rhs_specs[i] for i in range(rhs_ndim) if i not in rhs_cdims
        )
        out_specs = (*lhs_non_cspecs_gathered, *rhs_non_cspecs)

        # Bias and Pre-GeLU sharding is based on GEMM output
        bias_specs = out_specs[len(lhs_non_cspecs_gathered) : ]
        gelu_specs = out_specs

        # Auxiliary input/output specs depend on bulk vs. non-bulk overlap
        aux_out_specs = (None, )
        if self.output_all_gathered_lhs:
            # Auxiliary output is the same as the LHS spec, except the gathered dimension unsharded
            self._set_gather_dim(lhs_specs.index(lhs_lspec))
            aux_out_specs = list(lhs_specs).copy()
            aux_out_specs[lhs_specs.index(lhs_lspec)] = None

        return (
            (lhs_specs, rhs_specs, bias_specs, gelu_specs, aux_in_specs),
            (out_specs, bias_specs, gelu_specs, aux_out_specs),
            (None, None, None),
        )

    def _get_reduce_scatter_rules(self, lhs_specs, rhs_specs, aux_in_specs, dimension_numbers):
        contracting_dims, batch_dims = dimension_numbers
        lhs_ndim, rhs_ndim = map(len, (lhs_specs, rhs_specs))
        lhs_cdims, rhs_cdims, lhs_bdims, rhs_bdims = map(
            sanitize_dims, 2 * [lhs_ndim, rhs_ndim], contracting_dims + batch_dims
        )

        (_, lhs_cspec), (_, rhs_cspec) = self._check_operand_specs(
            lhs_specs, rhs_specs, ((lhs_cdims, rhs_cdims), (lhs_bdims, rhs_bdims))
        )
        assert lhs_cspec == rhs_cspec == self.tp_resource, (
            "CommOverlapHelper: Non-batched contracting dimensions of LHS and RHS operands for "
            "GEMM->RS overlap must be sharded over the tensor-parallel resource "
            f"{self.tp_resource}, but got LHS:{lhs_cspec} and RHS:{rhs_cspec} sharding instead."
        )

        # GEMM->RS overlap: Require non-contracting non-batch dimensions to be unsharded (e.g. FSDP)
        # LHS: (B, M, K) --(XLA-AG)-> (B, None, K)
        # RHS: (N, K) --(XLA-AG)-> (None, K)
        # OUT: (B, None, K) x (B, None, K) = (B, None, None) --(UB-RS)-> (B, M, None)
        lhs_specs = tuple(
            None if i not in lhs_bdims + lhs_cdims else lhs_specs[i] for i in range(lhs_ndim)
        )
        rhs_specs = tuple(
            None if i not in rhs_bdims + rhs_cdims else rhs_specs[i] for i in range(rhs_ndim)
        )

        # GEMM output is the internal communication buffer, but we will use the XLA output buffer
        # as the final reduce-scattered output so we shard it accordingly here.
        lhs_specs_scattered = list(lhs_specs).copy()
        for i in range(lhs_ndim):
            if i not in lhs_bdims:
                # Update only the first non-batch leading dimension to the TP resource
                lhs_specs_scattered[i] = self.tp_resource
                break
        lhs_non_cspecs_scattered = tuple(
            lhs_specs_scattered[i] for i in range(lhs_ndim) if i not in lhs_cdims
        )
        rhs_non_cspecs = tuple(
            rhs_specs[i] for i in range(rhs_ndim) if i not in rhs_cdims
        )
        out_specs = (*lhs_non_cspecs_scattered, *rhs_non_cspecs)
        self._set_scatter_dim(out_specs.index(self.tp_resource))


        # Bias and Pre-GeLU sharding is based on GEMM output
        bias_specs = out_specs[len(lhs_non_cspecs_scattered) : ]
        gelu_specs = out_specs

        return (
            (lhs_specs, rhs_specs, bias_specs, gelu_specs, aux_in_specs),
            (out_specs, bias_specs, gelu_specs, (None, )),
            (None, None, None),
        )

    def get_partitioning_rules(self, lhs_specs, rhs_specs, aux_in_specs, dimension_numbers):
        """
        Correct operand specs to partititions suitable for the GemmPrimitive, and infer the
        partition specs of the outputs.
        """
        if self.is_bulk():
            return self._get_bulk_overlap_rules(
                lhs_specs, rhs_specs, aux_in_specs, dimension_numbers
            )

        impl_map = {
            tex.CommOverlapType.NONE : self._get_no_overlap_rules,
            tex.CommOverlapType.AG : self._get_all_gather_rules,
            tex.CommOverlapType.RS : self._get_reduce_scatter_rules,
        }
        return impl_map[self.comm_type](lhs_specs, rhs_specs, aux_in_specs, dimension_numbers)


@dataclass(frozen=True)
class CommOverlapHelperSet:
    """
    A set of CommOverlapHelper objects that provide complementary comm+GEMM overlap configurations
    for FPROP, DGRAD and WGRAD GEMMs in FWD/BWD passes through Dense-layers.
    """
    fprop: CommOverlapHelper = field(default=None)
    dgrad: CommOverlapHelper = field(default=None)
    wgrad: CommOverlapHelper = field(default=None)

    def _sanity_check(self):
        if not self.fprop.is_enabled:
            assert self.dgrad is None or not self.dgrad.is_enabled, (
                "CommOverlapHelperSet: Comm+GEMM overlap for DGRAD requires comm+GEMM overlap "
                "for FPROP to be enabled first."
            )
            assert self.wgrad is None or not self.wgrad.is_enabled, (
                "CommOverlapHelperSet: Comm+GEMM overlap for WGRAD requires comm+GEMM overlap "
                "for FPROP to be enabled first."
            )
            return

        assert not self.fprop.is_bulk(), (
            "CommOverlapHelperSet: Comm+GEMM overlap for FPROP does not support bulk collectives."
        )

        if self.fprop.is_all_gather():
            if self.dgrad is not None:
                if self.fprop.output_all_gathered_lhs:
                    assert not self.dgrad.is_enabled, (
                        "CommOverlapHelperSet: AG->GEMM FPROP does not have a corresponding DGRAD "
                        "overlap when it is configured to return a copy of the all-gathered LHS "
                        "operand as the auxiliary output."
                    )

                elif self.dgrad.is_enabled:
                    assert (
                        (self.dgrad.is_bulk() and self.dgrad.is_all_gather())
                        or (not self.dgrad_is_bulk() and self.dgrad.is_reduce_scatter())
                    ), (
                        "CommOverlapHelperSet: AG->GEMM FPROP requires DGRAD overlap to be either "
                        "BULK-AG or GEMM->RS."
                    )

            if self.wgrad is not None:
                if (
                    self.dgrad is not None
                    and self.dgrad.is_enabled
                    and self.dgrad.is_bulk()  # not checking all-gather because we enforced it above
                ):
                    assert (
                        self.wgrad.is_enabled
                        and self.wgrad.is_bulk()
                        and self.wgrad.is_reduce_scatter()
                    ), (
                        "CommOverlapHelperSet: AG->GEMM FPROP with BULK-AG DGRAD requires "
                        "WGRAD to overlap with BULK-RS."
                    )
                else:
                    assert not self.wgrad.is_enabled, (
                        "CommOverlapHelperSet: AG->GEMM FPROP does not have a corresponding WGRAD "
                        "overlap when DGRAD does not overlap with BULK-AG."
                    )

        elif self.fprop.is_reduce_scatter():
            if self.dgrad is not None and self.dgrad.is_enabled:
                assert not self.dgrad.is_bulk() and self.dgrad.is_all_gather(), (
                    "CommOverlapHelperSet: GEMM->RS overlap in FPROP requires DGRAD overlap to "
                    "be AG->GEMM."
                )

            if self.wgrad is not None:
                assert not self.wgrad.is_enabled, (
                    "CommOverlapHelperSet: GEMM->RS overlap in FPROP does not have a "
                    "corresponding WGRAD overlap."
                )

    def __post_init__(self):
        if self.fprop is None:
            object.__setattr__(self, "fprop", CommOverlapHelper())
            object.__setattr__(self, "dgrad", CommOverlapHelper())
            object.__setattr__(self, "wgrad", CommOverlapHelper())

        self._sanity_check()

        if self.fprop.is_enabled:
            # FWD/BWD paths with overlap:
            #
            # 1. AG->GEMM: (B, M, None) --(LHS AG)-> (B, None, None) x (None, N) = (B, None, N)
            #    DGRAD + Bulk-AG: (B, None, N) x (None, N)^T = (B, None, None)
            #                     (B, M, None) --(LHS bulk-AG)-> (B, None, None)
            #    WGRAD + Bulk-RS: (B, None, None)^T x (B, None, N) = (None, N)
            #                     (B, None, None) --(DGRAD bulk RS)-> (B, M, None)
            #
            # 2. GEMM->RS in FPROP: (B, None, K) x (K, None) = (B, None, None) --(RS)-> (B, M, None)
            #    AG->DGRAD: (B, M, None) --(GRAD AG)-> (B, None, None) x (K, None)^T = (B, None, K)
            #    WGRAD w/ AG-GRAD from DGRAD: (B, None, K)^T x (B, None, None) = (K, None)

            if self.dgrad is None:
                if self.fprop.is_all_gather() and self.fprop.output_all_gathered_lhs:
                    # If the AG->GEMM FPROP already saved the all-gathered LHS in the autograd
                    # context, we don't need to overlap a BULK-AG for it with DGRAD.
                    object.__setattr__(self, "dgrad", CommOverlapHelper())

                else:
                    # Otherwise, AG->GEMM FPROP needs BULK-AG DGRAD, and GEMM->RS FPROP needs
                    # AG->GEMM DGRAD w/ all-gathered gradient returned as auxiliary output to be
                    # re-used in WGRAD.
                    object.__setattr__(
                        self,
                        "dgrad",
                        CommOverlapHelper(
                            method=(
                                tex.CommOverlapMethod.BULK
                                if self.fprop.is_all_gather()
                                else tex.CommOverlapMethod.RING_EXCHANGE
                            ),
                            comm_type=tex.CommOverlapType.AG,
                            buffer_shape=self.fprop.buffer_shape,
                            buffer_dtype=self.fprop.buffer_dtype,
                            tp_size=self.fprop.tp_size,
                            tp_resource=self.fprop.tp_resource,
                            sp_resource=self.fprop.sp_resource,
                            output_all_gathered_lhs=self.fprop.is_reduce_scatter(),
                        )
                    )

            if self.wgrad is None:
                if (
                    self.fprop.is_all_gather()
                    and self.dgrad.is_enabled
                    and self.dgrad.is_bulk()
                    and self.dgrad.is_all_gather()
                ):
                    # If FPROP does AG->GEMM and DGRAD does BULK-AG, WGRAD needs to do a BULK-RS
                    object.__setattr__(
                        self,
                        "wgrad",
                        CommOverlapHelper(
                            method=tex.CommOverlapMethod.BULK,
                            comm_type=tex.CommOverlapType.RS,
                            buffer_shape=self.fprop.buffer_shape,
                            buffer_dtype=self.fprop.buffer_dtype,
                            tp_size=self.fprop.tp_size,
                            tp_resource=self.fprop.tp_resource,
                            sp_resource=self.fprop.sp_resource,
                        )
                    )

                else:
                    # Otherwise, WGRAD does not support comm+GEMM overlap
                    object.__setattr__(self, "wgrad", CommOverlapHelper())


class GemmPrimitive(BasePrimitive):
    """
    Primitive for cuBLAS GEMM
    """

    name = "te_gemm_ffi"
    multiple_results = True
    impl_static_args = (7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
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
        aux_in,
        out_dtype,
        dimension_numbers,
        lhs_quantized_colwise,
        rhs_quantized_colwise,
        scaling_mode,
        fuse_bias,
        fuse_gelu,
        grad,
        use_split_accumulator,
        comm_overlap,
    ):
        del lhs_quantized_colwise, rhs_quantized_colwise, use_split_accumulator

        # Sanity-check operand layouts and types
        operand_ndims = (lhs.ndim, rhs.ndim)
        contracting_dims, _ = dimension_numbers
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
        out_shape = [*lhs_non_contracting_shape, *rhs_non_contracting_shape]
        output = jax.core.ShapedArray(shape=out_shape, dtype=out_dtype)

        # Auxiliary output for comm+GEMM overlap
        aux_out_shape = (0, )
        aux_out_dtype = jnp.bfloat16
        if comm_overlap.is_enabled:
            if comm_overlap.is_bulk():
                # Bulk overlap will all-gather or reduce-scatter the tensor in the auxiliary input
                # and return the result of the collective in the auxiliary output
                assert aux_in.size > 0, (
                    "cuBLAS GEMM w/ bulk collective overlap requires an auxiliary input."
                )
                assert aux_in.ndim > 1, (
                    "cuBLAS GEMM w/ bulk collective overlap only supports multidimensional "
                    "auxiliary inputs."
                )

                aux_out_shape = list(aux_in.shape).copy()
                aux_out_dtype = aux_in.dtype
                if comm_overlap.sharded_impl:
                    if comm_overlap["comm_type"] == tex.CommOverlapType.AG:
                        aux_out_shape[comm_overlap.gather_dim] *= comm_overlap.tp_size
                    else:
                        assert aux_in.shape[comm_overlap.scatter_dim] % comm_overlap.tp_size, (
                            "cuBLAS GEMM w/ bulk reduce-scatter overlap requires the auxiliary "
                            "input to be divisible by tensor-parallel size in dimension index "
                            f"{comm_overlap.scatter_dim}."
                        )
                        aux_out_shape[comm_overlap.scatter_dim] = (
                            aux_out_shape[comm_overlap.scatter_dim] // comm_overlap.tp_size
                        )

            elif comm_overlap.is_all_gather():
                # Sharded abstract multiplies gathered dimension by TP size
                if comm_overlap.sharded_impl:
                    out_shape[comm_overlap.gather_dim] *= comm_overlap.tp_size
                    output = jax.core.ShapedArray(shape=out_shape, dtype=out_dtype)

                # AG->GEMM overlap can copy all-gathered LHS into the auxiliary buffer
                if comm_overlap.output_all_gathered_lhs:
                    aux_out_shape = list(lhs.shape).copy()
                    aux_out_dtype = lhs.dtype

                    # Sharded abstract multiplies gathered dimension by TP size
                    if comm_overlap.sharded_impl:
                        aux_out_shape[comm_overlap.gather_dim] *= comm_overlap.tp_size
            elif comm_overlap.is_reduce_scatter():
                # GEMM->RS auxiliary output is the reduce-scattered output
                rs_out_shape = list(out_shape).copy()

                # Sharded abstract divides scattered dimension by TP size
                if comm_overlap.sharded_impl:
                    rs_out_shape[comm_overlap.scatter_dim] = (
                        rs_out_shape[comm_overlap.scatter_dim] // comm_overlap.tp_size
                    )

                output = jax.core.ShapedArray(shape=rs_out_shape, dtype=out_dtype)

        aux_out = jax.core.ShapedArray(shape=aux_out_shape, dtype=aux_out_dtype)

        # Validate bias -- shape always depends on pure GEMM output even for GEMM->RS overlap
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

        # Validate pre-GeLU -- shape always depends on pure GEMM output even for GEMM->RS overlap
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
        if comm_overlap.is_enabled:
            workspace_size *= comm_overlap.num_max_streams

        # cuBLAS requires workspace pointers aligned to 256 bytes but XLA does not guarantee that
        # so we add to the size here and align the pointer in the C++ custom call.
        workspace_size += 256
        workspace = jax.core.ShapedArray(shape=(workspace_size,), dtype=jnp.uint8)

        return output, bias_grad, pre_gelu_out, aux_out, lhs_swizzle, rhs_swizzle, workspace

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
        aux_in,
        out_dtype,
        dimension_numbers,
        lhs_quantized_colwise,
        rhs_quantized_colwise,
        scaling_mode,
        fuse_bias,
        fuse_gelu,
        grad,
        use_split_accumulator,
        comm_overlap,
    ):
        del lhs_quantized_colwise, rhs_quantized_colwise, out_dtype
        contracting_dims, _ = dimension_numbers
        lhs_aval, _, rhs_aval, *_ = ctx.avals_in
        lhs_cdims, rhs_cdims = map(sanitize_dims, (lhs_aval.ndim, rhs_aval.ndim), contracting_dims)
        lhs_transposed, rhs_transposed = _get_gemm_layout(
            (lhs_aval.ndim, rhs_aval.ndim), (lhs_cdims, rhs_cdims)
        )

        args = (lhs, lhs_scale_inv, rhs, rhs_scale_inv, bias, gelu_input, aux_in)
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
        kwargs.update(comm_overlap.get_lowering_kwargs())

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
        aux_in,
        out_dtype,
        dimension_numbers,
        lhs_quantized_colwise,
        rhs_quantized_colwise,
        scaling_mode,
        fuse_bias,
        fuse_gelu,
        grad,
        use_split_accumulator,
        comm_overlap,

    ):
        lhs_cdims, rhs_cdims = map(sanitize_dims, (lhs.ndim, rhs.ndim), dimension_numbers[0])
        lhs_transposed, rhs_transposed = _get_gemm_layout(
            (lhs.ndim, rhs.ndim), (lhs_cdims, rhs_cdims)
        )

        lhs_scale_inv = apply_padding_to_scale_inv(
            lhs_scale_inv,
            scaling_mode,
            lhs.shape,
            is_colwise=lhs_quantized_colwise,
            flatten_axis=max(lhs_cdims) + 1 if lhs_transposed else min(lhs_cdims),
        )
        rhs_scale_inv = apply_padding_to_scale_inv(
            rhs_scale_inv,
            scaling_mode,
            rhs.shape,
            is_colwise=rhs_quantized_colwise,
            flatten_axis=min(rhs_cdims) if rhs_transposed else max(rhs_cdims) + 1,
        )

        outputs = GemmPrimitive.inner_primitive.bind(
            lhs,
            lhs_scale_inv,
            rhs,
            rhs_scale_inv,
            bias,
            gelu_input,
            aux_in,
            out_dtype=out_dtype,
            dimension_numbers=dimension_numbers,
            lhs_quantized_colwise=lhs_quantized_colwise,
            rhs_quantized_colwise=rhs_quantized_colwise,
            scaling_mode=scaling_mode,
            fuse_bias=fuse_bias,
            fuse_gelu=fuse_gelu,
            grad=grad,
            use_split_accumulator=use_split_accumulator,
            comm_overlap=comm_overlap,
        )
        return outputs[:-3]  # discard workspace arrays

    @staticmethod
    def batcher(
        batched_args,
        batch_dims,
        out_dtype,
        dimension_numbers,
        lhs_quantized_colwise,
        rhs_quantized_colwise,
        scaling_mode,
        fuse_bias,
        fuse_gelu,
        grad,
        use_split_accumulator,
        comm_overlap,
    ):
        assert GemmPrimitive.outer_primitive is not None
        lhs, _, rhs, *_, aux_in_bdims = batched_args
        lhs_bdims, _, rhs_bdims, *_ = batch_dims
        contracting_dims, batch_dims = dimension_numbers
        arg_lhs_bdims, arg_rhs_bdims = map(sanitize_dims, (lhs.ndim, rhs.ndim), batch_dims)
        arg_lhs_bdims = (None,) if len(arg_lhs_bdims) == 0 else arg_lhs_bdims
        assert all(bdim == arg_bdim for bdim, arg_bdim in zip(lhs_bdims, arg_lhs_bdims)), (
            "User-specified batch dimension(s) for cuBLAS GEMM LHS operand does not match batch "
            f"dimensions inferred by JAX/XLA, expected {lhs_bdims} but got {arg_lhs_bdims}."
        )
        arg_rhs_bdims = (None,) if len(arg_rhs_bdims) == 0 else arg_rhs_bdims
        assert all(bdim == arg_bdim for bdim, arg_bdim in zip(rhs_bdims, arg_rhs_bdims)), (
            "User-specified batch dimension(s) for cuBLAS GEMM RHS operand does not match batch "
            f"dimensions inferred by JAX/XLA, expected {lhs_bdims} but got {arg_lhs_bdims}."
        )

        # Output is batched like the non-contracting batch dimensions of the LHS operand
        lhs_cdims = sanitize_dims(lhs.ndim, contracting_dims)
        lhs_non_contracting_bdims = tuple(dim for dim in lhs_bdims if dim not in lhs_cdims)
        out_bdims = (None,) if len(lhs_non_contracting_bdims) == 0 else lhs_non_contracting_bdims

        # Bias gradient is never batched
        bias_bdims = (None,)

        # Pre-GeLU output, if exists, is batched like GEMM output
        pre_gelu_bdims = (None,)
        if fuse_gelu and not grad:
            pre_gelu_bdims = out_bdims

        aux_out_bdims = (None, )
        if comm_overlap.is_enabled:
            if comm_overlap.is_bulk():
                # Bulk overlap auxiliary output must have the same batch dims as the auxiliary input
                aux_out_bdims = aux_in_bdims
            elif comm_overlap.is_all_gather() and comm_overlap.output_all_gathered_lhs:
                # AG->GEMM overlap with all-gathered LHS output must have same batch dims as
                # sharded LHS input
                aux_out_bdims = arg_lhs_bdims

        return (
            GemmPrimitive.outer_primitive.bind(
                *batched_args,
                out_dtype=out_dtype,
                dimension_numbers=dimension_numbers,
                lhs_quantized_colwise=lhs_quantized_colwise,
                rhs_quantized_colwise=rhs_quantized_colwise,
                scaling_mode=scaling_mode,
                fuse_bias=fuse_bias,
                fuse_gelu=fuse_gelu,
                grad=grad,
                use_split_accumulator=use_split_accumulator,
                comm_overlap=comm_overlap,
            ),
            (out_bdims, bias_bdims, pre_gelu_bdims, aux_out_bdims),
        )

    @staticmethod
    def infer_sharding_from_operands(
        out_dtype,
        dimension_numbers,
        lhs_quantized_colwise,
        rhs_quantized_colwise,
        scaling_mode,
        fuse_bias,
        fuse_gelu,
        grad,
        use_split_accumulator,
        comm_overlap,
        mesh,
        arg_infos,
        result_infos,
    ):
        del (
            out_dtype,
            lhs_quantized_colwise,
            rhs_quantized_colwise,
            scaling_mode,
            grad,
        )
        del use_split_accumulator, result_infos

        lhs_specs, _, rhs_specs, *_, aux_in_specs = map(get_padded_spec, arg_infos)
        (
            _, (out_specs, bias_grad_specs, pre_gelu_specs, aux_out_specs), *_
        ) = comm_overlap.get_partitioning_rules(
            lhs_specs, rhs_specs, aux_in_specs, dimension_numbers
        )

        # Discard bias gradient and pre-GeLU output specs based on fusion choices
        if not fuse_bias:
            bias_grad_specs = (None,)
        if not fuse_gelu:
            pre_gelu_specs = (None,)

        # Assemble output shardings
        out_shardings = list(
            map(
                lambda specs: NamedSharding(mesh, PartitionSpec(*specs)),
                (out_specs, bias_grad_specs, pre_gelu_specs, aux_out_specs)
            )
        )

        return out_shardings

    @staticmethod
    def partition(
        out_dtype,
        dimension_numbers,
        lhs_quantized_colwise,
        rhs_quantized_colwise,
        scaling_mode,
        fuse_bias,
        fuse_gelu,
        grad,
        use_split_accumulator,
        comm_overlap,
        mesh,
        arg_infos,
        result_infos,
    ):
        del result_infos

        lhs_specs, _, rhs_specs, *_, aux_in_specs = map(get_padded_spec, arg_infos)
        (
            (lhs_specs, rhs_specs, bias_specs, gelu_input_specs, aux_in_specs),
            (out_specs, bias_grad_specs, pre_gelu_specs, aux_out_specs),
            (all_reduce_spec, reduce_scatter_spec, scatter_dim),
        ) = comm_overlap.get_partitioning_rules(
            lhs_specs, rhs_specs, aux_in_specs, dimension_numbers
        )

        # Block scale inverses match their operands, but tensor scale inverses are unsharded.
        lhs_scale_specs = (None, )
        rhs_scale_specs = (None, )
        if scaling_mode.is_1d_block_scaling() and not comm_overlap.is_enabled:
            lhs_scale_specs = lhs_specs
            rhs_scale_specs = rhs_specs

        # Discard bias and pre-GeLU specs based on fusion choices
        if not fuse_bias:
            bias_specs = (None,)
            bias_grad_specs = (None,)
        if not fuse_gelu:
            gelu_input_specs = (None,)
            pre_gelu_specs = (None,)

        # Assemble argument shardings
        arg_shardings = tuple(
            map(
                lambda specs: NamedSharding(mesh, PartitionSpec(*specs)),
                (
                    lhs_specs,
                    lhs_scale_specs,
                    rhs_specs,
                    rhs_scale_specs,
                    bias_specs,
                    gelu_input_specs,
                    aux_in_specs
                ),
            )
        )

        # Assemble output shardings
        out_shardings = list(
            map(
                lambda specs: NamedSharding(mesh, PartitionSpec(*specs)),
                (out_specs, bias_grad_specs, pre_gelu_specs, aux_out_specs),
            )
        )

        def _sharded_impl(lhs, lhs_scale_inv, rhs, rhs_scale_inv, bias, gelu_input, aux_in):
            comm_overlap._set_sharded_impl(True)
            outputs = GemmPrimitive.impl(
                lhs,
                lhs_scale_inv,
                rhs,
                rhs_scale_inv,
                bias,
                gelu_input,
                aux_in,
                out_dtype=out_dtype,
                dimension_numbers=dimension_numbers,
                lhs_quantized_colwise=lhs_quantized_colwise,
                rhs_quantized_colwise=rhs_quantized_colwise,
                scaling_mode=scaling_mode,
                fuse_bias=fuse_bias,
                fuse_gelu=fuse_gelu,
                grad=grad,
                use_split_accumulator=use_split_accumulator,
                comm_overlap=comm_overlap,
            )
            comm_overlap._set_sharded_impl(False)

            # All-Reduce/Reduce-Scatter GEMM output
            if all_reduce_spec is not None:
                outputs[0] = jax.lax.psum(outputs[0], all_reduce_spec)
                if fuse_gelu and not grad:
                    outputs[2] = jax.lax.psum(outputs[2], all_reduce_spec)
            elif reduce_scatter_spec is not None:
                outputs[0] = jax.lax.psum_scatter(
                    outputs[0], reduce_scatter_spec, scatter_dimension=scatter_dim, tiled=True
                )
                if fuse_gelu and not grad:
                    outputs[2] = jax.lax.psum_scatter(
                        outputs[2], reduce_scatter_spec, scatter_dimension=scatter_dim, tiled=True
                    )

            return outputs

        return mesh, _sharded_impl, out_shardings, arg_shardings

    @staticmethod
    def shardy_sharding_rule(*args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "TE cuBLAS GEMM custom op does not support the Shardy partitioner. You can disable the "
            'custom op by setting `NVTE_JAX_CUSTOM_CALLS_RE="^(?!GemmPrimitive$).+$"` in the '
            "environment, which will make GEMM operations in TE will execute with native "
            "`jax.lax.dot_general` and `jax.nn.scaled_matmul` calls."
        )


register_primitive(GemmPrimitive)


def gemm_uses_jax_dot() -> bool:
    """Check if the GEMM call directs to the TE custom cuBLAS call or native JAX dot."""
    return not GemmPrimitive.enabled()


def _get_scale_inv_without_padding(scaled_tensor):
    return remove_padding_from_scale_inv(
        scaled_tensor.scale_inv,
        scaled_tensor.scaling_mode,
        scaled_tensor.data.shape,
        is_colwise=scaled_tensor.is_colwise,
        flatten_axis=scaled_tensor.flatten_axis,
    )


def _te_gemm(
    lhs: Union[jax.Array, ScaledTensor],
    rhs: Union[jax.Array, ScaledTensor],
    bias: jax.Array = None,
    gelu_input: jax.Array = None,
    aux_in: jax.Array = None,
    lhs_quantizer: Quantizer = None,
    rhs_quantizer: Quantizer = None,
    dimension_numbers: Tuple[Tuple[Sequence[int], Sequence[int]]] = (((-1,), (0,)), ((), ())),
    fuse_bias: bool = False,
    fuse_gelu: bool = False,
    grad: bool = False,
    use_split_accumulator: bool = QuantizeConfig.FP8_2X_ACC_FPROP,
    comm_overlap: CommOverlapHelper = CommOverlapHelper(),
) -> Tuple[jax.Array, ...]:
    # Prepare non-quantized GEMM operands
    lhs_data = lhs
    rhs_data = rhs
    lhs_scale_inv = jnp.empty(0, dtype=jnp.float32)
    rhs_scale_inv = jnp.empty(0, dtype=jnp.float32)
    scaling_mode = ScalingMode.NO_SCALING
    contracting_dims, batch_dims = dimension_numbers
    lhs_is_transposed, rhs_is_transposed = _get_gemm_layout((lhs.ndim, rhs.ndim), contracting_dims)
    lhs_cdims, rhs_cdims = map(sanitize_dims, (lhs.ndim, rhs.ndim), contracting_dims)
    lhs_bdims, rhs_bdims = map(sanitize_dims, (lhs.ndim, rhs.ndim), batch_dims)

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
        lhs_scale_inv = _get_scale_inv_without_padding(lhs_q)
        if lhs_q.data_layout == "T":
            lhs_cdims = transpose_dims(lhs_q.ndim, lhs_cdims, flatten_axis=lhs_q.flatten_axis)
            lhs_bdims = transpose_dims(lhs_q.ndim, lhs_bdims, flatten_axis=lhs_q.flatten_axis)

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
        rhs_scale_inv = _get_scale_inv_without_padding(rhs_q)
        if rhs_q.data_layout == "T":
            rhs_cdims = transpose_dims(rhs_q.ndim, rhs_cdims, flatten_axis=rhs_q.flatten_axis)
            rhs_bdims = transpose_dims(rhs_q.ndim, rhs_bdims, flatten_axis=rhs_q.flatten_axis)

    # Dummy empties for bias, gelu and aux_in
    out_dtype = lhs_q.dq_dtype if isinstance(lhs_q, ScaledTensor) else lhs_data.dtype
    if bias is None or not (fuse_bias and not grad):
        bias = jnp.empty(0, dtype=out_dtype)
    if gelu_input is None or not (fuse_gelu and grad):
        gelu_input = jnp.empty(0, dtype=out_dtype)
    if aux_in is None or not comm_overlap.is_enabled:
        aux_in = jnp.empty(0, dtype=jnp.bfloat16)

    return GemmPrimitive.outer_primitive.bind(
        lhs_data,
        lhs_scale_inv,
        rhs_data,
        rhs_scale_inv,
        bias,
        gelu_input,
        aux_in,
        out_dtype=out_dtype,
        dimension_numbers=((lhs_cdims, rhs_cdims), (lhs_bdims, rhs_bdims)),
        lhs_quantized_colwise=lhs_q.is_colwise if isinstance(lhs_q, ScaledTensor) else False,
        rhs_quantized_colwise=rhs_q.is_colwise if isinstance(rhs_q, ScaledTensor) else False,
        scaling_mode=scaling_mode,
        fuse_bias=fuse_bias,
        fuse_gelu=fuse_gelu,
        grad=grad,
        use_split_accumulator=use_split_accumulator,
        comm_overlap=comm_overlap,
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
        lhs_contract = transpose_dims(lhs.data.ndim, lhs_contract, flatten_axis=lhs.flatten_axis)
        lhs_batch = transpose_dims(lhs.data.ndim, lhs_batch, flatten_axis=lhs.flatten_axis)
    if rhs.data_layout == "T":
        rhs_contract = transpose_dims(rhs.data.ndim, rhs_contract, flatten_axis=rhs.flatten_axis)
        rhs_batch = transpose_dims(rhs.data.ndim, rhs_batch, flatten_axis=rhs.flatten_axis)

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
    dimension_numbers: Tuple[Tuple[Sequence[int], Sequence[int]]] = (((-1,), (0,)), ((), ())),
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
    dimension_numbers: Tuple[Tuple[Sequence[int], Sequence[int]]], default = (((-1, ), (0, )), ((), ()))
        Tuple of two tuples of sequences representing the contracting and batched dimensions,
        respectively. The first sequence in each tuple represents the contracting/batched
        dimensions of the LHS operand, and the second sequence represents the contracting/batched
        dimensions of the RHS operand.
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
    comm_overlap: CommOverlapHelper, default = None
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
        return _jax_gemm(lhs, rhs, dimension_numbers[0], lhs_quantizer, rhs_quantizer)

    outputs = _te_gemm(
        lhs,
        rhs,
        lhs_quantizer=lhs_quantizer,
        rhs_quantizer=rhs_quantizer,
        dimension_numbers=dimension_numbers,
        **kwargs,
    )

    # Discard empty outputs
    grad = kwargs.get("grad", False)
    comm_overlap = kwargs.get("comm_overlap", CommOverlapHelper())
    clean_outputs = outputs[0]  # first output is the final result and is never empty
    if (fuse_bias and grad) or (fuse_gelu and not grad) or comm_overlap.has_aux_output():
        clean_outputs = (outputs[0],)
        if fuse_bias and grad:  # only return bias gradient if it exists
            clean_outputs += (outputs[1],)
        if fuse_gelu and not grad:  # only return pre-GeLU output if it exists
            clean_outputs += (outputs[2],)
        if comm_overlap.has_aux_output():
            # only return aux output for bulk overlap or non-bulk all-gather overlap
            # with gathered LHS output
            clean_outputs += (outputs[3],)
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
