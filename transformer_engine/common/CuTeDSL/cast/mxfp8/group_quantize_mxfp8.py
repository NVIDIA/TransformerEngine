# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Grouped MXFP8 quantization kernel implemented in CuTeDSL.

Strategy-aligned port of group_quantize_mxfp8.cuh. The scheduling, descriptor
management and per-tensor scale addressing mirror the CUDA kernel one-for-one:

  * `is_single_tensor` reps (SAME_BOTH_DIMS, VARYING_FIRST_DIM) launch ONE CTA per
    128x128 chunk and address the group through ONE static TMA descriptor with
    global block offsets -- the CUDA `tensor_map_*_static` "direct mapper" path.
    For SAME_BOTH_DIMS the CUDA grid is linearized per tensor (X, Y-in-tensor,
    tensor) while this kernel linearizes it flat over the stacked rows; both
    require every member's row count to be a multiple of CHUNK_DIM_Y, and under
    that precondition the two decode to the identical (block_offset_Y, block_id_X)
    for every block index.
  * the other reps launch grid=(workers_per_tensor, num_tensors) and bind
    tensor_id to blockIdx.y, so a CTA grid-strides only within its own tensor and
    never re-resolves which tensor a chunk belongs to. They get per-tensor
    descriptors written by a prologue kernel (the CuTeDSL analog of
    update_tma_descriptors filling g_tensor_maps) and acquired with a tensormap
    proxy fence.
  * per-tensor scale bases/strides follow the CUDA formulas:
        scales_* += is_single_tensor ? 0 : tensor_base / 32
        stride_rowwise = roundup(cols/32, 4)   stride_colwise = roundup(cols, 128)

Both kernels dropped the older flat persistent grid that strided across tensor
boundaries (CUDA's decode_job / advance_to_next_job, removed in #3483); the
per-tensor grid above is what replaced it on both sides.

Mechanics that provably yield the same bytes may differ: the mbarrier pipeline is
expressed with PipelineTmaAsync instead of hand-rolled mbarriers. As in CUDA, the
scales of out-of-bounds columns in a chunk (the scale-row padding) are written as 0.

Scope: everything group_quantize_mxfp8.cuh covers except 2D block scaling -- the
cast-noop flag, fused activation (IS_ACT) and activation derivative (IS_DACT), dbias,
compact and GEMM-swizzled scales, rowwise and/or colwise, and all four shape
representations. Differences from CUDA:
  * the grouped amax pointer is accepted and left untouched, as the CUDA kernel does;
  * GEMM-swizzled colwise scales are only produced for the single-tensor reps. For
    VARYING_LAST_DIM / VARYING_BOTH_DIMS the CUDA kernel adds the tensor base to the
    colwise swizzled index twice, which is only in bounds for the first member, so the
    C++ bridge falls back to CUDA instead of reproducing it;
  * dbias partial sums are accumulated in the CUDA kernel's order (a running column sum
    with colwise output, otherwise per-thread sums reduced across the CTA), and the C++
    bridge reduces the workspace with the same grouped_reduce_dbias.

Like the CUDA kernel, every member's first dim must be a multiple of 128 (and, for the
varying-last reps, its last dim too). The kernel prints the same diagnostics as
get_tensor_rows_num / get_tensor_cols_num when a group violates this and, like
NVTE_DEVICE_ERROR in a release build, carries on.

Measured, deliberately NOT changed:
  - sO_row and sO_col are both allocated unconditionally, where CUDA sizes only the
    direction in use. Sizing them conditionally does work -- ncu confirms the shared-memory
    occupancy limit goes 6 -> 9 CTAs/SM for a single-direction bf16 config -- but it is a
    small LOSS on GB200, not a win: rep_med_sbd bf16 colwise 54.9 -> 56.1 us, rowwise
    56.5 -> 57.0 us (fp32 rowwise gains ~1%). The kernel is DRAM-bandwidth-bound at
    ~6.3 TB/s, so extra resident CTAs only add contention. Verified by a control that kept
    the conditional code but padded SMEM back to the old size: timings returned exactly to
    the unconditional numbers, so the effect is the occupancy, not codegen. Revisit if a
    future variant (dbias / activation) makes this kernel latency- rather than
    bandwidth-bound.
"""

# pylint: disable=missing-class-docstring

import logging
import os
from typing import Optional, Type

import cutlass
from cutlass import cute
from cutlass import pipeline
from cutlass import Boolean, Float32, Int32, Int64, Float8E8M0FNU
from cutlass.cute.nvgpu import cpasync
from cutlass.utils import TensorMapManager, TensorMapUpdateMode
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module
import tvm_ffi

from transformer_engine.common.CuTeDSL.utils import (
    str_to_cutlass_dtype,
    device_compute_capability,
)
from transformer_engine.common.CuTeDSL.cast.mxfp8.quantize_mxfp8 import (
    MXFP8_BLOCK_SCALING_SIZE,
    SYM_N_DIVISIBILITY,
    FP8E4M3_MAX_NORM_RCP,
    FP8E5M2_MAX_NORM_RCP,
    SUPPORTED_ACTIVATIONS,
    SUPPORTED_DACTIVATIONS,
    derive_swizzled_scale_layout,
    noop_flag_is_set,
    quantize_rowwise_mxfp8,
    quantize_colwise_mxfp8,
)

CUTEDSL_DEBUG_LOGGING = os.environ.get("CUTEDSL_DEBUG_LOGGING", "0") == "1"
logger = logging.getLogger("transformer_engine.cutedsl.mxfp8")

THREADS_PER_WARP = 32
BYTES_PER_TENSORMAP = 128
# Descriptor slots per tensor: input, rowwise output, colwise output, activation input.
NUM_TENSORMAPS = 4
ACT_INPUT_SLOT = 3
# One extra slot holds per-tensor (rows, cols, base_elts), so the main kernel never has to
# binary-search the offsets array. Mirrors TensorMapStorage::rows/cols/offsets upstream.
META_SLOT = NUM_TENSORMAPS
NUM_WORKSPACE_SLOTS = NUM_TENSORMAPS + 1

# Shape representations, mirroring ShapeRepresentation in common/utils.cuh.
SAME_BOTH_DIMS = "same_both_dims"
VARYING_FIRST_DIM = "varying_first_dim"
VARYING_LAST_DIM = "varying_last_dim"
VARYING_BOTH_DIMS = "varying_both_dims"
SUPPORTED_SHAPE_REPS = (SAME_BOTH_DIMS, VARYING_FIRST_DIM, VARYING_LAST_DIM, VARYING_BOTH_DIMS)

# Upper bound on the group size (MAX_SUPPORTED_TENSOR_DESCRIPTORS in grouped_tma.cuh); sizes
# the fixed binary search over the offsets.
MAX_SUPPORTED_TENSORS = 64


class MXFP8GroupQuantizeConfig:
    """Compile-time config for the grouped MXFP8 quantize kernel."""

    def __init__(
        self,
        dtype: str,
        fp8_dtype: str,
        rowwise: bool,
        colwise: bool,
        shape_rep: str,
        with_gemm_swizzled_scales: bool = False,
        with_dbias: bool = False,
        with_dact: bool = False,
        with_act: bool = False,
        activation: str = "none",
    ):
        if dtype not in ("Float32", "Float16", "BFloat16"):
            raise ValueError(f"unknown input dtype {dtype!r}; expected Float32|Float16|BFloat16")
        self.DTYPE = str_to_cutlass_dtype(dtype)
        self.DTYPE_STR = dtype
        if fp8_dtype not in ("Float8E4M3", "Float8E5M2"):
            raise ValueError(
                f"unknown FP8 dtype {fp8_dtype!r}; expected 'Float8E4M3' or 'Float8E5M2'"
            )
        self.FP8_DTYPE = str_to_cutlass_dtype(fp8_dtype)
        self.FP8_DTYPE_STR = fp8_dtype
        if not (rowwise or colwise):
            raise ValueError("at least one of rowwise or colwise must be true")
        self.ROWWISE = rowwise
        self.COLWISE = colwise
        if shape_rep not in SUPPORTED_SHAPE_REPS:
            raise ValueError(
                f"unsupported shape representation {shape_rep!r}; expected one of"
                f" {SUPPORTED_SHAPE_REPS}"
            )
        self.SHAPE_REP = shape_rep
        # Mirrors `is_single_tensor` in group_quantize_mxfp8.cuh.
        self.IS_SINGLE_TENSOR = shape_rep in (SAME_BOTH_DIMS, VARYING_FIRST_DIM)
        self.MAX_NORM_RCP = (
            FP8E4M3_MAX_NORM_RCP if fp8_dtype == "Float8E4M3" else FP8E5M2_MAX_NORM_RCP
        )

        self.WITH_GEMM_SWIZZLED_SCALES = with_gemm_swizzled_scales
        if with_gemm_swizzled_scales and colwise and not self.IS_SINGLE_TENSOR:
            # The CUDA kernel offsets the colwise swizzled scales of these representations by
            # the tensor base twice, which only lands inside the buffer for the first member.
            raise ValueError(
                "GEMM-swizzled colwise scales are only supported for single-tensor representations"
            )
        if with_dbias and not self.IS_SINGLE_TENSOR:
            # mxfp8::group_quantize raises for this.
            raise ValueError("dbias is only supported for tensors with a common last dimension")
        self.WITH_DBIAS = with_dbias

        if with_dact and with_act:
            raise ValueError("with_dact and with_act cannot both be set")
        if with_dact:
            if activation not in SUPPORTED_DACTIVATIONS:
                raise ValueError(
                    f"unknown activation {activation!r} for with_dact=True; expected one of"
                    f" {sorted(SUPPORTED_DACTIVATIONS)}"
                )
            self.ACTIVATION = activation
        elif with_act:
            if activation not in SUPPORTED_ACTIVATIONS:
                raise ValueError(
                    f"unknown activation {activation!r} for with_act=True; expected one of"
                    f" {sorted(SUPPORTED_ACTIVATIONS)}"
                )
            self.ACTIVATION = activation
        else:
            if activation != "none":
                raise ValueError("activation must be none when with_dact and with_act are False")
            self.ACTIVATION = None
        self.WITH_DACT = with_dact
        self.WITH_ACT = with_act

    def __str__(self):
        return (
            f"MXFP8GroupQuantizeConfig(dtype={self.DTYPE_STR}, fp8_dtype={self.FP8_DTYPE_STR}, "
            f"rowwise={self.ROWWISE}, colwise={self.COLWISE}, shape_rep={self.SHAPE_REP}, "
            f"swizzled={self.WITH_GEMM_SWIZZLED_SCALES}, with_dbias={self.WITH_DBIAS}, "
            f"with_dact={self.WITH_DACT}, with_act={self.WITH_ACT}, "
            f"activation={self.ACTIVATION})"
        )

    __repr__ = __str__


class MXFP8GroupQuantizeKernel:
    """Grouped MXFP8 quantize mirroring group_quantize_mxfp8_kernel's strategy."""

    # TunableConfig / derived constants from group_quantize_mxfp8.cuh.
    CHUNK_DIM_Y = 128
    CHUNK_DIM_X = 128
    THREADS_PER_CHUNK = 128
    STATIC_PERSISTENT_BLOCKS_PER_SM = 24
    ELTS_PER_CHUNK = CHUNK_DIM_Y * CHUNK_DIM_X
    THREADS_X = CHUNK_DIM_X // MXFP8_BLOCK_SCALING_SIZE  # 4
    THREADS_Y = THREADS_PER_CHUNK // THREADS_X  # 32
    BUFF_DIM_Y = THREADS_Y  # 32
    BUFF_DIM_X = CHUNK_DIM_X  # 128
    # Each block of (CHUNK_DIM_Y, CHUNK_DIM_X) consists of STAGES tiles of (BUFF_DIM_X, BUFF_DIM_Y) stacked vertically
    STAGES = CHUNK_DIM_Y // BUFF_DIM_Y  # 4
    PIPELINE_DEPTH = 2  # PREFETCH_STAGES(1) + 1
    NUM_WARPS = THREADS_PER_CHUNK // THREADS_PER_WARP  # 4

    # Rowwise vectorization constants (mirror MXFP8QuantizeKernel / CUDA PACK_SIZE).
    PACK_SIZE = 4
    WAVES = MXFP8_BLOCK_SCALING_SIZE // PACK_SIZE  # 8
    THREADS_PER_BANK = (32 * 4) // MXFP8_BLOCK_SCALING_SIZE  # 4

    def __init__(self, cfg: MXFP8GroupQuantizeConfig, SM_COUNT: int):
        self.cfg = cfg
        self.SM_COUNT = SM_COUNT
        # CastConfig<VARYING_BOTH_DIMS> widens the chunk to 128x256, i.e. STAGES_X = 2 tiles of
        # BUFF_DIM_X columns, each traversed in STAGES row stages before moving right.
        self.STAGES_X = 2 if cfg.SHAPE_REP == VARYING_BOTH_DIMS else 1
        self.CHUNK_WIDTH = self.CHUNK_DIM_X * self.STAGES_X
        # The CUDA kernel honors the noop flag only without fused activations or dbias.
        self.CHECK_NOOP_FLAG = not (cfg.WITH_ACT or cfg.WITH_DACT or cfg.WITH_DBIAS)
        # Like IS_CACHED_ACT_OP in CUDA: with both directions, the colwise pass caches the
        # activation in the input tile for the rowwise pass. ReLU is fused into the conversion
        # instead, which yields the same bytes.
        self.CACHE_ACTIVATION = (
            (cfg.WITH_ACT or cfg.WITH_DACT)
            and cfg.ROWWISE
            and cfg.COLWISE
            and cfg.ACTIVATION != "relu"
        )
        # CUDA reduces dbias in the colwise pass when there is one, else in the rowwise pass.
        self.DBIAS_IN_COLWISE = cfg.WITH_DBIAS and cfg.COLWISE
        self.DBIAS_IN_ROWWISE = cfg.WITH_DBIAS and not cfg.COLWISE

    # ---------------------------------------------------------------- helpers
    @cute.jit
    def _tensor_rows_cols(
        self, tensor_id, mFirstDims, mLastDims, first_logical_dim, last_logical_dim
    ):
        """Get the shape (rows, cols) of the tensor by tensor_id."""
        cfg = self.cfg
        if cutlass.const_expr(cfg.SHAPE_REP in (VARYING_FIRST_DIM, VARYING_BOTH_DIMS)):
            rows = Int32(mFirstDims[tensor_id])
        else:
            rows = Int32(first_logical_dim)
        if cutlass.const_expr(cfg.SHAPE_REP in (VARYING_LAST_DIM, VARYING_BOTH_DIMS)):
            cols = Int32(mLastDims[tensor_id])
        else:
            cols = Int32(last_logical_dim)
        return rows, cols

    @cute.jit
    def _find_tensor_from_offsets(self, mOffsets, num_tensors, offset: Int64):
        """Index of the tensor whose element range holds `offset` (find_tensor_from_offsets)."""
        low = Int32(1)
        hi = Int32(num_tensors)
        # Enough bisection steps for any group of up to MAX_SUPPORTED_TENSORS members.
        for _ in cutlass.range_constexpr(MAX_SUPPORTED_TENSORS.bit_length()):
            if low < hi:
                mid = low + (hi - low) // 2
                if Int64(mOffsets[mid]) <= offset:
                    low = mid + 1
                else:
                    hi = mid
        return low - 1

    @cute.jit
    def _scale_tensor(self, mS, base: Int64, layout):
        """View the scale buffer from element `base` on with `layout`."""
        return cute.make_tensor(
            cute.make_ptr(
                Float8E8M0FNU,
                mS.iterator.toint() + base,
                cute.AddressSpace.gmem,
                assumed_align=4,
            ),
            layout,
        )

    @cute.jit
    def _rowwise_scales(self, mS_row, base: Int64, rows, cols):
        """Rowwise scales of a (rows, cols) tensor at `base`, tiled per 32x128 stage."""
        if cutlass.const_expr(self.cfg.WITH_GEMM_SWIZZLED_SCALES):
            mS_t, _ = derive_swizzled_scale_layout(
                rows, cols, True, False, self._scale_tensor(mS_row, base, cute.make_layout(1)), None
            )
        else:
            # Rowwise scale's divisibility guarantee: (128, 4)
            stride = cute.round_up(cute.ceil_div(cols, MXFP8_BLOCK_SCALING_SIZE), 4)
            mS_t = self._scale_tensor(
                mS_row, base, cute.make_layout((rows, stride), stride=(stride, 1))
            )
        return cute.zipped_divide(
            mS_t, (self.BUFF_DIM_Y, self.BUFF_DIM_X // MXFP8_BLOCK_SCALING_SIZE)
        )

    @cute.jit
    def _colwise_scales(self, mS_col, base: Int64, rows, cols):
        """Colwise scales of a (rows, cols) tensor at `base`, tiled per 32x128 stage."""
        if cutlass.const_expr(self.cfg.WITH_GEMM_SWIZZLED_SCALES):
            _, mS_t = derive_swizzled_scale_layout(
                rows, cols, False, True, None, self._scale_tensor(mS_col, base, cute.make_layout(1))
            )
        else:
            # Colwise scale's divisibility guarantee: (4, 128)
            stride = cute.round_up(cols, 128)
            mS_t = self._scale_tensor(
                mS_col,
                base,
                cute.make_layout((rows // MXFP8_BLOCK_SCALING_SIZE, stride), stride=(stride, 1)),
            )
        return cute.zipped_divide(
            mS_t, (self.BUFF_DIM_Y // MXFP8_BLOCK_SCALING_SIZE, self.BUFF_DIM_X)
        )

    # ------------------------------------------------------------ entry point
    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mO_row: cute.Tensor,
        mO_col: cute.Tensor,
        mS_row: cute.Tensor,
        mS_col: cute.Tensor,
        mOffsets: cute.Tensor,  # int64[num_tensors + 1], CSR element offsets
        mFirstDims: cute.Tensor,  # int64[num_tensors] (VARYING_FIRST_DIM / VARYING_BOTH_DIMS)
        mLastDims: cute.Tensor,  # int64[num_tensors] (VARYING_LAST_DIM / VARYING_BOTH_DIMS)
        mTensormaps: cute.Tensor,  # int64[num_tensors, NUM_WORKSPACE_SLOTS, 16]
        mNoop: cute.Pointer,  # f32 cast_noop flag; may be null, checked on device
        mActInput: Optional[cute.Tensor],  # activation input, only with WITH_DACT
        mWorkspace: Optional[cute.Tensor],  # f32 partial dbias, only with WITH_DBIAS
        stream: CUstream,
    ):
        if cutlass.const_expr(CUTEDSL_DEBUG_LOGGING):
            cute.printf(f"[CuTeDSL] MXFP8GroupQuantizeKernel.__call__() cfg: {self.cfg}\n")

        cfg = self.cfg
        first_logical_dim = mX.shape[0]
        last_logical_dim = mX.shape[1]
        # Number of group members. Do NOT derive this from mOffsets: a caller is free to
        # pass a length-num_tensors stub for SAME_BOTH_DIMS (where the offsets array is
        # unused), which would make `mOffsets.shape[0] - 1` read one too few and divide by
        # zero at num_tensors == 1. The per-tensor descriptor workspace is num_tensors long
        # by construction -- one slot set per member -- so it is the reliable source.
        num_tensors = mTensormaps.shape[0]

        smem_tile_layout = cute.make_ordered_layout(
            (self.BUFF_DIM_Y, self.BUFF_DIM_X), order=(1, 0)
        )
        cta_tiler = (self.BUFF_DIM_Y, self.BUFF_DIM_X)

        op_load = cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_x, tma_src = cpasync.make_tiled_tma_atom(
            op_load, mX, smem_tile_layout, cta_tiler, num_multicast=1
        )
        tma_atom_act = None
        tma_src_act = None
        if cutlass.const_expr(cfg.WITH_DACT):
            tma_atom_act, tma_src_act = cpasync.make_tiled_tma_atom(
                op_load, mActInput, smem_tile_layout, cta_tiler, num_multicast=1
            )
        op_store = cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_out_row, tma_dst_out_row = cpasync.make_tiled_tma_atom(
            op_store, mO_row, smem_tile_layout, cta_tiler, num_multicast=1
        )
        tma_atom_out_col, tma_dst_out_col = cpasync.make_tiled_tma_atom(
            op_store, mO_col, smem_tile_layout, cta_tiler, num_multicast=1
        )

        if cutlass.const_expr(cfg.IS_SINGLE_TENSOR):
            # How many blocks does the grouped tensor have in both directions
            work_blocks_X = cute.ceil_div(Int32(last_logical_dim), self.CHUNK_DIM_X)
            work_blocks_Y = cute.ceil_div(Int32(first_logical_dim), self.CHUNK_DIM_Y)
            # Each CTA handles one chunk. With every member's rows a multiple of CHUNK_DIM_Y,
            # this linear order is the CUDA kernel's (X, Y-in-tensor, tensor) order.
            grid = [work_blocks_X * work_blocks_Y, 1, 1]
        else:
            # The work-block grid is per-tensor here: each CTA derives its own block range
            # from its tensor's extents, so work_blocks_X/Y are unused on this path.
            work_blocks_Y = Int32(1)
            work_blocks_X = Int32(1)
            # Persistent worker count, mirroring get_launch_config() in
            # group_quantize_mxfp8.cuh: SM_COUNT * STATIC_PERSISTENT_BLOCKS_PER_SM workers
            # split evenly across tensors, clamped to the average number of chunks a tensor
            # holds. The element count would wrap Int32, so the estimate
            # DIVUP(elts_total, CHUNK_DIM_Y * TILE_DIM_X) is formed without it.
            n_tensors = cutlass.max(Int32(num_tensors), Int32(1))  # never divide by zero
            if cutlass.const_expr(cfg.SHAPE_REP == VARYING_BOTH_DIMS):
                # The logical shape is [1, total].
                estimated_work_blocks = cute.ceil_div(Int32(last_logical_dim), self.ELTS_PER_CHUNK)
            else:
                # Exact: the first extent is compiled with divisibility CHUNK_DIM_Y.
                estimated_work_blocks = cute.ceil_div(
                    (Int32(first_logical_dim) // self.CHUNK_DIM_Y) * Int32(last_logical_dim),
                    self.ELTS_PER_CHUNK // self.CHUNK_DIM_Y,
                )
            estimated_work_blocks = cute.ceil_div(estimated_work_blocks, self.STAGES_X)
            requested_workers_per_tensor = cutlass.max(
                Int32(1),
                Int32(self.SM_COUNT * self.STATIC_PERSISTENT_BLOCKS_PER_SM) // n_tensors,
            )
            average_work_blocks_per_tensor = cutlass.max(
                Int32(1), cute.ceil_div(estimated_work_blocks, n_tensors)
            )
            workers_per_tensor = cutlass.min(
                requested_workers_per_tensor, average_work_blocks_per_tensor
            )
            grid = [workers_per_tensor, Int32(num_tensors), 1]

        # Only the multi-tensor representations need per-tensor descriptors.
        if cutlass.const_expr(not cfg.IS_SINGLE_TENSOR):
            self.update_descriptors_kernel(
                mX,
                mO_row,
                mO_col,
                mActInput,
                mOffsets,
                mFirstDims,
                mLastDims,
                mTensormaps,
                first_logical_dim,
                last_logical_dim,
                mX.element_type,
                tma_atom_x,
                tma_atom_out_row,
                tma_atom_out_col,
                tma_atom_act,
            ).launch(grid=[num_tensors, 1, 1], block=[THREADS_PER_WARP, 1, 1], stream=stream)

        self.kernel(
            mS_row,
            mS_col,
            mOffsets,
            mFirstDims,
            mTensormaps,
            mNoop,
            mWorkspace,
            first_logical_dim,
            last_logical_dim,
            num_tensors,
            work_blocks_X,
            mX.element_type,
            tma_atom_x,
            tma_src,
            tma_atom_act,
            tma_src_act,
            tma_atom_out_row,
            tma_dst_out_row,
            tma_atom_out_col,
            tma_dst_out_col,
        ).launch(
            grid=grid,
            block=[self.THREADS_PER_CHUNK, 1, 1],
            stream=stream,
        )

    # ------------------------------------------------- descriptor prologue
    @cute.kernel
    def update_descriptors_kernel(
        self,
        mX,
        mO_row,
        mO_col,
        mActInput,
        mOffsets,
        mFirstDims,
        mLastDims,
        mTensormaps,
        first_logical_dim,
        last_logical_dim,
        dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
        tma_atom_x,
        tma_atom_orow,
        tma_atom_ocol,
        tma_atom_act,
    ):
        """One CTA per tensor: point that tensor's TMA descriptors at its own block.

        CuTeDSL analog of common::update_tma_descriptors writing g_tensor_maps[].
        """
        cfg = self.cfg
        tensor_id, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        rows, cols = self._tensor_rows_cols(
            tensor_id, mFirstDims, mLastDims, first_logical_dim, last_logical_dim
        )
        base_elts = Int64(mOffsets[tensor_id])

        # Same diagnostics as get_tensor_rows_num / get_tensor_cols_num. Like NVTE_DEVICE_ERROR
        # in a release build, they only print.
        if tidx == 0:
            if rows % 128 != 0:
                cute.printf(
                    "tensor %d: First dimension of each tensor in a group must be divisible"
                    " by 128.\n",
                    tensor_id,
                )
            if cols % 128 != 0:
                cute.printf(
                    "tensor %d: For varying last dimensions support, the last dimension of each"
                    " tensor in a group must be divisible by 128.\n",
                    tensor_id,
                )

        # Publish this tensor's geometry for the main kernel (written even when empty).
        meta = mTensormaps[(tensor_id, META_SLOT, None)]
        meta[0] = Int64(rows)
        meta[1] = Int64(cols)
        meta[2] = base_elts

        tmap = TensorMapManager(TensorMapUpdateMode.GMEM, BYTES_PER_TENSORMAP)
        desc_x = tmap.get_tensormap_ptr(mTensormaps[(tensor_id, 0, None)].iterator)
        desc_orow = tmap.get_tensormap_ptr(mTensormaps[(tensor_id, 1, None)].iterator)
        desc_ocol = tmap.get_tensormap_ptr(mTensormaps[(tensor_id, 2, None)].iterator)
        desc_act = tmap.get_tensormap_ptr(mTensormaps[(tensor_id, ACT_INPUT_SLOT, None)].iterator)

        # Zero-sized groups: creating a descriptor with a zero extent is invalid,
        # so skip (the main kernel skips these tensors as well).
        if rows > 0 and cols > 0:
            member_layout = cute.make_layout((rows, cols), stride=(cols, 1))

            def member_view(tensor, elt_dtype):
                return cute.make_tensor(
                    cute.make_ptr(
                        elt_dtype,
                        tensor.iterator.toint() + base_elts * (elt_dtype.width // 8),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    member_layout,
                )

            views = [member_view(mX, dtype)]
            atoms = [tma_atom_x]
            descs = [desc_x]
            tmap.init_tensormap_from_atom(tma_atom_x, desc_x, 0)
            if cutlass.const_expr(cfg.ROWWISE):
                views.append(member_view(mO_row, cfg.FP8_DTYPE))
                atoms.append(tma_atom_orow)
                descs.append(desc_orow)
                tmap.init_tensormap_from_atom(tma_atom_orow, desc_orow, 0)
            if cutlass.const_expr(cfg.COLWISE):
                views.append(member_view(mO_col, cfg.FP8_DTYPE))
                atoms.append(tma_atom_ocol)
                descs.append(desc_ocol)
                tmap.init_tensormap_from_atom(tma_atom_ocol, desc_ocol, 0)
            if cutlass.const_expr(cfg.WITH_DACT):
                views.append(member_view(mActInput, dtype))
                atoms.append(tma_atom_act)
                descs.append(desc_act)
                tmap.init_tensormap_from_atom(tma_atom_act, desc_act, 0)
            tmap.fence_tensormap_initialization()
            tmap.update_tensormap(
                tuple(views),
                tuple(atoms),
                tuple(descs),
                0,
                (),  # smem staging is unused in GMEM update mode
            )

    # ------------------------------------------------------------ main kernel
    @cute.kernel
    def kernel(
        self,
        mS_row,
        mS_col,
        mOffsets,
        mFirstDims,
        mTensormaps,
        mNoop,
        mWorkspace,
        first_logical_dim,
        last_logical_dim,
        num_tensors,
        work_blocks_X,
        dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
        tma_atom_x,
        tma_src,
        tma_atom_act,
        tma_src_act,
        tma_atom_out_row,
        tma_dst_out_row,
        tma_atom_out_col,
        tma_dst_out_col,
    ):
        """No-op the CTA when the noop flag is set, else run the quantize main loop."""
        skip_execution = Boolean(False)
        if cutlass.const_expr(self.CHECK_NOOP_FLAG):
            skip_execution = noop_flag_is_set(mNoop)
        if not skip_execution:
            self._kernel_main(
                mS_row,
                mS_col,
                mOffsets,
                mFirstDims,
                mTensormaps,
                mWorkspace,
                first_logical_dim,
                last_logical_dim,
                num_tensors,
                work_blocks_X,
                dtype,
                tma_atom_x,
                tma_src,
                tma_atom_act,
                tma_src_act,
                tma_atom_out_row,
                tma_dst_out_row,
                tma_atom_out_col,
                tma_dst_out_col,
            )

    @cute.jit
    def _kernel_main(
        self,
        mS_row,
        mS_col,
        mOffsets,
        mFirstDims,
        mTensormaps,
        mWorkspace,
        first_logical_dim,
        last_logical_dim,
        num_tensors,
        work_blocks_X,
        dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
        tma_atom_x,
        tma_src,
        tma_atom_act,
        tma_src_act,
        tma_atom_out_row,
        tma_dst_out_row,
        tma_atom_out_col,
        tma_dst_out_col,
    ):
        cfg = self.cfg
        FP8_DTYPE = cfg.FP8_DTYPE
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        gdx, _, _ = cute.arch.grid_dim()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if cutlass.const_expr(cfg.SHAPE_REP == VARYING_FIRST_DIM):
            # The first CTA validates every member's rows, as the CUDA kernel does. Like
            # NVTE_DEVICE_ERROR in a release build, this only prints.
            if bidx == 0:
                if tidx < num_tensors:
                    if Int64(mFirstDims[tidx]) % 128 != 0:
                        cute.printf(
                            "tensor %d: First dimension of each tensor in a group must be"
                            " divisible by 128.\n",
                            tidx,
                        )

        # --- shared memory (allocated once, reused across jobs) ---
        @cute.struct
        class SharedStorage:
            mbar: cute.struct.MemRange[cute.Int64, 2 * self.PIPELINE_DEPTH]
            sX: cute.struct.Align[
                cute.struct.MemRange[
                    dtype, self.BUFF_DIM_Y * self.BUFF_DIM_X * self.PIPELINE_DEPTH
                ],
                128,
            ]
            sO_row: cute.struct.Align[
                cute.struct.MemRange[
                    FP8_DTYPE, self.BUFF_DIM_Y * self.BUFF_DIM_X * self.PIPELINE_DEPTH
                ],
                128,
            ]
            sO_col: cute.struct.Align[
                cute.struct.MemRange[
                    FP8_DTYPE, self.BUFF_DIM_Y * self.BUFF_DIM_X * self.PIPELINE_DEPTH
                ],
                128,
            ]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        tile_layout = cute.make_layout(
            ((self.BUFF_DIM_Y, self.BUFF_DIM_X), self.PIPELINE_DEPTH),
            stride=((self.BUFF_DIM_X, 1), self.BUFF_DIM_Y * self.BUFF_DIM_X),
        )
        sX = storage.sX.get_tensor(tile_layout)
        sO_row = storage.sO_row.get_tensor(tile_layout)
        sO_col = storage.sO_col.get_tensor(tile_layout)

        sActInput = None
        if cutlass.const_expr(cfg.WITH_DACT):

            @cute.struct
            class DactStorage:
                sActInput: cute.struct.Align[
                    cute.struct.MemRange[
                        dtype, self.BUFF_DIM_Y * self.BUFF_DIM_X * self.PIPELINE_DEPTH
                    ],
                    128,
                ]

            sActInput = smem.allocate(DactStorage).sActInput.get_tensor(tile_layout)

        sDbias = None
        if cutlass.const_expr(self.DBIAS_IN_ROWWISE):
            # Padded like the CUDA kernel's partial_dbias_rowwise to avoid bank conflicts.
            DBIAS_BUFF_WIDTH = self.THREADS_X * (MXFP8_BLOCK_SCALING_SIZE + 1)

            @cute.struct
            class DbiasStorage:
                sDbias: cute.struct.MemRange[Float32, self.THREADS_Y * DBIAS_BUFF_WIDTH]

            sDbias = smem.allocate(DbiasStorage).sDbias.get_tensor(
                cute.make_layout((self.THREADS_Y, self.BUFF_DIM_X), stride=(DBIAS_BUFF_WIDTH, 1))
            )

        # Grad and activation input share each stage's barrier.
        tx_count = self.BUFF_DIM_Y * self.BUFF_DIM_X * dtype.width // 8
        if cutlass.const_expr(cfg.WITH_DACT):
            tx_count *= 2
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar.data_ptr(),
            num_stages=self.PIPELINE_DEPTH,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.NUM_WARPS),
            tx_count=tx_count,
            cta_layout_vmnk=None,
        )
        prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.PIPELINE_DEPTH
        )
        cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.PIPELINE_DEPTH
        )

        # TMA partitions built from the representative views. For multi-tensor reps
        # the descriptor is swapped per tensor and the tile coords are tensor-local.
        gX_tiled = cute.zipped_divide(tma_src, (self.BUFF_DIM_Y, self.BUFF_DIM_X))
        tXsX, tXgX = cpasync.tma_partition(tma_atom_x, 0, cute.make_layout(1), sX, gX_tiled)
        tXsA = None
        tXgA = None
        if cutlass.const_expr(cfg.WITH_DACT):
            gA_tiled = cute.zipped_divide(tma_src_act, (self.BUFF_DIM_Y, self.BUFF_DIM_X))
            tXsA, tXgA = cpasync.tma_partition(
                tma_atom_act, 0, cute.make_layout(1), sActInput, gA_tiled
            )
        gO_row_tiled = cute.zipped_divide(tma_dst_out_row, (self.BUFF_DIM_Y, self.BUFF_DIM_X))
        tXsO_row, tXgO_row = cpasync.tma_partition(
            tma_atom_out_row, 0, cute.make_layout(1), sO_row, gO_row_tiled
        )
        gO_col_tiled = cute.zipped_divide(tma_dst_out_col, (self.BUFF_DIM_Y, self.BUFF_DIM_X))
        tXsO_col, tXgO_col = cpasync.tma_partition(
            tma_atom_out_col, 0, cute.make_layout(1), sO_col, gO_col_tiled
        )

        tmap = TensorMapManager(TensorMapUpdateMode.GMEM, BYTES_PER_TENSORMAP)
        cute.arch.sync_threads()

        # If the CTA has work to do
        has_work = Boolean(True)
        # Metadata of the tensor that owns this block
        tensor_rows = Int32(0)
        tensor_cols = Int32(0)
        # Element offset of this tensor within the group: Int64 (CUDA uses size_t), since a
        # group can exceed 2^31 elements even when every individual extent is small.
        tensor_base = Int64(0)
        # Block's offset and id in this individual tensor / global single tensor
        block_offset_Y = Int32(0)
        block_id_X = Int32(0)

        first_block_id = Int32(0)
        blocks_in_tensor = Int32(1)
        block_stride = Int32(1)
        block_columns_in_tensor = Int32(1)

        if cutlass.const_expr(cfg.IS_SINGLE_TENSOR):
            # grid = [work_blocks_X * work_blocks_Y, 1, 1]
            block_id_Y = Int32(bidx) // work_blocks_X
            block_id_X = Int32(bidx) % work_blocks_X
            # View the grouped tensor as a single tensor of shape (first_logical_dim, last_logical_dim)
            tensor_rows = Int32(first_logical_dim)
            tensor_cols = Int32(last_logical_dim)
            # Which row does this block start from
            block_offset_Y = block_id_Y * self.CHUNK_DIM_Y
            if cutlass.const_expr(cfg.SHAPE_REP == VARYING_FIRST_DIM):
                # logical_shape may describe graph-safe capacity beyond the active tensors,
                # whose total element count is the last CSR offset.
                total_elts = Int64(mOffsets[mOffsets.shape[0] - 1])
                if Int64(block_offset_Y) * Int64(last_logical_dim) >= total_elts:
                    has_work = Boolean(False)
        else:
            # grid = [workers_per_tensor, Int32(num_tensors), 1]
            tensor_id = Int32(bidy)
            # Extract tensor's metadata
            meta = mTensormaps[(tensor_id, META_SLOT, None)]
            tensor_rows = Int32(meta[0])
            tensor_cols = Int32(meta[1])
            tensor_base = Int64(meta[2])
            if tensor_rows > 0 and tensor_cols > 0:
                # How many blocks does this tensor have in both directions
                block_columns_in_tensor = cute.ceil_div(tensor_cols, self.CHUNK_WIDTH)
                block_rows_in_tensor = cute.ceil_div(tensor_rows, self.CHUNK_DIM_Y)
                # How many blocks does this tensor have
                blocks_in_tensor = block_columns_in_tensor * block_rows_in_tensor
                # Which block (1D index) does this CTA start from
                first_block_id = Int32(bidx)
                # gdx is workers_per_tensor (how many CTAs are assigned to this tensor)
                block_stride = Int32(gdx)
                # If my first block_id is already beyond the tensor's last block, I have no work to do
                if first_block_id >= blocks_in_tensor:
                    has_work = Boolean(False)
            else:
                # This tensor is empty, so this CTA has no work to do
                has_work = Boolean(False)

        partitions = (tXsX, tXgX, tXsA, tXgA, tXsO_row, tXgO_row, tXsO_col, tXgO_col)
        atoms = (tma_atom_x, tma_atom_act, tma_atom_out_row, tma_atom_out_col)

        if has_work:
            if cutlass.const_expr(cfg.IS_SINGLE_TENSOR):
                # For single tensor case we don't use tensor descriptors
                descs = (None, None, None, None)

                # Rowwise scales span the whole group: members are stacked on 128-row
                # boundaries, so each member's swizzled tiles follow the previous member's.
                row_scales = None
                if cutlass.const_expr(cfg.ROWWISE):
                    row_scales = self._rowwise_scales(mS_row, Int64(0), tensor_rows, tensor_cols)
                col_scales = None
                col_scale_row0 = block_offset_Y
                col_scale_rows = tensor_rows
                if cutlass.const_expr(cfg.COLWISE):
                    col_scale_base = Int64(0)
                    if cutlass.const_expr(cfg.WITH_GEMM_SWIZZLED_SCALES):
                        # Colwise swizzled scale indices restart at each member and depend
                        # on its rows (process_colwise_stage), so address the member that
                        # owns this chunk.
                        member_rows = Int32(0)
                        member_row0 = Int32(0)
                        if cutlass.const_expr(cfg.SHAPE_REP == SAME_BOTH_DIMS):
                            member_rows = tensor_rows // Int32(num_tensors)
                            member_row0 = block_offset_Y // member_rows * member_rows
                        else:
                            member_id = self._find_tensor_from_offsets(
                                mOffsets,
                                num_tensors,
                                Int64(block_offset_Y) * Int64(tensor_cols),
                            )
                            member_rows = Int32(mFirstDims[member_id])
                            member_row0 = Int32(Int64(mOffsets[member_id]) // Int64(tensor_cols))
                        col_scale_base = (
                            Int64(member_row0)
                            * Int64(cute.round_up(tensor_cols, 128))
                            // MXFP8_BLOCK_SCALING_SIZE
                        )
                        col_scale_row0 = block_offset_Y - member_row0
                        col_scale_rows = member_rows
                    col_scales = self._colwise_scales(
                        mS_col, col_scale_base, col_scale_rows, tensor_cols
                    )

                cute.arch.sync_threads()

                self._process_block(
                    block_offset_Y,
                    block_id_X,
                    tensor_rows,
                    tensor_cols,
                    row_scales,
                    col_scales,
                    col_scale_row0,
                    col_scale_rows,
                    block_offset_Y // self.CHUNK_DIM_Y,
                    mWorkspace,
                    sDbias,
                    descs,
                    tmap,
                    warp_idx,
                    tidx,
                    sX,
                    sActInput,
                    sO_row,
                    sO_col,
                    partitions,
                    atoms,
                    mainloop_pipeline,
                    prod_state,
                    cons_state,
                )
            else:
                # For multi-tensor case, retrieve tensor descriptors we processed early in the prologue kernel
                desc_x = tmap.get_tensormap_ptr(mTensormaps[(tensor_id, 0, None)].iterator)
                desc_out_row = tmap.get_tensormap_ptr(mTensormaps[(tensor_id, 1, None)].iterator)
                desc_out_col = tmap.get_tensormap_ptr(mTensormaps[(tensor_id, 2, None)].iterator)
                desc_act = tmap.get_tensormap_ptr(
                    mTensormaps[(tensor_id, ACT_INPUT_SLOT, None)].iterator
                )
                # Acquire the descriptors on ONE thread, as the CUDA kernel does
                # (`leading_thread` in group_quantize_mxfp8.cuh); the sync_threads below
                # publishes it CTA-wide. Running the tensormap acquire fence on all 128
                # threads is correct but very expensive -- it more than doubles the
                # kernel time on the multi-tensor path (4096x14336 bidirectional:
                # 133 us -> 58 us), since the cost scales with threads x descriptors.
                if tidx == 0:
                    tmap.fence_tensormap_update(desc_x)
                    if cutlass.const_expr(cfg.WITH_DACT):
                        tmap.fence_tensormap_update(desc_act)
                    if cutlass.const_expr(cfg.ROWWISE):
                        tmap.fence_tensormap_update(desc_out_row)
                    if cutlass.const_expr(cfg.COLWISE):
                        tmap.fence_tensormap_update(desc_out_col)
                descs = (desc_x, desc_act, desc_out_row, desc_out_col)

                # This tensor's scales start at tensor_base / 32 in both directions.
                scale_base = tensor_base // Int64(MXFP8_BLOCK_SCALING_SIZE)
                row_scales = None
                if cutlass.const_expr(cfg.ROWWISE):
                    row_scales = self._rowwise_scales(mS_row, scale_base, tensor_rows, tensor_cols)
                col_scales = None
                if cutlass.const_expr(cfg.COLWISE):
                    col_scales = self._colwise_scales(mS_col, scale_base, tensor_rows, tensor_cols)

                cute.arch.sync_threads()

                # Grid-stride over this tensor's own chunks; the descriptors never change.
                block_id = first_block_id
                job_finished = Boolean(False)
                while not job_finished:
                    block_id_Y_in_tensor = block_id // block_columns_in_tensor
                    block_id_X_in_tensor = block_id % block_columns_in_tensor
                    block_offset_Y_in_tensor = block_id_Y_in_tensor * self.CHUNK_DIM_Y
                    if cutlass.const_expr(self.STAGES_X == 1):
                        self._process_block(
                            block_offset_Y_in_tensor,
                            block_id_X_in_tensor,
                            tensor_rows,
                            tensor_cols,
                            row_scales,
                            col_scales,
                            block_offset_Y_in_tensor,
                            tensor_rows,
                            Int32(0),  # dbias is only supported for single-tensor reps
                            mWorkspace,
                            sDbias,
                            descs,
                            tmap,
                            warp_idx,
                            tidx,
                            sX,
                            sActInput,
                            sO_row,
                            sO_col,
                            partitions,
                            atoms,
                            mainloop_pipeline,
                            prod_state,
                            cons_state,
                        )
                    else:
                        # The chunk's column tiles in order, stopping at the tensor's last
                        # column like the CUDA kernel's stages_X = DIVUP(chunk_cols, TILE_DIM_X).
                        chunk_col0 = block_id_X_in_tensor * self.CHUNK_WIDTH
                        tiles_X = cutlass.min(
                            Int32(self.STAGES_X),
                            cute.ceil_div(tensor_cols - chunk_col0, self.BUFF_DIM_X),
                        )
                        for stage_X in cutlass.range(tiles_X, unroll=1):
                            self._process_block(
                                block_offset_Y_in_tensor,
                                block_id_X_in_tensor * self.STAGES_X + stage_X,
                                tensor_rows,
                                tensor_cols,
                                row_scales,
                                col_scales,
                                block_offset_Y_in_tensor,
                                tensor_rows,
                                Int32(0),  # dbias is only supported for single-tensor reps
                                mWorkspace,
                                sDbias,
                                descs,
                                tmap,
                                warp_idx,
                                tidx,
                                sX,
                                sActInput,
                                sO_row,
                                sO_col,
                                partitions,
                                atoms,
                                mainloop_pipeline,
                                prod_state,
                                cons_state,
                            )
                    # Find the next block to process
                    block_id = block_id + block_stride
                    if block_id >= blocks_in_tensor:
                        job_finished = Boolean(True)

        # Drain every TMA store before the CTA releases its shared-memory source buffers.
        if warp_idx == 0:
            cute.arch.cp_async_bulk_wait_group(0, read=False)
        cute.arch.sync_threads()

    def _issue_load(
        self,
        pipeline_obj,
        prod_state,
        tile_y,
        tile_x,
        atoms,
        partitions,
        tmap,
        descs,
    ):
        """Emit the 32x128 TMA load(s) of one stage into the current pipeline buffer.

        Caller gates this on warp 0 and advances `prod_state` afterwards -- the advance
        must happen outside the gate or the mutated SSA values stay trapped in the scf.if.
        """
        tma_atom_x, tma_atom_act, _, _ = atoms
        tXsX, tXgX, tXsA, tXgA, _, _, _, _ = partitions
        desc_x, desc_act, _, _ = descs
        # Wait for the consumer to finish using this SMEM buffer
        pipeline_obj.producer_acquire(prod_state)
        barrier = pipeline_obj.producer_get_barrier(prod_state)
        loads = [(tma_atom_x, tXgX, tXsX, desc_x)]
        if cutlass.const_expr(self.cfg.WITH_DACT):
            loads.append((tma_atom_act, tXgA, tXsA, desc_act))
        for atom, tXg, tXs, desc in loads:
            if cutlass.const_expr(self.cfg.IS_SINGLE_TENSOR):
                cute.copy(
                    atom,
                    tXg[(None, (tile_y, tile_x))],
                    tXs[(None, prod_state.index)],
                    tma_bar_ptr=barrier,
                )
            else:
                # Every member shares tXg's tile-coordinate arithmetic (the coefficients are
                # just the tile size); tma_desc_ptr supplies this member's geometry.
                cute.copy(
                    atom,
                    tXg[(None, (tile_y, tile_x))],
                    tXs[(None, prod_state.index)],
                    tma_bar_ptr=barrier,
                    tma_desc_ptr=tmap.get_tensormap_ptr(desc, cute.AddressSpace.generic),
                )
        # Notify the consumer that this SMEM buffer is ready for consumption
        pipeline_obj.producer_commit(prod_state)

    @cute.jit
    def _process_block(
        self,
        block_offset_Y,  # Row offset of this chunk (global for single-tensor, else tensor-local)
        block_id_X,  # Column-chunk index within the tensor
        rows,  # Rows of the rowwise-scale view (the group for single-tensor, else the tensor)
        cols,  # Number of columns in this tensor
        row_scales,  # Rowwise scales tiled per stage, rows counted like block_offset_Y
        col_scales,  # Colwise scales tiled per stage
        col_scale_row0,  # Row of this chunk in the colwise-scale view
        col_scale_rows,  # Rows of the colwise-scale view
        dbias_row,  # Row of the dbias workspace this chunk reduces into
        mWorkspace,  # f32 partial dbias workspace (WITH_DBIAS)
        sDbias,  # SMEM buffer for the rowwise dbias reduction (rowwise-only dbias)
        descs,  # Per-tensor descriptors (x, act, out_row, out_col), None if single-tensor
        tmap,  # TensorMapManager for managing TMA descriptors
        warp_idx,
        tidx,
        sX,  # SMEM input ring
        sActInput,  # SMEM activation input ring (WITH_DACT)
        sO_row,  # SMEM rowwise output ring
        sO_col,  # SMEM colwise output ring
        partitions,  # TMA partitions (x, act, out_row, out_col)
        atoms,  # TMA atoms (x, act, out_row, out_col)
        mainloop_pipeline: cutlass.pipeline.PipelineTmaAsync,
        prod_state,
        cons_state,
    ):
        """Quantize one 128x128 tile of a chunk in STAGES slices of BUFF_DIM_Y rows."""
        cfg = self.cfg
        _, _, tma_atom_out_row, tma_atom_out_col = atoms
        _, _, _, _, tXsO_row, tXgO_row, tXsO_col, tXgO_col = partitions
        _, _, desc_out_row, desc_out_col = descs
        block_offset_X = block_id_X * self.CHUNK_DIM_X

        # This chunk's coordinates in the tile grid (32x128 TMA boxes, not elements).
        tile_id_Y = block_offset_Y // self.BUFF_DIM_Y
        tile_id_X = block_id_X
        col_scale_tile_Y = col_scale_row0 // self.BUFF_DIM_Y

        # Per-chunk dbias accumulators, in the CUDA kernel's summation order: a running
        # column sum over the chunk's rows (colwise), or per-thread partial sums over its
        # stages that the whole CTA reduces afterwards (rowwise-only).
        dbias_col = Float32(0.0)
        dbias_row_acc = None
        if cutlass.const_expr(self.DBIAS_IN_ROWWISE):
            dbias_row_acc = cute.make_rmem_tensor(
                layout_or_shape=cute.make_layout((MXFP8_BLOCK_SCALING_SIZE,), stride=(1,)),
                dtype=Float32,
            )
            for c in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE):
                dbias_row_acc[c] = Float32(0.0)

        # Fill every buffer up front, then issue one more each time a stage is consumed.
        for prologue_stage in cutlass.range_constexpr(self.PIPELINE_DEPTH):
            if warp_idx == 0:
                self._issue_load(
                    mainloop_pipeline,
                    prod_state,
                    tile_id_Y + prologue_stage,
                    tile_id_X,
                    atoms,
                    partitions,
                    tmap,
                    descs,
                )
            prod_state.advance()

        for stage in cutlass.range_constexpr(self.STAGES):
            # Wait for at most DEPTH-1 iters on the fly, which means the the last DEPTH iter has finished
            # so we can reuse its SMEM output buffer
            # (input buffer is managed by the producer and consumer pipeline states)
            if warp_idx == 0:
                cute.arch.cp_async_bulk_wait_group(self.PIPELINE_DEPTH - 1, read=True)
            # Wait for this stage's input buffer to be filled by the producer
            mainloop_pipeline.consumer_wait(cons_state)
            cute.arch.sync_threads()
            sX_tile = sX[(None, cons_state.index)]
            sAct_tile = None
            if cutlass.const_expr(cfg.WITH_DACT):
                sAct_tile = sActInput[(None, cons_state.index)]
            row_tile = tile_id_Y + stage

            if cutlass.const_expr(cfg.COLWISE):
                _, dbias_col = quantize_colwise_mxfp8(
                    sX_tile,
                    sAct_tile,
                    sO_col[(None, cons_state.index)],
                    cute.flatten(col_scales[(None, (col_scale_tile_Y + stage, tile_id_X))]),
                    cfg.MAX_NORM_RCP,
                    (col_scale_tile_Y + stage) * self.BUFF_DIM_Y,
                    block_offset_X,
                    col_scale_rows,
                    cols,
                    ACTIVATION=cfg.ACTIVATION,
                    DTYPE=cfg.DTYPE,
                    FP8_DTYPE=cfg.FP8_DTYPE,
                    SWIZZLE=cfg.WITH_GEMM_SWIZZLED_SCALES,
                    TILE_X=self.BUFF_DIM_X,
                    TILE_Y=self.BUFF_DIM_Y,
                    WITH_ACT=cfg.WITH_ACT,
                    WITH_DACT=cfg.WITH_DACT,
                    WITH_DBIAS=self.DBIAS_IN_COLWISE,
                    CACHE_ACTIVATION=self.CACHE_ACTIVATION,
                    ZERO_OOB_SCALES=True,
                    dbias_init=dbias_col,
                )
            if cutlass.const_expr(self.CACHE_ACTIVATION):
                # The rowwise pass reads the activation the colwise pass cached in sX.
                cute.arch.sync_threads()
            if cutlass.const_expr(cfg.ROWWISE):
                quantize_rowwise_mxfp8(
                    sX_tile,
                    None if self.CACHE_ACTIVATION else sAct_tile,
                    sO_row[(None, cons_state.index)],
                    cute.flatten(row_scales[(None, (row_tile, tile_id_X))]),
                    cfg.MAX_NORM_RCP,
                    row_tile * self.BUFF_DIM_Y,
                    block_offset_X,
                    rows,
                    cols,
                    ACTIVATION=None if self.CACHE_ACTIVATION else cfg.ACTIVATION,
                    DTYPE=cfg.DTYPE,
                    FP8_DTYPE=cfg.FP8_DTYPE,
                    TILE_X=self.BUFF_DIM_X,
                    TILE_Y=self.BUFF_DIM_Y,
                    WAVES=self.WAVES,
                    THREADS_PER_BANK=self.THREADS_PER_BANK,
                    PACK_SIZE=self.PACK_SIZE,
                    WITH_ACT=cfg.WITH_ACT and not self.CACHE_ACTIVATION,
                    WITH_DACT=cfg.WITH_DACT and not self.CACHE_ACTIVATION,
                    WITH_DBIAS=self.DBIAS_IN_ROWWISE,
                    dbias_acc=dbias_row_acc,
                    ZERO_OOB_SCALES=True,
                )

            # Force consumer's write to SMEM to be visible to TMA stores later
            cute.arch.fence_proxy("async.shared", space="cta")
            # Only after everyone finishes computation then this stage can be considered as "consumed"
            cute.arch.sync_threads()
            # I'm done with my input SMEM buffer, so the producer can write the next stage's data into it
            mainloop_pipeline.consumer_release(cons_state)

            # I just freed my input SMEM buffer (stage), so the producer now can use it for writing
            # (stage+DEPTH) stage's data if that stage exists
            if cutlass.const_expr(stage + self.PIPELINE_DEPTH < self.STAGES):
                if warp_idx == 0:
                    self._issue_load(
                        mainloop_pipeline,
                        prod_state,
                        tile_id_Y + stage + self.PIPELINE_DEPTH,
                        tile_id_X,
                        atoms,
                        partitions,
                        tmap,
                        descs,
                    )
                prod_state.advance()

            # Write result to GMEM via TMA
            if warp_idx == 0:
                stores = []
                if cutlass.const_expr(cfg.ROWWISE):
                    stores.append((tma_atom_out_row, tXsO_row, tXgO_row, desc_out_row))
                if cutlass.const_expr(cfg.COLWISE):
                    stores.append((tma_atom_out_col, tXsO_col, tXgO_col, desc_out_col))
                for atom, tXs, tXg, desc in stores:
                    if cutlass.const_expr(cfg.IS_SINGLE_TENSOR):
                        cute.copy(
                            atom,
                            tXs[(None, cons_state.index)],
                            tXg[(None, (row_tile, tile_id_X))],
                        )
                    else:
                        cute.copy(
                            atom,
                            tXs[(None, cons_state.index)],
                            tXg[(None, (row_tile, tile_id_X))],
                            tma_desc_ptr=tmap.get_tensormap_ptr(desc, cute.AddressSpace.generic),
                        )
                # Commit all TMA operations of this iteration
                cute.arch.cp_async_bulk_commit_group()

            cons_state.advance()

        if cutlass.const_expr(cfg.WITH_DBIAS):
            if cutlass.const_expr(self.DBIAS_IN_ROWWISE):
                dbias_col = self._reduce_rowwise_dbias(sDbias, tidx, dbias_row_acc)
            # One partial-dbias row per chunk, as in the CUDA kernel's dbias_workspace.
            dbias_x = block_offset_X + tidx
            if dbias_x < cols:
                mWorkspace[(dbias_row, dbias_x)] = dbias_col

    @cute.jit
    def _reduce_rowwise_dbias(self, sDbias, tidx, dbias_row_acc):
        """Reduce the per-thread rowwise partial sums to one sum per column, in the order of the
        CUDA kernel's partial_dbias_rowwise reduction."""
        _, tv_write = cute.make_layout_tv(
            thr_layout=cute.make_layout(
                (self.THREADS_Y, self.THREADS_X), stride=(self.THREADS_X, 1)
            ),
            val_layout=cute.make_layout(
                (1, MXFP8_BLOCK_SCALING_SIZE), stride=(MXFP8_BLOCK_SCALING_SIZE, 1)
            ),
        )
        sDbias_write = cute.composition(sDbias, tv_write)
        bank_group = (tidx % THREADS_PER_WARP) // self.THREADS_PER_BANK
        offset = bank_group * self.PACK_SIZE
        for w in cutlass.range_constexpr(self.WAVES):
            # Undo the bank-conflict rotation quantize_rowwise_mxfp8 accumulated in.
            start = (w * self.PACK_SIZE + offset) % MXFP8_BLOCK_SCALING_SIZE
            for i in cutlass.range_constexpr(self.PACK_SIZE):
                sDbias_write[(tidx, start + i)] = dbias_row_acc[w * self.PACK_SIZE + i]
        cute.arch.sync_threads()
        # Thread tidx sums column tidx over the THREADS_Y partial rows.
        dbias = Float32(0.0)
        for i in cutlass.range_constexpr(self.THREADS_Y):
            dbias += sDbias[(i, tidx)]
        # The buffer is rewritten by the next chunk.
        cute.arch.sync_threads()
        return dbias


def compile_cutedsl_function_from_cfg(cfg: MXFP8GroupQuantizeConfig):
    """Return the compiled CuTeDSL function object for the given grouped config."""
    # CUDA requires the group's first logical dim to be a multiple of 128 (and each
    # tensor's rows likewise). VARYING_BOTH_DIMS is the exception: its logical shape is
    # [1, total]. The last dim only needs the 16-byte TMA row alignment; a partial 32-element
    # scale block at the end of a row is zero-filled by TMA, as in the CUDA kernel.
    if cfg.SHAPE_REP == VARYING_BOTH_DIMS:
        sym_M = cute.sym_int32()
    else:
        sym_M = cute.sym_int32(divisibility=128)
    sym_N = cute.sym_int32(divisibility=SYM_N_DIVISIBILITY)
    logical_shape = (sym_M, sym_N)

    out_dtype = cfg.FP8_DTYPE
    scale_dtype = cutlass.Float8E8M0FNU

    def g2d(dtype, shape=logical_shape, align=16):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            stride_order=(1, 0),
            memspace=cute.AddressSpace.gmem,
            assumed_align=align,
        )

    def g1d(dtype, align=4):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            (cute.sym_int32(),),
            stride_order=(0,),
            memspace=cute.AddressSpace.gmem,
            assumed_align=align,
        )

    # The kernel only takes the base address of the scale buffers (per-tensor strides
    # are derived from cols), so their fake shape is a flat 1D byte run.
    tensormaps_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int64,
        (cute.sym_int32(), NUM_WORKSPACE_SLOTS, BYTES_PER_TENSORMAP // 8),
        stride_order=(2, 1, 0),
        memspace=cute.AddressSpace.gmem,
        assumed_align=128,
    )
    # The cast-noop flag is an always-present f32 pointer instead of an optional tensor, so
    # that one compiled kernel serves both an absent and a present flag (noop_flag_is_set).
    noop_fake = cute.runtime.nullptr(Float32, mem_space=cute.AddressSpace.gmem, assumed_align=4)
    act_input_fake = g2d(cfg.DTYPE) if cfg.WITH_DACT else None
    workspace_fake = (
        g2d(Float32, shape=(cute.sym_int32(), cute.sym_int32()), align=4)
        if cfg.WITH_DBIAS
        else None
    )

    from cutlass.utils import HardwareInfo  # pylint: disable=import-outside-toplevel

    sm_count = HardwareInfo().get_device_multiprocessor_count()
    kernel_obj = MXFP8GroupQuantizeKernel(cfg, sm_count)
    return cute.compile(
        kernel_obj,
        g2d(cfg.DTYPE),  # mX
        g2d(out_dtype),  # mO_row
        g2d(out_dtype),  # mO_col
        g1d(scale_dtype),  # mS_row
        g1d(scale_dtype),  # mS_col
        g1d(cutlass.Int64, align=8),  # mOffsets
        g1d(cutlass.Int64, align=8),  # mFirstDims
        g1d(cutlass.Int64, align=8),  # mLastDims
        tensormaps_fake,  # mTensormaps
        noop_fake,  # mNoop
        act_input_fake,  # mActInput
        workspace_fake,  # mWorkspace
        cute.runtime.make_fake_stream(),
        options="--enable-tvm-ffi",
    )


def get_mxfp8_group_quantization_function(
    fn_name: str,
    dtype: str,
    fp8_dtype: str,
    rowwise: bool,
    colwise: bool,
    shape_rep: str,
    with_gemm_swizzled_scales: bool,
    with_dbias: bool,
    with_dact: bool,
    with_act: bool,
    activation: str,
) -> bool:
    """Compile the grouped MXFP8 quantize kernel for this config and register it in the TVM-FFI
    global registry under EXACTLY `fn_name` (the key the C++ dispatcher built; Python treats it as
    an opaque name). Returns True if a kernel is successfully registered under `fn_name` (the C++
    side then fetches it with GetGlobal(fn_name)); False if the config is unsupported, so the caller
    caches the negative result and falls back to the CUDA C++ grouped kernel.
    """
    try:
        # Already registered (e.g. by a prior call) -> supported.
        if tvm_ffi.get_global_func(fn_name, allow_missing=True) is not None:
            return True

        major, minor = device_compute_capability()
        if major < 10:
            logger.warning(
                "CuTeDSL MXFP8 backend requires compute capability >= 10.0 (Blackwell), "
                "but detected %d.%d; falling back to the CUDA C++ kernel.",
                major,
                minor,
            )
            return False

        try:
            cfg = MXFP8GroupQuantizeConfig(
                dtype=dtype,
                fp8_dtype=fp8_dtype,
                rowwise=rowwise,
                colwise=colwise,
                shape_rep=shape_rep,
                with_gemm_swizzled_scales=with_gemm_swizzled_scales,
                with_dbias=with_dbias,
                with_dact=with_dact,
                with_act=with_act,
                activation=activation,
            )
        except ValueError as e:
            logger.warning(
                "CuTeDSL grouped MXFP8 backend does not support this config, "
                "falling back to the CUDA C++ kernel: %s",
                e,
            )
            return False

        logger.debug("Compiling CuTeDSL grouped MXFP8 quantization kernel for %s", cfg)
        compiled = compile_cutedsl_function_from_cfg(cfg)
        # Register the native TVM-FFI function rather than its Python argument-parsing wrapper;
        # see get_mxfp8_quantization_function for why.
        native = getattr(compiled, "__tvm_ffi_object__", lambda: None)()
        tvm_ffi.register_global_func(
            fn_name, native if native is not None else compiled, override=True
        )
        return True
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(
            "CuTeDSL grouped MXFP8 kernel compilation & registration failed, falling back to the"
            " CUDA C++ kernel: %s",
            e,
        )
        # Unconditionally fallback to CUDA path because we can't tell if this exception is
        # transient or permanent.
        return False
