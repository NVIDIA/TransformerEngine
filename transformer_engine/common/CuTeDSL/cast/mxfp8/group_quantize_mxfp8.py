# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Grouped MXFP8 quantization kernel implemented in CuTeDSL.

Strategy-aligned port of group_quantize_mxfp8.cuh. The scheduling, descriptor
management and per-tensor scale addressing mirror the CUDA kernel one-for-one:

  * `is_single_tensor` reps (SAME_BOTH_DIMS, VARYING_FIRST_DIM) launch ONE CTA per
    128x128 job and address the group through ONE static TMA descriptor with
    global job offsets -- the CUDA `tensor_map_*_static` "direct mapper" path.
    For SAME_BOTH_DIMS the CUDA grid is linearized per tensor (X, Y-in-tensor,
    tensor) while this kernel linearizes it flat over the stacked rows; both
    require every member's row count to be a multiple of 128, and under
    that precondition the two decode to the identical (job_start_row, job_id_X)
    for every job index.
  * the other reps launch grid=(workers_per_tensor, num_tensors) and bind
    tensor_id to blockIdx.y, so a CTA grid-strides only within its own tensor and
    never re-resolves which tensor a job belongs to. They get per-tensor
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
scales of out-of-bounds columns in a job (the scale-row padding) are written as 0.

Scope: everything group_quantize_mxfp8.cuh covers except 2D block scaling -- the
cast-noop flag, fused activation (IS_ACT) and activation derivative (IS_DACT), dbias,
compact and GEMM-swizzled scales, rowwise and/or colwise, and all four shape
representations. Differences from CUDA:
  * the grouped amax pointer is accepted and left untouched, as the CUDA kernel does;
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
from cutlass.cute.testing import assert_ as runtime_assert
from cutlass.utils import TensorMapManager, TensorMapUpdateMode
from cutlass.utils import HardwareInfo
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
        # Only can view the grouped tensor as one tensor when the last dim is same across all tensors
        self.IS_SINGLE_TENSOR = shape_rep in (SAME_BOTH_DIMS, VARYING_FIRST_DIM)
        self.MAX_NORM_RCP = (
            FP8E4M3_MAX_NORM_RCP if fp8_dtype == "Float8E4M3" else FP8E5M2_MAX_NORM_RCP
        )

        self.WITH_GEMM_SWIZZLED_SCALES = with_gemm_swizzled_scales
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

    # Target persistent CTA count per SM for sizing the grid
    STATIC_PERSISTENT_WORKERS_PER_SM = 24
    # The shape of one pipeline stage processed by a CTA
    TILE_ROWS = 32
    TILE_COLS = 128
    PIPELINE_DEPTH = 2
    # The number of elements in a tile
    ELTS_PER_TILE = TILE_ROWS * TILE_COLS
    # CTA shape
    THREADS_PER_CTA = 128
    NUM_WARPS = THREADS_PER_CTA // THREADS_PER_WARP  # 4
    THREADS_X = TILE_COLS // MXFP8_BLOCK_SCALING_SIZE  # 4
    THREADS_Y = THREADS_PER_CTA // THREADS_X  # 32
    # How many elements a thread handles in a wave
    PACK_SIZE = 4
    # How many waves needed to handle a MXFP8 block
    WAVES = MXFP8_BLOCK_SCALING_SIZE // PACK_SIZE  # 8
    # How many threads per bank -- for avoiding bank conflicts
    THREADS_PER_BANK = (32 * 4) // MXFP8_BLOCK_SCALING_SIZE  # 4

    def __init__(self, cfg: MXFP8GroupQuantizeConfig, SM_COUNT: int):
        self.cfg = cfg
        self.SM_COUNT = SM_COUNT
        # A CTA processes (NUM_TILES_Y, NUM_TILES_X) tiles, NUM_STAGES tiles in total
        self.NUM_TILES_X = 2 if cfg.SHAPE_REP == VARYING_BOTH_DIMS else 1
        self.NUM_TILES_Y = 4
        self.NUM_STAGES = self.NUM_TILES_X * self.NUM_TILES_Y
        self.ELTS_PER_CTA = self.ELTS_PER_TILE * self.NUM_STAGES
        # The CUDA kernel honors the noop flag only without fused activations or dbias.
        self.CHECK_NOOP_FLAG = not (cfg.WITH_ACT or cfg.WITH_DACT or cfg.WITH_DBIAS)
        # The colwise pass caches the activation in the input tile for the rowwise pass.
        # Note: ReLU is fused into the conversion instead, which yields the same bytes.
        self.CACHE_ACTIVATION = (
            (cfg.WITH_ACT or cfg.WITH_DACT)
            and cfg.ROWWISE
            and cfg.COLWISE
            and cfg.ACTIVATION != "relu"
        )
        # Prefer to reduce dbias in the colwise pass if quantized in columnwise, else in the rowwise pass.
        self.DBIAS_IN_COLWISE = cfg.WITH_DBIAS and cfg.COLWISE
        self.DBIAS_IN_ROWWISE = cfg.WITH_DBIAS and not cfg.COLWISE

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
            mS_t, (self.TILE_ROWS, self.TILE_COLS // MXFP8_BLOCK_SCALING_SIZE)
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
            mS_t, (self.TILE_ROWS // MXFP8_BLOCK_SCALING_SIZE, self.TILE_COLS)
        )

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mO_row: Optional[cute.Tensor],
        mO_col: Optional[cute.Tensor],
        mS_row: cute.Tensor,
        mS_col: cute.Tensor,
        mOffsets: cute.Tensor,  # int64[num_tensors + 1], CSR element offsets
        mFirstDims: Optional[
            cute.Tensor
        ],  # int64[num_tensors] (VARYING_FIRST_DIM / VARYING_BOTH_DIMS)
        mLastDims: Optional[
            cute.Tensor
        ],  # int64[num_tensors] (VARYING_LAST_DIM / VARYING_BOTH_DIMS)
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
        if cutlass.const_expr(cfg.SHAPE_REP == VARYING_BOTH_DIMS):
            runtime_assert(
                first_logical_dim == 1, "VARYING_BOTH_DIMS requires logical shape [1, total]"
            )

        num_tensors = mTensormaps.shape[0]
        runtime_assert(num_tensors > 0, "Grouped quantization requires at least one tensor")

        # A TMA atom copies a TILE at a time
        smem_tile_layout = cute.make_ordered_layout((self.TILE_ROWS, self.TILE_COLS), order=(1, 0))
        cta_tiler = (self.TILE_ROWS, self.TILE_COLS)

        # TMA atom for loading the input tensor and the activation input (if WITH_DACT)
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

        # TMA atom for storing the rowwise and colwise outputs (if enabled)
        op_store = cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_out_row = None
        tma_dst_out_row = None
        if cutlass.const_expr(cfg.ROWWISE):
            tma_atom_out_row, tma_dst_out_row = cpasync.make_tiled_tma_atom(
                op_store, mO_row, smem_tile_layout, cta_tiler, num_multicast=1
            )
        tma_atom_out_col = None
        tma_dst_out_col = None
        if cutlass.const_expr(cfg.COLWISE):
            tma_atom_out_col, tma_dst_out_col = cpasync.make_tiled_tma_atom(
                op_store, mO_col, smem_tile_layout, cta_tiler, num_multicast=1
            )

        if cutlass.const_expr(cfg.IS_SINGLE_TENSOR):
            # How many CTAs does the grouped tensor have in both directions
            jobs_Y = cute.ceil_div(Int32(first_logical_dim), (self.TILE_ROWS * self.NUM_TILES_Y))
            jobs_X = cute.ceil_div(Int32(last_logical_dim), self.TILE_COLS * self.NUM_TILES_X)
            # Flatten it to an 1D grid
            grid = [jobs_X * jobs_Y, 1, 1]
        else:
            # A placeholder for the kernel signature only; we won't use it in non-single tensor cases
            jobs_X = None
            # Estimate the total jobs across the group; each job has NUM_TILES_X * NUM_TILES_Y tiles
            if cutlass.const_expr(cfg.SHAPE_REP == VARYING_BOTH_DIMS):
                # Note: when VARYING_BOTH_DIMS, the first_logical_dim must be 1
                estimated_jobs = cute.ceil_div(
                    Int32(first_logical_dim) * Int32(last_logical_dim), self.ELTS_PER_CTA
                )
            elif cutlass.const_expr(cfg.SHAPE_REP == VARYING_LAST_DIM):
                # Same as VARYING_BOTH_DIMS but we divide 128 before multiplying to avoid overflowing Int32
                # because the first logical dimension is always 128-aligned when not VARYING_BOTH_DIMS
                # so they are equivalent
                estimated_jobs = cute.ceil_div(
                    (Int32(first_logical_dim) // 128) * Int32(last_logical_dim),
                    self.ELTS_PER_CTA // 128,
                )
            else:
                raise ValueError(f"unexpected shape representation {cfg.SHAPE_REP!r}")

            # Divide the persistent worker budget evenly across tensors, with at least
            # one worker per tensor. Each worker may process several jobs.
            requested_CTAs_per_tensor = cutlass.max(
                Int32(1),
                Int32(self.SM_COUNT * self.STATIC_PERSISTENT_WORKERS_PER_SM) // Int32(num_tensors),
            )
            # In average how many jobs per tensor (only an average, the actual jobs per tensor may vary)
            average_jobs_per_tensor = cutlass.max(
                Int32(1), cute.ceil_div(estimated_jobs, Int32(num_tensors))
            )
            # Don't launch more CTAs than the average jobs per tensor in case
            # STATIC_PERSISTENT_WORKERS_PER_SM causes redundancy
            CTAs_per_tensor = cutlass.min(requested_CTAs_per_tensor, average_jobs_per_tensor)
            grid = [CTAs_per_tensor, Int32(num_tensors), 1]

        # Only the multi-tensor representations need per-tensor descriptors.
        if cutlass.const_expr(not cfg.IS_SINGLE_TENSOR):
            self._update_descriptors_kernel(
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
            jobs_X,
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
            block=[self.THREADS_PER_CTA, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def _update_descriptors_kernel(
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
        """Update the per-tensor TMA descriptors for the group quantization kernel.

        mTensormaps: int64[num_tensors, NUM_WORKSPACE_SLOTS, 16], where the slots are:
        - 0: input tensor
        - 1: rowwise output tensor
        - 2: colwise output tensor
        - 3: activation input tensor (only with WITH_DACT)
        - 4: metadata (rows, cols, base_offset)
        """
        cfg = self.cfg

        # For the descriptor update kernel, grid=[num_tensors, 1, 1]
        tensor_id, _, _ = cute.arch.block_idx()
        # Figure out how many rows and columns this tensor has, and where its first element is in the group.
        if cutlass.const_expr(cfg.SHAPE_REP in (VARYING_FIRST_DIM, VARYING_BOTH_DIMS)):
            rows = Int32(mFirstDims[tensor_id])
        else:
            rows = Int32(first_logical_dim)
        if cutlass.const_expr(cfg.SHAPE_REP in (VARYING_LAST_DIM, VARYING_BOTH_DIMS)):
            cols = Int32(mLastDims[tensor_id])
        else:
            cols = Int32(last_logical_dim)
        base_offset = Int64(mOffsets[tensor_id])

        # Same diagnostics as get_tensor_rows_num / get_tensor_cols_num. Like NVTE_DEVICE_ERROR
        # in a release build, they only print.
        tidx, _, _ = cute.arch.thread_idx()
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

        meta = mTensormaps[(tensor_id, META_SLOT, None)]
        meta[0] = Int64(rows)
        meta[1] = Int64(cols)
        meta[2] = base_offset

        tmap = TensorMapManager(TensorMapUpdateMode.GMEM, BYTES_PER_TENSORMAP)
        # Obtain the pointers of these descriptors
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
                        tensor.iterator.toint() + base_offset * (elt_dtype.width // 8),
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
                tmap.init_tensormap_from_atom(tma_atom_orow, desc_orow, 0)
                descs.append(desc_orow)
            if cutlass.const_expr(cfg.COLWISE):
                views.append(member_view(mO_col, cfg.FP8_DTYPE))
                atoms.append(tma_atom_ocol)
                tmap.init_tensormap_from_atom(tma_atom_ocol, desc_ocol, 0)
                descs.append(desc_ocol)
            if cutlass.const_expr(cfg.WITH_DACT):
                views.append(member_view(mActInput, dtype))
                atoms.append(tma_atom_act)
                tmap.init_tensormap_from_atom(tma_atom_act, desc_act, 0)
                descs.append(desc_act)
            tmap.fence_tensormap_initialization()
            # Update descriptors in global memory with views and atoms
            tmap.update_tensormap(
                tuple(views),
                tuple(atoms),
                tuple(descs),
                0,
                (),  # smem staging is unused in GMEM update mode
            )

    @cute.jit
    def _make_shared_storage(
        self,
        smem: cutlass.Constexpr,
        dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
    ):
        """Allocate pipeline buffers and optional activation input and dbias storage."""
        FP8_DTYPE = self.cfg.FP8_DTYPE
        tile_layout = cute.make_layout(
            ((self.TILE_ROWS, self.TILE_COLS), self.PIPELINE_DEPTH),
            stride=((self.TILE_COLS, 1), self.TILE_ROWS * self.TILE_COLS),
        )

        sX = None
        sO_row = None
        sO_col = None
        sActInput = None

        if cutlass.const_expr(self.cfg.ROWWISE and self.cfg.COLWISE):

            @cute.struct
            class SharedStorage:
                mbar: cute.struct.MemRange[cute.Int64, 2 * self.PIPELINE_DEPTH]
                sX: cute.struct.Align[
                    cute.struct.MemRange[
                        dtype, self.TILE_ROWS * self.TILE_COLS * self.PIPELINE_DEPTH
                    ],
                    128,
                ]
                sO_row: cute.struct.Align[
                    cute.struct.MemRange[
                        FP8_DTYPE, self.TILE_ROWS * self.TILE_COLS * self.PIPELINE_DEPTH
                    ],
                    128,
                ]
                sO_col: cute.struct.Align[
                    cute.struct.MemRange[
                        FP8_DTYPE, self.TILE_ROWS * self.TILE_COLS * self.PIPELINE_DEPTH
                    ],
                    128,
                ]

            storage = smem.allocate(SharedStorage)
            sX = storage.sX.get_tensor(tile_layout)
            sO_row = storage.sO_row.get_tensor(tile_layout)
            sO_col = storage.sO_col.get_tensor(tile_layout)

        elif cutlass.const_expr(self.cfg.ROWWISE):

            @cute.struct
            class SharedStorage:
                mbar: cute.struct.MemRange[cute.Int64, 2 * self.PIPELINE_DEPTH]
                sX: cute.struct.Align[
                    cute.struct.MemRange[
                        dtype, self.TILE_ROWS * self.TILE_COLS * self.PIPELINE_DEPTH
                    ],
                    128,
                ]
                sO_row: cute.struct.Align[
                    cute.struct.MemRange[
                        FP8_DTYPE, self.TILE_ROWS * self.TILE_COLS * self.PIPELINE_DEPTH
                    ],
                    128,
                ]

            storage = smem.allocate(SharedStorage)
            sX = storage.sX.get_tensor(tile_layout)
            sO_row = storage.sO_row.get_tensor(tile_layout)

        elif cutlass.const_expr(self.cfg.COLWISE):

            @cute.struct
            class SharedStorage:
                mbar: cute.struct.MemRange[cute.Int64, 2 * self.PIPELINE_DEPTH]
                sX: cute.struct.Align[
                    cute.struct.MemRange[
                        dtype, self.TILE_ROWS * self.TILE_COLS * self.PIPELINE_DEPTH
                    ],
                    128,
                ]
                sO_col: cute.struct.Align[
                    cute.struct.MemRange[
                        FP8_DTYPE, self.TILE_ROWS * self.TILE_COLS * self.PIPELINE_DEPTH
                    ],
                    128,
                ]

            storage = smem.allocate(SharedStorage)
            sX = storage.sX.get_tensor(tile_layout)
            sO_col = storage.sO_col.get_tensor(tile_layout)

        if cutlass.const_expr(self.cfg.WITH_DACT):

            @cute.struct
            class DactStorage:
                sActInput: cute.struct.Align[
                    cute.struct.MemRange[
                        dtype, self.TILE_ROWS * self.TILE_COLS * self.PIPELINE_DEPTH
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
                cute.make_layout((self.THREADS_Y, self.TILE_COLS), stride=(DBIAS_BUFF_WIDTH, 1))
            )

        return storage.mbar.data_ptr(), sX, sO_row, sO_col, sActInput, sDbias

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
        jobs_X,
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
                jobs_X,
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
        jobs_X,
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

        smem = cutlass.utils.SmemAllocator()
        # Layouts for sX, sO_row, sO_col, sActInput:
        # ((TILE_ROWS, TILE_COLS), PIPELINE_DEPTH):((TILE_COLS, 1), TILE_ROWS * TILE_COLS)
        # Layout for sDbias:
        # (THREADS_Y, TILE_COLS):(THREADS_X * (MXFP8_BLOCK_SCALING_SIZE + 1), 1)
        mbar_ptr, sX, sO_row, sO_col, sActInput, sDbias = self._make_shared_storage(smem, dtype)

        tx_count = self.TILE_ROWS * self.TILE_COLS * dtype.width // 8
        if cutlass.const_expr(cfg.WITH_DACT):
            tx_count *= 2
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=mbar_ptr,
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

        gX_tiled = cute.zipped_divide(tma_src, (self.TILE_ROWS, self.TILE_COLS))
        tXsX, tXgX = cpasync.tma_partition(tma_atom_x, 0, cute.make_layout(1), sX, gX_tiled)

        tXsA = None
        tXgA = None
        if cutlass.const_expr(cfg.WITH_DACT):
            gA_tiled = cute.zipped_divide(tma_src_act, (self.TILE_ROWS, self.TILE_COLS))
            tXsA, tXgA = cpasync.tma_partition(
                tma_atom_act, 0, cute.make_layout(1), sActInput, gA_tiled
            )

        tXsO_row = None
        tXgO_row = None
        if cutlass.const_expr(cfg.ROWWISE):
            gO_row_tiled = cute.zipped_divide(tma_dst_out_row, (self.TILE_ROWS, self.TILE_COLS))
            tXsO_row, tXgO_row = cpasync.tma_partition(
                tma_atom_out_row, 0, cute.make_layout(1), sO_row, gO_row_tiled
            )

        tXsO_col = None
        tXgO_col = None
        if cutlass.const_expr(cfg.COLWISE):
            gO_col_tiled = cute.zipped_divide(tma_dst_out_col, (self.TILE_ROWS, self.TILE_COLS))
            tXsO_col, tXgO_col = cpasync.tma_partition(
                tma_atom_out_col, 0, cute.make_layout(1), sO_col, gO_col_tiled
            )

        tmap = TensorMapManager(TensorMapUpdateMode.GMEM, BYTES_PER_TENSORMAP)
        cute.arch.sync_threads()

        # If the CTA has work to do
        has_work = Boolean(True)
        # Metadata of the tensor that owns this job
        tensor_rows = Int32(0)
        tensor_cols = Int32(0)
        # Element offset of this tensor within the group: Int64 (CUDA uses size_t), since a
        # group can exceed 2^31 elements even when every individual extent is small.
        tensor_base = Int64(0)
        # Job's starting row and id in this individual tensor / global single tensor
        job_start_row = Int32(0)
        job_id_X = Int32(0)

        first_job_id = Int32(0)
        jobs_in_tensor = Int32(1)
        job_stride = Int32(1)
        jobs_X_in_tensor = Int32(1)

        if cutlass.const_expr(cfg.IS_SINGLE_TENSOR):
            # grid = [jobs_X * jobs_Y, 1, 1]
            job_id_Y = Int32(bidx) // jobs_X
            job_id_X = Int32(bidx) % jobs_X
            # View the grouped tensor as a single tensor of shape (first_logical_dim, last_logical_dim)
            tensor_rows = Int32(first_logical_dim)
            tensor_cols = Int32(last_logical_dim)
            # Which row does this job start from
            job_start_row = job_id_Y * (self.TILE_ROWS * self.NUM_TILES_Y)
            if cutlass.const_expr(cfg.SHAPE_REP == VARYING_FIRST_DIM):
                total_elts = Int64(mOffsets[mOffsets.shape[0] - 1])
                # If the starting element is already beyond the last element of the group, this CTA can stop
                if Int64(job_start_row) * Int64(last_logical_dim) >= total_elts:
                    has_work = Boolean(False)
            # When SAME_BOTH_DIM, M is always divisible by 128, which is exactly TILE_ROWS * NUM_TILES_Y,
            # so no need to check for the last job's starting row being beyond the last row of the tensor.
        else:
            # grid = [workers_per_tensor, Int32(num_tensors), 1]
            tensor_id = Int32(bidy)
            # Extract tensor's metadata
            meta = mTensormaps[(tensor_id, META_SLOT, None)]
            tensor_rows = Int32(meta[0])
            tensor_cols = Int32(meta[1])
            tensor_base = Int64(meta[2])
            if tensor_rows > 0 and tensor_cols > 0:
                # How many jobs does this tensor have in both directions
                jobs_X_in_tensor = cute.ceil_div(tensor_cols, (self.TILE_COLS * self.NUM_TILES_X))
                jobs_Y_in_tensor = cute.ceil_div(tensor_rows, (self.TILE_ROWS * self.NUM_TILES_Y))
                # How many jobs does this tensor have
                jobs_in_tensor = jobs_X_in_tensor * jobs_Y_in_tensor
                # Which job (1D index) does this CTA start from, which is also their worker ID
                first_job_id = Int32(bidx)
                # gdx is how many CTAs are assigned to this tensor
                job_stride = Int32(gdx)
                # If my first job is already beyond the tensor's last job, I have no work to do
                if first_job_id >= jobs_in_tensor:
                    has_work = Boolean(False)
            else:
                # This tensor is empty, so this CTA has no work to do
                has_work = Boolean(False)

        partitions = (tXsX, tXgX, tXsA, tXgA, tXsO_row, tXgO_row, tXsO_col, tXgO_col)
        atoms = (tma_atom_x, tma_atom_act, tma_atom_out_row, tma_atom_out_col)

        if has_work:
            # For single tensor case, each CTA only processes one job
            if cutlass.const_expr(cfg.IS_SINGLE_TENSOR):
                # For single tensor case we don't use tensor descriptors
                descs = (None, None, None, None)

                # Rowwise scales span the whole group: members are stacked on 128-row
                # boundaries, so each member's swizzled tiles follow the previous member's.
                row_scales = None
                if cutlass.const_expr(cfg.ROWWISE):
                    row_scales = self._rowwise_scales(mS_row, Int64(0), tensor_rows, tensor_cols)

                col_scales = None
                col_scale_row_start = job_start_row
                col_scale_rows = tensor_rows
                if cutlass.const_expr(cfg.COLWISE):
                    col_scale_base = Int64(0)
                    if cutlass.const_expr(cfg.WITH_GEMM_SWIZZLED_SCALES):
                        # Colwise swizzled scale indices restart at each member and depend
                        # on its rows (process_colwise_stage), so address the member that
                        # owns this job.
                        member_rows = Int32(0)
                        member_row_start = Int32(0)
                        if cutlass.const_expr(cfg.SHAPE_REP == SAME_BOTH_DIMS):
                            member_rows = tensor_rows // Int32(num_tensors)
                            member_row_start = job_start_row // member_rows * member_rows
                        else:
                            member_id = self._find_tensor_from_offsets(
                                mOffsets,
                                num_tensors,
                                Int64(job_start_row) * Int64(tensor_cols),
                            )
                            member_rows = Int32(mFirstDims[member_id])
                            member_row_start = Int32(
                                Int64(mOffsets[member_id]) // Int64(tensor_cols)
                            )
                        col_scale_base = (
                            Int64(member_row_start)
                            * Int64(cute.round_up(tensor_cols, 128))
                            // MXFP8_BLOCK_SCALING_SIZE
                        )
                        col_scale_row_start = job_start_row - member_row_start
                        col_scale_rows = member_rows
                    col_scales = self._colwise_scales(
                        mS_col, col_scale_base, col_scale_rows, tensor_cols
                    )

                cute.arch.sync_threads()

                self._process_job_strip(
                    job_start_row,
                    job_id_X,
                    tensor_rows,
                    tensor_cols,
                    row_scales,
                    col_scales,
                    col_scale_row_start,
                    col_scale_rows,
                    job_start_row // (self.TILE_ROWS * self.NUM_TILES_Y),
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
                # For non-single tensor case, we use persistent kernel so each CTA keeps processing jobs until none is left
                desc_x = tmap.get_tensormap_ptr(mTensormaps[(tensor_id, 0, None)].iterator)
                desc_out_row = tmap.get_tensormap_ptr(mTensormaps[(tensor_id, 1, None)].iterator)
                desc_out_col = tmap.get_tensormap_ptr(mTensormaps[(tensor_id, 2, None)].iterator)
                desc_act = tmap.get_tensormap_ptr(
                    mTensormaps[(tensor_id, ACT_INPUT_SLOT, None)].iterator
                )

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

                # Make sure all threads see the updated descriptors and scales before processing any jobs.
                cute.arch.sync_threads()

                # Grid-stride over this tensor's own jobs; the descriptors never change.
                job_id = first_job_id
                job_finished = Boolean(False)
                while not job_finished:
                    job_id_Y_in_tensor = job_id // jobs_X_in_tensor
                    job_id_X_in_tensor = job_id % jobs_X_in_tensor
                    job_start_row_in_tensor = job_id_Y_in_tensor * (
                        self.TILE_ROWS * self.NUM_TILES_Y
                    )
                    if cutlass.const_expr(self.NUM_TILES_X == 1):
                        self._process_job_strip(
                            job_start_row_in_tensor,
                            job_id_X_in_tensor,
                            tensor_rows,
                            tensor_cols,
                            row_scales,
                            col_scales,
                            job_start_row_in_tensor,
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
                        # The job's column tiles in order, stopping at the tensor's last
                        # column like the CUDA kernel's stages_X = DIVUP(chunk_cols, TILE_DIM_X).
                        job_start_col = job_id_X_in_tensor * (self.TILE_COLS * self.NUM_TILES_X)
                        tiles_X = cutlass.min(
                            Int32(self.NUM_TILES_X),
                            cute.ceil_div(tensor_cols - job_start_col, self.TILE_COLS),
                        )
                        for stage_X in cutlass.range(tiles_X, unroll=1):
                            self._process_job_strip(
                                job_start_row_in_tensor,
                                job_id_X_in_tensor * self.NUM_TILES_X + stage_X,
                                tensor_rows,
                                tensor_cols,
                                row_scales,
                                col_scales,
                                job_start_row_in_tensor,
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
                    # Find the next job to process
                    job_id = job_id + job_stride
                    if job_id >= jobs_in_tensor:
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
    def _process_job_strip(
        self,
        job_start_row,  # Row offset of this job (global for single-tensor, else tensor-local)
        column_tile_id,  # Column-tile index within the tensor
        rows,  # Rows of the rowwise-scale view (the group for single-tensor, else the tensor)
        cols,  # Number of columns in this tensor
        row_scales,  # Rowwise scales tiled per stage, rows counted like job_start_row
        col_scales,  # Colwise scales tiled per stage
        col_scale_row_start,  # Row of this job in the colwise-scale view
        col_scale_rows,  # Rows of the colwise-scale view
        dbias_row,  # Row of the dbias workspace this job reduces into
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
        """Quantize NUM_TILES_Y vertically stacked tiles in one column strip of a job."""
        cfg = self.cfg
        _, _, tma_atom_out_row, tma_atom_out_col = atoms
        _, _, _, _, tXsO_row, tXgO_row, tXsO_col, tXgO_col = partitions
        _, _, desc_out_row, desc_out_col = descs
        job_start_col = column_tile_id * self.TILE_COLS

        # This job's coordinates in the tile grid (32x128 TMA boxes, not elements).
        tile_id_Y = job_start_row // self.TILE_ROWS
        tile_id_X = column_tile_id
        col_scale_tile_Y = col_scale_row_start // self.TILE_ROWS

        # Per-job dbias accumulators, in the CUDA kernel's summation order: a running
        # column sum over the job's rows (colwise), or per-thread partial sums over its
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

        for stage in cutlass.range_constexpr(self.NUM_TILES_Y):
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
                    (col_scale_tile_Y + stage) * self.TILE_ROWS,
                    job_start_col,
                    col_scale_rows,
                    cols,
                    ACTIVATION=cfg.ACTIVATION,
                    DTYPE=cfg.DTYPE,
                    FP8_DTYPE=cfg.FP8_DTYPE,
                    SWIZZLE=cfg.WITH_GEMM_SWIZZLED_SCALES,
                    TILE_X=self.TILE_COLS,
                    TILE_Y=self.TILE_ROWS,
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
                    row_tile * self.TILE_ROWS,
                    job_start_col,
                    rows,
                    cols,
                    ACTIVATION=None if self.CACHE_ACTIVATION else cfg.ACTIVATION,
                    DTYPE=cfg.DTYPE,
                    FP8_DTYPE=cfg.FP8_DTYPE,
                    TILE_X=self.TILE_COLS,
                    TILE_Y=self.TILE_ROWS,
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
            if cutlass.const_expr(stage + self.PIPELINE_DEPTH < self.NUM_TILES_Y):
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
            # One partial-dbias row per job, as in the CUDA kernel's dbias_workspace.
            dbias_x = job_start_col + tidx
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
        # The buffer is rewritten by the next job.
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

    out_col_fake = (
        cute.runtime.make_fake_compact_tensor(
            out_dtype,
            logical_shape,
            stride_order=(1, 0),
            memspace=cute.AddressSpace.gmem,
            assumed_align=16,
        )
        if cfg.COLWISE
        else None
    )

    out_row_fake = (
        cute.runtime.make_fake_compact_tensor(
            out_dtype,
            logical_shape,
            stride_order=(1, 0),
            memspace=cute.AddressSpace.gmem,
            assumed_align=16,
        )
        if cfg.ROWWISE
        else None
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
    act_input_fake = (
        cute.runtime.make_fake_compact_tensor(
            cfg.DTYPE,
            logical_shape,
            stride_order=(1, 0),
            memspace=cute.AddressSpace.gmem,
            assumed_align=16,
        )
        if cfg.WITH_DACT
        else None
    )
    workspace_fake = (
        cute.runtime.make_fake_compact_tensor(
            Float32,
            (cute.sym_int32(), cute.sym_int32()),
            stride_order=(1, 0),
            memspace=cute.AddressSpace.gmem,
            assumed_align=4,
        )
        if cfg.WITH_DBIAS
        else None
    )

    first_dims_fake = (
        cute.runtime.make_fake_compact_tensor(
            cutlass.Int64,
            (cute.sym_int32(),),
            stride_order=(0,),
            memspace=cute.AddressSpace.gmem,
            assumed_align=8,
        )
        if cfg.SHAPE_REP in (VARYING_FIRST_DIM, VARYING_BOTH_DIMS)
        else None
    )

    last_dims_fake = (
        cute.runtime.make_fake_compact_tensor(
            cutlass.Int64,
            (cute.sym_int32(),),
            stride_order=(0,),
            memspace=cute.AddressSpace.gmem,
            assumed_align=8,
        )
        if cfg.SHAPE_REP in (VARYING_LAST_DIM, VARYING_BOTH_DIMS)
        else None
    )

    sm_count = HardwareInfo().get_device_multiprocessor_count()
    kernel_obj = MXFP8GroupQuantizeKernel(cfg, sm_count)
    return cute.compile(
        kernel_obj,
        cute.runtime.make_fake_compact_tensor(  # mX
            cfg.DTYPE,
            logical_shape,
            stride_order=(1, 0),
            memspace=cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        out_row_fake,  # mO_row
        out_col_fake,  # mO_col
        cute.runtime.make_fake_compact_tensor(  # mS_row
            scale_dtype,
            (cute.sym_int32(),),
            stride_order=(0,),
            memspace=cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        cute.runtime.make_fake_compact_tensor(  # mS_col
            scale_dtype,
            (cute.sym_int32(),),
            stride_order=(0,),
            memspace=cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        cute.runtime.make_fake_compact_tensor(  # mOffsets
            cutlass.Int64,
            (cute.sym_int32(),),
            stride_order=(0,),
            memspace=cute.AddressSpace.gmem,
            assumed_align=8,
        ),
        first_dims_fake,  # mFirstDims
        last_dims_fake,  # mLastDims
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
