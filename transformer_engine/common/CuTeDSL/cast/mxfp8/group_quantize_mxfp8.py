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
expressed with PipelineTmaAsync instead of hand-rolled mbarriers, and
out-of-bounds scale padding is skipped rather than explicitly zeroed (the CUDA
kernel writes 0 there; downstream only consumes the meaningful region).

Scope: cast-only (no dbias / activation / dact / amax), compact (non-swizzled)
scales, rowwise and/or colwise. VARYING_BOTH_DIMS is not handled here (its
logical shape [1, total] is not tileable); the C++ bridge falls back to CUDA.

Known gap vs CUDA: the CUDA kernel validates on device that every member's first
dim is a multiple of 128 (get_tensor_rows_num -> NVTE_DEVICE_ERROR). This kernel
shares the precondition but cannot check it -- CuTeDSL has no device-side assert,
and for VARYING_FIRST_DIM the per-tensor extents live in device memory, so the
C++ bridge cannot check them either without a sync. A violating group mis-tiles
silently here where CUDA raises.
"""

"""
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
from typing import Type

import cutlass
from cutlass import cute
from cutlass import pipeline
from cutlass import Boolean, Int32, Int64, Float8E8M0FNU
from cutlass.cute.nvgpu import cpasync
from cutlass.tensor_utils import TensorMapManager, TensorMapUpdateMode
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module
import tvm_ffi

from transformer_engine.common.CuTeDSL.utils import (
    str_to_cutlass_dtype,
    device_compute_capability,
)
from transformer_engine.common.CuTeDSL.cast.mxfp8.quantize_mxfp8 import (
    MXFP8_BLOCK_SCALING_SIZE,
    FP8E4M3_MAX_NORM_RCP,
    FP8E5M2_MAX_NORM_RCP,
    quantize_rowwise_mxfp8,
    quantize_colwise_mxfp8,
)

CUTEDSL_DEBUG_LOGGING = os.environ.get("CUTEDSL_DEBUG_LOGGING", "0") == "1"
logger = logging.getLogger("transformer_engine.cutedsl.mxfp8")

THREADS_PER_WARP = 32
BYTES_PER_TENSORMAP = 128
# Descriptor slots per tensor: input, rowwise output, colwise output.
NUM_TENSORMAPS = 3
# One extra slot holds per-tensor (rows, cols, base_elts), so the main kernel never has to
# binary-search the offsets array. Mirrors TensorMapStorage::rows/cols/offsets upstream.
META_SLOT = NUM_TENSORMAPS
NUM_WORKSPACE_SLOTS = NUM_TENSORMAPS + 1

# Shape representations, mirroring ShapeRepresentation in common/utils.cuh.
SAME_BOTH_DIMS = "same_both_dims"
VARYING_FIRST_DIM = "varying_first_dim"
VARYING_LAST_DIM = "varying_last_dim"
SUPPORTED_SHAPE_REPS = (SAME_BOTH_DIMS, VARYING_FIRST_DIM, VARYING_LAST_DIM)


class MXFP8GroupQuantizeConfig:
    """Compile-time config for the grouped MXFP8 quantize kernel."""

    def __init__(self, dtype: str, fp8_dtype: str, rowwise: bool, colwise: bool, shape_rep: str):
        if dtype not in ("fp32", "fp16", "bf16"):
            raise ValueError(f"unknown input dtype {dtype!r}; expected fp32|fp16|bf16")
        self.DTYPE = str_to_cutlass_dtype(dtype)
        self.DTYPE_STR = dtype
        if fp8_dtype not in ("fp8_e4m3fn", "fp8_e5m2"):
            raise ValueError(f"unknown FP8 dtype {fp8_dtype!r}; expected fp8_e4m3fn|fp8_e5m2")
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
            FP8E4M3_MAX_NORM_RCP if fp8_dtype == "fp8_e4m3fn" else FP8E5M2_MAX_NORM_RCP
        )

    def __str__(self):
        return (
            f"MXFP8GroupQuantizeConfig(dtype={self.DTYPE_STR}, fp8_dtype={self.FP8_DTYPE_STR}, "
            f"rowwise={self.ROWWISE}, colwise={self.COLWISE}, shape_rep={self.SHAPE_REP})"
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

    # ---------------------------------------------------------------- helpers
    @cute.jit
    def _tensor_rows_cols(
        self, tensor_id, mFirstDims, mLastDims, first_logical_dim, last_logical_dim
    ):
        """Get the shape (rows, cols) of the tensor by tensor_id."""
        cfg = self.cfg
        if cutlass.const_expr(cfg.SHAPE_REP == VARYING_FIRST_DIM):
            rows = Int32(mFirstDims[tensor_id])
        else:
            rows = Int32(first_logical_dim)
        if cutlass.const_expr(cfg.SHAPE_REP == VARYING_LAST_DIM):
            cols = Int32(mLastDims[tensor_id])
        else:
            cols = Int32(last_logical_dim)
        return rows, cols

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
        mFirstDims: cute.Tensor,  # int64[num_tensors] (VARYING_FIRST_DIM)
        mLastDims: cute.Tensor,  # int64[num_tensors] (VARYING_LAST_DIM)
        mTensormaps: cute.Tensor,  # int64[num_tensors, NUM_TENSORMAPS, 16]
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
        # by construction -- one slot set per member -- so it is the reliable source, and it
        # is only consulted on the multi-tensor path that actually needs the workspace.
        num_tensors = mTensormaps.shape[0]

        smem_tile_layout = cute.make_ordered_layout(
            (self.BUFF_DIM_Y, self.BUFF_DIM_X), order=(1, 0)
        )
        cta_tiler = (self.BUFF_DIM_Y, self.BUFF_DIM_X)
        print(f"mx={mX}, smem_tile_layout={smem_tile_layout}, cta_tiler={cta_tiler}\n")

        op_load = cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_x, tma_src = cpasync.make_tiled_tma_atom(
            op_load, mX, smem_tile_layout, cta_tiler, num_multicast=1
        )
        print(f"tma_atom_x={tma_atom_x}\n")
        print(f"tma_src={tma_src}\n")
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
            # Each CTA handles a block from one individual tensor
            grid = [work_blocks_X * work_blocks_Y, 1, 1]
        else:
            # The work-block grid is per-tensor here, not global: each CTA derives its own
            # block range from its tensor's extents (see the kernel's `else` branch), so
            # work_blocks_X/Y are dead on this path -- the kernel reads them only under
            # IS_SINGLE_TENSOR. Pass dummies rather than computing first*last, which is an
            # ELEMENT count and would wrap Int32 past 2^31 elements.
            work_blocks_Y = Int32(1)
            work_blocks_X = Int32(1)
            # Persistent worker count, mirroring get_launch_config() in
            # group_quantize_mxfp8.cuh. There are SM_COUNT * STATIC_PERSISTENT_BLOCKS_PER_SM
            # workers in total, split evenly across tensors -- but clamped to the average
            # number of chunks a tensor actually holds, so a group of small tensors does not
            # launch CTAs that only reach the `first_block_id >= blocks_in_tensor` early-out.
            n_tensors = cutlass.max(Int32(num_tensors), Int32(1))  # never divide by zero
            # CUDA's DIVUP(elts_total, CHUNK_DIM_Y * TILE_DIM_X) / STAGES_X, where TILE_DIM_X
            # is BUFF_DIM_X here and STAGES_X is 1 for every rep this kernel supports. The
            # element product would wrap Int32 past 2^31 elements, so divide the first extent
            # by CHUNK_DIM_Y up front -- it is exact, the kernel is compiled with
            # sym_int32(divisibility=CHUNK_DIM_Y) on that extent -- and keep the whole
            # estimate in Int32. Feeding an Int64 into the grid poisons the tile arithmetic
            # downstream ('cute.make_tile' expects width=32).
            estimated_work_blocks = cute.ceil_div(
                (Int32(first_logical_dim) // self.CHUNK_DIM_Y) * Int32(last_logical_dim),
                self.ELTS_PER_CHUNK // self.CHUNK_DIM_Y,
            )
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
            # Each group of workers_per_tensor CTAs serves one tensor, and we launch
            # num_tensors such groups to cover the whole group.
            grid = [workers_per_tensor, Int32(num_tensors), 1]

        # Only manually create descriptors for the non-single-tensor case because we will need to manually
        # overwrite the descriptors as we visit different groups
        if cutlass.const_expr(not cfg.IS_SINGLE_TENSOR):
            self.update_descriptors_kernel(
                mX,
                mO_row,
                mO_col,
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
            ).launch(grid=[num_tensors, 1, 1], block=[THREADS_PER_WARP, 1, 1], stream=stream)

        self.kernel(
            mS_row,
            mS_col,
            mOffsets,
            mFirstDims,
            mLastDims,
            mTensormaps,
            first_logical_dim,
            last_logical_dim,
            num_tensors,
            work_blocks_X,
            work_blocks_Y,
            mX.element_type,
            tma_atom_x,
            tma_src,
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
    ):
        """One CTA per tensor: point that tensor's TMA descriptors at its own block.

        CuTeDSL analog of common::update_tma_descriptors writing g_tensor_maps[].
        """
        cfg = self.cfg
        tensor_id, _, _ = cute.arch.block_idx()
        rows, cols = self._tensor_rows_cols(
            tensor_id, mFirstDims, mLastDims, first_logical_dim, last_logical_dim
        )
        base_elts = Int64(mOffsets[tensor_id])

        # Publish this tensor's geometry for the main kernel (written even when empty).
        meta = mTensormaps[(tensor_id, META_SLOT, None)]
        meta[0] = Int64(rows)
        meta[1] = Int64(cols)
        meta[2] = base_elts

        tmap = TensorMapManager(TensorMapUpdateMode.GMEM, BYTES_PER_TENSORMAP)
        desc_x = tmap.get_tensormap_ptr(mTensormaps[(tensor_id, 0, None)].iterator)
        desc_orow = tmap.get_tensormap_ptr(mTensormaps[(tensor_id, 1, None)].iterator)
        desc_ocol = tmap.get_tensormap_ptr(mTensormaps[(tensor_id, 2, None)].iterator)

        # Zero-sized groups: creating a descriptor with a zero extent is invalid,
        # so skip (the main kernel skips these jobs via job_has_work).
        if rows > 0 and cols > 0:
            gX = cute.make_tensor(
                cute.make_ptr(
                    dtype,
                    mX.iterator.toint() + base_elts * (dtype.width // 8),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                cute.make_layout((rows, cols), stride=(cols, 1)),
            )
            gO_row = cute.make_tensor(
                cute.make_ptr(
                    cfg.FP8_DTYPE,
                    mO_row.iterator.toint() + base_elts,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                cute.make_layout((rows, cols), stride=(cols, 1)),
            )
            gO_col = cute.make_tensor(
                cute.make_ptr(
                    cfg.FP8_DTYPE,
                    mO_col.iterator.toint() + base_elts,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                cute.make_layout((rows, cols), stride=(cols, 1)),
            )

            tmap.init_tensormap_from_atom(tma_atom_x, desc_x, 0)
            if cutlass.const_expr(cfg.ROWWISE):
                tmap.init_tensormap_from_atom(tma_atom_orow, desc_orow, 0)
            if cutlass.const_expr(cfg.COLWISE):
                tmap.init_tensormap_from_atom(tma_atom_ocol, desc_ocol, 0)
            tmap.fence_tensormap_initialization()

            if cutlass.const_expr(cfg.ROWWISE and cfg.COLWISE):
                tmap.update_tensormap(
                    (gX, gO_row, gO_col),
                    (tma_atom_x, tma_atom_orow, tma_atom_ocol),
                    (desc_x, desc_orow, desc_ocol),
                    0,
                    (),  # smem staging is unused in GMEM update mode
                )
            elif cutlass.const_expr(cfg.ROWWISE):
                tmap.update_tensormap(
                    (gX, gO_row),
                    (tma_atom_x, tma_atom_orow),
                    (desc_x, desc_orow),
                    0,
                    (),  # smem staging is unused in GMEM update mode
                )
            else:
                tmap.update_tensormap(
                    (gX, gO_col),
                    (tma_atom_x, tma_atom_ocol),
                    (desc_x, desc_ocol),
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
        mLastDims,
        mTensormaps,
        first_logical_dim,
        last_logical_dim,
        num_tensors,
        work_blocks_X,
        work_blocks_Y,
        dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
        tma_atom_x,
        tma_src,
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

        storage = cutlass.utils.SmemAllocator().allocate(SharedStorage)
        tile_layout = cute.make_layout(
            ((self.BUFF_DIM_Y, self.BUFF_DIM_X), self.PIPELINE_DEPTH),
            stride=((self.BUFF_DIM_X, 1), self.BUFF_DIM_Y * self.BUFF_DIM_X),
        )
        sX = storage.sX.get_tensor(tile_layout)
        print(f"sX={sX}\n")
        sO_row = storage.sO_row.get_tensor(tile_layout)
        sO_col = storage.sO_col.get_tensor(tile_layout)

        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar.data_ptr(),
            num_stages=self.PIPELINE_DEPTH,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.NUM_WARPS),
            tx_count=self.BUFF_DIM_Y * self.BUFF_DIM_X * dtype.width // 8,
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
        gO_row_tiled = cute.zipped_divide(tma_dst_out_row, (self.BUFF_DIM_Y, self.BUFF_DIM_X))
        tXsO_row, tXgO_row = cpasync.tma_partition(
            tma_atom_out_row, 0, cute.make_layout(1), sO_row, gO_row_tiled
        )
        gO_col_tiled = cute.zipped_divide(tma_dst_out_col, (self.BUFF_DIM_Y, self.BUFF_DIM_X))
        tXsO_col, tXgO_col = cpasync.tma_partition(
            tma_atom_out_col, 0, cute.make_layout(1), sO_col, gO_col_tiled
        )
        print(f"tma_atom_x={tma_atom_x}\n")
        print(f"tma_src={tma_src}\n")
        print(f"gX_tiled={gX_tiled}\n")
        print(f"tXsX={tXsX}\n")
        print(f"tXgX={tXgX}\n")

        tmap = TensorMapManager(TensorMapUpdateMode.GMEM, BYTES_PER_TENSORMAP)
        cute.arch.sync_threads()

        # If the CTA has work to do
        has_work = Boolean(True)
        # Metadata of the tensor that owns this block
        tensor_rows = Int32(0)
        tensor_cols = Int32(0)
        # Element offset of this tensor within the group: Int64 (CUDA uses size_t), since a
        # group can exceed 2^31 elements even when every individual extent is small.
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
                # Check if this block contains any valid tokens since the last offset <= logical_first_dim * logical_last_dim
                # Last CSR offset == the group's total element count. Indexed off the
                # array's own length so it does not depend on num_tensors.
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
                block_columns_in_tensor = cute.ceil_div(tensor_cols, self.CHUNK_DIM_X)
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

        if has_work:
            if cutlass.const_expr(cfg.IS_SINGLE_TENSOR):
                # For single tensor case we don't use tensor descriptors
                desc_x = desc_out_row = desc_out_col = None

                cute.arch.sync_threads()

                self._process_block(
                    block_offset_Y,
                    block_id_X,
                    tensor_rows,
                    tensor_cols,
                    tensor_base,
                    desc_x,
                    desc_out_row,
                    desc_out_col,
                    mS_row,
                    mS_col,
                    first_logical_dim,
                    tmap,
                    warp_idx,
                    tidx,
                    sX,
                    sO_row,
                    sO_col,
                    tXsX,
                    tXgX,
                    tXsO_row,
                    tXgO_row,
                    tXsO_col,
                    tXgO_col,
                    tma_atom_x,
                    tma_atom_out_row,
                    tma_atom_out_col,
                    mainloop_pipeline,
                    prod_state,
                    cons_state,
                )
            else:
                # For multi-tensor case, retrieve tensor descriptors we processed early in the prologue kernel
                desc_x = tmap.get_tensormap_ptr(mTensormaps[(tensor_id, 0, None)].iterator)
                desc_out_row = tmap.get_tensormap_ptr(mTensormaps[(tensor_id, 1, None)].iterator)
                desc_out_col = tmap.get_tensormap_ptr(mTensormaps[(tensor_id, 2, None)].iterator)
                # Acquire the descriptors on ONE thread, as the CUDA kernel does
                # (`leading_thread` in group_quantize_mxfp8.cuh); the sync_threads below
                # publishes it CTA-wide. Running the tensormap acquire fence on all 128
                # threads is correct but very expensive -- it more than doubles the
                # kernel time on the multi-tensor path (4096x14336 bidirectional:
                # 133 us -> 58 us), since the cost scales with threads x descriptors.
                if tidx == 0:
                    tmap.fence_tensormap_update(desc_x)
                    if cutlass.const_expr(cfg.ROWWISE):
                        tmap.fence_tensormap_update(desc_out_row)
                    if cutlass.const_expr(cfg.COLWISE):
                        tmap.fence_tensormap_update(desc_out_col)

                cute.arch.sync_threads()

                # Grid-stride over this tensor's own chunks; the descriptors never change.
                block_id = first_block_id
                job_finished = Boolean(False)
                while not job_finished:
                    block_id_Y_in_tensor = block_id // block_columns_in_tensor
                    block_id_X_in_tensor = block_id % block_columns_in_tensor
                    self._process_block(
                        block_id_Y_in_tensor * self.CHUNK_DIM_Y,
                        block_id_X_in_tensor,
                        tensor_rows,
                        tensor_cols,
                        tensor_base,
                        desc_x,
                        desc_out_row,
                        desc_out_col,
                        mS_row,
                        mS_col,
                        first_logical_dim,
                        tmap,
                        warp_idx,
                        tidx,
                        sX,
                        sO_row,
                        sO_col,
                        tXsX,
                        tXgX,
                        tXsO_row,
                        tXgO_row,
                        tXsO_col,
                        tXgO_col,
                        tma_atom_x,
                        tma_atom_out_row,
                        tma_atom_out_col,
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
        tma_atom_x,
        tXgX,
        tXsX,
        tmap,
        desc_x,
    ):
        """Emit one 32x128 TMA load into the current pipeline buffer.

        Caller gates this on warp 0 and advances `prod_state` afterwards -- the advance
        must happen outside the gate or the mutated SSA values stay trapped in the scf.if.
        """
        # Wait for the consumer to finish using this SMEM buffer
        pipeline_obj.producer_acquire(prod_state)
        if cutlass.const_expr(self.cfg.IS_SINGLE_TENSOR):
            cute.copy(
                tma_atom_x,
                tXgX[(None, (tile_y, tile_x))],
                tXsX[(None, prod_state.index)],
                tma_bar_ptr=pipeline_obj.producer_get_barrier(prod_state),
            )
        else:
            # Every member shares tXgX's tile-coordinate arithmetic (the coefficients are
            # just the tile size); tma_desc_ptr supplies this member's geometry.
            cute.copy(
                tma_atom_x,
                tXgX[(None, (tile_y, tile_x))],
                tXsX[(None, prod_state.index)],
                tma_bar_ptr=pipeline_obj.producer_get_barrier(prod_state),
                tma_desc_ptr=tmap.get_tensormap_ptr(desc_x, cute.AddressSpace.generic),
            )
        # Notify the consumer that this SMEM buffer is ready for consumption
        pipeline_obj.producer_commit(prod_state)

    @cute.jit
    def _process_block(
        self,
        block_offset_Y_in_tensor,  # Row offset of this chunk (global for single-tensor, else tensor-local)
        block_id_X_in_tensor,  # Column-chunk index within the tensor
        rows,  # Number of rows in this tensor (logical shape)
        cols,  # Number of columns in this tensor (logical shape)
        tensor_base,  # Int64 element offset of this tensor in the group (0 for single-tensor)
        desc_x,  # Per-tensor descriptors, already acquired by the caller (None if single-tensor)
        desc_out_row,
        desc_out_col,
        mS_row,  # Grouped rowwise scales
        mS_col,  # Grouped colwise scales
        first_logical_dim,  # First dimension of the grouped tensor
        tmap,  # TensorMapManager for managing TMA descriptors
        warp_idx,
        tidx,
        sX,  # SMEM input for this block
        sO_row,  # SMEM rowwise output for this block
        sO_col,  # SMEM colwise output for this block
        tXsX,  # Tiled sX for TMA
        tXgX,  # Tiled gX for TMA
        tXsO_row,  # Tiled sO_row for TMA
        tXgO_row,  # Tiled gO_row for TMA
        tXsO_col,  # Tiled sO_col for TMA
        tXgO_col,  # Tiled gO_col for TMA
        tma_atom_x,  # TMA atom for input
        tma_atom_out_row,  # TMA atom for rowwise output
        tma_atom_out_col,  # TMA atom for colwise output
        mainloop_pipeline: cutlass.pipeline.PipelineTmaAsync,
        prod_state,
        cons_state,
    ):
        """Quantize one 128x128 chunk in STAGES slices of BUFF_DIM_Y rows."""
        cfg = self.cfg
        block_offset_X = block_id_X_in_tensor * self.CHUNK_DIM_X

        # This tensor's rowwise scales
        scale_rows = Int32(first_logical_dim) if cutlass.const_expr(cfg.IS_SINGLE_TENSOR) else rows
        scale_base = (
            Int64(0)
            if cutlass.const_expr(cfg.IS_SINGLE_TENSOR)
            else tensor_base // Int64(MXFP8_BLOCK_SCALING_SIZE)
        )

        if cutlass.const_expr(cfg.ROWWISE):
            # Rowwise scale's divisibility guarantee: (128, 4)
            scale_row_stride = cute.round_up(cute.ceil_div(cols, MXFP8_BLOCK_SCALING_SIZE), 4)
            # Advance to this tensor's rowwise scales
            mS_row_t = cute.make_tensor(
                cute.make_ptr(
                    Float8E8M0FNU,
                    mS_row.iterator.toint() + scale_base,
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                ),
                cute.make_layout((scale_rows, scale_row_stride), stride=(scale_row_stride, 1)),
            )
            mS_row_tiled = cute.zipped_divide(
                mS_row_t, (self.BUFF_DIM_Y, self.CHUNK_DIM_X // MXFP8_BLOCK_SCALING_SIZE)
            )

        if cutlass.const_expr(cfg.COLWISE):
            # Colwise scale's divisibility guarantee: (4, 128)
            scale_col_stride = cute.round_up(cols, 128)
            # Advance to this tensor's colwise scales
            mS_col_t = cute.make_tensor(
                cute.make_ptr(
                    Float8E8M0FNU,
                    mS_col.iterator.toint() + scale_base,
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                ),
                cute.make_layout(
                    (scale_rows // MXFP8_BLOCK_SCALING_SIZE, scale_col_stride),
                    stride=(scale_col_stride, 1),
                ),
            )
            mS_col_tiled = cute.zipped_divide(
                mS_col_t, (self.BUFF_DIM_Y // MXFP8_BLOCK_SCALING_SIZE, self.CHUNK_DIM_X)
            )

        cute.arch.sync_threads()

        # This chunk's coordinates in the tile grid (32x128 TMA boxes, not elements).
        tile_id_Y = block_offset_Y_in_tensor // self.BUFF_DIM_Y
        tile_id_X = block_id_X_in_tensor

        # Fill every buffer up front, then issue one more each time a stage is consumed.
        for prologue_stage in cutlass.range_constexpr(self.PIPELINE_DEPTH):
            if warp_idx == 0:
                self._issue_load(
                    mainloop_pipeline,
                    prod_state,
                    tile_id_Y + prologue_stage,
                    tile_id_X,
                    tma_atom_x,
                    tXgX,
                    tXsX,
                    tmap,
                    desc_x,
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
            row_tile = tile_id_Y + stage
            tile_row_start = block_offset_Y_in_tensor + stage * self.BUFF_DIM_Y

            if cutlass.const_expr(cfg.COLWISE):
                quantize_colwise_mxfp8(
                    sX_tile,
                    None,
                    sO_col[(None, cons_state.index)],
                    cute.flatten(mS_col_tiled[(None, (row_tile, tile_id_X))]),
                    cfg.MAX_NORM_RCP,
                    tile_row_start,
                    block_offset_X,
                    scale_rows,
                    cols,
                    ACTIVATION=None,
                    DTYPE=cfg.DTYPE,
                    FP8_DTYPE=cfg.FP8_DTYPE,
                    SWIZZLE=False,
                    TILE_X=self.BUFF_DIM_X,
                    TILE_Y=self.BUFF_DIM_Y,
                    SKIP_MASKING=False,
                )
            if cutlass.const_expr(cfg.ROWWISE):
                quantize_rowwise_mxfp8(
                    sX_tile,
                    None,
                    sO_row[(None, cons_state.index)],
                    cute.flatten(mS_row_tiled[(None, (row_tile, tile_id_X))]),
                    cfg.MAX_NORM_RCP,
                    tile_row_start,
                    block_offset_X,
                    scale_rows,
                    cols,
                    ACTIVATION=None,
                    DTYPE=cfg.DTYPE,
                    FP8_DTYPE=cfg.FP8_DTYPE,
                    TILE_X=self.BUFF_DIM_X,
                    TILE_Y=self.BUFF_DIM_Y,
                    WAVES=self.WAVES,
                    THREADS_PER_BANK=self.THREADS_PER_BANK,
                    PACK_SIZE=self.PACK_SIZE,
                    SKIP_MASKING=False,
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
                        tma_atom_x,
                        tXgX,
                        tXsX,
                        tmap,
                        desc_x,
                    )
                prod_state.advance()

            # Write result to GMEM via TMA
            if warp_idx == 0:
                if cutlass.const_expr(cfg.ROWWISE):
                    if cutlass.const_expr(cfg.IS_SINGLE_TENSOR):
                        cute.copy(
                            tma_atom_out_row,
                            tXsO_row[(None, cons_state.index)],
                            tXgO_row[(None, (row_tile, tile_id_X))],
                        )
                    else:
                        cute.copy(
                            tma_atom_out_row,
                            tXsO_row[(None, cons_state.index)],
                            tXgO_row[(None, (row_tile, tile_id_X))],
                            tma_desc_ptr=tmap.get_tensormap_ptr(
                                desc_out_row, cute.AddressSpace.generic
                            ),
                        )
                if cutlass.const_expr(cfg.COLWISE):
                    if cutlass.const_expr(cfg.IS_SINGLE_TENSOR):
                        cute.copy(
                            tma_atom_out_col,
                            tXsO_col[(None, cons_state.index)],
                            tXgO_col[(None, (row_tile, tile_id_X))],
                        )
                    else:
                        cute.copy(
                            tma_atom_out_col,
                            tXsO_col[(None, cons_state.index)],
                            tXgO_col[(None, (row_tile, tile_id_X))],
                            tma_desc_ptr=tmap.get_tensormap_ptr(
                                desc_out_col, cute.AddressSpace.generic
                            ),
                        )
                # Commit all TMA operations of this iteration
                cute.arch.cp_async_bulk_commit_group()

            cons_state.advance()


def compile_cutedsl_function_from_cfg(cfg: MXFP8GroupQuantizeConfig):
    """Return the compiled CuTeDSL function object for the given grouped config."""
    # CUDA requires the group's first logical dim to be a multiple of 128 (and each
    # tensor's rows/cols likewise); MXFP8 needs the last dim divisible by 32.
    sym_M = cute.sym_int32(divisibility=128)
    sym_N = cute.sym_int32(divisibility=MXFP8_BLOCK_SCALING_SIZE)
    logical_shape = (sym_M, sym_N)

    out_dtype = cfg.FP8_DTYPE
    scale_dtype = cutlass.Float8E8M0FNU

    def g2d(dtype, align=16):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            logical_shape,
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
    in_fake = g2d(cfg.DTYPE)
    out_row_fake = g2d(out_dtype)
    out_col_fake = g2d(out_dtype)
    scale_row_fake = g1d(scale_dtype)
    scale_col_fake = g1d(scale_dtype)
    offsets_fake = g1d(cutlass.Int64, align=8)
    first_dims_fake = g1d(cutlass.Int64, align=8)
    last_dims_fake = g1d(cutlass.Int64, align=8)
    tensormaps_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int64,
        (cute.sym_int32(), NUM_WORKSPACE_SLOTS, BYTES_PER_TENSORMAP // 8),
        stride_order=(2, 1, 0),
        memspace=cute.AddressSpace.gmem,
        assumed_align=128,
    )

    from cutlass.utils import HardwareInfo  # pylint: disable=import-outside-toplevel

    sm_count = HardwareInfo().get_device_multiprocessor_count()
    kernel_obj = MXFP8GroupQuantizeKernel(cfg, sm_count)
    return cute.compile(
        kernel_obj,
        in_fake,
        out_row_fake,
        out_col_fake,
        scale_row_fake,
        scale_col_fake,
        offsets_fake,
        first_dims_fake,
        last_dims_fake,
        tensormaps_fake,
        cute.runtime.make_fake_stream(),
        options="--enable-tvm-ffi",
    )


# TEMPORARY (demo only): same as compile_cutedsl_function_from_cfg but with every
# extent pinned to a compile-time constant, so traced layouts print concrete numbers
# (e.g. `(384,256):(1@1,1@0)`) instead of `?{div=128}`. Delete once done inspecting.
def compile_cutedsl_function_from_cfg_static(
    cfg: MXFP8GroupQuantizeConfig,
    M_total: int,
    N: int,
    num_tensors: int,
    scale_row_numel: int,
    scale_col_numel: int,
):
    """Compile with fully static shapes. Only accepts inputs of exactly these extents."""
    logical_shape = (M_total, N)

    def g2d(dtype, align=16):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            logical_shape,
            stride_order=(1, 0),
            memspace=cute.AddressSpace.gmem,
            assumed_align=align,
        )

    def g1d(dtype, numel, align=4):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            (numel,),
            stride_order=(0,),
            memspace=cute.AddressSpace.gmem,
            assumed_align=align,
        )

    scale_dtype = cutlass.Float8E8M0FNU
    tensormaps_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int64,
        (num_tensors, NUM_WORKSPACE_SLOTS, BYTES_PER_TENSORMAP // 8),
        stride_order=(2, 1, 0),
        memspace=cute.AddressSpace.gmem,
        assumed_align=128,
    )

    from cutlass.utils import HardwareInfo  # pylint: disable=import-outside-toplevel

    sm_count = HardwareInfo().get_device_multiprocessor_count()
    return cute.compile(
        MXFP8GroupQuantizeKernel(cfg, sm_count),
        g2d(cfg.DTYPE),
        g2d(cfg.FP8_DTYPE),
        g2d(cfg.FP8_DTYPE),
        g1d(scale_dtype, scale_row_numel),
        g1d(scale_dtype, scale_col_numel),
        g1d(cutlass.Int64, num_tensors + 1, align=8),
        g1d(cutlass.Int64, num_tensors, align=8),
        g1d(cutlass.Int64, num_tensors, align=8),
        tensormaps_fake,
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
) -> bool:
    """Compile the grouped MXFP8 quantize kernel for this config and register it in the
    TVM-FFI global registry under EXACTLY `fn_name`. Returns True on success (the C++
    dispatcher then fetches it with GetGlobal(fn_name)); False if unsupported, so the
    caller falls back to the CUDA C++ grouped kernel.
    """
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
        )
    except ValueError as e:
        logger.warning(
            "CuTeDSL grouped MXFP8 backend does not support this config, "
            "falling back to the CUDA C++ kernel: %s",
            e,
        )
        return False

    logger.debug("Compiling CuTeDSL grouped MXFP8 quantization kernel for %s", cfg)
    try:
        compiled = compile_cutedsl_function_from_cfg(cfg)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(
            "CuTeDSL grouped MXFP8 kernel compilation failed, "
            "falling back to the CUDA C++ kernel: %s",
            e,
        )
        return False
    tvm_ffi.register_global_func(fn_name, compiled, override=True)
    return True
