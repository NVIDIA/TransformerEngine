# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MXFP8 quantization kernel implemented in CuTeDSL.

Replicates the core logic of quantize_mxfp8.cuh: given a 2D tensor of BF16/FP16
values, quantize to MXFP8 format (FP8E4M3 data + E8M0 per-block scales).

Matches the C++ kernel's tile dimensions and thread layout:
  CHUNK_DIM_Y = 64, CHUNK_DIM_X = 64, THREADS_PER_CHUNK = 64
  BUFF_DIM_Y  = 32, BUFF_DIM_X  = 64, STAGES = 2
  SCALE_DIM   = 32 (elements per MXFP8 scaling block)

Grid: (ceil(N / 64), ceil(M / 64))
Each block processes a 64x64 chunk in 2 stages of 32x64 tiles loaded into
shared memory.
"""

from transformer_engine.common.cutedsl.cutedsl_utils import str_to_te_dtype, torch_to_cutlass_dtype
from transformer_engine.pytorch.tensor.flex_tensor import FlexQuantizer

from typing import Literal, Optional, Type, Union

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import Float32, Int32, Uint8

import hashlib
import tvm_ffi as _tvm_ffi

from transformer_engine.common.cutedsl.cast.quantization_utils import (
    FP8E4M3_MAX_NORM_RCP,
    FP8E5M2_MAX_NORM_RCP,
    _bitcast_f32_to_i32,
    quantize_colwise_mxfp8,
    quantize_rowwise_mxfp8,
)

# MXFP8 settings
MXFP8_BLOCK_SIZE = 32 # Number of elements per MXFP8 scale block. They will share the same E8M0 scale factor
SCALE_DIM = MXFP8_BLOCK_SIZE

# Double-buffering for async copy + compute overlap
BUFFER_NUM = 2

# Vectorised access constants for bank-conflict avoidance (rowwise pass)
PACK_SIZE = 4                              # Elements per vector load
WAVES = SCALE_DIM // PACK_SIZE             # Each thread reads 8 waves with each wave reads 4 packed bf16, so it reads a whole MXFP8 block in total
THREADS_PER_WARP = 32
TOTAL_BANKS_WIDTH = (32 * 4) // 1  # 32 banks × 4 bytes, in bytes (uint8 stride)
THREADS_PER_BANK = TOTAL_BANKS_WIDTH // SCALE_DIM  # 4 threads per bank

# Tiling sizes
NUM_STAGES = 2 # Pipeline depth of the producer/consumer ring buffer for the TMA-G2S input loads (PipelineTmaAsync stage count)
NUM_TILES = 2 # Each CTA process 2 tiles along the Y (row, slowest-changing) dimension
TILE_Y = 32 # Each tile has 32 rows, so each CTA handles 32 * 2 rows in total
TILE_X = 64 # Each tile has 64 columns

# CTA size
THREADS_PER_CHUNK = 64
NUM_WARPS = THREADS_PER_CHUNK // 32

class MXFP8QuantizeConfig:

    def __init__(self, 
                 dtype: torch.dtype,
                 dtype_row: Union[Literal["e4m3", "e5m2", "none"]], 
                 dtype_column: Union[Literal["e4m3", "e5m2", "none"]],
                 with_gemm_swizzled_scales=False):
        self.DTYPE = dtype
        self.DTYPE_ROW = dtype_row
        self.ROWWISE = dtype_row != "none"
        self.COLUMNWISE = dtype_column != "none"
        self.DTYPE_COLUMN = dtype_column
        self.WITH_GEMM_SWIZZLED_SCALES = with_gemm_swizzled_scales
        # No amax / no activation for the MXFP8 path (kept as
        # const-expr-false flags so the kernel's amax/activation branches are
        # dead-stripped). NVFP4 (which needs amax) will flip these later.
        self.WITH_AMAX = False
        self.ACTIVATION = None

class MXFP8QuantizeKernel:
    """MXFP8 quantization with shared-memory tiling (rowwise, colwise, or both).

    Matches C++ kernel's BIDIMENSIONAL scaling mode:
      Grid  (ceil(N/64), ceil(M/64))
      Block (64)
      Each block processes a 64x64 chunk in 2 stages of 32x64.

    Per stage, the tile is loaded into shared memory once.  The colwise
    pass reads columns from smem first, then the rowwise pass reads rows.
    When both directions are enabled, global memory is read only once per
    element — matching the C++ single-pass behaviour.

    Thread mappings (per stage):
      Colwise:  thread tidx handles column tidx, 32 rows (stride BUFF_DIM_X).
      Rowwise:  tid_Y = tidx // 2 -> row, tid_X = tidx % 2 -> scale-block.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,  # Input tensor to quantize
        mO_row: Optional[cute.Tensor], mS_row: Optional[cute.Tensor],  # Rowwise data + scale
        mA_row: Optional[cute.Tensor],  # Rowwise amax (None for MXFP8)
        mO_col: Optional[cute.Tensor], mS_col: Optional[cute.Tensor],  # Colwise data + scale
        mA_col: Optional[cute.Tensor],  # Colwise amax (None for MXFP8)
        rng_state: Optional[cute.Tensor],  # SR seed/offset (None when SR disabled)
        stream: cuda.CUstream,  # launch stream (C++ passes the handle as an int64 scalar)
    ):
        M = mX.shape[0]
        N = mX.shape[1]
        cfg = self.cfg
        num_scale_cols = N // SCALE_DIM
        num_scale_rows = M // SCALE_DIM
        
        # Rewrap mS_row / mS_col with the GEMM-swizzled layout when requested.
        # Wrapper passes in a tensor with the compact (M, N/32):(N/32, 1) layout
        # (built from a compact fake-ptr at compile time), and we re-view the
        # underlying buffer here so the per-block scale stores below land at the
        # cuBLAS-swizzled byte offsets.
        # See https://docs.nvidia.com/cuda/cublas/#d-block-scaling-factors-layout
        # and swizzle_demo.svg for a visual of the byte permutation.
        if cutlass.const_expr(cfg.WITH_GEMM_SWIZZLED_SCALES):
            num_tiles_M = (M + 127) // 128
            num_tiles_SC = (num_scale_cols + 3) // 4   # = ceil(N / 128)
            num_tiles_SR = (num_scale_rows + 3) // 4   # = ceil(M / 128)
            num_tiles_N = (N + 127) // 128
            # row i = i_lo + 32 * (i_hi + 4 * tile_Y);  col j = j_lo + 4 * tile_X.
            # Within one 128×4 tile: byte = i_lo*16 + i_hi*4 + j_lo.
        
            # Tile-major outer dims add (tile_Y * num_tiles_SC + tile_X) * 512.
            # For example, if M=256, N=512, then num_scale_cols = 16, num_scale_rows = 8, and num_tiles_M=2, num_tiles_SC=4, num_tiles_SR=2, num_tiles_N=4
            # The swizzled layout is ((32, 4, 2), (4, 4)):((16, 4, 2048), (1, 512))
            if cutlass.const_expr(cfg.ROWWISE):
                mS_row = cute.make_tensor(
                    mS_row.iterator,
                    cute.make_layout(
                        ((32, 4, num_tiles_M), (4, num_tiles_SC)),
                        stride=((16, 4, num_tiles_SC * 512), (1, 512)),
                    ),
                )
            # Colwise: same swizzle, axes swap roles — col axis gets the 32×4
            # inner decomp, scale-row axis gets the 4-extent dim.
            if cutlass.const_expr(cfg.COLUMNWISE):
                mS_col = cute.make_tensor(
                    mS_col.iterator,
                    cute.make_layout(
                        ((4, num_tiles_SR), (32, 4, num_tiles_N)),
                        stride=((1, 512), (16, 4, num_tiles_SR * 512)),
                    ),
                )
        
        # Divide by the STAGE tile (TILE_Y, TILE_X // SCALE_DIM), not the CTA
        # tile. Each CTA owns NUM_TILES consecutive row-tiles; the kernel walks
        # them by indexing GRID's row dim with `bidy * NUM_TILES + stage` (cute
        # auto-decomposes a flat coord onto GRID's hierarchical row modes).
        #
        # Critically, this is the only divide that cleanly cuts both layouts:
        #   - compact `(M, N/32):(N/32, 1)`  → SCALE_TILE = (32, 2):(N/32, 1)
        #   - swizzled `((32,4,n_M),(4,n_SC)):((16,4,n_SC·512),(1,512))`
        #                                    → SCALE_TILE = (32, 2):(16, 1)
        # The bigger (TILE_Y * NUM_TILES, ...) divide we used before tangles the
        # swizzle's (32, 4) row hierarchy under flatten + sub-divide chain.
        
        # Declare TMA descriptors on the host side.
        # make_tiled_tma_atom returns the UNTILED gmem tensor with basis strides.
        # Tile it inside the kernel with zipped_divide so each coord selects
        # one (TILE_Y, TILE_X) tile.
        smem_tile_layout = cute.make_ordered_layout((TILE_Y, TILE_X), order=(1, 0))
        cta_tiler = (TILE_Y, TILE_X)
        
        # Input: TMA G2S (bf16/fp16 → smem).
        op_load = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_atom, tma_src = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op_load, mX, smem_tile_layout, cta_tiler, num_multicast=1,
        )
        
        # Output: TMA S2G (uint8 smem → gmem) for both directions. Creating
        # both atoms unconditionally — if a direction is disabled the kernel
        # simply won't dispatch its copy, and the atom cost is negligible.
        op_store = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
        out_smem_layout = cute.make_ordered_layout((TILE_Y, TILE_X), order=(1, 0))
        tma_atom_out_row = None
        tma_dst_out_row = None
        tma_atom_out_col = None
        tma_dst_out_col = None
        if cutlass.const_expr(cfg.ROWWISE):
            tma_atom_out_row, tma_dst_out_row = cute.nvgpu.cpasync.make_tiled_tma_atom(
                op_store, mO_row, out_smem_layout, cta_tiler, num_multicast=1,
            )
        if cutlass.const_expr(cfg.COLUMNWISE):
            tma_atom_out_col, tma_dst_out_col = cute.nvgpu.cpasync.make_tiled_tma_atom(
                op_store, mO_col, out_smem_layout, cta_tiler, num_multicast=1,
            )
        
        # CUDA launches in (0,0), (1,0), (2,0)... order, so we should make N the leading dimension for better access pattern 
        # So consecutive blocks will move along the N dimension first, which is the innermost dimension in memory and we can use cache better
        grid = [
            cute.ceil_div(Int32(N), TILE_X),
            cute.ceil_div(M, TILE_Y * NUM_TILES),
        ]
        block = [THREADS_PER_CHUNK,]
        
        self.kernel(
            mX, mS_row, mS_col, None,  # mAmax = None (no amax for the MXFP8 path)
            mX.element_type,
            tma_atom, tma_src,
            tma_atom_out_row, tma_dst_out_row,
            tma_atom_out_col, tma_dst_out_col,
        ).launch(
            grid=grid,
            block=block,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX,
        mS_row,
        mS_col,
        mAmax,
        dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
        tma_atom, tma_src, # how to use TMA to copy the input
        tma_atom_out_row, tma_dst_out_row, # how to use TMA to copy the rowwise output
        tma_atom_out_col, tma_dst_out_col, # how to use TMA to copy the colwise output
    ):
        cfg = self.cfg

        if cutlass.const_expr(cfg.ROWWISE):
            mS_row = cute.zipped_divide(mS_row, (TILE_Y, TILE_X // SCALE_DIM))
        if cutlass.const_expr(cfg.COLUMNWISE):
            mS_col = cute.zipped_divide(mS_col, (TILE_Y // SCALE_DIM, TILE_X))
        # For M=256, N=512:
        # Non-swizzled: https://kainzhong.github.io/CuTe-Layout-Visualizer/?key=zipped_divide-%28256%2C+16%29%3A%2816%2C+1%29-32%0A2
        # Swizzled: https://kainzhong.github.io/CuTe-Layout-Visualizer/?key=zipped_divide-%28%2832%2C+4%2C+2%29%2C+%284%2C+4%29%29%3A%28%2816%2C+4%2C+2048%29%2C+%281%2C+512%29%29-32%0A2
        # print(f"mS_row after zipped_divide: {mS_row}")

        # FP8 output smem, one 32×64 tile per stage per enabled direction.
        # Allocating a dead sO_col in rowwise-only (or sO_row in colwise-only)
        # bumps per-CTA smem from 12 KB to 16 KB, which drops occupancy and
        # regresses the single-direction path by ~8-10% at 16384^2. Match
        # C++ and only allocate what the active pass actually uses.
        # sAmax holds one f32 per warp for the cross-warp amax reduction —
        # negligible (8 bytes for NUM_WARPS=2) and we always allocate so the
        # struct doesn't fork on a 4th const-expr (cfg.WITH_AMAX) dimension.
        if cutlass.const_expr(cfg.ROWWISE and cfg.COLUMNWISE):
            @cute.struct
            class SharedStorage:
                mbar_storage: cute.struct.MemRange[cute.Int64, 2 * NUM_STAGES]
                sX: cute.struct.Align[
                    cute.struct.MemRange[dtype, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sO_row: cute.struct.Align[
                    cute.struct.MemRange[Uint8, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sO_col: cute.struct.Align[
                    cute.struct.MemRange[Uint8, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sAmax: cute.struct.MemRange[Float32, NUM_WARPS]
        elif cutlass.const_expr(cfg.ROWWISE and not cfg.COLUMNWISE):
            @cute.struct
            class SharedStorage:
                mbar_storage: cute.struct.MemRange[cute.Int64, 2 * NUM_STAGES]
                sX: cute.struct.Align[
                    cute.struct.MemRange[dtype, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sO_row: cute.struct.Align[
                    cute.struct.MemRange[Uint8, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sAmax: cute.struct.MemRange[Float32, NUM_WARPS]
        elif cutlass.const_expr(cfg.ROWWISE):
            @cute.struct
            class SharedStorage:
                mbar_storage: cute.struct.MemRange[cute.Int64, 2 * NUM_STAGES]
                sX: cute.struct.Align[
                    cute.struct.MemRange[dtype, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sO_row: cute.struct.Align[
                    cute.struct.MemRange[Uint8, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sAmax: cute.struct.MemRange[Float32, NUM_WARPS]
        else:
            @cute.struct
            class SharedStorage:
                mbar_storage: cute.struct.MemRange[cute.Int64, 2 * NUM_STAGES]
                sX: cute.struct.Align[
                    cute.struct.MemRange[dtype, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sO_col: cute.struct.Align[
                    cute.struct.MemRange[Uint8, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
                sAmax: cute.struct.MemRange[Float32, NUM_WARPS]
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Per-stage shmem tile is 2D (TILE_Y, TILE_X); stages laid out back-to-back.
        # Mode 0 is hierarchical ((TILE_Y, TILE_X),) so it matches the rank/shape
        # of gX_tiled[(None, (ty, tx))] produced by zipped_divide.
        # sX[(None, stage)] selects one (TILE_Y, TILE_X) tile.
        sX = storage.sX.get_tensor(
            cute.make_layout(
                ((TILE_Y, TILE_X), NUM_STAGES),
                stride=((TILE_X, 1), TILE_Y * TILE_X),
            )
        )
        if cutlass.const_expr(cfg.ROWWISE):
            sO_row = storage.sO_row.get_tensor(
                cute.make_layout(
                    ((TILE_Y, TILE_X), NUM_STAGES),
                    stride=((TILE_X, 1), TILE_Y * TILE_X),
                )
            )
        if cutlass.const_expr(cfg.COLUMNWISE):
            sO_col = storage.sO_col.get_tensor(
                cute.make_layout(
                    ((TILE_Y, TILE_X), NUM_STAGES),
                    stride=((TILE_X, 1), TILE_Y * TILE_X),
                )
            )

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Prefetch TMA descriptor (one-time; warp-0 only).
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom)

        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        # Producer: `arrive_and_expect_tx` is wrapped in `elect_one`, so only
        # one lane of warp 0 arrives on the full barrier per stage → arrive_count=1.
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        # Consumer: `consumer_release` arrives only on the `is_signalling_thread`
        # (lane 0 of each warp), so arrive_count = num_warps per stage.
        num_warps = THREADS_PER_CHUNK // 32
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_warps)

        # Bytes transferred per TMA copy: one (TILE_Y, TILE_X) tile of dtype.
        tx_count = TILE_Y * TILE_X * dtype.width // 8

        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_storage.data_ptr(),
            num_stages=NUM_STAGES,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=tx_count,
            cta_layout_vmnk=None,   # single-CTA, no cluster/multicast
        )

        prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, NUM_STAGES
        )
        cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, NUM_STAGES
        )

        M = mX.shape[0]
        N = mX.shape[1]

        num_tiles = cutlass.min(
            NUM_TILES,
            cute.ceil_div(M - bidy * TILE_Y * NUM_TILES, TILE_Y),
        )

        # Tile the TMA gmem view: ((TILE_Y, TILE_X), (M/TILE_Y, N/TILE_X)).
        gX_tiled = cute.zipped_divide(tma_src, (TILE_Y, TILE_X))

        # Partition sX/gX for the TMA atom (single-CTA, no cluster/multicast).
        tXsX, tXgX = cute.nvgpu.cpasync.tma_partition(
            tma_atom,
            0, # Use the only CTA to do the TMA copy
            cute.make_layout(1), # This cluster only has 1 CTAs
            sX,
            gX_tiled,
        )

        # Same partitioning for S2G outputs: sO_row → mO_row and sO_col → mO_col.
        if cutlass.const_expr(cfg.ROWWISE):
            gO_row_tiled = cute.zipped_divide(tma_dst_out_row, (TILE_Y, TILE_X))
            tXsO_row, tXgO_row = cute.nvgpu.cpasync.tma_partition(
                tma_atom_out_row,
                0,
                cute.make_layout(1),
                sO_row,
                gO_row_tiled,
            )
        if cutlass.const_expr(cfg.COLUMNWISE):
            gO_col_tiled = cute.zipped_divide(tma_dst_out_col, (TILE_Y, TILE_X))
            tXsO_col, tXgO_col = cute.nvgpu.cpasync.tma_partition(
                tma_atom_out_col,
                0,
                cute.make_layout(1),
                sO_col,
                gO_col_tiled,
            )

        # print(f"sX: {sX}\n")
        # print(f"gX_tiled: {gX_tiled}\n")
        # print(f"tXsX: {tXsX}\n")
        # print(f"tXgX: {tXgX}\n")

        # Ensure barrier init is visible to all threads before the pipeline is used.
        cute.arch.sync_threads()

        # ---- Producer: warp 0 issues one TMA copy per tile. ----
        if warp_idx == 0:
            for stage in cutlass.range(num_tiles, unroll=1):
                mainloop_pipeline.producer_acquire(prod_state)
                tile_y = bidy * NUM_TILES + stage
                cute.copy(
                    tma_atom,
                    tXgX[(None, (tile_y, bidx))],
                    tXsX[(None, prod_state.index)],
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(prod_state),
                )
                mainloop_pipeline.producer_commit(prod_state)
                prod_state.advance()

        # Per-thread amax accumulator across all stages of this CTA. Combined
        # with the per-warp redux + cross-warp shmem reduce + atomic at the
        # bottom to produce a global max(|x|) in mAmax. Initialised to 0
        # since amax is non-negative.
        if cutlass.const_expr(cfg.WITH_AMAX):
            block_amax = Float32(0.0)

        # ---- Consumer: all threads quantize each completed tile. ----
        for stage in cutlass.range(num_tiles, unroll=1):
            mainloop_pipeline.consumer_wait(cons_state)
            sX_tile = sX[(None, stage)]          # (TILE_Y, TILE_X) bf16

            """
            grid = [
                cute.ceil_div(Int32(N), TILE_X),
                cute.ceil_div(M, TILE_Y * NUM_TILES),
            ]
            So to obtain the tile that belongs to this CTA.
            """
            # This is just block's x axis idx
            tile_idx_x = bidx
            # Each CTA has `NUM_TILES` tiles. Each stage we need to obtain the tile for that specific stage. 
            # So the tile index along Y dimension is `bidy * NUM_TILES + stage`
            tile_idx_y = bidy * NUM_TILES + stage

            if cutlass.const_expr(cfg.COLUMNWISE):
                # The first row that belongs to this CTA. Each CTA handles NUM_TILES of (TILE_Y, TILE_X) tiles stacked vertically,
                # and each stage handles one of them.
                sO_col_tile = sO_col[(None, stage)]
                mS_col_stage = cute.flatten(mS_col[(None, (tile_idx_y, tile_idx_x))])

                amax_c = self._process_colwise(
                    sX_tile, sO_col_tile,
                    mS_col_stage,
                    tile_idx_y * TILE_Y, bidx * TILE_X, M, N,
                )

            if cutlass.const_expr(cfg.ROWWISE):
                sO_row_tile = sO_row[(None, stage)]
                # mS_row is ((SCALE_TILE), (GRID)) where SCALE_TILE = (32, 2).
                # Each CTA owns NUM_TILES consecutive row-tiles of GRID. cute
                # auto-decomposes the flat row coord `bidy * NUM_TILES + stage`
                # onto GRID's hierarchical row modes — which is the
                # (i_hi, tile_Y) tile-major order for swizzled, and the plain
                # row-tile order for compact. Same source, both layouts correct.
                mS_row_stage = cute.flatten(mS_row[(None, (tile_idx_y, tile_idx_x))])
                # print(f"s0_row_tile: {sO_row_tile}\n")
                # print(f"sO_row: {sO_row}\n")
                # print(f"mS_row: {mS_row}\n")
                # print(f"mS_row_stage: {mS_row_stage}\n")
                # print(f"mS_row_stage: {mS_row_stage}\n")
                amax_r = self._process_rowwise(
                    sX_tile, sO_row_tile,
                    mS_row_stage,
                    tile_idx_y * TILE_Y, bidx * TILE_X, M, N,
                )

            # Make all smem stores (sO_row and/or sO_col) visible to the TMA
            # async proxy, then block-sync so warp 0 sees the fences from all
            # warps before issuing the bulk store(s). Matches the C++
            # reference's fence_proxy + __syncthreads pattern.
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            cute.arch.sync_threads()

            if warp_idx == 0:
                tile_y = bidy * NUM_TILES + stage
                if cutlass.const_expr(cfg.ROWWISE):
                    cute.copy(
                        tma_atom_out_row,
                        tXsO_row[(None, stage)],
                        tXgO_row[(None, (tile_y, bidx))],
                    )
                if cutlass.const_expr(cfg.COLUMNWISE):
                    cute.copy(
                        tma_atom_out_col,
                        tXsO_col[(None, stage)],
                        tXgO_col[(None, (tile_y, bidx))],
                    )
                cute.arch.cp_async_bulk_commit_group()

            mainloop_pipeline.consumer_release(cons_state)
            cons_state.advance()

        # Wait for in-flight TMA stores so data is visible to the host
        # before the kernel returns.
        cute.arch.cp_async_bulk_wait_group(0, read=False)

                # ---- amax block reduction + cross-CTA atomic ----------------------
        # 1) intra-warp: redux.sync.fmax.f32 (sm_80+, single instruction).
        # 2) cross-warp: NUM_WARPS shmem floats + sync_threads.
        # 3) cross-CTA: int-atomic-max on the f32 bit pattern. Since amax is
        #    always ≥ 0, IEEE-754 bit ordering on positives matches float
        #    magnitude ordering, so atomic_max on i32 bits gives the right
        #    result. (atomic_max_float32 also exists but its pointer
        #    normalisation is broken as of this CuTeDSL build.)
        if cutlass.const_expr(cfg.WITH_AMAX):
            warp_amax = cute.arch.warp_redux_sync(block_amax, kind="fmax")
            sAmax = storage.sAmax.get_tensor(cute.make_layout(NUM_WARPS))
            lane_idx = tidx % 32
            if lane_idx == 0:
                sAmax[warp_idx] = warp_amax
            cute.arch.sync_threads()
            if tidx == 0:
                cta_amax = Float32(0.0)
                for w in cutlass.range_constexpr(NUM_WARPS):
                    cta_amax = cute.arch.fmax(cta_amax, sAmax[w])
                amax_i32 = cute.make_tensor(
                    cute.recast_ptr(mAmax.iterator, dtype=Int32),
                    cute.make_layout(1),
                )
                cute.arch.atomic_max(
                    amax_i32.iterator, _bitcast_f32_to_i32(cta_amax),
                )


    @cute.jit
    def _process_rowwise(
        self,
        sX_tile,        # (TILE_Y, TILE_X) bf16/fp16 smem view, post-TMA
        sO_row_tile,    # (TILE_Y, TILE_X) uint8 smem view (rowwise FP8 output)
        mS_row_stage,   # rowwise scale tensor (1D swizzled, or 2D linear)
        tile_row_start, # Int32 — global row of this stage's row 0
        tile_col_start, # Int32 — global col of this CTA's col 0
        M, N,           # Int32 — full input extents, for OOB masking
    ):
        """Rowwise MXFP8 pass: thread `(tid_Y, tid_X) = (tidx % 32, tidx // 32)`
        owns one 32-element scale block (row `tid_Y`, columns `tid_X*32 .. +32`).

        The bank-group swizzle `((w + bank_group) * PACK_SIZE) % SCALE_DIM`
        staggers each 4-thread group's starting wave, which otherwise would
        collide on smem banks since all lanes in a warp read different rows
        at the same column offset.

        Writes quantized bytes into `sO_row_tile` as u32s (one per wave);
        caller is responsible for the TMA S2G flush.
        """
        cfg = self.cfg
        max_norm_rcp = FP8E4M3_MAX_NORM_RCP
        if cutlass.const_expr(cfg.DTYPE_ROW == "e5m2"):
            max_norm_rcp = FP8E5M2_MAX_NORM_RCP
        return quantize_rowwise_mxfp8(
            sX_tile,
            sO_row_tile,
            mS_row_stage,
            max_norm_rcp,
            tile_row_start,
            tile_col_start,
            M,
            N,
            ACTIVATION=None,
            DTYPE=cfg.DTYPE,
            FP8_DTYPE=cfg.DTYPE_ROW,
            TILE_Y=TILE_Y,
            SCALE_DIM=SCALE_DIM,
            WAVES=WAVES,
            THREADS_PER_WARP=THREADS_PER_WARP,
            THREADS_PER_BANK=THREADS_PER_BANK,
            PACK_SIZE=PACK_SIZE
        )

    @cute.jit
    def _process_colwise(
        self,
        sX_tile,        # (TILE_Y, TILE_X) bf16/fp16 smem view, post-TMA
        sO_col_tile,    # (TILE_Y, TILE_X) uint8 smem view (colwise FP8 output)
        mS_col_stage,   # colwise scale tensor (1D swizzled, or 2D linear)
        tile_row_start, # Int32 — global row of this stage's row 0
        tile_col_start, # Int32 — global col of this CTA's col 0
        M, N,           # Int32 — full input extents, for OOB masking
    ):
        """Colwise MXFP8 pass: thread `tidx` owns column `tidx` of the (32, 64)
        smem tile — 32 elements down. Writes quantized bytes into `sO_col_tile`
        so the caller can flush with a TMA S2G — matches C++'s
        `out_colwise_data_sh` + `cp.async.bulk.tensor.2d.shared_to_global`.
        """
        cfg = self.cfg
        max_norm_rcp = FP8E4M3_MAX_NORM_RCP
        if cutlass.const_expr(cfg.DTYPE_COLUMN == "e5m2"):
            max_norm_rcp = FP8E5M2_MAX_NORM_RCP
        return quantize_colwise_mxfp8(
            sX_tile,
            sO_col_tile,
            mS_col_stage,
            max_norm_rcp,
            tile_row_start,
            tile_col_start,
            M, N,
            ACTIVATION=None,
            DTYPE=cfg.DTYPE,
            FP8_DTYPE=cfg.DTYPE_COLUMN,
            SWIZZLE=cfg.WITH_GEMM_SWIZZLED_SCALES,
            TILE_X=TILE_X,
            TILE_Y=TILE_Y,
            SCALE_DIM=SCALE_DIM,
        )

def _cfg_to_fn_name(cfg, M, N) -> str:
    """Deterministic registry key from (cfg, shape)."""
    key = (cfg.DTYPE.__name__, cfg.DTYPE_ROW, cfg.DTYPE_COLUMN,
           int(cfg.ROWWISE), int(cfg.COLUMNWISE),
           int(cfg.WITH_GEMM_SWIZZLED_SCALES), int(cfg.WITH_AMAX),
           cfg.ACTIVATION or "none",
           M, N)
    h = hashlib.sha1(repr(key).encode()).hexdigest()[:16]
    return f"mxfp8_{h}"

_compile_cache_tvm_ffi: dict = {}

def _get_compiled_kernel(cfg, M, N):
    """Compile the kernel for THIS (cfg, M, N) with LITERAL shapes — every
    dimension is a constexpr int, so the AOT wrapper's per-arg type collapses
    to `{ void* data; }` (no shape array, no shape check at call time).

    Tradeoff vs sym_int: one compile per (cfg, M, N) instead of one per cfg.
    Memory cost is small; the per-call saving is ~7-8 us. Cache key already
    includes (M, N) so we never recompile."""
    cache = _compile_cache_tvm_ffi
    fn_name = _cfg_to_fn_name(cfg, M, N)
    if fn_name in cache:
        return cache[fn_name], fn_name

    kernel_obj = MXFP8QuantizeKernel(cfg)

    # TE allocates scale tensors at this padded shape regardless of swizzle
    # (see MXFP8Quantizer::get_scale_shape in transformer_engine/pytorch/csrc):
    #   rowwise:    (roundup(M, 128),    roundup(N // 32, 4))
    #   columnwise: (roundup(M // 32, 4), roundup(N, 128))
    SCALE_R = (((M + 127) // 128) * 128, ((N + 127) // 128) * 4)
    SCALE_C = (((M + 127) // 128) * 4,   ((N + 127) // 128) * 128)
    WS_M = (M + TILE_Y * NUM_TILES - 1) // (TILE_Y * NUM_TILES)

    # stride_order=(1, 0): row-major, dim 1 stride 1. 1D: (0,).
    kw_rm16_2d = dict(stride_order=(1, 0),
                      memspace=cute.AddressSpace.gmem, assumed_align=16)
    kw_rm4_2d  = dict(stride_order=(1, 0),
                      memspace=cute.AddressSpace.gmem, assumed_align=4)
    kw_rm4_1d  = dict(stride_order=(0,),
                      memspace=cute.AddressSpace.gmem, assumed_align=4)
    def fake(dtype, shape, kw):
        return cute.runtime.make_fake_compact_tensor(dtype, shape, **kw)

    in_fake        = fake(cfg.DTYPE,  (M, N),    kw_rm16_2d)
    out_row_fake   = fake(cute.Uint8, (M, N),    kw_rm16_2d) if cfg.ROWWISE    else None
    scale_row_fake = fake(cute.Uint8, SCALE_R,   kw_rm16_2d) if cfg.ROWWISE    else None
    out_col_fake   = fake(cute.Uint8, (M, N),    kw_rm16_2d) if cfg.COLUMNWISE else None
    scale_col_fake = fake(cute.Uint8, SCALE_C,   kw_rm16_2d) if cfg.COLUMNWISE else None
    # No amax / no SR for the MXFP8 path: these slots are None. The kernel's
    # __call__ takes them as Optional and dead-strips the amax/SR branches.
    amax_row_fake = None
    amax_col_fake = None
    rng_state_fake = None
    # Explicit stream arg (kept in the tvm-ffi signature, not env-stream): C++
    # passes the CUDA stream handle as an int64 scalar, decoded as int-as-ptr.
    stream_fake = cute.runtime.make_fake_stream()

    compiled = cute.compile(
        kernel_obj,
        in_fake,                                 # mX
        out_row_fake, scale_row_fake, amax_row_fake,  # mO_row, mS_row, mA_row
        out_col_fake, scale_col_fake, amax_col_fake,  # mO_col, mS_col, mA_col
        rng_state_fake,                          # rng_state
        stream_fake,                             # stream
        options="--enable-tvm-ffi",
    )
    cache[fn_name] = compiled
    return compiled, fn_name


def get_mxfp8_quantizer(
    x: torch.Tensor,
    dtype_row: Literal["e4m3", "e5m2", "none"] = "e4m3",
    dtype_col: Literal["e4m3", "e5m2", "none"] = "e4m3",
    with_gemm_swizzled_scales: bool = False,
) -> FlexQuantizer:
    """Compile + register the MXFP8 kernel and return a
    FlexQuantizer wired to it.

    The compiled CuTeDSL function is registered into the tvm-ffi global
    registry under its deterministic name; the returned quantizer carries that
    name in ``quantize_func`` so the C++ ``FlexQuantizer::quantize`` can
    resolve and dispatch to it via ``Function::GetGlobalRequired``.

    Note on ``with_gemm_swizzled_scales``: when True, the scale tensors are
    emitted in the cuBLAS GEMM-swizzled layout, which pads them up to whole
    128x4 tiles. Because the kernel only writes the valid blocks, the quantizer
    zeros the scale buffers on every quantize (FlexQuantizer::quantize issues a
    cudaMemsetAsync before dispatch) so the padded entries cuBLAS reads are 0.
    With swizzle=False no swizzling/zeroing is done and the scale padding is
    undefined.
    """
    M, N = x.shape
    cutlass_dtype = torch_to_cutlass_dtype(x.dtype)
    cfg = MXFP8QuantizeConfig(
        cutlass_dtype,
        dtype_row=dtype_row,
        dtype_column=dtype_col,
        with_gemm_swizzled_scales=with_gemm_swizzled_scales,
    )
    compiled, fn_name = _get_compiled_kernel(cfg, M, N)
    _tvm_ffi.register_global_func(fn_name, compiled, override=True)

    quantizer = FlexQuantizer(
        dtype_row=str_to_te_dtype(dtype_row),
        dtype_column=str_to_te_dtype(dtype_col),
        quantize_func=fn_name,
        # Dequant isn't implemented for this storage-only milestone, but
        # FlexQuantizer::quantize (C++) rejects an empty dequantize_func, so
        # carry a placeholder name. It is never resolved/called by quantize().
        dequantize_func=f"{fn_name}_dequant", # TODO: this is fake, implement dequant and remove the placeholder
        stochastic_rounding=False,
    )
    # Storage-only milestone: no full PyTorch-tensor compatibility / GEMM yet.
    quantizer.internal = True
    quantizer.optimize_for_gemm = with_gemm_swizzled_scales
    return quantizer