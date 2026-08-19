# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Device and host-launch logic for the tuned NVFP4 quantize-transpose kernel."""

from typing import Optional

import cutlass
from cutlass import cute
from cutlass import (
    BFloat16,
    Float4E2M1FN,
    Float8E4M3FN,
    Float32,
    Int32,
    Int64,
    Uint8,
    Uint32,
    Uint64,
)
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module

from transformer_engine.common.CuTeDSL.cast.nvfp4.utils import (
    NVFP4_BLOCK_SCALING_SIZE,
    NVFP4_SCALE_PAD_INNER,
    NVFP4_SCALE_PAD_OUTER,
    compute_block_decode_sf,
    compute_block_encode_sf,
    compute_global_encode_sf,
    cvt_f32x8_to_fp4x8,
    cvt_f32x8_to_fp4x8_sr,
    mul_cvt_bf16x8_to_fp4x8,
    mul_cvt_bf16x8_to_fp4x8_sr,
)
from transformer_engine.common.CuTeDSL.philox_rng import PhiloxRng
from transformer_engine.common.CuTeDSL.utils import (
    CUTEDSL_DEBUG_LOGGING,
    fabs_f32,
    make_prmt_u32,
    noop_flag_is_set,
    pack_u32x2,
    packed16_kit,
    validate_tensor,
)


# The bf16 packed-op kit (max.xorsign.abs.bf16x2 and exact widening helpers).
bf16_kit = packed16_kit(BFloat16)

# Interleave low/high bf16 halves of two pair-registers for the colwise pass.
prmt_lo_u32 = make_prmt_u32(0x5410)
prmt_hi_u32 = make_prmt_u32(0x7632)


class NVFP4QuantizeTransposeTuned1DKernel:
    """Tuned kernel to cast to NVFP4 and transpose.

    Each thread block processes a CHUNK_DIM_Y x CHUNK_DIM_X _chunk_ of the input tensor as a grid
    of TILE_DIM x TILE_DIM _tiles_, walked sequentially with PREFETCH_STAGES tiles in flight
    beyond the current one: TMA loads a tile to SMEM, the CTA quantizes it into staged SMEM
    output buffers, and TMA stores those while the next tile loads.

    The thread arrangements are the CUDA kernel's. Rowwise, a thread owns one whole scaling
    block (16 row-adjacent elements) per iteration, making the block amax an intra-thread
    reduction, with the 16-byte shared memory reads staggered by bank group. Columnwise, a
    thread owns two adjacent columns of one 16-row scaling block, reading pairs, with the block
    row staggered by warp so consecutive lanes read different SMEM rows.
    """

    CHUNK_DIM_Y = 128
    CHUNK_DIM_X = 128
    TILE_DIM = 64
    THREADS = 128
    PREFETCH_STAGES = 1

    # Derived tiling constants (names follow the CUDA kernel).
    NUM_BUFFERS = PREFETCH_STAGES + 1
    STAGES_Y = CHUNK_DIM_Y // TILE_DIM
    STAGES_X = CHUNK_DIM_X // TILE_DIM
    STAGES = STAGES_Y * STAGES_X
    SCALES_PER_TILE = TILE_DIM // NVFP4_BLOCK_SCALING_SIZE  # 4
    SCALES_PER_CHUNK_X = CHUNK_DIM_X // NVFP4_BLOCK_SCALING_SIZE  # 8
    SCALES_PER_CHUNK_Y = CHUNK_DIM_Y // NVFP4_BLOCK_SCALING_SIZE  # 8
    THREADS_X_ROWWISE = TILE_DIM // NVFP4_BLOCK_SCALING_SIZE  # 4
    THREADS_Y_ROWWISE = THREADS // THREADS_X_ROWWISE  # 32
    ITERATIONS_ROWWISE = TILE_DIM // THREADS_Y_ROWWISE  # 2
    PACK_SIZE = 8  # elements per vectorized SMEM access
    WAVES = NVFP4_BLOCK_SCALING_SIZE // PACK_SIZE  # 2
    # Threads that span the 32 4-byte SMEM banks at 16 bf16 per thread (the rowwise swizzle).
    THREADS_PER_BANK = (32 * 4 * 8) // 4 // NVFP4_BLOCK_SCALING_SIZE  # 16
    assert(NUM_BUFFERS <= STAGES) # otherwise, prefetch loop would read OOB

    def __init__(self, cfg):
        self.cfg = cfg

    # Host-side kernel launch
    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,  # (M, N) bf16 input
        mO_row: cute.Tensor,  # (M, N) fp4 rowwise output
        mS_row: cute.Tensor,  # (roundup(M, 128), roundup(ceil(N / 16), 4)) e4m3 scales
        mO_col: Optional[cute.Tensor],  # (N, M) fp4 transposed output
        mS_col: Optional[cute.Tensor],  # (roundup(N, 128), roundup(ceil(M / 16), 4)) e4m3 scales
        mAmaxRow: cute.Tensor,  # (M,) f32 per-row amax if ROW_SCALED_NVFP4, else (1,) global amax
        mAmaxCol: Optional[cute.Tensor],  # (1,) f32 global amax of the transposed output
        mNoop: cute.Pointer,  # f32 cast-noop flag; may be null, checked on device
        mRngState: Optional[cute.Tensor],  # (2,) i64 Philox {seed, offset}
        stream: CUstream,
    ):
        """AOT-compiled host entrypoint. `quantize_transpose_nvfp4_cutedsl.cuh` passes these
        arguments in this exact order via tvm-ffi, and the config fixes which of the optional
        ones are present (see `compile_cutedsl_function_from_cfg`).

        M, N are the input's *flattened* 2D dims, both multiples of NVFP4_SHAPE_ALIGNMENT; a
        rank > 2 input already arrives flattened.
        All tensors are row-major (with rightmost stride 1).
        FP4 extents are logical element counts, not the halved extents of the uint8 buffer TE
        actually allocates.
        """
        if cutlass.const_expr(CUTEDSL_DEBUG_LOGGING):
            cute.printf(
                "[CuTeDSL] NVFP4QuantizeTransposeTuned1DKernel.__call__() with config:"
                f" {self.cfg}\n"
            )

        ## Validation

        # Validate input and output tensor layouts
        M, N = mX.shape
        mX_layout = cute.make_ordered_layout((M, N), order=(1, 0))
        mO_row_layout = mX_layout
        mO_col_layout = cute.make_ordered_layout((N, M), order=(1, 0))
        validate_tensor(mX, mX_layout, BFloat16)
        validate_tensor(mO_row, mO_row_layout, Float4E2M1FN)
        validate_tensor(mO_col, mO_col_layout, Float4E2M1FN)

        # Validate scaling factor tensor layouts
        mS_row_layout = cute.make_ordered_layout(
            (
                cute.round_up(M, NVFP4_SCALE_PAD_OUTER),
                cute.round_up(cute.ceil_div(N, NVFP4_BLOCK_SCALING_SIZE), NVFP4_SCALE_PAD_INNER),
            ),
            order=(1, 0),
        )
        mS_col_layout = cute.make_ordered_layout(
            (
                cute.round_up(N, NVFP4_SCALE_PAD_OUTER),
                cute.round_up(cute.ceil_div(M, NVFP4_BLOCK_SCALING_SIZE), NVFP4_SCALE_PAD_INNER),
            ),
            order=(1, 0),
        )
        validate_tensor(mS_row, mS_row_layout, Float8E4M3FN)
        validate_tensor(mS_col, mS_col_layout, Float8E4M3FN)

        # Validate amax tensor layouts
        if cutlass.const_expr(self.cfg.ROW_SCALED_NVFP4):
            mAmaxRow_layout = cute.make_layout((M,))
            validate_tensor(mAmaxRow, mAmaxRow_layout, Float32)
        else:
            mAmaxRow_layout = cute.make_layout((1,))
            validate_tensor(mAmaxRow, mAmaxRow_layout, Float32)
        mAmaxCol_layout = cute.make_layout((1,))
        validate_tensor(mAmaxCol, mAmaxCol_layout, Float32)

        # Validate RNG state tensor layout
        mRngState_layout = cute.make_layout((2,))
        validate_tensor(mRngState, mRngState_layout, Int64)

        ## TMA descriptors
        # The FP4 outputs are moved as bytes: two elements to a byte, so the byte view of the
        # (M, N) rowwise output is (M, N/2) and a 64x64-element tile is a 64x32-byte box. The
        # 32-divisibility of both dims is what keeps every row stride a multiple of 16B as TMA
        # requires: 2*N bytes for the bf16 input, N/2 and M/2 for the fp4 outputs.
        mO_row_bytes = cute.recast_tensor(mO_row, Uint8)

        tile_in_layout = cute.make_ordered_layout((self.TILE_DIM, self.TILE_DIM), order=(1, 0))
        tile_out_layout = cute.make_ordered_layout(
            (self.TILE_DIM, self.TILE_DIM // 2), order=(1, 0)
        )

        op_load = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        op_store = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()

        tma_atom_in, tma_view_in = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op_load, mX, tile_in_layout, (self.TILE_DIM, self.TILE_DIM), num_multicast=1
        )
        tma_atom_row, tma_view_row = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op_store,
            mO_row_bytes,
            tile_out_layout,
            (self.TILE_DIM, self.TILE_DIM // 2),
            num_multicast=1,
        )
        if cutlass.const_expr(self.cfg.RETURN_TRANSPOSE):
            mO_col_bytes = cute.recast_tensor(mO_col, Uint8)
            tma_atom_col, tma_view_col = cute.nvgpu.cpasync.make_tiled_tma_atom(
                op_store,
                mO_col_bytes,
                tile_out_layout,
                (self.TILE_DIM, self.TILE_DIM // 2),
                num_multicast=1,
            )
        else:
            tma_atom_col = None
            tma_view_col = None

        ## Grid: one CTA per chunk; X indexes columns and Y rows, as in the CUDA kernel's
        ## ctaid_X / ctaid_Y.
        grid = [
            cute.ceil_div(N, self.CHUNK_DIM_X),
            cute.ceil_div(M, self.CHUNK_DIM_Y),
            1,
        ]

        self.kernel(
            mX,
            mS_row,
            mS_col,
            mAmaxRow,
            mAmaxCol,
            mNoop,
            mRngState,
            tma_atom_in,
            tma_view_in,
            tma_atom_row,
            tma_view_row,
            tma_atom_col,
            tma_view_col,
        ).launch(grid=grid, block=[self.THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mS_row: cute.Tensor,
        mS_col: Optional[cute.Tensor],
        mAmaxRow: cute.Tensor,
        mAmaxCol: Optional[cute.Tensor],
        mNoop: cute.Pointer,
        mRngState: Optional[cute.Tensor],
        tma_atom_in: cute.CopyAtom,
        tma_view_in: cute.Tensor,
        tma_atom_row: cute.CopyAtom,
        tma_view_row: cute.Tensor,
        tma_atom_col: Optional[cute.CopyAtom],
        tma_view_col: Optional[cute.Tensor],
    ):
        """Device entry for the NVFP4 tuned-1D quantize-transpose kernel."""
        if not noop_flag_is_set(mNoop):
            self.kernel_main(
                mX,
                mS_row,
                mS_col,
                mAmaxRow,
                mAmaxCol,
                mRngState,
                tma_atom_in,
                tma_view_in,
                tma_atom_row,
                tma_view_row,
                tma_atom_col,
                tma_view_col,
            )

    @cute.jit
    def kernel_main(
        self,
        mX: cute.Tensor,
        mS_row: cute.Tensor,
        mS_col: Optional[cute.Tensor],
        mAmaxRow: cute.Tensor,
        mAmaxCol: Optional[cute.Tensor],
        mRngState: Optional[cute.Tensor],
        tma_atom_in: cute.CopyAtom,
        tma_view_in: cute.Tensor,
        tma_atom_row: cute.CopyAtom,
        tma_view_row: cute.Tensor,
        tma_atom_col: Optional[cute.CopyAtom],
        tma_view_col: Optional[cute.Tensor],
    ):
        # -- Trace time --
        cfg = self.cfg
        TILE = self.TILE_DIM

        # Shared memory layout
        if cutlass.const_expr(cfg.RETURN_TRANSPOSE):

            @cute.struct
            class SharedStorage:
                mbar_storage: cute.struct.MemRange[cute.Int64, self.NUM_BUFFERS]
                sX: cute.struct.Align[
                    cute.struct.MemRange[BFloat16, TILE * TILE * self.NUM_BUFFERS], 128
                ]
                sO_row: cute.struct.Align[
                    cute.struct.MemRange[Uint8, TILE * (TILE // 2) * self.NUM_BUFFERS], 128
                ]
                sO_col: cute.struct.Align[
                    cute.struct.MemRange[Uint8, TILE * (TILE // 2) * self.NUM_BUFFERS], 128
                ]
                sS_row: cute.struct.Align[
                    cute.struct.MemRange[Float8E4M3FN, self.CHUNK_DIM_Y * self.SCALES_PER_CHUNK_X],
                    16,
                ]
                sS_col: cute.struct.Align[
                    cute.struct.MemRange[Float8E4M3FN, self.CHUNK_DIM_X * self.SCALES_PER_CHUNK_Y],
                    16,
                ]

        else:

            @cute.struct
            class SharedStorage:
                mbar_storage: cute.struct.MemRange[cute.Int64, self.NUM_BUFFERS]
                sX: cute.struct.Align[
                    cute.struct.MemRange[BFloat16, TILE * TILE * self.NUM_BUFFERS], 128
                ]
                sO_row: cute.struct.Align[
                    cute.struct.MemRange[Uint8, TILE * (TILE // 2) * self.NUM_BUFFERS], 128
                ]
                sS_row: cute.struct.Align[
                    cute.struct.MemRange[Float8E4M3FN, self.CHUNK_DIM_Y * self.SCALES_PER_CHUNK_X],
                    16,
                ]

        # "Allocate" shared memory
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Create views into the shared memory
        buffered_tile = lambda inner_cols: cute.make_layout(
            ((TILE, inner_cols), self.NUM_BUFFERS),
            stride=((inner_cols, 1), TILE * inner_cols),
        )
        sX = storage.sX.get_tensor(buffered_tile(TILE))
        sO_row = storage.sO_row.get_tensor(buffered_tile(TILE // 2))
        sS_row = storage.sS_row.get_tensor(
            cute.make_layout(
                (self.CHUNK_DIM_Y, self.SCALES_PER_CHUNK_X), stride=(self.SCALES_PER_CHUNK_X, 1)
            )
        )
        if cutlass.const_expr(cfg.RETURN_TRANSPOSE):
            sO_col = storage.sO_col.get_tensor(buffered_tile(TILE // 2))
            sS_col = storage.sS_col.get_tensor(
                cute.make_layout(
                    (self.CHUNK_DIM_X, self.SCALES_PER_CHUNK_Y),
                    stride=(self.SCALES_PER_CHUNK_Y, 1),
                )
            )
        else:
            sO_col = None
            sS_col = None

        # Bind GMEM and SMEM tensor views for TMA
        gX_tiled = cute.zipped_divide(tma_view_in, (TILE, TILE))
        tXsX, tXgX = cute.nvgpu.cpasync.tma_partition(
            tma_atom_in, 0, cute.make_layout(1), sX, gX_tiled
        )
        gO_row_tiled = cute.zipped_divide(tma_view_row, (TILE, TILE // 2))
        tOsO_row, tOgO_row = cute.nvgpu.cpasync.tma_partition(
            tma_atom_row, 0, cute.make_layout(1), sO_row, gO_row_tiled
        )
        if cutlass.const_expr(cfg.RETURN_TRANSPOSE):
            gO_col_tiled = cute.zipped_divide(tma_view_col, (TILE, TILE // 2))
            tOsO_col, tOgO_col = cute.nvgpu.cpasync.tma_partition(
                tma_atom_col, 0, cute.make_layout(1), sO_col, gO_col_tiled
            )

        # -- Runtime --
        rows, cols = mX.shape
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()  # (chunk_x, chunk_y)
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch TMA descriptor of the input
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_in)

        # Initialize mbarriers for TMA G2S input tensor copy
        mbar = storage.mbar_storage.data_ptr()
        if warp_idx == 0:
            with cute.arch.elect_one():
                for b in cutlass.range_constexpr(self.NUM_BUFFERS):
                    cute.arch.mbarrier_init(mbar + b, 1)
                # release mbarrier_init from elected thread to all threads in cluster (here: CTA)
                cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads() # acquire mbarrier_init by CTA (cluster)

        # Prefetch NUM_BUFFERS=PREFETCH_STAGES tiles + the first tile to process
        tx_count = TILE * TILE * 2 # each load is a TILE x TILE x bf16 (2 bytes) tile 
        if warp_idx == 0:
            for s in cutlass.range_constexpr(self.NUM_BUFFERS):
                tile_coord = (
                    bidy * self.STAGES_Y + s // self.STAGES_X,
                    bidx * self.STAGES_X + s % self.STAGES_X,
                )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar + s, tx_count)
                cute.copy(
                    tma_atom_in,
                    tXgX[(None, tile_coord)],
                    tXsX[(None, s)],
                    tma_bar_ptr=mbar + s,
                )

        # Compute global encode scales from supplied amax
        if cutlass.const_expr(not cfg.ROW_SCALED_NVFP4):
            S_enc_rowwise = compute_global_encode_sf(mAmaxRow[0])
        else:
            # Per-row encode scales are drawn inside the rowwise pass instead.
            S_enc_rowwise = Float32(1.0)
        if cutlass.const_expr(cfg.RETURN_TRANSPOSE):
            S_enc_colwise = compute_global_encode_sf(mAmaxCol[0])
        else:
            S_enc_colwise = Float32(1.0)

        # Construct RNG state for SR
        rng = None
        if cutlass.const_expr(cfg.USE_STOCHASTIC_ROUNDING):
            grid_dim_x, _, _ = cute.arch.grid_dim()
            # Contrary to CUDA C++ version, calculate in Int64 for correctness
            rng_sequence = (
                Int64(tidx)
                + Int64(bidx) * self.THREADS
                + Int64(bidy) * Int64(grid_dim_x) * self.THREADS
            )
            rng = PhiloxRng(
                seed=Uint64(mRngState[0].ir_value()),
                subsequence=Uint64(rng_sequence.ir_value()),
                offset=Uint64(mRngState[1].ir_value()),
            )

        # Main loop over tiles/stages in chunk
        for stage in cutlass.range_constexpr(self.STAGES):
            stage_y = stage // self.STAGES_X
            stage_x = stage % self.STAGES_X
            parity = (stage // self.NUM_BUFFERS) % 2
            buf = stage % self.NUM_BUFFERS

            # Wait for TMA G2S tile load to complete
            cute.arch.mbarrier_wait(mbar + buf, parity)

            sX_tile = sX[(None, buf)]
            sO_row_tile = sO_row[(None, buf)]

            self.rowwise_tile(
                sX_tile,
                sO_row_tile,
                sS_row,
                mAmaxRow,
                S_enc_rowwise,
                bidy * self.CHUNK_DIM_Y,
                rows,
                stage_y,
                stage_x,
                rng,
            )
            if cutlass.const_expr(cfg.RETURN_TRANSPOSE):
                self.colwise_tile(
                    sX_tile,
                    sO_col[(None, buf)],
                    sS_col,
                    S_enc_colwise,
                    stage_y,
                    stage_x,
                    rng,
                )

            # Make the SMEM output writes visible to the TMA async proxy. The dedicated
            # fence.proxy.async.shared::cta, not the far costlier generic membar
            # cute.arch.fence_proxy would emit.
            cute.arch.fence_view_async_shared()
            # Before the barrier: wait for every prior TMA store to have finished reading its
            # staged output buffer. The barrier then broadcasts that to all threads, making the
            # next stage's writes into that buffer safe, and orders this stage's SMEM output
            # writes before the TMA store below. One barrier serves all three purposes; this is
            # the only syncthreads in the loop.
            if warp_idx == 0:
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            cute.arch.sync_threads()

            if warp_idx == 0:
                # Store this tile's outputs. The transposed tile of chunk-relative tile (y, x)
                # lands at tile (x, y) of the transposed tensor.
                cute.copy(
                    tma_atom_row,
                    tOsO_row[(None, buf)],
                    tOgO_row[
                        (
                            None,
                            (bidy * self.STAGES_Y + stage_y, bidx * self.STAGES_X + stage_x),
                        )
                    ],
                )
                if cutlass.const_expr(cfg.RETURN_TRANSPOSE):
                    cute.copy(
                        tma_atom_col,
                        tOsO_col[(None, buf)],
                        tOgO_col[
                            (
                                None,
                                (bidx * self.STAGES_X + stage_x, bidy * self.STAGES_Y + stage_y),
                            )
                        ],
                    )
                cute.arch.cp_async_bulk_commit_group()

                # Refill the buffer this stage just consumed: every thread's reads of it
                # finished before the syncthreads above.
                if cutlass.const_expr(stage + self.NUM_BUFFERS < self.STAGES):
                    next_stage = stage + self.NUM_BUFFERS
                    tile_coord = (
                        bidy * self.STAGES_Y + next_stage // self.STAGES_X,
                        bidx * self.STAGES_X + next_stage % self.STAGES_X,
                    )
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(mbar + buf, tx_count)
                    cute.copy(
                        tma_atom_in,
                        tXgX[(None, tile_coord)],
                        tXsX[(None, buf)],
                        tma_bar_ptr=mbar + buf,
                    )

        ## Epilogue: flush the chunk's staged scales with predicated stores. The last stage's
        ## syncthreads ordered every thread's scale writes before this point. This is the one
        ## place anything is predicated: the data tensors are covered by TMA's out-of-bounds
        ## handling, the scale tensors are not.
        self.flush_scales(
            sS_row, mS_row, bidy * self.CHUNK_DIM_Y, bidx * self.SCALES_PER_CHUNK_X, rows, cols
        )
        if cutlass.const_expr(cfg.RETURN_TRANSPOSE):
            self.flush_scales(
                sS_col, mS_col, bidx * self.CHUNK_DIM_X, bidy * self.SCALES_PER_CHUNK_Y, cols, rows
            )

        # Wait for in-flight TMA stores before the kernel returns.
        cute.arch.cp_async_bulk_wait_group(0, read=False)

    @cute.jit
    def rowwise_tile(
        self,
        sX_tile: cute.Tensor,  # (TILE, TILE) bf16 SMEM input tile
        sO_row_tile: cute.Tensor,  # (TILE, TILE/2) u8 SMEM staged rowwise output
        sS_row: cute.Tensor,  # (CHUNK_DIM_Y, SCALES_PER_CHUNK_X) e4m3 SMEM staged scales
        mAmaxRow: cute.Tensor,  # per-row amax (ROW_SCALED) or the global amax (unused here)
        S_enc: Float32,  # global rowwise encode scale (ignored when ROW_SCALED)
        chunk_row0: Int32,  # global row of the chunk's first row
        rows: Int32,
        stage_y: cutlass.Constexpr,
        stage_x: cutlass.Constexpr,
        rng,
    ):
        """Quantize one SMEM tile rowwise: a thread owns one 16-element scaling block per
        iteration, so the block amax never leaves the thread. Mirrors rowwise_scaling in the
        CUDA kernel, including the bank-group stagger of the two 16-byte reads per block."""
        cfg = self.cfg
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % 32
        bank_group = lane // self.THREADS_PER_BANK  # which 16-byte wave this thread starts on
        tid_y = tidx // self.THREADS_X_ROWWISE
        tid_x = tidx % self.THREADS_X_ROWWISE

        if cutlass.const_expr(cfg.USE_FAST_MATH):
            sf_type = BFloat16
        else:
            sf_type = Float32

        # The tile as (row * 8-element group, element): a flat first mode so a slice needs only
        # one dynamic index, each group one 16-byte vectorized SMEM access. A fresh view on the
        # tile's pointer rather than cute.composition, whose right operand maps into the tile's
        # colexicographic element order, not its memory order.
        groups_per_row = self.TILE_DIM // self.PACK_SIZE
        sX_groups = cute.make_tensor(
            sX_tile.iterator,
            cute.make_layout(
                (self.TILE_DIM * groups_per_row, self.PACK_SIZE), stride=(self.PACK_SIZE, 1)
            ),
        )
        # The staged output as u32: one conversion of 8 elements fills one u32.
        sO_u32 = cute.make_tensor(
            cute.recast_ptr(sO_row_tile.iterator, dtype=Uint32),
            cute.make_layout(
                (self.TILE_DIM * self.TILE_DIM // 8,),
                stride=(1,),
            ),
        )
        u32_per_row = self.TILE_DIM // 8

        for it in cutlass.range_constexpr(self.ITERATIONS_ROWWISE):
            row = tid_y + it * self.THREADS_Y_ROWWISE

            # Read this thread's scaling block, one staggered 8-element wave at a time, and
            # accumulate the block amax as packed bf16 pairs (max.xorsign.abs.bf16x2).
            frg = []
            frg_u32 = []
            for w in cutlass.range_constexpr(self.WAVES):
                frg.append(cute.make_rmem_tensor(self.PACK_SIZE, BFloat16))
                frg_u32.append(
                    cute.make_tensor(
                        cute.recast_ptr(frg[w].iterator, dtype=Uint32),
                        cute.make_layout((self.PACK_SIZE // 2,), stride=(1,)),
                    )
                )
            amax_2x = Int32(0)
            for w in cutlass.range_constexpr(self.WAVES):
                group = tid_x * self.WAVES + ((w + bank_group) % self.WAVES)
                cute.autovec_copy(sX_groups[row * groups_per_row + group, None], frg[w])
                for j in cutlass.range_constexpr(self.PACK_SIZE // 2):
                    amax_2x = bf16_kit.abs_max_x2(amax_2x, Int32(frg_u32[w][j].ir_value()))
            block_amax = cute.arch.fmax(
                fabs_f32(bf16_kit.x2_lo_to_f32(amax_2x)),
                fabs_f32(bf16_kit.x2_hi_to_f32(amax_2x)),
            )

            # The encode scale: global, or this row's own under row-scaled quantization. An
            # out-of-bounds row (the CUDA kernel gives it encode scale 1.0) produces nothing
            # observable -- its data rows are dropped by the output TMA and its scale rows are
            # not flushed -- so the amax load clamps the index instead of branching.
            if cutlass.const_expr(cfg.ROW_SCALED_NVFP4):
                row_idx = chunk_row0 + stage_y * self.TILE_DIM + row
                S_enc_block = compute_global_encode_sf(mAmaxRow[cutlass.min(row_idx, rows - 1)])
            else:
                S_enc_block = S_enc

            block_decode_sf = compute_block_decode_sf(block_amax, S_enc_block)
            sS_row[stage_y * self.TILE_DIM + row, stage_x * self.SCALES_PER_TILE + tid_x] = (
                block_decode_sf
            )
            coeff = compute_block_encode_sf(block_decode_sf, S_enc_block, sf_type)

            # Scale and convert one wave (8 elements, one u32 of nibbles) at a time, storing to
            # the same staggered group the wave was read from. Each coefficient type gets the
            # multiply the CUDA kernel pairs it with: an fma against a zero addend for the bf16
            # one (which flushes a -0 product to +0, and E2M1 has a signed zero, so the
            # instruction has to match) and a plain f32 multiply for the f32 one.
            for w in cutlass.range_constexpr(self.WAVES):
                group = tid_x * self.WAVES + ((w + bank_group) % self.WAVES)
                if cutlass.const_expr(cfg.USE_FAST_MATH):
                    coeff_f32 = coeff.to(Float32)
                    if cutlass.const_expr(cfg.USE_STOCHASTIC_ROUNDING):
                        rbits03 = rng.get_rbits()
                        rbits47 = rng.get_rbits()
                        out = mul_cvt_bf16x8_to_fp4x8_sr(
                            frg_u32[w][0],
                            frg_u32[w][1],
                            frg_u32[w][2],
                            frg_u32[w][3],
                            coeff_f32,
                            rbits03,
                            rbits47,
                        )
                    else:
                        out = mul_cvt_bf16x8_to_fp4x8(
                            frg_u32[w][0], frg_u32[w][1], frg_u32[w][2], frg_u32[w][3], coeff_f32
                        )
                else:
                    scaled = []
                    for j in cutlass.range_constexpr(self.PACK_SIZE // 2):
                        pair = Int32(frg_u32[w][j].ir_value())
                        scaled.append(bf16_kit.x2_lo_to_f32(pair) * coeff)
                        scaled.append(bf16_kit.x2_hi_to_f32(pair) * coeff)
                    if cutlass.const_expr(cfg.USE_STOCHASTIC_ROUNDING):
                        rbits03 = rng.get_rbits()
                        rbits47 = rng.get_rbits()
                        out = cvt_f32x8_to_fp4x8_sr(*scaled, rbits03, rbits47)
                    else:
                        out = cvt_f32x8_to_fp4x8(*scaled)
                sO_u32[row * u32_per_row + group] = out

    @cute.jit
    def colwise_tile(
        self,
        sX_tile: cute.Tensor,  # (TILE, TILE) bf16 SMEM input tile
        sO_col_tile: cute.Tensor,  # (TILE, TILE/2) u8 SMEM staged transposed output
        sS_col: cute.Tensor,  # (CHUNK_DIM_X, SCALES_PER_CHUNK_Y) e4m3 SMEM staged scales
        S_enc: Float32,
        stage_y: cutlass.Constexpr,
        stage_x: cutlass.Constexpr,
        rng,
    ):
        """Quantize one SMEM tile columnwise into the transposed staged output: a thread owns
        two adjacent columns of one 16-row scaling block, read as bf16 pairs, with the block row
        staggered by warp so consecutive lanes hit different SMEM rows (conflict-free 4-byte
        reads). Mirrors colwise_scaling in the CUDA kernel."""
        cfg = self.cfg
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % 32
        warp = tidx // 32
        tid_y = (lane // 2 + warp) % self.SCALES_PER_TILE  # which 16-row block

        if cutlass.const_expr(cfg.USE_FAST_MATH):
            sf_type = BFloat16
        else:
            sf_type = Float32

        # The tile as (row, column pair): a pair-register read per row.
        sX_pairs = cute.make_tensor(
            cute.recast_ptr(sX_tile.iterator, dtype=Uint32),
            cute.make_layout(
                (self.TILE_DIM * (self.TILE_DIM // 2),),
                stride=(1,),
            ),
        )
        pairs_per_row = self.TILE_DIM // 2
        # The transposed staged output as u64: one 16-element block of a transposed row is 8
        # bytes, one two-conversion store.
        sO_u64 = cute.make_tensor(
            cute.recast_ptr(sO_col_tile.iterator, dtype=Int64),
            cute.make_layout(
                (self.TILE_DIM * self.TILE_DIM // 16,),
                stride=(1,),
            ),
        )
        u64_per_row = self.TILE_DIM // 16

        # Read the two columns' 16-row block as pairs, accumulating both block amaxes at once
        # in the packed halves.
        row0 = tid_y * NVFP4_BLOCK_SCALING_SIZE
        pairs = []
        amax_2x = Int32(0)
        for i in cutlass.range_constexpr(NVFP4_BLOCK_SCALING_SIZE):
            pair = sX_pairs[(row0 + i) * pairs_per_row + lane]
            pairs.append(pair)
            amax_2x = bf16_kit.abs_max_x2(amax_2x, Int32(pair.ir_value()))
        block_amax = [
            fabs_f32(bf16_kit.x2_lo_to_f32(amax_2x)),
            fabs_f32(bf16_kit.x2_hi_to_f32(amax_2x)),
        ]

        for w in cutlass.range_constexpr(2):
            block_decode_sf = compute_block_decode_sf(block_amax[w], S_enc)
            sS_col[
                stage_x * self.TILE_DIM + 2 * lane + w,
                stage_y * self.SCALES_PER_TILE + tid_y,
            ] = block_decode_sf
            coeff = compute_block_encode_sf(block_decode_sf, S_enc, sf_type)

            outs = []
            if cutlass.const_expr(cfg.USE_FAST_MATH):
                coeff_f32 = coeff.to(Float32)
                # Repack this column's elements into adjacent bf16 pairs for the fma-based
                # conversion: bytes of (row 2j, row 2j+1) picked by a byte permute.
                prmt = prmt_lo_u32 if w == 0 else prmt_hi_u32
                packed = [
                    prmt(pairs[2 * j], pairs[2 * j + 1])
                    for j in range(NVFP4_BLOCK_SCALING_SIZE // 2)
                ]
                for e in cutlass.range_constexpr(NVFP4_BLOCK_SCALING_SIZE // 8):
                    if cutlass.const_expr(cfg.USE_STOCHASTIC_ROUNDING):
                        rbits03 = rng.get_rbits()
                        rbits47 = rng.get_rbits()
                        outs.append(
                            mul_cvt_bf16x8_to_fp4x8_sr(
                                packed[4 * e],
                                packed[4 * e + 1],
                                packed[4 * e + 2],
                                packed[4 * e + 3],
                                coeff_f32,
                                rbits03,
                                rbits47,
                            )
                        )
                    else:
                        outs.append(
                            mul_cvt_bf16x8_to_fp4x8(
                                packed[4 * e],
                                packed[4 * e + 1],
                                packed[4 * e + 2],
                                packed[4 * e + 3],
                                coeff_f32,
                            )
                        )
            else:
                widen = bf16_kit.x2_lo_to_f32 if w == 0 else bf16_kit.x2_hi_to_f32
                scaled = [
                    widen(Int32(pairs[i].ir_value())) * coeff
                    for i in range(NVFP4_BLOCK_SCALING_SIZE)
                ]
                for e in cutlass.range_constexpr(NVFP4_BLOCK_SCALING_SIZE // 8):
                    if cutlass.const_expr(cfg.USE_STOCHASTIC_ROUNDING):
                        rbits03 = rng.get_rbits()
                        rbits47 = rng.get_rbits()
                        outs.append(
                            cvt_f32x8_to_fp4x8_sr(*scaled[8 * e : 8 * e + 8], rbits03, rbits47)
                        )
                    else:
                        outs.append(cvt_f32x8_to_fp4x8(*scaled[8 * e : 8 * e + 8]))

            sO_u64[(2 * lane + w) * u64_per_row + tid_y] = pack_u32x2(outs[0], outs[1])

    @cute.jit
    def flush_scales(
        self,
        sS: cute.Tensor,  # (128, 8) e4m3 SMEM staged scales for the whole chunk
        mS: cute.Tensor,  # padded gmem scale tensor
        chunk_row0: Int32,  # global row of the chunk's first scale row
        chunk_sf_col0: Int32,  # global scale column of the chunk's first scale
        outer: Int32,  # valid rows of the scale tensor (input rows / cols)
        inner_elems: Int32,  # input elements along the scaled direction (cols / rows)
        # -> valid scale columns = inner_elems / 16
    ):
        """Write one chunk's staged scale bytes out, one row per thread, predicated against the
        input's real extents: the padding region of the scale tensors is left untouched and
        rows/columns past the input's edge are skipped, like the CUDA kernel's Vec-based scale
        store. An interior chunk takes the vectorized path -- one 8-byte SMEM read and two
        4-byte gmem stores (the padded scale stride only guarantees 4-byte alignment) -- since
        the elementwise fallback's divergent byte stores are what an unoptimized epilogue
        spends most of the kernel's stalls on."""
        tidx, _, _ = cute.arch.thread_idx()
        row_global = chunk_row0 + tidx
        count = cutlass.min(
            Int32(self.SCALES_PER_CHUNK_X),
            (inner_elems - chunk_sf_col0 * NVFP4_BLOCK_SCALING_SIZE) // NVFP4_BLOCK_SCALING_SIZE,
        )
        frg = cute.make_rmem_tensor(self.SCALES_PER_CHUNK_X, Float8E4M3FN)
        cute.autovec_copy(sS[tidx, None], frg)
        frg_u32 = cute.make_tensor(
            cute.recast_ptr(frg.iterator, dtype=Uint32),
            cute.make_layout((self.SCALES_PER_CHUNK_X // 4,), stride=(1,)),
        )
        # A u32 view of the scale tensor: its padded inner extent (= the row stride) is a
        # multiple of NVFP4_SCALE_PAD_INNER = 4, so rows stay u32-aligned, and a chunk's first
        # scale column is a multiple of SCALES_PER_CHUNK_X.
        scale_stride = cute.size(mS.shape[1])
        mS_u32 = cute.make_tensor(
            cute.recast_ptr(mS.iterator, dtype=Uint32),
            cute.make_layout((mS.shape[0], scale_stride // 4), stride=(scale_stride // 4, 1)),
        )
        if row_global < outer:
            if count == self.SCALES_PER_CHUNK_X:
                for c4 in cutlass.range_constexpr(self.SCALES_PER_CHUNK_X // 4):
                    mS_u32[row_global, chunk_sf_col0 // 4 + c4] = frg_u32[c4]
            else:
                for c in cutlass.range_constexpr(self.SCALES_PER_CHUNK_X):
                    if c < count:
                        mS[row_global, chunk_sf_col0 + c] = frg[c]


