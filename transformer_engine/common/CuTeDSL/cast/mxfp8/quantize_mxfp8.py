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
import logging

import transformer_engine
from transformer_engine.common.CuTeDSL.utils import str_to_cutlass_dtype
import transformer_engine_torch as tex

from typing import Optional, Type

import torch
import transformer_engine_torch as tex

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import Float32, Int64, Int32, Int16, Uint8, Uint32
from cuda.bindings.driver import CUstream

import hashlib
import tvm_ffi

from .mxfp8_utils import (
    SUPPORTED_ACTIVATIONS,
    SUPPORTED_DACTIVATIONS,
    FP8E4M3_MAX_NORM_RCP,
    FP8E5M2_MAX_NORM_RCP,
    _bitcast_f32_to_i32,
    _cvt_f32_to_fp8,
    _cvt_f32x2_to_fp8x2,
    _is_packed16,
    _packed16_kit,
    exp2f_rcp,
    fabs_f32,
    float_to_e8m0,
    quantize_colwise_mxfp8,
    quantize_rowwise_mxfp8,
)

# Per-backend logger, so a fallback warning is attributable to *this* CuTeDSL
# backend (the MXFP8 quantize backend). Other CuTeDSL backends should use their
# own `transformer_engine.cutedsl.<backend>` logger.
logger = logging.getLogger("transformer_engine.cutedsl.mxfp8")

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

# ---------------------------------------------------------------------------
# Kernel configuration
# ---------------------------------------------------------------------------
class MXFP8QuantizeConfig:

    def __init__(
        self,
        dtype: str,
        fp8_dtype: str,
        rowwise: bool,
        colwise: bool,
        with_gemm_swizzled_scales: bool,
        with_amax: bool,
        with_dbias: bool = False,
        with_dact: bool = False,
        with_act: bool = False,
        with_noop: bool = False,
        activation: Optional[str] = None
    ):
        if dtype is None or dtype not in ("fp32", "fp16", "bf16"):
            raise ValueError(f"unknown input dtype {dtype!r}; expected fp32|fp16|bf16")
        self.DTYPE = str_to_cutlass_dtype(dtype)
        self.DTYPE_STR = dtype  # readable input-dtype token, for __str__
        if fp8_dtype not in ("e4m3", "e5m2"):
            raise ValueError(f"unknown FP8 dtype {fp8_dtype!r}; expected 'e4m3' or 'e5m2'")
        self.FP8_DTYPE = fp8_dtype
        self.ROWWISE = rowwise
        self.COLWISE = colwise
        if not (rowwise or colwise):
            raise ValueError("at least one of rowwise or colwise must be true")
        self.WITH_GEMM_SWIZZLED_SCALES = with_gemm_swizzled_scales
        self.WITH_AMAX = with_amax
        if not with_dact and not with_act:
            if activation == "none":
                self.ACTIVATION = None
            else:
                raise ValueError("activation must be none when with_dact and with_act are both False")
        else:
            if with_dact and with_act:
                raise ValueError("with_dact and with_act cannot be true at the same time since they are used for different paths (bwd vs fwd)")
            elif with_dact:
                if activation in SUPPORTED_DACTIVATIONS:
                    self.ACTIVATION = activation
                else:
                    raise ValueError(f"unknown activation {activation!r} for with_dact=True; expected one of {sorted(SUPPORTED_DACTIVATIONS)}")
            elif with_act:
                if activation in SUPPORTED_ACTIVATIONS:
                    self.ACTIVATION = activation
                else:
                    raise ValueError(f"unknown activation {activation!r} for with_act=True; expected one of {sorted(SUPPORTED_ACTIVATIONS)}")
        self.WITH_DACT = with_dact
        self.WITH_ACT = with_act
        # dbias is the column reduction of the (post-act/dact) element. With colwise
        # output each thread owns a full column (trivial reduction); rowwise-only
        # uses a cross-thread smem reduction over THREADS_Y. Both mirror the CUDA
        # kernel's COLWISE_SCALING / rowwise dbias branches.
        self.WITH_DBIAS = with_dbias
        self.WITH_NOOP = with_noop
        self.MAX_NORM_RCP = FP8E4M3_MAX_NORM_RCP if fp8_dtype == "e4m3" else FP8E5M2_MAX_NORM_RCP

    def __str__(self):
        return (f"MXFP8QuantizeConfig(dtype={self.DTYPE_STR}, fp8_dtype={self.FP8_DTYPE}, "
                f"rowwise={self.ROWWISE}, colwise={self.COLWISE}, "
                f"swizzled={self.WITH_GEMM_SWIZZLED_SCALES}, with_amax={self.WITH_AMAX}, "
                f"with_dbias={self.WITH_DBIAS}, with_dact={self.WITH_DACT}, "
                f"with_act={self.WITH_ACT}, with_noop={self.WITH_NOOP}, "
                f"activation={self.ACTIVATION})")

    __repr__ = __str__

# ---------------------------------------------------------------------------
# Unified MXFP8 quantization kernel — shared memory tiled, single-pass
# ---------------------------------------------------------------------------
class MXFP8QuantizeSmemKernel:
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
        mX: cute.Tensor, # Input tensor to quantize
        mO_row: Optional[cute.Tensor], mS_row: Optional[cute.Tensor], # Rowwise output and scale tensors
        mO_col: Optional[cute.Tensor], mS_col: Optional[cute.Tensor], # Colwise output and scale tensors
        mAmax: Optional[cute.Tensor], # Global amax accumulator, only used in WITH_AMAX path
        mNoop: Optional[cute.Tensor], # 1-element cast_noop flag, only used in WITH_NOOP path
        # Backward-only slots, present to mirror the CUDA mxfp8::quantize signature
        # (act_input / dbias / workspace). NOT used yet — None on the forward path;
        # WITH_DACT/WITH_DBIAS configs are rejected upstream so these never carry data.
        mActInput: Optional[cute.Tensor],
        mDbias: Optional[cute.Tensor],
        mWorkspace: Optional[cute.Tensor],
        stream: CUstream,
    ):
        M = mX.shape[0]
        N = mX.shape[1]
        cfg = self.cfg
        max_norm_rcp = cfg.MAX_NORM_RCP
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
            if cutlass.const_expr(cfg.COLWISE):
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

        # Backward (dact): the activation input is a second G2S load, identical to
        # mX's. The kernel computes `grad · act'(act_input)`; here mX carries grad.
        tma_atom_act = None
        tma_src_act = None
        if cutlass.const_expr(cfg.WITH_DACT):
            tma_atom_act, tma_src_act = cute.nvgpu.cpasync.make_tiled_tma_atom(
                op_load, mActInput, smem_tile_layout, cta_tiler, num_multicast=1,
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
        if cutlass.const_expr(cfg.COLWISE):
            tma_atom_out_col, tma_dst_out_col = cute.nvgpu.cpasync.make_tiled_tma_atom(
                op_store, mO_col, out_smem_layout, cta_tiler, num_multicast=1,
            )
        
        # Decide when to perform dbias reduction
        DBIAS_REDUCTION_COLWISE: cutlass.Constexpr = False
        DBIAS_REDUCTION_ROWWISE: cutlass.Constexpr = False
        if cutlass.const_expr(cfg.WITH_DBIAS):
            # We prefer to perform dbias reduction in the colwise pass since it doesn't require shuffle
            if cutlass.const_expr(cfg.COLWISE):
                DBIAS_REDUCTION_COLWISE = True
            else:
                DBIAS_REDUCTION_ROWWISE = True

        # CUDA launches in (0,0), (1,0), (2,0)... order, so we should make N the leading dimension for better access pattern 
        # So consecutive blocks will move along the N dimension first, which is the innermost dimension in memory and we can use cache better
        grid = [
            cute.ceil_div(Int32(N), TILE_X),
            cute.ceil_div(M, TILE_Y * NUM_TILES),
        ]
        block = [THREADS_PER_CHUNK,]
        
        self.kernel(
            mX, mS_row, mS_col, mAmax, mNoop, mWorkspace,
            max_norm_rcp, mX.element_type,
            tma_atom, tma_src,
            tma_atom_out_row, tma_dst_out_row,
            tma_atom_out_col, tma_dst_out_col,
            tma_atom_act, tma_src_act,
        ).launch(
            grid=grid,
            block=block,
            stream=stream,
        )

    # Device entry (launched by __call__). Reads the cast_noop flag and runs the
    # work only if it is not set — matching the CUDA kernel's
    # `if (noop[0]==1.0f) return;`. When WITH_NOOP is off, mNoop is None and the
    # whole check is compiled out (so no flag is read).
    @cute.kernel
    def kernel(
        self,
        mX,
        mS_row,
        mS_col,
        mAmax,
        mNoop,
        mWorkspace,
        max_norm_rcp,
        dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
        tma_atom, tma_src, # how to use TMA to copy the input
        tma_atom_out_row, tma_dst_out_row, # how to use TMA to copy the rowwise output
        tma_atom_out_col, tma_dst_out_col, # how to use TMA to copy the colwise output
        tma_atom_act, tma_src_act, # dact only: how to copy the activation input
    ):
        cfg = self.cfg
        # `not const_expr(WITH_NOOP)` is a compile-time True when noop is disabled,
        # so Python short-circuits the `or` and never reads mNoop[0] (it is None).
        if not cutlass.const_expr(cfg.WITH_NOOP) or mNoop[0] != Float32(1.0):
            self._kernel_main(
                mX, mS_row, mS_col, mAmax, mWorkspace,
                max_norm_rcp, dtype,
                tma_atom, tma_src,
                tma_atom_out_row, tma_dst_out_row,
                tma_atom_out_col, tma_dst_out_col,
                tma_atom_act, tma_src_act,
            )

    # The actual quantize work. MUST be @cute.jit (not @cute.kernel): it is invoked
    # from the @cute.kernel `kernel` wrapper under a runtime noop branch, and only a
    # separately-traced @cute.jit callable may allocate shared memory inside such a
    # branch (an inlined/undecorated method or a nested @cute.kernel would fail).
    @cute.jit
    def _kernel_main(
        self,
        mX,
        mS_row,
        mS_col,
        mAmax,
        mWorkspace,
        max_norm_rcp,
        dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
        tma_atom, tma_src, # how to use TMA to copy the input
        tma_atom_out_row, tma_dst_out_row, # how to use TMA to copy the rowwise output
        tma_atom_out_col, tma_dst_out_col, # how to use TMA to copy the colwise output
        tma_atom_act, tma_src_act, # dact only: how to copy the activation input
    ):
        cfg = self.cfg

        if cutlass.const_expr(cfg.ROWWISE):
            mS_row = cute.zipped_divide(mS_row, (TILE_Y, TILE_X // SCALE_DIM))
        if cutlass.const_expr(cfg.COLWISE):
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
        if cutlass.const_expr(cfg.ROWWISE and cfg.COLWISE):
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
        elif cutlass.const_expr(cfg.ROWWISE and not cfg.COLWISE):
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

        # dact: the activation-input tile lives in its own smem buffer, same
        # shape/layout as sX. Allocated separately so the 4 SharedStorage variants
        # above don't have to fork again on WITH_DACT.
        if cutlass.const_expr(cfg.WITH_DACT):
            @cute.struct
            class DactStorage:
                sActInput: cute.struct.Align[
                    cute.struct.MemRange[dtype, TILE_Y * TILE_X * NUM_STAGES], 128
                ]
            dact_storage = smem.allocate(DactStorage)
            sActInput = dact_storage.sActInput.get_tensor(
                cute.make_layout(
                    ((TILE_Y, TILE_X), NUM_STAGES),
                    stride=((TILE_X, 1), TILE_Y * TILE_X),
                )
            )

        # Rowwise-only dbias needs a cross-thread (over THREADS_Y) smem reduction,
        # since each rowwise thread owns a row, not a column. Buffer is
        # [THREADS_Y][THREADS_X*(SCALE_DIM+1)] f32 — the +1 per scale-block padding
        # avoids bank conflicts, matching CUDA's DBIAS_BUFF_WIDTH.
        DBIAS_REDUCTION_ROWWISE = cutlass.const_expr(cfg.WITH_DBIAS and not cfg.COLWISE)
        DBIAS_BUFF_WIDTH = (TILE_X // SCALE_DIM) * (SCALE_DIM + 1)
        if cutlass.const_expr(DBIAS_REDUCTION_ROWWISE):
            @cute.struct
            class DbiasStorage:
                sDbias: cute.struct.MemRange[Float32, TILE_Y * DBIAS_BUFF_WIDTH]
            dbias_storage = smem.allocate(DbiasStorage)
            sDbias = dbias_storage.sDbias.get_tensor(
                cute.make_layout(TILE_Y * DBIAS_BUFF_WIDTH)
            )

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
        if cutlass.const_expr(cfg.COLWISE):
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
            if cutlass.const_expr(cfg.WITH_DACT):
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_act)

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
        # dact loads two tiles (grad + act_input) under the same per-stage barrier,
        # so the barrier must expect both copies' bytes.
        tx_count = TILE_Y * TILE_X * dtype.width // 8
        if cutlass.const_expr(cfg.WITH_DACT):
            tx_count *= 2

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

        # dact: identical partition for the activation-input load.
        if cutlass.const_expr(cfg.WITH_DACT):
            gA_tiled = cute.zipped_divide(tma_src_act, (TILE_Y, TILE_X))
            tXsA, tXgA = cute.nvgpu.cpasync.tma_partition(
                tma_atom_act,
                0,
                cute.make_layout(1),
                sActInput,
                gA_tiled,
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
        if cutlass.const_expr(cfg.COLWISE):
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
                if cutlass.const_expr(cfg.WITH_DACT):
                    cute.copy(
                        tma_atom_act,
                        tXgA[(None, (tile_y, bidx))],
                        tXsA[(None, prod_state.index)],
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

        # Per-thread partial dbias: thread tidx owns column tidx of the colwise
        # tile and accumulates its column sum over this CTA's rows (both stages).
        # Written to workspace[bidy, col] below; reduced over row-blocks separately.
        if cutlass.const_expr(cfg.WITH_DBIAS):
            block_dbias = Float32(0.0)
        # Rowwise-only dbias: each thread holds per-column partials for its 32-col
        # block, summed across stages, then cross-thread reduced (over THREADS_Y)
        # into block_dbias after the loop.
        if cutlass.const_expr(DBIAS_REDUCTION_ROWWISE):
            rowwise_dbias_arr = cute.make_rmem_tensor(
                layout_or_shape=cute.make_layout((SCALE_DIM,), stride=(1,)),
                dtype=Float32,
            )
            for c in cutlass.range_constexpr(SCALE_DIM):
                rowwise_dbias_arr[c] = Float32(0.0)

        # ---- Consumer: all threads quantize each completed tile. ----
        for stage in cutlass.range(num_tiles, unroll=1):
            mainloop_pipeline.consumer_wait(cons_state)
            sX_tile = sX[(None, stage)]          # (TILE_Y, TILE_X) bf16 (grad for dact)
            sActInput_tile = None
            if cutlass.const_expr(cfg.WITH_DACT):
                sActInput_tile = sActInput[(None, stage)]  # (TILE_Y, TILE_X) act_input

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
            if cutlass.const_expr(cfg.COLWISE):
                # The first row that belongs to this CTA. Each CTA handles NUM_TILES of (TILE_Y, TILE_X) tiles stacked vertically,
                # and each stage handles one of them.
                sO_col_tile = sO_col[(None, stage)]
                mS_col_stage = cute.flatten(mS_col[(None, (tile_idx_y, tile_idx_x))])

                amax_c, dbias_c = self._process_colwise(
                    sX_tile, sO_col_tile,
                    mS_col_stage, max_norm_rcp,
                    tile_idx_y * TILE_Y, bidx * TILE_X, M, N,
                    sActInput_tile,
                )
                if cutlass.const_expr(cfg.WITH_AMAX):
                    block_amax = cute.arch.fmax(block_amax, amax_c)
                if cutlass.const_expr(cfg.WITH_DBIAS):
                    block_dbias += dbias_c
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
                    mS_row_stage, max_norm_rcp,
                    tile_idx_y * TILE_Y, bidx * TILE_X, M, N,
                    sActInput_tile,
                    rowwise_dbias_arr if cutlass.const_expr(DBIAS_REDUCTION_ROWWISE) else None,
                )

                if cutlass.const_expr(cfg.WITH_AMAX):
                    block_amax = cute.arch.fmax(block_amax, amax_r)

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
                if cutlass.const_expr(cfg.COLWISE):
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

        # ---- rowwise-only dbias: cross-thread reduction over THREADS_Y ---------
        # In the rowwise pass each thread owns a row, so its rowwise_dbias_arr holds
        # per-column partials for its 32-col block. Transpose through smem so thread
        # tidx ends up owning column tidx of the chunk (mirrors CUDA's
        # partial_dbias_rowwise smem buffer + reduce over THREADS_Y).
        if cutlass.const_expr(DBIAS_REDUCTION_ROWWISE):
            THREADS_X = TILE_X // SCALE_DIM        # scale-blocks per row (=2)
            tid_Y = tidx // THREADS_X
            tid_X = tidx % THREADS_X
            for c in cutlass.range_constexpr(SCALE_DIM):
                sDbias[tid_Y * DBIAS_BUFF_WIDTH + tid_X * (SCALE_DIM + 1) + c] = \
                    rowwise_dbias_arr[c]
            cute.arch.sync_threads()
            # thread tidx owns column tidx; +block skips the per-block padding slot.
            block = tidx // SCALE_DIM
            block_dbias = Float32(0.0)
            for i in cutlass.range_constexpr(TILE_Y):
                block_dbias += sDbias[i * DBIAS_BUFF_WIDTH + tidx + block]

        # ---- dbias: write this CTA's per-column partial to the workspace -------
        # Thread tidx owns column (bidx*TILE_X + tidx). Each CTA-row-block (bidy)
        # contributes one row of the (blocks_Y, N) fp32 workspace; the reduction
        # over blocks_Y to the final dbias[N] is a separate step.
        if cutlass.const_expr(cfg.WITH_DBIAS):
            dbias_col = bidx * TILE_X + tidx
            if dbias_col < N:
                mWorkspace[(bidy, dbias_col)] = block_dbias

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
        max_norm_rcp,
        tile_row_start, # Int32 — global row of this stage's row 0
        tile_col_start, # Int32 — global col of this CTA's col 0
        M, N,           # Int32 — full input extents, for OOB masking
        sActInput_tile=None,  # (TILE_Y, TILE_X) act_input tile (dact only)
        dbias_acc=None,       # rmem Float32[SCALE_DIM] dbias accumulator (rowwise-only dbias)
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
        return quantize_rowwise_mxfp8(
            sX_tile,
            sO_row_tile,
            mS_row_stage,
            max_norm_rcp,
            tile_row_start,
            tile_col_start,
            M,
            N,
            ACTIVATION=cfg.ACTIVATION,
            DTYPE=cfg.DTYPE,
            ROWWISE=cfg.ROWWISE,
            COLWISE=cfg.COLWISE,
            FP8_DTYPE=cfg.FP8_DTYPE,
            TILE_Y=TILE_Y,
            SCALE_DIM=SCALE_DIM,
            WAVES=WAVES,
            THREADS_PER_WARP=THREADS_PER_WARP,
            THREADS_PER_BANK=THREADS_PER_BANK,
            PACK_SIZE=PACK_SIZE,
            WITH_ACT=cfg.WITH_ACT,
            WITH_DACT=cfg.WITH_DACT,
            sA_tile=sActInput_tile,
            DBIAS_REDUCTION=cfg.WITH_DBIAS and not cfg.COLWISE,
            dbias_acc=dbias_acc,
        )

    @cute.jit
    def _process_colwise(
        self,
        sX_tile,        # (TILE_Y, TILE_X) bf16/fp16 smem view, post-TMA
        sO_col_tile,    # (TILE_Y, TILE_X) uint8 smem view (colwise FP8 output)
        mS_col_stage,   # colwise scale tensor (1D swizzled, or 2D linear)
        max_norm_rcp,
        tile_row_start, # Int32 — global row of this stage's row 0
        tile_col_start, # Int32 — global col of this CTA's col 0
        M, N,           # Int32 — full input extents, for OOB masking
        sActInput_tile=None,  # (TILE_Y, TILE_X) act_input tile (dact only)
    ):
        """Colwise MXFP8 pass: thread `tidx` owns column `tidx` of the (32, 64)
        smem tile — 32 elements down. Writes quantized bytes into `sO_col_tile`
        so the caller can flush with a TMA S2G — matches C++'s
        `out_colwise_data_sh` + `cp.async.bulk.tensor.2d.shared_to_global`.
        """
        cfg = self.cfg
        return quantize_colwise_mxfp8(
            sX_tile,
            sO_col_tile,
            mS_col_stage,
            max_norm_rcp,
            tile_row_start,
            tile_col_start,
            M,
            N,
            ACTIVATION=cfg.ACTIVATION,
            DTYPE=cfg.DTYPE,
            FP8_DTYPE=cfg.FP8_DTYPE,
            SWIZZLE=cfg.WITH_GEMM_SWIZZLED_SCALES,
            TILE_X=TILE_X,
            TILE_Y=TILE_Y,
            SCALE_DIM=SCALE_DIM,
            WITH_ACT=cfg.WITH_ACT,
            WITH_DACT=cfg.WITH_DACT,
            sA_tile=sActInput_tile,
            WITH_DBIAS=cfg.WITH_DBIAS,
        )

def compile_cutedsl_function_from_cfg(cfg):
    """
    Return the compiled CuTeDSL function object for the given MXFP8 quantization config.
    """

    kernel_obj = MXFP8QuantizeSmemKernel(cfg)

    # stride_order=(1, 0): row-major, dim 1 stride 1. 1D: (0,).
    kw_rm16_2d = dict(stride_order=(1, 0),
                      memspace=cute.AddressSpace.gmem, assumed_align=16)
    kw_rm4_2d  = dict(stride_order=(1, 0),
                      memspace=cute.AddressSpace.gmem, assumed_align=4)
    kw_rm4_1d  = dict(stride_order=(0,),
                      memspace=cute.AddressSpace.gmem, assumed_align=4)
    def fake(dtype, shape, kw):
        return cute.runtime.make_fake_compact_tensor(dtype, shape, **kw)

    
    # M, N must be divisible by the MXFP8 scale-block size (SCALE_DIM = 32) — the
    # same alignment the CUDA C++ kernel requires. The C++ dispatcher gates on the
    # matching value (kCuTeDSLMXFP8ShapeAlignment in cast/dispatch/quantize.cuh)
    # and falls back to CUDA for anything not divisible by it, so tvm-ffi never
    # sees a shape this kernel can't accept.
    sym_M = cute.sym_int32(divisibility=SCALE_DIM)
    sym_N = cute.sym_int32(divisibility=SCALE_DIM)
    in_shape = out_shape = (sym_M, sym_N)
    # TE allocates scale tensors at a padded shape (see
    # MXFP8Quantizer::get_scale_shape in transformer_engine/pytorch/csrc):
    #   rowwise:    (roundup(M, 128),     roundup(N // 32, 4))
    #   columnwise: (roundup(M // 32, 4), roundup(N, 128))
    # These padded extents are NOT M/N (and SymInt has no `//`/`+`), so give the
    # scales their own fresh syms carrying the divisibility the padding
    # guarantees (rowwise: 128 x 4; colwise: 4 x 128).
    scale_r_shape = (cute.sym_int32(divisibility=128), cute.sym_int32(divisibility=4))
    scale_c_shape = (cute.sym_int32(divisibility=4),   cute.sym_int32(divisibility=128))
    # Scale dim-1 is only 4-byte-divisible, so a 16-byte alignment promise would
    # be a lie for many shapes; the per-block scale stores are byte-wise anyway,
    # so 4-byte alignment loses nothing.
    scale_kw = kw_rm4_2d

    in_fake        = fake(cfg.DTYPE,  in_shape,      kw_rm16_2d)
    out_row_fake   = fake(cute.Uint8, out_shape,     kw_rm16_2d) if cfg.ROWWISE   else None
    scale_row_fake = fake(cute.Uint8, scale_r_shape, scale_kw)   if cfg.ROWWISE   else None
    out_col_fake   = fake(cute.Uint8, out_shape,     kw_rm16_2d) if cfg.COLWISE   else None
    scale_col_fake = fake(cute.Uint8, scale_c_shape, scale_kw)   if cfg.COLWISE   else None
    amax_fake      = fake(Float32, (1,), kw_rm4_1d)              if cfg.WITH_AMAX else None
    noop_fake      = fake(Float32, (1,), kw_rm4_1d)              if cfg.WITH_NOOP else None
    # Backward-only slots (act_input/dbias/workspace). Always None today —
    # WITH_DACT/WITH_DBIAS are rejected in the config — but kept in the compile
    # signature so the tvm-ffi protocol matches the CUDA mxfp8::quantize args.
    act_input_fake = fake(cfg.DTYPE, in_shape, kw_rm16_2d)       if cfg.WITH_DACT  else None
    # dbias: the kernel never writes the dbias tensor — it writes per-row-block
    # partials into the workspace (shape (blocks_Y, N) fp32, blocks_Y = ceil(M/64),
    # set by the C++ worker's size query). The final reduction lives elsewhere, so
    # mDbias stays None and only the workspace fake is built.
    dbias_fake     = None
    ws_shape       = (cute.sym_int32(), sym_N)  # (blocks_Y, N); N ties to input N
    workspace_fake = fake(Float32, ws_shape, kw_rm4_2d)         if cfg.WITH_DBIAS else None

    compiled = cute.compile(
        kernel_obj,
        in_fake,                            # mX
        out_row_fake,   scale_row_fake,     # mO_row, mS_row
        out_col_fake,   scale_col_fake,     # mO_col, mS_col
        amax_fake,                          # mAmax
        noop_fake,                          # mNoop (1-element cast_noop flag)
        act_input_fake,                     # mActInput (backward slot, unused)
        dbias_fake,                         # mDbias    (backward slot, unused)
        workspace_fake,                     # mWorkspace(backward slot, unused)
        cute.runtime.make_fake_stream(),    # stream (compiled as an explicit tvm-ffi
                                            # "handle" arg; C++ passes the CUDA stream
                                            # as void*)
        options="--enable-tvm-ffi",
    )
    return compiled

def get_mxfp8_quantization_function(
    fn_name: str,
    dtype: str,
    fp8_dtype: str,
    rowwise: bool,
    colwise: bool,
    with_gemm_swizzled_scales: bool,
    with_amax: bool,
    with_dbias: bool,
    with_dact: bool,
    with_act: bool,
    with_noop: bool,
    activation: str,
) -> bool:
    """Compile the MXFP8 quantize kernel for this config and register it in the
    TVM-FFI global registry under EXACTLY `fn_name` (the key the C++ dispatcher
    built; Python treats it as an opaque name). Returns True if a kernel is
    registered under `fn_name` (the C++ side then fetches it with
    GetGlobal(fn_name)); False if the config is unsupported, so the caller caches
    the negative result and falls back to the CUDA C++ kernel.

    The registry owns the compiled kernel's lifetime — important because it wraps
    a Python object, and tvm-ffi releases registry entries at interpreter
    shutdown (whereas a C++-held handle would be released after finalize → crash).
    """
    # Already registered (e.g. by a prior call) -> supported.
    if tvm_ffi.get_global_func(fn_name, allow_missing=True) is not None:
        return True

    try:
        cfg = MXFP8QuantizeConfig(
            dtype=dtype,
            fp8_dtype=fp8_dtype,
            rowwise=rowwise,
            colwise=colwise,
            with_gemm_swizzled_scales=with_gemm_swizzled_scales,
            with_amax=with_amax,
            with_dbias=with_dbias,
            with_dact=with_dact,
            with_act=with_act,
            with_noop=with_noop,
            activation=activation,
        )
    except ValueError as e:
        # The exception message states exactly why the config is unsupported
        # (unknown dtype/activation, dbias not implemented, ...). Surfacing it as a
        # warning lets the C++ dispatcher's CUDA fallback be recognized as expected.
        logger.warning(f"CuTeDSL MXFP8 backend does not support this config, "
                       f"falling back to the CUDA C++ kernel: {e}")
        return False

    logger.debug(f"Compiling CuTeDSL MXFP8 quantization kernel for {cfg}")
    compiled = compile_cutedsl_function_from_cfg(cfg)
    tvm_ffi.register_global_func(fn_name, compiled, override=True)

    return True

# Exposed so the C++ dispatcher can request on-demand compilation by name.
tvm_ffi.register_global_func("get_mxfp8_quantization_function", get_mxfp8_quantization_function, override=True)
