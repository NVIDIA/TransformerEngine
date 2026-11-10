# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MXFP8 2D block-scaling quantization kernel implemented in CuTeDSL.

Ports the kIs2DBlockScaling path of quantize_mxfp8_kernel (quantize_mxfp8.cuh): every 32x32
block of the input shares one E8M0 scale, which the rowwise scales repeat down the block's 32
rows and the colwise scales repeat across its 32 columns.

The TMA pipeline is the general G2S kernel's; only the per-tile math differs, so this subclasses
it and swaps in its own tile function. A tile is 32 rows by TILE_COLS, i.e. TILE_COLS // 32
blocks side by side, and the colwise pass maps warp w onto columns [32w, 32w + 32), which is
exactly block w, so a warp reduction of the per-column amaxes yields the block amax. The rowwise
pass reads it back from SMEM, or, when there is no colwise pass, reduces its per-row amaxes over
the block's 32 rows itself.

Only a plain cast is implemented: the C APIs that fuse dbias or an activation pass no
quantization config, so only plain casts (nvte_quantize_v2, nvte_multi_tensor_quantize) can
request 2D scaling.
"""

from typing import Optional, Type

import cutlass
from cutlass import cute
from cutlass import Float32, Int32, Int64, Uint16, Uint32, Uint8

from transformer_engine.common.CuTeDSL.utils import (
    device_compute_capability,
    is_packed16,
    abs_max_x2_bf16,
    abs_max_x2_f16,
    abs_max_scalar_bf16,
    abs_max_scalar_f16,
    to_f32_bf16,
    to_f32_f16,
    x2_lo_to_f32_bf16,
    x2_lo_to_f32_f16,
    x2_hi_to_f32_bf16,
    x2_hi_to_f32_f16,
    fabs_f32,
    exp2f_rcp,
    pack_f32x2,
    unpack_i64_to_i32x2,
)
from transformer_engine.common.CuTeDSL.utils_fp8 import (
    get_cvt_f32x2_to_fp8x2_func,
    cvt_f32_to_fp8e8m0fnu,
    mul_f32x2_cvt_f32x4_to_fp8x4,
    mul_f32x2_cvt_packed16x4_to_fp8x4,
)
from .quantize_mxfp8_common import (
    MXFP8QuantizeConfig,
    MXFP8_BLOCK_SCALING_SIZE,
    THREADS_PER_WARP,
)
from .quantize_mxfp8_g2s_general import MXFP8QuantizeGeneralG2SKernel


@cute.jit
def quantize_colwise_mxfp8_2d(
    sX_tile: cute.Tensor,  # (TILE_Y, TILE_X) input smem view, post-TMA
    sO_col_tile: cute.Tensor,  # (TILE_Y, TILE_X) fp8 smem view (colwise FP8 output)
    mS_col_stage: cute.Tensor,  # colwise scale tensor (1D swizzled, or 2D linear)
    sBlockAmax: cute.Tensor,  # receives the amax of block w at index w
    MAX_NORM_RCP: cutlass.Constexpr[float],
    tile_row_start: Int32,
    tile_col_start: Int32,
    M: Int32,
    N: Int32,
    DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
    FP8_DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
    SWIZZLE: cutlass.Constexpr[bool],
    TILE_X: cutlass.Constexpr[int],
    TILE_Y: cutlass.Constexpr[int],
    SKIP_SCALE_BOUNDS: cutlass.Constexpr[bool],
):
    """Quantize one SMEM tile colwise with one scale per 32x32 block; returns the block amax."""
    tidx, _, _ = cute.arch.thread_idx()

    # Each thread handles one vertical MXFP8 block, so we have 1 thread along the Y direction since one tile only has
    # one block in the Y (column) direction
    CTA_THREADS_Y = MXFP8_BLOCK_SCALING_SIZE // TILE_Y  # 1 thread along the Y direction
    _, tv_layout = cute.make_layout_tv(
        thr_layout=cute.make_layout((CTA_THREADS_Y, TILE_X), stride=(TILE_X, 1)),
        val_layout=cute.make_layout((MXFP8_BLOCK_SCALING_SIZE, 1), stride=(1, 1)),
    )
    sX_thread = cute.composition(sX_tile, tv_layout)[tidx, None]
    sO_thread = cute.composition(sO_col_tile, tv_layout)[tidx, None]

    # Gather this thread's 32 values from the MXFP8 block and its own amax.
    # Note: TMA zero-fills past M and N, so the padding values never win when comparing amax.
    in_c = [None] * MXFP8_BLOCK_SCALING_SIZE
    if cutlass.const_expr(is_packed16(DTYPE)):
        abs_max_scalar = abs_max_scalar_f16 if DTYPE is cutlass.Float16 else abs_max_scalar_bf16
        to_f32 = to_f32_f16 if DTYPE is cutlass.Float16 else to_f32_bf16
        in_c[0] = sX_thread[0]
        amax_value = in_c[0]
        for i in cutlass.range_constexpr(1, MXFP8_BLOCK_SCALING_SIZE):
            in_c[i] = sX_thread[i]
            amax_value = abs_max_scalar(amax_value, in_c[i])
        amax_c = fabs_f32(to_f32(amax_value))
    else:
        to_f32 = None
        in_c[0] = Float32(sX_thread[0])
        amax_c = fabs_f32(in_c[0])
        for i in cutlass.range_constexpr(1, MXFP8_BLOCK_SCALING_SIZE):
            in_c[i] = Float32(sX_thread[i])
            amax_c = cute.arch.fmax(amax_c, fabs_f32(in_c[i]))

    # Reduce the amaxes of the 32 threads in this warp to get the block amax, and let the leader thread
    # of each warp write it to SMEM. 
    # Use f32 redux if it is available (blackwell family); otherwise use butterfly shuffle for reduction
    if cutlass.const_expr(device_compute_capability() in ((10, 0), (10, 3))):
        block_amax = cute.arch.warp_redux_sync(amax_c, kind="fmax")
    else:
        block_amax = amax_c
        for k in cutlass.range_constexpr(THREADS_PER_WARP.bit_length() - 1):
            block_amax = cute.arch.fmax(
                block_amax, cute.arch.shuffle_sync_bfly(block_amax, offset=1 << k)
            )
    if tidx % THREADS_PER_WARP == 0:
        sBlockAmax[tidx // THREADS_PER_WARP] = block_amax

    biased_exp_c = cvt_f32_to_fp8e8m0fnu(block_amax * MAX_NORM_RCP)
    if cutlass.const_expr(SWIZZLE):
        scale_coord = (0, tidx % 32, tidx // 32)
    else:
        scale_coord = (0, tidx)
    if cutlass.const_expr(SKIP_SCALE_BOUNDS):
        mS_col_stage[scale_coord] = biased_exp_c
    elif tile_row_start < M and tile_col_start + tidx < N:
        mS_col_stage[scale_coord] = biased_exp_c

    inv_scale_c = exp2f_rcp(biased_exp_c)
    cvt_x2_func = get_cvt_f32x2_to_fp8x2_func(FP8_DTYPE)
    sO_thread_fp8 = cute.make_tensor(
        cute.recast_ptr(sO_thread.iterator, dtype=FP8_DTYPE), sO_thread.layout
    )
    for j in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE // 2):
        lo, hi = 2 * j, 2 * j + 1
        x_lo, x_hi = in_c[lo], in_c[hi]
        if cutlass.const_expr(to_f32 is not None):
            x_lo, x_hi = to_f32(x_lo), to_f32(x_hi)
        pair = cvt_x2_func(x_hi * inv_scale_c, x_lo * inv_scale_c)
        sO_thread_fp8[lo] = Uint8(pair & Uint16(0xFF)).bitcast(FP8_DTYPE)
        sO_thread_fp8[hi] = Uint8(pair >> Uint16(8)).bitcast(FP8_DTYPE)

    return block_amax


@cute.jit
def quantize_rowwise_mxfp8_2d(
    sX_tile: cute.Tensor,  # (TILE_Y, TILE_X) input smem view, post-TMA
    sO_row_tile: cute.Tensor,  # (TILE_Y, TILE_X) fp8 smem view (rowwise FP8 output)
    mS_row_stage: cute.Tensor,  # rowwise scale tensor (1D swizzled, or 2D linear)
    sBlockAmax: cute.Tensor,  # block amaxes from the colwise pass, or scratch for the reduction
    MAX_NORM_RCP: cutlass.Constexpr[float],
    tile_row_start: Int32,
    tile_col_start: Int32,
    M: Int32,
    N: Int32,
    DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
    FP8_DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
    TILE_X: cutlass.Constexpr[int],
    TILE_Y: cutlass.Constexpr[int],
    SKIP_SCALE_BOUNDS: cutlass.Constexpr[bool],
    BLOCK_AMAX_IN_SMEM: cutlass.Constexpr[
        bool
    ],  # If colwise pass has already left the block amaxes in sBlockAmax
):
    """Quantize one SMEM tile rowwise with one scale per 32x32 block; returns the block amax.

    With BLOCK_AMAX_IN_SMEM the colwise pass has already left the block amaxes in sBlockAmax
    (and the caller synchronized); otherwise they are reduced here, which syncs the CTA.
    """
    tidx, _, _ = cute.arch.thread_idx()

    # Same vectorized, bank-staggered loads as the general kernel's rowwise pass.
    PACK_SIZE = 4
    WAVES = MXFP8_BLOCK_SCALING_SIZE // PACK_SIZE
    THREADS_PER_BANK = 32 * 4 // MXFP8_BLOCK_SCALING_SIZE

    # Each thread covers one horizontal MXFP8 block, so we have TILE_X // MXFP8_BLOCK_SCALING_SIZE threads
    # along X (row direction)
    CTA_THREADS_X = TILE_X // MXFP8_BLOCK_SCALING_SIZE
    NUM_WARPS = TILE_Y * CTA_THREADS_X // THREADS_PER_WARP
    # A CTA consists of NUM_WARPS warps stacked together
    thr_layout = cute.blocked_product(
        cute.make_layout(
            (THREADS_PER_WARP // CTA_THREADS_X, CTA_THREADS_X), stride=(CTA_THREADS_X, 1)
        ),  # Thread layout in a warp, (16, 2)
        cute.make_layout((NUM_WARPS, 1)),  # Warp layout in a CTA (2, 1)
    )

    _, tv_layout = cute.make_layout_tv(
        thr_layout=thr_layout,
        val_layout=cute.make_layout((1, MXFP8_BLOCK_SCALING_SIZE), stride=(0, 1)),
    )

    sX_thread = cute.composition(sX_tile, tv_layout)[tidx, None]
    sO_thread = cute.composition(sO_row_tile, tv_layout)[tidx, None]
    sO_thread_u32 = cute.make_tensor(
        cute.recast_ptr(sO_thread.iterator, dtype=Uint32),
        cute.make_layout((MXFP8_BLOCK_SCALING_SIZE // 4,), stride=(1,)),
    )

    # Coordinates in thr_layout: (lane_y, lane_x) within the warp, warps stacked along Y.
    warp_id = tidx // THREADS_PER_WARP
    lane_id = tidx % THREADS_PER_WARP
    lane_y = lane_id // CTA_THREADS_X
    lane_x = lane_id % CTA_THREADS_X
    thread_y = warp_id * (THREADS_PER_WARP // CTA_THREADS_X) + lane_y
    thread_x = lane_x

    # Each thread start reading from the specfic bank based on its thread ID so they can do their best to access different banks
    # to avoid bank conflict.
    bank_group = (tidx % THREADS_PER_WARP) // THREADS_PER_BANK
    # The offset this thread should start reading from based on what's its first bank to access.
    offset = bank_group * PACK_SIZE

    if cutlass.const_expr(is_packed16(DTYPE)):
        sX_thread_rw_i64 = cute.make_tensor(
            cute.recast_ptr(sX_thread.iterator, dtype=Int64),
            cute.make_layout((1, MXFP8_BLOCK_SCALING_SIZE // 4), stride=(0, 1)),
        )
        in_r = [[None, None] for _ in range(WAVES)]
        for w in cutlass.range_constexpr(WAVES):
            idx = (w + offset // 4) % (MXFP8_BLOCK_SCALING_SIZE // 4)
            in_r[w][0], in_r[w][1] = unpack_i64_to_i32x2(sX_thread_rw_i64[0, idx])
    else:
        sX_thread_rw = cute.make_tensor(
            sX_thread.iterator,
            cute.make_layout((1, MXFP8_BLOCK_SCALING_SIZE), stride=(0, 1)),
        )
        in_r = [[None] * PACK_SIZE for _ in range(WAVES)]
        for w in cutlass.range_constexpr(WAVES):
            start = (w * PACK_SIZE + offset) % MXFP8_BLOCK_SCALING_SIZE
            for i in cutlass.range_constexpr(PACK_SIZE):
                in_r[w][i] = Float32(sX_thread_rw[0, start + i])

    if cutlass.const_expr(BLOCK_AMAX_IN_SMEM):
        # If colwise pass has already writes the block amaxes into sBlockAmax, just read it back.
        block_amax = sBlockAmax[thread_x]
    else:
        # Otherwise we need to compute the block amaxes in this rowwise pass
        if cutlass.const_expr(is_packed16(DTYPE)):
            abs_max_x2 = abs_max_x2_f16 if DTYPE is cutlass.Float16 else abs_max_x2_bf16
            x2_lo_to_f32 = x2_lo_to_f32_f16 if DTYPE is cutlass.Float16 else x2_lo_to_f32_bf16
            x2_hi_to_f32 = x2_hi_to_f32_f16 if DTYPE is cutlass.Float16 else x2_hi_to_f32_bf16
            # Load the fisst wave's two values into amax_2x, then fold in the rest of the waves.
            amax_2x = abs_max_x2(in_r[0][0], in_r[0][1])
            for w in cutlass.range_constexpr(1, WAVES):
                amax_2x = abs_max_x2(amax_2x, in_r[w][0])
                amax_2x = abs_max_x2(amax_2x, in_r[w][1])
            amax_r = cute.arch.fmax(
                fabs_f32(x2_lo_to_f32(amax_2x)), fabs_f32(x2_hi_to_f32(amax_2x))
            )
        else:
            vals = [in_r[w][i] for w in range(WAVES) for i in range(PACK_SIZE)]
            amax_r = fabs_f32(vals[0])
            for j in cutlass.range_constexpr(1, len(vals)):
                amax_r = cute.arch.fmax(amax_r, fabs_f32(vals[j]))

        # Use butterfly shuffles to reduce the 32 threads' local amax in this warp to get 2 block amax
        # for the left half and the right half of the warp (each warp with shape (16, 1))
        for k in cutlass.range_constexpr((THREADS_PER_WARP // CTA_THREADS_X).bit_length() - 1):
            amax_r = cute.arch.fmax(
                amax_r, cute.arch.shuffle_sync_bfly(amax_r, offset=CTA_THREADS_X << k)
            )

        # Write the half-warp block amaxes to SMEM, then let the first warp reduce them to get the final block amax.
        sWarpBlockAmax = cute.make_tensor(
            sBlockAmax.iterator,
            cute.make_layout((NUM_WARPS, CTA_THREADS_X), stride=(CTA_THREADS_X, 1)),
        )
        # Let the threads in the first row in each half warp write its partial warp amax to SMEM
        # Warp shape is (16, 2) so each warp writes 2 partial amax for a (16, 32) block which will be reduced to produce
        # the (32, 32) block amax
        if lane_y == 0:
            sWarpBlockAmax[(warp_id, lane_x)] = amax_r
        cute.arch.sync_threads()

        # Reduce 2 warps' amax to the whole (32, 32) block's amax
        block_amax = sWarpBlockAmax[(0, thread_x)]
        for w in cutlass.range_constexpr(1, NUM_WARPS):
            block_amax = cute.arch.fmax(block_amax, sWarpBlockAmax[(w, thread_x)])

    biased_exp_r = cvt_f32_to_fp8e8m0fnu(block_amax * MAX_NORM_RCP)
    if cutlass.const_expr(SKIP_SCALE_BOUNDS):
        mS_row_stage[(thread_y, thread_x)] = biased_exp_r
    elif tile_row_start + thread_y < M and tile_col_start + thread_x * MXFP8_BLOCK_SCALING_SIZE < N:
        mS_row_stage[(thread_y, thread_x)] = biased_exp_r

    inv_scale_r = exp2f_rcp(biased_exp_r)
    scale_2x = pack_f32x2(inv_scale_r, inv_scale_r)
    if cutlass.const_expr(is_packed16(DTYPE)):
        mul_cvt_x4_func = mul_f32x2_cvt_packed16x4_to_fp8x4(DTYPE, FP8_DTYPE)
    else:
        mul_cvt_x4_func = mul_f32x2_cvt_f32x4_to_fp8x4(FP8_DTYPE)
    for w in cutlass.range_constexpr(WAVES):
        idx = ((w * PACK_SIZE + offset) % MXFP8_BLOCK_SCALING_SIZE) // 4
        if cutlass.const_expr(is_packed16(DTYPE)):
            sO_thread_u32[idx] = mul_cvt_x4_func(in_r[w][0], in_r[w][1], scale_2x)
        else:
            sO_thread_u32[idx] = mul_cvt_x4_func(
                in_r[w][0], in_r[w][1], in_r[w][2], in_r[w][3], scale_2x
            )

    return block_amax


@cute.jit
def quantize_mxfp8_smem_tile_2d(  # pylint: disable=unused-argument
    sX_tile: cute.Tensor,
    sActInput_tile: Optional[cute.Tensor],
    sO_row_tile: Optional[cute.Tensor],
    mS_row_stage: Optional[cute.Tensor],
    sO_col_tile: Optional[cute.Tensor],
    mS_col_stage: Optional[cute.Tensor],
    rowwise_dbias_acc: Optional[cute.Tensor],
    tile_row_start: Int32,
    tile_col_start: Int32,
    M: Int32,
    N: Int32,
    ROWWISE: cutlass.Constexpr[bool],
    COLWISE: cutlass.Constexpr[bool],
    WITH_AMAX: cutlass.Constexpr[bool],
    MAX_NORM_RCP: cutlass.Constexpr[float],
    DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
    FP8_DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
    WITH_GEMM_SWIZZLED_SCALES: cutlass.Constexpr[bool],
    TILE_X: cutlass.Constexpr[int],
    TILE_Y: cutlass.Constexpr[int],
    SKIP_SCALE_BOUNDS: cutlass.Constexpr[bool],
    smem: cutlass.memory.SmemAllocator,
    **_fusion_args,  # The general tile function's fusion knobs, all off for a plain cast.
):
    """2D quantize a SMEM tile; returns (amax, None)."""
    NUM_BLOCKS_X = TILE_X // MXFP8_BLOCK_SCALING_SIZE
    NUM_WARPS = TILE_Y * NUM_BLOCKS_X // THREADS_PER_WARP

    # One amax per (warp, block) pair, for the rowwise-only cross-warp reduction.
    @cute.struct
    class BlockAmaxStorage:  # pylint: disable=missing-class-docstring
        sBlockAmax: cute.struct.MemRange[Float32, NUM_WARPS * NUM_BLOCKS_X]

    sBlockAmax = smem.allocate(BlockAmaxStorage).sBlockAmax.get_tensor(
        cute.make_layout(NUM_WARPS * NUM_BLOCKS_X)
    )

    block_amax = Float32(0.0)
    if cutlass.const_expr(COLWISE):
        block_amax = quantize_colwise_mxfp8_2d(
            sX_tile,
            sO_col_tile,
            mS_col_stage,
            sBlockAmax,
            MAX_NORM_RCP,
            tile_row_start,
            tile_col_start,
            M,
            N,
            DTYPE=DTYPE,
            FP8_DTYPE=FP8_DTYPE,
            SWIZZLE=WITH_GEMM_SWIZZLED_SCALES,
            TILE_X=TILE_X,
            TILE_Y=TILE_Y,
            SKIP_SCALE_BOUNDS=SKIP_SCALE_BOUNDS,
        )
        if cutlass.const_expr(ROWWISE):
            # Publish the block amaxes to the rowwise pass.
            cute.arch.sync_threads()
    if cutlass.const_expr(ROWWISE):
        block_amax = quantize_rowwise_mxfp8_2d(
            sX_tile,
            sO_row_tile,
            mS_row_stage,
            sBlockAmax,
            MAX_NORM_RCP,
            tile_row_start,
            tile_col_start,
            M,
            N,
            DTYPE=DTYPE,
            FP8_DTYPE=FP8_DTYPE,
            TILE_X=TILE_X,
            TILE_Y=TILE_Y,
            SKIP_SCALE_BOUNDS=SKIP_SCALE_BOUNDS,
            BLOCK_AMAX_IN_SMEM=COLWISE,
        )
    amax = None
    if cutlass.const_expr(WITH_AMAX):
        amax = block_amax
    return amax, None


class MXFP8Quantize2DKernel(MXFP8QuantizeGeneralG2SKernel):
    """MXFP8 quantization with one scale per 32x32 block (plain cast only).

    Reuses the general G2S kernel's pipeline with its own tile function and tile configuration.
    """

    _quantize_smem_tile = staticmethod(quantize_mxfp8_smem_tile_2d)

    def __init__(
        self,
        cfg: MXFP8QuantizeConfig,
        SKIP_INPUT_MASKING: bool = False,
        SKIP_SCALE_BOUNDS: bool = False,
    ):
        assert cfg.USE_2D_QUANTIZATION
        # 2D quantization is only supported for plain casts and no fusion should be requested
        assert not (cfg.WITH_DBIAS or cfg.WITH_ACT or cfg.WITH_DACT)

        super().__init__(cfg, SKIP_INPUT_MASKING, SKIP_SCALE_BOUNDS)

        # Overrides the general kernel's tile configuration with the 2D kernel's
        self._NUM_TILES = 2
        self._USE_STATIC_NUM_TILES = True
        self._MIN_BLOCKS_PER_MP = 0

        self._NUM_WARPS = 2
        self._THREADS_PER_CTA = self._NUM_WARPS * THREADS_PER_WARP

        # In rowwise pass,
        self._TILE_ROWS = MXFP8_BLOCK_SCALING_SIZE
        # In colwise pass, each thread covers one vertical MXFP8 block, and since THREADS_PER_WARP == MXFP8_BLOCK_SCALING_SIZE,
        # two warps cover 2 * MXFP8_BLOCK_SCALING_SIZE columns
        self._TILE_COLS = self._NUM_WARPS * MXFP8_BLOCK_SCALING_SIZE
