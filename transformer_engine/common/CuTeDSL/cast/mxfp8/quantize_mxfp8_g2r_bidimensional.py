# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Register-resident CuTeDSL bidimensional MXFP8 cast, ported from cast_bidim.cuh."""

from typing import Callable, Optional, Type

import cutlass
from cutlass import cute
from cutlass import Int32, Int64, Uint32, Uint8
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module

from transformer_engine.common.CuTeDSL.utils import (
    abs_max_x2_bf16,
    bf16_mx_scale_reciprocal,
    bf16_mx_scale_reciprocal_pair,
    bf16_mx_scale_bytes_pair,
    create_l2_policy,
)
from transformer_engine.common.CuTeDSL.utils_fp8 import mul_bf16x2x2_cvt_bf16x4_to_fp8x4
from .quantize_mxfp8_common import (
    MXFP8_BLOCK_SCALING_SIZE,
    MXFP8QuantizeConfig,
    MXFP8QuantizeKernelBase,
    bf16_pair_magnitude,
    noop_flag_is_set,
)


@cute.jit
def _quantize_rowwise(
    rX: cute.Tensor,
    gO_row_thread: cute.Tensor,
    gS_row_thread: cute.Tensor,
    # Atoms
    output_store_atom: cute.CopyAtom,
    row_scale_store_atom: cute.CopyAtom,
    # Miscellaneous
    cache_policy: Int64,
    scale_and_pack: cutlass.Constexpr[Callable[..., Uint32]],
    # Constexprs
    THREADS_X_PER_MX_BLOCK: cutlass.Constexpr[int],
    ELEMENTS_Y_PER_THREAD: cutlass.Constexpr[int],
    ELEMENTS_X_PER_THREAD: cutlass.Constexpr[int],
    SCALE_DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
    FP8_DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
):
    tidx, _, _ = cute.arch.thread_idx()
    MAX_EXPONENT = 8 if FP8_DTYPE is cutlass.Float8E4M3FN else 15

    @cute.jit
    def extract_mx_block_amax(values_i32: cute.Tensor) -> Uint32:
        """Reduce a 32-column MXFP8 block across its two or four lane owners."""
        amax = values_i32[0]
        for bf16x2_idx in cutlass.range_constexpr(1, cute.size(values_i32)):
            amax = abs_max_x2_bf16(amax, values_i32[bf16x2_idx])
        for stage in cutlass.range_constexpr(THREADS_X_PER_MX_BLOCK.bit_length() - 1):
            amax = abs_max_x2_bf16(amax, cute.arch.shuffle_sync_bfly(amax, 1 << stage))
        return bf16_pair_magnitude(amax)

    rS_row = cute.make_rmem_tensor((1,), SCALE_DTYPE)
    rO_row = cute.make_rmem_tensor((ELEMENTS_X_PER_THREAD,), FP8_DTYPE)
    rO_row_u32 = cute.recast_tensor(rO_row, Uint32)

    # Only one thread from threads that share the MXFP8 block writes the rowwise scale
    is_writer_thread = tidx % THREADS_X_PER_MX_BLOCK == 0
    for row_idx in cutlass.range_constexpr(ELEMENTS_Y_PER_THREAD):
        rX_row_i32 = cute.recast_tensor(rX[None, row_idx], Int32)
        reciprocal = bf16_mx_scale_reciprocal(extract_mx_block_amax(rX_row_i32), MAX_EXPONENT)
        reciprocal_pair = reciprocal | (reciprocal << Uint32(16))
        for pack_idx in cutlass.range_constexpr(
            cute.size(rO_row_u32)
        ):  # 4 output bytes in one pack
            rO_row_u32[pack_idx] = scale_and_pack(
                rX_row_i32[2 * pack_idx],
                reciprocal_pair,
                rX_row_i32[2 * pack_idx + 1],
                reciprocal_pair,
            )
        # rO_row_u32 and rO_row share the same memory
        cute.copy(
            output_store_atom, rO_row, gO_row_thread[(None, row_idx),], cache_policy=cache_policy
        )
        if is_writer_thread:
            rS_row[0] = Uint8(bf16_mx_scale_bytes_pair(reciprocal_pair)).bitcast(SCALE_DTYPE)
            cute.copy(
                row_scale_store_atom,
                rS_row,
                cute.local_tile(gS_row_thread, (1,), (row_idx,)),
                cache_policy=cache_policy,
            )


@cute.jit
def _gather_partial_colwise_amaxes(
    rX: cute.Tensor,
    # Constexprs
    CTA_THREADS_Y: cutlass.Constexpr[int],
    CTA_THREADS_X: cutlass.Constexpr[int],
    ELEMENTS_Y_PER_THREAD: cutlass.Constexpr[int],
    ELEMENTS_X_PER_THREAD: cutlass.Constexpr[int],
    DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
) -> cute.Tensor:
    """
    rX is a RMEM fragment of layout (ELEMENTS_X_PER_THREAD, ELEMENTS_Y_PER_THREAD) that contains this thread's data

    It returns a SMEM buffer of layout (ELEMENTS_X_PER_THREAD, (CTA_THREADS_Y, CTA_THREADS_X)), where each thread
    reduces their partial columnwise amax of shape (ELEMENTS_X_PER_THREAD,) and contributes it to the SMEM buffer
    """
    tidx, _, _ = cute.arch.thread_idx()

    # We first reduce each thread's rows into one maximum per column before reduce across different threads in the CTA
    rAmax_col = cute.make_rmem_tensor((ELEMENTS_X_PER_THREAD,), DTYPE)

    # Gather the amax for each column that this thread sweeps
    rX_i32 = cute.recast_tensor(rX, Int32)
    rAmax_col_i32 = cute.recast_tensor(rAmax_col, Int32)
    for bf16x2_idx in cutlass.range_constexpr(cute.size(rX_i32, mode=[0])):
        amax = rX_i32[bf16x2_idx, 0]
        for row_idx in cutlass.range_constexpr(1, ELEMENTS_Y_PER_THREAD):
            amax = abs_max_x2_bf16(amax, rX_i32[bf16x2_idx, row_idx])
        rAmax_col_i32[bf16x2_idx] = amax

    # Keep eight BF16 values (16 bytes) contiguous. XOR BF16 offset bit 6 into
    # bit 3 so each 128-bit warp transaction uses all 32 banks without overlap.
    scratch_swizzle = cute.make_swizzle(1, 3, 3)

    # Prepare amax reduction scratch space in SMEM.
    sAmaxs_col_scratch_layout = cute.make_composed_layout(
        scratch_swizzle,
        0,
        cute.zipped_product(
            rAmax_col.layout,  # Each thread contributes (ELEMENTS_X_PER_THREAD,)
            cute.make_layout((CTA_THREADS_Y, CTA_THREADS_X)),  # Total threads in the CTA
        ),
    )
    allocator = cutlass.memory.SmemAllocator()
    sAmaxs_col_scratch = allocator.allocate_tensor(
        DTYPE, sAmaxs_col_scratch_layout, byte_alignment=16
    )

    # Every thread writes its partial maxima using its flat thread ID.
    cute.autovec_copy(rAmax_col, sAmaxs_col_scratch[None, tidx])
    cute.arch.sync_threads()

    return sAmaxs_col_scratch


@cute.jit
def _reduce_partial_colwise_amaxes(
    sAmaxs_col_scratch: cute.Tensor,
    # Constexprs
    CTA_THREADS_Y: cutlass.Constexpr[int],
    CTA_THREADS_X: cutlass.Constexpr[int],
    ELEMENTS_Y_PER_THREAD: cutlass.Constexpr[int],
    ELEMENTS_X_PER_THREAD: cutlass.Constexpr[int],
    DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
    FP8_DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
) -> cute.Tensor:
    """
    sAmaxs_col_scratch: the SMEM buffer of layout (ELEMENTS_X_PER_THREAD, (CTA_THREADS_Y, CTA_THREADS_X))
        that contains each threads' own columnwise amax of its ELEMENTS_X_PER_THREAD columns

    This function reduces the partial amax buffer of layout (ELEMENTS_X_PER_THREAD, (CTA_THREADS_Y, CTA_THREADS_X))
    along the Y axis, to a single columnwise scale SMEM buffer of layout (ELEMENTS_X_PER_THREAD * CTA_THREADS_X),
    which is the return value
    """
    tidx, _, _ = cute.arch.thread_idx()

    TILE_COLS = CTA_THREADS_X * ELEMENTS_X_PER_THREAD

    # Each reducer owns one column pair and visits its partial maximum from
    # every warp. Column pairs precede warp rows in the scratch's logical order.
    REDUCTION_THREADS_NUM = CTA_THREADS_X * ELEMENTS_X_PER_THREAD // 2
    is_reduction_thread = tidx < REDUCTION_THREADS_NUM
    # How many bf16x2 pairs we need to reduce
    REDUCTION_i32_PAIRS = TILE_COLS // REDUCTION_THREADS_NUM // 2

    # Use the same 16-byte swizzle as the partial-max scratch. Consecutive
    # column-pair reducers still access distinct banks through this layout.
    scratch_swizzle = cute.make_swizzle(1, 3, 3)
    # SMEM buffer for colwise scale reduction where mode 0 is how many reciprocals a thread has,
    # and mode 1 is how many threads we have in X axis, and the layout size is just TILE_COLS
    sS_col_layout = cute.make_composed_layout(
        scratch_swizzle,
        0,
        cute.zipped_product(
            cute.make_layout((ELEMENTS_X_PER_THREAD,)),
            cute.make_layout((CTA_THREADS_X,)),
        ),
    )
    allocator = cutlass.memory.SmemAllocator()
    sS_col = allocator.allocate_tensor(DTYPE, sS_col_layout, byte_alignment=16)
    sS_col_i32 = cute.recast_tensor(sS_col, Int32)

    # Colwise amaxes layout: each reduction thread owns CTA_THREADS_Y rows, with REDUCTION_i32_PAIRS column pairs
    _, tv_col_amaxes_layout = cute.make_layout_tv(
        thr_layout=cute.make_layout((REDUCTION_THREADS_NUM, 1), stride=(1, REDUCTION_THREADS_NUM)),
        val_layout=cute.make_layout(
            (REDUCTION_i32_PAIRS, CTA_THREADS_Y), stride=(1, REDUCTION_i32_PAIRS)
        ),
    )
    # Colwise rcp layout: each reduction thread owns only 1 row, with REDUCTION_i32_PAIRS column pairs
    _, tv_col_rcp_layout = cute.make_layout_tv(
        thr_layout=cute.make_layout((REDUCTION_THREADS_NUM, 1), stride=(1, REDUCTION_THREADS_NUM)),
        val_layout=cute.make_layout((REDUCTION_i32_PAIRS, 1), stride=(1, REDUCTION_i32_PAIRS)),
    )

    sS_col_i32 = cute.composition(sS_col_i32, tv_col_rcp_layout)

    # Each reduction process `REDUCTION_i32_PAIRS` bf16x2 pairs by visiting all rows (CTA_THREADS_Y)
    # to extract the amax of that column
    sAmaxs_col_scratch_i32 = cute.recast_tensor(sAmaxs_col_scratch, Int32)
    MAX_EXPONENT = 8 if FP8_DTYPE is cutlass.Float8E4M3FN else 15
    if is_reduction_thread:
        sAmaxs_col_frag_i32 = cute.composition(sAmaxs_col_scratch_i32, tv_col_amaxes_layout)[
            tidx, None
        ]
        # Expose the thread's value mode as (partial rows, column pairs).
        sAmaxs_col_frag_i32 = cute.composition(
            sAmaxs_col_frag_i32,
            cute.make_layout((CTA_THREADS_Y, REDUCTION_i32_PAIRS), stride=(REDUCTION_i32_PAIRS, 1)),
        )

        # Iterative over all column pairs
        for pair_idx in cutlass.range_constexpr(REDUCTION_i32_PAIRS):
            # Select the pair column
            sAmaxs_col_frag_i32_rows = sAmaxs_col_frag_i32[None, pair_idx]

            # Reduce across all rows' partial amaxes to obtain the columnwise amax for this column pair
            amaxX2 = sAmaxs_col_frag_i32_rows[0]
            for scratch_row_idx in cutlass.range_constexpr(1, CTA_THREADS_Y):
                amaxX2 = abs_max_x2_bf16(amaxX2, sAmaxs_col_frag_i32_rows[scratch_row_idx])

            # Obtain the columnwise scale reciprocal for this column pair
            sS_col_i32[tidx, pair_idx] = bf16_mx_scale_reciprocal_pair(amaxX2, MAX_EXPONENT)
    cute.arch.sync_threads()

    return sS_col


@cute.jit
def _write_colwise_scale(
    rS_col_rcp: cute.Tensor,
    gS_col_thread: cute.Tensor,
    # Atoms
    col_scale_atom: cute.CopyAtom,
    # Miscellaneous
    cache_policy: Int64,
    # Constexprs
    CTA_THREADS_Y: cutlass.Constexpr[int],
    CTA_THREADS_X: cutlass.Constexpr[int],
    ELEMENTS_Y_PER_THREAD: cutlass.Constexpr[int],
    ELEMENTS_X_PER_THREAD: cutlass.Constexpr[int],
    DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
    FP8_DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
    SCALE_DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
):
    """
    rS_col_rcp: RMEM fragment of this thread's columnwise scale reciprocals with shape (ELEMENTS_X_PER_THREAD,)
    """
    tidx, _, _ = cute.arch.thread_idx()

    rS_col_rcp_u32 = cute.recast_tensor(rS_col_rcp, Uint32)

    # Only threads from CTA's the first row writes the scale fragments
    if tidx < CTA_THREADS_X:
        rS_col = cute.make_rmem_tensor((ELEMENTS_X_PER_THREAD,), SCALE_DTYPE)
        rS_col_u32 = cute.recast_tensor(rS_col, Uint32)
        # Get the colwise scale from the reciprocal
        for pack_idx in cutlass.range_constexpr(
            cute.size(rS_col_u32)
        ):  # 4 output bytes in one pack
            rS_col_u32[pack_idx] = bf16_mx_scale_bytes_pair(rS_col_rcp_u32[2 * pack_idx]) | (
                bf16_mx_scale_bytes_pair(rS_col_rcp_u32[2 * pack_idx + 1]) << Uint32(16)
            )
        # Use RMEM buffer to stash the columnwise scale and flush it to GMEM using vectorized store at once
        cute.copy(col_scale_atom, rS_col, gS_col_thread, cache_policy=cache_policy)


@cute.jit
def _quantize_colwise(
    rX: cute.Tensor,
    rS_col_rcp: cute.Tensor,
    gO_col_thread: cute.Tensor,
    # Atoms
    store_atom: cute.CopyAtom,
    # Miscellaneous
    cache_policy: Int64,
    scale_and_pack: cutlass.Constexpr[Callable[..., Uint32]],
    # Constexprs
    ELEMENTS_Y_PER_THREAD: cutlass.Constexpr[int],
    ELEMENTS_X_PER_THREAD: cutlass.Constexpr[int],
    FP8_DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
):
    """
    rX: RMEM fragment of this thread's rows with shape (ELEMENTS_X_PER_THREAD, ELEMENTS_Y_PER_THREAD)
    rS_col_rcp: RMEM fragment of this thread's columnwise scale reciprocals with shape (ELEMENTS_X_PER_THREAD,)
    gO_col_thread: GMEM fragment of this thread's columnwise quantized output with shape
        (ELEMENTS_X_PER_THREAD, ELEMENTS_Y_PER_THREAD)

    This function quantizes its fragments using the input and columnwise scales, and write to its output fragment
    """
    rS_col_rcp_u32 = cute.recast_tensor(rS_col_rcp, Uint32)

    rO_col = cute.make_rmem_tensor((ELEMENTS_X_PER_THREAD,), FP8_DTYPE)
    rO_col_u32 = cute.recast_tensor(rO_col, Uint32)

    # Each thread quantizes its own rows with the columnwise reciprocals and write its output to GMEM.
    for row_idx in cutlass.range_constexpr(ELEMENTS_Y_PER_THREAD):
        rX_row_i32 = cute.recast_tensor(rX[None, row_idx], Int32)
        for pack_idx in cutlass.range_constexpr(
            cute.size(rO_col_u32)
        ):  # 4 output bytes in one pack
            rO_col_u32[pack_idx] = scale_and_pack(
                rX_row_i32[2 * pack_idx],
                rS_col_rcp_u32[2 * pack_idx],
                rX_row_i32[2 * pack_idx + 1],
                rS_col_rcp_u32[2 * pack_idx + 1],
            )
        cute.copy(store_atom, rO_col, gO_col_thread[(None, row_idx),], cache_policy=cache_policy)


@cute.jit
def quantize_bidimensional_mxfp8_g2r(
    gX_tile: cute.Tensor,
    gO_row_tile: cute.Tensor,
    gS_row_tile: cute.Tensor,
    gO_col_tile: cute.Tensor,
    gS_col_tile: cute.Tensor,
    cfg: cutlass.Constexpr[MXFP8QuantizeConfig],
    CTA_THREADS_Y: cutlass.Constexpr[int],
    CTA_THREADS_X: cutlass.Constexpr[int],
    ELEMENTS_Y_PER_THREAD: cutlass.Constexpr[int],
    ELEMENTS_X_PER_THREAD: cutlass.Constexpr[int],
    cache_policy: Int64,
):
    """Quantize one (TILE_ROWS, TILE_COLS) GMEM tile in both directions.

    TILE_ROWS = CTA_THREADS_Y * ELEMENTS_Y_PER_THREAD = 32, so every columnwise
    MXFP8 block is contained in the tile. Each warp owns ELEMENTS_Y_PER_THREAD
    consecutive rows, and each lane owns ELEMENTS_X_PER_THREAD adjacent columns.
    Load the tile once, then reuse its registers for both quantization passes.
    """
    # Shape of the tile
    TILE_ROWS = CTA_THREADS_Y * ELEMENTS_Y_PER_THREAD

    # How many threads cooperate to process a single MXFP8 block (32 elements) for both quantized directions
    THREADS_X_PER_MX_BLOCK = MXFP8_BLOCK_SCALING_SIZE // ELEMENTS_X_PER_THREAD
    THREADS_Y_PER_MX_BLOCK = MXFP8_BLOCK_SCALING_SIZE // ELEMENTS_Y_PER_THREAD

    SCALE_DTYPE = cutlass.Float8E8M0FNU

    tidx, _, _ = cute.arch.thread_idx()

    thread_layout = cute.make_layout((CTA_THREADS_Y, CTA_THREADS_X), stride=(CTA_THREADS_X, 1))
    _, tv_layout = cute.make_layout_tv(
        thr_layout=thread_layout,
        val_layout=cute.make_layout(
            (ELEMENTS_Y_PER_THREAD, ELEMENTS_X_PER_THREAD), stride=(ELEMENTS_X_PER_THREAD, 1)
        ),
    )

    gX_thread = cute.composition(gX_tile, tv_layout)[tidx, None]
    gO_row_thread = cute.composition(gO_row_tile, tv_layout)[tidx, None]
    gO_col_thread = cute.composition(gO_col_tile, tv_layout)[tidx, None]

    # Group CTA threads by THREADS_X_PER_MX_BLOCK since they share the same rowwise scale for a MXFP8 block.
    # And assign stride 0 to that submode
    tv_row_scale = cute.make_layout(
        (
            (THREADS_X_PER_MX_BLOCK, (CTA_THREADS_X // THREADS_X_PER_MX_BLOCK, CTA_THREADS_Y)),
            ELEMENTS_Y_PER_THREAD,
        ),
        stride=(
            (0, (TILE_ROWS, ELEMENTS_Y_PER_THREAD)),
            1,
        ),
    )
    gS_row_thread = cute.composition(gS_row_tile, tv_row_scale)[tidx, None]

    # Group CTA threads by THREADS_Y_PER_MX_BLOCK since they share the same columnwise scale for a MXFP8 block.
    # And assign stride 0 to that submode
    tv_col_scale = cute.make_layout(
        (
            (CTA_THREADS_X, (THREADS_Y_PER_MX_BLOCK, CTA_THREADS_Y // THREADS_Y_PER_MX_BLOCK)),
            ELEMENTS_X_PER_THREAD,
        ),
        stride=(
            (ELEMENTS_X_PER_THREAD, (0, 1)),
            1,
        ),
    )
    # If CTA_THREADS_Y // THREADS_Y_PER_MX_BLOCK == 1, this will help compiler derive alignment
    # Otherwise coalesce will not do anything which is still fine
    tv_col_scale = cute.coalesce(tv_col_scale, target_profile=(1, 1))
    gS_col_thread = cute.composition(gS_col_tile, tv_col_scale)[tidx, None]

    # Input fragment contains all rows it handles; output fragment contains only one row and we write it back row by row
    rX = cute.make_rmem_tensor((ELEMENTS_X_PER_THREAD, ELEMENTS_Y_PER_THREAD), cfg.DTYPE)

    load_atom = cute.make_copy_atom(
        cute.nvgpu.CopyG2ROp(),
        cfg.DTYPE,
        num_bits_per_copy=ELEMENTS_X_PER_THREAD * cfg.DTYPE.width,
        invariant=True,
    )
    store_atom = cute.make_copy_atom(
        cute.nvgpu.CopyR2GOp(),
        cfg.FP8_DTYPE,
        num_bits_per_copy=ELEMENTS_X_PER_THREAD * cfg.FP8_DTYPE.width,
    )

    row_scale_atom = cute.make_copy_atom(cute.nvgpu.CopyR2GOp(), SCALE_DTYPE, num_bits_per_copy=8)
    col_scale_atom = cute.make_copy_atom(
        cute.nvgpu.CopyR2GOp(), SCALE_DTYPE, num_bits_per_copy=ELEMENTS_X_PER_THREAD * 8
    )

    scale_and_pack = mul_bf16x2x2_cvt_bf16x4_to_fp8x4(cfg.FP8_DTYPE)

    # Copy all rows a thread sweeps into the RMEM fragment at once
    for row_idx in cutlass.range_constexpr(ELEMENTS_Y_PER_THREAD):
        cute.copy(
            load_atom, gX_thread[(None, row_idx),], rX[None, row_idx], cache_policy=cache_policy
        )

    _quantize_rowwise(
        rX,
        gO_row_thread,
        gS_row_thread,
        output_store_atom=store_atom,
        row_scale_store_atom=row_scale_atom,
        cache_policy=cache_policy,
        scale_and_pack=scale_and_pack,
        THREADS_X_PER_MX_BLOCK=THREADS_X_PER_MX_BLOCK,
        ELEMENTS_Y_PER_THREAD=ELEMENTS_Y_PER_THREAD,
        ELEMENTS_X_PER_THREAD=ELEMENTS_X_PER_THREAD,
        SCALE_DTYPE=SCALE_DTYPE,
        FP8_DTYPE=cfg.FP8_DTYPE,
    )

    sAmaxs_col_scratch = _gather_partial_colwise_amaxes(
        rX,
        CTA_THREADS_Y=CTA_THREADS_Y,
        CTA_THREADS_X=CTA_THREADS_X,
        ELEMENTS_Y_PER_THREAD=ELEMENTS_Y_PER_THREAD,
        ELEMENTS_X_PER_THREAD=ELEMENTS_X_PER_THREAD,
        DTYPE=cfg.DTYPE,
    )

    sS_col = _reduce_partial_colwise_amaxes(
        sAmaxs_col_scratch,
        CTA_THREADS_Y=CTA_THREADS_Y,
        CTA_THREADS_X=CTA_THREADS_X,
        ELEMENTS_Y_PER_THREAD=ELEMENTS_Y_PER_THREAD,
        ELEMENTS_X_PER_THREAD=ELEMENTS_X_PER_THREAD,
        DTYPE=cfg.DTYPE,
        FP8_DTYPE=cfg.FP8_DTYPE,
    )

    # Copy the reduced columnwise scale reciprocals to RMEM.
    # Threads that quantize different rows share the same columnwise scale
    rS_col_rcp = cute.make_rmem_tensor((ELEMENTS_X_PER_THREAD,), cfg.DTYPE)
    cute.autovec_copy(sS_col[None, tidx % CTA_THREADS_X], rS_col_rcp)

    _write_colwise_scale(
        rS_col_rcp,
        gS_col_thread,
        col_scale_atom=col_scale_atom,
        cache_policy=cache_policy,
        CTA_THREADS_Y=CTA_THREADS_Y,
        CTA_THREADS_X=CTA_THREADS_X,
        ELEMENTS_Y_PER_THREAD=ELEMENTS_Y_PER_THREAD,
        ELEMENTS_X_PER_THREAD=ELEMENTS_X_PER_THREAD,
        DTYPE=cfg.DTYPE,
        FP8_DTYPE=cfg.FP8_DTYPE,
        SCALE_DTYPE=SCALE_DTYPE,
    )

    _quantize_colwise(
        rX,
        rS_col_rcp,
        gO_col_thread,
        store_atom=store_atom,
        cache_policy=cache_policy,
        scale_and_pack=scale_and_pack,
        ELEMENTS_Y_PER_THREAD=ELEMENTS_Y_PER_THREAD,
        ELEMENTS_X_PER_THREAD=ELEMENTS_X_PER_THREAD,
        FP8_DTYPE=cfg.FP8_DTYPE,
    )


class MXFP8QuantizeRegisterBidimensionalKernel(MXFP8QuantizeKernelBase):
    """Load one CTA tile into registers and quantize it along both axes.

    TILE_ROWS = _CTA_THREADS_Y * _ELEMENTS_Y_PER_THREAD = 32.
    TILE_COLS = _CTA_THREADS_X * ELEMENTS_X_PER_THREAD = 256 or 512.
    The 32-row tile contains complete columnwise MXFP8 blocks; shared memory
    combines the partial column maxima from the eight warps. Requires aligned
    BF16 input, M divisible by 32, N divisible by 256 and plain scale layouts.
    """

    _CTA_THREADS_Y = 8
    _CTA_THREADS_X = 32

    def __init__(self, cfg: MXFP8QuantizeConfig):
        self.cfg = cfg
        # One CTA only quantize one columnwise MXFP8 block at a time
        self._ELEMENTS_Y_PER_THREAD = MXFP8_BLOCK_SCALING_SIZE // self._CTA_THREADS_Y  # 4

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mO_row: Optional[cute.Tensor],
        mS_row: Optional[cute.Tensor],
        mO_col: Optional[cute.Tensor],
        mS_col: Optional[cute.Tensor],
        mAmax: Optional[cute.Tensor],
        mNoop: cute.Pointer,
        mDActInput: Optional[cute.Tensor],
        mWorkspace: Optional[cute.Tensor],
        stream: CUstream,
    ):
        M, N = mX.shape
        TILE_ROWS = self._CTA_THREADS_Y * self._ELEMENTS_Y_PER_THREAD
        WIDE_TILE_COLS = self._CTA_THREADS_X * 16
        if N % WIDE_TILE_COLS == 0:
            if Int64(M // TILE_ROWS) * (N // WIDE_TILE_COLS) >= 4096:
                self._launch(
                    mX,
                    mO_row,
                    mS_row,
                    mO_col,
                    mS_col,
                    mNoop,
                    stream,
                    ELEMENTS_X_PER_THREAD=16,
                    CLUSTER_WIDTH=4,
                )
            else:
                self._launch(
                    mX,
                    mO_row,
                    mS_row,
                    mO_col,
                    mS_col,
                    mNoop,
                    stream,
                    ELEMENTS_X_PER_THREAD=16,
                    CLUSTER_WIDTH=1,
                )
        else:
            self._launch(
                mX,
                mO_row,
                mS_row,
                mO_col,
                mS_col,
                mNoop,
                stream,
                ELEMENTS_X_PER_THREAD=8,
                CLUSTER_WIDTH=8,
            )

    @cute.jit
    def _launch(
        self,
        mX: cute.Tensor,
        mO_row: cute.Tensor,
        mS_row: cute.Tensor,
        mO_col: cute.Tensor,
        mS_col: cute.Tensor,
        mNoop: cute.Pointer,
        stream: CUstream,
        ELEMENTS_X_PER_THREAD: cutlass.Constexpr[int],
        CLUSTER_WIDTH: cutlass.Constexpr[int],
    ):
        assert ELEMENTS_X_PER_THREAD in (8, 16)

        # Dispatch checks these alignments before selecting this kernel.
        mX = cute.make_tensor(mX.iterator.align(32), mX.layout)
        mS_col = cute.make_tensor(mS_col.iterator.align(16), mS_col.layout)
        M, N = mX.shape
        TILE_ROWS = self._CTA_THREADS_Y * self._ELEMENTS_Y_PER_THREAD
        TILE_COLS = self._CTA_THREADS_X * ELEMENTS_X_PER_THREAD
        THREADS_PER_CTA = self._CTA_THREADS_Y * self._CTA_THREADS_X

        grid_cols = N // TILE_COLS
        grid_rows = M // TILE_ROWS
        cluster_cols = Int32(CLUSTER_WIDTH)
        # Adjust cluster size so that the CTAs in the column direction can be evenly divided into clusters.
        while cluster_cols > 1 and grid_cols % cluster_cols != 0:
            cluster_cols -= 1
        # Keep column CTAs on grid.x so clusters read adjacent column tiles.
        grid = [grid_cols, grid_rows]
        block = [THREADS_PER_CTA]
        cluster = [cluster_cols, 1, 1]

        # Hardcode the layout using static N for some of the commonly used shapes
        if N == 2048:
            payload_layout = cute.make_layout((M, 2048), stride=(2048, 1))
            gX = cute.make_tensor(mX.iterator, payload_layout)
            gO_row = cute.make_tensor(mO_row.iterator, payload_layout)
            gO_col = cute.make_tensor(mO_col.iterator, payload_layout)
            gS_row = cute.make_tensor(
                mS_row.iterator,
                cute.make_layout((cute.size(mS_row, mode=[0]), 64), stride=(64, 1)),
            )
            gS_col = cute.make_tensor(
                mS_col.iterator,
                cute.make_layout((cute.size(mS_col, mode=[0]), 2048), stride=(2048, 1)),
            )
            self.kernel(
                gX,
                gO_row,
                gS_row,
                gO_col,
                gS_col,
                mNoop,
                ELEMENTS_X_PER_THREAD=ELEMENTS_X_PER_THREAD,
            ).launch(grid=grid, block=block, cluster=cluster, min_blocks_per_mp=4, stream=stream)
        elif N == 4096:
            payload_layout = cute.make_layout((M, 4096), stride=(4096, 1))
            gX = cute.make_tensor(mX.iterator, payload_layout)
            gO_row = cute.make_tensor(mO_row.iterator, payload_layout)
            gO_col = cute.make_tensor(mO_col.iterator, payload_layout)
            gS_row = cute.make_tensor(
                mS_row.iterator,
                cute.make_layout((cute.size(mS_row, mode=[0]), 128), stride=(128, 1)),
            )
            gS_col = cute.make_tensor(
                mS_col.iterator,
                cute.make_layout((cute.size(mS_col, mode=[0]), 4096), stride=(4096, 1)),
            )
            self.kernel(
                gX,
                gO_row,
                gS_row,
                gO_col,
                gS_col,
                mNoop,
                ELEMENTS_X_PER_THREAD=ELEMENTS_X_PER_THREAD,
            ).launch(grid=grid, block=block, cluster=cluster, min_blocks_per_mp=4, stream=stream)
        else:
            self.kernel(
                mX,
                mO_row,
                mS_row,
                mO_col,
                mS_col,
                mNoop,
                ELEMENTS_X_PER_THREAD=ELEMENTS_X_PER_THREAD,
            ).launch(grid=grid, block=block, cluster=cluster, min_blocks_per_mp=4, stream=stream)

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mO_row: cute.Tensor,
        mS_row: cute.Tensor,
        mO_col: cute.Tensor,
        mS_col: cute.Tensor,
        mNoop: cute.Pointer,
        ELEMENTS_X_PER_THREAD: cutlass.Constexpr[int],
    ):
        """Skip the CTA when the noop flag is set; otherwise run the kernel body."""
        if not noop_flag_is_set(mNoop):
            self._kernel_main(
                mX,
                mO_row,
                mS_row,
                mO_col,
                mS_col,
                ELEMENTS_X_PER_THREAD=ELEMENTS_X_PER_THREAD,
            )

    @cute.jit
    def _kernel_main(
        self,
        mX: cute.Tensor,
        mO_row: cute.Tensor,
        mS_row: cute.Tensor,
        mO_col: cute.Tensor,
        mS_col: cute.Tensor,
        ELEMENTS_X_PER_THREAD: cutlass.Constexpr[int],
    ):
        """Construct CTA tile views and run bidimensional quantization."""
        cta_col, cta_row, _ = cute.arch.block_idx()
        cta_coord = (cta_row, cta_col)

        TILE_ROWS = self._CTA_THREADS_Y * self._ELEMENTS_Y_PER_THREAD
        TILE_COLS = self._CTA_THREADS_X * ELEMENTS_X_PER_THREAD

        TILER = (TILE_ROWS, TILE_COLS)
        ROW_SCALE_TILER = (TILE_ROWS, TILE_COLS // 32)
        COL_SCALE_TILER = (TILE_ROWS // 32, TILE_COLS)

        gX_tile = cute.local_tile(mX, TILER, cta_coord)
        gO_row_tile = cute.local_tile(mO_row, TILER, cta_coord)
        gO_col_tile = cute.local_tile(mO_col, TILER, cta_coord)
        gS_row_tile = cute.local_tile(mS_row, ROW_SCALE_TILER, cta_coord)
        gS_col_tile = cute.local_tile(mS_col, COL_SCALE_TILER, cta_coord)

        cache_policy = create_l2_policy(True)

        quantize_bidimensional_mxfp8_g2r(
            gX_tile,
            gO_row_tile,
            gS_row_tile,
            gO_col_tile,
            gS_col_tile,
            self.cfg,
            CTA_THREADS_Y=self._CTA_THREADS_Y,
            CTA_THREADS_X=self._CTA_THREADS_X,
            ELEMENTS_Y_PER_THREAD=self._ELEMENTS_Y_PER_THREAD,
            ELEMENTS_X_PER_THREAD=ELEMENTS_X_PER_THREAD,
            cache_policy=cache_policy,
        )
