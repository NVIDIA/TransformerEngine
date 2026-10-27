# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Register-resident CuTeDSL bidimensional MXFP8 cast, ported from cast_bidim.cuh."""

from typing import Callable, Optional, Type

import cutlass
from cutlass import cute
from cutlass.utils import HardwareInfo
from cutlass import Boolean, Float32, Int32, Int64, Uint16, Uint32, Uint8
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module

from transformer_engine.common.CuTeDSL.utils import (
    abs_max_x2_bf16,
    exp2f_rcp,
    fabs_f32,
    bf16_pair_magnitude,
    bf16_mx_scale_reciprocal,
    bf16_mx_scale_reciprocal_pair,
    bf16_mx_scale_bytes_pair,
    create_l2_policy,
    max_num_bits_per_copy_ld_global,
    x2_hi_to_f32_bf16,
    x2_lo_to_f32_bf16,
    cvt_f32x2_to_bf16x2,
)
from transformer_engine.common.CuTeDSL.activations import (
    DERIVATIVE_SLOPES,
    DERIVATIVE_TABLE_WINDOWS,
    FORWARD_TABLE_WINDOWS,
    activation_table_bytes,
    table_derivative_2x,
    table_forward_2x,
)
from transformer_engine.common.CuTeDSL.utils_fp8 import (
    cvt_f32_to_fp8e8m0fnu,
    get_cvt_f32x2_to_fp8x2_func,
    mul_bf16x2x2_cvt_bf16x4_to_fp8x4,
)
from .quantize_mxfp8_common import (
    MXFP8_BLOCK_SCALING_SIZE,
    MXFP8QuantizeConfig,
    MXFP8QuantizeKernelBase,
    activation_op,
    dactivation_op,
    derive_swizzled_scale_layout,
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
    EDGE: cutlass.Constexpr[bool] = False,
    rows_left: Int32 = 0,
):
    """With EDGE, only the thread's first rows_left rows are stored."""
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
        if cutlass.const_expr(EDGE):
            if row_idx < rows_left:
                cute.copy(
                    output_store_atom,
                    rO_row,
                    gO_row_thread[(None, row_idx),],
                    cache_policy=cache_policy,
                )
                if is_writer_thread:
                    rS_row[0] = Uint8(bf16_mx_scale_bytes_pair(reciprocal_pair)).bitcast(
                        SCALE_DTYPE
                    )
                    cute.copy(
                        row_scale_store_atom,
                        rS_row,
                        cute.local_tile(gS_row_thread, (1,), (row_idx,)),
                        cache_policy=cache_policy,
                    )
        else:
            cute.copy(
                output_store_atom,
                rO_row,
                gO_row_thread[(None, row_idx),],
                cache_policy=cache_policy,
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
    CTA_THREADS_X: cutlass.Constexpr[int],
    ELEMENTS_X_PER_THREAD: cutlass.Constexpr[int],
    SCALE_DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
    EDGE: cutlass.Constexpr[bool] = False,
    cols_ok: Boolean = True,
):
    """
    rS_col_rcp: RMEM fragment of this thread's columnwise scale reciprocals with shape (ELEMENTS_X_PER_THREAD,)
    With EDGE, a thread whose columns are past the tensor (not cols_ok) writes none.
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
        if cutlass.const_expr(EDGE):
            if cols_ok:
                cute.copy(col_scale_atom, rS_col, gS_col_thread, cache_policy=cache_policy)
        else:
            cute.copy(col_scale_atom, rS_col, gS_col_thread, cache_policy=cache_policy)


@cute.jit
def _apply_colwise_scales(
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
    EDGE: cutlass.Constexpr[bool] = False,
    rows_left: Int32 = 0,
):
    """
    With EDGE, only the thread's first rows_left rows are stored.

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
        if cutlass.const_expr(EDGE):
            if row_idx < rows_left:
                cute.copy(
                    store_atom, rO_col, gO_col_thread[(None, row_idx),], cache_policy=cache_policy
                )
        else:
            cute.copy(
                store_atom, rO_col, gO_col_thread[(None, row_idx),], cache_policy=cache_policy
            )


@cute.jit
def _quantize_colwise(
    rX: cute.Tensor,
    gO_col_thread: cute.Tensor,
    gS_col_thread: cute.Tensor,
    # Atoms
    output_store_atom: cute.CopyAtom,
    col_scale_store_atom: cute.CopyAtom,
    # Miscellaneous
    cache_policy: Int64,
    scale_and_pack: cutlass.Constexpr[Callable[..., Uint32]],
    # Constexprs
    CTA_THREADS_Y: cutlass.Constexpr[int],
    CTA_THREADS_X: cutlass.Constexpr[int],
    ELEMENTS_Y_PER_THREAD: cutlass.Constexpr[int],
    ELEMENTS_X_PER_THREAD: cutlass.Constexpr[int],
    SCALE_DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
    DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
    FP8_DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
    EDGE: cutlass.Constexpr[bool] = False,
    rows_left: Int32 = 0,
    cols_ok: Boolean = True,
):
    """Reduce columnwise amaxes across the CTA, write the columnwise scales, and quantize rX with them."""
    tidx, _, _ = cute.arch.thread_idx()

    sAmaxs_col_scratch = _gather_partial_colwise_amaxes(
        rX,
        CTA_THREADS_Y=CTA_THREADS_Y,
        CTA_THREADS_X=CTA_THREADS_X,
        ELEMENTS_Y_PER_THREAD=ELEMENTS_Y_PER_THREAD,
        ELEMENTS_X_PER_THREAD=ELEMENTS_X_PER_THREAD,
        DTYPE=DTYPE,
    )

    sS_col = _reduce_partial_colwise_amaxes(
        sAmaxs_col_scratch,
        CTA_THREADS_Y=CTA_THREADS_Y,
        CTA_THREADS_X=CTA_THREADS_X,
        ELEMENTS_X_PER_THREAD=ELEMENTS_X_PER_THREAD,
        DTYPE=DTYPE,
        FP8_DTYPE=FP8_DTYPE,
    )

    # Copy the reduced columnwise scale reciprocals to RMEM.
    # Threads that quantize different rows share the same columnwise scale
    rS_col_rcp = cute.make_rmem_tensor((ELEMENTS_X_PER_THREAD,), DTYPE)
    cute.autovec_copy(sS_col[None, tidx % CTA_THREADS_X], rS_col_rcp)

    _write_colwise_scale(
        rS_col_rcp,
        gS_col_thread,
        col_scale_atom=col_scale_store_atom,
        cache_policy=cache_policy,
        CTA_THREADS_X=CTA_THREADS_X,
        ELEMENTS_X_PER_THREAD=ELEMENTS_X_PER_THREAD,
        SCALE_DTYPE=SCALE_DTYPE,
        EDGE=EDGE,
        cols_ok=cols_ok,
    )

    _apply_colwise_scales(
        rX,
        rS_col_rcp,
        gO_col_thread,
        store_atom=output_store_atom,
        cache_policy=cache_policy,
        scale_and_pack=scale_and_pack,
        ELEMENTS_Y_PER_THREAD=ELEMENTS_Y_PER_THREAD,
        ELEMENTS_X_PER_THREAD=ELEMENTS_X_PER_THREAD,
        FP8_DTYPE=FP8_DTYPE,
        EDGE=EDGE,
        rows_left=rows_left,
    )


# FP16 and FP32 inputs scale in FP32, as the CUDA register-tiled kernel and the generic kernel do
# (2^(127-e) is often outside FP16's range): a block's scale is
# e8m0(max(amax * MAX_NORM_RCP, 0)), so a block of only NaNs gets the scale of 0 like the
# generic kernel's 0-seeded fmaxf, and each value is multiplied in FP32 by 2^(127-e) and then
# converted to FP8.


@cute.jit
def _scale_from_amax_f32(amax: Float32, MAX_NORM_RCP: cutlass.Constexpr[float]):
    """The E8M0 scale of a block with this amax, and its FP32 reciprocal."""
    biased = cvt_f32_to_fp8e8m0fnu(cute.arch.fmax(amax * Float32(MAX_NORM_RCP), Float32(0.0)))
    return biased, exp2f_rcp(biased)


@cute.jit
def _quantize_rowwise_f32_scaled(
    rX: cute.Tensor,
    gO_row_thread: cute.Tensor,
    gS_row_thread: cute.Tensor,
    output_store_atom: cute.CopyAtom,
    row_scale_store_atom: cute.CopyAtom,
    cache_policy: Int64,
    THREADS_X_PER_MX_BLOCK: cutlass.Constexpr[int],
    ELEMENTS_Y_PER_THREAD: cutlass.Constexpr[int],
    ELEMENTS_X_PER_THREAD: cutlass.Constexpr[int],
    SCALE_DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
    FP8_DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
    MAX_NORM_RCP: cutlass.Constexpr[float],
    EDGE: cutlass.Constexpr[bool] = False,
    rows_left: Int32 = 0,
):
    """_quantize_rowwise for FP16 and FP32 values, scaled in FP32."""
    tidx, _, _ = cute.arch.thread_idx()
    cvt_x2 = get_cvt_f32x2_to_fp8x2_func(FP8_DTYPE)
    rS_row = cute.make_rmem_tensor((1,), SCALE_DTYPE)
    rO_row = cute.make_rmem_tensor((ELEMENTS_X_PER_THREAD,), FP8_DTYPE)
    rO_row_u16 = cute.recast_tensor(rO_row, Uint16)
    is_writer_thread = tidx % THREADS_X_PER_MX_BLOCK == 0
    for row_idx in cutlass.range_constexpr(ELEMENTS_Y_PER_THREAD):
        amax = Float32(0.0)
        for col in cutlass.range_constexpr(ELEMENTS_X_PER_THREAD):
            amax = cute.arch.fmax(amax, fabs_f32(Float32(rX[col, row_idx])))
        # The block's amax across the lanes that share it.
        for stage in cutlass.range_constexpr(THREADS_X_PER_MX_BLOCK.bit_length() - 1):
            amax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, 1 << stage))
        biased, reciprocal = _scale_from_amax_f32(amax, MAX_NORM_RCP)
        for pair in cutlass.range_constexpr(ELEMENTS_X_PER_THREAD // 2):
            rO_row_u16[pair] = cvt_x2(
                Float32(rX[2 * pair + 1, row_idx]) * reciprocal,
                Float32(rX[2 * pair, row_idx]) * reciprocal,
            )
        row_ok = Boolean(True)
        if cutlass.const_expr(EDGE):
            row_ok = row_idx < rows_left
        if row_ok:
            cute.copy(
                output_store_atom,
                rO_row,
                gO_row_thread[(None, row_idx),],
                cache_policy=cache_policy,
            )
            if is_writer_thread:
                rS_row[0] = biased
                cute.copy(
                    row_scale_store_atom,
                    rS_row,
                    cute.local_tile(gS_row_thread, (1,), (row_idx,)),
                    cache_policy=cache_policy,
                )


@cute.jit
def _quantize_colwise_f32_scaled(
    rX: cute.Tensor,
    gO_col_thread: cute.Tensor,
    gS_col_thread: cute.Tensor,
    output_store_atom: cute.CopyAtom,
    col_scale_store_atom: cute.CopyAtom,
    cache_policy: Int64,
    CTA_THREADS_Y: cutlass.Constexpr[int],
    CTA_THREADS_X: cutlass.Constexpr[int],
    ELEMENTS_Y_PER_THREAD: cutlass.Constexpr[int],
    ELEMENTS_X_PER_THREAD: cutlass.Constexpr[int],
    SCALE_DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
    FP8_DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
    MAX_NORM_RCP: cutlass.Constexpr[float],
    EDGE: cutlass.Constexpr[bool] = False,
    rows_left: Int32 = 0,
    cols_ok: Boolean = True,
):
    """_quantize_colwise for FP16 and FP32 values, scaled in FP32. Thread x of each warp row
    holds columns [x * ELEMENTS_X_PER_THREAD, (x + 1) * ELEMENTS_X_PER_THREAD)."""
    tidx, _, _ = cute.arch.thread_idx()
    THREADS = CTA_THREADS_Y * CTA_THREADS_X
    TILE_COLS = CTA_THREADS_X * ELEMENTS_X_PER_THREAD
    cvt_x2 = get_cvt_f32x2_to_fp8x2_func(FP8_DTYPE)
    allocator = cutlass.memory.SmemAllocator()
    # Each thread's column maxima over its rows, contiguous per thread.
    sPartial = allocator.allocate_tensor(
        Float32,
        cute.make_layout((ELEMENTS_X_PER_THREAD, THREADS), stride=(1, ELEMENTS_X_PER_THREAD)),
        byte_alignment=16,
    )
    sScale = allocator.allocate_tensor(
        SCALE_DTYPE, cute.make_layout((TILE_COLS,)), byte_alignment=16
    )
    sReciprocal = allocator.allocate_tensor(
        Float32, cute.make_layout((TILE_COLS,)), byte_alignment=16
    )

    rPartial = cute.make_rmem_tensor((ELEMENTS_X_PER_THREAD,), Float32)
    for col in cutlass.range_constexpr(ELEMENTS_X_PER_THREAD):
        amax = Float32(0.0)
        for row_idx in cutlass.range_constexpr(ELEMENTS_Y_PER_THREAD):
            amax = cute.arch.fmax(amax, fabs_f32(Float32(rX[col, row_idx])))
        rPartial[col] = amax
    cute.autovec_copy(rPartial, sPartial[None, tidx])
    cute.arch.sync_threads()

    # One thread per column reduces the warps' maxima: column c is thread (c / X) of each warp
    # row, element c % X.
    if tidx < TILE_COLS:
        owner = tidx // ELEMENTS_X_PER_THREAD
        element = tidx % ELEMENTS_X_PER_THREAD
        amax = Float32(0.0)
        for warp_row in cutlass.range_constexpr(CTA_THREADS_Y):
            amax = cute.arch.fmax(amax, sPartial[element, warp_row * CTA_THREADS_X + owner])
        biased, reciprocal = _scale_from_amax_f32(amax, MAX_NORM_RCP)
        sScale[tidx] = biased
        sReciprocal[tidx] = reciprocal
    cute.arch.sync_threads()

    column0 = (tidx % CTA_THREADS_X) * ELEMENTS_X_PER_THREAD
    # The first warp row writes the scales.
    if tidx < CTA_THREADS_X:
        rS_col = cute.make_rmem_tensor((ELEMENTS_X_PER_THREAD,), SCALE_DTYPE)
        for col in cutlass.range_constexpr(ELEMENTS_X_PER_THREAD):
            rS_col[col] = sScale[column0 + col]
        if cols_ok:
            cute.copy(col_scale_store_atom, rS_col, gS_col_thread, cache_policy=cache_policy)

    rReciprocal = cute.make_rmem_tensor((ELEMENTS_X_PER_THREAD,), Float32)
    for col in cutlass.range_constexpr(ELEMENTS_X_PER_THREAD):
        rReciprocal[col] = sReciprocal[column0 + col]
    rO_col = cute.make_rmem_tensor((ELEMENTS_X_PER_THREAD,), FP8_DTYPE)
    rO_col_u16 = cute.recast_tensor(rO_col, Uint16)
    for row_idx in cutlass.range_constexpr(ELEMENTS_Y_PER_THREAD):
        for pair in cutlass.range_constexpr(ELEMENTS_X_PER_THREAD // 2):
            rO_col_u16[pair] = cvt_x2(
                Float32(rX[2 * pair + 1, row_idx]) * rReciprocal[2 * pair + 1],
                Float32(rX[2 * pair, row_idx]) * rReciprocal[2 * pair],
            )
        row_ok = Boolean(True)
        if cutlass.const_expr(EDGE):
            row_ok = row_idx < rows_left
        if row_ok:
            cute.copy(
                output_store_atom,
                rO_col,
                gO_col_thread[(None, row_idx),],
                cache_policy=cache_policy,
            )


@cute.jit
def quantize_mxfp8_gmem_tile_bidimensional(
    gX_tile: cute.Tensor,
    gA_tile: Optional[cute.Tensor],
    gO_row_tile: Optional[cute.Tensor],
    gS_row_tile: Optional[cute.Tensor],
    gO_col_tile: Optional[cute.Tensor],
    gS_col_tile: Optional[cute.Tensor],
    ROWWISE: cutlass.Constexpr[bool],
    COLWISE: cutlass.Constexpr[bool],
    ACTIVATION: cutlass.Constexpr[Optional[str]],
    ACTIVATION_TABLE: cutlass.Constexpr[str],
    sTable: Optional[cute.Tensor],
    sDbias: Optional[cute.Tensor],
    EDGE: cutlass.Constexpr[bool],
    rows_valid: Int32,
    cols_valid: Int32,
    WITH_AMAX: cutlass.Constexpr[bool],
    WITH_ACT: cutlass.Constexpr[bool],
    WITH_DACT: cutlass.Constexpr[bool],
    DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
    FP8_DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
    MAX_NORM_RCP: cutlass.Constexpr[float],
    WITH_GEMM_SWIZZLED_SCALES: cutlass.Constexpr[bool],
    CTA_THREADS_Y: cutlass.Constexpr[int],
    CTA_THREADS_X: cutlass.Constexpr[int],
    ELEMENTS_Y_PER_THREAD: cutlass.Constexpr[int],
    ELEMENTS_X_PER_THREAD: cutlass.Constexpr[int],
    cache_policy: Int64,
):
    """Quantize one (TILE_ROWS, TILE_COLS) GMEM tile in the directions enabled by ROWWISE / COLWISE.

    TILE_ROWS = CTA_THREADS_Y * ELEMENTS_Y_PER_THREAD = 32, so every columnwise
    MXFP8 block is contained in the tile. Each warp owns ELEMENTS_Y_PER_THREAD
    consecutive rows, and each lane owns ELEMENTS_X_PER_THREAD adjacent columns.
    Load the tile once, then reuse its registers for each enabled quantization pass.
    The tensors of a disabled direction are unused and may be None.
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

    # Input fragment contains all rows it handles; output fragment contains only one row and we write it back row by row
    rX = cute.make_rmem_tensor((ELEMENTS_X_PER_THREAD, ELEMENTS_Y_PER_THREAD), DTYPE)

    load_atom = cute.make_copy_atom(
        cute.nvgpu.CopyG2ROp(),
        DTYPE,
        num_bits_per_copy=min(
            ELEMENTS_X_PER_THREAD * DTYPE.width,
            max_num_bits_per_copy_ld_global(),
        ),
        # TODO(kainingz): due to https://github.com/NVIDIA/cutlass/issues/3726 using
        # invariant=True here will drop the cache policy
        invariant=False,
    )
    store_atom = cute.make_copy_atom(
        cute.nvgpu.CopyR2GOp(),
        FP8_DTYPE,
        num_bits_per_copy=ELEMENTS_X_PER_THREAD * FP8_DTYPE.width,
    )

    scale_and_pack = mul_bf16x2x2_cvt_bf16x4_to_fp8x4(FP8_DTYPE)

    # With EDGE the tile ends at rows_valid rows and cols_valid columns: the thread's rows from
    # rows_left on are past the tensor, and all of them when its columns are (a thread's columns
    # lie all inside or all past the tensor, as N % 16 == 0). They load as zeros, which leave
    # the column amaxes and dbias unchanged, and are not stored.
    rows_left = Int32(ELEMENTS_Y_PER_THREAD)
    cols_ok = Boolean(True)
    if cutlass.const_expr(EDGE):
        thread_origin = cute.composition(cute.make_identity_tensor(gX_tile.shape), tv_layout)[
            tidx, None
        ][((0, 0),)]
        cols_ok = thread_origin[1] < cols_valid
        rows_left = rows_valid - thread_origin[0]
        if not cols_ok:
            rows_left = Int32(0)

    # Copy all rows a thread sweeps into the RMEM fragment at once
    for row_idx in cutlass.range_constexpr(ELEMENTS_Y_PER_THREAD):
        if cutlass.const_expr(EDGE):
            # Predicated loads rather than branches, so the rows' loads still issue together.
            rX[None, row_idx].fill(0)
            row_pred = cute.make_rmem_tensor(
                cute.make_layout((ELEMENTS_X_PER_THREAD, (1,)), stride=(0, (0,))), Boolean
            )
            row_pred[0] = row_idx < rows_left
            cute.copy(
                load_atom,
                gX_thread[(None, row_idx),],
                rX[None, row_idx],
                pred=row_pred,
                cache_policy=cache_policy,
            )
        else:
            cute.copy(
                load_atom, gX_thread[(None, row_idx),], rX[None, row_idx], cache_policy=cache_policy
            )

    if cutlass.const_expr(WITH_ACT or WITH_DACT):
        rA = cute.make_rmem_tensor((ELEMENTS_X_PER_THREAD, ELEMENTS_Y_PER_THREAD), DTYPE)
        if cutlass.const_expr(WITH_DACT):
            act_load_atom = cute.make_copy_atom(
                cute.nvgpu.CopyG2ROp(),
                DTYPE,
                num_bits_per_copy=min(
                    ELEMENTS_X_PER_THREAD * DTYPE.width,
                    max_num_bits_per_copy_ld_global(),
                ),
                invariant=False,
            )
            # Activation tile has the same layout as the input tile so we can reuse the tv layout
            gA_thread = cute.composition(gA_tile, tv_layout)[tidx, None]
            for row_idx in cutlass.range_constexpr(ELEMENTS_Y_PER_THREAD):
                if cutlass.const_expr(EDGE):
                    rA[None, row_idx].fill(0)
                    act_row_pred = cute.make_rmem_tensor(
                        cute.make_layout((ELEMENTS_X_PER_THREAD, (1,)), stride=(0, (0,))), Boolean
                    )
                    act_row_pred[0] = row_idx < rows_left
                    cute.copy(
                        act_load_atom,
                        gA_thread[(None, row_idx),],
                        rA[None, row_idx],
                        pred=act_row_pred,
                        cache_policy=cache_policy,
                    )
                else:
                    cute.copy(
                        act_load_atom,
                        gA_thread[(None, row_idx),],
                        rA[None, row_idx],
                        cache_policy=cache_policy,
                    )
        # With dbias, the FP32 products are staged for the column sums (see _kernel_main).
        if cutlass.const_expr(sDbias is not None):
            sDbias_thread = cute.composition(sDbias, tv_layout)[tidx, None]
        # Apply activation row by row
        if cutlass.const_expr(DTYPE is not cutlass.BFloat16):
            # FP16 and FP32: per value in FP32, rounded to the input type, as the generic
            # kernel does; dbias takes the FP32 products before rounding.
            for row_idx in cutlass.range_constexpr(ELEMENTS_Y_PER_THREAD):
                if cutlass.const_expr(sDbias is not None):
                    rProducts = cute.make_rmem_tensor((ELEMENTS_X_PER_THREAD,), Float32)
                for col in cutlass.range_constexpr(ELEMENTS_X_PER_THREAD):
                    x = Float32(rX[col, row_idx])
                    if cutlass.const_expr(WITH_DACT):
                        dop = dactivation_op(ACTIVATION, ACTIVATION_TABLE)
                        x = x * dop(Float32(rA[col, row_idx]))
                        if cutlass.const_expr(sDbias is not None):
                            rProducts[col] = x
                    else:
                        x = activation_op(ACTIVATION, ACTIVATION_TABLE)(x)
                    rX[col, row_idx] = DTYPE(x)
                if cutlass.const_expr(sDbias is not None):
                    cute.autovec_copy(rProducts, sDbias_thread[(None, row_idx),])
        else:
            for row_idx in cutlass.range_constexpr(ELEMENTS_Y_PER_THREAD):
                rX_i32 = cute.recast_tensor(rX[None, row_idx], Int32)
                rA_i32 = cute.recast_tensor(rA[None, row_idx], Int32)
                if cutlass.const_expr(sDbias is not None):
                    rProducts = cute.make_rmem_tensor((ELEMENTS_X_PER_THREAD,), Float32)
                for i in cutlass.range_constexpr(ELEMENTS_X_PER_THREAD // 2):
                    if cutlass.const_expr(ACTIVATION_TABLE != "none"):
                        # Read the activation from the table, as the CUDA register-tiled kernel does.
                        if cutlass.const_expr(WITH_DACT):
                            low, high, end, stride = DERIVATIVE_TABLE_WINDOWS[
                                (ACTIVATION, ACTIVATION_TABLE)
                            ]
                            d_lo, d_hi = table_derivative_2x(
                                rA_i32[i],
                                cute.recast_tensor(sTable, Float32),
                                LO=low,
                                HI=high,
                                SATURATED_END=end,
                                STRIDE=stride,
                                SLOPE=DERIVATIVE_SLOPES[ACTIVATION],
                            )
                            lo = x2_lo_to_f32_bf16(rX_i32[i]) * d_lo
                            hi = x2_hi_to_f32_bf16(rX_i32[i]) * d_hi
                            if cutlass.const_expr(sDbias is not None):
                                rProducts[2 * i] = lo
                                rProducts[2 * i + 1] = hi
                            rX_i32[i] = cvt_f32x2_to_bf16x2(lo, hi)
                        else:
                            low, high = FORWARD_TABLE_WINDOWS[ACTIVATION]
                            rX_i32[i] = table_forward_2x(
                                rX_i32[i],
                                cute.recast_tensor(sTable, Uint16),
                                LOW_SENTINEL=low - 1,
                                HIGH_SENTINEL=high + 1,
                            )
                    else:
                        lo = x2_lo_to_f32_bf16(rX_i32[i])
                        hi = x2_hi_to_f32_bf16(rX_i32[i])
                        if cutlass.const_expr(WITH_DACT):
                            dop = dactivation_op(ACTIVATION, ACTIVATION_TABLE)
                            lo = lo * dop(x2_lo_to_f32_bf16(rA_i32[i]))
                            hi = hi * dop(x2_hi_to_f32_bf16(rA_i32[i]))
                            if cutlass.const_expr(sDbias is not None):
                                rProducts[2 * i] = lo
                                rProducts[2 * i + 1] = hi
                        else:
                            op = activation_op(ACTIVATION, ACTIVATION_TABLE)
                            lo = op(lo)
                            hi = op(hi)
                        rX_i32[i] = cvt_f32x2_to_bf16x2(lo, hi)
                if cutlass.const_expr(sDbias is not None):
                    cute.autovec_copy(rProducts, sDbias_thread[(None, row_idx),])

    # The thread's amax of the values it quantizes (the activation rounded to BF16). Like the
    # CUDA kernels' fmaxf, it ignores NaN: the BF16 max returns the other operand of a NaN, and
    # the FP32 fmax drops a pair of NaNs.
    thread_amax = Float32(0.0)
    if cutlass.const_expr(WITH_AMAX and DTYPE is not cutlass.BFloat16):
        for row_idx in cutlass.range_constexpr(ELEMENTS_Y_PER_THREAD):
            for col in cutlass.range_constexpr(ELEMENTS_X_PER_THREAD):
                thread_amax = cute.arch.fmax(thread_amax, fabs_f32(Float32(rX[col, row_idx])))
    elif cutlass.const_expr(WITH_AMAX):
        rX_all_i32 = cute.recast_tensor(rX, Int32)
        pair_amax = rX_all_i32[0]
        for word in cutlass.range_constexpr(1, cute.size(rX_all_i32)):
            pair_amax = abs_max_x2_bf16(pair_amax, rX_all_i32[word])
        thread_amax = cute.arch.fmax(
            fabs_f32(x2_lo_to_f32_bf16(pair_amax)), fabs_f32(x2_hi_to_f32_bf16(pair_amax))
        )

    if cutlass.const_expr(ROWWISE):
        gO_row_thread = cute.composition(gO_row_tile, tv_layout)[tidx, None]

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

        row_scale_atom = cute.make_copy_atom(
            cute.nvgpu.CopyR2GOp(), SCALE_DTYPE, num_bits_per_copy=8
        )

        if cutlass.const_expr(DTYPE is not cutlass.BFloat16):
            _quantize_rowwise_f32_scaled(
                rX,
                gO_row_thread,
                gS_row_thread,
                output_store_atom=store_atom,
                row_scale_store_atom=row_scale_atom,
                cache_policy=cache_policy,
                THREADS_X_PER_MX_BLOCK=THREADS_X_PER_MX_BLOCK,
                ELEMENTS_Y_PER_THREAD=ELEMENTS_Y_PER_THREAD,
                ELEMENTS_X_PER_THREAD=ELEMENTS_X_PER_THREAD,
                SCALE_DTYPE=SCALE_DTYPE,
                FP8_DTYPE=FP8_DTYPE,
                MAX_NORM_RCP=MAX_NORM_RCP,
                EDGE=EDGE,
                rows_left=rows_left,
            )
        else:
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
                FP8_DTYPE=FP8_DTYPE,
                EDGE=EDGE,
                rows_left=rows_left,
            )

    if cutlass.const_expr(COLWISE):
        gO_col_thread = cute.composition(gO_col_tile, tv_layout)[tidx, None]

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

        # GEMM-swizzled colwise scales are not contiguous (adjacent columns land 16 bytes apart),
        # so they are stored a byte at a time, as the general kernel does.
        col_scale_atom = cute.make_copy_atom(
            cute.nvgpu.CopyR2GOp(),
            SCALE_DTYPE,
            num_bits_per_copy=8 if WITH_GEMM_SWIZZLED_SCALES else ELEMENTS_X_PER_THREAD * 8,
        )

        if cutlass.const_expr(DTYPE is not cutlass.BFloat16):
            _quantize_colwise_f32_scaled(
                rX,
                gO_col_thread,
                gS_col_thread,
                output_store_atom=store_atom,
                col_scale_store_atom=col_scale_atom,
                cache_policy=cache_policy,
                CTA_THREADS_Y=CTA_THREADS_Y,
                CTA_THREADS_X=CTA_THREADS_X,
                ELEMENTS_Y_PER_THREAD=ELEMENTS_Y_PER_THREAD,
                ELEMENTS_X_PER_THREAD=ELEMENTS_X_PER_THREAD,
                SCALE_DTYPE=SCALE_DTYPE,
                FP8_DTYPE=FP8_DTYPE,
                MAX_NORM_RCP=MAX_NORM_RCP,
                EDGE=EDGE,
                rows_left=rows_left,
                cols_ok=cols_ok,
            )
        else:
            _quantize_colwise(
                rX,
                gO_col_thread,
                gS_col_thread,
                output_store_atom=store_atom,
                col_scale_store_atom=col_scale_atom,
                cache_policy=cache_policy,
                scale_and_pack=scale_and_pack,
                CTA_THREADS_Y=CTA_THREADS_Y,
                CTA_THREADS_X=CTA_THREADS_X,
                ELEMENTS_Y_PER_THREAD=ELEMENTS_Y_PER_THREAD,
                ELEMENTS_X_PER_THREAD=ELEMENTS_X_PER_THREAD,
                SCALE_DTYPE=SCALE_DTYPE,
                DTYPE=DTYPE,
                FP8_DTYPE=FP8_DTYPE,
                EDGE=EDGE,
                rows_left=rows_left,
                cols_ok=cols_ok,
            )

    return thread_amax


class MXFP8QuantizeBidimensionalG2RKernel(MXFP8QuantizeKernelBase):
    """Load one CTA tile into registers and quantize it along both axes.

    TILE_ROWS = _CTA_THREADS_Y * _ELEMENTS_Y_PER_THREAD = 32.
    TILE_COLS = _CTA_THREADS_X * ELEMENTS_X_PER_THREAD = 256 or 512.
    The 32-row tile contains complete columnwise MXFP8 blocks; shared memory
    combines the partial column maxima from the eight warps. Requires aligned
    BF16 input, M divisible by 32 and N divisible by 256; writes packed or GEMM-swizzled
    scales.
    """

    _CTA_THREADS_Y = 8
    _CTA_THREADS_X = 32
    # With a table, the grid holds about this many columns' worth of CTAs (16384 CTAs of 256
    # columns, as in the CUDA register-tiled kernel); each CTA walks the remaining row tiles.
    _WALK_TARGET_COLS = 16384 * 256

    @staticmethod
    def supports(cfg: MXFP8QuantizeConfig) -> bool:
        # BF16 packed math; FP16 and FP32 scaled in FP32, only with a fused activation.
        # dbias only fused with an activation derivative, like the CUDA register-tiled kernel;
        # amax only with a fused activation; plain casts with amax take the general kernel.
        return (
            (not cfg.WITH_AMAX or cfg.WITH_ACT or cfg.WITH_DACT)
            and (not cfg.WITH_DBIAS or cfg.WITH_DACT)
            and (
                cfg.DTYPE is cutlass.BFloat16
                or MXFP8QuantizeBidimensionalG2RKernel.takes_scaled_in_f32(cfg)
            )
            and cfg.COLWISE
        )

    @staticmethod
    def takes_scaled_in_f32(cfg: MXFP8QuantizeConfig) -> bool:
        """Whether an FP16 or FP32 request takes this kernel: the fused activations where it is
        faster than the general kernel. GB200, geometric mean speedup over TE main, general ->
        this kernel: FP16 GeLU 1.46 -> 1.62, SiLU 1.41 -> 1.69, dGeLU 1.56 -> 1.69, dSiLU
        1.28 -> 1.76, dbias_dSiLU 1.48 -> 1.74; FP32 SiLU 1.32 -> 1.54, dGeLU 1.38 -> 1.52,
        dSiLU 1.10 -> 1.85, dbias_dSiLU 1.18 -> 1.53. dbias_dGeLU (FP16 1.75 -> 1.50, FP32
        1.47 -> 1.25) and FP32 GeLU (1.20 -> 1.16) keep the general kernel."""
        if not (cfg.WITH_ACT or cfg.WITH_DACT):
            return False
        if cfg.WITH_DBIAS and cfg.ACTIVATION == "dgelu":
            return False
        return not (cfg.DTYPE is cutlass.Float32 and cfg.ACTIVATION == "gelu")

    @staticmethod
    @cute.jit
    def eligible_to_dispatch(
        cfg: cutlass.Constexpr[MXFP8QuantizeConfig],
        mX: cute.Tensor,
        mO_row: Optional[cute.Tensor],
        mS_row: Optional[cute.Tensor],
        mO_col: Optional[cute.Tensor],
        mS_col: Optional[cute.Tensor],
        mNoop: cute.Pointer,
        mDActInput: Optional[cute.Tensor],
    ) -> Boolean:
        M, N = mX.shape
        act_input_aligned = Boolean(True)
        if cutlass.const_expr(mDActInput is not None):
            act_input_aligned = mDActInput.iterator.toint() % 32 == 0
        # Whole 32-row by 256-column tiles (any shape with N % 16 == 0 for the requests in
        # takes_tails), grid.y within limits, and aligned input and colwise-scale pointers.
        shape_ok = Boolean(True)
        if cutlass.const_expr(MXFP8QuantizeBidimensionalG2RKernel.takes_tails(cfg)):
            shape_ok = N % 16 == 0
        else:
            shape_ok = M % 32 == 0 and N % 256 == 0
        return (
            shape_ok
            and (M + 31) // 32 <= 65535
            and mX.iterator.toint() % 32 == 0
            and act_input_aligned
            and mS_col.iterator.toint() % 16 == 0
            and mNoop.toint() == 0
        )

    @staticmethod
    def takes_tails(cfg: MXFP8QuantizeConfig) -> bool:
        """Whether this config takes row and column tails (any row count, N % 16 == 0): the
        derivatives and the requests with a table or amax, as the CUDA register-tiled kernel
        does. The others keep whole 32-row by 256-column tiles."""
        return (
            cfg.ACTIVATION_TABLE != "none"
            or cfg.WITH_DACT
            or cfg.WITH_AMAX
            or cfg.DTYPE is not cutlass.BFloat16
        )

    def __init__(self, cfg: MXFP8QuantizeConfig):
        self.cfg = cfg
        # FP32 values take twice the registers: twice the warps, each with half the rows, and
        # half the resident CTAs per SM, as in the CUDA register-tiled kernel.
        self._MIN_BLOCKS_PER_MP = 4
        if cfg.DTYPE is cutlass.Float32:
            self._CTA_THREADS_Y = 16
            self._MIN_BLOCKS_PER_MP = 2
        # One CTA only quantize one columnwise MXFP8 block at a time
        self._ELEMENTS_Y_PER_THREAD = MXFP8_BLOCK_SCALING_SIZE // self._CTA_THREADS_Y  # 4
        # SMs of the device compiled for, which set the walk length with a table or dbias.
        self._sm_count = 0
        if cfg.ACTIVATION_TABLE != "none" or cfg.WITH_DBIAS:
            from cuda.core import (
                Device,
            )  # pylint: disable=import-outside-toplevel,no-name-in-module

            self._sm_count = HardwareInfo(Device().device_id).get_device_multiprocessor_count()

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mO_row: Optional[cute.Tensor],
        mS_row: Optional[cute.Tensor],
        mO_col: Optional[cute.Tensor],
        mS_col: Optional[cute.Tensor],
        mAmax: Optional[cute.Tensor],
        mNoop: cute.Pointer,  # Unused: the dispatcher routes noop calls to the general kernel
        mDActInput: Optional[cute.Tensor],
        mWorkspace: Optional[cute.Tensor],
        mActTable: cute.Pointer,  # activation table; null without one
        stream: CUstream,
    ):
        M, N = mX.shape
        TILE_ROWS = self._CTA_THREADS_Y * self._ELEMENTS_Y_PER_THREAD
        WIDE_TILE_COLS = self._CTA_THREADS_X * 16
        # A derivative holds the gradient and the activation input in registers: 256-column tiles
        # keep it under the register budget (wider tiles spill), and clusters do not help it.
        # With dbias, the staged FP32 products of a 256-column tile take 32 KB of shared memory.
        # FP16 and FP32 values are scaled in FP32 and take the same tiles.
        if cutlass.const_expr(self.cfg.WITH_DACT or self.cfg.DTYPE is not cutlass.BFloat16):
            self._launch(
                mX,
                mDActInput,
                mO_row,
                mS_row,
                mO_col,
                mS_col,
                mActTable,
                mWorkspace,
                mAmax,
                stream,
                ELEMENTS_X_PER_THREAD=8,
                CLUSTER_WIDTH=1,
            )
        elif N % WIDE_TILE_COLS == 0:
            if Int64(M // TILE_ROWS) * (N // WIDE_TILE_COLS) >= 4096:
                self._launch(
                    mX,
                    mDActInput,
                    mO_row,
                    mS_row,
                    mO_col,
                    mS_col,
                    mActTable,
                    mWorkspace,
                    mAmax,
                    stream,
                    ELEMENTS_X_PER_THREAD=16,
                    CLUSTER_WIDTH=4,
                )
            else:
                self._launch(
                    mX,
                    mDActInput,
                    mO_row,
                    mS_row,
                    mO_col,
                    mS_col,
                    mActTable,
                    mWorkspace,
                    mAmax,
                    stream,
                    ELEMENTS_X_PER_THREAD=16,
                    CLUSTER_WIDTH=1,
                )
        else:
            self._launch(
                mX,
                mDActInput,
                mO_row,
                mS_row,
                mO_col,
                mS_col,
                mActTable,
                mWorkspace,
                mAmax,
                stream,
                ELEMENTS_X_PER_THREAD=8,
                CLUSTER_WIDTH=8,
            )

    @cute.jit
    def _launch(
        self,
        mX: cute.Tensor,
        mA: Optional[cute.Tensor],
        mO_row: cute.Tensor,
        mS_row: cute.Tensor,
        mO_col: cute.Tensor,
        mS_col: cute.Tensor,
        mActTable: cute.Pointer,
        mWorkspace: Optional[cute.Tensor],
        mAmax: Optional[cute.Tensor],
        stream: CUstream,
        ELEMENTS_X_PER_THREAD: cutlass.Constexpr[int],
        CLUSTER_WIDTH: cutlass.Constexpr[int],
    ):
        assert ELEMENTS_X_PER_THREAD in (8, 16)

        # Dispatch checks these alignments before selecting this kernel.
        mX = cute.make_tensor(mX.iterator.align(32), mX.layout)
        mS_col = cute.make_tensor(mS_col.iterator.align(16), mS_col.layout)
        if cutlass.const_expr(self.cfg.WITH_DACT):
            mA = cute.make_tensor(mA.iterator.align(32), mA.layout)
        M, N = mX.shape
        TILE_ROWS = self._CTA_THREADS_Y * self._ELEMENTS_Y_PER_THREAD
        TILE_COLS = self._CTA_THREADS_X * ELEMENTS_X_PER_THREAD
        THREADS_PER_CTA = self._CTA_THREADS_Y * self._CTA_THREADS_X

        grid_cols = N // TILE_COLS
        grid_rows = M // TILE_ROWS
        if cutlass.const_expr(self.takes_tails(self.cfg)):
            grid_cols = (N + TILE_COLS - 1) // TILE_COLS
            grid_rows = (M + TILE_ROWS - 1) // TILE_ROWS
        cluster_cols = Int32(CLUSTER_WIDTH)
        # Adjust cluster size so that the CTAs in the column direction can be evenly divided into clusters.
        while cluster_cols > 1 and grid_cols % cluster_cols != 0:
            cluster_cols -= 1
        # With a table, each CTA walks several row tiles so that it copies the table once for
        # all of them, as the CUDA register-tiled kernel does: halve the grid while it holds
        # more than _WALK_TARGET_COLS / TILE_COLS CTAs and the row tiles divide evenly, and
        # walk at least two tiles unless that leaves fewer than two CTAs per SM.
        walk = Int32(1)
        if cutlass.const_expr(self.cfg.ACTIVATION_TABLE != "none" or self.cfg.WITH_DBIAS):
            while (grid_rows // walk) % 2 == 0 and grid_cols * (grid_rows // walk) > (
                self._WALK_TARGET_COLS // TILE_COLS
            ):
                walk *= 2
            if (
                walk == 1
                and grid_rows % 2 == 0
                and grid_cols * (grid_rows // 2) >= 2 * self._sm_count
            ):
                walk = Int32(2)
        # With dbias the grid and the walk count 64-row chunks, whose two row tiles a CTA takes
        # back to back: a CTA walks at least one chunk.
        if cutlass.const_expr(self.cfg.WITH_DBIAS):
            grid_rows = (grid_rows + 1) // 2  # the last chunk may hold one tile
            if walk > 1:
                walk = walk // 2
        # Keep column CTAs on grid.x so clusters read adjacent column tiles.
        grid = [grid_cols, grid_rows // walk]
        block = [THREADS_PER_CTA]
        cluster = [cluster_cols, 1, 1]

        # Hardcode the layout using static N for some of the commonly used shapes
        if N == 2048:
            payload_layout = cute.make_layout((M, 2048), stride=(2048, 1))
            gX = cute.make_tensor(mX.iterator, payload_layout)
            gA = (
                cute.make_tensor(mA.iterator, payload_layout)
                if cutlass.const_expr(self.cfg.WITH_DACT)
                else None
            )
            gO_row = (
                cute.make_tensor(mO_row.iterator, payload_layout)
                if cutlass.const_expr(self.cfg.ROWWISE)
                else None
            )
            gO_col = cute.make_tensor(mO_col.iterator, payload_layout)
            if cutlass.const_expr(self.cfg.WITH_GEMM_SWIZZLED_SCALES):
                gS_row, gS_col = derive_swizzled_scale_layout(
                    M, 2048, self.cfg.ROWWISE, True, mS_row, mS_col
                )
            else:
                gS_row = (
                    cute.make_tensor(
                        mS_row.iterator,
                        cute.make_layout((cute.size(mS_row, mode=[0]), 64), stride=(64, 1)),
                    )
                    if cutlass.const_expr(self.cfg.ROWWISE)
                    else None
                )
                gS_col = cute.make_tensor(
                    mS_col.iterator,
                    cute.make_layout((cute.size(mS_col, mode=[0]), 2048), stride=(2048, 1)),
                )
            # A tensor with a row or column tail takes the kernel whose tiles are predicated.
            if cutlass.const_expr(self.takes_tails(self.cfg)) and (
                M % TILE_ROWS != 0 or N % TILE_COLS != 0
            ):
                self.kernel(
                    gX,
                    gA,
                    gO_row,
                    gS_row,
                    gO_col,
                    gS_col,
                    mActTable,
                    walk,
                    mWorkspace,
                    mAmax,
                    ELEMENTS_X_PER_THREAD=ELEMENTS_X_PER_THREAD,
                    TAILS=True,
                ).launch(
                    grid=grid,
                    block=block,
                    cluster=cluster,
                    min_blocks_per_mp=self._MIN_BLOCKS_PER_MP,
                    stream=stream,
                )
            else:
                self.kernel(
                    gX,
                    gA,
                    gO_row,
                    gS_row,
                    gO_col,
                    gS_col,
                    mActTable,
                    walk,
                    mWorkspace,
                    mAmax,
                    ELEMENTS_X_PER_THREAD=ELEMENTS_X_PER_THREAD,
                    TAILS=False,
                ).launch(
                    grid=grid,
                    block=block,
                    cluster=cluster,
                    min_blocks_per_mp=self._MIN_BLOCKS_PER_MP,
                    stream=stream,
                )
        elif N == 4096:
            payload_layout = cute.make_layout((M, 4096), stride=(4096, 1))
            gX = cute.make_tensor(mX.iterator, payload_layout)
            gA = (
                cute.make_tensor(mA.iterator, payload_layout)
                if cutlass.const_expr(self.cfg.WITH_DACT)
                else None
            )
            gO_row = (
                cute.make_tensor(mO_row.iterator, payload_layout)
                if cutlass.const_expr(self.cfg.ROWWISE)
                else None
            )
            gO_col = cute.make_tensor(mO_col.iterator, payload_layout)
            if cutlass.const_expr(self.cfg.WITH_GEMM_SWIZZLED_SCALES):
                gS_row, gS_col = derive_swizzled_scale_layout(
                    M, 4096, self.cfg.ROWWISE, True, mS_row, mS_col
                )
            else:
                gS_row = (
                    cute.make_tensor(
                        mS_row.iterator,
                        cute.make_layout((cute.size(mS_row, mode=[0]), 128), stride=(128, 1)),
                    )
                    if cutlass.const_expr(self.cfg.ROWWISE)
                    else None
                )
                gS_col = cute.make_tensor(
                    mS_col.iterator,
                    cute.make_layout((cute.size(mS_col, mode=[0]), 4096), stride=(4096, 1)),
                )
            # A tensor with a row or column tail takes the kernel whose tiles are predicated.
            if cutlass.const_expr(self.takes_tails(self.cfg)) and (
                M % TILE_ROWS != 0 or N % TILE_COLS != 0
            ):
                self.kernel(
                    gX,
                    gA,
                    gO_row,
                    gS_row,
                    gO_col,
                    gS_col,
                    mActTable,
                    walk,
                    mWorkspace,
                    mAmax,
                    ELEMENTS_X_PER_THREAD=ELEMENTS_X_PER_THREAD,
                    TAILS=True,
                ).launch(
                    grid=grid,
                    block=block,
                    cluster=cluster,
                    min_blocks_per_mp=self._MIN_BLOCKS_PER_MP,
                    stream=stream,
                )
            else:
                self.kernel(
                    gX,
                    gA,
                    gO_row,
                    gS_row,
                    gO_col,
                    gS_col,
                    mActTable,
                    walk,
                    mWorkspace,
                    mAmax,
                    ELEMENTS_X_PER_THREAD=ELEMENTS_X_PER_THREAD,
                    TAILS=False,
                ).launch(
                    grid=grid,
                    block=block,
                    cluster=cluster,
                    min_blocks_per_mp=self._MIN_BLOCKS_PER_MP,
                    stream=stream,
                )
        else:
            if cutlass.const_expr(self.cfg.WITH_GEMM_SWIZZLED_SCALES):
                gS_row, gS_col = derive_swizzled_scale_layout(
                    M, N, self.cfg.ROWWISE, True, mS_row, mS_col
                )
            else:
                gS_row, gS_col = mS_row, mS_col
            # A tensor with a row or column tail takes the kernel whose tiles are predicated.
            if cutlass.const_expr(self.takes_tails(self.cfg)) and (
                M % TILE_ROWS != 0 or N % TILE_COLS != 0
            ):
                self.kernel(
                    mX,
                    mA,
                    mO_row,
                    gS_row,
                    mO_col,
                    gS_col,
                    mActTable,
                    walk,
                    mWorkspace,
                    mAmax,
                    ELEMENTS_X_PER_THREAD=ELEMENTS_X_PER_THREAD,
                    TAILS=True,
                ).launch(
                    grid=grid,
                    block=block,
                    cluster=cluster,
                    min_blocks_per_mp=self._MIN_BLOCKS_PER_MP,
                    stream=stream,
                )
            else:
                self.kernel(
                    mX,
                    mA,
                    mO_row,
                    gS_row,
                    mO_col,
                    gS_col,
                    mActTable,
                    walk,
                    mWorkspace,
                    mAmax,
                    ELEMENTS_X_PER_THREAD=ELEMENTS_X_PER_THREAD,
                    TAILS=False,
                ).launch(
                    grid=grid,
                    block=block,
                    cluster=cluster,
                    min_blocks_per_mp=self._MIN_BLOCKS_PER_MP,
                    stream=stream,
                )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mA: Optional[cute.Tensor],
        mO_row: cute.Tensor,
        mS_row: cute.Tensor,
        mO_col: cute.Tensor,
        mS_col: cute.Tensor,
        mActTable: cute.Pointer,
        walk: Int32,
        mWorkspace: Optional[cute.Tensor],
        mAmax: Optional[cute.Tensor],
        ELEMENTS_X_PER_THREAD: cutlass.Constexpr[int],
        TAILS: cutlass.Constexpr[bool] = False,
    ):
        """Run the kernel body. Like the CUDA specialized cast kernels it does not consume the
        cast_noop flag: the dispatcher routes calls with one to the general kernel."""
        self._kernel_main(
            mX,
            mA,
            mO_row,
            mS_row,
            mO_col,
            mS_col,
            mActTable,
            walk,
            mWorkspace,
            mAmax,
            ELEMENTS_X_PER_THREAD=ELEMENTS_X_PER_THREAD,
            TAILS=TAILS,
        )

    @cute.jit
    def _data_tile(
        self,
        mT: cute.Tensor,
        row_tile: Int32,
        cta_col: Int32,
        TILER: cutlass.Constexpr,
        DYNAMIC_STRIDE: cutlass.Constexpr[bool],
        ALIGNMENT: cutlass.Constexpr[int],
    ) -> cute.Tensor:
        """The (row_tile, cta_col) tile of mT. With a run-time row stride, the tile's base takes the
        64-bit product once and offsets within the tile are 32-bit, as in the CUDA register-tiled
        kernel: the full 64-bit layout costs registers and instructions on every access."""
        if cutlass.const_expr(DYNAMIC_STRIDE):
            N = mT.shape[1]
            base = Int64(row_tile) * Int64(TILER[0]) * Int64(N) + Int64(cta_col) * Int64(TILER[1])
            return cute.make_tensor(
                (mT.iterator + base).align(ALIGNMENT),
                cute.make_layout(TILER, stride=(Int32(N), 1)),
            )
        return cute.local_tile(mT, TILER, (row_tile, cta_col))

    @cute.jit
    def _kernel_main(
        self,
        mX: cute.Tensor,
        mA: Optional[cute.Tensor],
        mO_row: cute.Tensor,
        mS_row: cute.Tensor,
        mO_col: cute.Tensor,
        mS_col: cute.Tensor,
        mActTable: cute.Pointer,
        walk: Int32,
        mWorkspace: Optional[cute.Tensor],
        mAmax: Optional[cute.Tensor],
        ELEMENTS_X_PER_THREAD: cutlass.Constexpr[int],
        TAILS: cutlass.Constexpr[bool] = False,
    ):
        """Run bidimensional quantization on the CTA's row tiles: blockIdx.y, and with a table
        every gridDim.y-th row tile after it, walk tiles in all."""
        cta_col, first_cta_row, _ = cute.arch.block_idx()
        _, grid_rows, _ = cute.arch.grid_dim()

        cache_policy = create_l2_policy(True)

        # With a table, copy it into shared memory once for all of the CTA's row tiles.
        sTable = None
        if cutlass.const_expr(self.cfg.ACTIVATION_TABLE != "none"):
            TABLE_WORDS = (
                activation_table_bytes(self.cfg.ACTIVATION, self.cfg.ACTIVATION_TABLE) // 4
            )
            tidx, _, _ = cute.arch.thread_idx()
            allocator = cutlass.memory.SmemAllocator()
            sTable = allocator.allocate_tensor(
                Uint32, cute.make_layout((TABLE_WORDS,)), byte_alignment=16
            )
            # 16-byte pieces: TE allocates the table 16-byte aligned and padded.
            piece_layout = cute.make_layout((4, TABLE_WORDS // 4), stride=(1, 4))
            gTable = cute.make_tensor(
                cute.recast_ptr(mActTable, dtype=Uint32).align(16), piece_layout
            )
            sTable_pieces = cute.make_tensor(sTable.iterator, piece_layout)
            for piece in cutlass.range(
                tidx, TABLE_WORDS // 4, self._CTA_THREADS_Y * self._CTA_THREADS_X, unroll=1
            ):
                cute.autovec_copy(gTable[None, piece], sTable_pieces[None, piece])
            cute.arch.sync_threads()

        thread_amax = Float32(0.0)
        if cutlass.const_expr(self.cfg.WITH_DBIAS):
            # dbias in the CUDA kernels' order: each 64-row chunk's FP32 products summed down the
            # column in row order, one workspace row per chunk (reduced by the caller). The CTA
            # takes the chunk's two row tiles back to back; each stages its products in shared
            # memory, and one thread per column adds them to the column's running sum.
            TILE_ROWS = self._CTA_THREADS_Y * self._ELEMENTS_Y_PER_THREAD
            TILE_COLS = self._CTA_THREADS_X * ELEMENTS_X_PER_THREAD
            tidx, _, _ = cute.arch.thread_idx()
            sDbias = cutlass.memory.SmemAllocator().allocate_tensor(
                Float32,
                cute.make_layout((TILE_ROWS, TILE_COLS), stride=(TILE_COLS, 1)),
                byte_alignment=16,
            )
            for step in cutlass.range(walk, unroll=1):
                chunk = step * grid_rows + first_cta_row
                column_sum = Float32(0.0)
                # A run-time loop, so the tile body (and its shared memory) is emitted once.
                for half in cutlass.range(2, unroll=1):
                    # With a row tail, the last chunk of an odd row-tile count has no second
                    # tile; its rows would add zeros, which leave a sum that starts at +0
                    # unchanged.
                    tile_in_tensor = Boolean(True)
                    if cutlass.const_expr(TAILS):
                        tile_in_tensor = (2 * chunk + half) * TILE_ROWS < mX.shape[0]
                    if tile_in_tensor:
                        tile_amax = self._quantize_row_tile(
                            mX,
                            mA,
                            mO_row,
                            mS_row,
                            mO_col,
                            mS_col,
                            sTable,
                            2 * chunk + half,
                            cta_col,
                            cache_policy,
                            sDbias=sDbias,
                            ELEMENTS_X_PER_THREAD=ELEMENTS_X_PER_THREAD,
                            TAILS=TAILS,
                        )
                        thread_amax = cute.arch.fmax(thread_amax, tile_amax)
                        cute.arch.sync_threads()
                        if tidx < TILE_COLS:
                            for row in cutlass.range_constexpr(TILE_ROWS):
                                column_sum = column_sum + sDbias[row, tidx]
                        # The next tile overwrites the staged products.
                        cute.arch.sync_threads()
                if tidx < TILE_COLS and cta_col * TILE_COLS + tidx < mX.shape[1]:
                    mWorkspace[chunk, cta_col * TILE_COLS + tidx] = column_sum
        elif cutlass.const_expr(self.cfg.ACTIVATION_TABLE == "none"):
            tile_amax = self._quantize_row_tile(
                mX,
                mA,
                mO_row,
                mS_row,
                mO_col,
                mS_col,
                sTable,
                first_cta_row,
                cta_col,
                cache_policy,
                ELEMENTS_X_PER_THREAD=ELEMENTS_X_PER_THREAD,
                TAILS=TAILS,
            )
            thread_amax = cute.arch.fmax(thread_amax, tile_amax)
        else:
            for step in cutlass.range(walk, unroll=1):
                tile_amax = self._quantize_row_tile(
                    mX,
                    mA,
                    mO_row,
                    mS_row,
                    mO_col,
                    mS_col,
                    sTable,
                    step * grid_rows + first_cta_row,
                    cta_col,
                    cache_policy,
                    ELEMENTS_X_PER_THREAD=ELEMENTS_X_PER_THREAD,
                    TAILS=TAILS,
                )
                thread_amax = cute.arch.fmax(thread_amax, tile_amax)

        if cutlass.const_expr(self.cfg.WITH_AMAX):
            # The CTA's amax, then an atomic max on the float's bits (amaxes are not negative),
            # as the general kernel does.
            tidx, _, _ = cute.arch.thread_idx()
            warp_amax = cute.arch.warp_redux_sync(thread_amax, kind="fmax")
            NUM_WARPS = self._CTA_THREADS_Y * self._CTA_THREADS_X // 32
            sAmax = cutlass.memory.SmemAllocator().allocate_tensor(
                Float32, cute.make_layout((NUM_WARPS,)), byte_alignment=4
            )
            if tidx % 32 == 0:
                sAmax[tidx // 32] = warp_amax
            cute.arch.sync_threads()
            if tidx == 0:
                cta_amax = Float32(0.0)
                for w in cutlass.range_constexpr(NUM_WARPS):
                    cta_amax = cute.arch.fmax(cta_amax, sAmax[w])
                cute.arch.atomic_max(
                    cute.recast_ptr(mAmax.iterator, dtype=Int32), cta_amax.bitcast(Int32)
                )

    @cute.jit
    def _quantize_row_tile(
        self,
        mX: cute.Tensor,
        mA: Optional[cute.Tensor],
        mO_row: cute.Tensor,
        mS_row: cute.Tensor,
        mO_col: cute.Tensor,
        mS_col: cute.Tensor,
        sTable: Optional[cute.Tensor],
        row_tile: Int32,
        cta_col: Int32,
        cache_policy: Int64,
        ELEMENTS_X_PER_THREAD: cutlass.Constexpr[int],
        sDbias: Optional[cute.Tensor] = None,
        TAILS: cutlass.Constexpr[bool] = False,
    ) -> Float32:
        """Quantize the (row_tile, cta_col) tile; return the thread's amax of it (0 without
        WITH_AMAX)."""
        TILE_ROWS = self._CTA_THREADS_Y * self._ELEMENTS_Y_PER_THREAD
        TILE_COLS = self._CTA_THREADS_X * ELEMENTS_X_PER_THREAD
        TILER = (TILE_ROWS, TILE_COLS)
        ROW_SCALE_TILER = (TILE_ROWS, TILE_COLS // 32)
        COL_SCALE_TILER = (TILE_ROWS // 32, TILE_COLS)
        # With a table or a derivative and a row stride known only at run time, index each data
        # tile from its own base address with 32-bit offsets (see _data_tile).
        DYNAMIC_STRIDE = (
            self.cfg.ACTIVATION_TABLE != "none"
            or self.cfg.WITH_DACT
            or self.cfg.DTYPE is not cutlass.BFloat16
        ) and not cute.is_static(mX.layout.stride[0])
        cta_coord = (row_tile, cta_col)
        gX_tile = self._data_tile(mX, row_tile, cta_col, TILER, DYNAMIC_STRIDE, 32)
        gA_tile = None
        if cutlass.const_expr(self.cfg.WITH_DACT):
            gA_tile = self._data_tile(mA, row_tile, cta_col, TILER, DYNAMIC_STRIDE, 32)
        gO_row_tile = None
        gS_row_tile = None
        if cutlass.const_expr(self.cfg.ROWWISE):
            gO_row_tile = self._data_tile(mO_row, row_tile, cta_col, TILER, DYNAMIC_STRIDE, 16)
            gS_row_tile = cute.local_tile(mS_row, ROW_SCALE_TILER, cta_coord)
        gO_col_tile = self._data_tile(mO_col, row_tile, cta_col, TILER, DYNAMIC_STRIDE, 16)
        gS_col_tile = cute.local_tile(mS_col, COL_SCALE_TILER, cta_coord)

        # With TAILS (a tensor with a row or column tail) every tile is predicated: rows and
        # columns past the tensor are zero-filled and not stored. A second, unpredicated variant for the whole tiles in
        # the same kernel would double the shared memory of the tile body.
        return quantize_mxfp8_gmem_tile_bidimensional(
            gX_tile,
            gA_tile,
            gO_row_tile,
            gS_row_tile,
            gO_col_tile,
            gS_col_tile,
            ROWWISE=self.cfg.ROWWISE,
            COLWISE=self.cfg.COLWISE,
            ACTIVATION=self.cfg.ACTIVATION,
            ACTIVATION_TABLE=self.cfg.ACTIVATION_TABLE,
            sTable=sTable,
            sDbias=sDbias,
            EDGE=TAILS,
            rows_valid=mX.shape[0] - row_tile * TILE_ROWS,
            cols_valid=mX.shape[1] - cta_col * TILE_COLS,
            WITH_AMAX=self.cfg.WITH_AMAX,
            WITH_ACT=self.cfg.WITH_ACT,
            WITH_DACT=self.cfg.WITH_DACT,
            DTYPE=self.cfg.DTYPE,
            FP8_DTYPE=self.cfg.FP8_DTYPE,
            MAX_NORM_RCP=self.cfg.MAX_NORM_RCP,
            WITH_GEMM_SWIZZLED_SCALES=self.cfg.WITH_GEMM_SWIZZLED_SCALES,
            CTA_THREADS_Y=self._CTA_THREADS_Y,
            CTA_THREADS_X=self._CTA_THREADS_X,
            ELEMENTS_Y_PER_THREAD=self._ELEMENTS_Y_PER_THREAD,
            ELEMENTS_X_PER_THREAD=ELEMENTS_X_PER_THREAD,
            cache_policy=cache_policy,
        )
