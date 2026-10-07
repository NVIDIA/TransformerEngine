# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MXFP8 quantization kernel implemented in CuTeDSL.

Replicates the core logic of quantize_mxfp8.cuh: given a 2D tensor of BF16/FP16
values, quantize to MXFP8 format (FP8E4M3 data + E8M0 per-block scales).

BF16 cast-only calls with aligned input use the register-resident ports of
specialized/cast_rowwise.cuh and specialized/cast_bidim.cuh. Other configurations
use the one-lane rowwise, staged bidimensional or general kernels. All paths share the
TVM-FFI entrypoint.

Matches the C++ kernel's tile dimensions and thread layout:
  CHUNK_DIM_Y = 64, CHUNK_DIM_X = 64, THREADS_PER_CTA = 64
  BUFF_DIM_Y  = 32, BUFF_DIM_X  = 64, STAGES = 2
  MXFP8_BLOCK_SCALING_SIZE   = 32 (elements per MXFP8 scaling block)

Grid: (ceil(N / 64), ceil(M / 64))
Each block processes a 64x64 chunk in 2 stages of 32x64 tiles loaded into
shared memory.
"""

# Local @cute.struct classes are SMEM-layout descriptors that need no docstrings.
# pylint: disable=missing-class-docstring

from typing import Optional, Type

import cutlass
from cutlass import cute
from cutlass import pipeline
from cutlass import Float32, Int32, Int64, Uint16, Uint32, Uint8
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module

from transformer_engine.common.CuTeDSL.utils import (
    is_packed16,
    abs_max_x2_bf16,
    abs_max_x2_f16,
    max_x2_bf16,
    max_x2_f16,
    max_scalar_bf16,
    max_scalar_f16,
    abs_max_scalar_bf16,
    abs_max_scalar_f16,
    to_f32_bf16,
    to_f32_f16,
    x2_lo_to_f32_bf16,
    x2_lo_to_f32_f16,
    x2_hi_to_f32_bf16,
    x2_hi_to_f32_f16,
    truncate_f32_bf16,
    truncate_f32_f16,
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
    CUTEDSL_DEBUG_LOGGING,
    MXFP8QuantizeConfig,
    MXFP8QuantizeKernelBase,
    MXFP8_BLOCK_SCALING_SIZE,
    SUPPORTED_ACTIVATIONS,
    SUPPORTED_DACTIVATIONS,
    SYM_N_DIVISIBILITY,
    THREADS_PER_WARP,
    derive_swizzled_scale_layout,
    noop_flag_is_set,
)


@cute.jit
def quantize_rowwise_mxfp8(
    sX_tile: cute.Tensor,  # (TILE_Y, TILE_X) bf16/fp16 smem view, post-TMA
    sActInput_tile: Optional[cute.Tensor],  # (TILE_Y, TILE_X) act-input smem tile (dact only)
    sO_row_tile: cute.Tensor,  # (TILE_Y, TILE_X) fp8 smem view (rowwise FP8 output)
    mS_row_stage: cute.Tensor,  # rowwise scale tensor (1D swizzled, or 2D linear)
    MAX_NORM_RCP: cutlass.Constexpr[float],
    tile_row_start: Int32,  # global row index of this stage's row 0
    tile_col_start: Int32,  # global col index of this CTA's col 0
    M: Int32,
    N: Int32,
    ACTIVATION: cutlass.Constexpr[str | None],
    DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
    FP8_DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
    TILE_X: cutlass.Constexpr[int],
    TILE_Y: cutlass.Constexpr[int],
    WAVES: cutlass.Constexpr[int],
    THREADS_PER_BANK: cutlass.Constexpr[int],
    PACK_SIZE: cutlass.Constexpr[int],
    SKIP_INPUT_MASKING: cutlass.Constexpr[bool] = False,
    SKIP_SCALE_BOUNDS: cutlass.Constexpr[bool] = False,
    WITH_ACT: cutlass.Constexpr[bool] = False,
    WITH_DACT: cutlass.Constexpr[bool] = False,
    WITH_DBIAS: cutlass.Constexpr[bool] = False,
    dbias_acc: Optional[cute.Tensor] = None,  #  only needed when WITH_DBIAS is True
):
    """Quantize one SMEM tile rowwise to MXFP8 (per-row 32-elt block scales); returns the tile amax."""
    tidx, _, _ = cute.arch.thread_idx()

    CTA_THREADS_Y = TILE_Y  # threads per column (rows per tile)
    CTA_THREADS_X = TILE_X // MXFP8_BLOCK_SCALING_SIZE  # threads per row (chunks per row)

    _, tv_layout = cute.make_layout_tv(
        thr_layout=cute.make_layout((CTA_THREADS_Y, CTA_THREADS_X), stride=(CTA_THREADS_X, 1)),
        val_layout=cute.make_layout((1, MXFP8_BLOCK_SCALING_SIZE), stride=(0, 1)),
    )

    sX_tv = cute.composition(sX_tile, tv_layout)
    sO_tv = cute.composition(sO_row_tile, tv_layout)

    # I/O Elements that belong to this thread
    sX_thread = sX_tv[tidx, None]  # shape (32,) bf16
    sO_thread = sO_tv[tidx, None]  # shape (32,) fp8

    sO_thread_u32_ptr = cute.recast_ptr(sO_thread.iterator, dtype=Uint32)
    sO_thread_u32 = cute.make_tensor(
        sO_thread_u32_ptr,
        cute.make_layout(
            (MXFP8_BLOCK_SCALING_SIZE // 4,), stride=(1,)
        ),  # 1 uint32 is 4 fp8 elements
    )

    # PTX allows to fuse relu activation in `cvt.rn.satfinite` unless we need to reduction for dbias
    FUSE_RELU = cutlass.const_expr(ACTIVATION == "relu") and not WITH_DBIAS
    # For this fast path we can read in pack of 2 instead of reading individual f16 / bf16 element,
    # and keep the elements in half precision instead of upcasting every one of them to f32.
    USE_HALF_PRECISION = is_packed16(DTYPE) and (ACTIVATION is None or FUSE_RELU)

    amax_r = Float32(0.0)

    # Each thread start reading from the specfic bank based on its thread ID so they can do their best to access different banks
    # to avoid bank conflict.
    bank_group = (tidx % THREADS_PER_WARP) // THREADS_PER_BANK
    # The offset this thread should start reading from based on what's its first bank to access.
    offset = bank_group * PACK_SIZE
    if cutlass.const_expr(USE_HALF_PRECISION):
        # If no activation, f16 / bf16 and rowwise quantization, we can read 2 f16 / bf16 at once in a pack
        # and use max.xorsign.abs.f16x2 / max.xorsign.abs.bf16x2 to compute
        max_x2 = max_x2_f16 if DTYPE is cutlass.Float16 else max_x2_bf16
        abs_max_x2 = abs_max_x2_f16 if DTYPE is cutlass.Float16 else abs_max_x2_bf16
        x2_lo_to_f32 = x2_lo_to_f32_f16 if DTYPE is cutlass.Float16 else x2_lo_to_f32_bf16
        x2_hi_to_f32 = x2_hi_to_f32_f16 if DTYPE is cutlass.Float16 else x2_hi_to_f32_bf16
        sX_thread_rw_i64 = cute.make_tensor(
            cute.recast_ptr(sX_thread.iterator, dtype=Int64),
            cute.make_layout(
                (1, MXFP8_BLOCK_SCALING_SIZE // 4), stride=(0, 1)
            ),  # 1 int64 is 4 fp16/bf16 elements
        )
        # Each wave reads its 4 elements (PACK_SIZE) as one 8-byte vectorized load
        in_r = [[None, None] for _ in range(WAVES)]
        for w in cutlass.range_constexpr(WAVES):
            idx = (w + offset // 4) % (MXFP8_BLOCK_SCALING_SIZE // 4)
            in_r[w][0], in_r[w][1] = unpack_i64_to_i32x2(sX_thread_rw_i64[0, idx])

        if cutlass.const_expr(WITH_DBIAS):
            for w in cutlass.range_constexpr(WAVES):
                dbias_acc[w * PACK_SIZE + 0] += x2_lo_to_f32(in_r[w][0])
                dbias_acc[w * PACK_SIZE + 1] += x2_hi_to_f32(in_r[w][0])
                dbias_acc[w * PACK_SIZE + 2] += x2_lo_to_f32(in_r[w][1])
                dbias_acc[w * PACK_SIZE + 3] += x2_hi_to_f32(in_r[w][1])

        amax_2x = in_r[0][0]
        # Each wave will use max.xorsign.abs.f16x2 or max.xorsign.abs.bf16x2 to compare 2 packed elements in parallel
        for w in cutlass.range_constexpr(WAVES):
            if cutlass.const_expr(FUSE_RELU):
                # Skip the first iteration because we assigned in_r[0][0] to amax_2x already
                if cutlass.const_expr(w > 0):
                    # If we fuse relu then we don't want to do abs since negative value will be set to 0 and they will lose comparison automatically
                    amax_2x = max_x2(amax_2x, in_r[w][0])
                amax_2x = max_x2(amax_2x, in_r[w][1])
            else:
                # Skip the first iteration because we assigned in_r[0][0] to amax_2x already
                if cutlass.const_expr(w > 0):
                    amax_2x = abs_max_x2(amax_2x, in_r[w][0])
                amax_2x = abs_max_x2(amax_2x, in_r[w][1])
        if cutlass.const_expr(FUSE_RELU):
            # Compare the 2 packed max without abs
            amax_r = cute.arch.fmax(
                x2_lo_to_f32(amax_2x),
                x2_hi_to_f32(amax_2x),
            )
            # For relu the max is at least 0
            amax_r = cute.arch.fmax(amax_r, Float32(0.0))
        else:
            # Compare the 2 packed abs max
            amax_r = cute.arch.fmax(
                fabs_f32(x2_lo_to_f32(amax_2x)),
                fabs_f32(x2_hi_to_f32(amax_2x)),
            )
    else:
        # Since we need to do computation on individual f16 / bf16 elements, we can't read in pack
        sX_thread_rw = cute.make_tensor(
            sX_thread.iterator,
            cute.make_layout((1, MXFP8_BLOCK_SCALING_SIZE), stride=(0, 1)),
        )

        if cutlass.const_expr(WITH_DACT):
            # Backward: out = grad · act'(act_input). sX is grad, sA is act_input.
            dop = SUPPORTED_DACTIVATIONS[ACTIVATION]
            sA_thread = cute.composition(sActInput_tile, tv_layout)[tidx, None]
            sA_thread_rw = cute.make_tensor(
                sA_thread.iterator,
                cute.make_layout((1, MXFP8_BLOCK_SCALING_SIZE), stride=(0, 1)),
            )
        elif cutlass.const_expr(WITH_ACT):
            op = SUPPORTED_ACTIVATIONS[ACTIVATION]

        if cutlass.const_expr(is_packed16(DTYPE) and ACTIVATION is not None):
            truncate_f32 = truncate_f32_f16 if DTYPE is cutlass.Float16 else truncate_f32_bf16

        # Each wave we read PACK_SIZE elements, and we have WAVES waves, so we read WAVES * PACK_SIZE (= MXFP8_BLOCK_SCALING_SIZE) elements in total.
        in_r = [[None] * PACK_SIZE for _ in range(WAVES)]
        # Don't repeat these OOB masking computation in the loop
        thread_row_start = tile_row_start + tidx // CTA_THREADS_X
        thread_col_start = tile_col_start + (tidx % CTA_THREADS_X) * MXFP8_BLOCK_SCALING_SIZE
        thread_row_oob = thread_row_start >= M
        # PACK_SIZE must divide every term of `thread_col_start + start` and N itself, or a
        # pack could straddle the N boundary and the check below would be wrong. TILE_X is a
        # tunable, so it is asserted rather than assumed.
        assert MXFP8_BLOCK_SCALING_SIZE % PACK_SIZE == 0
        assert SYM_N_DIVISIBILITY % PACK_SIZE == 0
        assert TILE_X % PACK_SIZE == 0
        for w in cutlass.range_constexpr(WAVES):
            start = (w * PACK_SIZE + offset) % MXFP8_BLOCK_SCALING_SIZE
            # thread_col_start + start >= N means "if the first element in this pack is OOB"
            # If the first element is OOB, then (obviously) elements after it within this pack must be OOB
            # If the first element is not OOB, then elements after it within this pack must be not OOB either.
            # Proof:
            # (1) when the first element is not OOB, we have thread_col_start + start < N,
            #     which implies N - (thread_col_start + start) > 0
            # (2) with the assertions above:
            #     - N is divisible by SYM_N_DIVISIBILITY, so N is also divisible by PACK_SIZE
            #     - thread_col_start = tile_col_start + (tidx % CTA_THREADS_X) * MXFP8_BLOCK_SCALING_SIZE is divisible by PACK_SIZE
            #       because TILE_X and MXFP8_BLOCK_SCALING_SIZE are both divisible by PACK_SIZE
            #     - start = (w * PACK_SIZE + bank_group * PACK_SIZE) % MXFP8_BLOCK_SCALING_SIZE is also divisible by PACK_SIZE
            #       because MXFP8_BLOCK_SCALING_SIZE is a multiple of PACK_SIZE
            #     so N - (thread_col_start + start) must be divisible by PACK_SIZE
            # Therefore, N - (thread_col_start + start) >= PACK_SIZE because PACK_SIZE is the minimal number
            # for (1) and (2) to hold, which implies thread_col_start + start + PACK_SIZE -1 < N
            # Hence, we only need to check if the first element is OOB to determine if the whole pack is OOB
            thread_col_oob = thread_col_start + start >= N
            for i in cutlass.range_constexpr(PACK_SIZE):
                x = Float32(sX_thread_rw[0, start + i])
                if cutlass.const_expr(WITH_DACT):
                    # out = grad · act'(act_input)
                    x = x * dop(Float32(sA_thread_rw[0, start + i]))
                # If IS_ACT, apply activation function to x in f32
                elif cutlass.const_expr(WITH_ACT):
                    # If it's relu, we can handle it later
                    if not cutlass.const_expr(FUSE_RELU):
                        x = op(x)
                    if not cutlass.const_expr(SKIP_INPUT_MASKING):
                        # If the input shape is not divisible by the tile size,
                        # TMA would zero-fills the input tile outside its logical MxN bounds.
                        # This is fine for non-activation cases, but for activation cases,
                        # op(0) might not be 0 which will pollute the amax and dbias.
                        # So we must manually mask the OOB region here.
                        if thread_row_oob or thread_col_oob:
                            x = Float32(0.0)
                # Accumulate to the per-thread dbias register buffer for this tile if WITH_DBIAS
                if cutlass.const_expr(WITH_DBIAS):
                    # dbias_acc is register buffer so we can just write without bank conflict
                    dbias_acc[w * PACK_SIZE + i] += x
                # If 16-bit input with activation, truncate to IType
                if cutlass.const_expr(is_packed16(DTYPE) and ACTIVATION is not None):
                    x = truncate_f32(x)
                in_r[w][i] = x
                if cutlass.const_expr(FUSE_RELU):
                    amax_r = cute.arch.fmax(
                        amax_r, x
                    )  # For relu cases, we don't need abs since negative values will be 0 so they lose comparison automatically
                else:
                    amax_r = cute.arch.fmax(amax_r, fabs_f32(x))
        if cutlass.const_expr(FUSE_RELU):
            amax_r = cute.arch.fmax(amax_r, Float32(0.0))  # If relu, the amax is at least 0

    biased_exp_r = cvt_f32_to_fp8e8m0fnu(amax_r * MAX_NORM_RCP)

    # mS_row_stage has logical shape (32, 2) and we have 64 threads where each is mapped to one scale factor
    # The TV layout is equivalent to TV layout with thr_layout=(32, 2):(2, 1), val_layout=(1,)
    # but it's too trival so let's just index it directly without using layout
    # Note this is the logical layout, which is on top of the swizzled / non-swizzled scale factor layout
    # that mappes the logical index to the physical offset

    # For irregular shapes, skip the scale store if this thread's logical row / col-block lies past the input's actual extents.
    # TMA already zero-fills OOB input reads and drops OOB output writes; only the direct scale-byte gmem store needs an explicit guard.
    # If the input shape is divisible by the tile size, then we won't access OOB regions because
    # we never we only access tiles we actually need (num_tiles) which are never OOB
    if cutlass.const_expr(SKIP_SCALE_BOUNDS):
        mS_row_stage[(tidx // CTA_THREADS_X, tidx % CTA_THREADS_X)] = biased_exp_r
    else:
        scale_row = tile_row_start + tidx // CTA_THREADS_X
        scale_col_first_elt = tile_col_start + (tidx % CTA_THREADS_X) * MXFP8_BLOCK_SCALING_SIZE
        if scale_row < M and scale_col_first_elt < N:
            mS_row_stage[(tidx // CTA_THREADS_X, tidx % CTA_THREADS_X)] = biased_exp_r

    inv_scale_r = exp2f_rcp(biased_exp_r)  # f32 reciprocal of the scale
    scale_2x = pack_f32x2(inv_scale_r, inv_scale_r)
    if cutlass.const_expr(USE_HALF_PRECISION):
        mul_cvt_x4_func = mul_f32x2_cvt_packed16x4_to_fp8x4(DTYPE, FP8_DTYPE, FUSE_RELU)
    else:
        mul_cvt_x4_func = mul_f32x2_cvt_f32x4_to_fp8x4(FP8_DTYPE, FUSE_RELU)

    for w in cutlass.range_constexpr(WAVES):
        idx = (w * 4 + offset) % MXFP8_BLOCK_SCALING_SIZE
        idx = idx // 4
        if cutlass.const_expr(USE_HALF_PRECISION):
            # Convert 2 packed f16/bf16 pairs to 4 fp8 in one fused op
            sO_thread_u32[idx] = mul_cvt_x4_func(in_r[w][0], in_r[w][1], scale_2x)
        else:
            # Convert 4 f32 to 4 fp8 in one fused op
            sO_thread_u32[idx] = mul_cvt_x4_func(
                in_r[w][0], in_r[w][1], in_r[w][2], in_r[w][3], scale_2x
            )

    return amax_r


@cute.jit
def quantize_colwise_mxfp8(
    sX_tile: cute.Tensor,  # (TILE_Y, TILE_X) bf16/fp16 smem view, post-TMA
    sActInput_tile: Optional[cute.Tensor],  # (TILE_Y, TILE_X) act-input smem tile (dact only)
    sO_col_tile: cute.Tensor,  # (TILE_Y, TILE_X) fp8 smem view (colwise FP8 output)
    mS_col_stage: cute.Tensor,  # colwise scale tensor (1D swizzled, or 2D linear)
    MAX_NORM_RCP: cutlass.Constexpr[float],
    tile_row_start: Int32,  # global row index of this stage's row 0
    tile_col_start: Int32,  # global col index of this CTA's col 0
    M: Int32,
    N: Int32,
    ACTIVATION: cutlass.Constexpr[str | None],
    DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
    FP8_DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
    SWIZZLE: cutlass.Constexpr[bool],
    TILE_X: cutlass.Constexpr[int],
    TILE_Y: cutlass.Constexpr[int],  # pylint: disable=unused-argument  # kept for API consistency
    SKIP_INPUT_MASKING: cutlass.Constexpr[bool] = False,
    SKIP_SCALE_BOUNDS: cutlass.Constexpr[bool] = False,
    WITH_ACT: cutlass.Constexpr[bool] = False,
    WITH_DACT: cutlass.Constexpr[bool] = False,
    WITH_DBIAS: cutlass.Constexpr[bool] = False,
    CACHE_ACTIVATION: cutlass.Constexpr[bool] = False,  # cache post-activation values to sX_tile
):
    """Quantize one SMEM tile colwise to MXFP8 (per-column 32-elt block scales); returns (amax, dbias_partial)."""
    tidx, _, _ = cute.arch.thread_idx()

    _, tv_layout = cute.make_layout_tv(
        thr_layout=cute.make_layout((1, TILE_X), stride=(TILE_X, 1)),
        val_layout=cute.make_layout((MXFP8_BLOCK_SCALING_SIZE, 1), stride=(1, 1)),
    )

    sX_tv = cute.composition(sX_tile, tv_layout)
    sO_tv = cute.composition(sO_col_tile, tv_layout)

    # I/O Elements that belong to this thread
    sX_thread = sX_tv[tidx, None]
    sO_thread = sO_tv[tidx, None]

    # PTX allows to fuse relu activation in `cvt.rn.satfinite` unless we need to reduction for dbias
    FUSE_RELU = cutlass.const_expr(ACTIVATION == "relu") and not WITH_DBIAS
    # Keep input in half precision format if possible
    USE_HALF_PRECISION = is_packed16(DTYPE) and (ACTIVATION is None or FUSE_RELU)
    dbias_partial = Float32(0.0)

    if cutlass.const_expr(USE_HALF_PRECISION):
        max_scalar = max_scalar_f16 if DTYPE is cutlass.Float16 else max_scalar_bf16
        abs_max_scalar = abs_max_scalar_f16 if DTYPE is cutlass.Float16 else abs_max_scalar_bf16
        to_f32 = to_f32_f16 if DTYPE is cutlass.Float16 else to_f32_bf16
        # If we can use the half precision format, then use the input tile directly since there is no need to upcast
        sX_thread_packed16 = cute.make_tensor(
            sX_thread.iterator,
            cute.make_layout((MXFP8_BLOCK_SCALING_SIZE,), stride=(TILE_X,)),
        )
        # Stash the strided column reads in registers (CUDA's in_colwise_IType):
        # the cvt loop below reuses them instead of re-reading smem.
        in_c = [None] * MXFP8_BLOCK_SCALING_SIZE
        for i in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE):
            in_c[i] = sX_thread_packed16[i]

        amax_value = in_c[0]
        # Skip the first iteration because we assigned in_c[0] to amax_value already
        for i in cutlass.range_constexpr(1, MXFP8_BLOCK_SCALING_SIZE):
            if cutlass.const_expr(FUSE_RELU):
                # If we fuse relu then we don't want to do abs since negative value will be set to 0
                # and they will lose comparison automatically
                amax_value = max_scalar(amax_value, in_c[i])
            else:
                amax_value = abs_max_scalar(amax_value, in_c[i])
        if cutlass.const_expr(FUSE_RELU):
            amax_c = cute.arch.fmax(to_f32(amax_value), Float32(0.0))
        else:
            amax_c = fabs_f32(to_f32(amax_value))
    else:
        # Otherwise we need to case input values to fp32. Allocate the register tensor and load from SMEM input tiles.
        rX_thread_f32 = cute.make_rmem_tensor(
            layout_or_shape=cute.make_layout((MXFP8_BLOCK_SCALING_SIZE,), stride=(1,)),
            dtype=Float32,
        )
        for i in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE):
            rX_thread_f32[i] = Float32(sX_thread[i])
        # Apply activation (fwd) or grad·act'(act_input) (bwd dact) in f32.
        if cutlass.const_expr(WITH_DACT):
            dop = SUPPORTED_DACTIVATIONS[ACTIVATION]
            sA_thread = cute.composition(sActInput_tile, tv_layout)[tidx, None]
            for i in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE):
                rX_thread_f32[i] = rX_thread_f32[i] * dop(Float32(sA_thread[i]))
        elif cutlass.const_expr(WITH_ACT):
            op = SUPPORTED_ACTIVATIONS[ACTIVATION]
            # Don't repeat these OOB masking computation in the loop
            thread_col_start = tile_col_start + tidx
            thread_col_oob = thread_col_start >= N
            for i in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE):
                # If it's relu, we can handle it later in the cvt
                if not cutlass.const_expr(FUSE_RELU):
                    rX_thread_f32[i] = op(rX_thread_f32[i])
                # If the input shape is not divisible by the tile size,
                # TMA would zero-fills the input tile outside its logical MxN bounds.
                # This is fine for non-activation cases, but for activation cases,
                # op(0) might not be 0 which will pollute the amax and dbias.
                # So we must manually mask the OOB region here.
                if not cutlass.const_expr(SKIP_INPUT_MASKING):
                    if tile_row_start + i >= M or thread_col_oob:
                        rX_thread_f32[i] = Float32(0.0)
        # Accumulate fp32 activations to DBIAS before we truncate to half precision when the input is half precision
        if cutlass.const_expr(WITH_DBIAS):
            for i in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE):
                dbias_partial += rX_thread_f32[i]
        # Truncate the activation (after we apply op) back to the half precision type if input is also half precision.
        if cutlass.const_expr(is_packed16(DTYPE) and ACTIVATION is not None):
            truncate_f32 = truncate_f32_f16 if DTYPE is cutlass.Float16 else truncate_f32_bf16
            for i in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE):
                rX_thread_f32[i] = truncate_f32(rX_thread_f32[i])
        # Columnwise is the preferred direction so it runs first. If it needs to cache the activation in the input tile
        # to let the rowwise pass read it, we need to cast and overwrite the input data in-place here
        if cutlass.const_expr(CACHE_ACTIVATION):
            for i in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE):
                sX_thread[i] = DTYPE(rX_thread_f32[i])
        amax_c = Float32(0.0)
        for i in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE):
            if cutlass.const_expr(FUSE_RELU):
                amax_c = cute.arch.fmax(amax_c, rX_thread_f32[i])
            else:
                amax_c = cute.arch.fmax(amax_c, fabs_f32(rX_thread_f32[i]))

    # Irregular shapes: skip when this stage's row range or this thread's
    # column lies past the input extents. TILE_Y == MXFP8_BLOCK_SCALING_SIZE so each stage
    # is exactly one scale-row; valid iff `tile_row_start < M`.
    biased_exp_c = cvt_f32_to_fp8e8m0fnu(amax_c * MAX_NORM_RCP)
    # If the input shape is divisible by the tile size, then we won't access OOB regions because
    # we never we only access tiles we actually need (num_tiles) which are never OOB
    if cutlass.const_expr(SKIP_SCALE_BOUNDS):
        if cutlass.const_expr(SWIZZLE):
            mS_col_stage[(0, tidx % 32, tidx // 32)] = biased_exp_c
        else:
            mS_col_stage[(0, tidx)] = biased_exp_c
    else:
        scale_col = tile_col_start + tidx
        if tile_row_start < M and scale_col < N:
            if cutlass.const_expr(SWIZZLE):
                mS_col_stage[(0, tidx % 32, tidx // 32)] = biased_exp_c
            else:
                mS_col_stage[(0, tidx)] = biased_exp_c

    inv_scale_c = exp2f_rcp(biased_exp_c)
    # cvt.rn.satfinite can be vectorized to convert 2 f32 to 2 fp8 in one instruction
    cvt_x2_func = get_cvt_f32x2_to_fp8x2_func(FP8_DTYPE, FUSE_RELU)
    sO_thread_fp8 = cute.make_tensor(
        cute.recast_ptr(sO_thread.iterator, dtype=FP8_DTYPE), sO_thread.layout
    )
    if cutlass.const_expr(USE_HALF_PRECISION):
        to_f32 = to_f32_f16 if DTYPE is cutlass.Float16 else to_f32_bf16
        for j in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE // 2):
            lo, hi = 2 * j, 2 * j + 1
            v_lo = to_f32(in_c[lo])
            v_hi = to_f32(in_c[hi])
            # Accumulate the per-thread column partial for dbias if WITH_DBIAS.
            # Kept as two adds in element order: f32 addition is not associative
            if cutlass.const_expr(WITH_DBIAS):
                dbias_partial += v_lo
                dbias_partial += v_hi
            pair = cvt_x2_func(v_hi * inv_scale_c, v_lo * inv_scale_c)
            sO_thread_fp8[lo] = Uint8(pair & Uint16(0xFF)).bitcast(FP8_DTYPE)
            sO_thread_fp8[hi] = Uint8(pair >> Uint16(8)).bitcast(FP8_DTYPE)
    else:
        for j in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE // 2):
            lo, hi = 2 * j, 2 * j + 1
            pair = cvt_x2_func(rX_thread_f32[hi] * inv_scale_c, rX_thread_f32[lo] * inv_scale_c)
            sO_thread_fp8[lo] = Uint8(pair & Uint16(0xFF)).bitcast(FP8_DTYPE)
            sO_thread_fp8[hi] = Uint8(pair >> Uint16(8)).bitcast(FP8_DTYPE)

    # Return this stage's per-column partial alongside amax; the caller accumulates
    # it across stages (a scalar can't be updated in-place through the arg).
    return amax_c, dbias_partial


@cute.jit
def dbias_reduction_colwise(
    sX_tile: cute.Tensor,  # (TILE_Y, TILE_X) bf16/fp16/fp32 smem view, post-TMA
    sA_tile: Optional[cute.Tensor],  # (TILE_Y, TILE_X) activation-input smem tile (dact only)
    ACTIVATION: cutlass.Constexpr[str | None],
    DTYPE: cutlass.Constexpr[Type[cutlass.Numeric]],
    TILE_X: cutlass.Constexpr[int],
    WITH_DACT: cutlass.Constexpr[bool] = False,
    CACHE_ACTIVATION: cutlass.Constexpr[bool] = False,  # cache post-activation values to sX_tile
):
    """Reduce one SMEM tile along rows (the dbias direction) and possibly cache dact values, with no quantization."""
    tidx, _, _ = cute.arch.thread_idx()

    # Same TV layout as quantize_colwise_mxfp8: thread tidx owns column tidx, all TILE_Y rows.
    _, tv_layout = cute.make_layout_tv(
        thr_layout=cute.make_layout((1, TILE_X), stride=(TILE_X, 1)),
        val_layout=cute.make_layout((MXFP8_BLOCK_SCALING_SIZE, 1), stride=(1, 1)),
    )
    sX_thread = cute.composition(sX_tile, tv_layout)[tidx, None]

    dbias_partial = Float32(0.0)
    if cutlass.const_expr(WITH_DACT):
        dop = SUPPORTED_DACTIVATIONS[ACTIVATION]
        sA_thread = cute.composition(sA_tile, tv_layout)[tidx, None]
        for i in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE):
            x = Float32(sX_thread[i]) * dop(Float32(sA_thread[i]))
            dbias_partial += x
            if cutlass.const_expr(CACHE_ACTIVATION):
                sX_thread[i] = DTYPE(x)
    else:
        for i in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE):
            dbias_partial += Float32(sX_thread[i])
    return dbias_partial


class MXFP8QuantizeKernel(MXFP8QuantizeKernelBase):
    """The MXFP8 quantization kernel that mirrors the standard (non-specialized) MXFP8 CUDA C++ quantization kernel
    with multiple fusions (activation, dbias, etc.).
    `__call__` method is the entrypoint which is AOT compiled. `self` will be captured so it's fixed per compiled kernel
    """

    # Vectorised access constants for bank-conflict avoidance (rowwise pass)
    _PACK_SIZE = 4  # Elements per vector load
    # Each thread reads 8 waves with each wave reads 4 packed bf16, so it reads a whole MXFP8 block in total
    _WAVES = MXFP8_BLOCK_SCALING_SIZE // _PACK_SIZE
    _TOTAL_BANKS_WIDTH = (32 * 4) // 1  # 32 banks × 4 bytes, in bytes
    _THREADS_PER_BANK = _TOTAL_BANKS_WIDTH // MXFP8_BLOCK_SCALING_SIZE  # 4 threads per bank
    _NUM_STAGES = 2  # The pipeline depth is always 2

    def __init__(
        self,
        cfg: MXFP8QuantizeConfig,
        SKIP_INPUT_MASKING: bool = False,
        SKIP_SCALE_BOUNDS: bool = False,
    ):
        self.cfg = cfg
        # If the input shape is divisible by the tile size or f(0)=0 holds for activaions,
        # we can skip masking inputs with zero in the kernel and save some instructions.
        self.SKIP_INPUT_MASKING = SKIP_INPUT_MASKING
        # If the input shape is divisible by the tile size, we can skip bounds check for scale writes and save some instructions.
        self.SKIP_SCALE_BOUNDS = SKIP_SCALE_BOUNDS
        # Only honor the noop flag when no activation or dbias is fused to match CUDA C++'s implementation
        self.CHECK_NOOP_FLAG: cutlass.const_expr = (
            not self.cfg.WITH_ACT and not self.cfg.WITH_DACT and not self.cfg.WITH_DBIAS
        )
        cast_dbias_only = cfg.WITH_DBIAS and not cfg.WITH_DACT and not cfg.WITH_ACT
        # Use a different tile size for dbias only config
        # No matter what tile size we use, each thread always handles a (1, MXFP8_BLOCK_SCALING_SIZE) chunk
        if cast_dbias_only:
            self._NUM_TILES = 4  # Each CTA handles 4 tiles stacked vertically
            self._THREADS_PER_CTA = 128
        else:
            self._NUM_TILES = 2  # Each CTA handles 2 tiles stacked vertically
            self._THREADS_PER_CTA = 64
        # Each thread handles a (1, MXFP8_BLOCK_SCALING_SIZE) chunk
        self._TILE_COLS = self._THREADS_PER_CTA
        self._TILE_ROWS = MXFP8_BLOCK_SCALING_SIZE
        self._NUM_WARPS = self._THREADS_PER_CTA // 32
        # We prefer to do dbias reduction in colwise which is easier (no cross-thread reduction needed).
        # Only do rowwise reduction when we don't quantize columnwisely when WITH_DBIAS is True.

        # If columnwise is not quantized, and dbias is fused, we will do a columnwise reduction only pass
        # (may also cache dact values) without quantization because doing dbias in rowwise requires a SMEM shuffle
        # which is even slower
        self.COLWISE_DBIAS_REDUCTION_ONLY = cfg.WITH_DBIAS and not cfg.COLWISE
        # If columnwise is quantized and dbias is fused, do dbias reduction in colwise which is easier
        # (no cross-thread reduction needed).
        self.DBIAS_REDUCTION_IN_COLWISE = cfg.WITH_DBIAS and cfg.COLWISE
        # Never do dbias reduction in rowwise because it is slow. Leave this so in case it's enabled in the future
        self.DBIAS_REDUCTION_IN_ROWWISE = False

        # Cache activation in-place in the SMEM input tile when we process both rowwise and colwise passes
        # so the activation is only computed once in the direction we favor (columnwise) and the other direction (rowwise)
        # reads the cached value instead of recomputing it.
        # Note: if activation is relu, there is no standalong relu applied because it's already fused into `cvt.rn.satfinite`
        # so it should be treated as "no activation"
        self.CACHE_ACTIVATION = (
            (cfg.WITH_ACT or cfg.WITH_DACT)
            and cfg.ROWWISE
            # Always cache activation in the colwise quantization or dbias reduction only pass
            and (cfg.COLWISE or self.COLWISE_DBIAS_REDUCTION_ONLY)
            and cfg.ACTIVATION != "relu"
        )
        # The global tensor amax (mAmax) is the max over ALL elements. Each direction's
        # per-block amaxes already span every element, so when both passes run we only
        # fold the global amax from one of them — favor colwise (matches the flags
        # above). The per-block *scale* amax is still computed in each pass for its own
        # scale; this only skips the redundant global comparison in the other pass.
        self.AMAX_FROM_COLWISE = cfg.WITH_AMAX and cfg.COLWISE
        self.AMAX_FROM_ROWWISE = cfg.WITH_AMAX and not cfg.COLWISE

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,  # Input tensor to quantize
        mO_row: Optional[cute.Tensor],
        mS_row: Optional[cute.Tensor],  # Rowwise output and scale tensors
        mO_col: Optional[cute.Tensor],
        mS_col: Optional[cute.Tensor],  # Colwise output and scale tensors
        mAmax: Optional[cute.Tensor],  # Global amax accumulator, only used when WITH_AMAX is True
        mNoop: cute.Pointer,  # f32 cast_noop flag; may be null, checked on device
        mDActInput: Optional[
            cute.Tensor
        ],  # Activation input for activation derivative fusion, only used when WITH_DACT is True
        mWorkspace: Optional[
            cute.Tensor
        ],  # Workspace for the dbias reduction, only used when WITH_DBIAS is True
        stream: CUstream,
    ):
        if cutlass.const_expr(CUTEDSL_DEBUG_LOGGING):
            cute.printf(f"[CuTeDSL] MXFP8QuantizeKernel.__call__() with config: {self.cfg}\n")

        M = mX.shape[0]
        N = mX.shape[1]
        cfg = self.cfg

        # If WITH_GEMM_SWIZZLED_SCALES is enabled, the output must satisfy cublas's swizzled layout
        # This is expressed as a CuTe layout applied to the output tensor so it can be transparent throughout the kernel implementation.
        # See https://docs.nvidia.com/cuda/cublas/#d-block-scaling-factors-layout for more details.
        if cutlass.const_expr(cfg.WITH_GEMM_SWIZZLED_SCALES):
            mS_row, mS_col = derive_swizzled_scale_layout(
                M, N, cfg.ROWWISE, cfg.COLWISE, mS_row, mS_col
            )

        # We have 2 stages in our pipeline where each stage loads / computes a (TILE_Y, TILE_X) tile
        smem_tile_layout = cute.make_ordered_layout(
            (self._TILE_ROWS, self._TILE_COLS), order=(1, 0)
        )
        cta_tiler = (self._TILE_ROWS, self._TILE_COLS)

        # Input TMA atoms
        op_load = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_atom, tma_src = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op_load,
            mX,
            smem_tile_layout,
            cta_tiler,
            num_multicast=1,
        )

        # Activation input TMA atoms for activation derivative fusion
        tma_atom_act = None
        tma_src_act = None
        if cutlass.const_expr(cfg.WITH_DACT):
            tma_atom_act, tma_src_act = cute.nvgpu.cpasync.make_tiled_tma_atom(
                op_load,
                mDActInput,
                smem_tile_layout,
                cta_tiler,
                num_multicast=1,
            )

        # Output TMA atoms
        op_store = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
        out_smem_layout = cute.make_ordered_layout((self._TILE_ROWS, self._TILE_COLS), order=(1, 0))
        tma_atom_out_row = None
        tma_dst_out_row = None
        tma_atom_out_col = None
        tma_dst_out_col = None
        if cutlass.const_expr(cfg.ROWWISE):
            tma_atom_out_row, tma_dst_out_row = cute.nvgpu.cpasync.make_tiled_tma_atom(
                op_store,
                mO_row,
                out_smem_layout,
                cta_tiler,
                num_multicast=1,
            )
        if cutlass.const_expr(cfg.COLWISE):
            tma_atom_out_col, tma_dst_out_col = cute.nvgpu.cpasync.make_tiled_tma_atom(
                op_store,
                mO_col,
                out_smem_layout,
                cta_tiler,
                num_multicast=1,
            )

        grid = [
            cute.ceil_div(Int32(N), self._TILE_COLS),
            cute.ceil_div(M, self._TILE_ROWS * self._NUM_TILES),
        ]
        block = [
            self._THREADS_PER_CTA,
        ]

        self.kernel(
            mX,
            mS_row,
            mS_col,
            mAmax,
            mNoop,
            mWorkspace,
            mX.element_type,
            tma_atom,
            tma_src,
            tma_atom_out_row,
            tma_dst_out_row,
            tma_atom_out_col,
            tma_dst_out_col,
            tma_atom_act,
            tma_src_act,
        ).launch(
            grid=grid,
            block=block,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mS_row: Optional[cute.Tensor],
        mS_col: Optional[cute.Tensor],
        mAmax: Optional[cute.Tensor],
        mNoop: cute.Pointer,
        mWorkspace: Optional[cute.Tensor],
        dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
        tma_atom: cute.CopyAtom,
        tma_src: cute.Tensor,  # Input TMA atoms
        tma_atom_out_row: Optional[cute.CopyAtom],
        tma_dst_out_row: Optional[cute.Tensor],  # Rowwise output TMA atoms
        tma_atom_out_col: Optional[cute.CopyAtom],
        tma_dst_out_col: Optional[cute.Tensor],  # Colwise output TMA atoms
        tma_atom_act: Optional[cute.CopyAtom],
        tma_src_act: Optional[
            cute.Tensor
        ],  # Activation derivative TMA atoms, None unless WITH_DACT
    ):
        """Device entry: no-op the CTA when the noop flag is set, else run the quantize main loop."""

        skip_execution = False
        if cutlass.const_expr(self.CHECK_NOOP_FLAG):
            skip_execution = noop_flag_is_set(mNoop)
        if not skip_execution:
            self._kernel_main(
                mX,
                mS_row,
                mS_col,
                mAmax,
                mWorkspace,
                dtype,
                tma_atom,
                tma_src,
                tma_atom_out_row,
                tma_dst_out_row,
                tma_atom_out_col,
                tma_dst_out_col,
                tma_atom_act,
                tma_src_act,
            )

    @cute.jit
    def _kernel_main(
        self,
        mX: cute.Tensor,
        mS_row: Optional[cute.Tensor],
        mS_col: Optional[cute.Tensor],
        mAmax: Optional[cute.Tensor],
        mWorkspace: Optional[cute.Tensor],
        dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
        tma_atom: cute.CopyAtom,
        tma_src: cute.Tensor,  # Input TMA atoms
        tma_atom_out_row: Optional[cute.CopyAtom],
        tma_dst_out_row: Optional[cute.Tensor],  # Rowwise output TMA atoms
        tma_atom_out_col: Optional[cute.CopyAtom],
        tma_dst_out_col: Optional[cute.Tensor],  # Colwise output TMA atoms
        tma_atom_act: Optional[cute.CopyAtom],
        tma_src_act: Optional[
            cute.Tensor
        ],  # Activation derivative TMA atoms, None unless WITH_DACT
    ):
        cfg = self.cfg

        if cutlass.const_expr(cfg.ROWWISE):
            mS_row = cute.zipped_divide(
                mS_row, (self._TILE_ROWS, self._TILE_COLS // MXFP8_BLOCK_SCALING_SIZE)
            )
        if cutlass.const_expr(cfg.COLWISE):
            mS_col = cute.zipped_divide(
                mS_col, (self._TILE_ROWS // MXFP8_BLOCK_SCALING_SIZE, self._TILE_COLS)
            )

        # Allocate shared memory for the input and rowwise / columnwise outputs
        FP8_DTYPE = cfg.FP8_DTYPE

        if cutlass.const_expr(cfg.ROWWISE and cfg.COLWISE):

            @cute.struct
            class SharedStorage:
                mbar_storage: cute.struct.MemRange[cute.Int64, 2 * self._NUM_STAGES]
                sX: cute.struct.Align[
                    cute.struct.MemRange[
                        dtype, self._TILE_ROWS * self._TILE_COLS * self._NUM_STAGES
                    ],
                    128,
                ]
                sO_row: cute.struct.Align[
                    cute.struct.MemRange[
                        FP8_DTYPE, self._TILE_ROWS * self._TILE_COLS * self._NUM_STAGES
                    ],
                    128,
                ]
                sO_col: cute.struct.Align[
                    cute.struct.MemRange[
                        FP8_DTYPE, self._TILE_ROWS * self._TILE_COLS * self._NUM_STAGES
                    ],
                    128,
                ]
                sAmax: cute.struct.MemRange[Float32, self._NUM_WARPS]

        elif cutlass.const_expr(cfg.ROWWISE and not cfg.COLWISE):

            @cute.struct
            class SharedStorage:
                mbar_storage: cute.struct.MemRange[cute.Int64, 2 * self._NUM_STAGES]
                sX: cute.struct.Align[
                    cute.struct.MemRange[
                        dtype, self._TILE_ROWS * self._TILE_COLS * self._NUM_STAGES
                    ],
                    128,
                ]
                sO_row: cute.struct.Align[
                    cute.struct.MemRange[
                        FP8_DTYPE, self._TILE_ROWS * self._TILE_COLS * self._NUM_STAGES
                    ],
                    128,
                ]
                sAmax: cute.struct.MemRange[Float32, self._NUM_WARPS]

        else:

            @cute.struct
            class SharedStorage:
                mbar_storage: cute.struct.MemRange[cute.Int64, 2 * self._NUM_STAGES]
                sX: cute.struct.Align[
                    cute.struct.MemRange[
                        dtype, self._TILE_ROWS * self._TILE_COLS * self._NUM_STAGES
                    ],
                    128,
                ]
                sO_col: cute.struct.Align[
                    cute.struct.MemRange[
                        FP8_DTYPE, self._TILE_ROWS * self._TILE_COLS * self._NUM_STAGES
                    ],
                    128,
                ]
                sAmax: cute.struct.MemRange[Float32, self._NUM_WARPS]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        # Apply the layout to the allocated shared memory buffers so the first rank is the tile (nested layout)
        # and the second rank is the pipeline stage
        sX = storage.sX.get_tensor(
            cute.make_layout(
                ((self._TILE_ROWS, self._TILE_COLS), self._NUM_STAGES),
                stride=((self._TILE_COLS, 1), self._TILE_ROWS * self._TILE_COLS),
            )
        )
        if cutlass.const_expr(cfg.ROWWISE):
            sO_row = storage.sO_row.get_tensor(
                cute.make_layout(
                    ((self._TILE_ROWS, self._TILE_COLS), self._NUM_STAGES),
                    stride=((self._TILE_COLS, 1), self._TILE_ROWS * self._TILE_COLS),
                )
            )
        if cutlass.const_expr(cfg.COLWISE):
            sO_col = storage.sO_col.get_tensor(
                cute.make_layout(
                    ((self._TILE_ROWS, self._TILE_COLS), self._NUM_STAGES),
                    stride=((self._TILE_COLS, 1), self._TILE_ROWS * self._TILE_COLS),
                )
            )

        # Allocate shared memory for the activation input used for the activation derivative fusion.
        if cutlass.const_expr(cfg.WITH_DACT):

            @cute.struct
            class DactStorage:
                sActInput: cute.struct.Align[
                    cute.struct.MemRange[
                        dtype, self._TILE_ROWS * self._TILE_COLS * self._NUM_STAGES
                    ],
                    128,
                ]

            dact_storage = smem.allocate(DactStorage)
            # Apply the same layout as the input
            sActInput = dact_storage.sActInput.get_tensor(
                cute.make_layout(
                    ((self._TILE_ROWS, self._TILE_COLS), self._NUM_STAGES),
                    stride=((self._TILE_COLS, 1), self._TILE_ROWS * self._TILE_COLS),
                )
            )

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Prefetch TMA descriptors
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom)
            if cutlass.const_expr(cfg.WITH_DACT):
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_act)

        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        # Only warp 0 is the producer (issues TMA)
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        # Every warp is the consumer (reads the data loaded by TMA)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self._NUM_WARPS)

        # Bytes transferred per TMA copy: one (TILE_Y, TILE_X) tile of dtype.
        tx_count = self._TILE_ROWS * self._TILE_COLS * dtype.width // 8
        # dact loads two tiles (grad + act_input) under the same per-stage barrier,
        # so the barrier must expect both copies' bytes.
        if cutlass.const_expr(cfg.WITH_DACT):
            tx_count *= 2

        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_storage.data_ptr(),
            num_stages=self._NUM_STAGES,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=tx_count,
            cta_layout_vmnk=None,  # single-CTA, no cluster/multicast
        )

        prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self._NUM_STAGES
        )
        cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self._NUM_STAGES
        )

        M = mX.shape[0]
        N = mX.shape[1]

        num_tiles = cutlass.min(
            self._NUM_TILES,
            cute.ceil_div(M - bidy * self._TILE_ROWS * self._NUM_TILES, self._TILE_ROWS),
        )

        # Tile the TMA gmem view: ((TILE_Y, TILE_X), (M/TILE_Y, N/TILE_X)).
        gX_tiled = cute.zipped_divide(tma_src, (self._TILE_ROWS, self._TILE_COLS))

        # Partition sX/gX for the TMA atom (single-CTA, no cluster/multicast).
        tXsX, tXgX = cute.nvgpu.cpasync.tma_partition(
            tma_atom,
            0,  # Use the only CTA to do the TMA copy
            cute.make_layout(1),  # This cluster only has 1 CTAs
            sX,
            gX_tiled,
        )

        # If WITH_DACT, partition the activation input for TMA as well in the same way
        if cutlass.const_expr(cfg.WITH_DACT):
            gA_tiled = cute.zipped_divide(tma_src_act, (self._TILE_ROWS, self._TILE_COLS))
            tXsA, tXgA = cute.nvgpu.cpasync.tma_partition(
                tma_atom_act,
                0,
                cute.make_layout(1),
                sActInput,
                gA_tiled,
            )

        # Partitioning for rowwise / columnwise outputs
        if cutlass.const_expr(cfg.ROWWISE):
            gO_row_tiled = cute.zipped_divide(tma_dst_out_row, (self._TILE_ROWS, self._TILE_COLS))
            tXsO_row, tXgO_row = cute.nvgpu.cpasync.tma_partition(
                tma_atom_out_row,
                0,
                cute.make_layout(1),
                sO_row,
                gO_row_tiled,
            )
        if cutlass.const_expr(cfg.COLWISE):
            gO_col_tiled = cute.zipped_divide(tma_dst_out_col, (self._TILE_ROWS, self._TILE_COLS))
            tXsO_col, tXgO_col = cute.nvgpu.cpasync.tma_partition(
                tma_atom_out_col,
                0,
                cute.make_layout(1),
                sO_col,
                gO_col_tiled,
            )

        # Ensure barrier init is visible to all threads before the pipeline is used.
        cute.arch.sync_threads()

        # Prologue: warp 0 prefetches up to NUM_STAGES tiles to fully fill the pipeline
        if warp_idx == 0:
            for s in cutlass.range_constexpr(self._NUM_STAGES):
                if s < num_tiles:
                    mainloop_pipeline.producer_acquire(prod_state)
                    tile_y = bidy * self._NUM_TILES + s
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

        # Per-thread amax accumulator
        if cutlass.const_expr(cfg.WITH_AMAX):
            per_thread_amax = Float32(0.0)

        # Prepare thread-level register accumulators for rowwise dbias reduction.
        # Each thread will process two (1, MXFP8_BLOCK_SCALING_SIZE) rows in two stages, and in each stage the thread will add the
        # (after dact applied) value to this register array with the same shape so it carries the the two stages' partial sum.
        # Then it will be written to a SMEM buffer to let the whole CTA do the reduction separately to yield
        # the final (1, TILE_X) dbias workspace output.
        rowwise_dbias_acc = None
        if cutlass.const_expr(self.DBIAS_REDUCTION_IN_ROWWISE):
            rowwise_dbias_acc = cute.make_rmem_tensor(
                layout_or_shape=cute.make_layout((MXFP8_BLOCK_SCALING_SIZE,), stride=(1,)),
                dtype=Float32,
            )
            # Zero the accumulator registers.
            for c in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE):
                rowwise_dbias_acc[c] = Float32(0.0)
            block_dbias = Float32(0.0)
        # Prepare thread-level register for columnwise dbias reduction.
        # Each thread will process two (MXFP8_BLOCK_SCALING_SIZE, 1) columns in two stages, and in each stage the thread will reduce the
        # (after dact applied) column to (1,) and add to this register.
        # Then this partial sum scalar will be written to the GMEM workspace buffer directly.
        if cutlass.const_expr(self.DBIAS_REDUCTION_IN_COLWISE or self.COLWISE_DBIAS_REDUCTION_ONLY):
            block_dbias = Float32(0.0)

        # Consumer: all warps fetch from the pipeline, process its tile, and issue a new load to the tile buffer it just consumed
        for tile_idx in cutlass.range(num_tiles, unroll=1):
            mainloop_pipeline.consumer_wait(cons_state)
            # Only allow at most _NUM_STAGES-1 stages to be in-flight, because this iteration will reuse the ring buffer
            # that is read _NUM_STAGES iterations ago so we must wait for whoever is reading that buffer to finish
            if warp_idx == 0:
                cute.arch.cp_async_bulk_wait_group(self._NUM_STAGES - 1, read=True)
            cute.arch.sync_threads()
            # The current pipeline stage index, which is the tile index modulo the number of stages.
            # This is used to index into the shared memory ring buffers that are wrapped around the number of stages.
            stage_idx = cons_state.index
            sX_tile = sX[(None, stage_idx)]
            # Also fetch the activation input if WITH_DACT
            sActInput_tile = None
            if cutlass.const_expr(cfg.WITH_DACT):
                sActInput_tile = sActInput[(None, stage_idx)]
            # Each CTA handles `NUM_TILES` tiles stacked vertically, so tile_idx_x is just the block index along X dimension
            # and tile_idx_y is the tile that this stage handles out of the `NUM_TILES` tiles
            tile_idx_x = bidx
            tile_idx_y = bidy * self._NUM_TILES + tile_idx
            # Process rowwise and colwise quantization separately
            if cutlass.const_expr(cfg.COLWISE):
                # The first row that belongs to this CTA. Each CTA handles NUM_TILES of (TILE_Y, TILE_X) tiles stacked vertically,
                # and each stage handles one of them.
                sO_col_tile = sO_col[(None, stage_idx)]
                mS_col_stage = cute.flatten(mS_col[(None, (tile_idx_y, tile_idx_x))])

                amax_c, dbias_c = self._process_colwise(
                    sX_tile,
                    sO_col_tile,
                    mS_col_stage,
                    tile_idx_y * self._TILE_ROWS,
                    bidx * self._TILE_COLS,
                    M,
                    N,
                    sActInput_tile,
                )
                if cutlass.const_expr(self.AMAX_FROM_COLWISE):
                    per_thread_amax = cute.arch.fmax(per_thread_amax, amax_c)
                if cutlass.const_expr(self.DBIAS_REDUCTION_IN_COLWISE):
                    block_dbias += dbias_c
            # If we don't quantize columnwise but we fuse dbias, do a dbias reduction only pass in the columnwise direction without quantization
            if cutlass.const_expr(self.COLWISE_DBIAS_REDUCTION_ONLY):
                block_dbias += self._dbias_only_colwise(sX_tile, sActInput_tile)
            # If we cache the activation in shared memory, we need to ensure that all threads have finished writing to the shared memory
            # from the columnwise pass before any thread reads from it in the rowwise pass.
            if cutlass.const_expr(self.CACHE_ACTIVATION):
                cute.arch.sync_threads()
            if cutlass.const_expr(cfg.ROWWISE):
                sO_row_tile = sO_row[(None, stage_idx)]
                # mS_row is ((SCALE_TILE), (GRID)) where SCALE_TILE = (32, 2).
                # Each CTA owns NUM_TILES consecutive row-tiles of GRID. cute
                # auto-decomposes the flat row coord `bidy * NUM_TILES + tile_idx`
                # onto GRID's hierarchical row modes — which is the
                # (i_hi, tile_Y) tile-major order for swizzled, and the plain
                # row-tile order for compact. Same source, both layouts correct.
                mS_row_stage = cute.flatten(mS_row[(None, (tile_idx_y, tile_idx_x))])
                amax_r = self._process_rowwise(
                    sX_tile,
                    sO_row_tile,
                    mS_row_stage,
                    tile_idx_y * self._TILE_ROWS,
                    bidx * self._TILE_COLS,
                    M,
                    N,
                    sActInput_tile,
                    rowwise_dbias_acc,
                )

                if cutlass.const_expr(self.AMAX_FROM_ROWWISE):
                    per_thread_amax = cute.arch.fmax(per_thread_amax, amax_r)

            # Make the shared-memory writes visible to the TMA's async proxy before the TMA reads them.
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            cute.arch.sync_threads()

            # We are done with this input pipeline SMEM buffer, signal the producer that it can write to this buffer
            mainloop_pipeline.consumer_release(cons_state)

            # Warp 0 issues TMA copy to write the quantized output tile from shared memory to global memory and then commits
            if warp_idx == 0:
                tile_y = bidy * self._NUM_TILES + tile_idx
                if cutlass.const_expr(cfg.ROWWISE):
                    cute.copy(
                        tma_atom_out_row,
                        tXsO_row[(None, stage_idx)],
                        tXgO_row[(None, (tile_y, bidx))],
                    )
                if cutlass.const_expr(cfg.COLWISE):
                    cute.copy(
                        tma_atom_out_col,
                        tXsO_col[(None, stage_idx)],
                        tXgO_col[(None, (tile_y, bidx))],
                    )
                cute.arch.cp_async_bulk_commit_group()

            cons_state.advance()

            # The pipeline is no longer fully filled after we consume this tile, so we fetch a new tile to fill the pipeline.
            # The next _NUM_STAGES-1 tiles are already in-flight, so the next tile to fetch is after _NUM_STAGES tiles.
            if warp_idx == 0:
                next_tile_idx = tile_idx + self._NUM_STAGES
                if next_tile_idx < num_tiles:
                    mainloop_pipeline.producer_acquire(prod_state)
                    tile_y = bidy * self._NUM_TILES + next_tile_idx
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
        # End of the main pipeline loop

        # Complete the cross-thread dbias reduction after each thread has its own per-thread partial sum after the rowwise quantization.
        if cutlass.const_expr(self.DBIAS_REDUCTION_IN_ROWWISE):
            # If we do the dbias reduction in the rowwise pass, each thread will have a (1, MXFP8_BLOCK_SCALING_SIZE) partial sum
            # and we need to write these to a SMEM buffer and let each thread reduce it in the columnwise direction
            block_dbias = self._dbias_reduction_rowwise_epilouge(smem, tidx, rowwise_dbias_acc)

        # Write the per-tile reduced dbias to the global workspace.
        if cutlass.const_expr(cfg.WITH_DBIAS):
            dbias_col = bidx * self._TILE_COLS + tidx
            if dbias_col < N:
                mWorkspace[(bidy, dbias_col)] = block_dbias

        if cutlass.const_expr(cfg.WITH_AMAX):
            sAmax = storage.sAmax.get_tensor(cute.make_layout(self._NUM_WARPS))
            self._amax_epilogue(sAmax, mAmax, tidx, warp_idx, per_thread_amax)

        # Wait for in-flight TMA stores so data is visible to the host
        # before the kernel returns.
        cute.arch.cp_async_bulk_wait_group(0, read=False)

    @cute.jit
    def _dbias_only_colwise(
        self,
        sX_tile: cute.Tensor,  # (TILE_Y, TILE_X) bf16/fp16 smem view, post-TMA
        sActInput_tile: Optional[
            cute.Tensor
        ] = None,  # (TILE_Y, TILE_X) act_input tile (dact only),
    ):
        return dbias_reduction_colwise(
            sX_tile,
            sActInput_tile,
            ACTIVATION=self.cfg.ACTIVATION,
            DTYPE=self.cfg.DTYPE,
            TILE_X=self._TILE_COLS,
            WITH_DACT=self.cfg.WITH_DACT,
            CACHE_ACTIVATION=self.CACHE_ACTIVATION,
        )

    @cute.jit
    def _dbias_reduction_rowwise_epilouge(
        self, smem: cutlass.utils.SmemAllocator, tidx: Int32, rowwise_dbias_acc: cute.Tensor
    ):
        # Pad the buffer to avoid bank conflicts. The logical shape is still the same. Only the stride is different.
        DBIAS_BUFF_WIDTH = (
            self._TILE_COLS // MXFP8_BLOCK_SCALING_SIZE * (MXFP8_BLOCK_SCALING_SIZE + 1)
        )

        # Allocate the SMEM buffer that all threads use to reduce the two-stage partial sum (per thread) to the
        # partial sum (per block).
        @cute.struct
        class DbiasStorage:
            sDbias: cute.struct.MemRange[Float32, self._TILE_ROWS * DBIAS_BUFF_WIDTH]

        dbias_storage = smem.allocate(DbiasStorage)
        sDbias = dbias_storage.sDbias.get_tensor(
            cute.make_layout((self._TILE_ROWS, self._TILE_COLS), stride=(DBIAS_BUFF_WIDTH, 1)),
        )
        _, tv_layout_dbias_write = cute.make_layout_tv(
            thr_layout=cute.make_layout(
                (self._TILE_ROWS, self._TILE_COLS // MXFP8_BLOCK_SCALING_SIZE),
                stride=(self._TILE_COLS // MXFP8_BLOCK_SCALING_SIZE, 1),
            ),
            val_layout=cute.make_layout(
                (1, MXFP8_BLOCK_SCALING_SIZE), stride=(MXFP8_BLOCK_SCALING_SIZE, 1)
            ),
        )
        sDbias_write = cute.composition(sDbias, tv_layout_dbias_write)
        # Each thread start reading from the specfic bank based on its thread ID so they can do their best to access different banks
        # to avoid bank conflict.
        bank_group = (tidx % THREADS_PER_WARP) // self._THREADS_PER_BANK
        # The offset this thread should start reading from based on what's its first bank to access.
        offset = bank_group * self._PACK_SIZE
        for w in cutlass.range_constexpr(
            self._WAVES
        ):  # Each thread starts from this offset when writing into SMEM to avoid bank conflict
            start = (w * self._PACK_SIZE + offset) % MXFP8_BLOCK_SCALING_SIZE
            for i in cutlass.range_constexpr(self._PACK_SIZE):
                # All threads write their per-thread partial sum results to the shared buffer.
                sDbias_write[(tidx, start + i)] = rowwise_dbias_acc[w * self._PACK_SIZE + i]
        cute.arch.sync_threads()
        # All threads reduce the cross-thread partial sums to the per-block partial sum.
        _, tv_layout_dbias_reduce = cute.make_layout_tv(
            thr_layout=cute.make_layout((1, self._TILE_COLS), stride=(self._TILE_COLS, 1)),
            val_layout=cute.make_layout((self._TILE_ROWS, 1), stride=(1, 1)),
        )
        sDbias_reduce = cute.composition(sDbias, tv_layout_dbias_reduce)
        # make_layout_tv yields a (thread, value) layout: thread=tidx -> column tidx,
        # value=i -> row i. So index [tidx, i] (thread first), summing the column's rows.
        block_dbias = Float32(0.0)
        for i in cutlass.range_constexpr(self._TILE_ROWS):
            block_dbias += sDbias_reduce[tidx, i]
        return block_dbias

    @cute.jit
    def _amax_epilogue(
        self,
        sAmax: cute.Tensor,
        mAmax: cute.Tensor,
        tidx: Int32,
        warp_idx: Int32,
        per_thread_amax: Float32,
    ):
        # Reduce and get the per-warp amax.
        warp_amax = cute.arch.warp_redux_sync(per_thread_amax, kind="fmax")
        # Write the per-warp amax to shared memory
        lane_idx = tidx % 32
        if lane_idx == 0:
            sAmax[warp_idx] = warp_amax
        cute.arch.sync_threads()
        if tidx == 0:
            cta_amax = Float32(0.0)
            # The first thread reduces all the per-warp amax to the per-CTA amax
            for w in cutlass.range_constexpr(self._NUM_WARPS):
                cta_amax = cute.arch.fmax(cta_amax, sAmax[w])
            amax_i32 = cute.make_tensor(
                cute.recast_ptr(mAmax.iterator, dtype=Int32),
                cute.make_layout(1),
            )
            # The first thread updates the global amax with an atomic max on the bitcasted float value
            cute.arch.atomic_max(
                amax_i32.iterator,
                cta_amax.bitcast(Int32),
            )

    @cute.jit
    def _process_rowwise(
        self,
        sX_tile: cute.Tensor,  # (TILE_Y, TILE_X) bf16/fp16 smem view, post-TMA
        sO_row_tile: cute.Tensor,  # (TILE_Y, TILE_X) fp8 smem view (rowwise FP8 output)
        mS_row_stage: cute.Tensor,  # rowwise scale tensor (1D swizzled, or 2D linear)
        tile_row_start: Int32,  # global row of this stage's row 0
        tile_col_start: Int32,  # global col of this CTA's col 0
        M: Int32,
        N: Int32,  # full input extents, for OOB masking
        sActInput_tile: Optional[cute.Tensor] = None,  # (TILE_Y, TILE_X) act_input tile (dact only)
        dbias_acc: Optional[
            cute.Tensor
        ] = None,  # rmem Float32[32] dbias accumulator (rowwise-only dbias)
    ):
        cfg = self.cfg
        return quantize_rowwise_mxfp8(
            sX_tile,
            None if self.CACHE_ACTIVATION else sActInput_tile,
            sO_row_tile,
            mS_row_stage,
            self.cfg.MAX_NORM_RCP,
            tile_row_start,
            tile_col_start,
            M,
            N,
            ACTIVATION=None if self.CACHE_ACTIVATION else cfg.ACTIVATION,
            DTYPE=cfg.DTYPE,
            FP8_DTYPE=cfg.FP8_DTYPE,
            TILE_X=self._TILE_COLS,
            TILE_Y=self._TILE_ROWS,
            WAVES=self._WAVES,
            THREADS_PER_BANK=self._THREADS_PER_BANK,
            PACK_SIZE=self._PACK_SIZE,
            SKIP_INPUT_MASKING=self.SKIP_INPUT_MASKING,
            SKIP_SCALE_BOUNDS=self.SKIP_SCALE_BOUNDS,
            WITH_ACT=cfg.WITH_ACT and not self.CACHE_ACTIVATION,
            WITH_DACT=cfg.WITH_DACT and not self.CACHE_ACTIVATION,
            WITH_DBIAS=self.DBIAS_REDUCTION_IN_ROWWISE,
            dbias_acc=dbias_acc,
        )

    @cute.jit
    def _process_colwise(
        self,
        sX_tile: cute.Tensor,  # (TILE_Y, TILE_X) bf16/fp16 smem view, post-TMA
        sO_col_tile: cute.Tensor,  # (TILE_Y, TILE_X) fp8 smem view (colwise FP8 output)
        mS_col_stage: cute.Tensor,  # colwise scale tensor (1D swizzled, or 2D linear)
        tile_row_start: Int32,  # global row of this stage's row 0
        tile_col_start: Int32,  # global col of this CTA's col 0
        M: Int32,
        N: Int32,  # full input extents, for OOB masking
        sActInput_tile: Optional[cute.Tensor] = None,  # (TILE_Y, TILE_X) act_input tile (dact only)
    ):
        cfg = self.cfg
        return quantize_colwise_mxfp8(
            sX_tile,
            sActInput_tile,
            sO_col_tile,
            mS_col_stage,
            self.cfg.MAX_NORM_RCP,
            tile_row_start,
            tile_col_start,
            M,
            N,
            ACTIVATION=cfg.ACTIVATION,
            DTYPE=cfg.DTYPE,
            FP8_DTYPE=cfg.FP8_DTYPE,
            SWIZZLE=cfg.WITH_GEMM_SWIZZLED_SCALES,
            TILE_X=self._TILE_COLS,
            TILE_Y=self._TILE_ROWS,
            SKIP_INPUT_MASKING=self.SKIP_INPUT_MASKING,
            SKIP_SCALE_BOUNDS=self.SKIP_SCALE_BOUNDS,
            WITH_ACT=cfg.WITH_ACT,
            WITH_DACT=cfg.WITH_DACT,
            WITH_DBIAS=self.DBIAS_REDUCTION_IN_COLWISE,
            CACHE_ACTIVATION=self.CACHE_ACTIVATION,
        )
