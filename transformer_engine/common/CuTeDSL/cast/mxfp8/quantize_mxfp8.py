# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MXFP8 quantization kernel implemented in CuTeDSL.

Replicates the core logic of quantize_mxfp8.cuh: given a 2D tensor of BF16/FP16
values, quantize to MXFP8 format (FP8E4M3 data + E8M0 per-block scales).

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

import abc
import logging
import os
from typing import Any, Dict, Optional, Type

import cutlass
from cutlass import cute
from cutlass import pipeline
from cutlass import Float32, Int16, Int32, Int64, Uint32, Uint8
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module
import tvm_ffi

from transformer_engine.common.CuTeDSL.utils import (
    _bitcast_f32_to_i32,
    device_compute_capability,
    str_to_cutlass_dtype,
    is_packed16,
    packed16_kit,
    fabs_f32,
    exp2f_rcp,
    pack_f32x2,
    unpack_i64_to_i32x2,
)
from transformer_engine.common.CuTeDSL.activations import (
    act_relu,
    act_gelu,
    act_silu,
    act_qgelu,
    act_srelu,
    dact_drelu,
    dact_dsrelu,
    dact_dsilu,
    dact_dqgelu,
    dact_dgelu,
)
from transformer_engine.common.CuTeDSL.utils_fp8 import (
    as_byte_tensor,
    get_cvt_f32_to_fp8_func,
    cvt_f32_to_fp8e8m0,
    mul_i64_cvt_f32x4_to_fp8x4,
    mul_f32x4_cvt_f32x4_to_fp8x4,
    mul_i64_cvt_packed16x4_to_fp8x4,
)

CUTEDSL_DEBUG_LOGGING = os.environ.get("CUTEDSL_DEBUG_LOGGING", "0") == "1"

logger = logging.getLogger("transformer_engine.cutedsl.mxfp8")

# Number of elements per MXFP8 scale block. They will share the same E8M0 scale factor
MXFP8_BLOCK_SCALING_SIZE = 32
# How many threads are in one warp
THREADS_PER_WARP = 32

# FP8E4M3 max representable value
FP8E4M3_MAX_NORM = 448.0
FP8E4M3_MAX_NORM_RCP = 1.0 / FP8E4M3_MAX_NORM
FP8E5M2_MAX_NORM = 57344.0
FP8E5M2_MAX_NORM_RCP = 1.0 / FP8E5M2_MAX_NORM


SUPPORTED_ACTIVATIONS = {
    "relu": act_relu,
    "gelu": act_gelu,
    "silu": act_silu,
    "qgelu": act_qgelu,
    "srelu": act_srelu,
}

SUPPORTED_DACTIVATIONS = {
    "drelu": dact_drelu,
    "dgelu": dact_dgelu,
    "dsilu": dact_dsilu,
    "dqgelu": dact_dqgelu,
    "dsrelu": dact_dsrelu,
}


@cute.jit
def derive_swizzled_scale_layout(
    M,
    N,
    ROWWISE: cutlass.Constexpr,
    COLWISE: cutlass.Constexpr,
    mS_row,
    mS_col,
):
    num_scale_cols = N // MXFP8_BLOCK_SCALING_SIZE
    num_scale_rows = M // MXFP8_BLOCK_SCALING_SIZE

    num_tiles_M = cute.ceil_div(M, 128)
    num_tiles_SC = cute.ceil_div(num_scale_cols, 4)
    num_tiles_SR = cute.ceil_div(num_scale_rows, 4)
    num_tiles_N = cute.ceil_div(N, 128)

    if cutlass.const_expr(ROWWISE):
        mS_row = cute.make_tensor(
            mS_row.iterator,
            cute.make_layout(
                ((32, 4, num_tiles_M), (4, num_tiles_SC)),
                stride=((16, 4, num_tiles_SC * 512), (1, 512)),
            ),
        )
    if cutlass.const_expr(COLWISE):
        mS_col = cute.make_tensor(
            mS_col.iterator,
            cute.make_layout(
                ((4, num_tiles_SR), (32, 4, num_tiles_N)),
                stride=((1, 512), (16, 4, num_tiles_SR * 512)),
            ),
        )
    return mS_row, mS_col


@cute.jit
def quantize_rowwise_mxfp8(
    sX_tile,  # (TILE_Y, TILE_X) bf16/fp16 smem view, post-TMA
    sA_tile,  # (TILE_Y, TILE_X) activation-input smem tile (dact only)
    sO_row_tile,  # (TILE_Y, TILE_X) uint8 smem view (rowwise FP8 output)
    mS_row_stage,  # rowwise scale tensor (1D swizzled, or 2D linear)
    max_norm_rcp,
    tile_row_start,  # Int32 — global row index of this stage's row 0
    # (= tile_idx_y * TILE_Y). Used to mask OOB scale stores
    # for irregular shapes.
    tile_col_start,  # Int32 — global col index of this CTA's col 0
    # (= bidx * TILE_X). Same purpose.
    M,
    N,  # Int32 — full tensor extents; OOB threads skip their
    # scale store.
    ACTIVATION,
    DTYPE,
    FP8_DTYPE,
    TILE_X,
    TILE_Y,
    WAVES,
    THREADS_PER_BANK,
    PACK_SIZE,
    WITH_ACT=False,
    WITH_DACT=False,
    WITH_DBIAS=False,  # rowwise-only dbias: accumulate per-column partials
    dbias_acc=None,  #  only needed when WITH_DBIAS is True
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
    sO_thread = sO_tv[tidx, None]  # shape (32,) uint8

    sO_thread_u32_ptr = cute.recast_ptr(sO_thread.iterator, dtype=Uint32)
    # Each wave it writes 32 bytes = 8 uint32s, so in 4 waves we write all 32 quantized elements.
    sO_thread_u32 = cute.make_tensor(
        sO_thread_u32_ptr,
        cute.make_layout(
            (MXFP8_BLOCK_SCALING_SIZE // 4,), stride=(1,)
        ),  # 1 uint32 is 4 fp8 elements
    )

    # PTX allows to fuse relu activation in `cvt.rn.satfinite`
    FUSE_RELU = cutlass.const_expr(ACTIVATION == "relu")
    # For this fast path we can read in pack of 2 instead of reading individual f16 / bf16 element.
    # dbias needs the per-element fp32 values to accumulate, so it forces the slow path.
    _row_fast = is_packed16(DTYPE) and (ACTIVATION is None or FUSE_RELU) and not WITH_DBIAS

    amax_r = Float32(0.0)

    # Each thread start reading from the specfic bank based on its thread ID so they can do their best to access different banks
    # to avoid bank conflict.
    bank_group = (tidx % THREADS_PER_WARP) // THREADS_PER_BANK
    # The offset this thread should start reading from based on what's its first bank to access.
    offset = bank_group * PACK_SIZE
    if cutlass.const_expr(_row_fast):
        # If no activation, f16 / bf16 and rowwise quantization, we can read 2 f16 / bf16 at once in a pack
        # and use max.xorsign.abs.f16x2 / max.xorsign.abs.bf16x2 to compute
        kit = packed16_kit(DTYPE)
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

        amax_2x = Int32(0)
        # Each wave will use max.xorsign.abs.f16x2 or max.xorsign.abs.bf16x2 to compare 2 packed elements in parallel
        for w in cutlass.range_constexpr(WAVES):
            if cutlass.const_expr(FUSE_RELU):
                # If we fuse relu then we don't want to do abs since negative value will be set to 0 and they will lose comparison automatically
                amax_2x = kit.max_x2(amax_2x, in_r[w][0])
                amax_2x = kit.max_x2(amax_2x, in_r[w][1])
            else:
                amax_2x = kit.abs_max_x2(amax_2x, in_r[w][0])
                amax_2x = kit.abs_max_x2(amax_2x, in_r[w][1])
        if cutlass.const_expr(FUSE_RELU):
            # Compare the 2 packed max without abs
            amax_r = cute.arch.fmax(
                kit.x2_lo_to_f32(amax_2x),
                kit.x2_hi_to_f32(amax_2x),
            )
            # For relu the max is at least 0
            amax_r = cute.arch.fmax(amax_r, Float32(0.0))
        else:
            # Compare the 2 packed abs max
            amax_r = cute.arch.fmax(
                fabs_f32(kit.x2_lo_to_f32(amax_2x)),
                fabs_f32(kit.x2_hi_to_f32(amax_2x)),
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
            sA_thread = cute.composition(sA_tile, tv_layout)[tidx, None]
            sA_thread_rw = cute.make_tensor(
                sA_thread.iterator,
                cute.make_layout((1, MXFP8_BLOCK_SCALING_SIZE), stride=(0, 1)),
            )
        elif cutlass.const_expr(WITH_ACT):
            op = SUPPORTED_ACTIVATIONS[ACTIVATION]

        if cutlass.const_expr(is_packed16(DTYPE) and ACTIVATION is not None):
            kit_act = packed16_kit(DTYPE)

        # Each wave we read PACK_SIZE elements, and we have WAVES waves, so we read WAVES * PACK_SIZE (= MXFP8_BLOCK_SCALING_SIZE) elements in total.
        in_r = [[None] * PACK_SIZE for _ in range(WAVES)]
        for w in cutlass.range_constexpr(WAVES):
            start = (w * PACK_SIZE + offset) % MXFP8_BLOCK_SCALING_SIZE
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
                # Accumulate to the per-thread dbias register buffer for this tile if WITH_DBIAS
                if cutlass.const_expr(WITH_DBIAS):
                    # dbias_acc is register buffer so we can just write without bank conflict
                    dbias_acc[w * PACK_SIZE + i] += x
                # If 16-bit input with activation, truncate to IType
                if cutlass.const_expr(is_packed16(DTYPE) and ACTIVATION is not None):
                    x = kit_act.truncate_f32(x)
                in_r[w][i] = x
                if cutlass.const_expr(FUSE_RELU):
                    amax_r = cute.arch.fmax(
                        amax_r, x
                    )  # For relu cases, we don't need abs since negative values will be 0 so they lose comparison automatically
                else:
                    amax_r = cute.arch.fmax(amax_r, fabs_f32(x))
        if cutlass.const_expr(FUSE_RELU):
            amax_r = cute.arch.fmax(amax_r, Float32(0.0))  # If relu, the amax is at least 0

    biased_exp_r = cvt_f32_to_fp8e8m0(amax_r * max_norm_rcp)

    # mS_row_stage has logical shape (32, 2) and we have 64 threads where each is mapped to one scale factor
    # The TV layout is equivalent to TV layout with thr_layout=(32, 2):(2, 1), val_layout=(1,)
    # but it's too trival so let's just index it directly without using layout
    # Note this is the logical layout, which is on top of the swizzled / non-swizzled scale factor layout
    # that mappes the logical index to the physical offset

    # For irregular shapes, skip the scale store if this thread's logical row / col-block lies past the input's actual extents.
    # TMA already zero-fills OOB input reads and drops OOB output writes; only the direct scale-byte gmem store needs an explicit guard.
    scale_row = tile_row_start + tidx // CTA_THREADS_X
    scale_col_first_elt = tile_col_start + (tidx % CTA_THREADS_X) * MXFP8_BLOCK_SCALING_SIZE
    if scale_row < M and scale_col_first_elt < N:
        mS_row_stage[(tidx // CTA_THREADS_X, tidx % CTA_THREADS_X)] = Uint8(biased_exp_r)

    inv_scale_r = exp2f_rcp(biased_exp_r)  # f32 reciprocal of the scale
    scale_2x = pack_f32x2(inv_scale_r, inv_scale_r)
    if cutlass.const_expr(_row_fast):
        mul_cvt_x4_func = mul_i64_cvt_packed16x4_to_fp8x4(DTYPE, FP8_DTYPE, FUSE_RELU)
    else:
        mul_cvt_x4_func = mul_i64_cvt_f32x4_to_fp8x4(FP8_DTYPE, FUSE_RELU)

    for w in cutlass.range_constexpr(WAVES):
        idx = (w * 4 + offset) % MXFP8_BLOCK_SCALING_SIZE
        idx = idx // 4
        if cutlass.const_expr(_row_fast):
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
    sX_tile,  # (TILE_Y, TILE_X) bf16/fp16 smem view, post-TMA
    sO_col_tile,  # (TILE_Y, TILE_X) uint8 smem view (colwise FP8 output)
    mS_col_stage,  # colwise scale tensor (1D swizzled, or 2D linear)
    max_norm_rcp,
    tile_row_start,  # Int32 — global row index of this stage's row 0
    # (= tile_idx_y * TILE_Y). Used to mask OOB scale stores
    # for irregular shapes.
    tile_col_start,  # Int32 — global col index of this CTA's col 0
    # (= bidx * TILE_X).
    M,
    N,  # Int32 — full tensor extents.
    ACTIVATION,
    DTYPE,
    FP8_DTYPE,
    SWIZZLE,
    TILE_X,
    TILE_Y,  # pylint: disable=unused-argument  # kept for API symmetry with the rowwise path
    WITH_ACT=False,  # forward: apply activation to the element
    WITH_DACT=False,  # backward: out = grad · act'(act_input)
    sA_tile=None,  # (TILE_Y, TILE_X) activation-input smem tile (dact only)
    WITH_DBIAS=False,  # also return this thread's column sum (pre-truncate)
    CACHE_ACTIVATION=False,  # overwrite sX_tile in place with the post-activation
    # (IType-truncated) values, so the rowwise pass can read
    # them instead of recomputing op
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

    USE_HALF_PRECISION = is_packed16(DTYPE) and ACTIVATION is None
    dbias_partial = Float32(0.0)

    if cutlass.const_expr(USE_HALF_PRECISION):
        kit = packed16_kit(DTYPE)
        # If we can use the half precision format, then use the input tile directly since there is no need to upcast
        sX_thread_i16 = cute.make_tensor(
            cute.recast_ptr(sX_thread.iterator, dtype=Int16),
            cute.make_layout((MXFP8_BLOCK_SCALING_SIZE,), stride=(TILE_X,)),
        )
        # Stash the strided column reads in registers (CUDA's in_colwise_IType):
        # the cvt loop below reuses them instead of re-reading smem.
        in_c = [None] * MXFP8_BLOCK_SCALING_SIZE
        amax_bits = Int16(0)
        for i in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE):
            in_c[i] = sX_thread_i16[i]
            amax_bits = kit.abs_max_scalar(amax_bits, in_c[i])
        amax_c = fabs_f32(kit.bits_to_f32(amax_bits))
    else:
        # Otherwise we need to case input values to fp32. Allocate the register tensor and load from SMEM input tiles.
        sX_thread_f32 = cute.make_rmem_tensor(
            layout_or_shape=cute.make_layout((MXFP8_BLOCK_SCALING_SIZE,), stride=(1,)),
            dtype=Float32,
        )
        for i in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE):
            sX_thread_f32[i] = Float32(sX_thread[i])
        # Apply activation (fwd) or grad·act'(act_input) (bwd dact) in f32.
        if cutlass.const_expr(WITH_DACT):
            dop = SUPPORTED_DACTIVATIONS[ACTIVATION]
            sA_thread = cute.composition(sA_tile, tv_layout)[tidx, None]
            for i in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE):
                sX_thread_f32[i] = sX_thread_f32[i] * dop(Float32(sA_thread[i]))
        elif cutlass.const_expr(WITH_ACT):
            op = SUPPORTED_ACTIVATIONS[ACTIVATION]
            for i in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE):
                sX_thread_f32[i] = op(sX_thread_f32[i])
        # Truncate the activation (after we apply op) back to the half precision type if input is also half precision.
        if cutlass.const_expr(is_packed16(DTYPE) and ACTIVATION is not None):
            kit_act = packed16_kit(DTYPE)
            for i in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE):
                sX_thread_f32[i] = kit_act.truncate_f32(sX_thread_f32[i])
        # Columnwise is the preferred direction so it runs first. If it needs to cache the activation in the input tile
        # to let the rowwise pass read it, we need to cast and overwrite the input data in-place here
        if cutlass.const_expr(CACHE_ACTIVATION):
            for i in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE):
                sX_thread[i] = DTYPE(sX_thread_f32[i])
        amax_c = Float32(0.0)
        for i in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE):
            amax_c = cute.arch.fmax(amax_c, fabs_f32(sX_thread_f32[i]))

    # Irregular shapes: skip when this stage's row range or this thread's
    # column lies past the input extents. TILE_Y == MXFP8_BLOCK_SCALING_SIZE so each stage
    # is exactly one scale-row; valid iff `tile_row_start < M`.
    biased_exp_c = cvt_f32_to_fp8e8m0(amax_c * max_norm_rcp)
    scale_col = tile_col_start + tidx
    if tile_row_start < M and scale_col < N:
        if cutlass.const_expr(SWIZZLE):
            mS_col_stage[(0, tidx % 32, tidx // 32)] = Uint8(biased_exp_c)
        else:
            mS_col_stage[(0, tidx)] = Uint8(biased_exp_c)

    inv_scale_c = exp2f_rcp(biased_exp_c)
    cvt_to_fp8_func = get_cvt_f32_to_fp8_func(FP8_DTYPE)
    if cutlass.const_expr(USE_HALF_PRECISION):
        kit_cast = packed16_kit(DTYPE)
        for i in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE):
            v_f32 = kit_cast.bits_to_f32(in_c[i])
            if cutlass.const_expr(WITH_DBIAS):
                dbias_partial += v_f32
            sO_thread[i] = Uint8(cvt_to_fp8_func(v_f32 * inv_scale_c))
    else:
        for i in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE):
            # Accumulate the per-thread column partial for dbias if WITH_DBIAS.
            if cutlass.const_expr(WITH_DBIAS):
                dbias_partial += sX_thread_f32[i]
            sO_thread[i] = Uint8(cvt_to_fp8_func(sX_thread_f32[i] * inv_scale_c))

    # Return this stage's per-column partial alongside amax; the caller accumulates
    # it across stages (a scalar can't be updated in-place through the arg).
    return amax_c, dbias_partial


@cute.jit
def quantize_bidimensional_mxfp8_swizzled(
    sX_tile: cute.Tensor,  # 32x64 input tile (already sliced to this stage), Sw<3,4,3> swizzled
    sO_row_tile: cute.Tensor,  # 32x64 rowwise-output tile
    sO_col_tile: cute.Tensor,  # 32x64 colwise-output tile
    sS_row_tile: cute.Tensor,  # (32, 2) smem rowwise-scale staging tile (flushed in the epilogue)
    sS_col_tile: cute.Tensor,  # (1, 64) smem colwise-scale staging tile (flushed in the epilogue)
    sColReduce_warp: cute.Tensor,  # (32,) SMEM fp32 columnwise scale reduction buffer for this warp
    WARPS_PER_CTA: cutlass.Constexpr,
    max_norm_rcp,
    DTYPE: cutlass.Constexpr,
    fp8_dtype: cutlass.Constexpr,
):
    """Quantize a pre-sliced 32x64 tile -> both rowwise and colwise MXFP8. Elements are
    addressed through TV layouts: a warp = 32 lanes = 32 rows (lane == row), and each
    thread owns one 32-col row segment (its 32-col block) and one output column.

    No bounds check: the caller only loops over valid tiles (`num_tiles`), and
    M % 32 == 0 means a tile can only be partial along N. The partial tile's
    out-of-bounds warp still runs, but harmlessly -- it reads TMA zero-filled smem,
    its output columns are masked by the caller's TMA store, and its scale writes
    (rowwise and colwise alike) go to staging slots whose flush targets are past-N
    padding columns of the respective scale tensors.
    """
    mul_cvt4 = mul_i64_cvt_f32x4_to_fp8x4(fp8_dtype)
    mul_cvt4_elemwise = mul_f32x4_cvt_f32x4_to_fp8x4(fp8_dtype)

    _, tv_layout = cute.make_layout_tv(
        thr_layout=cute.make_layout(((MXFP8_BLOCK_SCALING_SIZE, 1), WARPS_PER_CTA)),  # ((32, 1) 2)
        val_layout=cute.make_layout((1, MXFP8_BLOCK_SCALING_SIZE)),
    )
    _, tv_layout_rowwise_scale = cute.make_layout_tv(
        thr_layout=cute.make_layout(((MXFP8_BLOCK_SCALING_SIZE, 1), WARPS_PER_CTA)),  # ((32, 1) 2)
        val_layout=cute.make_layout((1, 1)),
    )
    _, tv_layout_colwise_scale = cute.make_layout_tv(
        thr_layout=cute.make_layout((1, (MXFP8_BLOCK_SCALING_SIZE, WARPS_PER_CTA))),  # (1, (32, 2))
        val_layout=cute.make_layout((1, 1)),
    )

    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx % MXFP8_BLOCK_SCALING_SIZE

    # Each composed [tidx, None] slice is 1-D (the value mode is flattened): the data
    # slices are size 32 (this thread's row segment), the scale slices size 1.
    tXsX = cute.composition(sX_tile, tv_layout)[tidx, None]  # (32,) input row segment
    tXsO_row = cute.composition(sO_row_tile, tv_layout)[tidx, None]  # (32,) rowwise out
    tXsO_col = cute.composition(sO_col_tile, tv_layout)[tidx, None]  # (32,) colwise out
    tSsS_row_tile = cute.composition(sS_row_tile, tv_layout_rowwise_scale)[tidx, None]  # (1,)
    tSsS_col_tile = cute.composition(sS_col_tile, tv_layout_colwise_scale)[tidx, None]  # (1,)

    rO_row = cute.make_rmem_tensor(MXFP8_BLOCK_SCALING_SIZE, Uint8)
    rO_col = cute.make_rmem_tensor(MXFP8_BLOCK_SCALING_SIZE, Uint8)
    rO_row_u32 = cute.make_tensor(
        cute.recast_ptr(rO_row.iterator, dtype=Uint32),
        cute.make_layout((MXFP8_BLOCK_SCALING_SIZE // 4,), stride=(1,)),
    )
    rO_col_u32 = cute.make_tensor(
        cute.recast_ptr(rO_col.iterator, dtype=Uint32),
        cute.make_layout((MXFP8_BLOCK_SCALING_SIZE // 4,), stride=(1,)),
    )
    sColReduce = cute.make_tensor(
        sColReduce_warp.iterator,
        cute.make_layout((MXFP8_BLOCK_SCALING_SIZE // 4, 4), stride=(4, 1)),
    )

    if cutlass.const_expr(is_packed16(DTYPE)):
        # If the input is bf16 / fp16, take this fast path and process 2 elements at a time in a packed i32
        kit = packed16_kit(DTYPE)
        rX = cute.make_rmem_tensor(MXFP8_BLOCK_SCALING_SIZE, DTYPE)
        # Do a vectorized load from SMEM to RMEM and unswizzle in the meantime.
        cute.autovec_copy(tXsX, rX)
        rX_2x = cute.make_tensor(
            cute.recast_ptr(rX.iterator, dtype=Int32),
            cute.make_layout((MXFP8_BLOCK_SCALING_SIZE // 2,), stride=(1,)),
        )
        sColReduce_2x = cute.make_tensor(
            cute.recast_ptr(sColReduce.iterator, dtype=Int64),
            cute.make_layout((MXFP8_BLOCK_SCALING_SIZE // 2,), stride=(1,)),
        )

        row_amax2 = Int32(0)
        for i in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE // 2):
            pair = rX_2x[i]
            row_amax2 = kit.abs_max_x2(row_amax2, pair)
            a_lo = fabs_f32(kit.x2_lo_to_f32(pair))
            a_hi = fabs_f32(kit.x2_hi_to_f32(pair))
            col_lo = cute.arch.warp_redux_sync(a_lo, kind="fmax")
            col_hi = cute.arch.warp_redux_sync(a_hi, kind="fmax")
            with cute.arch.elect_one():
                sColReduce_2x[i] = pack_f32x2(col_lo, col_hi)

        # Compute the rowwise scale factor
        row_amax = cute.arch.fmax(
            fabs_f32(kit.x2_lo_to_f32(row_amax2)), fabs_f32(kit.x2_hi_to_f32(row_amax2))
        )
        row_exp = cvt_f32_to_fp8e8m0(row_amax * max_norm_rcp)
        row_inv = exp2f_rcp(row_exp)
        tSsS_row_tile[0] = Uint8(row_exp)
        cute.arch.sync_warp()

        # Compute the colwise scale factor (only handle the one that belongs to this thread / lane)
        col_exp = cvt_f32_to_fp8e8m0(sColReduce_warp[lane] * max_norm_rcp)
        tSsS_col_tile[0] = Uint8(col_exp)
        sColReduce_warp[lane] = exp2f_rcp(col_exp)
        cute.arch.sync_warp()

        row_scale_2x = pack_f32x2(row_inv, row_inv)
        # Vectorized multiply-and-convert: 4 f32 → 4 fp8
        for j in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE // 4):
            # Vectorized load for 4 columnwise scale factors from SMEM to RMEM
            col_inv4 = cute.make_rmem_tensor(4, Float32)
            cute.autovec_copy(sColReduce[j, None], col_inv4)  # LDS.128, warp-broadcast
            p01 = rX_2x[2 * j]
            p23 = rX_2x[2 * j + 1]
            f0 = kit.x2_lo_to_f32(p01)
            f1 = kit.x2_hi_to_f32(p01)
            f2 = kit.x2_lo_to_f32(p23)
            f3 = kit.x2_hi_to_f32(p23)
            # For rowwise quantized values, they use the same scale for all 4 elements,
            # so we can just pass two row_inv to mul_cvt4 to apply it to all 4 elements at once.
            rO_row_u32[j] = mul_cvt4(f0, f1, f2, f3, row_scale_2x)
            # For columnwise quantized values, each element has its own scale; the
            # elementwise variant fuses the per-element multiply into the cvt sequence.
            rO_col_u32[j] = mul_cvt4_elemwise(
                f0, f1, f2, f3, col_inv4[0], col_inv4[1], col_inv4[2], col_inv4[3]
            )
        cute.autovec_copy(rO_row, tXsO_row)
        cute.autovec_copy(rO_col, tXsO_col)
    else:
        # If input is fp32, take this slow path and process it normally without packing
        rX = cute.make_rmem_tensor(MXFP8_BLOCK_SCALING_SIZE, Float32)
        cute.autovec_copy(tXsX, rX)

        row_amax = Float32(0.0)
        for c in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE):
            a = fabs_f32(rX[c])
            row_amax = cute.arch.fmax(row_amax, a)
            col_amax = cute.arch.warp_redux_sync(a, kind="fmax")
            with cute.arch.elect_one():
                sColReduce_warp[c] = col_amax

        # Compute the rowwise scale factor
        row_exp = cvt_f32_to_fp8e8m0(row_amax * max_norm_rcp)
        row_inv = exp2f_rcp(row_exp)
        tSsS_row_tile[0] = Uint8(row_exp)  # rowwise scale (this row-block, staged in smem)
        cute.arch.sync_warp()

        # Compute the colwise scale factor (only handle the one that belongs to this thread / lane)
        col_exp = cvt_f32_to_fp8e8m0(sColReduce_warp[lane] * max_norm_rcp)
        tSsS_col_tile[0] = Uint8(col_exp)  # colwise scale (this thread's column)
        sColReduce_warp[lane] = exp2f_rcp(col_exp)
        cute.arch.sync_warp()

        row_scale_2x = pack_f32x2(row_inv, row_inv)
        # Vectorized multiply-and-convert: 4 f32 → 4 fp8
        for j in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE // 4):
            # Vectorized load for 4 columnwise scale factors from SMEM to RMEM
            col_inv4 = cute.make_rmem_tensor(4, Float32)
            cute.autovec_copy(sColReduce[j, None], col_inv4)
            offset = 4 * j
            # For rowwise quantized values, they use the same scale for all 4 elements,
            # so we can just pass two row_inv to mul_cvt4 to apply it to all 4 elements at once.
            rO_row_u32[j] = mul_cvt4(
                rX[offset], rX[offset + 1], rX[offset + 2], rX[offset + 3], row_scale_2x
            )
            # For columnwise quantized values, each element has its own scale; the
            # elementwise variant fuses the per-element multiply into the cvt sequence.
            rO_col_u32[j] = mul_cvt4_elemwise(
                rX[offset],
                rX[offset + 1],
                rX[offset + 2],
                rX[offset + 3],
                col_inv4[0],
                col_inv4[1],
                col_inv4[2],
                col_inv4[3],
            )
        cute.autovec_copy(rO_row, tXsO_row)
        cute.autovec_copy(rO_col, tXsO_col)


class MXFP8QuantizeConfig:
    """Configs for the compiled CuTeDSL kernel. These will be fixed once the kernel is compiled and
    they will behave as const expressions.
    """

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
        activation: Optional[str] = None,
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
                raise ValueError(
                    "activation must be none when with_dact and with_act are both False"
                )
        else:
            if with_dact and with_act:
                raise ValueError(
                    "with_dact and with_act cannot be true at the same time since they are used for"
                    " different paths (bwd vs fwd)"
                )
            if with_dact:
                if activation in SUPPORTED_DACTIVATIONS:
                    self.ACTIVATION = activation
                else:
                    raise ValueError(
                        f"unknown activation {activation!r} for with_dact=True; expected one of"
                        f" {sorted(SUPPORTED_DACTIVATIONS)}"
                    )
            elif with_act:
                if activation in SUPPORTED_ACTIVATIONS:
                    self.ACTIVATION = activation
                else:
                    raise ValueError(
                        f"unknown activation {activation!r} for with_act=True; expected one of"
                        f" {sorted(SUPPORTED_ACTIVATIONS)}"
                    )
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
        return (
            f"MXFP8QuantizeConfig(dtype={self.DTYPE_STR}, fp8_dtype={self.FP8_DTYPE}, "
            f"rowwise={self.ROWWISE}, colwise={self.COLWISE}, "
            f"swizzled={self.WITH_GEMM_SWIZZLED_SCALES}, with_amax={self.WITH_AMAX}, "
            f"with_dbias={self.WITH_DBIAS}, with_dact={self.WITH_DACT}, "
            f"with_act={self.WITH_ACT}, with_noop={self.WITH_NOOP}, "
            f"activation={self.ACTIVATION})"
        )

    __repr__ = __str__


class MXFP8QuantizeKernelBase(abc.ABC):
    """Base class for MXFP8 quantize kernels."""

    def __init__(self, cfg: MXFP8QuantizeConfig):
        """Initialize the kernel with the given configuration and optional tunable parameters."""
        self.cfg = cfg

    def override_tuneable_configs(self, tuneable_cfgs: Optional[dict] = None):
        """Set the tunable configs as attributes of the kernel instance."""
        for name, value in (tuneable_cfgs or {}).items():
            setattr(self, name, value)

    @abc.abstractmethod
    def __call__(
        self,
        mX: cute.Tensor,
        mO_row: Optional[cute.Tensor],
        mS_row: Optional[cute.Tensor],
        mO_col: Optional[cute.Tensor],
        mS_col: Optional[cute.Tensor],
        mAmax: Optional[cute.Tensor],
        mNoop: Optional[cute.Tensor],
        mDActInput: Optional[cute.Tensor],
        mWorkspace: Optional[cute.Tensor],
        stream: CUstream,
    ):
        """
        Compiled kernel entrypoint (decorate with @cute.jit).
        All MXFP8 quantize kernels must implement this interface because this is our C++ call site's contract in `quantize_mxfp8_cutedsl.cuh`.
        C++ call site will pass the arguments in this exact order via tvm-ffi and the kernel must accept them even if they are not used.
        """


class MXFP8QuantizeKernel(MXFP8QuantizeKernelBase):
    """The MXFP8 quantization kernel that mirrors the standard (non-specialized) MXFP8 CUDA C++ quantization kernel
    with multiple fusions (activation, dbias, etc.).
    `__call__` method is the entrypoint which is AOT compiled. `self` will be captured so it's fixed per compiled kernel
    """

    # Vectorised access constants for bank-conflict avoidance (rowwise pass)
    _PACK_SIZE = 4  # Elements per vector load
    # Each thread reads 8 waves with each wave reads 4 packed bf16, so it reads a whole MXFP8 block in total
    _WAVES = MXFP8_BLOCK_SCALING_SIZE // _PACK_SIZE
    _TOTAL_BANKS_WIDTH = (32 * 4) // 1  # 32 banks × 4 bytes, in bytes (uint8 stride)
    _THREADS_PER_BANK = _TOTAL_BANKS_WIDTH // MXFP8_BLOCK_SCALING_SIZE  # 4 threads per bank

    def __init__(self, cfg):
        super().__init__(cfg)
        # We prefer to do dbias reduction in colwise which is easier (no cross-thread reduction needed).
        # Only do rowwise reduction when we don't quantize columnwisely when WITH_DBIAS is True.
        self.DBIAS_REDUCTION_COLWISE = cfg.WITH_DBIAS and cfg.COLWISE
        self.DBIAS_REDUCTION_ROWWISE = cfg.WITH_DBIAS and not cfg.COLWISE
        # Cache activation in-place in the SMEM input tile when we process both rowwise and colwise passes
        # so the activation is only computed once in the direction we favor (columnwise) and the other direction (rowwise)
        # reads the cached value instead of recomputing it.
        # Note: if activation is relu, there is no standalong relu applied because it's already fused into `cvt.rn.satfinite`
        # so it should be treated as "no activation"
        self.CACHE_ACTIVATION = (
            (cfg.WITH_ACT or cfg.WITH_DACT)
            and cfg.ROWWISE
            and cfg.COLWISE
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
        mNoop: Optional[cute.Tensor],  # 1-element cast_noop flag, only used when WITH_NOOP is True
        mDActInput: Optional[
            cute.Tensor
        ],  # Activation input for activation derivative fusion, only used when WITH_DACT is True
        mWorkspace: Optional[
            cute.Tensor
        ],  # Workspace for the dbias reduction, only used when WITH_DBIAS is True
        stream: CUstream,
        # This kernel allows these parameters to be tuned for performance.
        TUNEABLE_CFGS: cutlass.Constexpr = {
            "_NUM_STAGES": 2,
            "_NUM_TILES_STANDARD": 2,
            "_NUM_TILES_DBIAS_ONLY": 4,
            "_THREADS_PER_CTA_STANDARD": 64,
            "_THREADS_PER_CTA_DBIAS_ONLY": 128,
        },
    ):
        if cutlass.const_expr(CUTEDSL_DEBUG_LOGGING):
            cute.printf(f"[CuTeDSL] MXFP8QuantizeKernel.__call__() with config: {self.cfg}\n")

        self.override_tuneable_configs(TUNEABLE_CFGS)
        cfg = self.cfg
        # Cast + dbias with no activation gets the larger tile (CUDA CAST_DBIAS_ONLY).
        cast_dbias_only = cfg.WITH_DBIAS and not cfg.WITH_DACT and not cfg.WITH_ACT
        # Use a different tile size for dbias only config
        # No matter what tile size we use, each thread always handles a (1, MXFP8_BLOCK_SCALING_SIZE) chunk
        if cutlass.const_expr(cast_dbias_only):
            self._NUM_TILES = (
                self._NUM_TILES_DBIAS_ONLY
            )  # Each CTA handles 4 tiles stacked vertically
            self._THREADS_PER_CTA = self._THREADS_PER_CTA_DBIAS_ONLY
        else:
            self._NUM_TILES = (
                self._NUM_TILES_STANDARD
            )  # Each CTA handles 2 tiles stacked vertically
            self._THREADS_PER_CTA = self._THREADS_PER_CTA_STANDARD
        # Each thread handles a (1, MXFP8_BLOCK_SCALING_SIZE) chunk
        self._TILE_COLS = self._THREADS_PER_CTA
        self._TILE_ROWS = MXFP8_BLOCK_SCALING_SIZE
        self._NUM_WARPS = self._THREADS_PER_CTA // 32

        M = mX.shape[0]
        N = mX.shape[1]
        max_norm_rcp = cfg.MAX_NORM_RCP

        # The FFI boundary carries native FP8/E8M0 dtypes; the kernel works on bytes.
        if cutlass.const_expr(cfg.ROWWISE):
            mO_row = as_byte_tensor(mO_row)
            mS_row = as_byte_tensor(mS_row)
        if cutlass.const_expr(cfg.COLWISE):
            mO_col = as_byte_tensor(mO_col)
            mS_col = as_byte_tensor(mS_col)

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
            max_norm_rcp,
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
        mX,
        mS_row,
        mS_col,
        mAmax,
        mNoop,
        mWorkspace,
        max_norm_rcp,
        dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
        tma_atom,
        tma_src,  # Input TMA atoms
        tma_atom_out_row,
        tma_dst_out_row,  # Rowwise output TMA atoms
        tma_atom_out_col,
        tma_dst_out_col,  # Colwise output TMA atoms
        tma_atom_act,
        tma_src_act,  # Activation derivative TMA atoms, or None if WITH_DACT is False
    ):
        """Device entry: no-op the CTA when the noop flag is set, else run the quantize main loop."""
        cfg = self.cfg
        # Only check the noop flag when WITH_NOOP is True (the noop tensor is passed and it's not nullptr)
        # and both WITH_ACT and WITH_DACT are False (it's legitimate to skip the quantization because no fusion is involved)
        CHECK_NOOP_FLAG: cutlass.const_expr = (
            cfg.WITH_NOOP and not cfg.WITH_ACT and not cfg.WITH_DACT
        )
        # Only perform the runtime check to read the noop flag's value when the compiled kernel allows us to do so
        skip_execution = cutlass.const_expr(CHECK_NOOP_FLAG) and mNoop[0] == Float32(1.0)
        if not skip_execution:
            self._kernel_main(
                mX,
                mS_row,
                mS_col,
                mAmax,
                mWorkspace,
                max_norm_rcp,
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
        mX,
        mS_row,
        mS_col,
        mAmax,
        mWorkspace,
        max_norm_rcp,
        dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
        tma_atom,
        tma_src,  # Input TMA atoms
        tma_atom_out_row,
        tma_dst_out_row,  # Rowwise output TMA atoms
        tma_atom_out_col,
        tma_dst_out_col,  # Colwise output TMA atoms
        tma_atom_act,
        tma_src_act,  # Activation derivative TMA atoms, or None if WITH_DACT is False
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
                        Uint8, self._TILE_ROWS * self._TILE_COLS * self._NUM_STAGES
                    ],
                    128,
                ]
                sO_col: cute.struct.Align[
                    cute.struct.MemRange[
                        Uint8, self._TILE_ROWS * self._TILE_COLS * self._NUM_STAGES
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
                        Uint8, self._TILE_ROWS * self._TILE_COLS * self._NUM_STAGES
                    ],
                    128,
                ]
                sAmax: cute.struct.MemRange[Float32, self._NUM_WARPS]

        elif cutlass.const_expr(cfg.ROWWISE):

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
                        Uint8, self._TILE_ROWS * self._TILE_COLS * self._NUM_STAGES
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
                        Uint8, self._TILE_ROWS * self._TILE_COLS * self._NUM_STAGES
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
        if cutlass.const_expr(self.DBIAS_REDUCTION_ROWWISE):
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
        if cutlass.const_expr(self.DBIAS_REDUCTION_COLWISE):
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
                    max_norm_rcp,
                    tile_idx_y * self._TILE_ROWS,
                    bidx * self._TILE_COLS,
                    M,
                    N,
                    sActInput_tile,
                )
                if cutlass.const_expr(self.AMAX_FROM_COLWISE):
                    per_thread_amax = cute.arch.fmax(per_thread_amax, amax_c)
                if cutlass.const_expr(self.DBIAS_REDUCTION_COLWISE):
                    block_dbias += dbias_c
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
                    max_norm_rcp,
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
        if cutlass.const_expr(self.DBIAS_REDUCTION_ROWWISE):
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
    def _dbias_reduction_rowwise_epilouge(self, smem, tidx, rowwise_dbias_acc):
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
    def _amax_epilogue(self, sAmax, mAmax, tidx, warp_idx, per_thread_amax):
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
                _bitcast_f32_to_i32(cta_amax),
            )

    @cute.jit
    def _process_rowwise(
        self,
        sX_tile,  # (TILE_Y, TILE_X) bf16/fp16 smem view, post-TMA
        sO_row_tile,  # (TILE_Y, TILE_X) uint8 smem view (rowwise FP8 output)
        mS_row_stage,  # rowwise scale tensor (1D swizzled, or 2D linear)
        max_norm_rcp,
        tile_row_start,  # Int32 — global row of this stage's row 0
        tile_col_start,  # Int32 — global col of this CTA's col 0
        M,
        N,  # Int32 — full input extents, for OOB masking
        sActInput_tile=None,  # (TILE_Y, TILE_X) act_input tile (dact only)
        dbias_acc=None,  # rmem Float32[MXFP8_BLOCK_SCALING_SIZE] dbias accumulator (rowwise-only dbias)
    ):
        cfg = self.cfg
        return quantize_rowwise_mxfp8(
            sX_tile,
            None if self.CACHE_ACTIVATION else sActInput_tile,
            sO_row_tile,
            mS_row_stage,
            max_norm_rcp,
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
            WITH_ACT=cfg.WITH_ACT and not self.CACHE_ACTIVATION,
            WITH_DACT=cfg.WITH_DACT and not self.CACHE_ACTIVATION,
            WITH_DBIAS=self.DBIAS_REDUCTION_ROWWISE,
            dbias_acc=dbias_acc,
        )

    @cute.jit
    def _process_colwise(
        self,
        sX_tile,  # (TILE_Y, TILE_X) bf16/fp16 smem view, post-TMA
        sO_col_tile,  # (TILE_Y, TILE_X) uint8 smem view (colwise FP8 output)
        mS_col_stage,  # colwise scale tensor (1D swizzled, or 2D linear)
        max_norm_rcp,
        tile_row_start,  # Int32 — global row of this stage's row 0
        tile_col_start,  # Int32 — global col of this CTA's col 0
        M,
        N,  # Int32 — full input extents, for OOB masking
        sActInput_tile=None,  # (TILE_Y, TILE_X) act_input tile (dact only)
    ):
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
            TILE_X=self._TILE_COLS,
            TILE_Y=self._TILE_ROWS,
            WITH_ACT=cfg.WITH_ACT,
            WITH_DACT=cfg.WITH_DACT,
            sA_tile=sActInput_tile,
            WITH_DBIAS=self.DBIAS_REDUCTION_COLWISE,
            CACHE_ACTIVATION=self.CACHE_ACTIVATION,
        )


class MXFP8QuantizeSpecializedRowwiseKernel(MXFP8QuantizeKernelBase):
    """Specialized cast-only ROWWISE-only MXFP8 kernel. Requires N % 128 == 0 (full vectorizable column chunks).

    Plain rowwise-only quantize. Each thread owns one 32-element MXFP8 chunk and
    uses vectorized global loads/stores (no TMA used)."""

    def __init__(self, cfg):
        super().__init__(cfg)

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mO_row: Optional[cute.Tensor],
        mS_row: Optional[cute.Tensor],
        mO_col: Optional[cute.Tensor],
        mS_col: Optional[cute.Tensor],  # Unused, kept for API compatibility
        mAmax: Optional[cute.Tensor],  # Unused, kept for API compatibility
        mNoop: Optional[cute.Tensor],  # Unused, kept for API compatibility
        mDActInput: Optional[cute.Tensor],  # Unused, kept for API compatibility
        mWorkspace: Optional[cute.Tensor],  # Unused, kept for API compatibility
        stream: CUstream,
        # This kernel allows these parameters to be tuned for performance.
        TUNEABLE_CFGS: cutlass.Constexpr = {
            "_TILE_ROWS": 4,
            "_TILE_COLS": 1024,
            # If True, then this kernel will first write each thread's scale byte to a shared
            # memory buffer, then utilize vectorized store to flush the buffer to global memory.
            "_STASH_SCALE_TO_SMEM": True,
        },
    ):
        if cutlass.const_expr(CUTEDSL_DEBUG_LOGGING):
            cute.printf(
                "[CuTeDSL] MXFP8QuantizeSpecializedRowwiseKernel.__call__() with config:"
                f" {self.cfg}\n"
            )

        super().override_tuneable_configs(TUNEABLE_CFGS)
        # One thread per 32 elements scale block.
        self._THREADS_PER_CTA = self._TILE_ROWS * self._TILE_COLS // MXFP8_BLOCK_SCALING_SIZE

        M = mX.shape[0]
        N = mX.shape[1]

        # The FFI boundary carries native FP8/E8M0 dtypes; the kernel works on bytes.
        mO_row = as_byte_tensor(mO_row)
        mS_row = as_byte_tensor(mS_row)

        grid = [
            cute.ceil_div(Int32(N), self._TILE_COLS),
            cute.ceil_div(M, self._TILE_ROWS),
        ]
        block = [self._THREADS_PER_CTA]

        self.kernel(
            mX,
            mO_row,
            mS_row,
            self.cfg.MAX_NORM_RCP,
            mX.element_type,
        ).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def kernel(self, mX, mO_row, mS_row, max_norm_rcp, DTYPE):
        """Device entry for the specialized rowwise-only cast kernel (vectorized global loads/stores, no TMA)."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        M = mX.shape[0]
        N = mX.shape[1]

        # Each thread handles one 32-element MXFP8 chunk (= one scale block).
        # The 128 threads in the CTA are grouped as (4, 32), so they cover a
        # (4, 1024) input tile and the matching (4, 32) scale tile.
        CTA_Y = self._TILE_ROWS
        CTA_X = self._TILE_COLS // MXFP8_BLOCK_SCALING_SIZE
        tiler, tv_layout = cute.make_layout_tv(
            thr_layout=cute.make_layout((CTA_Y, CTA_X), stride=(CTA_X, 1)),
            val_layout=cute.make_layout(
                (1, MXFP8_BLOCK_SCALING_SIZE), stride=(MXFP8_BLOCK_SCALING_SIZE, 1)
            ),
        )
        tiler_scale, tv_layout_scale = cute.make_layout_tv(
            thr_layout=cute.make_layout((CTA_Y, CTA_X), stride=(CTA_X, 1)),
            val_layout=cute.make_layout((1, 1), stride=(1, 1)),
        )

        # Select the tile that belongs to this CTA, then the fragment per thread.
        mX_tile = cute.local_tile(mX, tiler, (bidy, bidx))
        mO_tile = cute.local_tile(mO_row, tiler, (bidy, bidx))
        mS_tile = cute.local_tile(mS_row, tiler_scale, (bidy, bidx))
        mX_thread = cute.composition(mX_tile, tv_layout)[tidx, None]
        mO_thread = cute.composition(mO_tile, tv_layout)[tidx, None]
        mS_thread = cute.composition(mS_tile, tv_layout_scale)[tidx, None]

        rX_thread = cute.make_rmem_tensor(
            cute.make_layout((1, MXFP8_BLOCK_SCALING_SIZE), stride=(MXFP8_BLOCK_SCALING_SIZE, 1)),
            dtype=DTYPE,
        )
        # Inputs widened to f32 once (reused by amax and the fused cvt). The FP8
        # output stays a uint8 fragment; we write it through a uint32 view so the
        # 4-wide mul_cvt drops one packed word per call (see the cvt loop).
        rX_f32 = cute.make_rmem_tensor(
            cute.make_layout((1, MXFP8_BLOCK_SCALING_SIZE), stride=(MXFP8_BLOCK_SCALING_SIZE, 1)),
            dtype=Float32,
        )
        rO_thread = cute.make_rmem_tensor(
            cute.make_layout((1, MXFP8_BLOCK_SCALING_SIZE), stride=(MXFP8_BLOCK_SCALING_SIZE, 1)),
            dtype=Uint8,
        )
        rO_u32 = cute.make_tensor(
            cute.recast_ptr(rO_thread.iterator, dtype=Uint32),
            cute.make_layout(
                (MXFP8_BLOCK_SCALING_SIZE // 4,), stride=(1,)
            ),  # Unit is Uint32, divide by 4 here
        )

        sS_thread = None
        if cutlass.const_expr(self._STASH_SCALE_TO_SMEM):

            @cute.struct
            class SharedStorage:
                buf: cute.struct.Align[cute.struct.MemRange[Uint8, CTA_Y * CTA_X], 16]

            storage = cutlass.utils.SmemAllocator().allocate(SharedStorage)
            sScale = storage.buf.get_tensor(cute.make_layout((CTA_Y, CTA_X), stride=(CTA_X, 1)))
            # sScale is (CTA_Y, CTA_X):(CTA_X, 1), which is the same layout as tv_layout_scale
            # so sS_thread is really just an 1 Uint8 buffer for this thread's scale byte.
            sS_thread = cute.composition(sScale, tv_layout_scale)[tidx, None]
            # Zero first so padding columns (cols past N/32 in the padded scale
            # matrix) flush as 0 and we never read uninitialized smem.
            sS_thread[0] = Uint8(0)
            cute.arch.sync_threads()

        row = bidy * self._TILE_ROWS + tidx // CTA_X
        col = bidx * self._TILE_COLS + (tidx % CTA_X) * MXFP8_BLOCK_SCALING_SIZE
        if row < M and col < N:
            cute.autovec_copy(mX_thread, rX_thread)

            # Widen once and reduce. bf16/fp16 -> f32 widening is exact, so the
            # amax matches the CUDA 16-bit abs_max path bit-for-bit.
            amax = Float32(0.0)
            for i in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE):
                rX_f32[0, i] = Float32(rX_thread[0, i])
                amax = cute.arch.fmax(amax, fabs_f32(rX_f32[0, i]))

            biased_exp = cvt_f32_to_fp8e8m0(amax * max_norm_rcp)
            if cutlass.const_expr(self._STASH_SCALE_TO_SMEM):
                sS_thread[0] = Uint8(biased_exp)
            else:
                mS_thread[0] = Uint8(biased_exp)

            # Rescale + FP8 cast, 4 elements per fused mul_cvt (one uint32 out),
            # then a vectorized store. Mirrors CUDA's _use_cvt_4x path.
            inv_scale = exp2f_rcp(biased_exp)
            scale_2x = pack_f32x2(inv_scale, inv_scale)
            mul_cvt4 = mul_i64_cvt_f32x4_to_fp8x4(self.cfg.FP8_DTYPE)
            for i in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE // 4):
                offset = 4 * i
                rO_u32[i] = mul_cvt4(
                    rX_f32[0, offset],
                    rX_f32[0, offset + 1],
                    rX_f32[0, offset + 2],
                    rX_f32[0, offset + 3],
                    scale_2x,
                )
            cute.autovec_copy(rO_thread, mO_thread)

        # Cooperative wide flush of the staged scales where padding columns flush as 0.
        if cutlass.const_expr(self._STASH_SCALE_TO_SMEM):
            cute.arch.sync_threads()
            padded_cols = mS_row.shape[1]
            if padded_cols % 16 == 0:
                # If columns is divisible by 16, use 16 bytes as the vectorized store width
                self._flush_scales_to_gmem(sScale, mS_tile, tidx, bidx, bidy, M, padded_cols, 16)
            else:
                # Otherwise use 4 bytes as the vectorized store width.
                # Note our fake tensor requires 4 divisibility so this is enforced as long as you can get here
                self._flush_scales_to_gmem(sScale, mS_tile, tidx, bidx, bidy, M, padded_cols, 4)

    @cute.jit
    def _flush_scales_to_gmem(self, sScale, mS_tile, tidx, bidx, bidy, M, padded_cols, width):
        """Flush the staged (CTA_Y, CTA_X) scale tile to gmem with vectorized stores."""
        CTA_Y = self._TILE_ROWS
        CTA_X = self._TILE_COLS // MXFP8_BLOCK_SCALING_SIZE
        # Previously each threads has 1 byte, but now we are doing vectorized store,
        # which means only a subset of threads will need to issue the store while other threads are not used.
        active_threads = CTA_X // width
        _, tv_flush = cute.make_layout_tv(
            thr_layout=cute.make_layout((CTA_Y, active_threads), stride=(active_threads, 1)),
            val_layout=cute.make_layout((1, width), stride=(width, 1)),
        )
        # We only need to use a subset of threads with shape (CTA_Y, active_threads) to write
        # so if the thread is outside of this subset, it will remain inactive
        if tidx < CTA_Y * active_threads:
            # Absolute position of the scale vector to write in the GMEM buffer
            thread_y = bidy * CTA_Y + tidx // active_threads
            thread_x = bidx * CTA_X + (tidx % active_threads) * width
            if thread_y < M and thread_x < padded_cols:
                cute.autovec_copy(
                    cute.composition(sScale, tv_flush)[tidx, None],
                    cute.composition(mS_tile, tv_flush)[tidx, None],
                )


class MXFP8QuantizeSpecializedBidimensionalKernel(MXFP8QuantizeKernelBase):
    """Specialized cast-only rowwise+colwise MXFP8 kernel (swizzled TMA, one 32x32 tile per warp)."""

    _WARPS_PER_CTA = 2
    _THREADS_PER_CTA = _WARPS_PER_CTA * THREADS_PER_WARP
    # A warp handles a 32x32 tile
    _WARP_ROWS = MXFP8_BLOCK_SCALING_SIZE
    _WARP_COLS = MXFP8_BLOCK_SCALING_SIZE
    # A CTA handles a tile consisted two 32x32 warp subtiles side-by-side at a time, each handled by a warp
    _TILE_ROWS = _WARP_ROWS
    _TILE_COLS = _WARP_COLS * _WARPS_PER_CTA

    # Rows and columns for the rowwise / columnwise scale factor tensor
    _SCALE_ROWS = _TILE_ROWS // MXFP8_BLOCK_SCALING_SIZE
    _SCALE_COLS = _TILE_COLS // MXFP8_BLOCK_SCALING_SIZE

    def __init__(self, cfg):
        super().__init__(cfg)

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mO_row: Optional[cute.Tensor],
        mS_row: Optional[cute.Tensor],
        mO_col: Optional[cute.Tensor],
        mS_col: Optional[cute.Tensor],
        mAmax: Optional[cute.Tensor],
        mNoop: Optional[cute.Tensor],
        mDActInput: Optional[cute.Tensor],
        mWorkspace: Optional[cute.Tensor],
        stream: CUstream,
        # This kernel allows these parameters to be tuned for performance.
        # Constexpr: baked at compile time; an unannotated dict would be
        # flattened into runtime scalar arguments instead.
        TUNEABLE_CFGS: cutlass.Constexpr = {
            "_NUM_TILES_X": 4,
            "_NUM_TILES_Y": 1,
            "_NUM_STAGES": 2,
        },
    ):
        if cutlass.const_expr(CUTEDSL_DEBUG_LOGGING):
            cute.printf(
                "[CuTeDSL] MXFP8QuantizeSpecializedBidimensionalKernel.__call__() with config:"
                f" {self.cfg}\n"
            )

        super().override_tuneable_configs(TUNEABLE_CFGS)
        self._NUM_TILES = self._NUM_TILES_X * self._NUM_TILES_Y

        M = mX.shape[0]
        N = mX.shape[1]

        # The FFI boundary carries native FP8/E8M0 dtypes; the kernel works on bytes.
        mO_row = as_byte_tensor(mO_row)
        mS_row = as_byte_tensor(mS_row)
        mO_col = as_byte_tensor(mO_col)
        mS_col = as_byte_tensor(mS_col)

        smem_tile_layout = cute.make_ordered_layout(
            (self._TILE_ROWS, self._TILE_COLS), order=(1, 0)
        )
        cta_tiler = (self._TILE_ROWS, self._TILE_COLS)
        # Apply 128B input Swizzle<3,4,3> to input tiles
        in_smem_layout = cute.make_composed_layout(cute.make_swizzle(3, 4, 3), 0, smem_tile_layout)
        # Apply 64B output swizzle<2,4,3> for output tiles
        out_smem_layout = cute.make_composed_layout(cute.make_swizzle(2, 4, 3), 0, smem_tile_layout)
        op_load = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        op_store = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()

        # Input TMA atom
        tma_atom, tma_src = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op_load,
            mX,
            in_smem_layout,
            cta_tiler,
            num_multicast=1,
        )
        # Rowwise output TMA atoms
        tma_atom_out_row, tma_dst_out_row = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op_store,
            mO_row,
            out_smem_layout,
            cta_tiler,
            num_multicast=1,
        )
        # Colwise output TMA atoms
        tma_atom_out_col, tma_dst_out_col = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op_store,
            mO_col,
            out_smem_layout,
            cta_tiler,
            num_multicast=1,
        )

        grid = [
            # Each CTA covers a (TILE_ROWS * NUM_TILES_Y, TILE_COLS * NUM_TILES_X) area.
            cute.ceil_div(Int32(N), self._TILE_COLS * self._NUM_TILES_X),
            cute.ceil_div(M, self._TILE_ROWS * self._NUM_TILES_Y),
        ]
        block = [self._THREADS_PER_CTA]
        self.kernel(
            mX,
            mS_row,
            mS_col,
            self.cfg.MAX_NORM_RCP,
            mX.element_type,
            tma_atom,
            tma_src,
            tma_atom_out_row,
            tma_dst_out_row,
            tma_atom_out_col,
            tma_dst_out_col,
        ).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def kernel(
        self,
        mX,
        mS_row,
        mS_col,
        max_norm_rcp,
        dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
        tma_atom,
        tma_src,
        tma_atom_out_row,
        tma_dst_out_row,
        tma_atom_out_col,
        tma_dst_out_col,
    ):
        """Device entry for the specialized bidimensional (rowwise+colwise) cast kernel."""

        bidx, bidy, _ = cute.arch.block_idx()

        M = mX.shape[0]
        N = mX.shape[1]

        # The CTA owns a NUM_TILES_Y x NUM_TILES_X arrangement of tiles, starting here.
        tile_x_base = bidx * self._NUM_TILES_X
        tile_y_base = bidy * self._NUM_TILES_Y

        num_tiles_x = cutlass.min(
            self._NUM_TILES_X,
            # Valid 64-col tiles remaining from this CTA's start column (bidx * span).
            cute.ceil_div(Int32(N) - bidx * self._TILE_COLS * self._NUM_TILES_X, self._TILE_COLS),
        )
        num_tiles_y = cutlass.min(
            self._NUM_TILES_Y,
            # Valid 32-row tiles remaining from this CTA's start row (bidy * span).
            cute.ceil_div(Int32(M) - bidy * self._TILE_ROWS * self._NUM_TILES_Y, self._TILE_ROWS),
        )
        num_tiles = num_tiles_x * num_tiles_y

        @cute.struct
        class SharedStorage:
            mbar_storage: cute.struct.MemRange[
                cute.Int64, 2 * self._NUM_STAGES
            ]  # (full,empty) per stage
            sX: cute.struct.Align[
                cute.struct.MemRange[dtype, self._TILE_ROWS * self._TILE_COLS * self._NUM_STAGES],
                128,
            ]
            sO_row: cute.struct.Align[
                cute.struct.MemRange[Uint8, self._TILE_ROWS * self._TILE_COLS * self._NUM_STAGES],
                128,
            ]
            sO_col: cute.struct.Align[
                cute.struct.MemRange[Uint8, self._TILE_ROWS * self._TILE_COLS * self._NUM_STAGES],
                128,
            ]
            # Per-warp colwise-reduce scratchpad
            sColReduce: cute.struct.Align[
                cute.struct.MemRange[Float32, THREADS_PER_WARP * self._WARPS_PER_CTA], 16
            ]
            # Staged rowwise scales for the whole CTA
            sScaleRow: cute.struct.Align[
                cute.struct.MemRange[
                    Uint8,
                    self._TILE_ROWS * self._NUM_TILES * self._TILE_COLS // MXFP8_BLOCK_SCALING_SIZE,
                ],
                16,
            ]
            # Staged colwise scales for the whole CTA
            sScaleCol: cute.struct.Align[
                cute.struct.MemRange[Uint8, self._NUM_TILES * self._TILE_COLS], 16
            ]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        tile_layout = cute.make_layout(
            ((self._TILE_ROWS, self._TILE_COLS), self._NUM_STAGES),
            stride=((self._TILE_COLS, 1), self._TILE_ROWS * self._TILE_COLS),
        )
        # SMEM input tile should have the same Swizzle<3,4,3> as the input TMA atom
        sX = storage.sX.get_tensor(tile_layout, swizzle=cute.make_swizzle(3, 4, 3))
        # SMEM output tiles should have the same Swizzle<2,4,3> as the output TMA atom
        sO_row = storage.sO_row.get_tensor(tile_layout, swizzle=cute.make_swizzle(2, 4, 3))
        sO_col = storage.sO_col.get_tensor(tile_layout, swizzle=cute.make_swizzle(2, 4, 3))
        # Reshape the per-warp colwise-reduce scratchpad to a (threads, warps) layout for easier indexing.
        sColReduce = storage.sColReduce.get_tensor(
            cute.make_layout((THREADS_PER_WARP, self._WARPS_PER_CTA), stride=(1, THREADS_PER_WARP))
        )
        # The whole rowwise scale factor SMEM tensor covered by a CTA
        sScaleRow2D = storage.sScaleRow.get_tensor(
            cute.make_ordered_layout(
                (
                    self._TILE_ROWS * self._NUM_TILES_Y,
                    self._SCALE_COLS * self._NUM_TILES_X,
                ),
                order=(1, 0),
            )
        )
        # Divide by a single tile's shape, so it becomes ((TILE_ROWS, SCALE_COLS), TILES) for easier indexing
        # where TILES = NUM_TILES_X * NUM_TILES_Y
        sScaleRow = cute.zipped_divide(sScaleRow2D, (self._TILE_ROWS, self._SCALE_COLS))
        # The whole colwise scale factor SMEM tensor covered by a CTA
        sScaleCol2D = storage.sScaleCol.get_tensor(
            cute.make_ordered_layout(
                (
                    self._SCALE_ROWS * self._NUM_TILES_Y,
                    self._TILE_COLS * self._NUM_TILES_X,
                ),
                order=(1, 0),
            )
        )
        # Divide by a single tile's shape, so it becomes ((SCALE_ROWS, TILE_COLS), TILES) for easier indexing
        # where TILES = NUM_TILES_X * NUM_TILES_Y
        sScaleCol = cute.zipped_divide(sScaleCol2D, (self._SCALE_ROWS, self._TILE_COLS))

        # Zero the scale SMEM buffers so partial tiles / padding columns flush as 0.
        tidx, _, _ = cute.arch.thread_idx()
        # View rowwise staging buffers as flat uint32 and stride the CTA over them.
        _ROW_SCALE_WORDS = self._TILE_ROWS * self._SCALE_COLS * self._NUM_TILES // 4
        sScaleRow_u32 = cute.make_tensor(
            cute.recast_ptr(sScaleRow2D.iterator, dtype=Uint32),
            cute.make_layout((_ROW_SCALE_WORDS,), stride=(1,)),
        )
        for i in cutlass.range_constexpr(cute.ceil_div(_ROW_SCALE_WORDS, self._THREADS_PER_CTA)):
            slot = i * self._THREADS_PER_CTA + tidx
            if slot < _ROW_SCALE_WORDS:
                sScaleRow_u32[slot] = Uint32(0)
        # View colwise staging buffers as flat uint32 and stride the CTA over them.
        _COL_SCALE_WORDS = self._SCALE_ROWS * self._TILE_COLS * self._NUM_TILES // 4
        sScaleCol_u32 = cute.make_tensor(
            cute.recast_ptr(sScaleCol2D.iterator, dtype=Uint32),
            cute.make_layout((_COL_SCALE_WORDS,), stride=(1,)),
        )
        for i in cutlass.range_constexpr(cute.ceil_div(_COL_SCALE_WORDS, self._THREADS_PER_CTA)):
            slot = i * self._THREADS_PER_CTA + tidx
            if slot < _COL_SCALE_WORDS:
                sScaleCol_u32[slot] = Uint32(0)

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom)

        # Only warp 0 is the producer (issues TMA)
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        # Every warp is the consumer
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self._WARPS_PER_CTA)
        # One TMA loads a tile
        tx_count = self._TILE_ROWS * self._TILE_COLS * dtype.width // 8

        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_storage.data_ptr(),
            num_stages=self._NUM_STAGES,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=tx_count,
            cta_layout_vmnk=None,
        )

        prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self._NUM_STAGES
        )
        cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self._NUM_STAGES
        )

        gX_tiled = cute.zipped_divide(tma_src, (self._TILE_ROWS, self._TILE_COLS))
        tXsX, tXgX = cute.nvgpu.cpasync.tma_partition(
            tma_atom, 0, cute.make_layout(1), sX, gX_tiled
        )
        gO_row_tiled = cute.zipped_divide(tma_dst_out_row, (self._TILE_ROWS, self._TILE_COLS))
        tXsO_row, tXgO_row = cute.nvgpu.cpasync.tma_partition(
            tma_atom_out_row, 0, cute.make_layout(1), sO_row, gO_row_tiled
        )
        gO_col_tiled = cute.zipped_divide(tma_dst_out_col, (self._TILE_ROWS, self._TILE_COLS))
        tXsO_col, tXgO_col = cute.nvgpu.cpasync.tma_partition(
            tma_atom_out_col, 0, cute.make_layout(1), sO_col, gO_col_tiled
        )

        cute.arch.sync_threads()

        # Prologue: warp 0 prefetches up to NUM_STAGES tiles to fully fill the pipeline
        if warp_idx == 0:
            for s in cutlass.range_constexpr(self._NUM_STAGES):
                if s < num_tiles:
                    # This tile's position in the CTA's NUM_TILES_Y x NUM_TILES_X arrangement.
                    tile_y = s // num_tiles_x
                    tile_x = s % num_tiles_x
                    mainloop_pipeline.producer_acquire(prod_state)
                    cute.copy(
                        tma_atom,
                        tXgX[(None, (tile_y_base + tile_y, tile_x_base + tile_x))],
                        tXsX[(None, prod_state.index)],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(prod_state),
                    )
                    mainloop_pipeline.producer_commit(prod_state)
                    prod_state.advance()

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

            # This tile's position in the CTA's NUM_TILES_Y x NUM_TILES_X arrangement.
            tile_y = tile_idx // num_tiles_x
            tile_x = tile_idx % num_tiles_x

            # Process the 32x64 SMEM tile for this stage
            quantize_bidimensional_mxfp8_swizzled(
                sX[None, stage_idx],
                sO_row[None, stage_idx],
                sO_col[None, stage_idx],
                sScaleRow[(None, (tile_y, tile_x))],
                sScaleCol[(None, (tile_y, tile_x))],
                sColReduce[
                    None, warp_idx
                ],  # Pick the per-warp colwise-reduce scratchpad for this warp
                self._WARPS_PER_CTA,
                max_norm_rcp,
                dtype,
                self.cfg.FP8_DTYPE,
            )

            # Make the smem output writes visible to the TMA async proxy, then store.
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()

            # We are done with this input pipeline SMEM buffer, signal the producer that it can write to this buffer
            mainloop_pipeline.consumer_release(cons_state)

            # Issue TMA and write the output to GMEM
            if warp_idx == 0:
                tile_coord = (tile_y_base + tile_y, tile_x_base + tile_x)
                cute.copy(
                    tma_atom_out_row, tXsO_row[(None, stage_idx)], tXgO_row[(None, tile_coord)]
                )
                cute.copy(
                    tma_atom_out_col, tXsO_col[(None, stage_idx)], tXgO_col[(None, tile_coord)]
                )
                cute.arch.cp_async_bulk_commit_group()

            cons_state.advance()

            # Producer: refill the buffer the consumer just freed with the tile
            # NUM_STAGES ahead.
            next_tile_idx = tile_idx + self._NUM_STAGES
            if next_tile_idx < num_tiles:
                if warp_idx == 0:
                    next_tile_y = next_tile_idx // num_tiles_x
                    next_tile_x = next_tile_idx % num_tiles_x
                    mainloop_pipeline.producer_acquire(prod_state)
                    cute.copy(
                        tma_atom,
                        tXgX[(None, (tile_y_base + next_tile_y, tile_x_base + next_tile_x))],
                        tXsX[(None, prod_state.index)],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(prod_state),
                    )
                    mainloop_pipeline.producer_commit(prod_state)
                    prod_state.advance()

        # TODO(kainingz): the rowwise WIDTH is hardcoded to 4. Consider using other width if we can convince
        # cute.autovec_copy that wider stores are safe (e.g. 16B) and the runtime row pitch allows it. This might
        # require us to change `scale_rowwise_shape`'s divisibility when we compile with fake tensors
        self._flush_scales_to_gmem(
            sScaleRow2D,
            mS_row,
            tidx,
            bidx,
            bidy,
            ROWS=self._TILE_ROWS * self._NUM_TILES_Y,
            COLS=self._SCALE_COLS * self._NUM_TILES_X,
            WIDTH=4,
        )
        self._flush_scales_to_gmem(
            sScaleCol2D,
            mS_col,
            tidx,
            bidx,
            bidy,
            ROWS=self._SCALE_ROWS * self._NUM_TILES_Y,
            COLS=self._TILE_COLS * self._NUM_TILES_X,
            WIDTH=16,
        )

        # Drain the final stores (their gmem writes must complete before the CTA exits).
        cute.arch.cp_async_bulk_wait_group(0, read=False)

    @cute.jit
    def _flush_scales_to_gmem(
        self,
        sScale2D,
        mS,
        tidx,
        bidx,
        bidy,
        ROWS: cutlass.Constexpr,
        COLS: cutlass.Constexpr,
        WIDTH: cutlass.Constexpr,
    ):
        """Flush a staged (ROWS, COLS) SMEM scale block (a plain row-major 2D tensor) to its (bidy, bidx) slice of the gmem scale tensor,
        where each GMEM slice has the same shape as this SMEM tile. Use `WIDTH` bytes per vectorized store.
        """
        mS_M = mS.shape[0]
        mS_N = mS.shape[1]
        # Obtain the GMEM slice for the output scale factor block of this CTA
        mS_tile = cute.local_tile(mS, (ROWS, COLS), (bidy, bidx))

        ACTIVE_THREAD_COLS = COLS // WIDTH
        _, tv_flush_layout = cute.make_layout_tv(
            thr_layout=cute.make_layout((ROWS, ACTIVE_THREAD_COLS), stride=(ACTIVE_THREAD_COLS, 1)),
            val_layout=cute.make_layout((1, WIDTH), stride=(WIDTH, 1)),
        )
        TOTAL_ACTIVE = ROWS * ACTIVE_THREAD_COLS

        # We may have more total active threads than threads in a CTA, so we do multiple waves of stores to flush the whole buffer
        for wave in cutlass.range_constexpr(cute.ceil_div(TOTAL_ACTIVE, self._THREADS_PER_CTA)):
            thread_idx = wave * self._THREADS_PER_CTA + tidx
            if thread_idx < TOTAL_ACTIVE:
                # Find the position for this slot's vectorized store in the GMEM scale factor buffer
                thread_y = bidy * ROWS + thread_idx // ACTIVE_THREAD_COLS
                thread_x = bidx * COLS + (thread_idx % ACTIVE_THREAD_COLS) * WIDTH
                # For rowwise we have N divisible by 4 and WIDTH=4, and for colwise we have N divisible by 128 and WIDTH=16,
                # so `thread_x < mS_N` with vectorized store is safe here.
                # A thread only writes to a single row so `thread_y < mS_M` is also safe here.
                if thread_y < mS_M and thread_x < mS_N:
                    cute.autovec_copy(
                        cute.composition(sScale2D, tv_flush_layout)[thread_idx, None],
                        cute.composition(mS_tile, tv_flush_layout)[thread_idx, None],
                    )


def get_kernel_class(cfg):
    """If no fusion is involved and the kernel only quantizes, dispatch to the specialized kernel for better performance."""
    plain_cast_only = (
        not cfg.WITH_GEMM_SWIZZLED_SCALES
        and not cfg.WITH_AMAX
        and not cfg.WITH_DBIAS
        and not cfg.WITH_DACT
        and not cfg.WITH_ACT
    )
    # Only dispatch to the specialized kernels for packed16 types (bf16/fp16)
    # TODO(kainingz): should specialized kernel consider WITH_NOOP? CUDA C++ kernel seems to ignore it completely.
    if plain_cast_only and is_packed16(cfg.DTYPE):
        if cfg.ROWWISE and not cfg.COLWISE:
            return MXFP8QuantizeSpecializedRowwiseKernel
        if cfg.ROWWISE and cfg.COLWISE:
            return MXFP8QuantizeSpecializedBidimensionalKernel
    return MXFP8QuantizeKernel


def compile_cutedsl_function_from_cfg(cfg):
    """
    Return the compiled CuTeDSL function object for the given MXFP8 quantization config.
    """

    # Route plain cast-only configs to the matching specialized kernel (mirrors the
    # CUDA dispatcher); everything else uses the general standard kernel.
    kernel_class = get_kernel_class(cfg)
    kernel_obj = kernel_class(cfg)
    # M, N must be divisible by the MXFP8 scale-block size (MXFP8_BLOCK_SCALING_SIZE = 32) — the
    # same alignment the CUDA C++ kernel requires.
    sym_M = cute.sym_int32(divisibility=MXFP8_BLOCK_SCALING_SIZE)
    sym_N = cute.sym_int32(divisibility=MXFP8_BLOCK_SCALING_SIZE)
    in_shape = out_shape = (sym_M, sym_N)
    # TE allocates scale tensors at a padded shape (see
    # MXFP8Quantizer::get_scale_shape in transformer_engine/pytorch/csrc):
    #   rowwise:    (roundup(M, 128),     roundup(N // 32, 4))
    #   columnwise: (roundup(M // 32, 4), roundup(N, 128))
    # These padded extents are NOT M/N (and SymInt has no `//`/`+`), so give the
    # scales their own fresh syms carrying the divisibility the padding
    # guarantees (rowwise: 128 x 4; colwise: 4 x 128).
    scale_rowwise_shape = (cute.sym_int32(divisibility=128), cute.sym_int32(divisibility=4))
    scale_colwise_shape = (cute.sym_int32(divisibility=4), cute.sym_int32(divisibility=128))
    ws_shape = (cute.sym_int32(), sym_N)  # (blocks_Y, N); N ties to input N
    # Native FP8/E8M0 dtypes at the FFI boundary (matches the DLPack dtype the C++
    # bridge sends); the kernels view these buffers as raw bytes internally.
    out_dtype = cutlass.Float8E4M3FN if cfg.FP8_DTYPE == "e4m3" else cutlass.Float8E5M2
    scale_dtype = cutlass.Float8E8M0FNU

    in_fake = cute.runtime.make_fake_compact_tensor(
        cfg.DTYPE, in_shape, stride_order=(1, 0), memspace=cute.AddressSpace.gmem, assumed_align=16
    )
    out_row_fake = (
        cute.runtime.make_fake_compact_tensor(
            out_dtype,
            out_shape,
            stride_order=(1, 0),
            memspace=cute.AddressSpace.gmem,
            assumed_align=16,
        )
        if cfg.ROWWISE
        else None
    )
    scale_row_fake = (
        cute.runtime.make_fake_compact_tensor(
            scale_dtype,
            scale_rowwise_shape,
            stride_order=(1, 0),
            memspace=cute.AddressSpace.gmem,
            assumed_align=4,
        )
        if cfg.ROWWISE
        else None
    )
    out_col_fake = (
        cute.runtime.make_fake_compact_tensor(
            out_dtype,
            out_shape,
            stride_order=(1, 0),
            memspace=cute.AddressSpace.gmem,
            assumed_align=16,
        )
        if cfg.COLWISE
        else None
    )
    scale_col_fake = (
        cute.runtime.make_fake_compact_tensor(
            scale_dtype,
            scale_colwise_shape,
            stride_order=(1, 0),
            memspace=cute.AddressSpace.gmem,
            assumed_align=4,
        )
        if cfg.COLWISE
        else None
    )
    amax_fake = (
        cute.runtime.make_fake_compact_tensor(
            Float32, (1,), stride_order=(0,), memspace=cute.AddressSpace.gmem, assumed_align=4
        )
        if cfg.WITH_AMAX
        else None
    )
    noop_fake = (
        cute.runtime.make_fake_compact_tensor(
            Float32, (1,), stride_order=(0,), memspace=cute.AddressSpace.gmem, assumed_align=4
        )
        if cfg.WITH_NOOP
        else None
    )
    act_input_fake = (
        cute.runtime.make_fake_compact_tensor(
            cfg.DTYPE,
            in_shape,
            stride_order=(1, 0),
            memspace=cute.AddressSpace.gmem,
            assumed_align=16,
        )
        if cfg.WITH_DACT
        else None
    )
    workspace_fake = (
        cute.runtime.make_fake_compact_tensor(
            Float32, ws_shape, stride_order=(1, 0), memspace=cute.AddressSpace.gmem, assumed_align=4
        )
        if cfg.WITH_DBIAS
        else None
    )

    compiled = cute.compile(
        kernel_obj,
        in_fake,  # mX
        out_row_fake,
        scale_row_fake,  # mO_row, mS_row
        out_col_fake,
        scale_col_fake,  # mO_col, mS_col
        amax_fake,  # mAmax
        noop_fake,  # mNoop (1-element cast_noop flag)
        act_input_fake,  # mDActInput (backward slot, unused)
        workspace_fake,  # mWorkspace(backward slot, unused)
        cute.runtime.make_fake_stream(),  # stream (compiled as an explicit tvm-ffi
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
    """Compile the MXFP8 quantize kernel for this config and register it in the TVM-FFI global registry
    under EXACTLY `fn_name` (the key the C++ dispatcher built; Python treats it as an opaque name).
    Returns True if a kernel is successfully registered under `fn_name` (the C++ side then fetches it with GetGlobal(fn_name));
    False if the config is unsupported, so the caller caches the negative result and falls back to the CUDA C++ kernel.
    """
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
        logger.warning(
            "CuTeDSL MXFP8 backend does not support this config, "
            "falling back to the CUDA C++ kernel: %s",
            e,
        )
        return False

    logger.debug("Compiling CuTeDSL MXFP8 quantization kernel for %s", cfg)
    try:
        compiled = compile_cutedsl_function_from_cfg(cfg)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(
            "CuTeDSL MXFP8 kernel compilation failed, falling back to the CUDA C++ kernel: %s",
            e,
        )
        return False
    tvm_ffi.register_global_func(fn_name, compiled, override=True)

    return True
