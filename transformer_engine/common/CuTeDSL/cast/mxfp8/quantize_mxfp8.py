# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MXFP8 quantization kernel implemented in CuTeDSL.

Replicates the core logic of quantize_mxfp8.cuh: given a 2D tensor of BF16/FP16
values, quantize to MXFP8 format (FP8E4M3 data + E8M0 per-block scales).

Matches the C++ kernel's tile dimensions and thread layout:
  CHUNK_DIM_Y = 64, CHUNK_DIM_X = 64, THREADS_PER_CTA = 64
  BUFF_DIM_Y  = 32, BUFF_DIM_X  = 64, STAGES = 2
  MXFP8_BLOCK_SIZE   = 32 (elements per MXFP8 scaling block)

Grid: (ceil(N / 64), ceil(M / 64))
Each block processes a 64x64 chunk in 2 stages of 32x64 tiles loaded into
shared memory.
"""
import logging
import os

from transformer_engine.common.CuTeDSL.utils import _bitcast_f32_to_i32, str_to_cutlass_dtype

from typing import Optional, Type

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import Float32, Int16, Int32, Uint32, Uint8
from cuda.bindings.driver import CUstream

import tvm_ffi

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

from transformer_engine.common.CuTeDSL.utils import (
    is_packed16,
    packed16_kit,
    fabs_f32,
    exp2f_rcp,
    pack_f32x2,
    mul_cvt_f32x4_to_fp8x4,
)
from transformer_engine.common.CuTeDSL.utils_fp8 import (
    get_cvt_f32_to_fp8_func, 
    get_cvt_f32x2_to_fp8x2_func,
    cvt_f32_to_fp8e8m0
)

CUTEDSL_DEBUG_LOGGING = os.environ.get("CUTEDSL_DEBUG_LOGGING", "0") == "1"

logger = logging.getLogger("transformer_engine.cutedsl.mxfp8")

# Number of elements per MXFP8 scale block. They will share the same E8M0 scale factor
MXFP8_BLOCK_SIZE = 32
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
def quantize_rowwise_mxfp8(
    sX_tile,        # (TILE_Y, TILE_X) bf16/fp16 smem view, post-TMA
    sA_tile,        # (TILE_Y, TILE_X) activation-input smem tile (dact only)
    sO_row_tile,    # (TILE_Y, TILE_X) uint8 smem view (rowwise FP8 output)
    mS_row_stage,   # rowwise scale tensor (1D swizzled, or 2D linear)
    max_norm_rcp,
    tile_row_start, # Int32 — global row index of this stage's row 0
                    # (= tile_idx_y * TILE_Y). Used to mask OOB scale stores
                    # for irregular shapes.
    tile_col_start, # Int32 — global col index of this CTA's col 0
                    # (= bidx * TILE_X). Same purpose.
    M, N,           # Int32 — full tensor extents; OOB threads skip their
                    # scale store.
    ACTIVATION,
    DTYPE,
    FP8_DTYPE,
    TILE_Y,
    WAVES,
    THREADS_PER_BANK,
    PACK_SIZE,
    WITH_ACT=False,
    WITH_DACT=False,
    WITH_DBIAS=False,  # rowwise-only dbias: accumulate per-column partials
    dbias_acc=None,         #  only needed when WITH_DBIAS is True
):
    tidx, _, _ = cute.arch.thread_idx()

    _, tv_layout = cute.make_layout_tv(
        thr_layout=cute.make_layout((TILE_Y, 2), stride=(2, 1)),
        val_layout=cute.make_layout((1, MXFP8_BLOCK_SIZE), stride=(0, 1))
    )

    sX_tv = cute.composition(sX_tile, tv_layout)
    sO_tv = cute.composition(sO_row_tile, tv_layout)

    # I/O Elements that belong to this thread
    sX_thread = sX_tv[tidx, None]   # shape (32,) bf16
    sO_thread = sO_tv[tidx, None]   # shape (32,) uint8

    sO_thread_u32_ptr = cute.recast_ptr(sO_thread.iterator, dtype=Uint32)
    # Each wave it writes 32 bytes = 8 uint32s, so in 4 waves we write all 32 quantized elements.
    sO_thread_u32 = cute.make_tensor(
        sO_thread_u32_ptr,
        cute.make_layout((MXFP8_BLOCK_SIZE // 4,), stride=(1,)), # 1 uint32 is 4 fp8 elements
    )

    # PTX allows to fuse relu activation in `cvt.rn.satfinite`
    FUSE_RELU = cutlass.const_expr(ACTIVATION == "relu")
    # For this fast path we can read in pack of 2 instead of reading individual f16 / bf16 element.
    # dbias needs the per-element fp32 values to accumulate, so it forces the slow path.
    _row_fast = (is_packed16(DTYPE) and (ACTIVATION is None or FUSE_RELU)
                 and not WITH_DBIAS)

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
        sX_thread_rw_i32 = cute.make_tensor(
            cute.recast_ptr(sX_thread.iterator, dtype=Int32),
            cute.make_layout((1, MXFP8_BLOCK_SIZE // 2), stride=(0, 1)), # 1 int32 is 2 fp16/bf16 elements
        )
        # Each wave we read 2 packed i32, which is 4 fp16/bf16 elements (PACK_SIZE)
        # In total we have 8 waves where each wave reads 4 elements, so we read 32 elements in total.
        in_r = [[None, None] for _ in range(WAVES)]
        for w in cutlass.range_constexpr(WAVES):
            idx = (w * 2 + offset // 2) % (MXFP8_BLOCK_SIZE // 2)
            in_r[w][0] = sX_thread_rw_i32[0, idx]
            in_r[w][1] = sX_thread_rw_i32[0, idx + 1]

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
            cute.make_layout((1, MXFP8_BLOCK_SIZE), stride=(0, 1)),
        )

        if cutlass.const_expr(WITH_DACT):
            # Backward: out = grad · act'(act_input). sX is grad, sA is act_input.
            dop = SUPPORTED_DACTIVATIONS[ACTIVATION]
            sA_thread = cute.composition(sA_tile, tv_layout)[tidx, None]
            sA_thread_rw = cute.make_tensor(
                sA_thread.iterator,
                cute.make_layout((1, MXFP8_BLOCK_SIZE), stride=(0, 1)),
            )
        elif cutlass.const_expr(WITH_ACT):
            op = SUPPORTED_ACTIVATIONS[ACTIVATION]

        if cutlass.const_expr(is_packed16(DTYPE) and ACTIVATION is not None):
            kit_act = packed16_kit(DTYPE)

        # Each wave we read PACK_SIZE elements, and we have WAVES waves, so we read WAVES * PACK_SIZE (= MXFP8_BLOCK_SIZE) elements in total.
        in_r = [[None] * PACK_SIZE for _ in range(WAVES)]
        for w in cutlass.range_constexpr(WAVES):
            start = (w * PACK_SIZE + offset) % MXFP8_BLOCK_SIZE
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
                    amax_r = cute.arch.fmax(amax_r, x) # For relu cases, we don't need abs since negative values will be 0 so they lose comparison automatically
                else:
                    amax_r = cute.arch.fmax(amax_r, fabs_f32(x))
        if cutlass.const_expr(FUSE_RELU):
            amax_r = cute.arch.fmax(amax_r, Float32(0.0)) # If relu, the amax is at least 0

    biased_exp_r = cvt_f32_to_fp8e8m0(amax_r * max_norm_rcp)

    # mS_row_stage has logical shape (32, 2) and we have 64 threads where each is mapped to one scale factor
    # The TV layout is equivalent to TV layout with thr_layout=(32, 2):(2, 1), val_layout=(1,)
    # but it's too trival so let's just index it directly without using layout
    # Note this is the logical layout, which is on top of the swizzled / non-swizzled scale factor layout
    # that mappes the logical index to the physical offset

    # For irregular shapes, skip the scale store if this thread's logical row / col-block lies past the input's actual extents. 
    # TMA already zero-fills OOB input reads and drops OOB output writes; only the direct scale-byte gmem store needs an explicit guard.
    scale_row = tile_row_start + tidx // 2
    scale_col_first_elt = tile_col_start + (tidx % 2) * MXFP8_BLOCK_SIZE
    if scale_row < M and scale_col_first_elt < N:
        mS_row_stage[(tidx // 2, tidx % 2)] = Uint8(biased_exp_r)

    inv_scale_r = exp2f_rcp(biased_exp_r) # f32 reciprocal of the scale
    # Fetch the conversion function based on the FP8 format
    cvt_f32x2 = get_cvt_f32x2_to_fp8x2_func(FP8_DTYPE)
    if cutlass.const_expr(_row_fast):
        kit_cast = packed16_kit(DTYPE)
        mul_cvt_x2 = kit_cast.mul_cvt_to_fp8x2(FP8_DTYPE, FUSE_RELU)
        # Pack `(inv_scale_r, inv_scale_r)` as a single 64-bit f32x2 once;
        # the per-wave mul_cvt consumes this directly.
        scale_2x = pack_f32x2(inv_scale_r, inv_scale_r)

    for w in cutlass.range_constexpr(WAVES):
        idx = (w * 4 + offset) % MXFP8_BLOCK_SIZE
        idx = idx // 4
        if cutlass.const_expr(_row_fast):
            # One fused PTX per <fmt>x2 pair: <fmt>x2 × f32x2 → fp8x2.
            # Byte layout: byte[0]=fp8(lo * s), byte[1]=fp8(hi * s).
            p01 = mul_cvt_x2(in_r[w][0], scale_2x)
            p23 = mul_cvt_x2(in_r[w][1], scale_2x)
        else:
            # cvt PTX semantics: `cvt.rn.satfinite.<fmt>.f32 d, a, b` gives
            # d[15:8]=fp8(a), d[7:0]=fp8(b). Pass (v1, v0) so the u16 low
            # byte ends up as fp8(v0) and the high byte as fp8(v1).
            v0 = in_r[w][0] * inv_scale_r
            v1 = in_r[w][1] * inv_scale_r
            v2 = in_r[w][2] * inv_scale_r
            v3 = in_r[w][3] * inv_scale_r
            p01 = cvt_f32x2(v1, v0, FUSE_RELU)  # u16 little-endian: v0,v1
            p23 = cvt_f32x2(v3, v2, FUSE_RELU)  # u16 little-endian: v2,v3
        quad = (p23 << Int32(16)) | p01
        sO_thread_u32[idx] = Uint32(quad)

    return amax_r

@cute.jit
def quantize_colwise_mxfp8(
    sX_tile,        # (TILE_Y, TILE_X) bf16/fp16 smem view, post-TMA
    sO_col_tile,    # (TILE_Y, TILE_X) uint8 smem view (colwise FP8 output)
    mS_col_stage,         # colwise scale tensor (1D swizzled, or 2D linear)
    max_norm_rcp,
    tile_row_start, # Int32 — global row index of this stage's row 0
                    # (= tile_idx_y * TILE_Y). Used to mask OOB scale stores
                    # for irregular shapes.
    tile_col_start, # Int32 — global col index of this CTA's col 0
                    # (= bidx * TILE_X).
    M, N,           # Int32 — full tensor extents.
    ACTIVATION,
    DTYPE,
    FP8_DTYPE,
    SWIZZLE,
    TILE_X,
    TILE_Y,
    WITH_ACT=False,     # forward: apply activation to the element
    WITH_DACT=False,    # backward: out = grad · act'(act_input)
    sA_tile=None,       # (TILE_Y, TILE_X) activation-input smem tile (dact only)
    WITH_DBIAS=False,   # also return this thread's column sum (pre-truncate)
    CACHE_ACTIVATION=False,  # overwrite sX_tile in place with the post-activation
                             # (IType-truncated) values, so the rowwise pass can read
                             # them instead of recomputing op
):
    tidx, _, _ = cute.arch.thread_idx()

    _, tv_layout = cute.make_layout_tv(
        thr_layout=cute.make_layout((1, TILE_X), stride=(TILE_X, 1)),
        val_layout=cute.make_layout((MXFP8_BLOCK_SIZE, 1), stride=(1, 1))
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
            cute.make_layout((MXFP8_BLOCK_SIZE,), stride=(TILE_X,)),
        )
        amax_bits = Int16(0)
        for i in cutlass.range_constexpr(MXFP8_BLOCK_SIZE):
            amax_bits = kit.abs_max_scalar(amax_bits, sX_thread_i16[i])
        amax_c = fabs_f32(kit.bits_to_f32(amax_bits))
    else:
        # Otherwise we need to case input values to fp32. Allocate the register tensor and load from SMEM input tiles.
        sX_thread_f32 = cute.make_rmem_tensor(
            layout_or_shape=cute.make_layout((MXFP8_BLOCK_SIZE,), stride=(1,)),
            dtype=Float32,
        )
        for i in cutlass.range_constexpr(MXFP8_BLOCK_SIZE):
            sX_thread_f32[i] = Float32(sX_thread[i])
        # Apply activation (fwd) or grad·act'(act_input) (bwd dact) in f32.
        if cutlass.const_expr(WITH_DACT):
            dop = SUPPORTED_DACTIVATIONS[ACTIVATION]
            sA_thread = cute.composition(sA_tile, tv_layout)[tidx, None]
            for i in cutlass.range_constexpr(MXFP8_BLOCK_SIZE):
                sX_thread_f32[i] = sX_thread_f32[i] * dop(Float32(sA_thread[i]))
        elif cutlass.const_expr(WITH_ACT):
            op = SUPPORTED_ACTIVATIONS[ACTIVATION]
            for i in cutlass.range_constexpr(MXFP8_BLOCK_SIZE):
                sX_thread_f32[i] = op(sX_thread_f32[i])
        # Truncate the activation (after we apply op) back to the half precision type if input is also half precision.
        if cutlass.const_expr(is_packed16(DTYPE) and ACTIVATION is not None):
            kit_act = packed16_kit(DTYPE)
            for i in cutlass.range_constexpr(MXFP8_BLOCK_SIZE):
                sX_thread_f32[i] = kit_act.truncate_f32(sX_thread_f32[i])
        # Columnwise is the preferred direction so it runs first. If it needs to cache the activation in the input tile
        # to let the rowwise pass read it, we need to cast and overwrite the input data in-place here
        if cutlass.const_expr(CACHE_ACTIVATION):
            for i in cutlass.range_constexpr(MXFP8_BLOCK_SIZE):
                sX_thread[i] = DTYPE(sX_thread_f32[i])
        amax_c = Float32(0.0)
        for i in cutlass.range_constexpr(MXFP8_BLOCK_SIZE):
            amax_c = cute.arch.fmax(amax_c, fabs_f32(sX_thread_f32[i]))

    # Irregular shapes: skip when this stage's row range or this thread's
    # column lies past the input extents. TILE_Y == MXFP8_BLOCK_SIZE so each stage
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
        for i in cutlass.range_constexpr(MXFP8_BLOCK_SIZE):
            v_f32 = kit_cast.bits_to_f32(sX_thread_i16[i])
            if cutlass.const_expr(WITH_DBIAS):
                dbias_partial += v_f32
            sO_thread[i] = Uint8(cvt_to_fp8_func(v_f32 * inv_scale_c))
    else:
        for i in cutlass.range_constexpr(MXFP8_BLOCK_SIZE):
            # Accumulate the per-thread column partial for dbias if WITH_DBIAS.
            if cutlass.const_expr(WITH_DBIAS):
                for i in cutlass.range_constexpr(MXFP8_BLOCK_SIZE):
                    dbias_partial += sX_thread_f32[i]
            sO_thread[i] = Uint8(cvt_to_fp8_func(sX_thread_f32[i] * inv_scale_c))

    # Return this stage's per-column partial alongside amax; the caller accumulates
    # it across stages (a scalar can't be updated in-place through the arg).
    return amax_c, dbias_partial

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

class MXFP8QuantizeKernel:
    """The MXFP8 quantization kernel that mirrors the standard (non-specialized) MXFP8 CUDA C++ quantization kernel
    with multiple fusions (activation, dbias, etc.).
    `__call__` method is the entrypoint which is AOT compiled. `self` will be captured so it's fixed per compiled kernel
    """

    # Vectorised access constants for bank-conflict avoidance (rowwise pass)
    _PACK_SIZE = 4 # Elements per vector load
    _WAVES = MXFP8_BLOCK_SIZE // _PACK_SIZE # Each thread reads 8 waves with each wave reads 4 packed bf16, so it reads a whole MXFP8 block in total
    _TOTAL_BANKS_WIDTH = (32 * 4) // 1  # 32 banks × 4 bytes, in bytes (uint8 stride)
    _THREADS_PER_BANK = _TOTAL_BANKS_WIDTH // MXFP8_BLOCK_SIZE  # 4 threads per bank
    
    # Tiling sizes
    _NUM_STAGES = 2 # Pipeline depth of the producer/consumer ring buffer for the TMA-G2S input loads (PipelineTmaAsync stage count)
    _NUM_TILES = 2 # Each CTA process 2 tiles along the Y (row, slowest-changing) dimension
    _TILE_Y = 32 # Each tile has 32 rows, so each CTA handles 32 * 2 rows in total
    _TILE_X = 64 # Each tile has 64 columns

    # CTA size
    _THREADS_PER_CTA = 64
    _NUM_WARPS = _THREADS_PER_CTA // 32

    def __init__(self, cfg):
        self.cfg = cfg
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
            and cfg.ROWWISE and cfg.COLWISE
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
        mX: cute.Tensor, # Input tensor to quantize
        mO_row: Optional[cute.Tensor], mS_row: Optional[cute.Tensor], # Rowwise output and scale tensors
        mO_col: Optional[cute.Tensor], mS_col: Optional[cute.Tensor], # Colwise output and scale tensors
        mAmax: Optional[cute.Tensor], # Global amax accumulator, only used when WITH_AMAX is True
        mNoop: Optional[cute.Tensor], # 1-element cast_noop flag, only used when WITH_NOOP is True
        mDActInput: Optional[cute.Tensor], # Activation input for activation derivative fusion, only used when WITH_DACT is True
        mWorkspace: Optional[cute.Tensor], # Workspace for the dbias reduction, only used when WITH_DBIAS is True
        stream: CUstream,
    ):
        if cutlass.const_expr(CUTEDSL_DEBUG_LOGGING):
            cute.printf(f"[CuTeDSL] MXFP8QuantizeKernel.__call__() with config: {self.cfg}\n")

        M = mX.shape[0]
        N = mX.shape[1]
        cfg = self.cfg
        max_norm_rcp = cfg.MAX_NORM_RCP
        num_scale_cols = N // MXFP8_BLOCK_SIZE
        num_scale_rows = M // MXFP8_BLOCK_SIZE
        
        # If WITH_GEMM_SWIZZLED_SCALES is enabled, the output must satisfy cublas's swizzled layout
        # This is expressed as a CuTe layout applied to the output tensor so it can be transparent throughout the kernel implementation.
        # See https://docs.nvidia.com/cuda/cublas/#d-block-scaling-factors-layout for more details.
        if cutlass.const_expr(cfg.WITH_GEMM_SWIZZLED_SCALES):
            num_tiles_M = (M + 127) // 128
            num_tiles_SC = (num_scale_cols + 3) // 4
            num_tiles_SR = (num_scale_rows + 3) // 4
            num_tiles_N = (N + 127) // 128
        
            if cutlass.const_expr(cfg.ROWWISE):
                mS_row = cute.make_tensor(
                    mS_row.iterator,
                    cute.make_layout(
                        ((32, 4, num_tiles_M), (4, num_tiles_SC)),
                        stride=((16, 4, num_tiles_SC * 512), (1, 512)),
                    ),
                )
            if cutlass.const_expr(cfg.COLWISE):
                mS_col = cute.make_tensor(
                    mS_col.iterator,
                    cute.make_layout(
                        ((4, num_tiles_SR), (32, 4, num_tiles_N)),
                        stride=((1, 512), (16, 4, num_tiles_SR * 512)),
                    ),
                )

        # We have 2 stages in our pipeline where each stage loads / computes a (TILE_Y, TILE_X) tile
        smem_tile_layout = cute.make_ordered_layout((self._TILE_Y, self._TILE_X), order=(1, 0))
        cta_tiler = (self._TILE_Y, self._TILE_X)

        # Input TMA atoms
        op_load = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_atom, tma_src = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op_load, mX, smem_tile_layout, cta_tiler, num_multicast=1,
        )

        # Activation input TMA atoms for activation derivative fusion
        tma_atom_act = None
        tma_src_act = None
        if cutlass.const_expr(cfg.WITH_DACT):
            tma_atom_act, tma_src_act = cute.nvgpu.cpasync.make_tiled_tma_atom(
                op_load, mDActInput, smem_tile_layout, cta_tiler, num_multicast=1,
            )

        # Output TMA atoms
        op_store = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
        out_smem_layout = cute.make_ordered_layout((self._TILE_Y, self._TILE_X), order=(1, 0))
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

        grid = [
            cute.ceil_div(Int32(N), self._TILE_X),
            cute.ceil_div(M, self._TILE_Y * self._NUM_TILES),
        ]
        block = [self._THREADS_PER_CTA,]
        
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
        tma_atom, tma_src, # Input TMA atoms
        tma_atom_out_row, tma_dst_out_row, # Rowwise output TMA atoms
        tma_atom_out_col, tma_dst_out_col, # Colwise output TMA atoms
        tma_atom_act, tma_src_act, # Activation derivative TMA atoms, or None if WITH_DACT is False
    ):
        cfg = self.cfg
        # If the noop tensor is not passed (compile-time check), or the noop tensor is not 1.0 (run-time check)
        # then we run the kernel for real. Otherwise, skip the quantization so this kernel becomes a no-op.
        if not cutlass.const_expr(cfg.WITH_NOOP) or mNoop[0] != Float32(1.0):
            self._kernel_main(
                mX, mS_row, mS_col, mAmax, mWorkspace,
                max_norm_rcp, dtype,
                tma_atom, tma_src,
                tma_atom_out_row, tma_dst_out_row,
                tma_atom_out_col, tma_dst_out_col,
                tma_atom_act, tma_src_act,
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
        tma_atom, tma_src, # Input TMA atoms
        tma_atom_out_row, tma_dst_out_row, # Rowwise output TMA atoms
        tma_atom_out_col, tma_dst_out_col, # Colwise output TMA atoms
        tma_atom_act, tma_src_act, # Activation derivative TMA atoms, or None if WITH_DACT is False
    ):
        cfg = self.cfg

        if cutlass.const_expr(cfg.ROWWISE):
            mS_row = cute.zipped_divide(mS_row, (self._TILE_Y, self._TILE_X // MXFP8_BLOCK_SIZE))
        if cutlass.const_expr(cfg.COLWISE):
            mS_col = cute.zipped_divide(mS_col, (self._TILE_Y // MXFP8_BLOCK_SIZE, self._TILE_X))

        # Allocate shared memory for the input and rowwise / columnwise outputs
        if cutlass.const_expr(cfg.ROWWISE and cfg.COLWISE):
            @cute.struct
            class SharedStorage:
                mbar_storage: cute.struct.MemRange[cute.Int64, 2 * self._NUM_STAGES]
                sX: cute.struct.Align[
                    cute.struct.MemRange[dtype, self._TILE_Y * self._TILE_X * self._NUM_STAGES], 128
                ]
                sO_row: cute.struct.Align[
                    cute.struct.MemRange[Uint8, self._TILE_Y * self._TILE_X * self._NUM_STAGES], 128
                ]
                sO_col: cute.struct.Align[
                    cute.struct.MemRange[Uint8, self._TILE_Y * self._TILE_X * self._NUM_STAGES], 128
                ]
                sAmax: cute.struct.MemRange[Float32, self._NUM_WARPS]
        elif cutlass.const_expr(cfg.ROWWISE and not cfg.COLWISE):
            @cute.struct
            class SharedStorage:
                mbar_storage: cute.struct.MemRange[cute.Int64, 2 * self._NUM_STAGES]
                sX: cute.struct.Align[
                    cute.struct.MemRange[dtype, self._TILE_Y * self._TILE_X * self._NUM_STAGES], 128
                ]
                sO_row: cute.struct.Align[
                    cute.struct.MemRange[Uint8, self._TILE_Y * self._TILE_X * self._NUM_STAGES], 128
                ]
                sAmax: cute.struct.MemRange[Float32, self._NUM_WARPS]
        elif cutlass.const_expr(cfg.ROWWISE):
            @cute.struct
            class SharedStorage:
                mbar_storage: cute.struct.MemRange[cute.Int64, 2 * self._NUM_STAGES]
                sX: cute.struct.Align[
                    cute.struct.MemRange[dtype, self._TILE_Y * self._TILE_X * self._NUM_STAGES], 128
                ]
                sO_row: cute.struct.Align[
                    cute.struct.MemRange[Uint8, self._TILE_Y * self._TILE_X * self._NUM_STAGES], 128
                ]
                sAmax: cute.struct.MemRange[Float32, self._NUM_WARPS]
        else:
            @cute.struct
            class SharedStorage:
                mbar_storage: cute.struct.MemRange[cute.Int64, 2 * self._NUM_STAGES]
                sX: cute.struct.Align[
                    cute.struct.MemRange[dtype, self._TILE_Y * self._TILE_X * self._NUM_STAGES], 128
                ]
                sO_col: cute.struct.Align[
                    cute.struct.MemRange[Uint8, self._TILE_Y * self._TILE_X * self._NUM_STAGES], 128
                ]
                sAmax: cute.struct.MemRange[Float32, self._NUM_WARPS]
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        # Apply the layout to the allocated shared memory buffers so the first rank is the tile (nested layout) 
        # and the second rank is the pipeline stage
        sX = storage.sX.get_tensor(
            cute.make_layout(
                ((self._TILE_Y, self._TILE_X), self._NUM_STAGES),
                stride=((self._TILE_X, 1), self._TILE_Y * self._TILE_X),
            )
        )
        if cutlass.const_expr(cfg.ROWWISE):
            sO_row = storage.sO_row.get_tensor(
                cute.make_layout(
                    ((self._TILE_Y, self._TILE_X), self._NUM_STAGES),
                    stride=((self._TILE_X, 1), self._TILE_Y * self._TILE_X),
                )
            )
        if cutlass.const_expr(cfg.COLWISE):
            sO_col = storage.sO_col.get_tensor(
                cute.make_layout(
                    ((self._TILE_Y, self._TILE_X), self._NUM_STAGES),
                    stride=((self._TILE_X, 1), self._TILE_Y * self._TILE_X),
                )
            )

        # Allocate shared memory for the activation input used for the activation derivative fusion.
        if cutlass.const_expr(cfg.WITH_DACT):
            @cute.struct
            class DactStorage:
                sActInput: cute.struct.Align[
                    cute.struct.MemRange[dtype, self._TILE_Y * self._TILE_X * self._NUM_STAGES], 128
                ]
            dact_storage = smem.allocate(DactStorage)
            # Apply the same layout as the input
            sActInput = dact_storage.sActInput.get_tensor(
                cute.make_layout(
                    ((self._TILE_Y, self._TILE_X), self._NUM_STAGES),
                    stride=((self._TILE_X, 1), self._TILE_Y * self._TILE_X),
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
        tx_count = self._TILE_Y * self._TILE_X * dtype.width // 8
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
            cta_layout_vmnk=None,   # single-CTA, no cluster/multicast
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
            cute.ceil_div(M - bidy * self._TILE_Y * self._NUM_TILES, self._TILE_Y),
        )

        # Tile the TMA gmem view: ((TILE_Y, TILE_X), (M/TILE_Y, N/TILE_X)).
        gX_tiled = cute.zipped_divide(tma_src, (self._TILE_Y, self._TILE_X))

        # Partition sX/gX for the TMA atom (single-CTA, no cluster/multicast).
        tXsX, tXgX = cute.nvgpu.cpasync.tma_partition(
            tma_atom,
            0, # Use the only CTA to do the TMA copy
            cute.make_layout(1), # This cluster only has 1 CTAs
            sX,
            gX_tiled,
        )

        # If WITH_DACT, partition the activation input for TMA as well in the same way
        if cutlass.const_expr(cfg.WITH_DACT):
            gA_tiled = cute.zipped_divide(tma_src_act, (self._TILE_Y, self._TILE_X))
            tXsA, tXgA = cute.nvgpu.cpasync.tma_partition(
                tma_atom_act,
                0,
                cute.make_layout(1),
                sActInput,
                gA_tiled,
            )

        # Partitioning for rowwise / columnwise outputs
        if cutlass.const_expr(cfg.ROWWISE):
            gO_row_tiled = cute.zipped_divide(tma_dst_out_row, (self._TILE_Y, self._TILE_X))
            tXsO_row, tXgO_row = cute.nvgpu.cpasync.tma_partition(
                tma_atom_out_row,
                0,
                cute.make_layout(1),
                sO_row,
                gO_row_tiled,
            )
        if cutlass.const_expr(cfg.COLWISE):
            gO_col_tiled = cute.zipped_divide(tma_dst_out_col, (self._TILE_Y, self._TILE_X))
            tXsO_col, tXgO_col = cute.nvgpu.cpasync.tma_partition(
                tma_atom_out_col,
                0,
                cute.make_layout(1),
                sO_col,
                gO_col_tiled,
            )

        # Ensure barrier init is visible to all threads before the pipeline is used.
        cute.arch.sync_threads()

        # ---- Producer: warp 0 issues one TMA copy per tile. ----
        if warp_idx == 0:
            for stage in cutlass.range(num_tiles, unroll=1):
                mainloop_pipeline.producer_acquire(prod_state)
                tile_y = bidy * self._NUM_TILES + stage
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
        # Each thread will process two (1, MXFP8_BLOCK_SIZE) rows in two stages, and in each stage the thread will add the
        # (after dact applied) value to this register array with the same shape so it carries the the two stages' partial sum.
        # Then it will be written to a SMEM buffer to let the whole CTA do the reduction separately to yield
        # the final (1, TILE_X) dbias workspace output.
        rowwise_dbias_acc = None
        if cutlass.const_expr(self.DBIAS_REDUCTION_ROWWISE):
            rowwise_dbias_acc = cute.make_rmem_tensor(
                layout_or_shape=cute.make_layout((MXFP8_BLOCK_SIZE,), stride=(1,)),
                dtype=Float32,
            )
            # Zero the accumulator registers.
            for c in cutlass.range_constexpr(MXFP8_BLOCK_SIZE):
                rowwise_dbias_acc[c] = Float32(0.0)
            block_dbias = Float32(0.0)
        # Prepare thread-level register for columnwise dbias reduction.
        # Each thread will process two (MXFP8_BLOCK_SIZE, 1) columns in two stages, and in each stage the thread will reduce the
        # (after dact applied) column to (1,) and add to this register.
        # Then this partial sum scalar will be written to the GMEM workspace buffer directly.
        if cutlass.const_expr(self.DBIAS_REDUCTION_COLWISE):
            block_dbias = Float32(0.0)

        # ---- Consumer: all threads quantize each completed tile. ----
        for stage in cutlass.range(num_tiles, unroll=1):
            mainloop_pipeline.consumer_wait(cons_state)
            sX_tile = sX[(None, stage)]
            sActInput_tile = None
            if cutlass.const_expr(cfg.WITH_DACT):
                sActInput_tile = sActInput[(None, stage)]
            tile_idx_x = bidx
            # Each CTA has `NUM_TILES` tiles. Each stage we need to obtain the tile for that specific stage. 
            # So the tile index along Y dimension is `bidy * NUM_TILES + stage`
            tile_idx_y = bidy * self._NUM_TILES + stage
            # Process rowwise and colwise quantization separately
            if cutlass.const_expr(cfg.COLWISE):
                # The first row that belongs to this CTA. Each CTA handles NUM_TILES of (TILE_Y, TILE_X) tiles stacked vertically,
                # and each stage handles one of them.
                sO_col_tile = sO_col[(None, stage)]
                mS_col_stage = cute.flatten(mS_col[(None, (tile_idx_y, tile_idx_x))])

                amax_c, dbias_c = self._process_colwise(
                    sX_tile, sO_col_tile,
                    mS_col_stage, max_norm_rcp,
                    tile_idx_y * self._TILE_Y, bidx * self._TILE_X, M, N,
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
                    tile_idx_y * self._TILE_Y, bidx * self._TILE_X, M, N,
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

            # Warp 0 issues TMA copy to write the quantized output tile from shared memory to global memory and then commits
            if warp_idx == 0:
                tile_y = bidy * self._NUM_TILES + stage
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

        # Complete the cross-thread dbias reduction after each thread has its own per-thread partial sum after the rowwise quantization.
        if cutlass.const_expr(self.DBIAS_REDUCTION_ROWWISE):
            # Allocate the SMEM buffer that all threads use to reduce the two-stage partial sum (per thread) to the 
            # partial sum (per block).

            # Pad the buffer to avoid bank conflicts. The logical shape is still the same. Only the stride is different.
            DBIAS_BUFF_WIDTH = self._TILE_X // MXFP8_BLOCK_SIZE * (MXFP8_BLOCK_SIZE + 1)
            @cute.struct
            class DbiasStorage:
                sDbias: cute.struct.MemRange[Float32, self._TILE_Y * DBIAS_BUFF_WIDTH]
            dbias_storage = smem.allocate(DbiasStorage)
            sDbias = dbias_storage.sDbias.get_tensor(
                cute.make_layout((self._TILE_Y, self._TILE_X), stride=(DBIAS_BUFF_WIDTH, 1)),
            )
            # Thread layout: (TILE_Y, 2); value layout: (1, MXFP8_BLOCK_SIZE) where TILE_X = 2 * MXFP8_BLOCK_SIZE
            # And each thread writes the (1, MXFP8_BLOCK_SIZE) partial sum to this (TILE_Y, TILE_X) buffer
            # and then each thread reads its (TILE_Y, 1) sDbias column and writes the reduced sum to the GMEM workspace.
            # Since TILE_X == THREADS_PER_CTA, this column reduction yields (TILE_Y, TILE_X) -> (1, TILE_X).
            _, tv_layout_dbias_write = cute.make_layout_tv(
                thr_layout=cute.make_layout((self._TILE_Y, 2), stride=(2, 1)), 
                val_layout=cute.make_layout((1, MXFP8_BLOCK_SIZE), stride=(MXFP8_BLOCK_SIZE, 1)),
            )
            sDbias_write = cute.composition(sDbias, tv_layout_dbias_write)
            # Each thread start reading from the specfic bank based on its thread ID so they can do their best to access different banks
            # to avoid bank conflict.
            bank_group = (tidx % THREADS_PER_WARP) // self._THREADS_PER_BANK
            # The offset this thread should start reading from based on what's its first bank to access.
            offset = bank_group * self._PACK_SIZE
            for w in cutlass.range_constexpr(self._WAVES):                # Each thread starts from this offset when writing into SMEM to avoid bank conflict
                start = (w * self._PACK_SIZE + offset) % MXFP8_BLOCK_SIZE
                for i in cutlass.range_constexpr(self._PACK_SIZE):
                    # All threads write their per-thread partial sum results to the shared buffer.
                    sDbias_write[(tidx, start + i)] = rowwise_dbias_acc[w * self._PACK_SIZE + i]
            cute.arch.sync_threads()
            # All threads reduce the cross-thread partial sums to the per-block partial sum.
            _, tv_layout_dbias_reduce = cute.make_layout_tv(
                thr_layout=cute.make_layout((1, self._TILE_X), stride=(self._TILE_X, 1)),
                val_layout=cute.make_layout((self._TILE_Y, 1), stride=(1, 1))
            )
            sDbias_reduce = cute.composition(sDbias, tv_layout_dbias_reduce)
            # make_layout_tv yields a (thread, value) layout: thread=tidx -> column tidx,
            # value=i -> row i. So index [tidx, i] (thread first), summing the column's rows.
            block_dbias = Float32(0.0)
            for i in cutlass.range_constexpr(self._TILE_Y):
                block_dbias += sDbias_reduce[tidx, i]

        # Write the per-tile reduced dbias to the global workspace. 
        if cutlass.const_expr(cfg.WITH_DBIAS):
            dbias_col = bidx * self._TILE_X + tidx
            if dbias_col < N:
                mWorkspace[(bidy, dbias_col)] = block_dbias

        if cutlass.const_expr(cfg.WITH_AMAX):
            # Reduce and get the per-warp amax.
            warp_amax = cute.arch.warp_redux_sync(per_thread_amax, kind="fmax")
            # Write the per-warp amax to shared memory
            sAmax = storage.sAmax.get_tensor(cute.make_layout(self._NUM_WARPS))
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
                    amax_i32.iterator, _bitcast_f32_to_i32(cta_amax),
                )

        # Wait for in-flight TMA stores so data is visible to the host
        # before the kernel returns.
        cute.arch.cp_async_bulk_wait_group(0, read=False)

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
        dbias_acc=None,       # rmem Float32[MXFP8_BLOCK_SIZE] dbias accumulator (rowwise-only dbias)
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
            TILE_Y=self._TILE_Y,
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
        sX_tile,        # (TILE_Y, TILE_X) bf16/fp16 smem view, post-TMA
        sO_col_tile,    # (TILE_Y, TILE_X) uint8 smem view (colwise FP8 output)
        mS_col_stage,   # colwise scale tensor (1D swizzled, or 2D linear)
        max_norm_rcp,
        tile_row_start, # Int32 — global row of this stage's row 0
        tile_col_start, # Int32 — global col of this CTA's col 0
        M, N,           # Int32 — full input extents, for OOB masking
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
            TILE_X=self._TILE_X,
            TILE_Y=self._TILE_Y,
            WITH_ACT=cfg.WITH_ACT,
            WITH_DACT=cfg.WITH_DACT,
            sA_tile=sActInput_tile,
            WITH_DBIAS=self.DBIAS_REDUCTION_COLWISE,
            CACHE_ACTIVATION=self.CACHE_ACTIVATION,
        )

class MXFP8QuantizeSpecializedRowwiseKernel:
    """Specialized cast-only ROWWISE-only MXFP8 kernel. Requires N % 128 == 0 (full vectorizable column chunks).

    Plain rowwise-only quantize. Each thread owns one 32-element MXFP8 chunk and
    uses vectorized global loads/stores (no TMA used)."""

    _TILE_Y = 4
    _TILE_X = 1024
    _THREADS_PER_CTA = 128

    def __init__(self, cfg):
        self.cfg = cfg
        self.STASH_SCALE_TO_SMEM = True

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mO_row: Optional[cute.Tensor], mS_row: Optional[cute.Tensor],
        mO_col: Optional[cute.Tensor], mS_col: Optional[cute.Tensor], # Unused, kept for API compatibility
        mAmax: Optional[cute.Tensor], # Unused, kept for API compatibility
        mNoop: Optional[cute.Tensor], # Unused, kept for API compatibility
        mDActInput: Optional[cute.Tensor], # Unused, kept for API compatibility
        mWorkspace: Optional[cute.Tensor], # Unused, kept for API compatibility
        stream: CUstream,
    ):
        if cutlass.const_expr(CUTEDSL_DEBUG_LOGGING):
            cute.printf(f"[CuTeDSL] MXFP8QuantizeSpecializedRowwiseKernel.__call__() with config: {self.cfg}\n")

        M = mX.shape[0]
        N = mX.shape[1]

        grid = [
            cute.ceil_div(Int32(N), self._TILE_X),
            cute.ceil_div(M, self._TILE_Y),
        ]
        block = [self._THREADS_PER_CTA]

        self.kernel(
            mX, mO_row, mS_row, self.cfg.MAX_NORM_RCP, mX.element_type,
        ).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def kernel(self, mX, mO_row, mS_row, max_norm_rcp, DTYPE):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        M = mX.shape[0]
        N = mX.shape[1]

        # Each thread handles one 32-element MXFP8 chunk (= one scale block).
        # The 128 threads in the CTA are grouped as (4, 32), so they cover a
        # (4, 1024) input tile and the matching (4, 32) scale tile.
        CTA_Y = self._TILE_Y
        CTA_X = self._TILE_X // MXFP8_BLOCK_SIZE
        tiler, tv_layout = cute.make_layout_tv(
            thr_layout=cute.make_layout((CTA_Y, CTA_X), stride=(CTA_X, 1)),
            val_layout=cute.make_layout((1, MXFP8_BLOCK_SIZE),
                                        stride=(MXFP8_BLOCK_SIZE, 1)),
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
            cute.make_layout((1, MXFP8_BLOCK_SIZE), stride=(MXFP8_BLOCK_SIZE, 1)),
            dtype=DTYPE,
        )
        # Inputs widened to f32 once (reused by amax and the fused cvt). The FP8
        # output stays a uint8 fragment; we write it through a uint32 view so the
        # 4-wide mul_cvt drops one packed word per call (see the cvt loop).
        rX_f32 = cute.make_rmem_tensor(
            cute.make_layout((1, MXFP8_BLOCK_SIZE), stride=(MXFP8_BLOCK_SIZE, 1)),
            dtype=Float32,
        )
        rO_thread = cute.make_rmem_tensor(
            cute.make_layout((1, MXFP8_BLOCK_SIZE), stride=(MXFP8_BLOCK_SIZE, 1)),
            dtype=Uint8,
        )
        rO_u32 = cute.make_tensor(
            cute.recast_ptr(rO_thread.iterator, dtype=Uint32),
            cute.make_layout((MXFP8_BLOCK_SIZE // 4,), stride=(1,)),
        )

        # TODO: review this kernel myself.


        # Each thread owns only one scale byte, so a direct RF->GMEM scale write
        # can't vectorize (128 scattered 1-byte stores). If STASH_SCALE_TO_SMEM,
        # stage the CTA's (CTA_Y, CTA_X) scale tile in SMEM, then flush it to gmem
        # with wide (uint32) stores. Compile-time gated.
        sS_slot = None
        if cutlass.const_expr(self.STASH_SCALE_TO_SMEM):
            @cute.struct
            class SharedStorage:
                buf: cute.struct.Align[cute.struct.MemRange[Uint8, CTA_Y * CTA_X], 16]
            storage = cutlass.utils.SmemAllocator().allocate(SharedStorage)
            sScale = storage.buf.get_tensor(cute.make_layout((CTA_Y, CTA_X), stride=(CTA_X, 1)))
            sS_slot = cute.composition(sScale, tv_layout_scale)[tidx, None]
            # Zero first so padding columns (cols past N/32 in the padded scale
            # matrix) flush as 0 and we never read uninitialized smem.
            sS_slot[0] = Uint8(0)
            cute.arch.sync_threads()

        row = bidy * self._TILE_Y + tidx // CTA_X
        col = bidx * self._TILE_X + (tidx % CTA_X) * MXFP8_BLOCK_SIZE
        if row < M and col < N:
            cute.autovec_copy(mX_thread, rX_thread)

            # Widen once and reduce. bf16/fp16 -> f32 widening is exact, so the
            # amax matches the CUDA 16-bit abs_max path bit-for-bit.
            amax = Float32(0.0)
            for i in cutlass.range_constexpr(MXFP8_BLOCK_SIZE):
                rX_f32[0, i] = Float32(rX_thread[0, i])
                amax = cute.arch.fmax(amax, fabs_f32(rX_f32[0, i]))

            biased_exp = cvt_f32_to_fp8e8m0(amax * max_norm_rcp)
            if cutlass.const_expr(self.STASH_SCALE_TO_SMEM):
                sS_slot[0] = Uint8(biased_exp)
            else:
                mS_thread[0] = Uint8(biased_exp)

            # Rescale + FP8 cast, 4 elements per fused mul_cvt (one uint32 out),
            # then a vectorized store. Mirrors CUDA's _use_cvt_4x path.
            inv_scale = exp2f_rcp(biased_exp)
            scale_2x = pack_f32x2(inv_scale, inv_scale)
            mul_cvt4 = mul_cvt_f32x4_to_fp8x4(self.cfg.FP8_DTYPE)
            for w in cutlass.range_constexpr(MXFP8_BLOCK_SIZE // 4):
                b = 4 * w
                rO_u32[w] = mul_cvt4(rX_f32[0, b], rX_f32[0, b + 1],
                                     rX_f32[0, b + 2], rX_f32[0, b + 3], scale_2x)
            cute.autovec_copy(rO_thread, mO_thread)

        # Cooperative wide flush of the staged scales: the first CTA_Y*(CTA_X/G)
        # threads each store one G-wide group, the rest idle. Pick the widest store
        # the runtime row pitch allows (mirrors CUDA's PreferredDataType + runtime
        # check): uint4 (16 scale bytes) when padded_cols % 16 == 0, else uint32
        # (4 bytes, always safe since padded_cols is a multiple of 4). A group never
        # straddles the allocation boundary (base and padded_cols are both multiples
        # of G), so flush a group iff its first column is in-allocation. Zeroed
        # padding columns flush as 0.
        if cutlass.const_expr(self.STASH_SCALE_TO_SMEM):
            cute.arch.sync_threads()
            padded_cols = mS_row.shape[1]
            if padded_cols % 16 == 0:
                self._flush_scales(sScale, mS_tile, tidx, bidx, bidy, M, padded_cols, 16)
            else:
                self._flush_scales(sScale, mS_tile, tidx, bidx, bidy, M, padded_cols, 4)

    @cute.jit
    def _flush_scales(self, sScale, mS_tile, tidx, bidx, bidy, M, padded_cols, G):
        """Flush the staged (CTA_Y, CTA_X) scale tile to gmem with G-wide stores."""
        CTA_Y = self._TILE_Y
        CTA_X = self._TILE_X // MXFP8_BLOCK_SIZE
        GROUPS = CTA_X // G
        _, tv_flush = cute.make_layout_tv(
            thr_layout=cute.make_layout((CTA_Y, GROUPS), stride=(GROUPS, 1)),
            val_layout=cute.make_layout((1, G), stride=(G, 1)),
        )
        if tidx < CTA_Y * GROUPS:
            frow = tidx // GROUPS
            fgroup = tidx % GROUPS
            if bidy * CTA_Y + frow < M and bidx * CTA_X + fgroup * G < padded_cols:
                cute.autovec_copy(
                    cute.composition(sScale, tv_flush)[tidx, None],
                    cute.composition(mS_tile, tv_flush)[tidx, None],
                )

class MXFP8QuantizeSpecializedBidimensionalKernel:
    """Specialized cast-only BIDIMENSIONAL (both-direction) MXFP8 kernel — the
    CuTeDSL counterpart of specialized/quantize_mxfp8.cuh::
    quantize_mxfp8_kernel_cast_only<…,true,true> (non-warp-specialized).

    Plain both-direction quantize. TMA-based 32x32 warp tiles producing both the
    rowwise and colwise scales/outputs from one staged tile; handles any N % 32.

    (Kernel body — TMA pipeline + dual-direction scale/cast — is implemented in a
    later round; this is a dispatch stub that only logs the routing for now.)"""

    def __init__(self, cfg):
        self.cfg = cfg

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mO_row: Optional[cute.Tensor], mS_row: Optional[cute.Tensor],
        mO_col: Optional[cute.Tensor], mS_col: Optional[cute.Tensor],
        mAmax: Optional[cute.Tensor],
        mNoop: Optional[cute.Tensor],
        mDActInput: Optional[cute.Tensor],
        mWorkspace: Optional[cute.Tensor],
        stream: CUstream,
    ):
        if cutlass.const_expr(CUTEDSL_DEBUG_LOGGING):
            cute.printf(f"[CuTeDSL] MXFP8QuantizeSpecializedBidimensionalKernel.__call__() with config: {self.cfg}\n")
        # TODO(next round): TMA-based 32x32-tile bidimensional cast-only kernel —
        # grid/launch + dual-direction (rowwise+colwise) scale/cast. No output is
        # produced yet; this stub exists so dispatch routing can be wired now.


def get_specialized_kernel_class(cfg):
    """If no fusion is involved and the kernel only quantizes, dispatch to the specialized kernel for better performance."""
    plain_cast_only = (
        not cfg.WITH_GEMM_SWIZZLED_SCALES
        and not cfg.WITH_AMAX and not cfg.WITH_DBIAS
        and not cfg.WITH_DACT and not cfg.WITH_ACT and not cfg.WITH_NOOP
    )
    if plain_cast_only:
        if cfg.ROWWISE and not cfg.COLWISE:
            return MXFP8QuantizeSpecializedRowwiseKernel
        if cfg.ROWWISE and cfg.COLWISE:
            # TODO: dispatch to the bidimensional specialized kernel once implemented; for now, use the general kernel.
            return MXFP8QuantizeKernel
    return MXFP8QuantizeKernel


def compile_cutedsl_function_from_cfg(cfg):
    """
    Return the compiled CuTeDSL function object for the given MXFP8 quantization config.
    """

    # Route plain cast-only configs to the matching specialized kernel (mirrors the
    # CUDA dispatcher); everything else uses the general standard kernel.
    kernel_class = get_specialized_kernel_class(cfg)
    kernel_obj = kernel_class(cfg)
    # M, N must be divisible by the MXFP8 scale-block size (MXFP8_BLOCK_SIZE = 32) — the
    # same alignment the CUDA C++ kernel requires.
    sym_M = cute.sym_int32(divisibility=MXFP8_BLOCK_SIZE)
    sym_N = cute.sym_int32(divisibility=MXFP8_BLOCK_SIZE)
    in_shape = out_shape = (sym_M, sym_N)
    # TE allocates scale tensors at a padded shape (see
    # MXFP8Quantizer::get_scale_shape in transformer_engine/pytorch/csrc):
    #   rowwise:    (roundup(M, 128),     roundup(N // 32, 4))
    #   columnwise: (roundup(M // 32, 4), roundup(N, 128))
    # These padded extents are NOT M/N (and SymInt has no `//`/`+`), so give the
    # scales their own fresh syms carrying the divisibility the padding
    # guarantees (rowwise: 128 x 4; colwise: 4 x 128).
    scale_rowwise_shape = (cute.sym_int32(divisibility=128), cute.sym_int32(divisibility=4))
    scale_colwise_shape = (cute.sym_int32(divisibility=4),   cute.sym_int32(divisibility=128))
    ws_shape       = (cute.sym_int32(), sym_N)  # (blocks_Y, N); N ties to input N

    in_fake        = cute.runtime.make_fake_compact_tensor(cfg.DTYPE,  in_shape, stride_order=(1, 0), memspace=cute.AddressSpace.gmem, assumed_align=16)
    out_row_fake   = cute.runtime.make_fake_compact_tensor(cute.Uint8, out_shape, stride_order=(1, 0), memspace=cute.AddressSpace.gmem, assumed_align=16) if cfg.ROWWISE else None
    scale_row_fake = cute.runtime.make_fake_compact_tensor(cute.Uint8, scale_rowwise_shape, stride_order=(1, 0), memspace=cute.AddressSpace.gmem, assumed_align=4) if cfg.ROWWISE else None
    out_col_fake   = cute.runtime.make_fake_compact_tensor(cute.Uint8, out_shape, stride_order=(1, 0), memspace=cute.AddressSpace.gmem, assumed_align=16) if cfg.COLWISE else None
    scale_col_fake = cute.runtime.make_fake_compact_tensor(cute.Uint8, scale_colwise_shape, stride_order=(1, 0), memspace=cute.AddressSpace.gmem, assumed_align=4) if cfg.COLWISE else None
    amax_fake      = cute.runtime.make_fake_compact_tensor(Float32, (1,), stride_order=(0,), memspace=cute.AddressSpace.gmem, assumed_align=4) if cfg.WITH_AMAX else None
    noop_fake      = cute.runtime.make_fake_compact_tensor(Float32, (1,), stride_order=(0,), memspace=cute.AddressSpace.gmem, assumed_align=4) if cfg.WITH_NOOP else None
    act_input_fake = cute.runtime.make_fake_compact_tensor(cfg.DTYPE, in_shape, stride_order=(1, 0), memspace=cute.AddressSpace.gmem, assumed_align=16) if cfg.WITH_DACT  else None
    workspace_fake = cute.runtime.make_fake_compact_tensor(Float32, ws_shape, stride_order=(1, 0), memspace=cute.AddressSpace.gmem, assumed_align=4) if cfg.WITH_DBIAS else None

    compiled = cute.compile(
        kernel_obj,
        in_fake,                            # mX
        out_row_fake,   scale_row_fake,     # mO_row, mS_row
        out_col_fake,   scale_col_fake,     # mO_col, mS_col
        amax_fake,                          # mAmax
        noop_fake,                          # mNoop (1-element cast_noop flag)
        act_input_fake,                     # mDActInput (backward slot, unused)
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
    """Compile the MXFP8 quantize kernel for this config and register it in the TVM-FFI global registry 
    under EXACTLY `fn_name` (the key the C++ dispatcher built; Python treats it as an opaque name). 
    Returns True if a kernel is successfully registered under `fn_name` (the C++ side then fetches it with GetGlobal(fn_name));
    False if the config is unsupported, so the caller caches the negative result and falls back to the CUDA C++ kernel.
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
