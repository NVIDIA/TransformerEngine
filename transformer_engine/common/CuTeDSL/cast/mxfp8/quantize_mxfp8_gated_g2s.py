# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Gated-activation MXFP8 casts in CuTeDSL, ported from gated_mxfp8.cuh.

The kernel reuses MXFP8QuantizeGeneralG2SKernel's TMA pipeline and its plain-cast SMEM tile
quantizer. What it adds is an elementwise pass that runs on each staged tile before it is
quantized: it combines the act and gate halves of the input (and the incoming gradient y_grad,
in backward) and writes the result back into the act (and gate) input buffer in the input dtype.
That is exactly the truncation the CUDA kernel applies before computing block amaxes, so the
plain cast that follows reproduces its output.

Gated input is [M, 2N] with [act | gate] along the last dim. Every operand is addressed through
(M, N) views (stride 2N, column offset 0 or N), mirroring the CUDA kernel's tensor maps:
  forward : y[M, N] = act(x[:, :N]) * gate(x[:, N:])
  backward: [act_grad | gate_grad][M, 2N], the gradients of the act and gate halves given
            y_grad[M, N]
"""

from typing import Optional

import cutlass
from cutlass import cute
from cutlass import pipeline
from cutlass import Float32, Int32, Uint8, Uint16, Uint32
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module

from transformer_engine.common.CuTeDSL.utils import (
    exp2f_rcp,
    fabs_f32,
    is_packed16,
    pack_f32x2,
    select_f32,
    truncate_f32_bf16,
    truncate_f32_f16,
)
from transformer_engine.common.CuTeDSL.activations import (
    act_clamped_silu,
    act_situ_gate,
    dact_dsitu_gate,
    identity,
    didentity,
    situ_up,
    dsitu_up,
    act_silu,
    dact_dsilu,
    sigmoid,
)
from transformer_engine.common.CuTeDSL.utils_fp8 import (
    cvt_f32_to_fp8e8m0fnu,
    get_cvt_f32x2_to_fp8x2_func,
    mul_f32x2_cvt_f32x4_to_fp8x4,
)
from .quantize_mxfp8_common import (
    CUTEDSL_DEBUG_LOGGING,
    GATED_ACTIVATIONS,
    MXFP8GatedQuantizeConfig,
    MXFP8_BLOCK_SCALING_SIZE,
    THREADS_PER_WARP,
    derive_swizzled_scale_layout,
)
from .quantize_mxfp8_g2s_general import (
    MXFP8QuantizeGeneralG2SKernel,
    quantize_mxfp8_smem_tile_general,
)


@cute.jit
def gated_elementwise_op(
    act_element: Float32,
    gate_element: Float32,
    y_grad_element: Optional[Float32],
    limit: Optional[Float32],  # Clamped SwiGLU only
    alpha: Optional[Float32],  # Clamped SwiGLU only
    glu_linear_offset: Optional[Float32],  # Clamped SwiGLU only
    beta1: Optional[Float32],  # SiTU-GLU only
    beta2: Optional[Float32],  # SiTU-GLU only
    ACTIVATION: cutlass.Constexpr[str],
    IS_BWD: cutlass.Constexpr[bool],
):
    """
    FWD: (y, None)                y = act(a) * gate(g)
    BWD: (act_grad, gate_grad)    act_grad = y_grad * gate(g) * dact(a),
                                  gate_grad = y_grad * act(a) * dgate(g)
    """

    if cutlass.const_expr(ACTIVATION == "clamped_swiglu"):
        # FWD gate function for clamped_swiglu:
        # gate(g) = clamp(g, -limit, limit) + glu_linear_offset
        gate = cute.arch.fmin(cute.arch.fmax(-limit, gate_element), limit) + glu_linear_offset
        # BWD dgate function for clamped_swiglu:
        # dgate(g) = 0   if g < -limit or g > limit
        #          = 1   if -limit <= g <= limit
        dgate = select_f32(fabs_f32(gate_element) > limit, Float32(0.0), Float32(1.0))
    elif cutlass.const_expr(ACTIVATION == "situglu"):
        gate = situ_up(gate_element, beta2)
        dgate = dsitu_up(gate_element, beta2)
    else:
        gate = identity(gate_element)
        dgate = didentity(gate_element)

    act_fn, dact_fn = GATED_ACTIVATIONS[ACTIVATION]

    if cutlass.const_expr(not IS_BWD):
        if cutlass.const_expr(ACTIVATION == "clamped_swiglu"):
            act = act_clamped_silu(act_element, limit, alpha)
        elif cutlass.const_expr(ACTIVATION == "situglu"):
            act = act_situ_gate(act_element, beta1)
        else:
            act = act_fn(act_element)
        y = act * gate
        outputs = (y, None)
    else:
        if cutlass.const_expr(ACTIVATION == "clamped_swiglu"):
            x = cute.arch.fmin(act_element, limit)
            # Somehow CUDA always uses fast math for this and we should too to match its output
            s = sigmoid(alpha * x, fast_math=True)
            act = x * s
            dact = select_f32(
                act_element <= limit, s + s * (Float32(1.0) - s) * alpha * x, Float32(0.0)
            )
        elif cutlass.const_expr(ACTIVATION == "situglu"):
            act = act_situ_gate(act_element, beta1)
            dact = dact_dsitu_gate(act_element, beta1)
        elif cutlass.const_expr(ACTIVATION == "swiglu"):
            # Match CUDA implementation's behavior too
            s = sigmoid(act_element, fast_math=True)
            act = act_silu(act_element, s)
            dact = dact_dsilu(act_element, s)
        else:
            act = act_fn(act_element)
            dact = dact_fn(act_element)
        act_grad = dact * y_grad_element * gate
        gate_grad = dgate * act * y_grad_element
        outputs = (act_grad, gate_grad)
    return outputs


@cute.jit
def gated_elementwise_smem_tile(
    sAct_tile: cute.Tensor,  # also the y tile in FWD / the act_grad tile in BWD
    sGate_tile: cute.Tensor,  # also the gate_grad tile in BWD
    sYGrad_tile: Optional[cute.Tensor],  # BWD only
    limit: Optional[Float32],  # Clamped SwiGLU only
    alpha: Optional[Float32],  # Clamped SwiGLU only
    glu_linear_offset: Optional[Float32],  # Clamped SwiGLU only
    beta1: Optional[Float32],  # SiTU-GLU only
    beta2: Optional[Float32],  # SiTU-GLU only
    ACTIVATION: cutlass.Constexpr[str],
    IS_BWD: cutlass.Constexpr[bool],
    ROWWISE: cutlass.Constexpr[bool],
    DTYPE: cutlass.Constexpr[type],
    TILE_X: cutlass.Constexpr[int],
):
    """Compute the gated op on one staged tile, in place, rounded to the input dtype."""
    tidx, _, _ = cute.arch.thread_idx()

    _, tv_layout = cute.make_layout_tv(
        thr_layout=cute.make_layout((1, TILE_X), stride=(TILE_X, 1)),
        val_layout=cute.make_layout((MXFP8_BLOCK_SCALING_SIZE, 1), stride=(1, 1)),
    )

    sAct_tile = cute.composition(sAct_tile, tv_layout)[tidx, None]
    sGate_tile = cute.composition(sGate_tile, tv_layout)[tidx, None]
    if cutlass.const_expr(IS_BWD):
        sYGrad_tile = cute.composition(sYGrad_tile, tv_layout)[tidx, None]

    if cutlass.const_expr(is_packed16(DTYPE)):
        truncate_f32 = truncate_f32_f16 if DTYPE is cutlass.Float16 else truncate_f32_bf16

    for i in cutlass.range_constexpr(MXFP8_BLOCK_SCALING_SIZE):
        act_element = Float32(sAct_tile[i])
        gate_element = Float32(sGate_tile[i])
        # Round to the input dtype, like the CUDA kernels do before taking the amax. The cast to
        # DTYPE below is then exact.
        if cutlass.const_expr(IS_BWD):
            y_grad_element = Float32(sYGrad_tile[i])
            act_grad_element, gate_grad_element = gated_elementwise_op(
                act_element,
                gate_element,
                y_grad_element,
                limit,
                alpha,
                glu_linear_offset,
                beta1,
                beta2,
                ACTIVATION=ACTIVATION,
                IS_BWD=IS_BWD,
            )
            if cutlass.const_expr(is_packed16(DTYPE)):
                act_grad_element = truncate_f32(act_grad_element)
                gate_grad_element = truncate_f32(gate_grad_element)
            sAct_tile[i] = DTYPE(act_grad_element)
            sGate_tile[i] = DTYPE(gate_grad_element)
        else:
            y_element, _ = gated_elementwise_op(
                act_element,
                gate_element,
                None,
                limit,
                alpha,
                glu_linear_offset,
                beta1,
                beta2,
                ACTIVATION=ACTIVATION,
                IS_BWD=IS_BWD,
            )
            if cutlass.const_expr(is_packed16(DTYPE)):
                y_element = truncate_f32(y_element)
            sAct_tile[i] = DTYPE(y_element)

    # The rowwise quantization pass reads elements other threads wrote so we need to sync here
    if cutlass.const_expr(ROWWISE):
        cute.arch.sync_threads()


# Colwise-only: threads per column, each owning MXFP8_BLOCK_SCALING_SIZE // COLWISE_SPLIT rows
COLWISE_SPLIT = 2


@cute.jit
def gated_quantize_colwise_split_tile(
    sAct_tile: cute.Tensor,
    sGate_tile: cute.Tensor,
    sYGrad_tile: Optional[cute.Tensor],  # BWD only
    sO_col_tiles: list,  # FWD: [y]; BWD: [act_grad, gate_grad] colwise FP8 SMEM tiles
    mS_col_stages: list,  # their colwise scales
    sAmax: cute.Tensor,  # (outputs, COLWISE_SPLIT, TILE_X) f32 SMEM, partial block amaxes
    limit: Optional[Float32],
    alpha: Optional[Float32],
    glu_linear_offset: Optional[Float32],
    beta1: Optional[Float32],
    beta2: Optional[Float32],
    tile_row_start: Int32,
    tile_col_start: Int32,
    M: Int32,
    N: Int32,
    ACTIVATION: cutlass.Constexpr[str],
    IS_BWD: cutlass.Constexpr[bool],
    DTYPE: cutlass.Constexpr[type],
    FP8_DTYPE: cutlass.Constexpr[type],
    MAX_NORM_RCP: cutlass.Constexpr[float],
    SWIZZLE: cutlass.Constexpr[bool],
    TILE_X: cutlass.Constexpr[int],
):
    """Gated op fused with colwise quantization, COLWISE_SPLIT threads per column (as
    gated_mxfp8.cuh's colwise-only kernel). Values stay in registers; the threads of a column
    combine their partial amaxes through sAmax."""
    tidx, _, _ = cute.arch.thread_idx()
    tx = tidx % TILE_X
    ty = tidx // TILE_X
    ROWS = MXFP8_BLOCK_SCALING_SIZE // COLWISE_SPLIT
    _, tv_layout = cute.make_layout_tv(
        thr_layout=cute.make_layout((COLWISE_SPLIT, TILE_X), stride=(TILE_X, 1)),
        val_layout=cute.make_layout((ROWS, 1), stride=(1, 1)),
    )
    sAct_t = cute.composition(sAct_tile, tv_layout)[tidx, None]
    sGate_t = cute.composition(sGate_tile, tv_layout)[tidx, None]
    if cutlass.const_expr(IS_BWD):
        sYGrad_t = cute.composition(sYGrad_tile, tv_layout)[tidx, None]
    if cutlass.const_expr(is_packed16(DTYPE)):
        truncate_f32 = truncate_f32_f16 if DTYPE is cutlass.Float16 else truncate_f32_bf16

    NUM_OUT = 2 if IS_BWD else 1
    vals = [[None] * ROWS for _ in range(NUM_OUT)]
    for i in cutlass.range_constexpr(ROWS):
        y_grad = Float32(sYGrad_t[i]) if cutlass.const_expr(IS_BWD) else None
        out0, out1 = gated_elementwise_op(
            Float32(sAct_t[i]),
            Float32(sGate_t[i]),
            y_grad,
            limit,
            alpha,
            glu_linear_offset,
            beta1,
            beta2,
            ACTIVATION=ACTIVATION,
            IS_BWD=IS_BWD,
        )
        outs = (out0, out1) if cutlass.const_expr(IS_BWD) else (out0,)
        for o in cutlass.range_constexpr(NUM_OUT):
            # Round to the input dtype before the amax, like the CUDA kernel
            v = outs[o]
            if cutlass.const_expr(is_packed16(DTYPE)):
                v = truncate_f32(v)
            vals[o][i] = v

    for o in cutlass.range_constexpr(NUM_OUT):
        # NaN start: loses every comparison, so an all-NaN block keeps a NaN amax
        amax = Float32(float("nan"))
        for i in cutlass.range_constexpr(ROWS):
            amax = cute.arch.fmax(amax, fabs_f32(vals[o][i]))
        sAmax[o, ty, tx] = amax
    cute.arch.sync_threads()

    cvt_x2 = get_cvt_f32x2_to_fp8x2_func(FP8_DTYPE, False)
    for o in cutlass.range_constexpr(NUM_OUT):
        amax = sAmax[o, 0, tx]
        for t in cutlass.range_constexpr(1, COLWISE_SPLIT):
            amax = cute.arch.fmax(amax, sAmax[o, t, tx])
        biased_exp = cvt_f32_to_fp8e8m0fnu(amax * MAX_NORM_RCP)
        if ty == 0:
            if tile_row_start < M and tile_col_start + tx < N:
                if cutlass.const_expr(SWIZZLE):
                    mS_col_stages[o][(0, tx % 32, tx // 32)] = biased_exp
                else:
                    mS_col_stages[o][(0, tx)] = biased_exp
        inv_scale = exp2f_rcp(biased_exp)
        sO_t = cute.composition(sO_col_tiles[o], tv_layout)[tidx, None]
        sO_fp8 = cute.make_tensor(cute.recast_ptr(sO_t.iterator, dtype=FP8_DTYPE), sO_t.layout)
        for j in cutlass.range_constexpr(ROWS // 2):
            lo, hi = 2 * j, 2 * j + 1
            pair = cvt_x2(vals[o][hi] * inv_scale, vals[o][lo] * inv_scale)
            sO_fp8[lo] = Uint8(pair & Uint16(0xFF)).bitcast(FP8_DTYPE)
            sO_fp8[hi] = Uint8(pair >> Uint16(8)).bitcast(FP8_DTYPE)


@cute.jit
def gated_quantize_rowwise_tile(
    sAct_tile: cute.Tensor,
    sGate_tile: cute.Tensor,
    sYGrad_tile: Optional[cute.Tensor],  # BWD only
    sO_row_tiles: list,  # FWD: [y]; BWD: [act_grad, gate_grad] rowwise FP8 SMEM tiles
    mS_row_stages: list,  # their rowwise scales
    limit: Optional[Float32],
    alpha: Optional[Float32],
    glu_linear_offset: Optional[Float32],
    beta1: Optional[Float32],
    beta2: Optional[Float32],
    tile_row_start: Int32,
    tile_col_start: Int32,
    M: Int32,
    N: Int32,
    ACTIVATION: cutlass.Constexpr[str],
    IS_BWD: cutlass.Constexpr[bool],
    DTYPE: cutlass.Constexpr[type],
    FP8_DTYPE: cutlass.Constexpr[type],
    MAX_NORM_RCP: cutlass.Constexpr[float],
    TILE_X: cutlass.Constexpr[int],
    TILE_Y: cutlass.Constexpr[int],
):
    """Gated op fused with rowwise quantization, as gated_mxfp8.cuh does: each thread reads its
    32-element row block of every input, computes the op in registers and quantizes the result,
    with the general kernel's rowwise thread mapping and bank rotation (_quantize_rowwise_mxfp8),
    instead of staging the result through SMEM."""
    tidx, _, _ = cute.arch.thread_idx()
    PACK_SIZE = 4
    WAVES = MXFP8_BLOCK_SCALING_SIZE // PACK_SIZE
    THREADS_PER_BANK = 32 * 4 // MXFP8_BLOCK_SCALING_SIZE
    THREADS_X = TILE_X // MXFP8_BLOCK_SCALING_SIZE
    _, tv_layout = cute.make_layout_tv(
        thr_layout=cute.make_layout((TILE_Y, THREADS_X), stride=(THREADS_X, 1)),
        val_layout=cute.make_layout((1, MXFP8_BLOCK_SCALING_SIZE), stride=(0, 1)),
    )
    row_layout = cute.make_layout((MXFP8_BLOCK_SCALING_SIZE,), stride=(1,))

    def thread_row(tile):
        return cute.make_tensor(cute.composition(tile, tv_layout)[tidx, None].iterator, row_layout)

    sAct_t = thread_row(sAct_tile)
    sGate_t = thread_row(sGate_tile)
    if cutlass.const_expr(IS_BWD):
        sYGrad_t = thread_row(sYGrad_tile)
    if cutlass.const_expr(is_packed16(DTYPE)):
        truncate_f32 = truncate_f32_f16 if DTYPE is cutlass.Float16 else truncate_f32_bf16

    # Start each thread on a different bank, as _quantize_rowwise_mxfp8 does.
    offset = (tidx % THREADS_PER_WARP) // THREADS_PER_BANK * PACK_SIZE
    NUM_OUT = 2 if IS_BWD else 1
    vals = [[[None] * PACK_SIZE for _ in range(WAVES)] for _ in range(NUM_OUT)]
    # NaN start: loses every comparison, so an all-NaN block keeps a NaN amax
    amax = [Float32(float("nan")) for _ in range(NUM_OUT)]
    for w in cutlass.range_constexpr(WAVES):
        start = (w * PACK_SIZE + offset) % MXFP8_BLOCK_SCALING_SIZE
        for i in cutlass.range_constexpr(PACK_SIZE):
            y_grad = Float32(sYGrad_t[start + i]) if cutlass.const_expr(IS_BWD) else None
            out0, out1 = gated_elementwise_op(
                Float32(sAct_t[start + i]),
                Float32(sGate_t[start + i]),
                y_grad,
                limit,
                alpha,
                glu_linear_offset,
                beta1,
                beta2,
                ACTIVATION=ACTIVATION,
                IS_BWD=IS_BWD,
            )
            outs = (out0, out1) if cutlass.const_expr(IS_BWD) else (out0,)
            for o in cutlass.range_constexpr(NUM_OUT):
                # Round to the input dtype before the amax, like the CUDA kernel
                v = outs[o]
                if cutlass.const_expr(is_packed16(DTYPE)):
                    v = truncate_f32(v)
                vals[o][w][i] = v
                amax[o] = cute.arch.fmax(amax[o], fabs_f32(v))

    row = tidx // THREADS_X
    block = tidx % THREADS_X
    cvt_x4 = mul_f32x2_cvt_f32x4_to_fp8x4(FP8_DTYPE, False)
    for o in cutlass.range_constexpr(NUM_OUT):
        biased_exp = cvt_f32_to_fp8e8m0fnu(amax[o] * MAX_NORM_RCP)
        if tile_row_start + row < M and tile_col_start + block * MXFP8_BLOCK_SCALING_SIZE < N:
            mS_row_stages[o][(row, block)] = biased_exp
        inv_scale = exp2f_rcp(biased_exp)
        scale_2x = pack_f32x2(inv_scale, inv_scale)
        sO_u32 = cute.make_tensor(
            cute.recast_ptr(
                cute.composition(sO_row_tiles[o], tv_layout)[tidx, None].iterator, dtype=Uint32
            ),
            cute.make_layout((MXFP8_BLOCK_SCALING_SIZE // 4,), stride=(1,)),
        )
        for w in cutlass.range_constexpr(WAVES):
            idx = (w * PACK_SIZE + offset) % MXFP8_BLOCK_SCALING_SIZE // 4
            v = vals[o][w]
            sO_u32[idx] = cvt_x4(v[0], v[1], v[2], v[3], scale_2x)


@cute.jit
def quantize_mxfp8_smem_tile_general_partial(
    sX_tile: cute.Tensor,
    sO_row_tile: Optional[cute.Tensor],
    mS_row_stage: Optional[cute.Tensor],
    sO_col_tile: Optional[cute.Tensor],
    mS_col_stage: Optional[cute.Tensor],
    tile_row_start: Int32,
    tile_col_start: Int32,
    M: Int32,
    N: Int32,
    ROWWISE: cutlass.Constexpr[bool],
    COLWISE: cutlass.Constexpr[bool],
    MAX_NORM_RCP: cutlass.Constexpr[float],
    DTYPE: cutlass.Constexpr[type],
    FP8_DTYPE: cutlass.Constexpr[type],
    WITH_GEMM_SWIZZLED_SCALES: cutlass.Constexpr[bool],
    TILE_X: cutlass.Constexpr[int],
    TILE_Y: cutlass.Constexpr[int],
):
    """A partial quantize_mxfp8_smem_tile_general function with some args fixed."""
    quantize_mxfp8_smem_tile_general(
        sX_tile,
        None,
        sO_row_tile,
        mS_row_stage,
        sO_col_tile,
        mS_col_stage,
        None,
        tile_row_start,
        tile_col_start,
        M,
        N,
        ROWWISE=ROWWISE,
        COLWISE=COLWISE,
        WITH_AMAX=False,
        WITH_ACT=False,
        WITH_DACT=False,
        ACTIVATION=None,
        MAX_NORM_RCP=MAX_NORM_RCP,
        DTYPE=DTYPE,
        FP8_DTYPE=FP8_DTYPE,
        WITH_GEMM_SWIZZLED_SCALES=WITH_GEMM_SWIZZLED_SCALES,
        TILE_X=TILE_X,
        TILE_Y=TILE_Y,
        # The gated op maps the zeros TMA fills out-of-bounds elements with to zero.
        SKIP_INPUT_MASKING=True,
        SKIP_SCALE_BOUNDS=False,
        CACHE_ACTIVATION=False,
        DBIAS_REDUCTION_IN_COLWISE=False,
        COLWISE_DBIAS_REDUCTION_ONLY=False,
        DBIAS_REDUCTION_IN_ROWWISE=False,
    )


@cute.jit
def _split_gated_scales(
    mS_row: Optional[cute.Tensor],
    mS_col: Optional[cute.Tensor],
    M: Int32,
    N: Int32,
    ROWWISE: cutlass.Constexpr[bool],
    COLWISE: cutlass.Constexpr[bool],
    SWIZZLED: cutlass.Constexpr[bool],
):
    """Split the scales of a gated input with logical shape [M, 2N] into the act and gate halves, each with logical shape [M, N]."""
    mS_row_act = mS_row_gate = mS_col_act = mS_col_gate = None
    if cutlass.const_expr(SWIZZLED):
        mS_row, mS_col = derive_swizzled_scale_layout(M, 2 * N, ROWWISE, COLWISE, mS_row, mS_col)
        # To slice the left half (act) and right half (gate) of the swizzled scales, we need to tile using the tiler
        # with the same mode pattern as the swizzled layout's shape:
        # mS_row: ((32, 4, ceil(M/128)), (4, ceil(N/64))) : ((16, 4, 512·ceil(N/64)), (1, 512))
        # mS_col: ((4, ceil(M/128)), (32, 4, ceil(N/64))) : ((1, 512), (16, 4, 512·ceil(M/128)))
        if cutlass.const_expr(ROWWISE):
            row_tiler = (
                cute.make_layout((32, 4, cute.ceil_div(M, 128))),  # slice all rows
                cute.make_layout(
                    (4, cute.ceil_div(N, 128))
                ),  # slice half columns so cdiv uses 128 instead of 64
            )
            mS_row_act = cute.local_tile(mS_row, row_tiler, (0, 0))
            mS_row_gate = cute.local_tile(mS_row, row_tiler, (0, 1))
        if cutlass.const_expr(COLWISE):
            col_tiler = (
                cute.make_layout((4, cute.ceil_div(M, 128))),  # slice all rows
                cute.make_layout(
                    (32, 4, cute.ceil_div(N, 128))
                ),  # slice half columns so cdiv uses 128 instead of 64
            )
            mS_col_act = cute.local_tile(mS_col, col_tiler, (0, 0))
            mS_col_gate = cute.local_tile(mS_col, col_tiler, (0, 1))
    else:
        # Compact scales are row-major, so each half is a block of scale columns
        if cutlass.const_expr(ROWWISE):
            row_tiler = (M, cute.ceil_div(N, MXFP8_BLOCK_SCALING_SIZE))
            mS_row_act = cute.local_tile(mS_row, row_tiler, (0, 0))
            mS_row_gate = cute.local_tile(mS_row, row_tiler, (0, 1))
        if cutlass.const_expr(COLWISE):
            col_tiler = (cute.ceil_div(M, MXFP8_BLOCK_SCALING_SIZE), N)
            mS_col_act = cute.local_tile(mS_col, col_tiler, (0, 0))
            mS_col_gate = cute.local_tile(mS_col, col_tiler, (0, 1))
    return mS_row_act, mS_row_gate, mS_col_act, mS_col_gate


class MXFP8GatedQuantizeG2SKernel(MXFP8QuantizeGeneralG2SKernel):
    """gated_mxfp8.cuh in CuTeDSL. The general kernel's pipeline, loading the act and gate halves
    (and y_grad in BWD) each stage and storing one (forward) or two (backward) quantized halves.

    Inherits the plain-cast tile config, which is gated_mxfp8.cuh's: 64 threads per CTA, each CTA
    covering two vertically stacked 32x64 tiles. Rowwise-only and colwise-only fuse the gated op
    into the quantization in registers, like gated_mxfp8.cuh; colwise-only uses 128 threads (two per
    column), like its colwise-only kernel. Both directions keep the elementwise pass, which computes
    the op once for the two quantization passes.
    """

    def __init__(self, cfg: MXFP8GatedQuantizeConfig):
        super().__init__(cfg)
        # Same occupancy hint as the general kernel with a fused activation
        self._MIN_BLOCKS_PER_MP = 1
        # Colwise-only splits each column over COLWISE_SPLIT threads (gated_mxfp8.cuh's
        # THREADS_PER_CHUNK_COLWISE), doubling the resident warps at the same SMEM per CTA
        self.COLWISE_ONLY = cfg.COLWISE and not cfg.ROWWISE
        # Rowwise-only fuses the gated op into the rowwise quantization (no SMEM round trip)
        self.ROWWISE_ONLY = cfg.ROWWISE and not cfg.COLWISE
        if self.COLWISE_ONLY:
            self._THREADS_PER_CTA = COLWISE_SPLIT * self._TILE_COLS
            self._NUM_WARPS = self._THREADS_PER_CTA // 32
        # Every pipeline stage has its own SMEM buffers, so a CTA with fewer tiles than stages
        # would allocate buffers it never fills
        assert (
            self._NUM_STAGES <= self._NUM_TILES
        ), f"{self._NUM_STAGES} pipeline stages for {self._NUM_TILES} tiles per CTA"

    @cute.jit
    def __call__(
        self,
        mGatedInput: cute.Tensor,  # [M, 2N] = [act | gate]
        mYGrad: Optional[cute.Tensor],  # [M, N], BWD only
        # Outputs, with C = N (FWD) or 2N (BWD)
        mO_row: Optional[cute.Tensor],  # [M, C]
        mS_row: Optional[cute.Tensor],  # [roundup(M, 128), roundup(C / 32, 4)]
        mO_col: Optional[cute.Tensor],  # [M, C], not transposed
        mS_col: Optional[cute.Tensor],  # [roundup(ceil(M / 32), 4), roundup(C, 128)]
        # Special activation params, None if not used by the activation
        limit: Optional[Float32],  # Clamped SwiGLU only
        alpha: Optional[Float32],  # Clamped SwiGLU only
        glu_linear_offset: Optional[Float32],  # Clamped SwiGLU only
        beta1: Optional[Float32],  # SiTUGLU only
        beta2: Optional[Float32],  # SiTUGLU only
        stream: CUstream,
    ):
        """
        mGatedInput: [a | g] with shape M x 2N, where a is the activation half and g is the gate half.
        mYGrad: None for FWD, y_grad with shape [M, N] for BWD
        FWD mode:
        y = quantize(act(a) * gate(g)) with shape [M, N]
        BWD mode:
        [act_grad | gate_grad] = quantize([ y_grad * gate(g) * dact(a)  |  y_grad * act(a) * dgate(g) ]) with shape [M, 2N]
        """
        if cutlass.const_expr(CUTEDSL_DEBUG_LOGGING):
            cute.printf(
                f"[CuTeDSL] MXFP8GatedQuantizeG2SKernel.__call__() with config: {self.cfg}\n"
            )

        M = mGatedInput.shape[0]
        # Pick any mO that is used just to get N
        mO = mO_row if mO_row is not None else mO_col
        # FWD: mO is [M, N]; BWD: mYGrad is [M, N]
        N = mO.shape[1] if not self.cfg.IS_BWD else mYGrad.shape[1]

        # The act and gate halves of the [M, 2N] input
        mAct = cute.local_tile(mGatedInput, (M, N), (0, 0))
        mGate = cute.local_tile(mGatedInput, (M, N), (0, 1))

        mO_row_act_grad = mO_row_gate_grad = mO_col_act_grad = mO_col_gate_grad = None  # BWD only
        mS_row_act_grad = mS_row_gate_grad = mS_col_act_grad = mS_col_gate_grad = None  # BWD only
        if cutlass.const_expr(self.cfg.IS_BWD):
            # The [M, 2N] output is [act_grad | gate_grad]; the kernel takes its two halves
            mS_row_act_grad, mS_row_gate_grad, mS_col_act_grad, mS_col_gate_grad = (
                _split_gated_scales(
                    mS_row,
                    mS_col,
                    M,
                    N,
                    self.cfg.ROWWISE,
                    self.cfg.COLWISE,
                    self.cfg.WITH_GEMM_SWIZZLED_SCALES,
                )
            )
            if cutlass.const_expr(self.cfg.ROWWISE):
                mO_row_act_grad = cute.local_tile(mO_row, (M, N), (0, 0))
                mO_row_gate_grad = cute.local_tile(mO_row, (M, N), (0, 1))
            if cutlass.const_expr(self.cfg.COLWISE):
                mO_col_act_grad = cute.local_tile(mO_col, (M, N), (0, 0))
                mO_col_gate_grad = cute.local_tile(mO_col, (M, N), (0, 1))
            # In BWD mode we won't use these, and we should set them to None so we don't build TMA atoms for them
            mO_row = mO_col = mS_row = mS_col = None
        elif cutlass.const_expr(self.cfg.WITH_GEMM_SWIZZLED_SCALES):
            mS_row, mS_col = derive_swizzled_scale_layout(
                M, N, self.cfg.ROWWISE, self.cfg.COLWISE, mS_row, mS_col
            )

        atoms = self._make_tma_atoms(
            mYGrad,
            mAct,
            mGate,
            mO_row,
            mO_col,
            mO_row_act_grad,
            mO_row_gate_grad,
            mO_col_act_grad,
            mO_col_gate_grad,
        )
        # The TMA atoms are spread from a list, which pylint cannot count
        self.kernel(  # pylint: disable=no-value-for-parameter
            M,
            N,
            mS_row,
            mS_col,
            mS_row_act_grad,
            mS_row_gate_grad,
            mS_col_act_grad,
            mS_col_gate_grad,
            limit,
            alpha,
            glu_linear_offset,
            beta1,
            beta2,
            *atoms,
        ).launch(
            grid=[
                cute.ceil_div(N, self._TILE_COLS),
                cute.ceil_div(M, self._TILE_ROWS * self._NUM_TILES),
            ],
            block=[self._THREADS_PER_CTA],
            min_blocks_per_mp=self._MIN_BLOCKS_PER_MP,
            stream=stream,
        )

    @cute.jit
    def _make_tma_atoms(
        self,
        mYGrad: Optional[cute.Tensor],
        mAct: cute.Tensor,
        mGate: cute.Tensor,
        mO_row: Optional[cute.Tensor],  # FWD only
        mO_col: Optional[cute.Tensor],  # FWD only
        mO_row_act_grad: Optional[cute.Tensor],  # BWD only
        mO_row_gate_grad: Optional[cute.Tensor],  # BWD only
        mO_col_act_grad: Optional[cute.Tensor],  # BWD only
        mO_col_gate_grad: Optional[cute.Tensor],  # BWD only
    ):
        """Build TMA atoms for input / quantized output; scales will not be loaded / stored by TMA."""
        smem_tile_layout = cute.make_ordered_layout(
            (self._TILE_ROWS, self._TILE_COLS), order=(1, 0)
        )
        tiler = (self._TILE_ROWS, self._TILE_COLS)
        op_load = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        op_store = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()

        atoms = []
        for op, mT in (
            (op_load, mYGrad),
            (op_load, mAct),
            (op_load, mGate),
            (op_store, mO_row),
            (op_store, mO_col),
            (op_store, mO_row_act_grad),
            (op_store, mO_row_gate_grad),
            (op_store, mO_col_act_grad),
            (op_store, mO_col_gate_grad),
        ):
            if cutlass.const_expr(mT is None):
                atoms.extend([None, None])
            else:
                atoms.extend(
                    cute.nvgpu.cpasync.make_tiled_tma_atom(
                        op, mT, smem_tile_layout, tiler, num_multicast=1
                    )
                )
        return atoms

    @cute.kernel
    def kernel(
        self,
        M: Int32,
        N: Int32,
        mS_row: Optional[cute.Tensor],  # FWD only
        mS_col: Optional[cute.Tensor],  # FWD only
        mS_row_act_grad: Optional[cute.Tensor],  # BWD only
        mS_row_gate_grad: Optional[cute.Tensor],  # BWD only
        mS_col_act_grad: Optional[cute.Tensor],  # BWD only
        mS_col_gate_grad: Optional[cute.Tensor],  # BWD only
        limit: Optional[Float32],
        alpha: Optional[Float32],
        glu_linear_offset: Optional[Float32],
        beta1: Optional[Float32],
        beta2: Optional[Float32],
        # TMA for mYGrad, only used in BWD
        tma_atom_y_grad: Optional[cute.CopyAtom],
        tma_src_y_grad: Optional[cute.Tensor],
        # TMA for mAct and mGate, used in both FWD and BWD
        tma_atom_act: cute.CopyAtom,
        tma_src_act: cute.Tensor,
        tma_atom_gate: cute.CopyAtom,
        tma_src_gate: cute.Tensor,
        # TMA for mO_row/col, only used in FWD
        tma_atom_row: Optional[cute.CopyAtom],
        tma_dst_row: Optional[cute.Tensor],
        tma_atom_col: Optional[cute.CopyAtom],
        tma_dst_col: Optional[cute.Tensor],
        # TMA for mO_row/col_act_grad/gate_grad, only used in BWD
        tma_atom_row_act_grad: Optional[cute.CopyAtom],
        tma_dst_row_act_grad: Optional[cute.Tensor],
        tma_atom_row_gate_grad: Optional[cute.CopyAtom],
        tma_dst_row_gate_grad: Optional[cute.Tensor],
        tma_atom_col_act_grad: Optional[cute.CopyAtom],
        tma_dst_col_act_grad: Optional[cute.Tensor],
        tma_atom_col_gate_grad: Optional[cute.CopyAtom],
        tma_dst_col_gate_grad: Optional[cute.Tensor],
    ):
        """
        FWD mode:
        mAct: activation half input
        mGate: gate half input
        mO_row/col: quantized output for row / column directions
        mS_row/col: output scales for row / column directions

        BWD mode:
        mYGrad: y_grad, the incoming gradient
        mAct: activation half input from FWD pass
        mGate: gate half input from FWD pass
        mO_row/col_act_grad: quantized activation gradient for row / column directions
        mO_row/col_gate_grad: quantized gate gradient for row / column directions
        mS_row/col_act_grad: output scales for activation gradient row / column directions
        mS_row/col_gate_grad: output scales for gate gradient row / column directions
        mO_row/col: quantized gradient (both activation and gate) for row / column directions
        mS_row/col: output scales for gradient (both activation and gate) row / column directions

        mAct: both FWD and BWD
        mGate: both FWD and BWD
        mYGrad: BWD only
        mO_row/col: both FWD (N) and BWD (2N)
        mS_row/col: both FWD (N) and BWD (2N)
        mO_row/col_act_grad: BWD only
        mO_row/col_gate_grad: BWD only
        mS_row/col_act_grad: BWD only
        mS_row/col_gate_grad: BWD only
        """

        cfg = self.cfg
        DTYPE = cfg.DTYPE
        TILE_ROWS = self._TILE_ROWS
        TILE_COLS = self._TILE_COLS
        NUM_STAGES = self._NUM_STAGES
        NUM_TILES = self._NUM_TILES

        # The scales of one (TILE_ROWS, TILE_COLS) tile
        row_scale_tile = (TILE_ROWS, TILE_COLS // MXFP8_BLOCK_SCALING_SIZE)
        col_scale_tile = (TILE_ROWS // MXFP8_BLOCK_SCALING_SIZE, TILE_COLS)

        # Shared memory: one ((TILE_ROWS, TILE_COLS), NUM_STAGES) ring buffer per operand
        smem = cutlass.memory.SmemAllocator()

        @cute.struct
        class BarrierStorage:  # pylint: disable=missing-class-docstring
            mbar_storage: cute.struct.MemRange[cute.Int64, 2 * NUM_STAGES]

        storage = smem.allocate(BarrierStorage)
        stages_layout = cute.make_layout(
            ((TILE_ROWS, TILE_COLS), NUM_STAGES),
            stride=((TILE_COLS, 1), TILE_ROWS * TILE_COLS),
        )

        # Allocate SMEM buffers for inputs
        sAct = smem.allocate_tensor(DTYPE, stages_layout, byte_alignment=128)
        sGate = smem.allocate_tensor(DTYPE, stages_layout, byte_alignment=128)
        sYGrad = None
        if cutlass.const_expr(cfg.IS_BWD):
            sYGrad = smem.allocate_tensor(DTYPE, stages_layout, byte_alignment=128)

        # Allocate SMEM buffers for outputs
        sO_row = sO_col = None
        sO_row_act_grad = sO_row_gate_grad = sO_col_act_grad = sO_col_gate_grad = None
        if cutlass.const_expr(cfg.ROWWISE):
            if cutlass.const_expr(cfg.IS_BWD):
                sO_row_act_grad = smem.allocate_tensor(
                    cfg.FP8_DTYPE, stages_layout, byte_alignment=128
                )
                sO_row_gate_grad = smem.allocate_tensor(
                    cfg.FP8_DTYPE, stages_layout, byte_alignment=128
                )
            else:
                sO_row = smem.allocate_tensor(cfg.FP8_DTYPE, stages_layout, byte_alignment=128)
        if cutlass.const_expr(cfg.COLWISE):
            if cutlass.const_expr(cfg.IS_BWD):
                sO_col_act_grad = smem.allocate_tensor(
                    cfg.FP8_DTYPE, stages_layout, byte_alignment=128
                )
                sO_col_gate_grad = smem.allocate_tensor(
                    cfg.FP8_DTYPE, stages_layout, byte_alignment=128
                )
            else:
                sO_col = smem.allocate_tensor(cfg.FP8_DTYPE, stages_layout, byte_alignment=128)

        sAmax = None
        if cutlass.const_expr(self.COLWISE_ONLY):
            # One slot per output, so FP32 forward keeps the CUDA kernel's 6 CTAs per SM
            num_outputs = 2 if cfg.IS_BWD else 1
            sAmax = smem.allocate_tensor(
                Float32,
                cute.make_layout((num_outputs, COLWISE_SPLIT, TILE_COLS)),
                byte_alignment=16,
            )

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, bidy, _ = cute.arch.block_idx()

        # Issue TMA descriptor prefetch using the TMA warp (warp 0)
        if warp_idx == 0:
            for load_atom in (tma_atom_act, tma_atom_gate, tma_atom_y_grad):
                if cutlass.const_expr(load_atom is not None):
                    cute.nvgpu.cpasync.prefetch_descriptor(load_atom)

        # TMA warp (warp 0)'s elected thread produces (issues TMA), every warp consumes
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self._NUM_WARPS)
        # Input has 2 tiles (act + gate) in FWD, 3 tiles (y_grad + act + gate) in BWD
        num_inputs = 3 if cfg.IS_BWD else 2
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_storage.data_ptr(),
            num_stages=NUM_STAGES,
            producer_group=producer_group,
            consumer_group=consumer_group,
            # Every input tile of a stage arrives on the same barrier
            tx_count=num_inputs * TILE_ROWS * TILE_COLS * DTYPE.width // 8,
            cta_layout_vmnk=None,
        )
        prod_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, NUM_STAGES)
        cons_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, NUM_STAGES)

        cta_coord = (bidy, bidx)
        # Shape of each individual tile. A CTA processes one tile at a time
        tiler = (TILE_ROWS, TILE_COLS)
        # The block processed by a CTA, consisting of NUM_TILES vertically stacked tiles
        cta_block = (NUM_TILES * TILE_ROWS, TILE_COLS)

        # Prepare parameters to pass to the SMEM tile quantize function per output (FWD: y; BWD:
        # act_grad, gate_grad), where input and quantized output are SMEM tiles and scales are GMEM
        # tiles
        # Prepare parameters to pass to the SMEM tile quantize function for all outputs
        # (FWD: y; BWD: act_grad, gate_grad)
        # Note: input, quantized output are SMEM tiles, and scales are GMEM tiles
        cta_block_row_scales = (NUM_TILES * TILE_ROWS, TILE_COLS // MXFP8_BLOCK_SCALING_SIZE)
        cta_block_col_scales = (NUM_TILES * TILE_ROWS // MXFP8_BLOCK_SCALING_SIZE, TILE_COLS)
        if cutlass.const_expr(cfg.IS_BWD):
            if cutlass.const_expr(cfg.ROWWISE):
                mS_row_act_grad = cute.local_tile(mS_row_act_grad, cta_block_row_scales, cta_coord)
                mS_row_gate_grad = cute.local_tile(
                    mS_row_gate_grad, cta_block_row_scales, cta_coord
                )
            if cutlass.const_expr(cfg.COLWISE):
                mS_col_act_grad = cute.local_tile(mS_col_act_grad, cta_block_col_scales, cta_coord)
                mS_col_gate_grad = cute.local_tile(
                    mS_col_gate_grad, cta_block_col_scales, cta_coord
                )
            # In BWD mode we output gradient for both the act and gate halves, which are stored in sAct and sGate respectively
            tile_quantizer_params = [
                (
                    sAct,
                    sO_row_act_grad,
                    mS_row_act_grad,
                    sO_col_act_grad,
                    mS_col_act_grad,
                ),  # We will write unquantized act_grad to sAct in-place
                (
                    sGate,
                    sO_row_gate_grad,
                    mS_row_gate_grad,
                    sO_col_gate_grad,
                    mS_col_gate_grad,
                ),  # We will write unquantized gate_grad to sGate in-place
            ]
        else:
            if cutlass.const_expr(cfg.ROWWISE):
                mS_row = cute.local_tile(mS_row, cta_block_row_scales, cta_coord)
            if cutlass.const_expr(cfg.COLWISE):
                mS_col = cute.local_tile(mS_col, cta_block_col_scales, cta_coord)
            # In FWD mode we only output the result after the gated op, which is stored in sAct
            tile_quantizer_params = [
                (sAct, sO_row, mS_row, sO_col, mS_col)
            ]  # We will write unquantized output to sAct in-place

        # Prepare input TMA (atom, smem partition, gmem partition) triples: act and gate, and BWD's y_grad
        TMA_load_atom_and_partitions = []
        for atom_in, src_in, sX_in in (
            (tma_atom_act, tma_src_act, sAct),
            (tma_atom_gate, tma_src_gate, sGate),
            (tma_atom_y_grad, tma_src_y_grad, sYGrad),
        ):
            if cutlass.const_expr(atom_in is not None):
                tXsX_in, tXgX_in = cute.nvgpu.cpasync.tma_partition(
                    atom_in,
                    0,
                    cute.make_layout(1),
                    sX_in,
                    cute.zipped_divide(cute.local_tile(src_in, cta_block, cta_coord), tiler),
                )
                TMA_load_atom_and_partitions.append((atom_in, tXsX_in, tXgX_in))

        # Prepare output TMA (atom, smem partition, gmem partition) triples of the outputs this config has
        TMA_store_atom_and_partitions = []
        for atom_out, dst_out, sO_out in (
            (tma_atom_row, tma_dst_row, sO_row),  # FWD only
            (tma_atom_col, tma_dst_col, sO_col),  # FWD only
            (tma_atom_row_act_grad, tma_dst_row_act_grad, sO_row_act_grad),  # BWD only
            (tma_atom_row_gate_grad, tma_dst_row_gate_grad, sO_row_gate_grad),  # BWD only
            (tma_atom_col_act_grad, tma_dst_col_act_grad, sO_col_act_grad),  # BWD only
            (tma_atom_col_gate_grad, tma_dst_col_gate_grad, sO_col_gate_grad),  # BWD only
        ):
            if cutlass.const_expr(atom_out is not None):
                tXsO_out, tXgO_out = cute.nvgpu.cpasync.tma_partition(
                    atom_out,
                    0,
                    cute.make_layout(1),
                    sO_out,
                    cute.zipped_divide(cute.local_tile(dst_out, cta_block, cta_coord), tiler),
                )
                TMA_store_atom_and_partitions.append((atom_out, tXsO_out, tXgO_out))

        # Barrier init must be visible to every thread
        cute.arch.sync_threads()

        # Prologue: fill the pipeline
        if warp_idx == 0:
            for s in cutlass.range_constexpr(NUM_STAGES):
                mainloop_pipeline.producer_acquire(prod_state)
                for load_atom, load_sX, load_gX in TMA_load_atom_and_partitions:
                    cute.copy(
                        load_atom,
                        load_gX[(None, (s, 0))],
                        load_sX[(None, prod_state.index)],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(prod_state),
                    )
                mainloop_pipeline.producer_commit(prod_state)
                prod_state.advance()

        for tile_idx in cutlass.range(NUM_TILES, unroll=1):
            mainloop_pipeline.consumer_wait(cons_state)
            # We only reuse previously used SMEM buffers when we have more tiles to process than pipeline stages, and only
            # after we've already finished one round and wrap around to the start of the ring buffer
            if cutlass.const_expr(NUM_TILES > NUM_STAGES):
                if tile_idx >= NUM_STAGES:
                    # Only allow at most NUM_STAGES - 1 store groups reading SMEM on the fly, so the tile NUM_STAGES
                    # ago must finish before we reuse its SMEM buffer
                    if warp_idx == 0:
                        cute.arch.cp_async_bulk_wait_group(NUM_STAGES - 1, read=True)
                    cute.arch.sync_threads()
            # The tile should be ready to use now

            stage_idx = cons_state.index
            tile_coord = (tile_idx, 0)  # within this CTA's block

            # Select current stage's SMEM tile
            sAct_tile = sAct[(None, stage_idx)]
            sGate_tile = sGate[(None, stage_idx)]
            sYGrad_tile = None
            if cutlass.const_expr(cfg.IS_BWD):
                sYGrad_tile = sYGrad[(None, stage_idx)]

            if cutlass.const_expr(self.COLWISE_ONLY):
                gated_quantize_colwise_split_tile(
                    sAct_tile,
                    sGate_tile,
                    sYGrad_tile,
                    [
                        sO_col_out[(None, stage_idx)]
                        for _, _, _, sO_col_out, _ in tile_quantizer_params
                    ],
                    [
                        cute.flatten(cute.local_tile(mS_col_cta, col_scale_tile, tile_coord))
                        for _, _, _, _, mS_col_cta in tile_quantizer_params
                    ],
                    sAmax,
                    limit,
                    alpha,
                    glu_linear_offset,
                    beta1,
                    beta2,
                    (bidy * NUM_TILES + tile_idx) * TILE_ROWS,
                    bidx * TILE_COLS,
                    M,
                    N,
                    ACTIVATION=cfg.GATED_ACTIVATION,
                    IS_BWD=cfg.IS_BWD,
                    DTYPE=DTYPE,
                    FP8_DTYPE=cfg.FP8_DTYPE,
                    MAX_NORM_RCP=cfg.MAX_NORM_RCP,
                    SWIZZLE=cfg.WITH_GEMM_SWIZZLED_SCALES,
                    TILE_X=TILE_COLS,
                )
            elif cutlass.const_expr(self.ROWWISE_ONLY):
                gated_quantize_rowwise_tile(
                    sAct_tile,
                    sGate_tile,
                    sYGrad_tile,
                    [
                        sO_row_out[(None, stage_idx)]
                        for _, sO_row_out, _, _, _ in tile_quantizer_params
                    ],
                    [
                        cute.flatten(cute.local_tile(mS_row_cta, row_scale_tile, tile_coord))
                        for _, _, mS_row_cta, _, _ in tile_quantizer_params
                    ],
                    limit,
                    alpha,
                    glu_linear_offset,
                    beta1,
                    beta2,
                    (bidy * NUM_TILES + tile_idx) * TILE_ROWS,
                    bidx * TILE_COLS,
                    M,
                    N,
                    ACTIVATION=cfg.GATED_ACTIVATION,
                    IS_BWD=cfg.IS_BWD,
                    DTYPE=DTYPE,
                    FP8_DTYPE=cfg.FP8_DTYPE,
                    MAX_NORM_RCP=cfg.MAX_NORM_RCP,
                    TILE_X=TILE_COLS,
                    TILE_Y=TILE_ROWS,
                )
            else:
                # Compute the gated op on this tile in fp32, cast back to the original dtype,
                # and store the result in-place in the SMEM tile without quantization.
                # FWD: write the output into sAct_tile
                # BWD: write the act half's gradient to sAct_tile, and write the gate half's gradient to sGate_tile
                gated_elementwise_smem_tile(
                    sAct_tile,  # we also store the FWD output / BWD act half's gradient here
                    sGate_tile,  # we also store the BWD gate half's gradient here
                    sYGrad_tile,
                    limit,
                    alpha,
                    glu_linear_offset,
                    beta1,
                    beta2,
                    ACTIVATION=cfg.GATED_ACTIVATION,
                    IS_BWD=cfg.IS_BWD,
                    ROWWISE=cfg.ROWWISE,
                    DTYPE=DTYPE,
                    TILE_X=TILE_COLS,
                )

                # Quantize every output of this tile into its SMEM buffers and scales
                for sX_out, sO_row_out, mS_row_cta, sO_col_out, mS_col_cta in tile_quantizer_params:
                    # Slice the SMEM / GMEM tile for the current iteration
                    sO_row_tile = None
                    mS_row_stage = None
                    if cutlass.const_expr(cfg.ROWWISE):
                        sO_row_tile = sO_row_out[(None, stage_idx)]
                        mS_row_stage = cute.flatten(
                            cute.local_tile(mS_row_cta, row_scale_tile, tile_coord)
                        )
                    sO_col_tile = None
                    mS_col_stage = None
                    if cutlass.const_expr(cfg.COLWISE):
                        sO_col_tile = sO_col_out[(None, stage_idx)]
                        mS_col_stage = cute.flatten(
                            cute.local_tile(mS_col_cta, col_scale_tile, tile_coord)
                        )
                    # Use the general MXFP8 SMEM tile quantize function to quantize it
                    quantize_mxfp8_smem_tile_general_partial(
                        sX_out[(None, stage_idx)],  # The SMEM tile to quantize
                        sO_row_tile,  # SMEM tile for rowwise quantized output
                        mS_row_stage,  # GMEM tile for rowwise quantized scales
                        sO_col_tile,  # SMEM tile for columnwise quantized output
                        mS_col_stage,  # GMEM tile for columnwise quantized scales
                        # The tile's first row and column in the whole tensor, for bounds checks
                        (bidy * NUM_TILES + tile_idx) * TILE_ROWS,
                        bidx * TILE_COLS,
                        M,
                        N,
                        ROWWISE=cfg.ROWWISE,
                        COLWISE=cfg.COLWISE,
                        MAX_NORM_RCP=cfg.MAX_NORM_RCP,
                        DTYPE=DTYPE,
                        FP8_DTYPE=cfg.FP8_DTYPE,
                        WITH_GEMM_SWIZZLED_SCALES=cfg.WITH_GEMM_SWIZZLED_SCALES,
                        TILE_X=TILE_COLS,
                        TILE_Y=TILE_ROWS,
                    )

            # Make the SMEM writes visible to the TMA's async proxy before it reads them
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()
            mainloop_pipeline.consumer_release(cons_state)

            if warp_idx == 0:
                # The tile quantize function has already written the quantized outputs to SMEM,
                # so we can issue TMA stores from SMEM to GMEM
                for store_atom, store_sO, store_gO in TMA_store_atom_and_partitions:
                    cute.copy(
                        store_atom,
                        store_sO[(None, stage_idx)],
                        store_gO[(None, (tile_idx, 0))],
                    )
                cute.arch.cp_async_bulk_commit_group()

            cons_state.advance()

            # Refill the stage just released with the tile NUM_STAGES ahead
            if warp_idx == 0:
                next_tile_idx = tile_idx + NUM_STAGES
                if next_tile_idx < NUM_TILES:
                    mainloop_pipeline.producer_acquire(prod_state)
                    for load_atom, load_sX, load_gX in TMA_load_atom_and_partitions:
                        cute.copy(
                            load_atom,
                            load_gX[(None, (next_tile_idx, 0))],
                            load_sX[(None, prod_state.index)],
                            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(prod_state),
                        )
                    mainloop_pipeline.producer_commit(prod_state)
                    prod_state.advance()

        # Only wait until the in-flight TMA stores have read SMEM, so the CTA retires early
        cute.arch.cp_async_bulk_wait_group(0, read=True)
