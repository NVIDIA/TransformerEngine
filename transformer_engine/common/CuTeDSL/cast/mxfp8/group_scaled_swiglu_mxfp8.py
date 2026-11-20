# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Grouped scaled SwiGLU fused with columnwise MXFP8 quantization, in CuTeDSL (GMEM -> RMEM).

Port of group_scaled_swiglu_mxfp8.cuh, the MoE backward recompute of the FC2 input:

    input  : x [M, 2H] = [act | gate], the FC1 output of a group stacked along the tokens
    prob   : [M] per-token router weight, in the input dtype
    output : columnwise MXFP8 of (silu(act) * gate) * prob, [M, H]

Clamped SwiGLU clamps act from above and gate on both sides, then offsets gate, as
gated_mxfp8.cuh's forward does. The C++ bridge is group_scaled_swiglu_mxfp8_cutedsl.cuh.
"""

import logging
from typing import Optional

import cutlass
from cutlass import cute
from cutlass import Boolean, Float32, Int32, Int64, Uint8, Uint16, Uint32, Float8E8M0FNU
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module

from transformer_engine.common.CuTeDSL.activations import scaled_swiglu_approx_x2_pair
from transformer_engine.common.CuTeDSL.utils import (
    abs_max_x2,
    bf16_mx_scale_bytes_pair,
    bf16_mx_scale_reciprocal_pair,
    exp2f_rcp,
    fabs_f32,
    pack_f32x2,
    str_to_cutlass_dtype,
    unpack_i64_to_i32x2,
    x2_hi_to_f32_f16,
    x2_lo_to_f32_f16,
    zero_b32_reg,
)
from transformer_engine.common.CuTeDSL.utils_fp8 import (
    cvt_f32_to_fp8e8m0fnu,
    mul_bf16x2x2_cvt_bf16x4_to_fp8x4,
    mul_f32x2_cvt_x2_to_fp8x2,
)
from .quantize_mxfp8_common import (
    FP8E4M3_MAX_NORM_RCP,
    FP8E5M2_MAX_NORM_RCP,
    MXFP8_BLOCK_SCALING_SIZE,
    THREADS_PER_WARP,
    _fake_gmem_tensor,
    compile_and_register_function,
    derive_swizzled_scale_layout,
    derive_swizzled_scale_vectorized_tv_layout,
    noop_flag_is_set,
)
from .group_quantize_mxfp8_common import (
    SAME_BOTH_DIMS,
    VARYING_FIRST_DIM,
    find_tensor_from_offsets,
)

logger = logging.getLogger("transformer_engine.cutedsl.mxfp8")

SCALED_SWIGLU_ACTIVATIONS = ("swiglu", "clamped_swiglu")


@cute.jit
def scaled_swiglu_quantize_mxfp8_gmem_tile_colwise(
    gAct_tile: cute.Tensor,  # (TILE_ROWS, TILE_COLS) act half of the input
    gGate_tile: cute.Tensor,  # (TILE_ROWS, TILE_COLS) gate half of the input
    gProb_tile: cute.Tensor,  # (TILE_ROWS,) per-row prob, in the input dtype
    gO_col_tile: cute.Tensor,  # (TILE_ROWS, TILE_COLS) columnwise FP8 output
    sS_col_tile: cute.Tensor,  # Uint8 colwise scales of the tile, in their GMEM order
    valid_cols: Int64,  # the tile's columns before the tensor ends
    limit: Optional[Float32],  # clamped SwiGLU only
    alpha: Optional[Float32],  # clamped SwiGLU only
    glu_linear_offset: Optional[Float32],  # clamped SwiGLU only
    DTYPE: cutlass.Constexpr[type],
    FP8_DTYPE: cutlass.Constexpr[type],
    MAX_NORM_RCP: cutlass.Constexpr[float],
    CLAMPED: cutlass.Constexpr[bool],
    WITH_GEMM_SWIZZLED_SCALES: cutlass.Constexpr[bool],
    WARPS_X: cutlass.Constexpr[int],
    ELEMENTS_Y_PER_THREAD: cutlass.Constexpr[int],
    ELEMENTS_X_PER_THREAD: cutlass.Constexpr[int],
):
    """Columnwise MXFP8 of (silu(act) * gate) * prob over one GMEM tile, in registers.

    Each warp owns a 32-row scaling block. A thread loads ELEMENTS_Y_PER_THREAD rows of
    ELEMENTS_X_PER_THREAD columns of both halves with one 16-byte load per row, and the
    lanes that share columns combine their amaxes by shuffles. The scales are staged in
    sS_col_tile, compact (row-major) or as one 128 x 4 GEMM-swizzled tile; columns past
    valid_cols get zero scales and no output.
    """
    _, TILE_COLS = gO_col_tile.shape
    ELEMENTS_X = ELEMENTS_X_PER_THREAD
    ELEMENTS_Y = ELEMENTS_Y_PER_THREAD
    THREAD_SHAPE = (ELEMENTS_Y, ELEMENTS_X)
    PAIRS_X = ELEMENTS_X // 2
    IS_BF16 = DTYPE is cutlass.BFloat16
    IS_FP32 = DTYPE is Float32
    # A pair of adjacent elements: one packed register for 16-bit input, two for FP32
    PAIR_DTYPE = Int64 if IS_FP32 else Int32
    # Shape of a warp
    LANES_Y = MXFP8_BLOCK_SCALING_SIZE // ELEMENTS_Y
    LANES_X = THREADS_PER_WARP // LANES_Y
    WARP_COLS = LANES_X * ELEMENTS_X
    # The warps sweep the tile's columns in COL_ITERS steps
    COL_ITERS = TILE_COLS // (WARPS_X * WARP_COLS)
    assert COL_ITERS * WARPS_X * WARP_COLS == TILE_COLS
    # log2 of the FP8 format's largest power of two, for the BF16 scale helpers
    MAX_EXPONENT = 8 if FP8_DTYPE is cutlass.Float8E4M3FN else 15

    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane_idx = cute.arch.lane_idx()
    warp_y = warp_idx // WARPS_X
    warp_x = warp_idx % WARPS_X
    lane_y = lane_idx // LANES_X
    lane_x = lane_idx % LANES_X
    # This thread's rows, as a tile coordinate in units of ELEMENTS_Y rows
    thread_row_coord = warp_y * LANES_Y + lane_y

    # Each lane loads the prob of one row of the block and hands it to the lanes on that row via warp shuffles
    half_prob_lane = Float32(0.5) * Float32(
        gProb_tile[warp_y * MXFP8_BLOCK_SCALING_SIZE + lane_idx]
    )
    half_prob = [
        cute.arch.shuffle_sync(half_prob_lane, lane_y * ELEMENTS_Y + row_idx)
        for row_idx in range(ELEMENTS_Y)
    ]

    scaled_swiglu = scaled_swiglu_approx_x2_pair(DTYPE, CLAMPED)

    half_alpha = Float32(0.5) * alpha if cutlass.const_expr(CLAMPED) else None
    amax_x2 = abs_max_x2(DTYPE) if cutlass.const_expr(not IS_FP32) else None

    # Quantization multiplies by packed BF16 reciprocals for BF16 input, as
    # quantize_mxfp8_g2r_bidim does, else by FP32 ones.
    if cutlass.const_expr(IS_BF16):
        scale_and_cvt = mul_bf16x2x2_cvt_bf16x4_to_fp8x4(FP8_DTYPE)
    else:
        scale_and_cvt = mul_f32x2_cvt_x2_to_fp8x2(DTYPE, FP8_DTYPE)

    load_atom = cute.make_copy_atom(
        cute.nvgpu.CopyG2ROp(), DTYPE, num_bits_per_copy=ELEMENTS_X * DTYPE.width
    )
    store_atom = cute.make_copy_atom(
        cute.nvgpu.CopyR2GOp(), FP8_DTYPE, num_bits_per_copy=ELEMENTS_X * FP8_DTYPE.width
    )

    for col_iter in cutlass.range_constexpr(COL_ITERS):
        # This thread's columns, as a tile coordinate in units of ELEMENTS_X columns
        thread_col_coord = (col_iter * WARPS_X + warp_x) * LANES_X + lane_x
        tile_col = thread_col_coord * ELEMENTS_X
        # H % 16 == 0, so a thread's columns are either all in bounds or all out
        in_bounds = Int64(tile_col) < valid_cols
        thread_coord = (thread_row_coord, thread_col_coord)

        # The scaled activation, rounded to the input dtype like the CUDA kernel does before
        # taking the amax, overwrites the act half in place.
        rAct = cute.make_rmem_tensor((ELEMENTS_X, ELEMENTS_Y), DTYPE)
        rGate = cute.make_rmem_tensor((ELEMENTS_X, ELEMENTS_Y), DTYPE)
        rY = cute.recast_tensor(rAct, PAIR_DTYPE)
        rGate_pairs = cute.recast_tensor(rGate, PAIR_DTYPE)

        # Per-column block amaxes: packed pairs for 16-bit input, else one FP32 per column
        if cutlass.const_expr(IS_FP32):
            amax = [Float32(0.0) for _ in range(ELEMENTS_X)]
        else:
            zero = zero_b32_reg()
            amax = [zero for _ in range(PAIRS_X)]

        if in_bounds:
            gAct_thread = cute.local_tile(gAct_tile, THREAD_SHAPE, thread_coord)
            gGate_thread = cute.local_tile(gGate_tile, THREAD_SHAPE, thread_coord)
            for row_idx in cutlass.range_constexpr(ELEMENTS_Y):
                cute.copy(load_atom, gAct_thread[row_idx, None], rAct[None, row_idx])
                cute.copy(load_atom, gGate_thread[row_idx, None], rGate[None, row_idx])

            for row_idx in cutlass.range_constexpr(ELEMENTS_Y):
                for pair_idx in cutlass.range_constexpr(PAIRS_X):
                    y = scaled_swiglu(
                        rY[pair_idx, row_idx],
                        rGate_pairs[pair_idx, row_idx],
                        half_prob[row_idx],
                        limit,
                        half_alpha,
                        glu_linear_offset,
                    )
                    rY[j, row_idx] = y
                    if cutlass.const_expr(IS_FP32):
                        y_lo, y_hi = unpack_i64_to_i32x2(y)
                        amax[2 * j] = cute.arch.fmax(amax[2 * j], fabs_f32(y_lo.bitcast(Float32)))
                        amax[2 * j + 1] = cute.arch.fmax(
                            amax[2 * j + 1], fabs_f32(y_hi.bitcast(Float32))
                        )
                    else:
                        amax[j] = amax_x2(amax[j], y)

        # Combine the amaxes of the lanes holding the other rows of the scaling block
        for i in cutlass.range_constexpr(len(amax)):
            for step in cutlass.range_constexpr(LANES_Y.bit_length() - 1):
                other = cute.arch.shuffle_sync_bfly(amax[i], LANES_X << step)
                if cutlass.const_expr(IS_FP32):
                    amax[i] = cute.arch.fmax(amax[i], other)
                else:
                    amax[i] = amax_x2(amax[i], other)

        # Scales, and the reciprocals quantization multiplies by
        rS = cute.make_rmem_tensor((ELEMENTS_X,), Float8E8M0FNU)
        rS_pairs = cute.recast_tensor(rS, Uint16)
        scale_rcp = [None] * PAIRS_X
        for pair_idx in cutlass.range_constexpr(PAIRS_X):
            if cutlass.const_expr(IS_BF16):
                scale_rcp[pair_idx] = bf16_mx_scale_reciprocal_pair(amax[pair_idx], MAX_EXPONENT)
                rS_pairs[pair_idx] = Uint16(bf16_mx_scale_bytes_pair(scale_rcp[pair_idx]))
            else:
                if cutlass.const_expr(IS_FP32):
                    amax_lo, amax_hi = amax[2 * pair_idx], amax[2 * pair_idx + 1]
                else:
                    amax_lo = x2_lo_to_f32_f16(amax[pair_idx] & Int32(0x7FFF7FFF))
                    amax_hi = x2_hi_to_f32_f16(amax[pair_idx] & Int32(0x7FFF7FFF))
                rS[2 * pair_idx] = cvt_f32_to_fp8e8m0fnu(amax_lo * MAX_NORM_RCP)
                rS[2 * pair_idx + 1] = cvt_f32_to_fp8e8m0fnu(amax_hi * MAX_NORM_RCP)
                scale_rcp[pair_idx] = pack_f32x2(
                    exp2f_rcp(rS[2 * pair_idx]), exp2f_rcp(rS[2 * pair_idx + 1])
                )

        # One lane per column stages the scales. Columns past valid_cols get zero scales, as in
        # the CUDA kernel.
        if lane_y == 0:
            if cutlass.const_expr(WITH_GEMM_SWIZZLED_SCALES):
                # In a tile, scale (r, c) is at (c % 32) * 16 + (c // 32 % 4) * 4 + r: a
                # thread's columns stay in one 32-column group, 16 bytes apart.
                tile_offset = (tile_col % 32) * 16 + (tile_col // 32) * 4 + warp_y
                sS_thread = cute.make_tensor(
                    sS_col_tile.iterator + tile_offset,
                    cute.make_layout((ELEMENTS_X,), stride=(16,)),
                )
            else:
                sS_thread = cute.make_tensor(
                    (sS_col_tile.iterator + warp_y * TILE_COLS + tile_col).align(ELEMENTS_X),
                    cute.make_layout((ELEMENTS_X,)),
                )
            cute.autovec_copy(cute.recast_tensor(rS, Uint8), sS_thread)

        if in_bounds:
            gO_col_thread = cute.local_tile(gO_col_tile, THREAD_SHAPE, thread_coord)
            rO = cute.make_rmem_tensor((ELEMENTS_X,), FP8_DTYPE)
            rO_x4 = cute.recast_tensor(rO, Uint32)
            rO_x2 = cute.recast_tensor(rO, Uint16)
            for row_idx in cutlass.range_constexpr(ELEMENTS_Y):
                if cutlass.const_expr(IS_BF16):
                    for w in cutlass.range_constexpr(PAIRS_X // 2):
                        rO_x4[w] = scale_and_cvt(
                            rY[2 * w, row_idx],
                            scale_rcp[2 * w],
                            rY[2 * w + 1, row_idx],
                            scale_rcp[2 * w + 1],
                        )
                else:
                    for j in cutlass.range_constexpr(PAIRS_X):
                        rO_x2[j] = scale_and_cvt(rY[j, row_idx], scale_rcp[j])
                cute.copy(store_atom, rO, gO_col_thread[row_idx, None])


class MXFP8GroupScaledSwiGLUConfig:
    """Compile-time config of the grouped scaled SwiGLU kernel."""

    def __init__(
        self,
        dtype: str,
        fp8_dtype: str,
        shape_rep: str,
        with_gemm_swizzled_scales: bool,
        activation: str,
    ):
        if dtype not in ("Float32", "Float16", "BFloat16"):
            raise ValueError(f"unknown input dtype {dtype!r}; expected Float32|Float16|BFloat16")
        if fp8_dtype not in ("Float8E4M3", "Float8E5M2"):
            raise ValueError(
                f"unknown FP8 dtype {fp8_dtype!r}; expected 'Float8E4M3' or 'Float8E5M2'"
            )
        # Like the CUDA kernel, only a group with a common last dim is supported.
        if shape_rep not in (SAME_BOTH_DIMS, VARYING_FIRST_DIM):
            raise ValueError(
                f"unsupported shape representation {shape_rep!r}; expected"
                f" {SAME_BOTH_DIMS!r} or {VARYING_FIRST_DIM!r}"
            )
        if activation not in SCALED_SWIGLU_ACTIVATIONS:
            raise ValueError(
                f"unknown activation {activation!r}; expected one of {SCALED_SWIGLU_ACTIVATIONS}"
            )
        self.DTYPE = str_to_cutlass_dtype(dtype)
        self.DTYPE_STR = dtype
        self.FP8_DTYPE = str_to_cutlass_dtype(fp8_dtype)
        self.FP8_DTYPE_STR = fp8_dtype
        self.MAX_NORM_RCP = (
            FP8E4M3_MAX_NORM_RCP if fp8_dtype == "Float8E4M3" else FP8E5M2_MAX_NORM_RCP
        )
        self.SHAPE_REP = shape_rep
        self.WITH_GEMM_SWIZZLED_SCALES = with_gemm_swizzled_scales
        self.ACTIVATION = activation
        self.CLAMPED = activation == "clamped_swiglu"

    def __str__(self):
        return (
            f"MXFP8GroupScaledSwiGLUConfig(dtype={self.DTYPE_STR},"
            f" fp8_dtype={self.FP8_DTYPE_STR}, shape_rep={self.SHAPE_REP},"
            f" swizzled={self.WITH_GEMM_SWIZZLED_SCALES}, activation={self.ACTIVATION})"
        )

    __repr__ = __str__

    def compile(self):
        """Return the compiled CuTeDSL function object for this config."""
        # The C++ bridge checks M % 128 == 0, as the CUDA launcher does, and H % 16 == 0.
        sym_M = cute.sym_int64(divisibility=128)
        sym_H = cute.sym_int64(divisibility=16)
        sym_2H = cute.sym_int64(divisibility=32)
        vfd = self.SHAPE_REP == VARYING_FIRST_DIM
        # Only clamped SwiGLU takes runtime parameters; plain SwiGLU is compiled with None in
        # their slots, and the C++ bridge passes None for them.
        params = [Float32(0.0) if self.CLAMPED else None] * 3  # limit, alpha, glu_linear_offset
        return cute.compile(
            MXFP8GroupScaledSwiGLUKernel(self),
            _fake_gmem_tensor(self.DTYPE, (sym_M, sym_2H), 16),  # mX
            _fake_gmem_tensor(self.DTYPE, (cute.sym_int64(),), self.DTYPE.width // 8),  # mProb
            _fake_gmem_tensor(self.FP8_DTYPE, (sym_M, sym_H), 16),  # mO_col
            _fake_gmem_tensor(Float8E8M0FNU, (cute.sym_int64(),), 16),  # mS_col
            _fake_gmem_tensor(Int64, (cute.sym_int32(),), 8) if vfd else None,  # mOffsets
            _fake_gmem_tensor(Int64, (cute.sym_int32(),), 8) if vfd else None,  # mFirstDims
            Int32(1),  # num_tensors, supplied at runtime
            cute.runtime.nullptr(Float32, mem_space=cute.AddressSpace.gmem, assumed_align=4),
            *params,
            cute.runtime.make_fake_stream(),
            options="--enable-tvm-ffi",
        )


class MXFP8GroupScaledSwiGLUKernel:
    """Grouped scaled SwiGLU + columnwise MXFP8, on group_scaled_swiglu_mxfp8_kernel's jobs.

    A CTA processes one (TILE_ROWS, TILE_COLS) job of the [M, H] output, as each iteration of
    the CUDA kernel's persistent CTAs does; every member's rows are a multiple of TILE_ROWS, so
    a job belongs to a single member. scaled_swiglu_quantize_mxfp8_gmem_tile_colwise computes
    the job in registers and stages its scales in SMEM, which one warp then writes with 16-byte
    stores.
    """

    # The shape of the job a CTA processes
    TILE_ROWS = 128
    TILE_COLS = 128

    # CTA shape: a warp handles a full MXFP8_BLOCK_SCALING_SIZE row block
    WARPS_Y = TILE_ROWS // MXFP8_BLOCK_SCALING_SIZE  # 4
    WARPS_X = 2
    NUM_WARPS = WARPS_Y * WARPS_X  # 8
    THREADS_PER_CTA = NUM_WARPS * THREADS_PER_WARP  # 256

    # A job's colwise scales: 4 rows of TILE_COLS, which is one 128 x 4 GEMM-swizzled tile
    SCALES_PER_JOB = TILE_ROWS // MXFP8_BLOCK_SCALING_SIZE * TILE_COLS  # 512
    # One warp writes a job's scales, one store per lane
    SCALE_STORE_BYTES = SCALES_PER_JOB // THREADS_PER_WARP  # 16

    def __init__(self, cfg: MXFP8GroupScaledSwiGLUConfig) -> None:
        self.cfg = cfg
        # Thread's amount of work depends on the dtype
        self.ELEMENTS_X_PER_THREAD = 128 // cfg.DTYPE.width
        self.ELEMENTS_Y_PER_THREAD = 8 if cfg.DTYPE is Float32 else 4
        # A hyperparameter tuned for better performance
        self.MIN_BLOCKS_PER_MP = 3 if cfg.DTYPE is Float32 else 4

    @cute.jit
    def _scale_tensor(self, mS: cute.Tensor, base: Int64, layout: cute.Layout) -> cute.Tensor:
        """View the scale buffer from element `base` on with `layout`."""
        return cute.make_tensor(
            cute.make_ptr(
                Float8E8M0FNU,
                mS.iterator.toint() + base,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            layout,
        )

    @cute.jit
    def _colwise_scales(
        self, mS_col: cute.Tensor, base: Int64, rows: Int64, cols: Int64
    ) -> cute.Tensor:
        """Colwise scales of a (rows, cols) tensor at `base`, indexed by (scale row, column)."""
        if cutlass.const_expr(self.cfg.WITH_GEMM_SWIZZLED_SCALES):
            _, mS_t = derive_swizzled_scale_layout(
                rows, cols, False, True, None, self._scale_tensor(mS_col, base, cute.make_layout(1))
            )
        else:
            # Colwise scale's divisibility guarantee: (4, 128)
            stride = cute.round_up(cols, 128)
            mS_t = self._scale_tensor(
                mS_col,
                base,
                cute.make_layout((rows // MXFP8_BLOCK_SCALING_SIZE, stride), stride=(stride, 1)),
            )
        return mS_t

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,  # [M, 2H] input, [act | gate]
        mProb: cute.Tensor,  # [>= M] per-token prob, in the input dtype
        mO_col: cute.Tensor,  # [M, H] columnwise FP8 output
        mS_col: cute.Tensor,  # flat colwise scales, compact or GEMM-swizzled per member
        mOffsets: Optional[cute.Tensor],  # int64[num_tensors + 1], CSR element offsets
        mFirstDims: Optional[cute.Tensor],  # int64[num_tensors] (VARYING_FIRST_DIM)
        num_tensors: Int32,
        mNoop: cute.Pointer,  # f32 cast_noop flag; may be null, checked on device
        limit: Optional[Float32],  # clamped SwiGLU only
        alpha: Optional[Float32],  # clamped SwiGLU only
        glu_linear_offset: Optional[Float32],  # clamped SwiGLU only
        stream: CUstream,
    ) -> None:
        first_logical_dim = mO_col.shape[0]
        last_logical_dim = mO_col.shape[1]
        # How many jobs does the grouped tensor have in both directions
        jobs_Y = cute.ceil_div(Int64(first_logical_dim), self.TILE_ROWS)
        jobs_X = cute.ceil_div(Int64(last_logical_dim), self.TILE_COLS)
        # Flatten it to an 1D grid
        grid = [Int32(jobs_X * jobs_Y), 1, 1]

        self.kernel(
            mX,
            mProb,
            mO_col,
            mS_col,
            mOffsets,
            mFirstDims,
            num_tensors,
            mNoop,
            limit,
            alpha,
            glu_linear_offset,
            jobs_X,
        ).launch(
            grid=grid,
            block=[self.THREADS_PER_CTA, 1, 1],
            min_blocks_per_mp=self.MIN_BLOCKS_PER_MP,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mProb: cute.Tensor,
        mO_col: cute.Tensor,
        mS_col: cute.Tensor,
        mOffsets: Optional[cute.Tensor],
        mFirstDims: Optional[cute.Tensor],
        num_tensors: Int32,
        mNoop: cute.Pointer,
        limit: Optional[Float32],
        alpha: Optional[Float32],
        glu_linear_offset: Optional[Float32],
        jobs_X: Int64,
    ) -> None:
        """No-op the CTA when the noop flag is set, else process its job."""
        skip_execution = noop_flag_is_set(mNoop)
        if not skip_execution:
            self._kernel_main(
                mX,
                mProb,
                mO_col,
                mS_col,
                mOffsets,
                mFirstDims,
                num_tensors,
                limit,
                alpha,
                glu_linear_offset,
                jobs_X,
            )

    @cute.jit
    def _kernel_main(
        self,
        mX: cute.Tensor,
        mProb: cute.Tensor,
        mO_col: cute.Tensor,
        mS_col: cute.Tensor,
        mOffsets: Optional[cute.Tensor],
        mFirstDims: Optional[cute.Tensor],
        num_tensors: Int32,
        limit: Optional[Float32],
        alpha: Optional[Float32],
        glu_linear_offset: Optional[Float32],
        jobs_X: Int64,
    ) -> None:
        cfg = self.cfg
        bidx, _, _ = cute.arch.block_idx()
        first_logical_dim = Int64(mO_col.shape[0])
        # All members share the stacked tensor's column count. The bridge checks it is a
        # multiple of 16, which keeps every thread's vector accesses aligned.
        member_cols = cute.assume(Int64(mO_col.shape[1]), divby=16)

        # grid = [jobs_X * jobs_Y, 1, 1]
        job_id_Y = Int64(bidx) // jobs_X
        job_id_X = Int64(bidx) % jobs_X
        job_start_row = job_id_Y * self.TILE_ROWS
        job_start_col = job_id_X * self.TILE_COLS

        # If the CTA has work to do
        has_work = Boolean(True)
        if cutlass.const_expr(cfg.SHAPE_REP == VARYING_FIRST_DIM):
            total_elts = Int64(mOffsets[mOffsets.shape[0] - 1])
            # Jobs past the group's last element (a buffer sized ahead for CUDA graphs) have
            # no work
            if job_start_row * member_cols >= total_elts:
                has_work = Boolean(False)

        if has_work:
            self._process_job(
                mX,
                mProb,
                mO_col,
                mS_col,
                mOffsets,
                mFirstDims,
                num_tensors,
                first_logical_dim,
                member_cols,
                job_start_row,
                job_start_col,
                limit,
                alpha,
                glu_linear_offset,
            )

    @cute.jit
    def _process_job(
        self,
        mX: cute.Tensor,
        mProb: cute.Tensor,
        mO_col: cute.Tensor,
        mS_col: cute.Tensor,
        mOffsets: Optional[cute.Tensor],
        mFirstDims: Optional[cute.Tensor],
        num_tensors: Int32,
        first_logical_dim: Int64,
        member_cols: Int64,
        job_start_row: Int64,
        job_start_col: Int64,
        limit: Optional[Float32],
        alpha: Optional[Float32],
        glu_linear_offset: Optional[Float32],
    ) -> None:
        """Compute and quantize a job's tile in registers, then write its colwise scales."""
        cfg = self.cfg
        TILE_SHAPE = (self.TILE_ROWS, self.TILE_COLS)

        in_stride = cute.assume(Int64(mX.shape[1]), divby=32)  # 2H
        in_layout = cute.make_layout(TILE_SHAPE, stride=(in_stride, 1))
        # Find the act and gate halves
        act_offset = job_start_row * in_stride + job_start_col
        gAct_tile = cute.make_tensor((mX.iterator + act_offset).align(16), in_layout)
        gGate_tile = cute.make_tensor((mX.iterator + act_offset + member_cols).align(16), in_layout)
        gProb_tile = cute.make_tensor(
            mProb.iterator + job_start_row, cute.make_layout(self.TILE_ROWS)
        )
        gO_col_tile = cute.make_tensor(
            (mO_col.iterator + job_start_row * member_cols + job_start_col).align(16),
            cute.make_layout(TILE_SHAPE, stride=(member_cols, 1)),
        )
        smem = cutlass.memory.SmemAllocator()
        sS_col_tile = smem.allocate_tensor(
            Uint8, cute.make_layout((self.SCALES_PER_JOB,)), byte_alignment=16
        )

        scaled_swiglu_quantize_mxfp8_gmem_tile_colwise(
            gAct_tile,
            gGate_tile,
            gProb_tile,
            gO_col_tile,
            sS_col_tile,
            member_cols - job_start_col,
            limit,
            alpha,
            glu_linear_offset,
            DTYPE=cfg.DTYPE,
            FP8_DTYPE=cfg.FP8_DTYPE,
            MAX_NORM_RCP=cfg.MAX_NORM_RCP,
            CLAMPED=cfg.CLAMPED,
            WITH_GEMM_SWIZZLED_SCALES=cfg.WITH_GEMM_SWIZZLED_SCALES,
            WARPS_X=self.WARPS_X,
            ELEMENTS_Y_PER_THREAD=self.ELEMENTS_Y_PER_THREAD,
            ELEMENTS_X_PER_THREAD=self.ELEMENTS_X_PER_THREAD,
        )

        cute.arch.sync_threads()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            self._store_colwise_scales(
                sS_col_tile,
                mS_col,
                mOffsets,
                mFirstDims,
                num_tensors,
                first_logical_dim,
                member_cols,
                job_start_row,
                job_start_col,
            )

    @cute.jit
    def _store_colwise_scales(
        self,
        sS_col: cute.Tensor,
        mS_col: cute.Tensor,
        mOffsets: Optional[cute.Tensor],
        mFirstDims: Optional[cute.Tensor],
        num_tensors: Int32,
        first_logical_dim: Int64,
        member_cols: Int64,
        job_start_row: Int64,
        job_start_col: Int64,
    ) -> None:
        """Write the job's staged colwise scales, one warp issuing 16-byte stores: a GEMM-swizzled
        job is one contiguous tile of its member's scales, a compact one is 4 rows of
        TILE_COLS."""
        cfg = self.cfg
        lane_idx = cute.arch.lane_idx()

        # Find how many rows this tensor member has and which row the member starts from
        if cutlass.const_expr(cfg.WITH_GEMM_SWIZZLED_SCALES and cfg.SHAPE_REP == VARYING_FIRST_DIM):
            # Swizzled scales are laid out per member, so find the member the job belongs to
            member_id = find_tensor_from_offsets(mOffsets, num_tensors, job_start_row * member_cols)
            member_rows = Int64(mFirstDims[member_id])
            member_start_row = Int64(mOffsets[member_id]) // member_cols
        else:
            # Compact scales of a group are those of the stacked tensor, and swizzled scales of
            # SAME_BOTH_DIMS have a single member (checked by the bridge), so the whole tensor
            # stands in for the member.
            member_rows = first_logical_dim
            member_start_row = Int64(0)
        # The starting offset for this tensor member's scale
        member_scale_base = (
            member_start_row * cute.round_up(member_cols, 128) // MXFP8_BLOCK_SCALING_SIZE
        )

        # Obtain the colwise scale tensor of this member only
        if cutlass.const_expr(self.cfg.WITH_GEMM_SWIZZLED_SCALES):
            # Create a flat tensor that points to the member's swizzled scales with 1D layout
            # Its layout doesn't matter because we will need to derive the layout from the swizzled pattern
            col_scales_flat = cute.make_tensor(
                cute.make_ptr(
                    Float8E8M0FNU,
                    mS_col.iterator.toint() + member_scale_base,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                cute.make_layout(1),  # Just a placeholder
            )
            _, col_scales = derive_swizzled_scale_layout(
                member_rows, member_cols, False, True, None, col_scales_flat
            )
        else:
            # Colwise scale's divisibility guarantee: (4, 128)
            stride = cute.round_up(member_cols, 128)
            col_scales = cute.make_tensor(
                cute.make_ptr(
                    Float8E8M0FNU,
                    mS_col.iterator.toint() + member_scale_base,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                cute.make_layout(
                    (member_rows // MXFP8_BLOCK_SCALING_SIZE, stride), stride=(stride, 1)
                ),
            )
        # The job's (4, TILE_COLS) scales, in GMEM and as staged in SMEM
        JOB_SCALE_ROWS = self.TILE_ROWS // MXFP8_BLOCK_SCALING_SIZE
        JOB_SCALE_COLS = self.TILE_COLS
        scale_start_row = (job_start_row - member_start_row) // MXFP8_BLOCK_SCALING_SIZE
        gS_job = cute.local_tile(
            cute.recast_tensor(col_scales, Uint8),
            (JOB_SCALE_ROWS, JOB_SCALE_COLS),  # scale tile shape
            (
                scale_start_row // JOB_SCALE_ROWS,
                job_start_col // JOB_SCALE_COLS,
            ),  # Coordinates for the scale tile
        )

        # Each lane stores 16 scales that are contiguous in both SMEM and GMEM
        if cutlass.const_expr(cfg.WITH_GEMM_SWIZZLED_SCALES):
            _, sS_job = derive_swizzled_scale_layout(
                self.TILE_ROWS, self.TILE_COLS, False, True, None, sS_col
            )
            # Each lane takes 16 consecutive bytes of the swizzled tile
            _, tv_layout = derive_swizzled_scale_vectorized_tv_layout(
                self.TILE_ROWS,
                self.TILE_COLS,
                False,
                True,
                num_threads=THREADS_PER_WARP,
                num_values=self.SCALE_STORE_BYTES,
            )
        else:
            sS_job = cute.make_tensor(
                sS_col.iterator,
                cute.make_layout((JOB_SCALE_ROWS, JOB_SCALE_COLS), stride=(JOB_SCALE_COLS, 1)),
            )
            # 8 lanes per row, each taking 16 consecutive columns
            _, tv_layout = cute.make_layout_tv(
                thr_layout=cute.make_layout((4, 8), stride=(8, 1)),
                val_layout=cute.make_layout((1, 16)),  # Each thread does 16 bytes vectorized store
            )

        src_frag = cute.composition(sS_job, tv_layout)[lane_idx, None]
        dst_frag = cute.composition(gS_job, tv_layout)[lane_idx, None]
        # The row stride of compact GMEM scales is runtime, so restate the 16-byte alignment
        src_frag = cute.make_tensor(
            src_frag.iterator.align(self.SCALE_STORE_BYTES), src_frag.layout
        )
        dst_frag = cute.make_tensor(
            dst_frag.iterator.align(self.SCALE_STORE_BYTES), dst_frag.layout
        )
        cute.autovec_copy(src_frag, dst_frag)


def get_mxfp8_group_scaled_swiglu_function(
    fn_name: str,
    dtype: str,
    fp8_dtype: str,
    shape_rep: str,
    with_gemm_swizzled_scales: bool,
    activation: str,
) -> bool:
    """Compile the grouped scaled SwiGLU kernel for this config and register it under `fn_name`.
    Returns False if the config is unsupported, so the C++ bridge falls back to the CUDA kernel.
    """
    try:
        cfg = MXFP8GroupScaledSwiGLUConfig(
            dtype=dtype,
            fp8_dtype=fp8_dtype,
            shape_rep=shape_rep,
            with_gemm_swizzled_scales=with_gemm_swizzled_scales,
            activation=activation,
        )
    except ValueError as e:
        logger.warning(
            "CuTeDSL grouped scaled SwiGLU does not support this config, "
            "falling back to the CUDA C++ kernel: %s",
            e,
        )
        return False
    return compile_and_register_function(fn_name, cfg)
