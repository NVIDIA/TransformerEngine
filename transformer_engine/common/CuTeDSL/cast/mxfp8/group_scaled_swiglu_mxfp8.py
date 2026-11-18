# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Grouped scaled SwiGLU fused with columnwise MXFP8 quantization, in CuTeDSL.

Port of group_scaled_swiglu_mxfp8.cuh, the MoE backward recompute of the FC2 input:

    input  : x [M, 2H] = [act | gate], the FC1 output of a group stacked along the tokens
    prob   : [M] per-token router weight, in the input dtype
    output : columnwise MXFP8 of (silu(act) * gate) * prob, [M, H]

Clamped SwiGLU clamps act from above and gate on both sides, then offsets gate, as
gated_mxfp8.cuh's forward does. The C++ bridge is group_scaled_swiglu_mxfp8_cutedsl.cuh.
"""

import functools
import logging
from typing import Callable, Optional

import cutlass
from cutlass import cute
from cutlass import Float32, Int32, Int64, Uint8, Uint16, Uint32
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module

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
)
from transformer_engine.common.CuTeDSL.utils_fp8 import (
    cvt_f32_to_fp8e8m0fnu,
    mul_bf16x2x2_cvt_bf16x4_to_fp8x4,
)
from .quantize_mxfp8_common import (
    FP8E4M3_MAX_NORM_RCP,
    FP8E5M2_MAX_NORM_RCP,
    MXFP8_BLOCK_SCALING_SIZE,
    THREADS_PER_WARP,
    _fake_gmem_tensor,
    compile_and_register_function,
    derive_swizzled_scale_layout,
    noop_flag_is_set,
)
from .group_quantize_mxfp8_common import (
    MAX_SUPPORTED_TENSORS,
    SAME_BOTH_DIMS,
    VARYING_FIRST_DIM,
)

logger = logging.getLogger("transformer_engine.cutedsl.mxfp8")

SCALED_SWIGLU_ACTIVATIONS = ("swiglu", "clamped_swiglu")


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
            _fake_gmem_tensor(self.FP8_DTYPE, (sym_M, sym_H), 16),  # mO
            _fake_gmem_tensor(cutlass.Float8E8M0FNU, (cute.sym_int64(),), 16),  # mS
            _fake_gmem_tensor(Int64, (cute.sym_int32(),), 8) if vfd else None,  # mOffsets
            _fake_gmem_tensor(Int64, (cute.sym_int32(),), 8) if vfd else None,  # mFirstDims
            Int32(1),  # num_tensors, supplied at runtime
            cute.runtime.nullptr(Float32, mem_space=cute.AddressSpace.gmem, assumed_align=4),
            *params,
            cute.runtime.make_fake_stream(),
            options="--enable-tvm-ffi",
        )


class MXFP8GroupScaledSwiGLUKernel:
    """Register-resident grouped scaled SwiGLU + columnwise MXFP8 quantization.

    A CTA takes one (CHUNK_ROWS, CHUNK_COLS) chunk of the [M, H] output, as the CUDA kernel's
    persistent CTAs do; every member's rows are a multiple of CHUNK_ROWS, so a chunk belongs to
    a single member. Each warp owns a 32-row scaling block of WARP_COLS columns: a lane loads
    ELEMS_Y rows of ELEMS_X columns of both halves with one 16-byte load per row, and the
    LANES_Y lanes that share columns combine their block amaxes by shuffles. The chunk's scales
    are gathered in shared memory and written with 16-byte stores.
    """

    CHUNK_ROWS = 128
    CHUNK_COLS = 128
    WARPS_Y = CHUNK_ROWS // MXFP8_BLOCK_SCALING_SIZE
    WARPS_X = 2
    THREADS_PER_CTA = WARPS_Y * WARPS_X * THREADS_PER_WARP
    # A chunk's scales: 4 rows of CHUNK_COLS, which is one 128 x 4 GEMM-swizzled tile.
    CHUNK_SCALES = CHUNK_ROWS // MXFP8_BLOCK_SCALING_SIZE * CHUNK_COLS
    SCALE_STORE_BYTES = 16

    def __init__(self, cfg: MXFP8GroupScaledSwiGLUConfig):
        self.cfg = cfg
        self.ELEMS_X = 128 // cfg.DTYPE.width
        # By default ptxas saves registers by issuing most of a lane's loads only when the math
        # needs them. These rows per lane and CTAs per SM keep most of them in flight instead:
        # 1.2x (FP32) to 1.3x (16-bit input) faster on GB200.
        self.ELEMS_Y = 8 if cfg.DTYPE is Float32 else 4
        self.MIN_BLOCKS_PER_MP = 3 if cfg.DTYPE is Float32 else 4
        self.LANES_Y = MXFP8_BLOCK_SCALING_SIZE // self.ELEMS_Y
        self.LANES_X = THREADS_PER_WARP // self.LANES_Y
        # log2 of the FP8 format's largest power of two, for the BF16 scale helpers.
        self.MAX_EXPONENT = 8 if cfg.FP8_DTYPE is cutlass.Float8E4M3FN else 15
        self.WARP_COLS = self.LANES_X * self.ELEMS_X
        # The warps sweep the chunk's columns in COL_ITERS steps.
        self.COL_ITERS = self.CHUNK_COLS // (self.WARPS_X * self.WARP_COLS)
        assert self.COL_ITERS * self.WARPS_X * self.WARP_COLS == self.CHUNK_COLS
        # One warp writes a chunk's scales, a 16-byte store per lane.
        assert self.CHUNK_SCALES == THREADS_PER_WARP * self.SCALE_STORE_BYTES

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,  # [M, 2H] input, [act | gate]
        mProb: cute.Tensor,  # [>= M] per-token prob, in the input dtype
        mO: cute.Tensor,  # [M, H] columnwise FP8 output
        mS: cute.Tensor,  # Flat columnwise scales, compact or GEMM-swizzled per member
        mOffsets: Optional[cute.Tensor],  # int64[num_tensors + 1] output element offsets
        mFirstDims: Optional[cute.Tensor],  # int64[num_tensors] member rows
        num_tensors: Int32,
        mNoop: cute.Pointer,
        limit: Optional[Float32],
        alpha: Optional[Float32],
        glu_linear_offset: Optional[Float32],
        stream: CUstream,
    ):
        jobs_X = cute.ceil_div(Int64(mO.shape[1]), self.CHUNK_COLS)
        num_jobs = jobs_X * (Int64(mO.shape[0]) // self.CHUNK_ROWS)
        self.kernel(
            mX,
            mProb,
            mO,
            mS,
            mOffsets,
            mFirstDims,
            num_tensors,
            mNoop,
            limit,
            alpha,
            glu_linear_offset,
            jobs_X,
        ).launch(
            grid=[Int32(num_jobs), 1, 1],
            block=[self.THREADS_PER_CTA, 1, 1],
            min_blocks_per_mp=self.MIN_BLOCKS_PER_MP,
            stream=stream,
        )

    @cute.jit
    def _scale_view(self, mS: cute.Tensor, member_start_row: Int64, member_rows: Int64, H: Int64):
        """The member's colwise scales, indexed by (scale row in the member, column)."""
        stride = cute.round_up(H, 128)
        base = cute.make_ptr(
            cutlass.Float8E8M0FNU,
            mS.iterator.toint() + member_start_row * stride // MXFP8_BLOCK_SCALING_SIZE,
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        if cutlass.const_expr(self.cfg.WITH_GEMM_SWIZZLED_SCALES):
            _, gS = derive_swizzled_scale_layout(
                member_rows, H, False, True, None, cute.make_tensor(base, cute.make_layout(1))
            )
        else:
            gS = cute.make_tensor(
                base,
                cute.make_layout(
                    (member_rows // MXFP8_BLOCK_SCALING_SIZE, stride), stride=(stride, 1)
                ),
            )
        return gS

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mProb: cute.Tensor,
        mO: cute.Tensor,
        mS: cute.Tensor,
        mOffsets: Optional[cute.Tensor],
        mFirstDims: Optional[cute.Tensor],
        num_tensors: Int32,
        mNoop: cute.Pointer,
        limit: Optional[Float32],
        alpha: Optional[Float32],
        glu_linear_offset: Optional[Float32],
        jobs_X: Int64,
    ):
        """Device entry: one CTA per chunk."""
        cfg = self.cfg
        bidx, _, _ = cute.arch.block_idx()
        M = Int64(mO.shape[0])
        # The bridge checks H % 16 == 0, which keeps every lane's vector accesses aligned.
        H = cute.assume(Int64(mO.shape[1]), divby=16)

        if not noop_flag_is_set(mNoop):
            chunk_row = Int64(bidx) // jobs_X * self.CHUNK_ROWS
            chunk_col = Int64(bidx) % jobs_X * self.CHUNK_COLS
            has_work = True
            if cutlass.const_expr(cfg.SHAPE_REP == VARYING_FIRST_DIM):
                # Chunks past the group's end (a buffer sized ahead for CUDA graphs) have no work.
                has_work = chunk_row * H < Int64(mOffsets[num_tensors])
            if has_work:
                self._process_chunk(
                    mX,
                    mProb,
                    mO,
                    mS,
                    mOffsets,
                    mFirstDims,
                    num_tensors,
                    chunk_row,
                    chunk_col,
                    M,
                    H,
                    limit,
                    alpha,
                    glu_linear_offset,
                )

    @cute.jit
    def _process_chunk(
        self,
        mX: cute.Tensor,
        mProb: cute.Tensor,
        mO: cute.Tensor,
        mS: cute.Tensor,
        mOffsets: Optional[cute.Tensor],
        mFirstDims: Optional[cute.Tensor],
        num_tensors: Int32,
        chunk_row: Int64,
        chunk_col: Int64,
        M: Int64,
        H: Int64,
        limit: Optional[Float32],
        alpha: Optional[Float32],
        glu_linear_offset: Optional[Float32],
    ):
        cfg = self.cfg
        DTYPE = cfg.DTYPE
        FP8_DTYPE = cfg.FP8_DTYPE
        ELEMS_X = self.ELEMS_X
        ELEMS_Y = self.ELEMS_Y
        PAIRS = ELEMS_X // 2
        BF16 = DTYPE is cutlass.BFloat16
        # A pair of adjacent elements: one packed register for 16-bit input, two for FP32.
        PAIR_DTYPE = Int64 if DTYPE is Float32 else Int32

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        warp_y = warp_idx // self.WARPS_X
        warp_x = warp_idx % self.WARPS_X
        lane_y = lane // self.LANES_X
        lane_x = lane % self.LANES_X

        block_row = chunk_row + warp_y * MXFP8_BLOCK_SCALING_SIZE  # This warp's scaling block
        thread_row = block_row + lane_y * ELEMS_Y
        in_stride = cute.assume(Int64(mX.shape[1]), divby=32)  # 2H

        # Each lane loads the prob of one row of the block and hands it to the lanes on that row.
        # Pre-halved to pair with the doubled activation; exact, as 0.5 is a power of two.
        half_prob_lane = Float32(0.5) * Float32(mProb[block_row + lane])
        half_prob = [
            cute.arch.shuffle_sync(half_prob_lane, lane_y * ELEMS_Y + k) for k in range(ELEMS_Y)
        ]
        swiglu_x2 = _swiglu_x2(DTYPE, cfg.CLAMPED)
        amax_x2 = abs_max_x2(DTYPE) if cutlass.const_expr(DTYPE is not Float32) else None
        half_alpha = Float32(0.5) * alpha if cutlass.const_expr(cfg.CLAMPED) else None

        load_atom = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(), DTYPE, num_bits_per_copy=ELEMS_X * DTYPE.width
        )
        store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyR2GOp(), FP8_DTYPE, num_bits_per_copy=ELEMS_X * FP8_DTYPE.width
        )
        # Per lane: ELEMS_X adjacent columns (mode 0) of ELEMS_Y rows (mode 1).
        frag_layout = cute.make_layout((ELEMS_X, ELEMS_Y), stride=(1, in_stride))
        out_frag_layout = cute.make_layout((ELEMS_X, ELEMS_Y), stride=(1, H))

        # The chunk's scales, in the order they are written to global memory.
        sS = cutlass.memory.SmemAllocator().allocate_tensor(
            Uint8, cute.make_layout((self.CHUNK_SCALES,)), byte_alignment=16
        )

        for col_iter in cutlass.range_constexpr(self.COL_ITERS):
            # This lane's first column, in the chunk and in the output.
            chunk_col_idx = (col_iter * self.WARPS_X + warp_x) * self.WARP_COLS + lane_x * ELEMS_X
            col = chunk_col + Int64(chunk_col_idx)
            # H % 16 == 0, so a lane's columns are either all in bounds or all out.
            in_bounds = col < H

            # The scaled activation, rounded to the input dtype like the CUDA kernel does before
            # taking the amax, overwrites the act half in place.
            rAct = cute.make_rmem_tensor((ELEMS_X, ELEMS_Y), DTYPE)
            rGate = cute.make_rmem_tensor((ELEMS_X, ELEMS_Y), DTYPE)
            rY = cute.recast_tensor(rAct, PAIR_DTYPE)
            rGate_x2 = cute.recast_tensor(rGate, PAIR_DTYPE)
            # Per-column block amaxes: packed pairs for 16-bit input, else one FP32 per column.
            if cutlass.const_expr(DTYPE is Float32):
                amax = [Float32(0.0) for _ in range(ELEMS_X)]
            else:
                # Packed max takes no immediates, so the zero must live in a register.
                zero = cute.arch.inline_ptx(
                    "mov.b32 {$w0}, 0;", write_only_types=[Int32], read_only_args=[]
                )
                amax = [zero for _ in range(PAIRS)]
            if in_bounds:
                act_offset = thread_row * in_stride + col
                gAct = cute.make_tensor(
                    cute.make_ptr(
                        DTYPE,
                        mX.iterator.toint() + act_offset * (DTYPE.width // 8),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    frag_layout,
                )
                gGate = cute.make_tensor(
                    cute.make_ptr(
                        DTYPE,
                        mX.iterator.toint() + (act_offset + H) * (DTYPE.width // 8),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    frag_layout,
                )
                for k in cutlass.range_constexpr(ELEMS_Y):
                    cute.copy(load_atom, gAct[None, k], rAct[None, k])
                    cute.copy(load_atom, gGate[None, k], rGate[None, k])

                for k in cutlass.range_constexpr(ELEMS_Y):
                    for j in cutlass.range_constexpr(PAIRS):
                        y = swiglu_x2(
                            rY[j, k],
                            rGate_x2[j, k],
                            half_prob[k],
                            limit,
                            half_alpha,
                            glu_linear_offset,
                        )
                        rY[j, k] = y
                        if cutlass.const_expr(DTYPE is Float32):
                            y_lo, y_hi = unpack_i64_to_i32x2(y)
                            amax[2 * j] = cute.arch.fmax(
                                amax[2 * j], fabs_f32(y_lo.bitcast(Float32))
                            )
                            amax[2 * j + 1] = cute.arch.fmax(
                                amax[2 * j + 1], fabs_f32(y_hi.bitcast(Float32))
                            )
                        else:
                            amax[j] = amax_x2(amax[j], y)

            # Combine the amaxes of the lanes holding the other rows of the scaling block.
            for i in cutlass.range_constexpr(len(amax)):
                for step in cutlass.range_constexpr(self.LANES_Y.bit_length() - 1):
                    other = cute.arch.shuffle_sync_bfly(amax[i], self.LANES_X << step)
                    if cutlass.const_expr(DTYPE is Float32):
                        amax[i] = cute.arch.fmax(amax[i], other)
                    else:
                        amax[i] = amax_x2(amax[i], other)

            # Scales, and what quantization multiplies by: packed BF16 reciprocals for BF16
            # input, as quantize_mxfp8_g2r_bidim does, else FP32 ones.
            rS = cute.make_rmem_tensor((ELEMS_X,), cutlass.Float8E8M0FNU)
            rS_x2 = cute.recast_tensor(rS, Uint16)
            scale_rcp = [None] * PAIRS
            for j in cutlass.range_constexpr(PAIRS):
                if cutlass.const_expr(BF16):
                    scale_rcp[j] = bf16_mx_scale_reciprocal_pair(amax[j], self.MAX_EXPONENT)
                    rS_x2[j] = Uint16(bf16_mx_scale_bytes_pair(scale_rcp[j]))
                else:
                    if cutlass.const_expr(DTYPE is Float32):
                        amax_lo, amax_hi = amax[2 * j], amax[2 * j + 1]
                    else:
                        amax_lo = x2_lo_to_f32_f16(amax[j] & Int32(0x7FFF7FFF))
                        amax_hi = x2_hi_to_f32_f16(amax[j] & Int32(0x7FFF7FFF))
                    rS[2 * j] = cvt_f32_to_fp8e8m0fnu(amax_lo * cfg.MAX_NORM_RCP)
                    rS[2 * j + 1] = cvt_f32_to_fp8e8m0fnu(amax_hi * cfg.MAX_NORM_RCP)
                    scale_rcp[j] = pack_f32x2(exp2f_rcp(rS[2 * j]), exp2f_rcp(rS[2 * j + 1]))

            # One lane per column stages the scales in the chunk's output order, to be written
            # out by _store_scales. Columns past H get zero scales, as in the CUDA kernel.
            if lane_y == 0:
                if cutlass.const_expr(cfg.WITH_GEMM_SWIZZLED_SCALES):
                    # In a tile, scale (r, c) is at (c % 32) * 16 + (c // 32 % 4) * 4 + r: a lane's
                    # columns stay in one 32-column group, 16 bytes apart.
                    tile_offset = (chunk_col_idx % 32) * 16 + (chunk_col_idx // 32) * 4 + warp_y
                    sS_thread = cute.make_tensor(
                        sS.iterator + tile_offset, cute.make_layout((ELEMS_X,), stride=(16,))
                    )
                else:
                    sS_thread = cute.make_tensor(
                        (sS.iterator + warp_y * self.CHUNK_COLS + chunk_col_idx).align(ELEMS_X),
                        cute.make_layout((ELEMS_X,)),
                    )
                cute.autovec_copy(cute.recast_tensor(rS, Uint8), sS_thread)

            if in_bounds:
                gO = cute.make_tensor(
                    cute.make_ptr(
                        FP8_DTYPE,
                        mO.iterator.toint() + thread_row * H + col,
                        cute.AddressSpace.gmem,
                        assumed_align=ELEMS_X,
                    ),
                    out_frag_layout,
                )
                rO = cute.make_rmem_tensor((ELEMS_X,), FP8_DTYPE)
                rO_x4 = cute.recast_tensor(rO, Uint32)
                rO_x2 = cute.recast_tensor(rO, Uint16)
                for k in cutlass.range_constexpr(ELEMS_Y):
                    if cutlass.const_expr(BF16):
                        scale_and_pack = mul_bf16x2x2_cvt_bf16x4_to_fp8x4(FP8_DTYPE)
                        for w in cutlass.range_constexpr(PAIRS // 2):
                            rO_x4[w] = scale_and_pack(
                                rY[2 * w, k],
                                scale_rcp[2 * w],
                                rY[2 * w + 1, k],
                                scale_rcp[2 * w + 1],
                            )
                    else:
                        scale_and_cvt = _mul_cvt_x2(DTYPE, FP8_DTYPE)
                        for j in cutlass.range_constexpr(PAIRS):
                            rO_x2[j] = scale_and_cvt(rY[j, k], scale_rcp[j])
                    cute.copy(store_atom, rO, gO[None, k])

        cute.arch.sync_threads()
        if warp_idx == 0:
            self._store_scales(
                sS, mS, mOffsets, mFirstDims, num_tensors, chunk_row, chunk_col, M, H
            )

    @cute.jit
    def _store_scales(
        self,
        sS: cute.Tensor,
        mS: cute.Tensor,
        mOffsets: Optional[cute.Tensor],
        mFirstDims: Optional[cute.Tensor],
        num_tensors: Int32,
        chunk_row: Int64,
        chunk_col: Int64,
        M: Int64,
        H: Int64,
    ):
        """Write the chunk's staged scales, one warp issuing 16-byte stores: a GEMM-swizzled
        chunk is one contiguous tile of its member's scales, a compact one is 4 rows of
        CHUNK_COLS."""
        lane = cute.arch.lane_idx()
        # The member owning this chunk; only GEMM-swizzled scales depend on it.
        member_start_row = Int64(0)
        member_rows = M
        if cutlass.const_expr(
            self.cfg.WITH_GEMM_SWIZZLED_SCALES and self.cfg.SHAPE_REP == VARYING_FIRST_DIM
        ):
            member_id = _find_member(mOffsets, num_tensors, chunk_row * H)
            member_start_row = Int64(mOffsets[member_id]) // H
            member_rows = Int64(mFirstDims[member_id])
        gS = self._scale_view(mS, member_start_row, member_rows, H)
        scale_row = (chunk_row - member_start_row) // MXFP8_BLOCK_SCALING_SIZE
        if cutlass.const_expr(self.cfg.WITH_GEMM_SWIZZLED_SCALES):
            offset = cute.crd2idx((scale_row, chunk_col), gS.layout) + Int64(
                lane * self.SCALE_STORE_BYTES
            )
        else:
            pieces_per_row = self.CHUNK_COLS // self.SCALE_STORE_BYTES
            offset = cute.crd2idx(
                (
                    scale_row + lane // pieces_per_row,
                    chunk_col + lane % pieces_per_row * self.SCALE_STORE_BYTES,
                ),
                gS.layout,
            )
        src = cute.make_tensor(
            (sS.iterator + lane * self.SCALE_STORE_BYTES).align(self.SCALE_STORE_BYTES),
            cute.make_layout((self.SCALE_STORE_BYTES,)),
        )
        dst = cute.make_tensor(
            cute.make_ptr(
                Uint8,
                gS.iterator.toint() + offset,
                cute.AddressSpace.gmem,
                assumed_align=self.SCALE_STORE_BYTES,
            ),
            cute.make_layout((self.SCALE_STORE_BYTES,)),
        )
        staged = cute.make_rmem_tensor((self.SCALE_STORE_BYTES,), Uint8)
        cute.autovec_copy(src, staged)
        cute.autovec_copy(staged, dst)


@cute.jit
def _find_member(mOffsets: cute.Tensor, num_tensors: Int32, offset: Int64) -> Int32:
    """find_tensor_from_offsets with the lanes of a warp checking the offsets in parallel: the
    member is the number of later members' start offsets at or before `offset`."""
    lane = cute.arch.lane_idx()
    member_id = Int32(0)
    for first in cutlass.range_constexpr(0, MAX_SUPPORTED_TENSORS, THREADS_PER_WARP):
        if num_tensors > first + 1:
            started = False
            later_member = first + 1 + lane
            if later_member < num_tensors:
                started = Int64(mOffsets[later_member]) <= offset
            member_id += cute.arch.popc(cute.arch.vote_ballot_sync(started))
    return member_id


@functools.cache
def _swiglu_x2(dtype, clamped: bool) -> Callable:
    """(silu(act) * gate) * prob of two adjacent elements, rounded to the input dtype.

    Computes in FP32 exactly as the CUDA kernel does, with f32x2 math: the activation is doubled
    (silu_approx_x2 in util/math.h) and paired with the halved prob. The pair is a packed register
    for 16-bit input (act, gate and the result) and two FP32 registers for FP32 input. Clamped
    SwiGLU clamps act from above and gate on both sides, then offsets gate.
    """
    if dtype is Float32:
        unpack = "mov.b64 vx, {$r0};\n\tmov.b64 vg, {$r1};\n\t"
        pack = "mov.b64 {$w0}, vy;\n\t"
        pair_type = Int64
    else:
        if dtype is cutlass.BFloat16:
            # A BF16 value is the upper half of its FP32 value.
            widen = (
                "prmt.b32 x0, 0x0, {$r0}, 0x5410;\n\tprmt.b32 x1, 0x0, {$r0}, 0x7632;\n\t"
                "prmt.b32 g0, 0x0, {$r1}, 0x5410;\n\tprmt.b32 g1, 0x0, {$r1}, 0x7632;\n\t"
            )
        else:
            widen = (
                "mov.b32 {h0, h1}, {$r0};\n\tcvt.f32.f16 x0, h0;\n\tcvt.f32.f16 x1, h1;\n\t"
                "mov.b32 {h0, h1}, {$r1};\n\tcvt.f32.f16 g0, h0;\n\tcvt.f32.f16 g1, h1;\n\t"
            )
        fmt = "bf16x2" if dtype is cutlass.BFloat16 else "f16x2"
        unpack = widen + "mov.b64 vx, {x0, x1};\n\tmov.b64 vg, {g0, g1};\n\t"
        pack = f"mov.b64 {{y0, y1}}, vy;\n\tcvt.rn.{fmt}.f32 {{$w0}}, y1, y0;\n\t"
        pair_type = Int32
    if clamped:
        # $r3: limit, $r4: alpha / 2, $r5: glu_linear_offset
        activation = (
            "mov.b64 {g0, g1}, vg;\n\t"
            "min.xorsign.abs.f32 g0, g0, {$r3};\n\tmin.xorsign.abs.f32 g1, g1, {$r3};\n\t"
            "mov.b64 vg, {g0, g1};\n\tmov.b64 vc, {{$r5}, {$r5}};\n\tadd.rn.f32x2 vg, vg, vc;\n\t"
            "mov.b64 {x0, x1}, vx;\n\tmin.f32 x0, {$r3}, x0;\n\tmin.f32 x1, {$r3}, x1;\n\t"
            "mov.b64 vx, {x0, x1};\n\tmov.b64 vc, {{$r4}, {$r4}};\n\tmul.rn.f32x2 vt, vc, vx;\n\t"
        )
    else:
        activation = "mov.b64 vc, 0x3F0000003F000000;\n\tmul.rn.f32x2 vt, vc, vx;\n\t"  # x / 2
    asm = (
        "{\n\t"
        ".reg.b64 vx, vg, vt, vc, vy;\n\t.reg.b32 x0, x1, g0, g1, t0, t1, y0, y1;\n\t"
        ".reg.b16 h0, h1;\n\t"
        + unpack
        + activation
        + "mov.b64 {t0, t1}, vt;\n\ttanh.approx.f32 t0, t0;\n\ttanh.approx.f32 t1, t1;\n\t"
        "mov.b64 vt, {t0, t1};\n\t"
        "fma.rn.f32x2 vt, vx, vt, vx;\n\t"  # 2 * silu(x) = x * tanh(x / 2) + x
        "mul.rn.f32x2 vy, vt, vg;\n\t"
        "mov.b64 vc, {{$r2}, {$r2}};\n\tmul.rn.f32x2 vy, vy, vc;\n\t" + pack + "}"
    )

    @cute.jit
    def fn(act, gate, half_prob, limit, half_alpha, glu_linear_offset):
        args = [act, gate, half_prob]
        if cutlass.const_expr(clamped):
            args += [limit, half_alpha, glu_linear_offset]
        return cute.arch.inline_ptx(asm, write_only_types=[pair_type], read_only_args=args)

    return fn


@functools.cache
def _mul_cvt_x2(dtype, fp8_dtype) -> Callable:
    """fp8(y * scale) of two adjacent elements, for FP16 or FP32 y (one packed register or two)
    and two FP32 scales; returns the two FP8 bytes, the first element in the low byte."""
    out_op = "e5m2x2" if fp8_dtype is cutlass.Float8E5M2 else "e4m3x2"
    if dtype is Float32:
        widen = "mov.b64 vy, {$r0};\n\t"
    else:
        widen = (
            "mov.b32 {h0, h1}, {$r0};\n\tcvt.f32.f16 y0, h0;\n\tcvt.f32.f16 y1, h1;\n\t"
            "mov.b64 vy, {y0, y1};\n\t"
        )
    asm = (
        "{\n\t.reg.b64 vy;\n\t.reg.b32 y0, y1;\n\t.reg.b16 h0, h1;\n\t"
        + widen
        + "mul.rn.f32x2 vy, vy, {$r1};\n\tmov.b64 {y0, y1}, vy;\n\t"
        f"cvt.rn.satfinite.{out_op}.f32 {{$w0}}, y1, y0;\n\t}}"
    )

    @cute.jit
    def fn(y, scale):
        return cute.arch.inline_ptx(asm, write_only_types=[Uint16], read_only_args=[y, scale])

    return fn


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
