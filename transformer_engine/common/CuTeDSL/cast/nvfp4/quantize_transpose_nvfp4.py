# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""NVFP4 quantization kernel implemented in CuTeDSL.

Replicates the core logic of quantize_transpose_nvfp4_tuned_1D.cuh: given a 2D tensor of BF16
values, quantize to NVFP4 (FP4E2M1 data + E4M3 per-block scales) and optionally also emit the
transposed (columnwise-scaled) output.

Main kernel logic resides in NVFP4QuantizeTransposeTuned1DKernel.

This is the initial version and only covers the rowwise, round-to-nearest configuration. It keeps
the CUDA kernel's thread arrangement, one thread owning whole scaling blocks, but not yet its data
movement: there is no shared memory, no TMA and no pipelining, so a CTA's chunk is a single tile
and the loads and stores go straight to global memory. The chunk is 64x64 rather than the CUDA
kernel's 128x128, and it is stated as a thread arrangement and a number of blocks per thread, with
the layout algebra deriving the tile, the grid and the loop bounds from those two shapes.

"""

import logging
import os
from typing import Optional

import cutlass
from cutlass import cute
from cutlass import Boolean, Float32, Int32, Int64, Uint32, Float4E2M1FN, Float8E4M3FN, BFloat16
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module
import tvm_ffi

from transformer_engine.common.CuTeDSL.utils import device_is_blackwell

CUTEDSL_DEBUG_LOGGING = os.environ.get("CUTEDSL_DEBUG_LOGGING", "0") == "1"

logger = logging.getLogger("transformer_engine.cutedsl.nvfp4")

# Number of elements per NVFP4 scale block (they share one E4M3 scale factor).
NVFP4_BLOCK_SCALING_SIZE = 16
# Input row/col divisibility the CUDA tuned-1D kernel requires (16B TMA alignment).
NVFP4_SHAPE_ALIGNMENT = 32
# Padding TE applies to the scale tensors' outer / inner dim (NVFP4Quantizer::get_scale_shape).
NVFP4_SCALE_PAD_OUTER = 128
NVFP4_SCALE_PAD_INNER = 4

FLOAT8E4M3_MAX = 448.0
FLOAT4E2M1_MAX = 6.0
BFLOAT16_MAX = 3.3895313892515355e38
FLOAT32_MAX = 3.4028234663852886e38


class NVFP4QuantizeConfig:
    """Instantiation parameters of the CuTE DSL kernel"""

    def __init__(
        self,
        use_stochastic_rounding: bool,
        use_fast_math: bool,
        row_scaled_nvfp4: bool,
        return_transpose: bool,
    ):
        self.USE_STOCHASTIC_ROUNDING = use_stochastic_rounding
        self.USE_FAST_MATH = use_fast_math
        if row_scaled_nvfp4 and return_transpose:
            raise ValueError("row-scaled NVFP4 quantization does not produce a transposed output")
        self.ROW_SCALED_NVFP4 = row_scaled_nvfp4
        self.RETURN_TRANSPOSE = return_transpose

    def __str__(self):
        return (
            f"NVFP4QuantizeConfig(use_stochastic_rounding={self.USE_STOCHASTIC_ROUNDING}, "
            f"use_fast_math={self.USE_FAST_MATH}, "
            f"row_scaled_nvfp4={self.ROW_SCALED_NVFP4}, "
            f"return_transpose={self.RETURN_TRANSPOSE})"
        )

    __repr__ = __str__


# Runs if CUTE_DSL_ENABLE_ASSERTIONS=1 or --enable-assertions present in cute.compile
def validate_tensor(tensor: Optional[cute.Tensor], expected_layout: cute.Layout, expected_dtype):
    if tensor is None:
        return
    cute.testing.assert_(tensor.layout == expected_layout, "Tensor layout does not match")
    cute.testing.assert_(tensor.element_type == expected_dtype, "Tensor dtype does not match")


def partition_scaling_blocks(
    t: cute.Tensor, tiler: cute.Shape, tv_layout: cute.Layout, chunk: cute.Coord, tidx: Int32
) -> cute.Tensor:
    """The scaling blocks one thread owns out of one CTA's chunk of `t`.

    Cut the chunk out, re-index it by (thread, value) instead of by position, keep this thread's
    values, and group them into the blocks that share a scale, giving ((block), (blocks)). The
    input, the output and a tensor of coordinates all go through this, which is what keeps them in
    step: none of them is indexed with arithmetic of its own.
    """
    chunk_values = cute.composition(cute.local_tile(t, tiler, chunk), tv_layout)[tidx, None]
    return cute.zipped_divide(chunk_values, (NVFP4_BLOCK_SCALING_SIZE,))


@cute.jit
def noop_flag_is_set(mNoop: cute.Pointer) -> Boolean:
    """Whether the cast_noop flag says this quantization is a no-op and must be skipped.

    mNoop is a pointer rather than a tensor so that one compiled kernel serves both a present and
    an absent flag, hence the address is checked before it is dereferenced, exactly like the CUDA
    C++ kernel's `noop != nullptr && noop[0] == 1.0f`. The two checks cannot be joined with `and`,
    which the DSL lowers to a non-short-circuiting op that would load from the null pointer.
    """
    flag_is_set = Boolean(False)
    if mNoop.toint() != Int64(0):
        flag_is_set = cute.make_tensor(mNoop, cute.make_layout((1,)))[0] == Float32(1.0)
    return flag_is_set


@cute.jit
def compute_global_encode_sf(global_amax: Float32) -> Float32:
    "like compute_global_encode_scaling_factor_FP4 in core_nvfp4.cuh"

    global_encode_scale = FLOAT8E4M3_MAX * FLOAT4E2M1_MAX / global_amax
    global_encode_scale = cute.arch.fmin(global_encode_scale, FLOAT32_MAX)
    if global_amax == 0.0 or global_encode_scale == 0.0:
        global_encode_scale = 1.0
    return global_encode_scale


@cute.jit
def compute_block_decode_sf(block_amax: Float32, global_encode_sf: Float32) -> Float8E4M3FN:
    "like quantization_and_transposition_SF::compute_decoding_scaling_factor in core_nvfp4.cuh"
    block_decode_sf = block_amax * (global_encode_sf * (1.0 / FLOAT4E2M1_MAX))
    block_decode_sf = cute.arch.fmin(block_decode_sf, FLOAT32_MAX)
    return block_decode_sf.to(Float8E4M3FN)


@cute.jit
def compute_block_encode_sf(
    block_decode_sf: Float8E4M3FN,
    global_encode_sf: Float32,
    sf_type: cutlass.Constexpr[type],
):
    "like compute_nvfp4_scaling_coefficient in quantize_tranpose_nvfp4_tuned_1D.cuh"
    if cutlass.const_expr(sf_type == Float32):
        global_decode_sf = 1.0 / global_encode_sf
        block_encode_sf = 1.0 / (block_decode_sf.to(Float32) * global_decode_sf)
        block_encode_sf = cute.arch.fmin(block_encode_sf, FLOAT32_MAX)
        return block_encode_sf
    elif cutlass.const_expr(sf_type == BFloat16):
        block_encode_sf = global_encode_sf / block_decode_sf.to(Float32)
        block_encode_sf = cute.arch.fmin(block_encode_sf, BFLOAT16_MAX)
        return block_encode_sf.to(BFloat16)
    else:
        raise ValueError("Unsupported scaling-factor type. Only FP32 and BF16 are supported.")


@dsl_user_op
def cvt_f32x8_to_fp4x8(
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    v4: Float32,
    v5: Float32,
    v6: Float32,
    v7: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """Convert eight f32 values to eight E2M1 values packed into four bytes, v0 in the lowest.

    `cvt d, a, b` puts fp4(a) in d[7:4] and fp4(b) in d[3:0], so each pair is fed as (odd, even)
    to keep the earlier element in the low nibble. This is the same instruction, and hence the
    same rounding and saturation, as mul_cvt_bf16_to_fp4_8x_round_to_nearest<float> in ptx.cuh.
    The scaling multiply is left to the caller rather than fused, which is exact because the
    instruction that one fuses is a plain `mul.f32x2`; the bf16 coefficient fuses an fma instead
    and is not interchangeable that way, hence mul_cvt_bf16x8_to_fp4x8 below.
    """
    asm = (
        "{\n"
        ".reg.b8 f0; .reg.b8 f1; .reg.b8 f2; .reg.b8 f3;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f0, $2, $1;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f1, $4, $3;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f2, $6, $5;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f3, $8, $7;\n\t"
        "mov.b32 $0, {f0, f1, f2, f3};\n\t"
        "}"
    )
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                v0.ir_value(loc=loc, ip=ip),
                v1.ir_value(loc=loc, ip=ip),
                v2.ir_value(loc=loc, ip=ip),
                v3.ir_value(loc=loc, ip=ip),
                v4.ir_value(loc=loc, ip=ip),
                v5.ir_value(loc=loc, ip=ip),
                v6.ir_value(loc=loc, ip=ip),
                v7.ir_value(loc=loc, ip=ip),
            ],
            asm,
            "=r,f,f,f,f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def mul_cvt_bf16x8_to_fp4x8(
    v01: Uint32,
    v23: Uint32,
    v45: Uint32,
    v67: Uint32,
    coeff: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """Scale eight bf16 values by a bf16 coefficient and pack them into eight E2M1 values.

    This is mul_cvt_bf16_to_fp4_8x_round_to_nearest<bf16> from ptx.cuh instruction for
    instruction, which matters for exactly one input: `fma` against a zero addend flushes a -0
    product to +0 where a plain multiply keeps the sign, and E2M1 has a signed zero, so scaling
    any other way would encode a -0 element as 0x8 instead of 0x0. Everything else about it
    agrees with widening to f32 and multiplying, since a bf16 product has at most 16 mantissa
    bits.

    The values arrive as four u32s of bf16 pairs, which is how they already sit in the fragment.
    The coefficient arrives widened to f32 and is rounded back to bf16 here, which is exact, and
    keeps the operands to the f32 and u32 registers the other PTX wrappers here use.
    """
    asm = (
        "{\n"
        ".reg.f32 zero;\n\t"
        "mov.b32 zero, 0;\n\t"
        ".reg.b16 c;\n\t"
        "cvt.rn.bf16.f32 c, $5;\n\t"
        ".reg.b16 h0, h1, h2, h3, h4, h5, h6, h7;\n\t"
        "mov.b32 {h0, h1}, $1;\n\t"
        "mov.b32 {h2, h3}, $2;\n\t"
        "mov.b32 {h4, h5}, $3;\n\t"
        "mov.b32 {h6, h7}, $4;\n\t"
        ".reg.f32 v0, v1, v2, v3, v4, v5, v6, v7;\n\t"
        "fma.rn.f32.bf16 v0, h0, c, zero;\n\t"
        "fma.rn.f32.bf16 v1, h1, c, zero;\n\t"
        "fma.rn.f32.bf16 v2, h2, c, zero;\n\t"
        "fma.rn.f32.bf16 v3, h3, c, zero;\n\t"
        "fma.rn.f32.bf16 v4, h4, c, zero;\n\t"
        "fma.rn.f32.bf16 v5, h5, c, zero;\n\t"
        "fma.rn.f32.bf16 v6, h6, c, zero;\n\t"
        "fma.rn.f32.bf16 v7, h7, c, zero;\n\t"
        ".reg.b8 f0, f1, f2, f3;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f0, v1, v0;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f1, v3, v2;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f2, v5, v4;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f3, v7, v6;\n\t"
        "mov.b32 $0, {f0, f1, f2, f3};\n\t"
        "}"
    )
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                v01.ir_value(loc=loc, ip=ip),
                v23.ir_value(loc=loc, ip=ip),
                v45.ir_value(loc=loc, ip=ip),
                v67.ir_value(loc=loc, ip=ip),
                coeff.ir_value(loc=loc, ip=ip),
            ],
            asm,
            "=r,r,r,r,r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


class NVFP4QuantizeTransposeTuned1DKernel:
    """Tuned kernel to cast to NVFP4 and transpose"""

    # Each thread block processes a _chunk_ of the input tensor, which is a 2D sub-tensor.
    # The quantization this kernel performs is almost a pure point-wise operation, except that
    # NVFP4_BLOCK_SCALING_SIZE elements share a single block scaling factor and the entire tensor
    # has a global scaling factor.
    # Therefore, the chunk also correponds to a 2D sub-tensor of all the other tensors.
    #
    # A chunk is made of multiple _tiles_, which are processed sequentially by the thread block.
    # While the tiles are processed sequentially, there may be more than one tile in-flight at a time.
    # PREFETCH_STAGES specifies how many tiles to prefetch to SMEM, in addition to the current tile.
    #
    # Optionally, the kernel can be PERSISTENT, in which case it will use Cluster Launch Control to
    # achieve dynamic persistent tile scheduling, through work stealing.

    # Tunable config (mirroring the CUDA version)
    # A chunk is one tile for now: without SMEM staging there is nothing to gain from a CTA
    # covering several tiles.
    PREFETCH_STAGES = 1
    PERSISTENT = False

    # The two shapes the rest of the tiling follows from. A row of threads covers a whole number of
    # scaling blocks, and one thread owns whole blocks, which is what makes the block amax an
    # intra-thread reduction needing no cross-thread communication. The chunk a CTA covers is not
    # spelled out: it is what these two make, 32 * 2 rows by 4 * 16 columns.
    THREAD_SHAPE = (32, 4)
    BLOCKS_PER_THREAD = 2

    def __init__(self, cfg):
        self.cfg = cfg

    def rowwise_partitioning(self):
        """How a chunk is divided over the CTA's threads, for the launch and the device code both.

        Stating the thread arrangement and what one thread owns is the whole input: make_layout_tv
        works out the tile a CTA covers from them, so the chunk dimensions, the thread count and
        the number of blocks a thread walks are all read back off the layouts afterwards rather
        than being computed here.
        """
        thr_layout = cute.make_ordered_layout(self.THREAD_SHAPE, order=(1, 0))
        # A value row is one scaling block: NVFP4_BLOCK_SCALING_SIZE elements along the input's
        # major mode, which is also what lets a block be a single vectorized access.
        tiler, tv_layout = cute.make_layout_tv(
            thr_layout,
            cute.make_ordered_layout(
                (self.BLOCKS_PER_THREAD, NVFP4_BLOCK_SCALING_SIZE), order=(1, 0)
            ),
        )
        # The same partitioning as the scales see it, one per block, so its tiler comes out as the
        # chunk's footprint in the scale tensor rather than in the input.
        tiler_scale, tv_layout_scale = cute.make_layout_tv(
            thr_layout, cute.make_layout((self.BLOCKS_PER_THREAD, 1))
        )
        return tiler, tv_layout, tiler_scale, tv_layout_scale

    # Host-side kernel launch
    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,  # (M, N) bf16 input
        mO_row: cute.Tensor,  # (M, N) fp4 rowwise output
        mS_row: cute.Tensor,  # (roundup(M, 128), roundup(ceil(N / 16), 4)) e4m3 scales
        mO_col: Optional[cute.Tensor],  # (N, M) fp4 transposed output
        mS_col: Optional[cute.Tensor],  # (roundup(N, 128), roundup(ceil(M / 16), 4)) e4m3 scales
        mAmaxRow: cute.Tensor,  # (M,) f32 per-row amax if ROW_SCALED_NVFP4, else (1,) global amax
        mAmaxCol: Optional[cute.Tensor],  # (1,) f32 global amax of the transposed output
        mNoop: cute.Pointer,  # f32 cast-noop flag; may be null, checked on device
        mRngState: Optional[cute.Tensor],  # (2,) i64 Philox {seed, offset}
        stream: CUstream,
    ):
        """AOT-compiled host entrypoint. `quantize_transpose_nvfp4_cutedsl.cuh` passes these
        arguments in this exact order via tvm-ffi, and the config fixes which of the optional
        ones are present (see `compile_cutedsl_function_from_cfg`).

        M, N are the input's *flattened* 2D dims, both multiples of NVFP4_SHAPE_ALIGNMENT; a
        rank > 2 input already arrives flattened.
        All tensors are row-major (with rightmost stride 1).
        FP4 extents are logical element counts, not the halved extents of the uint8 buffer TE
        actually allocates.
        """
        if cutlass.const_expr(CUTEDSL_DEBUG_LOGGING):
            cute.printf(
                "[CuTeDSL] NVFP4QuantizeTransposeTuned1DKernel.__call__() with config:"
                f" {self.cfg}\n"
            )

        # Declining here rather than in NVFP4QuantizeConfig keeps the config able to describe every
        # CUDA configuration: the exception propagates out of cute.compile and the dispatcher falls
        # back to the CUDA C++ kernel.
        if cutlass.const_expr(self.cfg.RETURN_TRANSPOSE):
            raise NotImplementedError("the transposed output is not implemented yet")
        if cutlass.const_expr(self.cfg.ROW_SCALED_NVFP4):
            raise NotImplementedError("row-scaled NVFP4 quantization is not implemented yet")
        if cutlass.const_expr(self.cfg.USE_STOCHASTIC_ROUNDING):
            raise NotImplementedError("stochastic rounding is not implemented yet")

        ## Validation

        # Validate input and output tensor layouts
        (M, N) = mX.shape
        mX_layout = cute.make_ordered_layout((M, N), order=(1, 0))
        mO_row_layout = mX_layout
        mO_col_layout = cute.make_ordered_layout((N, M), order=(1, 0))
        validate_tensor(mX, mX_layout, BFloat16)
        validate_tensor(mO_row, mO_row_layout, Float4E2M1FN)
        validate_tensor(mO_col, mO_col_layout, Float4E2M1FN)

        # Validate scaling factor tensor layouts
        mS_row_layout = cute.make_ordered_layout(
            (
                cute.round_up(M, NVFP4_SCALE_PAD_OUTER),
                cute.round_up(cute.ceil_div(N, NVFP4_BLOCK_SCALING_SIZE), NVFP4_SCALE_PAD_INNER),
            ),
            order=(1, 0),
        )
        mS_col_layout = cute.make_ordered_layout(
            (
                cute.round_up(N, NVFP4_SCALE_PAD_OUTER),
                cute.round_up(cute.ceil_div(M, NVFP4_BLOCK_SCALING_SIZE), NVFP4_SCALE_PAD_INNER),
            ),
            order=(1, 0),
        )
        validate_tensor(mS_row, mS_row_layout, Float8E4M3FN)
        validate_tensor(mS_col, mS_col_layout, Float8E4M3FN)

        # Validate amax tensor layouts
        if cutlass.const_expr(self.cfg.ROW_SCALED_NVFP4):
            mAmaxRow_layout = cute.make_layout((M,))
            validate_tensor(mAmaxRow, mAmaxRow_layout, Float32)
        else:
            mAmaxRow_layout = cute.make_layout((1,))
            validate_tensor(mAmaxRow, mAmaxRow_layout, Float32)
        mAmaxCol_layout = cute.make_layout((1,))
        validate_tensor(mAmaxCol, mAmaxCol_layout, Float32)

        # Validate RNG state tensor layout
        mRngState_layout = cute.make_layout((2,))
        validate_tensor(mRngState, mRngState_layout, Int64)

        ## Grid and block size calculation
        # One CTA per chunk, and a chunk is whatever tile the partitioning covers, so both the
        # grid and the block are read off it. X indexes columns and Y rows, as in the CUDA kernel's
        # ctaid_X / ctaid_Y.
        tiler, tv_layout, _, _ = self.rowwise_partitioning()
        chunks_m, chunks_n = cute.ceil_div(mX.shape, tiler)
        grid = [chunks_n, chunks_m, 1]
        block = [cute.size(tv_layout, mode=[0]), 1, 1]

        self.kernel(
            mX,
            mO_row,
            mS_row,
            mO_col,
            mS_col,
            mAmaxRow,
            mAmaxCol,
            mNoop,
            mRngState,
        ).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,  # (M, N) bf16 input
        mO_row: cute.Tensor,  # (M, N) fp4 rowwise output
        mS_row: cute.Tensor,  # (roundup(M, 128), roundup(ceil(N / 16), 4)) e4m3 scales
        mO_col: Optional[cute.Tensor],  # (N, M) fp4 transposed output
        mS_col: Optional[cute.Tensor],  # (roundup(N, 128), roundup(ceil(M / 16), 4)) e4m3 scales
        mAmaxRow: cute.Tensor,  # (M,) f32 per-row amax if ROW_SCALED_NVFP4, else (1,) global amax
        mAmaxCol: Optional[cute.Tensor],  # (1,) f32 global amax of the transposed output
        mNoop: cute.Pointer,  # f32 cast-noop flag; may be null, checked on device
        mRngState: Optional[cute.Tensor],  # (2,) i64 Philox {seed, offset}
    ):
        """Device entry for the NVFP4 tuned-1D quantize-transpose kernel."""
        if not noop_flag_is_set(mNoop):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, bidy, _ = cute.arch.block_idx()

            global_encode_sf = compute_global_encode_sf(mAmaxRow[0])
            if cutlass.const_expr(self.cfg.USE_FAST_MATH):
                sf_type = BFloat16
            else:
                sf_type = Float32

            tiler, tv_layout, tiler_scale, tv_layout_scale = self.rowwise_partitioning()

            # This thread's share of this CTA's chunk, as the scaling blocks it owns. The output is
            # partitioned by the same tiler as the input, which it can be because the FFI hands it
            # over with logical extents, and only then read as the bytes it is stored in: two
            # values to a byte and four bytes to what one conversion produces. The coordinates ride
            # along through the same partitioning, so the bounds check reads the coordinate a
            # thread is about to touch rather than recomputing it from the block and thread indices.
            chunk = (bidy, bidx)
            tXgX = partition_scaling_blocks(mX, tiler, tv_layout, chunk, tidx)
            tOgO = cute.recast_tensor(
                partition_scaling_blocks(mO_row, tiler, tv_layout, chunk, tidx), Uint32
            )
            tXcX = partition_scaling_blocks(
                cute.make_identity_tensor(mX.shape), tiler, tv_layout, chunk, tidx
            )
            # One scale per block, so the scales need no block mode of their own.
            tSgS = cute.composition(cute.local_tile(mS_row, tiler_scale, chunk), tv_layout_scale)[
                tidx, None
            ]

            # Registers for one block at a time, shaped by the gmem partition they are copied from
            # so the two are congruent by construction. The input is also read as the u32 pairs of
            # bf16 that the fast-math scaling instruction takes.
            frgX = cute.make_fragment_like(tXgX[None, 0])
            frgO = cute.make_fragment_like(tOgO[None, 0])
            frgX_u32 = cute.recast_tensor(frgX, Uint32)

            for blk in cutlass.range_constexpr(cute.size(tXgX, mode=[1])):
                # A block is NVFP4_BLOCK_SCALING_SIZE aligned columns of a single row and N is a
                # multiple of NVFP4_SHAPE_ALIGNMENT, so its first coordinate decides for the whole
                # block, and the scale it feeds shares its fate.
                if cute.elem_less(tXcX[0, blk], mX.shape):
                    cute.autovec_copy(tXgX[None, blk], frgX)
                    values = frgX.load().to(Float32)

                    # Widening bf16 to f32 is exact, so this matches the CUDA kernel's 16-bit
                    # abs-max bit for bit. There is no vector abs, hence max(max(v), -min(v)); a
                    # full reduction lowers to vector.reduction and yields a bare MLIR value, hence
                    # the Float32 wrapping.
                    largest = Float32(values.reduce(cute.ReductionOp.MAX, -Float32.inf, 0))
                    smallest = Float32(values.reduce(cute.ReductionOp.MIN, Float32.inf, 0))
                    block_amax = cute.arch.fmax(largest, -smallest)

                    block_decode_sf = compute_block_decode_sf(block_amax, global_encode_sf)
                    tSgS[blk] = block_decode_sf

                    coeff = compute_block_encode_sf(block_decode_sf, global_encode_sf, sf_type)

                    # Scale and convert eight values at a time, which is what the conversion
                    # instruction takes and the one place the vector has to be opened up; satfinite
                    # does the clamping to +-FLOAT4E2M1_MAX and the nibble packing. One conversion
                    # fills one of the block's u32s, so the fragment says how many there are. Each
                    # coefficient type gets the multiply the CUDA kernel pairs it with, an f32
                    # multiply for the f32 one and an fma against a zero addend for the bf16 one.
                    # The two agree on every input whose product is not zero, but the fma turns a
                    # -0 product into +0 and E2M1 has a signed zero, so the instruction has to
                    # match.
                    if cutlass.const_expr(self.cfg.USE_FAST_MATH):
                        coeff_f32 = coeff.to(Float32)
                        for pack in cutlass.range_constexpr(cute.size(frgO)):
                            u = 4 * pack
                            frgO[pack] = mul_cvt_bf16x8_to_fp4x8(
                                frgX_u32[u],
                                frgX_u32[u + 1],
                                frgX_u32[u + 2],
                                frgX_u32[u + 3],
                                coeff_f32,
                            )
                    else:
                        scaled = values * coeff
                        for pack in cutlass.range_constexpr(cute.size(frgO)):
                            v = 8 * pack
                            frgO[pack] = cvt_f32x8_to_fp4x8(
                                scaled[v],
                                scaled[v + 1],
                                scaled[v + 2],
                                scaled[v + 3],
                                scaled[v + 4],
                                scaled[v + 5],
                                scaled[v + 6],
                                scaled[v + 7],
                            )
                    cute.autovec_copy(frgO, tOgO[None, blk])


def compile_cutedsl_function_from_cfg(cfg):
    """
    Return the compiled CuTeDSL function object for the given NVFP4 quantization config.

    The fake tensors below are what fix the compiled ABI, so they must agree with `__call__`'s
    signature.
    """

    if not device_is_blackwell():
        raise RuntimeError("CuTeDSL NVFP4 backend requires compute capability >= 10.0 (Blackwell)")

    kernel_obj = NVFP4QuantizeTransposeTuned1DKernel(cfg)

    def _gmem(dtype, shape, stride_order, align):
        """Fake row-major GMEM tensor; `align` is the assumed base alignment in bytes."""
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            stride_order=stride_order,
            memspace=cute.AddressSpace.gmem,
            assumed_align=align,
        )

    # Flattened 2D input dims. Their divisibility is also what keeps every row stride a multiple
    # of 16B as TMA requires: 2*N bytes for the bf16 input, N/2 and M/2 for the fp4 outputs.
    sym_M = cute.sym_int32(divisibility=NVFP4_SHAPE_ALIGNMENT)
    sym_N = cute.sym_int32(divisibility=NVFP4_SHAPE_ALIGNMENT)

    # Scale extents are TE's padded ones (NVFP4Quantizer::get_scale_shape):
    #   mS_row: (roundup(M, 128), roundup(ceil(N / 16), 4))
    #   mS_col: (roundup(N, 128), roundup(ceil(M / 16), 4))
    # Neither is expressible in sym_M / sym_N (SymInt has no `//` or `+`), so the scales get
    # fresh syms carrying only the divisibility the padding guarantees.
    scale_row_shape = (
        cute.sym_int32(divisibility=NVFP4_SCALE_PAD_OUTER),
        cute.sym_int32(divisibility=NVFP4_SCALE_PAD_INNER),
    )
    scale_col_shape = (
        cute.sym_int32(divisibility=NVFP4_SCALE_PAD_OUTER),
        cute.sym_int32(divisibility=NVFP4_SCALE_PAD_INNER),
    )

    # mX / mO_row / mS_row: (M, N) input, (M, N) rowwise fp4 output and its scales. The fp4
    # extents are logical element counts, so the row stride is N while only N/2 bytes are
    # stored. That is the kernel's view of it; the compiled ABI is not, because the DSL rewrites
    # every Float4E2M1FN argument into the packed float4_e2m1fnx2 form DLPack can express, with
    # the contiguous extent and the strides above it halved. What the caller hands over is
    # therefore the packed tensor, which is what tvm_ffi_bridge.h builds for an FP4 TE tensor.
    in_fake = _gmem(BFloat16, (sym_M, sym_N), stride_order=(1, 0), align=16)
    out_row_fake = _gmem(Float4E2M1FN, (sym_M, sym_N), stride_order=(1, 0), align=16)
    scale_row_fake = _gmem(Float8E4M3FN, scale_row_shape, stride_order=(1, 0), align=4)

    # mO_col / mS_col: (N, M) transposed fp4 output and its scales, present iff a transpose is
    # produced.
    out_col_fake = (
        _gmem(Float4E2M1FN, (sym_N, sym_M), stride_order=(1, 0), align=16)
        if cfg.RETURN_TRANSPOSE
        else None
    )
    scale_col_fake = (
        _gmem(Float8E4M3FN, scale_col_shape, stride_order=(1, 0), align=4)
        if cfg.RETURN_TRANSPOSE
        else None
    )

    # mAmaxRow: unpadded per-row amax (M,) when row-scaled, else a single global amax (1,).
    amax_row_fake = (
        _gmem(Float32, (sym_M,), stride_order=(0,), align=4)
        if cfg.ROW_SCALED_NVFP4
        else _gmem(Float32, (1,), stride_order=(0,), align=4)
    )
    # mAmaxCol: (1,) global amax of the transposed output, present iff a transpose is produced.
    amax_col_fake = (
        _gmem(Float32, (1,), stride_order=(0,), align=4) if cfg.RETURN_TRANSPOSE else None
    )

    # mNoop: always-present f32 pointer to the single-element cast-noop flag. A pointer rather
    # than a tensor so the runtime value can be null without changing the compiled ABI -- the
    # kernel does the null-check and the noop[0] == 1.0f test on device. The null address here
    # is only a compile-time placeholder for the pointer the dispatcher passes at launch.
    noop_fake = cute.runtime.nullptr(Float32, mem_space=cute.AddressSpace.gmem, assumed_align=4)

    # mRngState: Philox {seed, offset}; present (and consumed) only for stochastic rounding.
    rng_state_fake = (
        _gmem(Int64, (2,), stride_order=(0,), align=8) if cfg.USE_STOCHASTIC_ROUNDING else None
    )

    compiled = cute.compile(
        kernel_obj,
        in_fake,  # mX
        out_row_fake,  # mO_row
        scale_row_fake,  # mS_row
        out_col_fake,  # mO_col
        scale_col_fake,  # mS_col
        amax_row_fake,  # mAmaxRow
        amax_col_fake,  # mAmaxCol
        noop_fake,  # mNoop
        rng_state_fake,  # mRngState
        cute.runtime.make_fake_stream(),  # stream
        options="--enable-tvm-ffi",
    )
    return compiled


def get_nvfp4_quantization_function(
    fn_name: str,
    use_stochastic_rounding: bool,
    use_fast_math: bool,
    row_scaled_nvfp4: bool,
    return_transpose: bool,
) -> bool:
    """Compile the NVFP4 quantize kernel for this config and register it in the TVM-FFI global
    registry under EXACTLY `fn_name` (the key the C++ dispatcher built). Returns True if a kernel
    is registered under `fn_name`, False if the config is unsupported or compilation failed, in
    which case the caller caches the negative result and falls back to the CUDA C++ kernel.
    """

    # Already registered (e.g. by a prior call) -> supported.
    if tvm_ffi.get_global_func(fn_name, allow_missing=True) is not None:
        return True

    try:
        cfg = NVFP4QuantizeConfig(
            use_stochastic_rounding=use_stochastic_rounding,
            use_fast_math=use_fast_math,
            row_scaled_nvfp4=row_scaled_nvfp4,
            return_transpose=return_transpose,
        )
    except ValueError as e:
        # The exception message states exactly why the config is unsupported. Surfacing it as a
        # warning lets the C++ dispatcher's CUDA fallback be recognized as expected.
        logger.warning(
            "CuTeDSL NVFP4 backend does not support this config, "
            "falling back to the CUDA C++ kernel: %s",
            e,
        )
        return False

    logger.debug("Compiling CuTeDSL NVFP4 quantization kernel for %s", cfg)
    try:
        compiled = compile_cutedsl_function_from_cfg(cfg)
    except NotImplementedError as e:
        # Raised while tracing a config this kernel does not cover yet, which is expected rather
        # than a failure, so it is logged apart from a genuine compilation error.
        logger.warning(
            "CuTeDSL NVFP4 backend does not implement this config yet, "
            "falling back to the CUDA C++ kernel: %s",
            e,
        )
        return False
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(
            "CuTeDSL NVFP4 kernel compilation failed, falling back to the CUDA C++ kernel: %s",
            e,
        )
        return False
    tvm_ffi.register_global_func(fn_name, compiled, override=True)

    return True
