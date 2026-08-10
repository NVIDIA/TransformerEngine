# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Constants, scaling math, and E2M1 conversion helpers for NVFP4 kernels."""

import cutlass
from cutlass import cute
from cutlass import BFloat16, Float32, Float8E4M3FN, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from transformer_engine.common.CuTeDSL.utils import (
    BFLOAT16_MAX,
    FLOAT4E2M1_MAX,
    FLOAT8E4M3_MAX,
    FLOAT32_MAX,
    select_f32,
)


# Number of elements per NVFP4 scale block (they share one E4M3 scale factor).
NVFP4_BLOCK_SCALING_SIZE = 16
# Input row/col divisibility the CUDA tuned-1D kernel requires (16B TMA alignment).
NVFP4_SHAPE_ALIGNMENT = 32
# Padding TE applies to the scale tensors' outer / inner dim (NVFP4Quantizer::get_scale_shape).
NVFP4_SCALE_PAD_OUTER = 128
NVFP4_SCALE_PAD_INNER = 4


@cute.jit
def compute_global_encode_sf(global_amax: Float32) -> Float32:
    """like compute_global_encode_scaling_factor_FP4 in core_nvfp4.cuh

    Branch-free: the degenerate-amax fallback is a select rather than an `if`, because the
    row-scaled configuration evaluates this per scaling block and the DSL lowers a dynamic `if`
    to real divergent branches where the CUDA kernel's ternary gets predicated selects.
    """
    global_encode_scale = FLOAT8E4M3_MAX * FLOAT4E2M1_MAX / global_amax
    global_encode_scale = cute.arch.fmin(global_encode_scale, FLOAT32_MAX)
    degenerate = (global_amax == 0.0) | (global_encode_scale == 0.0)
    return select_f32(degenerate, Float32(1.0), global_encode_scale)


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
def cvt_f32x8_to_fp4x8_sr(
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    v4: Float32,
    v5: Float32,
    v6: Float32,
    v7: Float32,
    rbits03: Uint32,
    rbits47: Uint32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """Stochastic-rounding variant of cvt_f32x8_to_fp4x8: the same packing, but through
    `cvt.rs.satfinite.e2m1x4.f32`, which rounds each conversion with random bits. One b16 result
    holds four elements, `{v3, v2, v1, v0}` keeps v0 in the lowest nibble, and each four-element
    group consumes one 32-bit random word, exactly like
    mul_cvt_bf16_to_fp4_8x_stochastic_rounding<float> in ptx.cuh (whose multiplies are plain
    `mul.f32`, which is what the caller's scaling multiply lowers to).
    """
    asm = (
        "{\n"
        ".reg.b16 b03, b47;\n\t"
        "cvt.rs.satfinite.e2m1x4.f32 b03, {$4, $3, $2, $1}, $9;\n\t"
        "cvt.rs.satfinite.e2m1x4.f32 b47, {$8, $7, $6, $5}, $10;\n\t"
        "mov.b32 $0, {b03, b47};\n\t"
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
                rbits03.ir_value(loc=loc, ip=ip),
                rbits47.ir_value(loc=loc, ip=ip),
            ],
            asm,
            "=r,f,f,f,f,f,f,f,f,r,r",
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


@dsl_user_op
def mul_cvt_bf16x8_to_fp4x8_sr(
    v01: Uint32,
    v23: Uint32,
    v45: Uint32,
    v67: Uint32,
    coeff: Float32,
    rbits03: Uint32,
    rbits47: Uint32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """Stochastic-rounding variant of mul_cvt_bf16x8_to_fp4x8: the same bf16-coefficient fma
    scaling, but converted through `cvt.rs.satfinite.e2m1x4.f32` with random bits, mirroring
    mul_cvt_bf16_to_fp4_8x_stochastic_rounding<bf16> in ptx.cuh."""
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
        ".reg.b16 b03, b47;\n\t"
        "cvt.rs.satfinite.e2m1x4.f32 b03, {v3, v2, v1, v0}, $6;\n\t"
        "cvt.rs.satfinite.e2m1x4.f32 b47, {v7, v6, v5, v4}, $7;\n\t"
        "mov.b32 $0, {b03, b47};\n\t"
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
                rbits03.ir_value(loc=loc, ip=ip),
                rbits47.ir_value(loc=loc, ip=ip),
            ],
            asm,
            "=r,r,r,r,r,f,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
