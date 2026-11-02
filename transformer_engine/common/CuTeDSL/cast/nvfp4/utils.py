# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Constants, scaling math, and E2M1 conversion helpers for NVFP4 kernels."""

from typing import Optional

import cutlass
from cutlass import cute
from cutlass import BFloat16, Boolean, Float32, Float8E4M3FN, Int32, Int64, Uint32, Uint64
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

FLOAT32_MAX = 3.4028234663852886e38
BFLOAT16_MAX = 3.3895313892515355e38
FLOAT8E4M3_MAX = 448.0
FLOAT4E2M1_MAX = 6.0

# Number of elements per NVFP4 scale block (they share one E4M3 scale factor).
NVFP4_BLOCK_SCALING_SIZE = 16
# Input row/col divisibility the CUDA tuned-1D kernel requires (16B TMA alignment).
NVFP4_SHAPE_ALIGNMENT = 32
# Padding TE applies to the scale tensors' outer / inner dim (NVFP4Quantizer::get_scale_shape).
NVFP4_SCALE_PAD_OUTER = 128
NVFP4_SCALE_PAD_INNER = 4


# Runs if CUTE_DSL_ENABLE_ASSERTIONS=1 or --enable-assertions present in cute.compile
def validate_tensor(tensor: Optional[cute.Tensor], expected_layout: cute.Layout, expected_dtype):
    """Assert at trace time that an optional tensor has the expected layout and dtype."""
    if tensor is None:
        return
    cute.testing.assert_(tensor.layout == expected_layout, "Tensor layout does not match")
    cute.testing.assert_(tensor.element_type == expected_dtype, "Tensor dtype does not match")


@dsl_user_op
def select_f32(cond: Boolean, if_true: Float32, if_false: Float32, *, loc=None, ip=None) -> Float32:
    """`cond ? if_true : if_false` as one select, never a branch."""
    return Float32(
        mlir_arith.select(
            cond.ir_value(loc=loc, ip=ip),
            if_true.ir_value(loc=loc, ip=ip),
            if_false.ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def pack_u32x2(lo: Uint32, hi: Uint32) -> Int64:
    """Pack two u32 into one 64-bit register (register-pair move, no real instruction)."""
    return cute.arch.inline_ptx(
        "mov.b64 {$w0}, {{$r0}, {$r1}};",
        write_only_types=[Int64],
        read_only_args=[lo, hi],
    )


def make_prmt_u32(selector: int):
    """A byte-permute op with the 16-bit selector baked in as an immediate.

    prmt.b32 indexes the eight source bytes {a0..a3, b0..b7} and each selector nibble picks the
    byte for one destination position, low nibble first. 0x5410 interleaves the low halves of a
    and b (a0 a1 b0 b1), 0x7632 the high halves.
    """

    @cute.jit
    def prmt_u32(a: Uint32, b: Uint32) -> Uint32:
        return cute.arch.inline_ptx(
            f"prmt.b32 {{$w0}}, {{$r0}}, {{$r1}}, {selector:#x};",
            write_only_types=[Uint32],
            read_only_args=[a, b],
        )

    return prmt_u32


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


# --- Register-resident kernels: global memory access by address ---
#
# The register kernels address global memory with 64-bit byte addresses rather than through
# layouts: every access is one row offset from a thread-constant base, and the cache hints and
# vector widths below are part of what makes them fast.


def _vec(prefix: str, count: int) -> str:
    return "{" + ", ".join(f"{{${prefix}{i}}}" for i in range(count)) + "}"


@cute.jit
def row_offset(row: Int32, pitch: Int32) -> Int64:
    """row * pitch for non-negative 32-bit operands. A product of two zero-extended words lowers
    to one IMAD.WIDE.U32; plain arithmetic rather than inline PTX, so ptxas can recompute an
    address instead of keeping it live."""
    return Int64(Uint64(Uint32(row)) * Uint64(Uint32(pitch)))


@cute.jit
def prefetch_l2(addr: Int64):
    """Bring one global cache line into L2. Names no destination register, so it can be issued
    ahead of a chain the thread is already stalled on."""
    cute.arch.inline_ptx("prefetch.global.L2 [{$r0}];", read_only_args=[addr])


@cute.jit
def ld_global_nc_evict_first_v8(addr: Int64):
    """32 bytes of input read once: read-only path, evicted first from L2. 32-byte aligned."""
    return list(
        cute.arch.inline_ptx(
            "ld.global.nc.L2::evict_first.v8.b32 " + _vec("w", 8) + ", [{$r0}];",
            write_only_types=[Uint32] * 8,
            read_only_args=[addr],
        )
    )


@cute.jit
def ld_global_evict_last_v8(addr: Int64):
    """32 bytes of input that is read again soon: coherent path (the read-only path would not
    keep the line for the re-read) and kept in L2 until then. 32-byte aligned."""
    return list(
        cute.arch.inline_ptx(
            "ld.global.L2::evict_last.v8.b32 " + _vec("w", 8) + ", [{$r0}];",
            write_only_types=[Uint32] * 8,
            read_only_args=[addr],
        )
    )


@cute.jit
def ld_global_v2(addr: Int64):
    """8 bytes, 8-byte aligned."""
    return list(
        cute.arch.inline_ptx(
            "ld.global.v2.b32 " + _vec("w", 2) + ", [{$r0}];",
            write_only_types=[Uint32] * 2,
            read_only_args=[addr],
        )
    )


@cute.jit
def unpack_u64(v: Int64):
    """The (low, high) words of a 64-bit value."""
    return cute.arch.inline_ptx(
        "mov.b64 {{$w0}, {$w1}}, {$r0};", write_only_types=[Uint32, Uint32], read_only_args=[v]
    )


@cute.jit
def st_global_u8(addr: Int64, value: Uint32):
    """Store the low byte of value."""
    cute.arch.inline_ptx("st.global.u8 [{$r0}], {$r1};", read_only_args=[addr, value])


@cute.jit
def st_global_u16(addr: Int64, value: Uint32):
    """Store the low two bytes of value, 2-byte aligned."""
    cute.arch.inline_ptx("st.global.u16 [{$r0}], {$r1};", read_only_args=[addr, value])


@cute.jit
def st_global_u32(addr: Int64, value: Uint32):
    """4-byte aligned."""
    cute.arch.inline_ptx("st.global.b32 [{$r0}], {$r1};", read_only_args=[addr, value])


@cute.jit
def st_global_v2(addr: Int64, w0: Uint32, w1: Uint32):
    """8 bytes, 8-byte aligned."""
    cute.arch.inline_ptx("st.global.v2.b32 [{$r0}], {{$r1}, {$r2}};", read_only_args=[addr, w0, w1])


@cute.jit
def st_global_v4(addr: Int64, words):
    """16 bytes, 16-byte aligned."""
    cute.arch.inline_ptx(
        "st.global.v4.b32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
        read_only_args=[addr] + list(words),
    )


@cute.jit
def st_global_v8(addr: Int64, words):
    """32 bytes (one sector), 32-byte aligned."""
    cute.arch.inline_ptx(
        "st.global.v8.b32 [{$r0}], {" + ", ".join(f"{{$r{i}}}" for i in range(1, 9)) + "};",
        read_only_args=[addr] + list(words),
    )


# --- Register-resident kernels: the numeric core ---
#
# A 16-element block is eight packed bf16 words; word h holds element 2h in its low half and
# 2h+1 in its high half. Block scale -> coefficient -> E2M1 follows TE's formulas
# (compute_decoding_scaling_factor, compute_nvfp4_scaling_coefficient<float>), so the output is
# bit-identical to the CUDA kernels.


@cute.jit
def swap_halves_b32(v: Uint32) -> Uint32:
    """Exchange the two 16-bit halves of a word."""
    return cute.arch.inline_ptx(
        "prmt.b32 {$w0}, {$r0}, {$r0}, 0x1032;", write_only_types=[Uint32], read_only_args=[v]
    )


@cute.jit
def rcp_rn_f32(x: Float32) -> Float32:
    """Correctly rounded reciprocal, bit for bit the IEEE quotient 1 / x."""
    return cute.arch.inline_ptx(
        "rcp.rn.f32 {$w0}, {$r0};", write_only_types=[Float32], read_only_args=[x]
    )


@cute.jit
def e4m3_code(x: Float32) -> Uint32:
    """The E4M3 byte of x (round to nearest, saturating to 448), as static_cast<fp8e4m3> does.
    Its clamp to FLT_MAX in compute_decoding_scaling_factor cannot change the byte: the only
    value it alters, +inf, saturates to the same 448."""
    return cute.arch.inline_ptx(
        "{ .reg .b16 h; cvt.rn.satfinite.e4m3x2.f32 h, {$r0}, {$r0}; "
        "cvt.u32.u16 {$w0}, h; and.b32 {$w0}, {$w0}, 255; }",
        write_only_types=[Uint32],
        read_only_args=[x],
    )


@cute.jit
def e4m3_value(code: Uint32) -> Float32:
    """The value of the E4M3 byte `code`, exactly."""
    return cute.arch.inline_ptx(
        "{ .reg .b16 c, lo, hi; .reg .b32 h; cvt.u16.u32 c, {$r0}; "
        "cvt.rn.f16x2.e4m3x2 h, c; mov.b32 {lo, hi}, h; cvt.f32.f16 {$w0}, lo; }",
        write_only_types=[Float32],
        read_only_args=[code],
    )


@cute.jit
def coefficient_of(code: Uint32, S_dec: Float32) -> Float32:
    """compute_nvfp4_scaling_coefficient<float> for scale byte `code`:
    min(1 / (scale * S_dec), FLT_MAX) with S_dec = 1 / S_enc computed once by the caller."""
    return cute.arch.fmin(rcp_rn_f32(e4m3_value(code) * S_dec), Float32(FLOAT32_MAX))


@cute.jit
def splat_f32x2(x: Float32) -> Int64:
    """x in both halves of a packed f32 pair, as mul.f32x2 consumes it."""
    return cute.arch.inline_ptx(
        "mov.b64 {$w0}, {{$r0}, {$r0}};", write_only_types=[Int64], read_only_args=[x]
    )


@cute.jit
def scale_bf16x2(v: Uint32, coeff2: Int64):
    """Both bf16 halves of v widened to f32 and multiplied by the packed coefficient, (lo, hi)."""
    return cute.arch.inline_ptx(
        "{ .reg .b32 lo, hi; .reg .b64 p, r; shl.b32 lo, {$r0}, 16; "
        "and.b32 hi, {$r0}, 0xffff0000; mov.b64 p, {lo, hi}; mul.f32x2 r, p, {$r1}; "
        "mov.b64 {{$w0}, {$w1}}, r; }",
        write_only_types=[Float32, Float32],
        read_only_args=[v, coeff2],
    )


@cute.jit
def scale_column_bf16(a: Uint32, b: Uint32, coeff2: Int64, LOW: cutlass.Constexpr[bool]):
    """Two rows of one column of a packed column pair, scaled: the low (LOW) or high halves of
    the row words a and b, as (row a, row b)."""
    if cutlass.const_expr(LOW):
        widen = "shl.b32 ea, {$r0}, 16; shl.b32 eb, {$r1}, 16; "
    else:
        widen = "and.b32 ea, {$r0}, 0xffff0000; and.b32 eb, {$r1}, 0xffff0000; "
    return cute.arch.inline_ptx(
        "{ .reg .b32 ea, eb; .reg .b64 p, r; "
        + widen
        + "mov.b64 p, {ea, eb}; mul.f32x2 r, p, {$r2}; mov.b64 {{$w0}, {$w1}}, r; }",
        write_only_types=[Float32, Float32],
        read_only_args=[a, b, coeff2],
    )


def abs_max_tree_bf16x2(words):
    """Packed abs-max over a power-of-two count of words, as a balanced tree: the magnitudes of
    both halves are the abs-maxes of the even and odd elements (signs are unreliable,
    max.xorsign.abs leaves the XOR of the operands' signs)."""
    # Imported here: utils.py of the package root is imported by every CuTeDSL module.
    from transformer_engine.common.CuTeDSL.utils import (  # pylint: disable=import-outside-toplevel
        abs_max_x2_bf16,
    )

    level = list(words)
    while len(level) > 1:
        level = [
            Uint32(abs_max_x2_bf16(Int32(level[2 * i]), Int32(level[2 * i + 1])))
            for i in range(len(level) // 2)
        ]
    return level[0]


@cute.jit
def block_scale_code(words, S_enc_div6: Float32) -> Uint32:
    """E4M3 scale byte of one rowwise block (eight packed words) for the encode scale
    S_enc_div6 = S_enc * (1 / 6)."""
    m = abs_max_tree_bf16x2(words)
    # Fold the two halves with the same unit: swap them and take one more abs-max.
    folded = abs_max_tree_bf16x2([m, swap_halves_b32(m)]) & Uint32(0x7FFF0000)
    return e4m3_code(folded.bitcast(Float32) * S_enc_div6)


@cute.jit
def pack_half_block(floats, r0: Uint32, r1: Uint32, STOCHASTIC: cutlass.Constexpr[bool]) -> Uint32:
    """Pack 8 scaled elements into one word of E2M1 nibbles, element 0 in the lowest; under
    stochastic rounding r0 and r1 are the random words of elements 0..3 and 4..7."""
    if cutlass.const_expr(STOCHASTIC):
        return cvt_f32x8_to_fp4x8_sr(*floats, r0, r1)
    return cvt_f32x8_to_fp4x8(*floats)


@cute.jit
def pack_row_block(words, coeff2: Int64, rbits, STOCHASTIC: cutlass.Constexpr[bool]):
    """Scale and pack one rowwise block (eight packed words) into two words; rbits is the block's
    draw under stochastic rounding. Half a block at a time keeps 8 floats live, not 16."""
    halves = []
    for half in cutlass.range_constexpr(2):
        floats = []
        for h in cutlass.range_constexpr(4 * half, 4 * half + 4):
            floats.extend(scale_bf16x2(words[h], coeff2))
        r0 = rbits[2 * half] if cutlass.const_expr(STOCHASTIC) else None
        r1 = rbits[2 * half + 1] if cutlass.const_expr(STOCHASTIC) else None
        halves.append(pack_half_block(floats, r0, r1, STOCHASTIC))
    return halves[0], halves[1]


@cute.jit
def pack_column_block(
    rows,
    coeff2: Int64,
    rbits,
    LOW: cutlass.Constexpr[bool],
    STOCHASTIC: cutlass.Constexpr[bool],
):
    """Columnwise twin of pack_row_block: rows[t] is row t of a packed column pair, and LOW
    selects its even column."""
    halves = []
    for half in cutlass.range_constexpr(2):
        floats = []
        for h in cutlass.range_constexpr(4 * half, 4 * half + 4):
            floats.extend(scale_column_bf16(rows[2 * h], rows[2 * h + 1], coeff2, LOW))
        r0 = rbits[2 * half] if cutlass.const_expr(STOCHASTIC) else None
        r1 = rbits[2 * half + 1] if cutlass.const_expr(STOCHASTIC) else None
        halves.append(pack_half_block(floats, r0, r1, STOCHASTIC))
    return halves[0], halves[1]
