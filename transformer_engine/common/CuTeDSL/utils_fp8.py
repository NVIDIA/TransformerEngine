# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FP8 conversion helpers (f32<->fp8 e4m3/e5m2/e8m0, fused mul+cvt PTX wrappers) for the CuTeDSL kernels."""

import logging
import os
import re

import cutlass
from cutlass import Float32, Int64, Int32, Int16, Uint32
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from transformer_engine.common.CuTeDSL.utils import FP32_MANTISSA_BITS, _bitcast_f32_to_i32

logger = logging.getLogger("transformer_engine.cutedsl.utils_fp8")


@dsl_user_op
def cvt_f32_to_fp8e4m3(val: Float32, *, loc=None, ip=None) -> Int32:
    """float32 -> fp8e4m3 conversion."""
    zero = Float32(0.0)
    result_i16 = Int16(
        llvm.inline_asm(
            T.i16(),
            [zero.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
            "cvt.rn.satfinite.e4m3x2.f32 $0, $1, $2;",
            "=h,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
    result_i32 = Int32(
        mlir_arith.extui(T.i32(), result_i16.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    )
    return result_i32 & Int32(0xFF)


@dsl_user_op
def cvt_f32_to_fp8e5m2(val: Float32, *, loc=None, ip=None) -> Int32:
    """float32 -> fp8e5m2 conversion."""
    zero = Float32(0.0)
    result_i16 = Int16(
        llvm.inline_asm(
            T.i16(),
            [zero.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
            "cvt.rn.satfinite.e5m2x2.f32 $0, $1, $2;",
            "=h,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
    result_i32 = Int32(
        mlir_arith.extui(T.i32(), result_i16.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    )
    return result_i32 & Int32(0xFF)


@dsl_user_op
def cvt_f32_to_fp8e8m0_non_blackwell(val: Float32, *, loc=None, ip=None) -> Int32:
    """float32 -> fp8e8m0 conversion (generic, pre-Blackwell).

    Software round-up of the biased exponent, mirroring ptx::float_to_e8m0's
    non-Blackwell branch (transformer_engine/common/util/ptx.cuh)."""
    val_i32 = _bitcast_f32_to_i32(val, loc=loc, ip=ip)
    rounded = val_i32 + Int32(0x7FFFFF)
    exponent = (rounded >> Int32(FP32_MANTISSA_BITS)) & Int32(0xFF)
    return Int32(
        mlir_arith.minsi(
            exponent.ir_value(loc=loc, ip=ip), Int32(254).ir_value(loc=loc, ip=ip), loc=loc, ip=ip
        )
    )


@dsl_user_op
def cvt_f32_to_fp8e8m0_blackwell(val: Float32, *, loc=None, ip=None) -> Int32:
    """float32 -> fp8e8m0 conversion (Blackwell, SM >= 100).

    Uses the hardware cvt.rp.satfinite.ue8m0x2.f32 instruction, mirroring
    ptx::float_to_e8m0's Blackwell branch. The x2 form packs two e8m0 bytes;
    we feed (0.0, val) so the low byte is e8m0(val) and mask it out."""
    zero = Float32(0.0)
    result_i16 = Int16(
        llvm.inline_asm(
            T.i16(),
            [zero.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
            "cvt.rp.satfinite.ue8m0x2.f32 $0, $1, $2;",
            "=h,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
    result_i32 = Int32(
        mlir_arith.extui(T.i32(), result_i16.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    )
    return result_i32 & Int32(0xFF)


def _build_mul_i64_cvt_f32x4_to_fp8x4(out_fmt: str, relu: bool = False):
    """Build a fused 4-wide `f32x4 * f32x2 -> fp8x4` PTX wrapper.

    Multiplies four f32 inputs by a broadcast inverse scale (passed as an
    f32x2 pack of (s, s)) and converts to FP8, packing the four bytes into one
    uint32: byte i = fp8(v_i * s). Two `mul.f32x2` + two `cvt...x2.f32` — the
    f32-input form of this op family (CUDA ptx::mul_cvt_4x).
    """
    out_op = "e4m3x2" if out_fmt == "e4m3" else "e5m2x2"
    asm = (
        "{\n"
        ".reg.b64 vp0; .reg.b64 vp1; .reg.b64 vp2; .reg.b64 vp3;\n\t"
        ".reg.b32 vs0; .reg.b32 vs1; .reg.b32 vs2; .reg.b32 vs3;\n\t"
        ".reg.b16 vo0; .reg.b16 vo1;\n\t"
        "mov.b64 vp0, {$1, $2};\n\t"
        "mov.b64 vp2, {$3, $4};\n\t"
        "mul.f32x2 vp1, vp0, $5;\n\t"
        "mul.f32x2 vp3, vp2, $5;\n\t"
        "mov.b64 {vs0, vs1}, vp1;\n\t"
        "mov.b64 {vs2, vs3}, vp3;\n\t"
        # cvt d, a, b => d[15:8]=fp8(a), d[7:0]=fp8(b); feed (hi, lo) so the low
        # byte holds the earlier element.
        f"cvt.rn.satfinite{".relu" if relu else ""}.{out_op}.f32 vo0, vs1, vs0;\n\t"
        f"cvt.rn.satfinite{".relu" if relu else ""}.{out_op}.f32 vo1, vs3, vs2;\n\t"
        "mov.b32 $0, {vo0, vo1};\n\t"
        "}"
    )

    @dsl_user_op
    def fn(
        v0: Float32, v1: Float32, v2: Float32, v3: Float32, scale_2x: Int64, *, loc=None, ip=None
    ) -> Uint32:
        return Uint32(
            llvm.inline_asm(
                T.i32(),
                [
                    v0.ir_value(loc=loc, ip=ip),
                    v1.ir_value(loc=loc, ip=ip),
                    v2.ir_value(loc=loc, ip=ip),
                    v3.ir_value(loc=loc, ip=ip),
                    scale_2x.ir_value(loc=loc, ip=ip),
                ],
                asm,
                "=r,f,f,f,f,l",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    return fn


def mul_i64_cvt_f32x4_to_fp8x4(fp8_dtype: str, relu: bool = False):
    """Return the fused 4-wide f32->FP8 multiply+cast op for the given FP8 format.

    The op takes (v0, v1, v2, v3, scale_2x) and returns a uint32 of four packed
    fp8 bytes, byte i = fp8(v_i * scale). `scale_2x` is pack_f32x2(s, s)."""
    return _build_mul_i64_cvt_f32x4_to_fp8x4("e5m2" if fp8_dtype == "e5m2" else "e4m3", relu)


def _build_mul_f32x4_cvt_f32x4_to_fp8x4(out_fmt: str, relu: bool = False):
    """Build a fused elementwise `f32x4 * f32x4 -> fp8x4` PTX wrapper.

    General elementwise multiply-and-convert: byte i = fp8(a_i * b_i). Two
    fma.rn.f32x2 against packed zeros + two cvt...x2.f32 (same sequence as
    CUDA's mul_cvt_4x(out, floatx4, floatx4) in ptx.cuh). fma (not mul) so a
    -0 product flushes to +0."""
    out_op = "e4m3x2" if out_fmt == "e4m3" else "e5m2x2"
    asm = (
        "{\n"
        ".reg.b64 va0; .reg.b64 va1; .reg.b64 vb0; .reg.b64 vb1;\n\t"
        ".reg.b64 vr0; .reg.b64 vr1; .reg.b64 zeros;\n\t"
        ".reg.b32 vs0; .reg.b32 vs1; .reg.b32 vs2; .reg.b32 vs3;\n\t"
        ".reg.b16 vo0; .reg.b16 vo1;\n\t"
        "mov.b64 zeros, {0x0, 0x0};\n\t"
        "mov.b64 va0, {$1, $2};\n\t"
        "mov.b64 va1, {$3, $4};\n\t"
        "mov.b64 vb0, {$5, $6};\n\t"
        "mov.b64 vb1, {$7, $8};\n\t"
        "fma.rn.f32x2 vr0, va0, vb0, zeros;\n\t"
        "fma.rn.f32x2 vr1, va1, vb1, zeros;\n\t"
        "mov.b64 {vs0, vs1}, vr0;\n\t"
        "mov.b64 {vs2, vs3}, vr1;\n\t"
        # cvt d, a, b => d[15:8]=fp8(a), d[7:0]=fp8(b); feed (hi, lo) so the low
        # byte holds the earlier element.
        f"cvt.rn.satfinite{".relu" if relu else ""}.{out_op}.f32 vo0, vs1, vs0;\n\t"
        f"cvt.rn.satfinite{".relu" if relu else ""}.{out_op}.f32 vo1, vs3, vs2;\n\t"
        "mov.b32 $0, {vo0, vo1};\n\t"
        "}"
    )

    @dsl_user_op
    def fn(
        a0: Float32,
        a1: Float32,
        a2: Float32,
        a3: Float32,
        b0: Float32,
        b1: Float32,
        b2: Float32,
        b3: Float32,
        *,
        loc=None,
        ip=None,
    ) -> Uint32:
        return Uint32(
            llvm.inline_asm(
                T.i32(),
                [
                    a0.ir_value(loc=loc, ip=ip),
                    a1.ir_value(loc=loc, ip=ip),
                    a2.ir_value(loc=loc, ip=ip),
                    a3.ir_value(loc=loc, ip=ip),
                    b0.ir_value(loc=loc, ip=ip),
                    b1.ir_value(loc=loc, ip=ip),
                    b2.ir_value(loc=loc, ip=ip),
                    b3.ir_value(loc=loc, ip=ip),
                ],
                asm,
                "=r,f,f,f,f,f,f,f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    return fn


def mul_f32x4_cvt_f32x4_to_fp8x4(fp8_dtype: str, relu: bool = False):
    """Return the fused elementwise f32x4*f32x4 -> FP8x4 op for the given FP8 format.

    The op takes (a0..a3, b0..b3) and returns a uint32 of four packed fp8
    bytes, byte i = fp8(a_i * b_i)."""
    return _build_mul_f32x4_cvt_f32x4_to_fp8x4("e5m2" if fp8_dtype == "e5m2" else "e4m3", relu)


def _build_mul_i64_cvt_packed16x4_to_fp8x4(in_fmt: str, out_fmt: str, relu: bool = False):
    """Build a fused `2x <in_fmt>x2 * f32x2 -> fp8x4` PTX wrapper.

    16-bit-input form of _build_mul_i64_cvt_f32x4_to_fp8x4: widens four packed
    bf16/f16 elements to f32 inside the asm, multiplies by the broadcast
    (s, s) pair, and converts: byte i = fp8(elt_i * s). The two u16 cvt
    results combine into the u32 via a register-pair mov (free)."""
    out_op = "e4m3x2" if out_fmt == "e4m3" else "e5m2x2"
    asm = (
        "{\n"
        ".reg.b64 vp0; .reg.b64 vp1; .reg.b64 vq0; .reg.b64 vq1;\n\t"
        ".reg.b32 v1; .reg.b32 v2; .reg.b32 v3; .reg.b32 v4;\n\t"
        ".reg.b16 vb1; .reg.b16 vb2; .reg.b16 vb3; .reg.b16 vb4;\n\t"
        ".reg.b16 vo0; .reg.b16 vo1;\n\t"
        "mov.b32 {vb1, vb2}, $1;\n\t"
        "mov.b32 {vb3, vb4}, $2;\n\t"
        f"cvt.f32.{in_fmt} v1, vb1;\n\t"
        f"cvt.f32.{in_fmt} v2, vb2;\n\t"
        f"cvt.f32.{in_fmt} v3, vb3;\n\t"
        f"cvt.f32.{in_fmt} v4, vb4;\n\t"
        "mov.b64 vp0, {v1, v2};\n\t"
        "mov.b64 vq0, {v3, v4};\n\t"
        "mul.f32x2 vp1, vp0, $3;\n\t"
        "mul.f32x2 vq1, vq0, $3;\n\t"
        "mov.b64 {v2, v1}, vp1;\n\t"
        "mov.b64 {v4, v3}, vq1;\n\t"
        # cvt d, a, b => d[15:8]=fp8(a), d[7:0]=fp8(b); feed (hi, lo) so the
        # low byte holds the earlier element.
        f"cvt.rn.satfinite{".relu" if relu else ""}.{out_op}.f32 vo0, v1, v2;\n\t"
        f"cvt.rn.satfinite{".relu" if relu else ""}.{out_op}.f32 vo1, v3, v4;\n\t"
        "mov.b32 $0, {vo0, vo1};\n\t"
        "}"
    )

    @dsl_user_op
    def fn(lo_2x: Int32, hi_2x: Int32, scale_2x: Int64, *, loc=None, ip=None) -> Uint32:
        return Uint32(
            llvm.inline_asm(
                T.i32(),
                [
                    lo_2x.ir_value(loc=loc, ip=ip),
                    hi_2x.ir_value(loc=loc, ip=ip),
                    scale_2x.ir_value(loc=loc, ip=ip),
                ],
                asm,
                "=r,r,r,l",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    return fn


def mul_i64_cvt_packed16x4_to_fp8x4(dtype, fp8_dtype: str, relu: bool = False):
    """Return the fused packed16x4 * broadcast-scale -> FP8x4 op.

    The op takes (lo_2x, hi_2x, scale_2x): two i32s of packed bf16/f16 pairs
    (elements 0-1, 2-3) and a pack_f32x2(s, s) pair; returns a uint32 of four
    packed fp8 bytes, byte i = fp8(elt_i * s)."""
    in_fmt = "f16" if dtype is cutlass.Float16 else "bf16"
    return _build_mul_i64_cvt_packed16x4_to_fp8x4(
        in_fmt, "e5m2" if fp8_dtype == "e5m2" else "e4m3", relu
    )


def _target_arch_is_blackwell() -> bool:
    """Return True for the Blackwell family (SM 10.0 / 11.0 / 12.0), which has the
    cvt.*.ue8m0x2.f32 hardware instruction. This mirrors the CUDA reference's
    ARCH_BLACKWELL_FAMILY gate (FamilySpecific<100/110/120> in
    transformer_engine/common/util/ptx.cuh) -- a family check, since the
    instruction is available across the family (verified on sm_120a) even though
    e.g. tcgen05 is not.

    The gate is the *compile target*, not the physical device, since that is what
    decides whether the instruction codegens: CUTE_DSL_ARCH if set (what
    cute.compile uses), else the current device's compute capability. Falls back
    to the non-Blackwell software path if the arch can't be determined."""
    try:
        arch = os.getenv("CUTE_DSL_ARCH")  # e.g. "sm_120a", the explicit compile target
        if arch:
            major_minor = re.search(r"(\d+)", arch).group(1)  # "120"
        else:
            from cuda.core import Device  # pylint: disable=no-name-in-module

            major_minor = Device().arch  # compute capability as digits, e.g. "120"
        # Trailing digit is the minor version; the rest is the major version.
        return int(major_minor[:-1]) in (10, 11, 12)
    # Best-effort detection: any failure means "assume non-Blackwell software path".
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.debug("e8m0 arch detection failed (%s); using software path", e)
        return False


# Pick the appropriate float32 -> fp8e8m0 conversion function based on the target architecture.
# Blackwell (SM >= 100) has a hardware instruction for this, while older architectures require a software implementation.
cvt_f32_to_fp8e8m0 = (
    cvt_f32_to_fp8e8m0_blackwell
    if _target_arch_is_blackwell()
    else cvt_f32_to_fp8e8m0_non_blackwell
)


@dsl_user_op
def cvt_f32x2_to_fp8e4m3x2(
    val_hi: Float32, val_lo: Float32, relu: bool = False, *, loc=None, ip=None
) -> Int32:
    """Convert two float32 values to two packed fp8e4m3fn bytes in one instruction.

    Returns an int32 where bits [7:0] = fp8(val_lo), bits [15:8] = fp8(val_hi).
    """
    result_i16 = Int16(
        llvm.inline_asm(
            T.i16(),
            [val_hi.ir_value(loc=loc, ip=ip), val_lo.ir_value(loc=loc, ip=ip)],
            f"cvt.rn.satfinite{".relu" if relu else ""}.e4m3x2.f32 $0, $1, $2;",
            "=h,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
    return Int32(mlir_arith.extui(T.i32(), result_i16.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def cvt_f32x2_to_fp8e5m2x2(
    val_hi: Float32, val_lo: Float32, relu: bool = False, *, loc=None, ip=None
) -> Int32:
    """Convert two float32 values to two packed fp8e5m2 bytes in one instruction.

    Returns an int32 where bits [7:0] = fp8(val_lo), bits [15:8] = fp8(val_hi).
    """
    result_i16 = Int16(
        llvm.inline_asm(
            T.i16(),
            [val_hi.ir_value(loc=loc, ip=ip), val_lo.ir_value(loc=loc, ip=ip)],
            f"cvt.rn.satfinite{".relu" if relu else ""}.e5m2x2.f32 $0, $1, $2;",
            "=h,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
    return Int32(mlir_arith.extui(T.i32(), result_i16.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


def get_cvt_f32_to_fp8_func(fp8_dtype: str):
    """Returns the float32 -> float8 conversion function for the given FP8 format."""
    if fp8_dtype == "e5m2":
        return cvt_f32_to_fp8e5m2
    return cvt_f32_to_fp8e4m3


def get_cvt_f32x2_to_fp8x2_func(fp8_dtype: str):
    """Returns the float32x2 -> float8x2 conversion function for the given FP8 format."""
    if fp8_dtype == "e5m2":
        return cvt_f32x2_to_fp8e5m2x2
    return cvt_f32x2_to_fp8e4m3x2
