import logging
import os
import re

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int64, Int32, Int16, Uint8, Uint32
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
            from cuda.core import Device

            major_minor = Device().arch  # compute capability as digits, e.g. "120"
        # Trailing digit is the minor version; the rest is the major version.
        return int(major_minor[:-1]) in (10, 11, 12)
    except Exception as e:  # pragma: no cover - detection is best-effort
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
