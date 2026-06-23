import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int64, Int32, Int16, Uint8, Uint32
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from transformer_engine.common.CuTeDSL.utils import FP32_MANTISSA_BITS, _bitcast_f32_to_i32

@dsl_user_op
def cvt_f32_to_fp8e4m3(val: Float32, *, loc=None, ip=None) -> Int32:
    """float32 -> fp8e4m3 conversion."""
    zero = Float32(0.0)
    result_i16 = Int16(llvm.inline_asm(
        T.i16(),
        [zero.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
        "cvt.rn.satfinite.e4m3x2.f32 $0, $1, $2;",
        "=h,f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))
    result_i32 = Int32(mlir_arith.extui(
        T.i32(), result_i16.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))
    return result_i32 & Int32(0xFF)


@dsl_user_op
def cvt_f32_to_fp8e5m2(val: Float32, *, loc=None, ip=None) -> Int32:
    """float32 -> fp8e5m2 conversion."""
    zero = Float32(0.0)
    result_i16 = Int16(llvm.inline_asm(
        T.i16(),
        [zero.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
        "cvt.rn.satfinite.e5m2x2.f32 $0, $1, $2;",
        "=h,f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))
    result_i32 = Int32(mlir_arith.extui(
        T.i32(), result_i16.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))
    return result_i32 & Int32(0xFF)


@dsl_user_op
def cvt_f32_to_fp8e8m0(val: Float32, *, loc=None, ip=None) -> Int32:
    """float32 -> fp8e8m0 conversion."""
    val_i32 = _bitcast_f32_to_i32(val, loc=loc, ip=ip)
    rounded = val_i32 + Int32(0x7FFFFF)
    exponent = (rounded >> Int32(FP32_MANTISSA_BITS)) & Int32(0xFF)
    return Int32(mlir_arith.minsi(
        exponent.ir_value(loc=loc, ip=ip),
        Int32(254).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def cvt_f32x2_to_fp8e4m3x2(val_hi: Float32, val_lo: Float32, relu: bool = False,
                             *, loc=None, ip=None) -> Int32:
    """Convert two float32 values to two packed fp8e4m3fn bytes in one instruction.

    Returns an int32 where bits [7:0] = fp8(val_lo), bits [15:8] = fp8(val_hi).
    """
    result_i16 = Int16(llvm.inline_asm(
        T.i16(),
        [val_hi.ir_value(loc=loc, ip=ip), val_lo.ir_value(loc=loc, ip=ip)],
        f"cvt.rn.satfinite{".relu" if relu else ""}.e4m3x2.f32 $0, $1, $2;",
        "=h,f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))
    return Int32(mlir_arith.extui(
        T.i32(), result_i16.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def cvt_f32x2_to_fp8e5m2x2(val_hi: Float32, val_lo: Float32, relu: bool = False,
                             *, loc=None, ip=None) -> Int32:
    """Convert two float32 values to two packed fp8e5m2 bytes in one instruction.

    Returns an int32 where bits [7:0] = fp8(val_lo), bits [15:8] = fp8(val_hi).
    """
    result_i16 = Int16(llvm.inline_asm(
        T.i16(),
        [val_hi.ir_value(loc=loc, ip=ip), val_lo.ir_value(loc=loc, ip=ip)],
        f"cvt.rn.satfinite{".relu" if relu else ""}.e5m2x2.f32 $0, $1, $2;",
        "=h,f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))
    return Int32(mlir_arith.extui(
        T.i32(), result_i16.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


def cvt_f32_to_fp8(fp8_dtype: str):
    """Returns the float32 -> float8 conversion function for the given FP8 format."""
    if fp8_dtype == "e5m2":
        return cvt_f32_to_fp8e5m2
    return cvt_f32_to_fp8e4m3


def cvt_f32x2_to_fp8x2(fp8_dtype: str):
    """Returns the float32x2 -> float8x2 conversion function for the given FP8 format."""
    if fp8_dtype == "e5m2":
        return cvt_f32x2_to_fp8e5m2x2
    return cvt_f32x2_to_fp8e4m3x2

