# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Low-level CuTeDSL helpers: bitcast/fma/exp2 intrinsics, f32 packing, and the 16-bit (bf16/fp16) packed-op kit."""

import logging
from types import SimpleNamespace
from typing import Optional

import cutlass
from cutlass import cute
from cutlass import Float32, Int64, Int32, Int16
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

_CUTLASS_DTYPE_FROM_STR = {
    "Float32": cutlass.Float32,
    "Float16": cutlass.Float16,
    "BFloat16": cutlass.BFloat16,
    "Float8E4M3": cutlass.Float8E4M3FN,
    "Float8E5M2": cutlass.Float8E5M2,
    "Float8E8M0": cutlass.Float8E8M0FNU,
    "Float4E2M1": cutlass.Float4E2M1FN,
}
_STR_FROM_CUTLASS_DTYPE = {v: k for k, v in _CUTLASS_DTYPE_FROM_STR.items()}

logger = logging.getLogger("transformer_engine.cutedsl.utils")


def device_compute_capability(device_id: Optional[int] = None) -> tuple:
    """(major, minor) compute capability of a CUDA device (current by default), or (0, 0) if it can't be queried."""
    from cuda.core import Device  # pylint: disable=no-name-in-module

    try:
        device_id = Device().device_id if device_id is None else device_id
        major_minor = Device(device_id).arch
        return (int(major_minor[:-1]), int(major_minor[-1])) if major_minor else (0, 0)
    except Exception as e:  # pylint: disable=broad-except
        logger.warning("Could not query CUDA device compute capability (%s); assuming (0, 0).", e)
        return (0, 0)


def device_is_blackwell(device_id: Optional[int] = None) -> bool:
    """Return True if the device (current device by default) is Blackwell family
    (SM 10.0 / 11.0 / 12.0). Run-time check, not compile-time."""
    major, minor = device_compute_capability(device_id)
    return (
        (major == 10 and minor == 0) or (major == 11 and minor == 0) or (major == 12 and minor == 0)
    )


def str_to_cutlass_dtype(dtype_str: str):
    """Convert a string dtype to a cutlass dtype, or None if unknown."""
    return _CUTLASS_DTYPE_FROM_STR.get(dtype_str, None)


def cutlass_dtype_to_str(dtype):
    """Convert a cutlass dtype back to its protocol string, or None if unknown."""
    return _STR_FROM_CUTLASS_DTYPE.get(dtype, None)


FP32_MANTISSA_BITS = 23


@cute.jit
def fabs_f32(val: Float32) -> Float32:
    """Compute the absolute value of a float32."""
    return (val.bitcast(Int32) & Int32(0x7FFFFFFF)).bitcast(Float32)


@dsl_user_op
def fma_f32(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    """Compute the fused multiply-add of three float32 values: a * b + c."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip), c.ir_value(loc=loc, ip=ip)],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def exp2f_rcp(scale_e8m0) -> Float32:
    """2^(127 - biased_exp) with special-case handling, for an e8m0 scale."""
    biased_exp = Int32(scale_e8m0.bitcast(cutlass.Uint8))
    new_exp = (Int32(254) - biased_exp) << Int32(FP32_MANTISSA_BITS)
    result = new_exp.bitcast(Float32)
    # CuTeDSL unrolls this fixed loop and emits predicate instructions for the
    # scalar conditionals, so this control flow does not hurt performance.
    for cmp_val, repl_bits in [(255, 0x7FFFFFFF), (254, 0x00400000), (0, 0x7F000000)]:
        if biased_exp == Int32(cmp_val):
            result = Int32(repl_bits).bitcast(Float32)
    return result


@dsl_user_op
def pack_f32x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Int64:
    """Pack two f32 scalars into a single 64-bit register (`floatx2` layout).

    Low 32 bits = `lo`, high 32 bits = `hi`. Uses `mov.b64 %dst, {%lo, %hi};`
    which lowers to a single register move — no actual memory traffic.
    """
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [lo.ir_value(loc=loc, ip=ip), hi.ir_value(loc=loc, ip=ip)],
            "mov.b64 $0, {$1, $2};",
            "=l,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def unpack_i64_to_i32x2(v: Int64):
    """Split a 64-bit value into (lo, hi) 32-bit halves.

    Inverse of pack_f32x2's register-pair layout. Lowers to register-pair
    aliasing in SASS (no real instructions), so an 8-byte smem load + this
    split costs one LDS.64 total."""
    lo = Int32(v)
    hi = Int32(v >> Int64(32))
    return lo, hi


def _build_packed16_kit(in_fmt: str):
    """Build a kit of PTX wrappers for a 16-bit input format so we don't have to repeat
    the same inline asm boilerplate code for FP16 and BF16 dtypes.

    `in_fmt` is the PTX format string ('bf16' or 'f16'). Returns a namespace
    with the per-format ops the rowwise/colwise inner loops need:

      abs_max_x2(Int32, Int32)  -> Int32   # `max.xorsign.abs.<fmt>x2`
      max_x2(Int32, Int32)      -> Int32   # `max.<fmt>x2`
      abs_max_scalar(Int16, Int16) -> Int16  # `max.xorsign.abs.<fmt>`
      max_scalar(Int16, Int16)  -> Int16   # `max.<fmt>`
      bits_to_f32(Int16) -> Float32          # widen one 16-bit element
      x2_lo_to_f32(Int32) -> Float32         # extract+widen low half
      x2_hi_to_f32(Int32) -> Float32         # extract+widen high half
    """

    @dsl_user_op
    def abs_max_x2(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
        return Int32(
            llvm.inline_asm(
                T.i32(),
                [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
                f"max.xorsign.abs.{in_fmt}x2 $0, $1, $2;",
                "=r,r,r",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def max_x2(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
        return Int32(
            llvm.inline_asm(
                T.i32(),
                [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
                f"max.{in_fmt}x2 $0, $1, $2;",
                "=r,r,r",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def max_scalar(a: Int16, b: Int16, *, loc=None, ip=None) -> Int16:
        return Int16(
            llvm.inline_asm(
                T.i16(),
                [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
                f"max.{in_fmt} $0, $1, $2;",
                "=h,h,h",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def abs_max_scalar(a: Int16, b: Int16, *, loc=None, ip=None) -> Int16:
        return Int16(
            llvm.inline_asm(
                T.i16(),
                [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
                f"max.xorsign.abs.{in_fmt} $0, $1, $2;",
                "=h,h,h",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
        )

    if in_fmt == "bf16":
        # bf16 == top 16 bits of f32 — widening is a free bit-shift.
        @cute.jit
        def bits_to_f32(bits: Int16) -> Float32:
            i32 = cutlass.Uint32(bits)
            return (i32 << cutlass.Uint32(16)).bitcast(Float32)

        @cute.jit
        def x2_lo_to_f32(bits: Int32) -> Float32:
            return ((bits & Int32(0xFFFF)) << Int32(16)).bitcast(Float32)

        @cute.jit
        def x2_hi_to_f32(bits: Int32) -> Float32:
            # `(x >> 16) << 16` ≡ `x & 0xFFFF0000`, sidestepping signed-literal
            # issues. Sign bits from the arith-right shift get zeroed by the
            # left shift.
            return ((bits >> Int32(16)) << Int32(16)).bitcast(Float32)

        @dsl_user_op
        def truncate_f32(val: Float32, *, loc=None, ip=None) -> Float32:
            """Round f32 to bf16 precision (round-to-nearest-even), keep f32.
            Matches C++'s `static_cast<float>(static_cast<bf16>(elt))`."""
            bf16_bits = Int16(
                llvm.inline_asm(
                    T.i16(),
                    [val.ir_value(loc=loc, ip=ip)],
                    "cvt.rn.bf16.f32 $0, $1;",
                    "=h,f",
                    has_side_effects=False,
                    is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                    loc=loc,
                    ip=ip,
                )
            )
            i32 = Int32(
                mlir_arith.extui(T.i32(), bf16_bits.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
            )
            return (i32 << Int32(16)).bitcast(Float32)

    else:
        # f16 has its own bit layout; widening requires `cvt.f32.f16`.
        @dsl_user_op
        def bits_to_f32(bits: Int16, *, loc=None, ip=None) -> Float32:
            return Float32(
                llvm.inline_asm(
                    T.f32(),
                    [bits.ir_value(loc=loc, ip=ip)],
                    "cvt.f32.f16 $0, $1;",
                    "=f,h",
                    has_side_effects=False,
                    is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                    loc=loc,
                    ip=ip,
                )
            )

        @cute.jit
        def x2_lo_to_f32(bits: Int32) -> Float32:
            return bits_to_f32(Int16(bits))

        @cute.jit
        def x2_hi_to_f32(bits: Int32) -> Float32:
            hi_shifted = bits >> Int32(16)
            return bits_to_f32(Int16(hi_shifted))

        @dsl_user_op
        def truncate_f32(val: Float32, *, loc=None, ip=None) -> Float32:
            """Round f32 to f16 precision, keep f32."""
            f16_bits = Int16(
                llvm.inline_asm(
                    T.i16(),
                    [val.ir_value(loc=loc, ip=ip)],
                    "cvt.rn.f16.f32 $0, $1;",
                    "=h,f",
                    has_side_effects=False,
                    is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                    loc=loc,
                    ip=ip,
                )
            )
            return Float32(
                llvm.inline_asm(
                    T.f32(),
                    [f16_bits.ir_value(loc=loc, ip=ip)],
                    "cvt.f32.f16 $0, $1;",
                    "=f,h",
                    has_side_effects=False,
                    is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                    loc=loc,
                    ip=ip,
                )
            )

    return SimpleNamespace(
        max_x2=max_x2,
        abs_max_x2=abs_max_x2,
        abs_max_scalar=abs_max_scalar,
        max_scalar=max_scalar,
        bits_to_f32=bits_to_f32,
        x2_lo_to_f32=x2_lo_to_f32,
        x2_hi_to_f32=x2_hi_to_f32,
        truncate_f32=truncate_f32,
    )


_BF16_KIT = _build_packed16_kit("bf16")
_F16_KIT = _build_packed16_kit("f16")


def is_packed16(dtype) -> bool:
    """True if `dtype` is one of the 16-bit packed input formats."""
    return dtype is cutlass.BFloat16 or dtype is cutlass.Float16


def packed16_kit(dtype):
    """Trace-time selector — pick a Packed16Kit for the input dtype."""
    if dtype is cutlass.Float16:
        return _F16_KIT
    return _BF16_KIT
