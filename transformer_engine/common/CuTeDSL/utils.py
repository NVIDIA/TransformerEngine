# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Low-level CuTeDSL helpers: bitcast/fma/exp2 intrinsics, f32 packing, and the 16-bit (bf16/fp16) packed-op kit."""

import functools
import logging
import os
from types import SimpleNamespace
from typing import Optional

import cutlass
from cutlass import cute
from cutlass import Boolean, Float32, Int64, Int32, Int16, Uint32, Uint64
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

_CUTLASS_DTYPE_FROM_STR = {
    "fp32": cutlass.Float32,
    "fp16": cutlass.Float16,
    "bf16": cutlass.BFloat16,
}
_STR_FROM_CUTLASS_DTYPE = {v: k for k, v in _CUTLASS_DTYPE_FROM_STR.items()}

logger = logging.getLogger("transformer_engine.cutedsl.utils")

CUTEDSL_DEBUG_LOGGING = os.environ.get("CUTEDSL_DEBUG_LOGGING", "0") == "1"


@functools.lru_cache(maxsize=None)
def device_compute_capability() -> tuple:
    """(major, minor) of CUDA device 0, or (0, 0) if it can't be queried."""
    from cuda.core import Device  # pylint: disable=no-name-in-module

    major_minor = Device().arch  # compute capability as digits, e.g. "120"
    return (int(major_minor[:-1]), int(major_minor[-1])) if major_minor else (0, 0)


@functools.lru_cache(maxsize=None)
def device_is_blackwell() -> bool:
    """Return True for the Blackwell family (SM 10.x / 11.x / 12.x)
    This is a run-time check, not a compile-time check. It check if the current device is Blackwell architecture.
    The minor version is deliberately not pinned: SM 10.3 (GB300) is Blackwell as much as 10.0 is.
    """
    major, _ = device_compute_capability()
    return major in (10, 11, 12)


def str_to_cutlass_dtype(dtype_str: str):
    """Convert a string dtype to a cutlass dtype, or None if unknown."""
    return _CUTLASS_DTYPE_FROM_STR.get(dtype_str, None)


def cutlass_dtype_to_str(dtype):
    """Convert a cutlass dtype back to its protocol string, or None if unknown."""
    return _STR_FROM_CUTLASS_DTYPE.get(dtype, None)



# Runs if CUTE_DSL_ENABLE_ASSERTIONS=1 or --enable-assertions present in cute.compile
def validate_tensor(tensor: Optional[cute.Tensor], expected_layout: cute.Layout, expected_dtype):
    if tensor is None:
        return
    cute.testing.assert_(tensor.layout == expected_layout, "Tensor layout does not match")
    cute.testing.assert_(tensor.element_type == expected_dtype, "Tensor dtype does not match")


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


FP32_MANTISSA_BITS = 23
FLOAT32_MAX = 3.4028234663852886e38
BFLOAT16_MAX = 3.3895313892515355e38
FLOAT8E4M3_MAX = 448.0
FLOAT4E2M1_MAX = 6.0


@dsl_user_op
def _bitcast_f32_to_i32(val: Float32, *, loc=None, ip=None) -> Int32:
    """Bitcast a float32 value to int32 without changing the bit pattern."""
    return Int32(mlir_arith.bitcast(T.i32(), val.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def _bitcast_i32_to_f32(val: Int32, *, loc=None, ip=None) -> Float32:
    """Bitcast an int32 value to float32 without changing the bit pattern."""
    return Float32(mlir_arith.bitcast(T.f32(), val.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def fabs_f32(val: Float32, *, loc=None, ip=None) -> Float32:
    """Compute the absolute value of a float32."""
    val_i32 = _bitcast_f32_to_i32(val, loc=loc, ip=ip)
    abs_i32 = val_i32 & Int32(0x7FFFFFFF)
    return _bitcast_i32_to_f32(abs_i32, loc=loc, ip=ip)


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
        )
    )


@dsl_user_op
def select_f32(cond: Boolean, if_true: Float32, if_false: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        mlir_arith.select(
            cond.ir_value(loc=loc, ip=ip),
            if_true.ir_value(loc=loc, ip=ip),
            if_false.ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def exp2f_rcp(biased_exp: Int32, *, loc=None, ip=None) -> Float32:
    """2^(127 - biased_exp) with special-case handling."""
    new_exp = (Int32(254) - biased_exp) << Int32(FP32_MANTISSA_BITS)
    result = _bitcast_i32_to_f32(new_exp, loc=loc, ip=ip)
    for cmp_val, repl_bits in [(255, 0x7FFFFFFF), (254, 0x00400000), (0, 0x7F000000)]:
        cond = mlir_arith.cmpi(
            mlir_arith.CmpIPredicate.eq,
            biased_exp.ir_value(loc=loc, ip=ip),
            Int32(cmp_val).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
        alt = _bitcast_i32_to_f32(Int32(repl_bits), loc=loc, ip=ip)
        result = Float32(
            mlir_arith.select(
                cond, alt.ir_value(loc=loc, ip=ip), result.ir_value(loc=loc, ip=ip), loc=loc, ip=ip
            )
        )
    return result


@dsl_user_op
def umulhi_u32(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """High 32 bits of the unsigned 32x32 product (`__umulhi`)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
            "mul.hi.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def u64_lo32(v: Uint64, *, loc=None, ip=None) -> Uint32:
    return Uint32(mlir_arith.trunci(T.i32(), v.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def u64_hi32(v: Uint64, *, loc=None, ip=None) -> Uint32:
    shifted = mlir_arith.shrui(
        v.ir_value(loc=loc, ip=ip), Uint64(32).ir_value(loc=loc, ip=ip), loc=loc, ip=ip
    )
    return Uint32(mlir_arith.trunci(T.i32(), shifted, loc=loc, ip=ip))


@dsl_user_op
def bool_to_u64(b: Boolean, *, loc=None, ip=None) -> Uint64:
    return Uint64(mlir_arith.extui(T.i64(), b.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


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
        )
    )


@dsl_user_op
def pack_u32x2(lo: Uint32, hi: Uint32, *, loc=None, ip=None) -> Int64:
    """Pack two u32 into one 64-bit register (register-pair move, no real instruction)."""
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [lo.ir_value(loc=loc, ip=ip), hi.ir_value(loc=loc, ip=ip)],
            "mov.b64 $0, {$1, $2};",
            "=l,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def unpack_i64_to_i32x2(v: Int64, *, loc=None, ip=None):
    """Split a 64-bit value into (lo, hi) 32-bit halves.

    Inverse of pack_f32x2's register-pair layout. Lowers to register-pair
    aliasing in SASS (no real instructions), so an 8-byte smem load + this
    split costs one LDS.64 total."""
    lo = Int32(mlir_arith.trunci(T.i32(), v.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))
    hi_64 = mlir_arith.shrui(
        v.ir_value(loc=loc, ip=ip), Int64(32).ir_value(loc=loc, ip=ip), loc=loc, ip=ip
    )
    hi = Int32(mlir_arith.trunci(T.i32(), hi_64, loc=loc, ip=ip))
    return lo, hi


def make_prmt_u32(selector: int):
    """A byte-permute op with the 16-bit selector baked in as an immediate.

    prmt.b32 indexes the eight source bytes {a0..a3, b0..b7} and each selector nibble picks the
    byte for one destination position, low nibble first. 0x5410 interleaves the low halves of a
    and b (a0 a1 b0 b1), 0x7632 the high halves.
    """

    @dsl_user_op
    def prmt_u32(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
        return Uint32(
            llvm.inline_asm(
                T.i32(),
                [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
                f"prmt.b32 $0, $1, $2, {selector:#x};",
                "=r,r,r",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    return prmt_u32


def _build_packed16_kit(in_fmt: str):
    """Build a kit of PTX wrappers for a 16-bit input format so we don't have to repeat
    the same inline asm boilerplate code for FP16 and BF16 dtypes.

    `in_fmt` is the PTX format string ('bf16' or 'f16'). Returns a namespace
    with the per-format ops the rowwise/colwise inner loops need:

      abs_max_x2(Int32, Int32)  -> Int32   # `max.xorsign.abs.<fmt>x2`
      abs_max_scalar(Int16, Int16) -> Int16  # `max.xorsign.abs.<fmt>`
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
            )
        )

    if in_fmt == "bf16":
        # bf16 == top 16 bits of f32 — widening is a free bit-shift.
        @dsl_user_op
        def bits_to_f32(bits: Int16, *, loc=None, ip=None) -> Float32:
            i32 = Int32(mlir_arith.extui(T.i32(), bits.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))
            return _bitcast_i32_to_f32(i32 << Int32(16), loc=loc, ip=ip)

        @dsl_user_op
        def x2_lo_to_f32(bits: Int32, *, loc=None, ip=None) -> Float32:
            return _bitcast_i32_to_f32((bits & Int32(0xFFFF)) << Int32(16), loc=loc, ip=ip)

        @dsl_user_op
        def x2_hi_to_f32(bits: Int32, *, loc=None, ip=None) -> Float32:
            # `(x >> 16) << 16` ≡ `x & 0xFFFF0000`, sidestepping signed-literal
            # issues. Sign bits from the arith-right shift get zeroed by the
            # left shift.
            return _bitcast_i32_to_f32((bits >> Int32(16)) << Int32(16), loc=loc, ip=ip)

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
                )
            )
            i32 = Int32(
                mlir_arith.extui(T.i32(), bf16_bits.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
            )
            return _bitcast_i32_to_f32(i32 << Int32(16), loc=loc, ip=ip)

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
                )
            )

        @dsl_user_op
        def x2_lo_to_f32(bits: Int32, *, loc=None, ip=None) -> Float32:
            lo_i16 = Int16(
                mlir_arith.trunci(T.i16(), bits.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
            )
            return bits_to_f32(lo_i16, loc=loc, ip=ip)

        @dsl_user_op
        def x2_hi_to_f32(bits: Int32, *, loc=None, ip=None) -> Float32:
            hi_shifted = bits >> Int32(16)
            hi_i16 = Int16(
                mlir_arith.trunci(T.i16(), hi_shifted.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
            )
            return bits_to_f32(hi_i16, loc=loc, ip=ip)

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
                )
            )

    return SimpleNamespace(
        max_x2=max_x2,
        abs_max_x2=abs_max_x2,
        abs_max_scalar=abs_max_scalar,
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
