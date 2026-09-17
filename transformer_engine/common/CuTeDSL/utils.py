# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Low-level CuTeDSL helpers for bitcasts, math intrinsics, and packed FP16/BF16 ops."""

import logging
from typing import Optional

import cutlass
from cutlass import cute
from cutlass import Float32, Int64, Int32, Uint16, Uint32

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


@cute.jit
def fma_f32(a: Float32, b: Float32, c: Float32) -> Float32:
    """Compute the fused multiply-add of three float32 values: a * b + c."""
    return cute.arch.inline_ptx(
        "fma.rn.f32 {$w0}, {$r0}, {$r1}, {$r2};",
        write_only_types=[Float32],
        read_only_args=[a, b, c],
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


@cute.jit
def pack_f32x2(lo: Float32, hi: Float32) -> Int64:
    """Pack two f32 scalars into a single 64-bit register (`floatx2` layout).

    Low 32 bits = `lo`, high 32 bits = `hi`. Uses `mov.b64 %dst, {%lo, %hi};`
    which lowers to a single register move — no actual memory traffic.
    """
    return cute.arch.inline_ptx(
        "mov.b64 {$w0}, {{$r0}, {$r1}};",
        write_only_types=[Int64],
        read_only_args=[lo, hi],
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


def _packed16_format(dtype) -> str:
    if dtype is cutlass.BFloat16:
        return "bf16"
    if dtype is cutlass.Float16:
        return "f16"
    raise ValueError(f"Expected BFloat16 or Float16, got {dtype}")


def abs_max_x2(dtype):
    """Return a packed x2 absolute-max operation for a 16-bit float dtype."""
    in_fmt = _packed16_format(dtype)

    @cute.jit
    def impl(a: Int32, b: Int32) -> Int32:
        return cute.arch.inline_ptx(
            f"max.xorsign.abs.{in_fmt}x2 {{$w0}}, {{$r0}}, {{$r1}};",
            write_only_types=[Int32],
            read_only_args=[a, b],
        )

    return impl


def max_x2(dtype):
    """Return a packed x2 max operation for a 16-bit float dtype."""
    in_fmt = _packed16_format(dtype)

    @cute.jit
    def impl(a: Int32, b: Int32) -> Int32:
        return cute.arch.inline_ptx(
            f"max.{in_fmt}x2 {{$w0}}, {{$r0}}, {{$r1}};",
            write_only_types=[Int32],
            read_only_args=[a, b],
        )

    return impl


def max_scalar(dtype):
    """Return a scalar max operation for a 16-bit float dtype."""
    in_fmt = _packed16_format(dtype)

    @cute.jit
    def impl(a: dtype, b: dtype) -> dtype:
        result = cute.arch.inline_ptx(
            f"max.{in_fmt} {{$w0}}, {{$r0}}, {{$r1}};",
            write_only_types=[Uint16],
            read_only_args=[a.bitcast(Uint16), b.bitcast(Uint16)],
        )
        return result.bitcast(dtype)

    return impl


def abs_max_scalar(dtype):
    """Return a scalar absolute-max operation for a 16-bit float dtype."""
    in_fmt = _packed16_format(dtype)

    @cute.jit
    def impl(a: dtype, b: dtype) -> dtype:
        result = cute.arch.inline_ptx(
            f"max.xorsign.abs.{in_fmt} {{$w0}}, {{$r0}}, {{$r1}};",
            write_only_types=[Uint16],
            read_only_args=[a.bitcast(Uint16), b.bitcast(Uint16)],
        )
        return result.bitcast(dtype)

    return impl


abs_max_x2_bf16 = abs_max_x2(cutlass.BFloat16)
abs_max_x2_f16 = abs_max_x2(cutlass.Float16)
max_x2_bf16 = max_x2(cutlass.BFloat16)
max_x2_f16 = max_x2(cutlass.Float16)
max_scalar_bf16 = max_scalar(cutlass.BFloat16)
max_scalar_f16 = max_scalar(cutlass.Float16)
abs_max_scalar_bf16 = abs_max_scalar(cutlass.BFloat16)
abs_max_scalar_f16 = abs_max_scalar(cutlass.Float16)


@cute.jit
def to_f32_bf16(value: cutlass.BFloat16) -> Float32:
    """Widen BF16 to FP32 using its shared bit representation."""
    bits = value.bitcast(Uint16)
    return (Uint32(bits) << Uint32(16)).bitcast(Float32)


@cute.jit
def x2_lo_to_f32_bf16(bits: Int32) -> Float32:
    return ((bits & Int32(0xFFFF)) << Int32(16)).bitcast(Float32)


@cute.jit
def x2_hi_to_f32_bf16(bits: Int32) -> Float32:
    # `(x >> 16) << 16` ≡ `x & 0xFFFF0000`, sidestepping signed-literal
    # issues. Sign bits from the arith-right shift get zeroed by the left shift.
    return ((bits >> Int32(16)) << Int32(16)).bitcast(Float32)


@cute.jit
def truncate_f32_bf16(val: Float32) -> Float32:
    """Round FP32 to BF16 precision (round-to-nearest-even), keeping FP32 storage."""
    bf16_bits = cute.arch.inline_ptx(
        "cvt.rn.bf16.f32 {$w0}, {$r0};",
        write_only_types=[Uint16],
        read_only_args=[val],
    )
    return to_f32_bf16(bf16_bits.bitcast(cutlass.BFloat16))


@cute.jit
def to_f32_f16(value: cutlass.Float16) -> Float32:
    """Widen FP16 to FP32."""
    return cute.arch.inline_ptx(
        "cvt.f32.f16 {$w0}, {$r0};",
        write_only_types=[Float32],
        read_only_args=[value.bitcast(Uint16)],
    )


@cute.jit
def x2_lo_to_f32_f16(bits: Int32) -> Float32:
    return to_f32_f16(Uint16(bits).bitcast(cutlass.Float16))


@cute.jit
def x2_hi_to_f32_f16(bits: Int32) -> Float32:
    hi_shifted = bits >> Int32(16)
    return to_f32_f16(Uint16(hi_shifted).bitcast(cutlass.Float16))


@cute.jit
def truncate_f32_f16(val: Float32) -> Float32:
    """Round FP32 to FP16 precision, keeping FP32 storage."""
    f16_bits = cute.arch.inline_ptx(
        "cvt.rn.f16.f32 {$w0}, {$r0};",
        write_only_types=[Uint16],
        read_only_args=[val],
    )
    return to_f32_f16(f16_bits.bitcast(cutlass.Float16))


def is_packed16(dtype) -> bool:
    """True if `dtype` is one of the 16-bit packed input formats."""
    return dtype is cutlass.BFloat16 or dtype is cutlass.Float16
