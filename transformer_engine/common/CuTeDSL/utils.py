# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Low-level CuTeDSL helpers for bitcasts, math intrinsics, and packed FP16/BF16 ops."""

import logging
from typing import Optional

import cutlass
from cutlass import cute
from cutlass import Boolean, Float32, Int64, Int32, Uint16, Uint32, Uint64
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass.cutlass_dsl import dsl_user_op

# This must match `inline std::string to_string(const DType type)`` in transformer_engine/common/common.h exactly
# which is how we encode types to strings in C++
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
    # pylint: disable=c-extension-no-member
    from cuda.bindings import runtime as cudart  # pylint: disable=no-name-in-module

    def check(result):
        err, *values = result
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA runtime error {err}")
        return values[0]

    try:
        if device_id is None:
            device_id = check(cudart.cudaGetDevice())
        attr = cudart.cudaDeviceAttr
        major = check(
            cudart.cudaDeviceGetAttribute(attr.cudaDevAttrComputeCapabilityMajor, device_id)
        )
        minor = check(
            cudart.cudaDeviceGetAttribute(attr.cudaDevAttrComputeCapabilityMinor, device_id)
        )
        return (major, minor)
    except Exception as e:  # pylint: disable=broad-except
        logger.warning("Could not query CUDA device compute capability (%s); assuming (0, 0).", e)
        return (0, 0)


def str_to_cutlass_dtype(dtype_str: str):
    """Convert a string dtype to a cutlass dtype, or None if unknown."""
    return _CUTLASS_DTYPE_FROM_STR.get(dtype_str, None)


def cutlass_dtype_to_str(dtype):
    """Convert a cutlass dtype back to its protocol string, or None if unknown."""
    return _STR_FROM_CUTLASS_DTYPE.get(dtype, None)


FP32_MANTISSA_BITS = 23
# Largest finite values, which the scaling math clamps to as the CUDA kernels do with
# TypeExtrema<T>::max. CuTeDSL's numeric types do not expose them.
FLOAT32_MAX = float.fromhex("0x1.fffffep127")
BFLOAT16_MAX = float.fromhex("0x1.fep127")


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
    """Extract and widen the low BF16 lane from a packed 32-bit value."""
    return ((bits & Int32(0xFFFF)) << Int32(16)).bitcast(Float32)


@cute.jit
def x2_hi_to_f32_bf16(bits: Int32) -> Float32:
    """Extract and widen the high BF16 lane from a packed 32-bit value."""
    # Clear the low 16 bits while preserving the high BF16 value. Use shifts because
    # the equivalent 0xFFFF0000 mask does not fit in a positive signed Int32 literal.
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
    """Extract and widen the low FP16 lane from a packed 32-bit value."""
    return to_f32_f16(Uint16(bits).bitcast(cutlass.Float16))


@cute.jit
def x2_hi_to_f32_f16(bits: Int32) -> Float32:
    """Extract and widen the high FP16 lane from a packed 32-bit value."""
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


def abs_max_tree_bf16x2(words):
    """Packed abs-max over a power-of-two count of words, as a balanced tree: the magnitudes of
    both halves are the abs-maxes of the even and odd elements (signs are unreliable,
    max.xorsign.abs leaves the XOR of the operands' signs)."""
    level = list(words)
    while len(level) > 1:
        level = [
            Uint32(abs_max_x2_bf16(Int32(level[2 * i]), Int32(level[2 * i + 1])))
            for i in range(len(level) // 2)
        ]
    return level[0]


# --- Global memory access by address ---
#
# The register-resident (g2r) kernels address global memory with 64-bit byte addresses rather
# than through layouts: every access is one row offset from a thread-constant base, and the cache
# hints and vector widths are part of what makes them fast.


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
