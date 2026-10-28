# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Elementwise activations and their derivatives (f32)
A CuTeDSL port of the CUDA C++ implementation in common/util/math.h."""

# Each function is a 1:1 port of its math.h counterpart (see module docstring).
# pylint: disable=missing-function-docstring
import os
import struct

import cutlass
from cutlass import cute
from cutlass import Float32, Int32, Uint32

from transformer_engine.common.CuTeDSL.utils import fma_f32

USE_FAST_MATH = os.environ.get("NVTE_USE_FAST_MATH", "0") == "1"


def tanh(x: Float32) -> Float32:
    return cute.math.tanh(x, fastmath=USE_FAST_MATH)


def exp(x: Float32) -> Float32:
    return cute.math.exp(x, fastmath=USE_FAST_MATH)


def act_relu(x: Float32) -> Float32:
    return cute.arch.fmax(x, Float32(0.0))


def act_gelu(x: Float32) -> Float32:
    A = Float32(0.79788456)  # sqrt(2/π) truncated to TE's 8-digit literal
    B = Float32(0.03567741)  # = sqrt(2/π) · 0.044715, same truncation
    return x * (Float32(0.5) + Float32(0.5) * tanh(x * (A + B * x * x)))


def act_silu(x: Float32) -> Float32:
    # x * sigmoid(x), as math.h computes it: x / (1 + exp(-x)) rounds differently in FP32.
    return x * sigmoid(x)


def act_qgelu(x: Float32) -> Float32:
    return x * sigmoid(Float32(1.702) * x)


def act_srelu(x: Float32) -> Float32:
    r = cute.arch.fmax(x, Float32(0.0))
    return r * r


@cute.jit
def dact_drelu(x: Float32) -> Float32:
    return Float32(x > Float32(0.0))


def dact_dsrelu(x: Float32) -> Float32:
    return cute.arch.fmax(Float32(2.0) * x, Float32(0.0))


def sigmoid(x: Float32) -> Float32:
    return Float32(1.0) / (Float32(1.0) + exp(-x))


def dsigmoid(x: Float32) -> Float32:
    s = sigmoid(x)
    return s * (Float32(1.0) - s)


def dact_dsilu(x: Float32) -> Float32:
    return fma_f32(x, dsigmoid(x), sigmoid(x))


def dact_dqgelu(x: Float32) -> Float32:
    ax = Float32(1.702) * x
    return ax * dsigmoid(ax) + sigmoid(ax)


def dact_dgelu(x: Float32) -> Float32:
    t = tanh(Float32(0.79788456) * x * (Float32(1.0) + Float32(0.044715) * x * x))
    return Float32(0.5) * x * (
        (Float32(1.0) - t * t) * (Float32(0.79788456) + Float32(0.1070322243) * x * x)
    ) + Float32(0.5) * (Float32(1.0) + t)


# BF16 activation tables (see common/util/activation_table.cuh, which builds them; the layouts
# below must match it). With a table, BF16 activations follow the table specification: GeLU and
# SiLU of -inf give -0, and the derivatives below the table window are the linear term
# 0.5 + slope*x. Kernels either read the table ("table_*_2x") or compute what it holds
# ("activation_op"/"dactivation_op").

# Forward tables: rows of (op(+m), op(-m)) BF16 entries for magnitudes m from LOW_SENTINEL to
# HIGH_SENTINEL; the LOW_SENTINEL row holds -inf, the HIGH_SENTINEL row +inf and -0.
FORWARD_TABLE_WINDOWS = {"gelu": (0x3B4B, 0x40A5), "silu": (0x3BB6, 0x42B1)}
# Derivative tables: FP32 entries for magnitudes [LO, HI], positive inputs from entry 0 and
# negative ones from entry STRIDE; from SATURATED_END on (and for NaN) the result is NaN.
# Keyed by (activation, layout); layout "exact" is the exact dSiLU table.
DERIVATIVE_TABLE_WINDOWS = {
    ("dgelu", "default"): (0x3980, 0x4100, 0x6044, 2048),
    ("dsilu", "default"): (0x3B00, 0x42B2, 0x7F80, 2048),
    ("dsilu", "exact"): (0x3300, 0x42B2, 0x7F80, 4096),
}
DERIVATIVE_SLOPES = {"dgelu": 0.79788456, "dsilu": 0.5}


def activation_table_bytes(activation: str, layout: str) -> int:
    """Size of the table for activation in layout, rounded up to 16 bytes as TE allocates it."""
    if activation in FORWARD_TABLE_WINDOWS:
        lo, hi = FORWARD_TABLE_WINDOWS[activation]
        rows = (hi + 1) - (lo - 1) + 1
        return (rows * 2 * 2 + 15) // 16 * 16
    lo, hi, _, stride = DERIVATIVE_TABLE_WINDOWS[(activation, layout)]
    return ((stride + hi - lo + 1) * 4 + 15) // 16 * 16


@cute.jit
def _select_below(x: Float32, bound: Float32, below: Float32, other: Float32) -> Float32:
    """below if |x| < bound, else other. Branch-free; NaN compares false."""
    return cute.arch.inline_ptx(
        "{ .reg .pred p; .reg .f32 m; abs.f32 m, {$r0}; setp.lt.f32 p, m, {$r1}; "
        "selp.f32 {$w0}, {$r2}, {$r3}, p; }",
        write_only_types=[Float32],
        read_only_args=[x, bound, below, other],
    )


def forward_table_spec(op):
    """op as the forward tables hold it (see the comment above), computed."""

    @cute.jit
    def table_spec(x: Float32) -> Float32:
        # Outside the window the table's closed forms equal util/math.h rounded to BF16, except
        # at -inf, where the formula gives NaN and the table -0. util/math.h gives -0 for every
        # finite x below -2^100, so clamp the input there; NaN fails the comparison.
        clamped = cute.arch.inline_ptx(
            "{ .reg .pred p; setp.lt.f32 p, {$r0}, {$r1}; selp.f32 {$w0}, {$r1}, {$r0}, p; }",
            write_only_types=[Float32],
            read_only_args=[x, Float32(-(2.0**100))],
        )
        return op(clamped)

    return table_spec


def derivative_table_spec(dop, activation: str, layout: str):
    """dop as the derivative table of activation in layout holds it, computed."""
    # Below the exact dSiLU table's window the linear term equals util/math.h.
    if layout == "exact":
        return dop
    lo = DERIVATIVE_TABLE_WINDOWS[(activation, layout)][0]
    window_low = struct.unpack("<f", struct.pack("<I", lo << 16))[0]
    slope = DERIVATIVE_SLOPES[activation]

    @cute.jit
    def table_spec(x: Float32) -> Float32:
        # Above the window util/math.h already gives the saturated value or NaN.
        linear = fma_f32(Float32(slope), x, Float32(0.5))
        return _select_below(x, Float32(window_low), linear, dop(x))

    return table_spec


@cute.jit
def table_forward_2x(
    x: Int32,
    sTable_u16: cute.Tensor,
    LOW_SENTINEL: cutlass.Constexpr[int],
    HIGH_SENTINEL: cutlass.Constexpr[int],
) -> Int32:
    """Forward activation of a BF16 pair (low lane in the low half) from a forward table in
    shared memory, as BF16 bits. Same instructions as forward_2x in util/activation_table.cuh."""
    # Clamp both magnitudes to the table rows. bf16x2 max/min ignore NaN, which lands on the
    # -inf row; min.NaN with the upper bound below then makes it NaN again.
    clamped = Uint32(
        cute.arch.inline_ptx(
            "{ .reg .b32 m, lo, hi; "
            f"mov.b32 lo, {LOW_SENTINEL * 0x10001}; mov.b32 hi, {HIGH_SENTINEL * 0x10001}; "
            "and.b32 m, {$r0}, 0x7fff7fff; max.bf16x2 m, m, lo; min.bf16x2 {$w0}, m, hi; }",
            write_only_types=[Int32],
            read_only_args=[x],
        )
    )
    # Both lanes' entry indices in one word: 2 * row + sign per half. Each stays below 0x10000,
    # so the halves never carry into each other.
    indices = ((clamped - Uint32(LOW_SENTINEL * 0x10001)) << Uint32(1)) + (
        (Uint32(x) >> Uint32(15)) & Uint32(0x10001)
    )
    entry_lo = Uint32(sTable_u16[Int32(indices & Uint32(0xFFFF))])
    entry_hi = Uint32(sTable_u16[Int32(indices >> Uint32(16))])
    entries = Int32(entry_lo | (entry_hi << Uint32(16)))
    # max(entry, RN(x/2)) picks RN(x/2) below the window; min.NaN(., max.NaN(x, -0)) picks x
    # above it for x > 0, -0 for x < 0, and NaN for NaN.
    return cute.arch.inline_ptx(
        "{ .reg .b32 half, upper, neg_zero, h; "
        "mov.b32 half, 0x3f003f00; mov.b32 neg_zero, 0x80008000; "
        "mul.rn.bf16x2 h, {$r0}, half; max.NaN.bf16x2 upper, {$r0}, neg_zero; "
        "max.bf16x2 h, {$r1}, h; min.NaN.bf16x2 {$w0}, h, upper; }",
        write_only_types=[Int32],
        read_only_args=[x, entries],
    )


@cute.jit
def table_derivative_2x(
    x: Int32,
    sTable_f32: cute.Tensor,
    LO: cutlass.Constexpr[int],
    HI: cutlass.Constexpr[int],
    SATURATED_END: cutlass.Constexpr[int],
    STRIDE: cutlass.Constexpr[int],
    SLOPE: cutlass.Constexpr[float],
):
    """Derivative of a BF16 pair (low lane in the low half) from a derivative table in shared
    memory, as (lo, hi) FP32. Same results as derivative_2x in util/activation_table.cuh."""
    magnitude = Uint32(x) & Uint32(0x7FFF7FFF)
    clamped = Uint32(
        cute.arch.inline_ptx(
            "{ .reg .b32 lo, hi; "
            f"mov.b32 lo, {LO * 0x10001}; mov.b32 hi, {HI * 0x10001}; "
            "max.bf16x2 {$w0}, {$r0}, lo; min.bf16x2 {$w0}, {$w0}, hi; }",
            write_only_types=[Int32],
            read_only_args=[Int32(magnitude)],
        )
    )
    bits = Uint32(x)
    # Both lanes' entry indices in one word: the sign bits (15 and 31), shifted down to STRIDE
    # (a power of two), select the negative half. Each index stays below 0x10000, so the halves
    # never carry into each other.
    SIGN_SHIFT = 15 - (STRIDE.bit_length() - 1)
    indices = (clamped - Uint32(LO * 0x10001)) + ((bits ^ magnitude) >> Uint32(SIGN_SHIFT))
    entry_lo = sTable_f32[Int32(indices & Uint32(0xFFFF))]
    entry_hi = sTable_f32[Int32(indices >> Uint32(16))]
    # Below the window: the linear term 0.5 + SLOPE*x, rounded once.
    linear_lo = fma_f32(Float32(SLOPE), (bits << Uint32(16)).bitcast(Float32), Float32(0.5))
    linear_hi = fma_f32(Float32(SLOPE), (bits & Uint32(0xFFFF0000)).bitcast(Float32), Float32(0.5))
    # One BF16x2 compare per condition sets both lanes' predicates. The unordered >= also holds
    # for NaN, which gives NaN like the saturated end.
    return cute.arch.inline_ptx(
        "{ .reg .pred below_lo, below_hi, nan_lo, nan_hi; .reg .b32 lo, end; mov.b32 lo,"
        f" {LO * 0x10001}; mov.b32 end, {SATURATED_END * 0x10001}; setp.lt.bf16x2"
        " below_lo|below_hi, {$r0}, lo; setp.geu.bf16x2 nan_lo|nan_hi, {$r0}, end; selp.f32"
        " {$w0}, {$r1}, {$r2}, below_lo; selp.f32 {$w1}, {$r3}, {$r4}, below_hi;"
        " selp.b32 {$w0}, 0x7fffffff, {$w0}, nan_lo; selp.b32 {$w1}, 0x7fffffff, {$w1},"
        " nan_hi; }",
        write_only_types=[Float32, Float32],
        read_only_args=[Int32(magnitude), linear_lo, entry_lo, linear_hi, entry_hi],
    )
