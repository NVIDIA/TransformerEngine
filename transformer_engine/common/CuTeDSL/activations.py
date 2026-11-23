# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Elementwise activations and their derivatives (f32)
A CuTeDSL port of the CUDA C++ implementation in common/util/math.h."""

# Each function is a 1:1 port of its math.h counterpart (see module docstring).
# pylint: disable=missing-function-docstring
import os
from typing import Callable, Optional

import cutlass
from cutlass import cute
from cutlass import Float32, Int32, Int64

from transformer_engine.common.CuTeDSL.utils import fma_f32

USE_FAST_MATH = os.environ.get("NVTE_USE_FAST_MATH", "0") == "1"


def tanh(x: Float32) -> Float32:
    return cute.math.tanh(x, fastmath=USE_FAST_MATH)


def exp(x: Float32, fast_math: bool = False) -> Float32:
    """exp(x); fast_math, or NVTE_USE_FAST_MATH, selects the fast approximate exp."""
    return cute.math.exp(x, fastmath=fast_math or USE_FAST_MATH)


def act_relu(x: Float32) -> Float32:
    return cute.arch.fmax(x, Float32(0.0))


def act_gelu(x: Float32) -> Float32:
    A = Float32(0.79788456)  # sqrt(2/π) truncated to TE's 8-digit literal
    B = Float32(0.03567741)  # = sqrt(2/π) · 0.044715, same truncation
    return x * (Float32(0.5) + Float32(0.5) * tanh(x * (A + B * x * x)))


def act_silu(x: Float32, sig: Optional[Float32] = None, fast_math: bool = False) -> Float32:
    # Recompute the sigmoid if not provided one
    if sig is None:
        sig = sigmoid(x, fast_math)
    return x * sig


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


def sigmoid(x: Float32, fast_math: bool = False) -> Float32:
    return Float32(1.0) / (Float32(1.0) + exp(-x, fast_math))


def dsigmoid(x: Float32, fast_math: bool = False) -> Float32:
    s = sigmoid(x, fast_math)
    return s * (Float32(1.0) - s)


def dact_dsilu(x: Float32, sig: Optional[Float32] = None, fast_math: bool = False) -> Float32:
    # Recompute the sigmoid if not provided one
    if sig is None:
        return fma_f32(x, dsigmoid(x, fast_math), sigmoid(x, fast_math))
    return x * sig * (Float32(1.0) - sig) + sig


def dact_dqgelu(x: Float32) -> Float32:
    ax = Float32(1.702) * x
    return ax * dsigmoid(ax) + sigmoid(ax)


def dact_dgelu(x: Float32) -> Float32:
    t = tanh(Float32(0.79788456) * x * (Float32(1.0) + Float32(0.044715) * x * x))
    return Float32(0.5) * x * (
        (Float32(1.0) - t * t) * (Float32(0.79788456) + Float32(0.1070322243) * x * x)
    ) + Float32(0.5) * (Float32(1.0) + t)


# ---- Gated activations (gated_mxfp8.cuh) ----
# The parameterized ones take their ClampedSwiGLUParam / SiTUGLUParam fields as
# trailing runtime arguments.


def act_sigmoid(x: Float32) -> Float32:
    return sigmoid(x)


def dact_dsigmoid(x: Float32) -> Float32:
    return dsigmoid(x)


def act_clamped_silu(x: Float32, limit: Float32, alpha: Float32) -> Float32:
    c = cute.arch.fmin(limit, x)
    return c * sigmoid(alpha * c)


def scaled_swiglu_approx_x2_pair(dtype, clamped: bool) -> Callable:
    """Return (silu_approx_x2(act) * gate) * scale of two adjacent elements, rounded to dtype.

    group_scaled_swiglu_mxfp8.cuh's activation, in f32x2 math. silu_approx_x2 returns twice the
    SiLU, so callers pass half the scale they want. Clamped SwiGLU clamps act from above and
    gate on both sides, then offsets gate. A pair is one packed register for 16-bit dtype and an
    Int64 of two FP32 values for FP32. The op takes (act, gate, half_scale, limit, half_alpha,
    glu_linear_offset); plain SwiGLU ignores the last three.
    """
    if dtype is Float32:
        unpack = "mov.b64 vx, {$r0};\n\tmov.b64 vg, {$r1};\n\t"
        pack = "mov.b64 {$w0}, vy;\n\t"
        pair_type = Int64
    else:
        if dtype is cutlass.BFloat16:
            # A BF16 value is the upper half of its FP32 value.
            widen = (
                "prmt.b32 x0, 0x0, {$r0}, 0x5410;\n\tprmt.b32 x1, 0x0, {$r0}, 0x7632;\n\t"
                "prmt.b32 g0, 0x0, {$r1}, 0x5410;\n\tprmt.b32 g1, 0x0, {$r1}, 0x7632;\n\t"
            )
        else:
            widen = (
                "mov.b32 {h0, h1}, {$r0};\n\tcvt.f32.f16 x0, h0;\n\tcvt.f32.f16 x1, h1;\n\t"
                "mov.b32 {h0, h1}, {$r1};\n\tcvt.f32.f16 g0, h0;\n\tcvt.f32.f16 g1, h1;\n\t"
            )
        fmt = "bf16x2" if dtype is cutlass.BFloat16 else "f16x2"
        unpack = widen + "mov.b64 vx, {x0, x1};\n\tmov.b64 vg, {g0, g1};\n\t"
        pack = f"mov.b64 {{y0, y1}}, vy;\n\tcvt.rn.{fmt}.f32 {{$w0}}, y1, y0;\n\t"
        pair_type = Int32
    if clamped:
        # $r3: limit, $r4: alpha / 2, $r5: glu_linear_offset
        activation = (
            "mov.b64 {g0, g1}, vg;\n\t"
            "min.xorsign.abs.f32 g0, g0, {$r3};\n\tmin.xorsign.abs.f32 g1, g1, {$r3};\n\t"
            "mov.b64 vg, {g0, g1};\n\tmov.b64 vc, {{$r5}, {$r5}};\n\tadd.rn.f32x2 vg, vg, vc;\n\t"
            "mov.b64 {x0, x1}, vx;\n\tmin.f32 x0, {$r3}, x0;\n\tmin.f32 x1, {$r3}, x1;\n\t"
            "mov.b64 vx, {x0, x1};\n\tmov.b64 vc, {{$r4}, {$r4}};\n\tmul.rn.f32x2 vt, vc, vx;\n\t"
        )
    else:
        activation = "mov.b64 vc, 0x3F0000003F000000;\n\tmul.rn.f32x2 vt, vc, vx;\n\t"  # x / 2
    asm = (
        "{\n\t"
        ".reg.b64 vx, vg, vt, vc, vy;\n\t.reg.b32 x0, x1, g0, g1, t0, t1, y0, y1;\n\t"
        ".reg.b16 h0, h1;\n\t"
        + unpack
        + activation
        + "mov.b64 {t0, t1}, vt;\n\ttanh.approx.f32 t0, t0;\n\ttanh.approx.f32 t1, t1;\n\t"
        "mov.b64 vt, {t0, t1};\n\t"
        "fma.rn.f32x2 vt, vx, vt, vx;\n\t"  # 2 * silu(x) = x * tanh(x / 2) + x
        "mul.rn.f32x2 vy, vt, vg;\n\t"
        "mov.b64 vc, {{$r2}, {$r2}};\n\tmul.rn.f32x2 vy, vy, vc;\n\t" + pack + "}"
    )

    @cute.jit
    def fn(act, gate, half_scale, limit, half_alpha, glu_linear_offset):
        args = [act, gate, half_scale]
        if cutlass.const_expr(clamped):
            args += [limit, half_alpha, glu_linear_offset]
        return cute.arch.inline_ptx(asm, write_only_types=[pair_type], read_only_args=args)

    return fn


def act_situ_gate(x: Float32, beta1: Float32) -> Float32:
    """SiTU-GLU's activation, applied to the act half."""
    return beta1 * tanh(x / beta1) * sigmoid(x)


def dact_dsitu_gate(x: Float32, beta1: Float32) -> Float32:
    """Derivative of act_situ_gate."""
    t = tanh(x / beta1)
    s = sigmoid(x)
    return (Float32(1.0) - t * t) * s + beta1 * t * s * (Float32(1.0) - s)


def identity(x: Float32) -> Float32:
    """The default gate: gating is ideally the identity."""
    return x


def didentity(x: Float32) -> Float32:  # pylint: disable=unused-argument
    """Derivative of identity."""
    return Float32(1.0)


def situ_up(x: Float32, beta2: Float32) -> Float32:
    """SiTU-GLU's gate transform, applied to the gate half."""
    return beta2 * tanh(x / beta2)


def dsitu_up(x: Float32, beta2: Float32) -> Float32:
    """Derivative of situ_up."""
    t = tanh(x / beta2)
    return Float32(1.0) - t * t
