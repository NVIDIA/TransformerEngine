# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import cutlass.cute as cute
from cutlass import Float32
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass.cutlass_dsl import T, dsl_user_op

from transformer_engine.common.CuTeDSL.utils import fma_f32


def act_relu(x: Float32) -> Float32:
    return cute.arch.fmax(x, Float32(0.0))


def act_gelu(x: Float32) -> Float32:
    """Tanh-approximation GELU. Constants and operator grouping match TE's
    `transformer_engine/common/util/math.h::gelu` exactly (factored form
    `x · (0.5 + 0.5·tanh(x·(a + b·x²)))`) so quantized output is bit-exact
    against the C++ fused IS_ACT path. Uses `cute.math.tanh(fastmath=False)`
    rather than the `tanh.approx.f32` PTX intrinsic — TE compiles activation
    kernels without `--use_fast_math` by default, so its `tanhf` is the
    IEEE-precise expansion."""
    A = Float32(0.79788456)  # sqrt(2/π) truncated to TE's 8-digit literal
    B = Float32(0.03567741)  # = sqrt(2/π) · 0.044715, same truncation
    return x * (Float32(0.5) + Float32(0.5) * cute.math.tanh(x * (A + B * x * x)))


def act_silu(x: Float32) -> Float32:
    """SiLU/Swish: x · σ(x) = x / (1 + e^-x).
    Matches TE's `silu` (`val / (1 + expf(-val))`)."""
    return x / (Float32(1.0) + cute.math.exp(-x, fastmath=True))


def act_qgelu(x: Float32) -> Float32:
    """Quick GELU: x · σ(1.702·x). Matches TE `qgelu_with_alpha(val, 1.702)` =
    `cval · (1 / (1 + expf(-1.702·cval)))` (multiply by sigmoid, not a divide)."""
    z = Float32(1.702) * x
    return x * (Float32(1.0) / (Float32(1.0) + cute.math.exp(-z, fastmath=True)))


def act_srelu(x: Float32) -> Float32:
    """Squared ReLU: x>0 ? x·x : 0 == (max(0,x))². Matches TE `srelu`."""
    r = cute.arch.fmax(x, Float32(0.0))
    return r * r


@dsl_user_op
def dact_drelu(x: Float32, *, loc=None, ip=None) -> Float32:
    """drelu: x > 0 ? 1 : 0. Matches math.h `drelu` (NaN → 0 via ordered compare)."""
    cond = mlir_arith.cmpf(
        mlir_arith.CmpFPredicate.OGT,
        x.ir_value(loc=loc, ip=ip),
        Float32(0.0).ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    return Float32(
        mlir_arith.select(
            cond,
            Float32(1.0).ir_value(loc=loc, ip=ip),
            Float32(0.0).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


def dact_dsrelu(x: Float32) -> Float32:
    """dsrelu: fmax(2x, 0). Matches math.h `dsrelu`."""
    return cute.arch.fmax(Float32(2.0) * x, Float32(0.0))


def sigmoid(x: Float32) -> Float32:
    """σ(x) = 1 / (1 + e^-x), same exp intrinsic as the forward silu/qgelu."""
    return Float32(1.0) / (Float32(1.0) + cute.math.exp(-x, fastmath=True))


def dact_dsilu(x: Float32) -> Float32:
    """dsilu: x·σ(x)·(1-σ(x)) + σ(x). Matches math.h `dsilu`
    (`cval·dsigmoid + sigmoid`, dsigmoid = s·(1-s))."""
    s = sigmoid(x)
    # cval·dsigmoid + sigmoid as one FFMA — matches nvcc's contraction of
    # math.h `dsilu` (`cval * dsigmoid + sigmoid`) so dbias is bit-exact.
    return fma_f32(x, s * (Float32(1.0) - s), s)


def dact_dqgelu(x: Float32) -> Float32:
    """dqgelu (alpha=1.702): a·x·dσ(a·x) + σ(a·x). Matches math.h
    `dqgelu_with_alpha(val, 1.702)`."""
    a = Float32(1.702)
    ax = a * x
    s = sigmoid(ax)
    return a * x * (s * (Float32(1.0) - s)) + s


def dact_dgelu(x: Float32) -> Float32:
    """dgelu (tanh approximation). Matches math.h `dgelu` term-for-term;
    same tanh argument as the forward `_act_gelu`."""
    t = cute.math.tanh(
        Float32(0.79788456) * x * (Float32(1.0) + Float32(0.044715) * x * x),
        fastmath=False,
    )
    return Float32(0.5) * x * (
        (Float32(1.0) - t * t) * (Float32(0.79788456) + Float32(0.1070322243) * x * x)
    ) + Float32(0.5) * (Float32(1.0) + t)
