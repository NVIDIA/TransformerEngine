# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Elementwise activations and their derivatives (f32)
A CuTeDSL port of the CUDA C++ implementation in common/util/math.h."""

# Each function is a 1:1 port of its math.h counterpart (see module docstring).
# pylint: disable=missing-function-docstring
import os

from cutlass import cute
from cutlass import Float32
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass.cutlass_dsl import dsl_user_op

from transformer_engine.common.CuTeDSL.utils import fma_f32


USE_FAST_MATH = os.environ.get("NVTE_BUILD_ACTIVATION_WITH_FAST_MATH", "0") == "1"


def act_relu(x: Float32) -> Float32:
    return cute.arch.fmax(x, Float32(0.0))


def act_gelu(x: Float32) -> Float32:
    A = Float32(0.79788456)  # sqrt(2/π) truncated to TE's 8-digit literal
    B = Float32(0.03567741)  # = sqrt(2/π) · 0.044715, same truncation
    return x * (
        Float32(0.5) + Float32(0.5) * cute.math.tanh(x * (A + B * x * x), fastmath=USE_FAST_MATH)
    )


def act_silu(x: Float32) -> Float32:
    return x / (Float32(1.0) + cute.math.exp(-x, fastmath=USE_FAST_MATH))


def act_qgelu(x: Float32) -> Float32:
    return x * sigmoid(Float32(1.702) * x)


def act_srelu(x: Float32) -> Float32:
    r = cute.arch.fmax(x, Float32(0.0))
    return r * r


@dsl_user_op
def dact_drelu(x: Float32, *, loc=None, ip=None) -> Float32:
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
    return cute.arch.fmax(Float32(2.0) * x, Float32(0.0))


def sigmoid(x: Float32) -> Float32:
    return Float32(1.0) / (Float32(1.0) + cute.math.exp(-x, fastmath=USE_FAST_MATH))


def dsigmoid(x: Float32) -> Float32:
    s = sigmoid(x)
    return s * (Float32(1.0) - s)


def dact_dsilu(x: Float32) -> Float32:
    return fma_f32(x, dsigmoid(x), sigmoid(x))


def dact_dqgelu(x: Float32) -> Float32:
    ax = Float32(1.702) * x
    return ax * dsigmoid(ax) + sigmoid(ax)


def dact_dgelu(x: Float32) -> Float32:
    t = cute.math.tanh(
        Float32(0.79788456) * x * (Float32(1.0) + Float32(0.044715) * x * x),
        fastmath=USE_FAST_MATH,
    )
    return Float32(0.5) * x * (
        (Float32(1.0) - t * t) * (Float32(0.79788456) + Float32(0.1070322243) * x * x)
    ) + Float32(0.5) * (Float32(1.0) + t)
