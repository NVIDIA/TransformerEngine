# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Elementwise activations and their derivatives (f32)
A CuTeDSL port of the CUDA C++ implementation in common/util/math.h."""

# Each function is a 1:1 port of its math.h counterpart (see module docstring).
# pylint: disable=missing-function-docstring
import os
from typing import Optional

from cutlass import cute
from cutlass import Float32

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
