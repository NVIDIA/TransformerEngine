# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Layer normalization operations for Transformer Engine in JAX.

This module provides optimized layer normalization operations for transformer
architectures, including support for different normalization types and quantization.
It implements various normalization strategies like LayerNorm and RMSNorm, with
optional zero-centered gamma and epsilon parameters.
"""

from functools import partial

import jax
import jax.numpy as jnp

from . import cpp_extensions as tex

from .quantize import (
    Quantizer,
)


def canonicalize_norm_type(x):
    """Convert normalization type string to canonical form.

    Args:
        x: Input normalization type string

    Returns:
        Canonicalized normalization type string
    """
    canonicalized = x.lower().strip().replace("-", "").replace("_", "")
    assert canonicalized in ["layernorm", "rmsnorm"]
    return canonicalized


def layernorm(
    x: jnp.ndarray,
    gamma: jnp.ndarray,
    beta: jnp.ndarray,
    norm_type: str,
    zero_centered_gamma: bool = False,
    epsilon: float = 1e-6,
    quantizer: Quantizer = None,
):
    """Apply layer normalization with optional quantization.

    This function implements layer normalization with support for different
    normalization types and optional quantization. It normalizes the input
    tensor using the provided gamma and beta parameters.

    Args:
        x: Input tensor to normalize
        gamma: Scale parameter for normalization
        beta: Shift parameter for normalization
        norm_type: Type of normalization to apply
        zero_centered_gamma: Whether to use zero-centered gamma
        epsilon: Small constant for numerical stability
        quantizer: Optional quantizer for quantizing the output

    Returns:
        Normalized output tensor
    """
    output = _layernorm(x, gamma, beta, norm_type, zero_centered_gamma, epsilon, quantizer)
    return output


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def _layernorm(x, gamma, beta, norm_type: str, zero_centered_gamma, epsilon, quantizer):
    """Internal implementation of layer normalization with custom VJP.

    This function implements the core layer normalization logic with support
    for custom vector-Jacobian product (VJP) for automatic differentiation.

    Args:
        x: Input tensor
        gamma: Scale parameter
        beta: Shift parameter
        norm_type: Type of normalization
        zero_centered_gamma: Whether to use zero-centered gamma
        epsilon: Small constant for numerical stability
        quantizer: Optional quantizer

    Returns:
        Normalized tensor
    """
    output, _ = _layernorm_fwd_rule(
        x, gamma, beta, norm_type, zero_centered_gamma, epsilon, quantizer
    )
    return output


def _layernorm_fwd_rule(x, gamma, beta, norm_type: str, zero_centered_gamma, epsilon, quantizer):
    """Forward pass rule for layer normalization.

    Args:
        x: Input tensor
        gamma: Scale parameter
        beta: Shift parameter
        norm_type: Type of normalization
        zero_centered_gamma: Whether to use zero-centered gamma
        epsilon: Small constant for numerical stability
        quantizer: Optional quantizer

    Returns:
        Tuple of (output, context) for backward pass
    """

    norm_type = canonicalize_norm_type(norm_type)
    output, mu, rsigma = tex.normalization_fwd(
        x, gamma, beta, zero_centered_gamma, epsilon, norm_type, quantizer
    )
    # This is a no-op for higher-precision tensors
    output = output.dequantize()

    return output, (x, mu, rsigma, gamma, beta, quantizer)


def _layernorm_bwd_rule(norm_type, zero_centered_gamma, epsilon, ctx, dz):
    """Backward pass rule for layer normalization.

    Args:
        norm_type: Type of normalization
        zero_centered_gamma: Whether to use zero-centered gamma
        epsilon: Small constant for numerical stability
        ctx: Context from forward pass
        dz: Gradient from upstream

    Returns:
        Tuple of gradients with respect to inputs
    """
    x, mu, rsigma, gamma, beta, quantizer = ctx

    dx, dgamma, dbeta = tex.normalization_bwd(
        dz, x, mu, rsigma, gamma, beta, zero_centered_gamma, epsilon, norm_type
    )
    return dx, dgamma, dbeta, quantizer


_layernorm.defvjp(_layernorm_fwd_rule, _layernorm_bwd_rule)
