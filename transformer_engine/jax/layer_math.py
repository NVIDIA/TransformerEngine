# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX implementation of layers"""
import jax
import jax.numpy as jnp


def layernorm(x, gamma, beta, zero_centered_gamma, eps):
    """
    JAX native layernorm implementations
    """
    x_ = jnp.asarray(x, jnp.float32)
    mean = jnp.mean(x_, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x_ - mean), axis=-1, keepdims=True)
    normed_input = (x_ - mean) * jax.lax.rsqrt(var + eps)
    if zero_centered_gamma:
        gamma += 1.
    return jnp.asarray(normed_input * gamma + beta).astype(x.dtype)


def rmsnorm(x, gamma, zero_centered_gamma, eps):
    """
    JAX native rmsnorm implementations
    """
    x_ = jnp.asarray(x, jnp.float32)
    var = jnp.mean(jnp.square(x_), axis=-1, keepdims=True)
    normed_input = x_ * jax.lax.rsqrt(var + eps)
    if zero_centered_gamma:
        gamma += 1.
    return jnp.asarray(normed_input * gamma).astype(x.dtype)


def layernorm_fp8(x, gamma, beta, scale, amax, out_dtype, zero_centered_gamma, eps):
    """
    JAX native layernorm fp8 implementations
    """
    x_ = jnp.asarray(x, jnp.float32)
    mean = jnp.mean(x_, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x_ - mean), axis=-1, keepdims=True)
    rsigma = jax.lax.rsqrt(var + eps)
    normed_input = (x_ - mean) * rsigma
    if zero_centered_gamma:
        gamma += 1.
    output = normed_input * gamma + beta
    casted_output = (scale * output).astype(out_dtype)
    updated_amax = jax.lax.max(amax, jnp.max(jnp.abs(output)).astype(amax.dtype))
    return casted_output, jnp.squeeze(mean, axis=-1), jnp.squeeze(rsigma, axis=-1), updated_amax


def rmsnorm_fp8(x, gamma, scale, amax, out_dtype, zero_centered_gamma, eps):
    """
    JAX native rmsnorm fp8 implementations
    """
    x_ = jnp.asarray(x, jnp.float32)
    var = jnp.mean(jnp.square(x_), axis=-1, keepdims=True)
    rsigma = jax.lax.rsqrt(var + eps)
    normed_input = x_ * rsigma
    if zero_centered_gamma:
        gamma += 1.
    output = normed_input * gamma
    casted_output = (scale * output).astype(out_dtype)
    updated_amax = jax.lax.max(amax, jnp.max(jnp.abs(output)).astype(amax.dtype))
    return casted_output, jnp.squeeze(rsigma, axis=-1), updated_amax
