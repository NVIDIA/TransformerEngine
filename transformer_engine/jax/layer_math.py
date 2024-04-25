# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX implementation of layers"""
import jax
import jax.numpy as jnp


def layernorm(x, gamma, beta, zero_centered_gamma, eps):
    """
    JAX native layernorm implementation
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
    JAX native rmsnorm implementation
    """
    x_ = jnp.asarray(x, jnp.float32)
    var = jnp.mean(jnp.square(x_), axis=-1, keepdims=True)
    normed_input = x_ * jax.lax.rsqrt(var + eps)
    if zero_centered_gamma:
        gamma += 1.
    return jnp.asarray(normed_input * gamma).astype(x.dtype)


def quantize(x, scale, q_dtype):
    """
    Quantize with scale
    """
    dtype_max = (jnp.finfo(q_dtype).max).astype(x.dtype)
    scale = scale.astype(x.dtype)
    clipped_scaled_x = jnp.clip((x * scale), -dtype_max, dtype_max)
    return clipped_scaled_x.astype(q_dtype)


def layernorm_fp8(x, gamma, beta, scale, amax, out_dtype, zero_centered_gamma, eps):
    """
    JAX native layernorm fp8 implementation
    """
    x_ = jnp.asarray(x, jnp.float32)
    mean = jnp.mean(x_, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x_ - mean), axis=-1, keepdims=True)
    rsigma = jax.lax.rsqrt(var + eps)
    normed_input = (x_ - mean) * rsigma
    if zero_centered_gamma:
        gamma += 1.
    output = normed_input * gamma + beta
    casted_output = quantize(output, scale, q_dtype=out_dtype)
    updated_amax = jax.lax.max(amax, jnp.max(jnp.abs(output)).astype(amax.dtype))
    return casted_output, jnp.squeeze(mean, axis=-1), jnp.squeeze(rsigma, axis=-1), updated_amax


def rmsnorm_fp8(x, gamma, scale, amax, out_dtype, zero_centered_gamma, eps):
    """
    JAX native rmsnorm fp8 implementation
    """
    x_ = jnp.asarray(x, jnp.float32)
    var = jnp.mean(jnp.square(x_), axis=-1, keepdims=True)
    rsigma = jax.lax.rsqrt(var + eps)
    normed_input = x_ * rsigma
    if zero_centered_gamma:
        gamma += 1.
    output = normed_input * gamma
    casted_output = quantize(output, scale, q_dtype=out_dtype)
    updated_amax = jax.lax.max(amax, jnp.max(jnp.abs(output)).astype(amax.dtype))
    return casted_output, jnp.squeeze(rsigma, axis=-1), updated_amax


def bias_gelu(inputs, bias):
    """
    JAX native bias_gelu implementation
    """
    return jax.nn.gelu(inputs + bias)


def gated_gelu(inputs):
    """
    JAX native gated gelu implementation
    inputs: (N, 2, H)
    """
    gelu_inputs, identity_inputs = jnp.split(inputs, [1], axis=-2)
    gelu_outputs = jax.nn.gelu(gelu_inputs)
    return jnp.squeeze(gelu_outputs * identity_inputs, axis=-2)


def gated_gelu_fp8(inputs, scale, amax, out_dtype):
    """
    JAX native gated gelu fp8 implementation
    """
    geglu_output = gated_gelu(inputs)
    casted_output = quantize(geglu_output, scale, q_dtype=out_dtype)
    updated_amax = jax.lax.max(amax, jnp.max(jnp.abs(geglu_output)).astype(amax.dtype))
    return casted_output, updated_amax


def cast_fp8(inputs, scale, amax, out_dtype):
    """
    JAX native fp8 casting implementation
    """
    casted_output = quantize(inputs, scale, q_dtype=out_dtype)
    updated_amax = jax.lax.max(amax, jnp.max(jnp.abs(inputs)).astype(amax.dtype))
    return casted_output, updated_amax


def _normalize_axis_boundary(axis, ndim):
    return axis if axis >= 0 else ndim + axis


def _multidim_transpose(shape, static_axis_boundary, transpose_axis_boundary):
    """
    te_cast_transpose_p multi-dims transpose

    static_axis_boundary: int, Indicate those axes <= static_axis_boundary would not be
        involved into transpose, -1 means all axes involve into transpose.
    transpose_axis_boundary: int, Indicate how to split multi-dimensions tensors to 2D matrix for
        transpose. Note, transpose_axis_boundary should be greater than static_axis_boundary

    examples:
        X in shape (dim0, dim1, dim2, dim3, dim4)

        static_axis_boundary == -1, transpose_axis_boundary == 2
            Xt = (dim2, dim3, dim4, dim0, dim1)

        static_axis_boundary == 0, transpose_axis_boundary == 2
            Xt = (dim0, dim2, dim3, dim4, dim1)

        static_axis_boundary == 0, transpose_axis_boundary == 3
            Xt = (dim0, dim3, dim4, dim1. dim2)
    """
    if static_axis_boundary < 0:
        static_axis_boundary = -1    # means no static axes
    assert static_axis_boundary < len(shape) - 2    # at least 2 remaining for transpose.
    transpose_start_idx = static_axis_boundary + 1
    transpose_axis_boundary = _normalize_axis_boundary(transpose_axis_boundary, len(shape))
    assert transpose_start_idx < transpose_axis_boundary
    return (*shape[:transpose_start_idx], *shape[transpose_axis_boundary:],
            *shape[transpose_start_idx:transpose_axis_boundary])


def transpose(inputs, static_axis_boundary, transpose_axis_boundary):
    """
    JAX native transpose implementation
    """
    axes = _multidim_transpose(range(inputs.ndim), static_axis_boundary, transpose_axis_boundary)
    return jnp.transpose(inputs, axes=axes)


def cast_transpose(inputs, scale, amax, out_dtype, static_axis_boundary, transpose_axis_boundary):
    """
    JAX native cast_transpose implementation
    """
    updated_amax = jax.lax.max(amax, jnp.max(jnp.abs(inputs)).astype(amax.dtype))
    casted_output = quantize(inputs, scale, q_dtype=out_dtype)
    casted_transposed_output = transpose(casted_output, static_axis_boundary,
                                         transpose_axis_boundary)
    return casted_output, casted_transposed_output, updated_amax
