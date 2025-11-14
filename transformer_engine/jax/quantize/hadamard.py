# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Randomized Hadamard Transform (RHT) utilities for JAX."""
from functools import lru_cache
import weakref

import jax.numpy as jnp


def get_wgrad_sign_vector() -> list[int]:
    """Get a fixed sign vector for the RHT used in NVFP4 weight gradient quantization."""
    return [1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1]


def get_sign_from_vector(vector: list[int]) -> int:
    """Convert a sign vector to a bitmask integer."""
    mask = 0
    for i, v in enumerate(vector):
        mask |= (v == -1) << i
    return mask


def apply_rht(x: jnp.ndarray, inverse=False) -> jnp.ndarray:
    """Apply the Randomized Hadamard Transform (RHT) to the input tensor."""
    h = get_rht_matrix()
    block_size = 16
    if inverse:
        h = jnp.linalg.inv(h.astype(jnp.float32)).astype(jnp.bfloat16)
    # TODO(jberchtold): These reshapes will break partitioning, fixme
    return (x.reshape(-1, block_size) @ h).reshape(x.shape)


_rht_matrix_cache = []


def _get_rht_matrix_impl(opaque_trace_state) -> jnp.ndarray:
    """Get the Randomized Hadamard Transform (RHT) matrix used in NVFP4 weight gradient quantization.

    Args:
        opaque_trace_state: Opaque trace state from JAX. This value is not used except to ensure that the function is re-traced when needed and jit tracers are not leaked.
    Returns:
        A (16, 16) bfloat16 matrix representing the RHT. This matrix is pre-multiplied by the random sign mask.
    """
    import scipy

    for trace_state, matrix in _rht_matrix_cache:
        if trace_state is opaque_trace_state:
            return matrix

    block_size = 16
    h = jnp.array(scipy.linalg.hadamard(block_size))

    # Apply the random sign mask
    s = jnp.array(get_wgrad_sign_vector(), dtype=jnp.int32)
    h = jnp.diag(s) @ h

    rht_matrix = (h / jnp.sqrt(block_size)).astype(jnp.bfloat16)
    _rht_matrix_cache.append((opaque_trace_state, rht_matrix))
    return rht_matrix


def get_rht_matrix() -> jnp.ndarray:
    """Get the Randomized Hadamard Transform (RHT) matrix used in NVFP4 weight gradient quantization.

    Returns:
        A (16, 16) bfloat16 matrix representing the RHT. This matrix is pre-multiplied by the random sign mask.
    """
    from jax._src.core import get_opaque_trace_state
    opaque_trace_state = get_opaque_trace_state()
    return _get_rht_matrix_impl(opaque_trace_state)
