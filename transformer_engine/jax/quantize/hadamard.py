# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Randomized Hadamard Transform (RHT) utilities for JAX."""
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


def get_rht_matrix() -> jnp.ndarray:
    """Get the Randomized Hadamard Transform (RHT) matrix used in NVFP4 weight gradient quantization.

    Returns:
        A (16, 16) bfloat16 matrix representing the RHT. This matrix is pre-multiplied by the random sign mask.
    """
    import scipy

    block_size = 16
    h = jnp.array(scipy.linalg.hadamard(block_size))

    # Apply the random sign mask
    s = jnp.array(get_wgrad_sign_vector(), dtype=jnp.int32)
    h = jnp.diag(s) @ h

    return (h / jnp.sqrt(block_size)).astype(jnp.bfloat16)
