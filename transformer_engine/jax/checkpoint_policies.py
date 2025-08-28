# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Checkpoint policies for Transformer Engine in JAX.

This module provides JAX checkpoint policies that are compatible with Transformer Engine's custom primitives.
"""

import jax
from .cpp_extensions.gemm import GemmPrimitive, GroupedGemmPrimitive


__all__ = [
    "te_gemms_saveable",
    "dots_and_te_gemms_with_no_batch_dims",
    "checkpoint_dots_and_te_gemms",
]


def te_gemms_saveable(prim, *_, **__) -> bool:
    """Checkpoint policy for Transformer Engine GEMMs."""
    is_te_gemm = prim in {GemmPrimitive.outer_primitive, GroupedGemmPrimitive.outer_primitive}
    # Workaround to include JAX's scaled_matmul until JAX checkpoint policies for dots are
    # updated to include it.
    is_jax_scaled_matmul = prim.name == "scaled_matmul_wrapper"

    return is_te_gemm or is_jax_scaled_matmul


dots_and_te_gemms_with_no_batch_dims = jax.checkpoint_policies.save_from_both_policies(
    jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
    te_gemms_saveable,
)

checkpoint_dots_and_te_gemms = jax.checkpoint_policies.save_from_both_policies(
    jax.checkpoint_policies.checkpoint_dots,
    te_gemms_saveable,
)
