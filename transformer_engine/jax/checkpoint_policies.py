# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Checkpoint policies for Transformer Engine in JAX.

This module provides JAX checkpoint policies that are compatible with Transformer Engine's custom primitives.
"""

import jax
from .cpp_extensions.gemm import GemmPrimitive, GroupedGemmPrimitive


__all__ = [
    "te_gemms_saveable",
    "te_gemms_with_no_batch_dims_saveable",
    "dots_and_te_gemms_with_no_batch_dims"
]

def te_gemms_saveable(prim, *_, **__) -> bool:
    """Checkpoint policy for Transformer Engine GEMMs."""
    return prim in {GemmPrimitive.outer_primitive, GroupedGemmPrimitive.outer_primitive}

# The TE GEMM primitive does not support batched GEMMs, so the batched and non-batched policies are the same.
te_gemms_with_no_batch_dims_saveable = te_gemms_saveable

dots_and_te_gemms_with_no_batch_dims = jax.checkpoint_policies.save_from_both_policies(
    jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
    te_gemms_with_no_batch_dims_saveable,
)