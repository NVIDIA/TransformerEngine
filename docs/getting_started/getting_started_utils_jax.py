# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Utility functions for Getting Started with Transformer Engine - JAX
====================================================================

Helper classes and functions for the getting started examples.
"""

import time
from typing import Callable, Any, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
import transformer_engine.jax as te
from transformer_engine.jax.sharding import MeshResource


def speedometer(
    apply_fn: Callable,
    params: Any,
    x: jnp.ndarray,
    forward_kwargs: dict = {},
    autocast_kwargs: Optional[dict] = None,
    timing_iters: int = 100,
    warmup_iters: int = 10,
    label: str = "benchmark",
) -> float:
    """Measure average forward + backward pass time for a JAX module.

    Args:
        apply_fn: JIT-compiled apply function
        params: Model parameters
        x: Input tensor
        forward_kwargs: Additional kwargs for forward pass
        autocast_kwargs: Kwargs for te.autocast context
        timing_iters: Number of timing iterations
        warmup_iters: Number of warmup iterations
        label: Optional label for logging

    Returns:
        Average time per iteration in milliseconds
    """
    if autocast_kwargs is None:
        autocast_kwargs = {"enabled": False}
    else:
        autocast_kwargs = dict(autocast_kwargs)
    autocast_kwargs.setdefault("mesh_resource", MeshResource())

    def loss_fn(params, x):
        y = apply_fn(params, x, **forward_kwargs)
        return jnp.sum(y)

    # JIT compile within autocast context
    with te.autocast(**autocast_kwargs):
        grad_fn = jax.jit(jax.value_and_grad(loss_fn))

        # Warmup runs
        for _ in range(warmup_iters):
            loss, grads = grad_fn(params, x)
            jax.block_until_ready((loss, grads))

        # Timing runs
        times = []
        for _ in range(timing_iters):
            start = time.perf_counter()
            loss, grads = grad_fn(params, x)
            jax.block_until_ready((loss, grads))
            times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times) * 1000
    print(f"Mean time: {avg_time:.3f} ms")
    return avg_time
