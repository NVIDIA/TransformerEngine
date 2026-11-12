# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Explicit TE cleanup while the JAX/XLA CUDA runtime remains usable."""

import sys

import jax
import transformer_engine_jax as tex


def release_cuda_state() -> None:
    """Release all TE-owned CUDA state at a process-wide restart boundary.

    Stop TE submissions on all host threads/ranks and retire captured graphs
    first. This waits for local GPU work, clears JAX compilation caches, and
    finalizes TE EP across all local devices. Borrowed NCCL communicators remain
    usable through cleanup; every collective rank participates in the same order.
    Model arrays and the XLA runtime survive.
    """
    from .cpp_extensions import ep as ep_extension  # pylint: disable=import-outside-toplevel

    jax.effects_barrier()
    for device in jax.local_devices(backend="gpu"):
        tex.synchronize_cuda_device(device.local_hardware_id)
    jax.clear_caches()
    if ep_extension._ep_config is not None:  # pylint: disable=protected-access
        from . import ep  # pylint: disable=import-outside-toplevel

        ep.ep_finalize()
    triton = sys.modules.get("transformer_engine.jax.triton_extensions.utils")
    if triton is not None:
        triton.release_cuda_state()
    tex.release_cuda_state()


def reset_cuda_state() -> None:
    """Reject a false-success reset of a lost JAX/XLA-owned CUDA runtime."""
    raise RuntimeError(
        "TE cannot recreate a lost JAX/XLA CUDA runtime. Restart the worker after "
        "context loss; use release_cuda_state while CUDA is usable for a "
        "context-preserving in-process restart. Native owners may use the common C API."
    )
