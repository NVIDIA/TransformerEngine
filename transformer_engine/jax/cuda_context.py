# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Explicit TE cleanup while the JAX/XLA CUDA runtime remains usable."""

import sys

import jax
import transformer_engine_jax as tex


def release_cuda_state(device: int) -> None:
    """Release device-scoped TE state at a coordinated restart boundary.

    Stop TE submissions on all host threads/ranks, complete asynchronous work and
    retire captured graphs first. This clears JAX's process-wide compilation
    cache so executables cannot retain communication owners. NCCL communicators
    must remain usable through cleanup; collective users call on every rank.
    Model arrays and the XLA runtime are not destroyed.
    """
    if not isinstance(device, int) or isinstance(device, bool) or device < 0:
        raise ValueError("device must be a nonnegative CUDA device ordinal")
    from .cpp_extensions import ep as ep_extension  # pylint: disable=import-outside-toplevel

    if ep_extension._ep_config is not None:  # pylint: disable=protected-access
        local_devices = jax.local_devices(backend="gpu")
        if len(local_devices) != 1 or local_devices[0].local_hardware_id != device:
            raise RuntimeError(
                "JAX EP spans process-local devices. Call ep_finalize on every rank "
                "before releasing individual devices."
            )
    jax.effects_barrier()
    tex.synchronize_cuda_device(device)
    jax.clear_caches()
    if ep_extension._ep_config is not None:  # pylint: disable=protected-access
        from . import ep  # pylint: disable=import-outside-toplevel

        ep.ep_finalize()
    triton = sys.modules.get("transformer_engine.jax.triton_extensions.utils")
    if triton is not None:
        triton.release_cuda_state()
    tex.release_cuda_state(device)


def reset_cuda_state(device: int) -> None:
    """Reject a false-success reset of a lost JAX/XLA-owned CUDA runtime."""
    if not isinstance(device, int) or isinstance(device, bool) or device < 0:
        raise ValueError("device must be a nonnegative CUDA device ordinal")
    raise RuntimeError(
        "TE cannot recreate a lost JAX/XLA CUDA runtime. Restart the worker after "
        "context loss; use release_cuda_state while CUDA is usable for a "
        "context-preserving in-process restart. Native owners may use the common C API."
    )
