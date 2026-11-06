# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Explicit Transformer Engine cleanup at an externally serialized restart boundary."""

import sys

import torch
import transformer_engine_torch as tex

from ._cuda_state import check_release_boundary, release_device_caches, release_fp8_registrations


def release_cuda_state() -> None:
    """Release all TE-owned CUDA state at a process-wide restart boundary.

    Stop every TE caller, finish backward, and retire captured/compiled graphs
    and external communication owners first. All ranks perform collective
    cleanup before destroying borrowed process groups. Model tensors, optimizer
    state, FP8 numerical history, RNG and the framework runtime are retained.
    Only devices with an existing primary context are synchronized and visited.
    This never resets CUDA or the framework allocator.
    """
    check_release_boundary()
    if torch.cuda.is_initialized():
        devices = [
            device
            for device in range(torch.cuda.device_count())
            if torch._C._cuda_hasPrimaryContext(device)  # pylint: disable=protected-access
        ]
        # Check every device before mutating any cache or entering synchronization.
        for device in devices:
            with torch.cuda.device(device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Release CUDA state outside graph capture")
        for device in devices:
            torch.cuda.synchronize(device)
        torch.compiler.reset()
        from . import ep  # pylint: disable=import-outside-toplevel
        from . import distributed  # pylint: disable=import-outside-toplevel
        from .module import base  # pylint: disable=import-outside-toplevel

        if ep._BOOTSTRAPPED:  # pylint: disable=protected-access
            with torch.cuda.device(ep._BOOTSTRAP_DEVICE):  # pylint: disable=protected-access
                ep.ep_finalize()
        pool_device = distributed._SYMM_MEM_POOL_DEVICE  # pylint: disable=protected-access
        if pool_device is not None:
            with torch.cuda.device(pool_device):
                distributed.release_symm_mem_pool()
        if base._ub_initialized:  # pylint: disable=protected-access
            with torch.cuda.device(base._ub_device):  # pylint: disable=protected-access
                base.destroy_ub()
        for device in devices:
            with torch.cuda.device(device):
                release_fp8_registrations(device)
                release_device_caches(device)
    tex.release_cuda_state()
    backend = sys.modules.get("transformer_engine.common.CuTeDSL")
    if backend is not None:
        backend.release_cuda_state()


def reset_cuda_state() -> None:
    """Invalidate common state after loss of all TE contexts, before CUDA initialization.

    A CUDA runtime already initialized by PyTorch cannot be repaired by TE:
    allocator blocks, streams and generators are framework-owned. In that case
    fail before touching stale tensors; restart the worker after actual context
    loss. For a context-preserving restart use release_cuda_state instead.
    Native callers with their own recovered allocator may use the common C API.
    """
    if torch.cuda.is_initialized():
        raise RuntimeError(
            "TE cannot recover an initialized PyTorch CUDA runtime after context loss. "
            "Restart the worker, or call release_cuda_state while the context is usable "
            "for a context-preserving in-process restart."
        )
    tex.reset_cuda_state()
