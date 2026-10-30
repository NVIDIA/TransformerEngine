# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Explicit Transformer Engine cleanup at an externally serialized restart boundary."""

import sys

import torch
import transformer_engine_torch as tex

from ._cuda_state import check_release_boundary, release_device_caches, release_fp8_registrations


def _check_device(device):
    if not isinstance(device, int) or isinstance(device, bool) or device < 0:
        raise ValueError("device must be a nonnegative CUDA device ordinal")


def release_cuda_state(device: int) -> None:
    """Release TE-owned state while CUDA and borrowed communicators are usable.

    Stop every TE caller, finish backward, and destroy captured/compiled graphs
    and external communication owners first. With EP this is a coordinated
    cleanup on every participating rank, before destroying process groups.
    Model parameters, optimizer state, FP8 numeric history, RNG and the framework
    runtime are retained. The process-wide Torch compilation cache is cleared.
    Workspaces and library handles are recreated lazily on their next use.
    This function never resets the CUDA context or the framework allocator.
    """
    _check_device(device)
    check_release_boundary()
    if torch.cuda.is_initialized():
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Release CUDA state outside graph capture")
            # Fail before changing ownership if the runtime cannot drain its work.
            torch.cuda.synchronize(device)
            torch.compiler.reset()
            from . import ep  # pylint: disable=import-outside-toplevel
            from . import distributed  # pylint: disable=import-outside-toplevel
            from .module import base  # pylint: disable=import-outside-toplevel

            if (
                ep._BOOTSTRAPPED and ep._BOOTSTRAP_DEVICE == device
            ):  # pylint: disable=protected-access
                ep.ep_finalize()
            pool_device = distributed._SYMM_MEM_POOL_DEVICE  # pylint: disable=protected-access
            if pool_device is not None and torch.device(pool_device).index == device:
                distributed.release_symm_mem_pool()
            if (
                base._ub_initialized and base._ub_device == device
            ):  # pylint: disable=protected-access
                base.destroy_ub()
            release_fp8_registrations(device)
            release_device_caches(device)
    tex.release_cuda_state(device)
    backend = sys.modules.get("transformer_engine.common.CuTeDSL")
    if backend is not None:
        backend.release_cuda_state(device)


def reset_cuda_state(device: int) -> None:
    """Invalidate common state after context loss, before PyTorch CUDA initialization.

    A CUDA runtime already initialized by PyTorch cannot be repaired by TE:
    allocator blocks, streams and generators are framework-owned. In that case
    fail before touching stale tensors; restart the worker after actual context
    loss. For a context-preserving restart use release_cuda_state instead.
    Native callers with their own recovered allocator may use the common C API.
    """
    _check_device(device)
    if torch.cuda.is_initialized():
        raise RuntimeError(
            "TE cannot recover an initialized PyTorch CUDA runtime after context loss. "
            "Restart the worker, or call release_cuda_state while the context is usable "
            "for a context-preserving in-process restart."
        )
    tex.reset_cuda_state(device)
