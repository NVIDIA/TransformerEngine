# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Distributed Newton-Schulz matrix orthogonalization via cuSolverMp."""

from typing import List, Optional

import torch
import torch.distributed as dist

import transformer_engine_torch as tex

_CUSOLVERMP_REQUIRED = (
    "Newton-Schulz requires Transformer Engine to be built with NVTE_WITH_CUSOLVERMP=1"
)


class CusolverMpCtx:
    """cuSolverMp context for Newton-Schulz matrix orthogonalization.

    Context creation is expensive; create once and reuse across multiple
    :func:`newton_schulz` calls.  Call :meth:`destroy` when done, or use as a
    context manager::

        with te.cusolvermp_ctx_create(group) as ctx:
            te.newton_schulz(x, ctx)
    """

    def __init__(self, ptr: int, nranks: int) -> None:
        self._ptr = ptr
        self.nranks = nranks

    def destroy(self) -> None:
        """Destroy the underlying cuSolverMp context."""
        tex.cusolvermp_ctx_destroy(self._ptr)

    def __enter__(self) -> "CusolverMpCtx":
        return self

    def __exit__(self, *_) -> None:
        self.destroy()


def _get_nccl_comm_ptr(group: dist.ProcessGroup) -> int:
    """Extract the raw NCCL communicator pointer from a PyTorch process group."""
    backend = dist.get_backend(group)
    if backend != "nccl":
        raise RuntimeError(f"Newton-Schulz requires NCCL backend, got '{backend}'")
    nccl_backend = group._get_backend(torch.device("cuda"))
    return nccl_backend._comm_ptr()


def cusolvermp_ctx_create(group: dist.ProcessGroup) -> CusolverMpCtx:
    """Create a cuSolverMp context for Newton-Schulz matrix orthogonalization.

    Context creation is expensive; callers should create the context once and
    reuse it across multiple :func:`newton_schulz` calls.  The context must be
    destroyed with :meth:`CusolverMpCtx.destroy` (or used as a context manager)
    when it is no longer needed.

    Parameters
    ----------
    group : torch.distributed.ProcessGroup
        Process group with NCCL backend for distributed communication.

    Returns
    -------
    CusolverMpCtx
        Context to be passed to :func:`newton_schulz`.
    """
    if not hasattr(tex, "cusolvermp_ctx_create"):
        raise RuntimeError(_CUSOLVERMP_REQUIRED)
    nccl_comm_ptr = _get_nccl_comm_ptr(group)
    nranks = dist.get_world_size(group)
    rank = dist.get_rank(group)
    ptr = tex.cusolvermp_ctx_create(nccl_comm_ptr, nranks, rank)
    return CusolverMpCtx(ptr, nranks)


def get_coefficients(num_iterations: int) -> List[float]:
    """Return the default coefficient schedule for Newton-Schulz."""
    quintic_coefficients = [
        4.0848,
        -6.8946,
        2.9270,
        3.9505,
        -6.3029,
        2.6377,
        3.7418,
        -5.5913,
        2.3037,
        2.8769,
        -3.1427,
        1.2046,
        2.8366,
        -3.0525,
        1.2012,
    ]
    if num_iterations == 5:
        return quintic_coefficients
    return [1.5, -0.5, 0.0] * num_iterations


def newton_schulz(
    x: torch.Tensor,
    ctx: CusolverMpCtx,
    num_iterations: int = 5,
    coefficients: Optional[List[float]] = None,
) -> None:
    """Compute Newton-Schulz matrix orthogonalization in-place on a distributed matrix.

    Parameters
    ----------
    x : torch.Tensor
        Local part of the distributed matrix (modified in-place).
        Must be a 2D CUDA tensor of type float32 or bfloat16.
        Columns are distributed across ranks.
    ctx : CusolverMpCtx
        cuSolverMp context created by :func:`cusolvermp_ctx_create`.
    num_iterations : int, optional
        Number of Newton-Schulz iterations. Default: 5.
    coefficients : list of float, optional
        Polynomial coefficients for the Newton-Schulz iteration.
    """
    if coefficients is None:
        coefficients = get_coefficients(num_iterations)
    if len(coefficients) != num_iterations * 3:
        raise ValueError(
            f"Unexpected number of coefficients: {len(coefficients)} for"
            f" {num_iterations} iterations"
        )

    if x.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {x.dim()}D")
    if x.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"Expected float32 or bfloat16 tensor, got {x.dtype}")
    if not x.is_contiguous():
        raise ValueError("Input tensor must be contiguous")
    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA device")

    if not hasattr(tex, "newton_schulz"):
        raise RuntimeError(_CUSOLVERMP_REQUIRED)

    # Global matrix dimensions; columns are distributed across ranks.
    m = x.size(0)
    n = x.size(1) * ctx.nranks

    tex.newton_schulz(ctx._ptr, m, n, x, num_iterations, coefficients)
