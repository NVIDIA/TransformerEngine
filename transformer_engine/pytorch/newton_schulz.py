# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Distributed Newton-Schulz inverse square root via cuSolverMp."""

from typing import List, Optional

import torch
import torch.distributed as dist

import transformer_engine_torch as tex


def _get_nccl_comm_ptr(group: dist.ProcessGroup) -> int:
    """Extract the raw NCCL communicator pointer from a PyTorch process group."""
    backend = dist.get_backend(group)
    if backend != "nccl":
        raise RuntimeError(f"newton_schulz requires NCCL backend, got '{backend}'")
    nccl_backend = group._get_backend(torch.device("cuda"))
    return nccl_backend._comm_ptr()


def newton_schulz(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    num_iterations: int = 5,
    coefficients: Optional[List[float]] = None,
) -> None:
    """Compute Newton-Schulz inverse square root in-place on a distributed matrix.

    Parameters
    ----------
    x : torch.Tensor
        Local part of the distributed matrix (modified in-place).
        Must be a 2D CUDA tensor of type float32 or bfloat16.
    group : torch.distributed.ProcessGroup
        Process group with NCCL backend for distributed communication.
    num_iterations : int, optional
        Number of Newton-Schulz iterations. Default: 5.
    coefficients : list of float, optional
        Polynomial coefficients for the Newton-Schulz iteration.
    """
    QUINTIC_COEFFICIENTS = [
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
    if coefficients is None:
        coefficients = QUINTIC_COEFFICIENTS if num_iterations==5 else [1.5, -0.5, 0.0] * num_iterations
    assert len(coefficients) == num_iterations * 3, f"Unexpected number of coefficients: {len(coefficients)} for {num_iterations} iterations"

    if x.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {x.dim()}D")
    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA device")

    nccl_comm_ptr = _get_nccl_comm_ptr(group)
    nranks = dist.get_world_size(group)
    rank = dist.get_rank(group)

    # Global matrix dimensions
    m = x.size(0) * nranks  # rows are distributed across ranks
    n = x.size(1)

    ctx_ptr = tex.cusolvermp_ctx_create(nccl_comm_ptr, nranks, rank)
    try:
        tex.newton_schulz(ctx_ptr, m, n, x, num_iterations, coefficients)
    finally:
        tex.cusolvermp_ctx_destroy(ctx_ptr)
