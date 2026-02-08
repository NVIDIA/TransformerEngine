# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Distributed Newton-Schulz inverse square root via cuSolverMp."""

from typing import List, Optional

import torch
import torch.distributed as dist

import transformer_engine_torch as tex


# Default quintic polynomial coefficients for 5-iteration Newton-Schulz
# from cuSolverMp sample: (3069/1024, -7175/1024, 9009/1024, -6435/1024, 2835/2048)
_DEFAULT_COEFFICIENTS = [
    3069.0 / 1024.0,
    -7175.0 / 1024.0,
    9009.0 / 1024.0,
    -6435.0 / 1024.0,
    2835.0 / 2048.0,
]


def _get_nccl_comm_ptr(group: dist.ProcessGroup) -> int:
    """Extract the raw NCCL communicator pointer from a PyTorch process group."""
    backend = dist.get_backend(group)
    if backend != "nccl":
        raise RuntimeError(
            f"newton_schulz requires NCCL backend, got '{backend}'"
        )
    # Access the NCCL communicator via the internal _get_backend method
    nccl_backend = group._get_backend(torch.device("cuda"))
    comm = nccl_backend.get_nccl_comm()
    return comm


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
        Default: quintic polynomial coefficients from cuSolverMp sample.
    """
    if coefficients is None:
        coefficients = _DEFAULT_COEFFICIENTS

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
        tex.newton_schulz(ctx_ptr, x, num_iterations, coefficients)
    finally:
        tex.cusolvermp_ctx_destroy(ctx_ptr)
