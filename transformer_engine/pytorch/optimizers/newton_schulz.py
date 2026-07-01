# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Distributed Newton-Schulz matrix orthogonalization via cuSolverMp."""

from itertools import chain, cycle, islice, repeat
from typing import Iterator, Literal, Optional, Sequence

import torch
import torch.distributed as dist

import transformer_engine_torch as tex


_COEFFICIENT_SETS = {
    # Values are rounded to closest representable in single precision.
    "simple": [
        (3.4445, -4.7750, 2.0315),
    ],
    "quintic": [
        # optimized for a quintic iteration.
        # Source: https://leloykun.github.io/ponder/muon-opt-coeffs/#how-do-we-optimize-the-coefficients
        # Numbers from: https://github.com/KellerJordan/modded-nanogpt/blob/0674386070ceb4dcd207e1aca747ffcea6c15250/train_gpt_medium.py#L45
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ],
    "polar_express": [
        # Polar Express iteration from: https://arxiv.org/abs/2505.16932
        # We include PolarExpress' division by 1.01^polynomial_degree (as stated in their Algorithm 1) in the coefficient list.
        # This is a safety factor for numerical stability.
        (8.2051, -22.9019, 16.4607),
        (4.0664, -2.8612, 0.5184),
        (3.9096, -2.8234, 0.5250),
        (3.2856, -2.4153, 0.4853),
        (2.2779, -1.6198, 0.3985),
        (1.8726, -1.2307, 0.3585),
        (1.8564, -1.2132, 0.3568),
        (1.8750, -1.2500, 0.3750),
    ],
    "cans": [
        # CANS from: http://arxiv.org/abs/2506.10935
        # CANS iteration (Remez + adaptive interval) based coefficients.
        # Source (for generating CANS coefficients): https://github.com/GrishKate/accelerating_orthogonalization/blob/main/polynomials.py
        (8.4703, -25.1081, 18.6293),
        (4.1828, -3.1087, 0.5806),
        (3.9619, -2.9541, 0.5630),
        (3.2866, -2.4647, 0.5074),
        (2.2737, -1.6447, 0.4162),
    ],
    "aol": [
        # from https://github.com/thib-s/flash-newton-schulz/blob/main/newton_schulz_triton.py#L511
        (4.0098, -7.0585, 2.4635),
        (3.4585, -5.5479, 2.5959),
        (2.7573, -3.2939, 1.4254),
        (2.7215, -3.0494, 1.3169),
    ],
}

NSCoeffT = Literal[_COEFFICIENT_SETS.keys()]

CoeffIterMode = Literal["cycle", "repeat_last"]
CoeffT = tuple[float, float, float]


def get_coefficient_iterator(
    steps: int,
    coefficient_sets: Sequence[CoeffT],
    mode: CoeffIterMode = "cycle",
) -> Iterator[CoeffT]:
    """Iterate through coefficient sets with configurable end behavior using itertools.

    Args:
        steps: The number of tuples to yield.
        coefficient_sets: A sequence of (a, b, c) coefficient tuples.
        mode: Iteration mode:
            - "cycle": After the last element, restart from the beginning.
            - "repeat_last": After the last element, keep yielding the last tuple.

    Yields:
        Tuples (a, b, c) from coefficient_sets according to the specified mode.

    Raises:
        ValueError: If coefficient_sets is empty.
        ValueError: If an invalid mode is provided.
    """
    if not coefficient_sets:
        raise ValueError("coefficient_sets must be non-empty.")

    base: Iterator[CoeffT]
    if mode == "cycle":
        base = cycle(coefficient_sets)
    elif mode == "repeat_last":
        # Chain the original list with an infinite repeat of the last item
        base = chain(coefficient_sets, repeat(coefficient_sets[-1]))
    else:
        raise ValueError(f"Invalid mode: {mode}. Expected 'cycle' or 'repeat_last'.")

    return islice(base, steps)


def get_coefficients(steps: int, coefficient_type: NSCoeffT = "quintic") -> list[CoeffT]:
    """Return the coefficient schedule for Newton-Schulz.

    Parameter ``coefficient_type`` can be one of the following
      - "simple": Default coefficient set.
      - "quintic": Quintic iteration with optimized coefficients.
      - "polar_express": Polar Express iteration with optimized coefficients.
      - "cans": CANS iteration with Remez + adaptive interval coefficients.
      - "aol": AOL coefficient set.
    """
    if coefficient_type not in _COEFFICIENT_SETS:
        raise ValueError("Invalid coefficient type: " + coefficient_type)
    iter_mode: CoeffIterMode = (
        "repeat_last" if coefficient_type in ("polar_express", "cans") else "cycle"
    )
    coeff_iter = get_coefficient_iterator(
        steps, _COEFFICIENT_SETS[coefficient_type], mode=iter_mode
    )
    return list(coeff_iter)


class CusolverMpCtx:
    """cuSolverMp context for Newton-Schulz matrix orthogonalization.

    Context creation is expensive; create once and reuse across multiple
    :func:`newton_schulz` calls.  Call :meth:`destroy` when done.
    """

    def __init__(self, group: dist.ProcessGroup) -> None:
        self.group = group
        self.nranks = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self._ptr = tex.cusolvermp_ctx_create(_get_nccl_comm_ptr(group), self.nranks, self.rank)

    def destroy(self) -> None:
        """Destroy the underlying cuSolverMp context."""
        if self._ptr is not None:
            tex.cusolvermp_ctx_destroy(self._ptr)
            self._ptr = None

    def __del__(self) -> None:
        # Called when the context is manually destroyed or during Python teardown
        self.destroy()


def _get_nccl_comm_ptr(group: dist.ProcessGroup) -> int:
    """Extract the raw NCCL communicator pointer from a PyTorch process group."""
    backend = dist.get_backend(group)
    if backend != "nccl":
        raise RuntimeError(f"Newton-Schulz requires NCCL backend, got '{backend}'")
    nccl_backend = group._get_backend(torch.device("cuda"))
    return nccl_backend._comm_ptr()


def newton_schulz(
    x: torch.Tensor,
    ctx: CusolverMpCtx,
    num_iterations: int = 5,
    coefficients: Optional[Sequence[CoeffT]] = None,
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
    coefficients : sequence of tuple[float, float, float], optional
        Polynomial coefficients for the Newton-Schulz iteration.
    """
    if coefficients is None:
        coefficients = get_coefficients(num_iterations)
    if len(coefficients) != num_iterations:
        raise ValueError(
            f"Unexpected number of coefficients: {len(coefficients)} for"
            f" {num_iterations} iterations"
        )
    flat_coefficients: list[float] = []
    for i, coeff in enumerate(coefficients):
        if len(coeff) != 3:
            raise ValueError(
                f"Expected coefficient tuple of length 3 at iteration {i}, got {len(coeff)}"
            )
        flat_coefficients.extend(coeff)

    if x.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {x.dim()}D")
    if x.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"Expected float32 or bfloat16 tensor, got {x.dtype}")
    if not x.is_contiguous():
        raise ValueError("Input tensor must be contiguous")
    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA device")

    # Global matrix dimensions; columns are distributed across ranks.
    m = x.size(0)
    n = x.size(1) * ctx.nranks

    tex.newton_schulz(ctx._ptr, m, n, x, num_iterations, flat_coefficients)


def _orthogonalize_replicated(
    x: torch.Tensor,
    ctx: CusolverMpCtx,
    num_iterations: int,
    coefficients: Optional[Sequence[CoeffT]],
    transpose: bool,
) -> torch.Tensor:
    """Orthogonalize a replicated matrix using the distributed column-sharded kernel."""
    work = x.mT.contiguous() if transpose else x
    if work.size(1) % ctx.nranks != 0:
        distributed_dim = 0 if transpose else 1
        raise ValueError(
            f"Tensor dimension {distributed_dim} with size {x.size(distributed_dim)} "
            f"must be divisible by tensor-parallel size {ctx.nranks}"
        )

    local_work = work.chunk(ctx.nranks, dim=1)[ctx.rank].contiguous()
    newton_schulz(local_work, ctx, num_iterations, coefficients=coefficients)

    output_shards = [torch.empty_like(local_work) for _ in range(ctx.nranks)]
    dist.all_gather(output_shards, local_work, group=ctx.group)
    output = torch.cat(output_shards, dim=1)
    return output.mT.contiguous() if transpose else output


def newton_schulz_tp(
    x: torch.Tensor,
    ctx: CusolverMpCtx,
    num_iterations: int = 5,
    coefficients: Optional[Sequence[CoeffT]] = None,
    partition_dim: Optional[int] = None,
    tp_mode: Literal["duplicated", "distributed"] = "duplicated",
) -> None:
    """Compute tensor-parallel Newton-Schulz orthogonalization in-place.

    This convenience wrapper handles replicated tensors and tensor-parallel
    shards while delegating the matrix orthogonalization to :func:`newton_schulz`.
    The underlying kernel expects columns to be distributed across ranks, so row
    partitions are transposed before the call and transposed back afterward.

    Parameters
    ----------
    x : torch.Tensor
        Local tensor to orthogonalize in-place.
    ctx : CusolverMpCtx
        cuSolverMp context created for the tensor-parallel process group.
    num_iterations : int, optional
        Number of Newton-Schulz iterations. Default: 5.
    coefficients : sequence of tuple[float, float, float], optional
        Polynomial coefficients for the Newton-Schulz iteration.
    partition_dim : int, optional
        Dimension along which ``x`` is partitioned. ``None`` treats ``x`` as a
        duplicated full tensor on every rank.
    tp_mode : {"duplicated", "distributed"}, optional
        ``"distributed"`` orthogonalizes the existing partition directly.
        ``"duplicated"`` first gathers the full tensor, orthogonalizes it, and
        copies this rank's partition back into ``x``.
    """
    if x.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {x.dim()}D")

    if partition_dim is not None:
        if partition_dim not in (0, 1):
            raise ValueError(f"Invalid partition_dim: {partition_dim}")
        if tp_mode not in ("duplicated", "distributed"):
            raise ValueError(f"Invalid tp_mode: {tp_mode}")

    if x.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"Expected float32 or bfloat16 tensor, got {x.dtype}")
    if not x.is_contiguous():
        raise ValueError("Input tensor must be contiguous")
    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA device")

    if partition_dim is None:
        output = _orthogonalize_replicated(
            x,
            ctx,
            num_iterations,
            coefficients,
            transpose=x.size(0) > x.size(1),
        )
        x.copy_(output)
        return

    if tp_mode == "duplicated":
        x_shards = [torch.empty_like(x) for _ in range(ctx.nranks)]
        dist.all_gather(x_shards, x, group=ctx.group)
        global_x = torch.cat(x_shards, dim=partition_dim)

        output = _orthogonalize_replicated(
            global_x,
            ctx,
            num_iterations,
            coefficients,
            transpose=global_x.size(0) > global_x.size(1),
        )

        local_start = ctx.rank * x.size(partition_dim)
        local_output = output.narrow(partition_dim, local_start, x.size(partition_dim))
        x.copy_(local_output)
    elif tp_mode == "distributed":
        if partition_dim == 0:
            x_t = x.mT.contiguous()
            newton_schulz(x_t, ctx, num_iterations, coefficients=coefficients)
            x.copy_(x_t.mT)
        else:
            newton_schulz(x, ctx, num_iterations, coefficients=coefficients)
    else:
        raise ValueError(f"Invalid tp_mode: {tp_mode}")
