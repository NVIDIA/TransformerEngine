# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Distributed Newton-Schulz test worker.

Launched via torchrun from test_newton_schulz.py.
"""

import argparse
import itertools
import os

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

from transformer_engine.pytorch.optimizers.newton_schulz import (
    CusolverMpCtx,
    get_coefficients,
    newton_schulz,
    newton_schulz_tp,
)


DTYPES = ("float32", "bfloat16")
COEFFICIENT_CONFIGS = ((5, "quintic"), (8, "polar_express"))


def newton_schulz_reference(
    in_x: torch.Tensor, coefficients: list[tuple[float, float, float]]
) -> torch.Tensor:
    """Local Newton-Schulz reference mirroring the provided Octave update."""
    x = in_x.clone()
    for a, b, c in coefficients:
        xxt = x @ x.mT
        x = a * x + b * xxt @ x + c * xxt @ xxt @ x
    return x


def _dtype_from_name(dtype: str) -> torch.dtype:
    if dtype == "float32":
        return torch.float32
    if dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")


def _test_tolerances(dtype: str) -> tuple[float, float]:
    return (5e-2, 5e-2) if dtype == "bfloat16" else (1e-2, 1e-2)


def _orthogonality_shapes(world_size: int) -> list[tuple[int, int]]:
    return [
        (world_size * 64, world_size * 64),
        (world_size * 64, world_size * 96),
        (world_size * 96, world_size * 64),
    ]


def _reference_shapes(world_size: int) -> list[tuple[int, int]]:
    return [(world_size * 64, world_size * 64)]


def _make_matrix(
    m: int,
    n: int,
    dtype: torch.dtype,
    rank: int,
) -> torch.Tensor:
    # Create a random matrix on rank 0 with singular values in (0, 1),
    # which keeps the Newton-Schulz iterations in the convergence regime.
    if rank == 0:
        torch.manual_seed(42)
        k = min(m, n)
        U, _ = torch.linalg.qr(
            torch.randn(m, k, device="cuda", dtype=torch.float32), mode="reduced"
        )
        V, _ = torch.linalg.qr(
            torch.randn(n, k, device="cuda", dtype=torch.float32), mode="reduced"
        )
        singular_values = torch.rand(k, device="cuda", dtype=torch.float32) * 0.8 + 0.1
        matrix = U @ torch.diag(singular_values) @ V.T
        matrix = matrix.to(dtype)
    else:
        matrix = torch.empty(m, n, device="cuda", dtype=dtype)

    dist.broadcast(matrix, src=0)
    return matrix


def _run_case(
    *,
    ctx: CusolverMpCtx,
    check: str,
    dtype_name: str,
    matrix_shape: tuple[int, int],
    num_iterations: int,
    coeff_type: str,
    api: str = "base",
    partition_dim: int = 1,
    tp_mode: str = "distributed",
) -> None:
    rank = ctx.rank
    world_size = ctx.nranks
    dtype = _dtype_from_name(dtype_name)
    m, n = matrix_shape
    coefficients = get_coefficients(num_iterations, coeff_type)
    atol, rtol = _test_tolerances(dtype_name)

    if api == "base" or partition_dim == 1:
        # Ensure the distributed column dimension is divisible by world_size.
        assert (
            n % world_size == 0
        ), f"Matrix columns {n} must be divisible by world_size {world_size}"
    else:
        assert m % world_size == 0, f"Matrix rows {m} must be divisible by world_size {world_size}"

    A = _make_matrix(m, n, dtype, rank)

    # Scatter columns for the base API. Scatter along partition_dim for the TP API.
    if api == "tp" and partition_dim == 0:
        local_rows = m // world_size
        x_local = A[rank * local_rows : (rank + 1) * local_rows, :].contiguous().clone()
        gather_dim = 0
    else:
        local_cols = n // world_size
        x_local = A[:, rank * local_cols : (rank + 1) * local_cols].contiguous().clone()
        gather_dim = 1

    if api == "tp":
        newton_schulz_tp(
            x_local,
            ctx,
            num_iterations,
            coefficients=coefficients,
            partition_dim=partition_dim,
            tp_mode=tp_mode,
        )
    else:
        newton_schulz(x_local, ctx, num_iterations, coefficients=coefficients)

    # Gather results
    gathered = [torch.empty_like(x_local) for _ in range(world_size)]
    dist.all_gather(gathered, x_local)
    X = torch.cat(gathered, dim=gather_dim)

    # Check: the resulting matrix should be orthogonal, or match a local reference.
    if check == "orthogonality":
        if m <= n:
            gram = X @ X.t()
            expected = torch.eye(m, device=gram.device, dtype=gram.dtype)
            label = "X @ X.t() - I"
        else:
            gram = X.t() @ X
            expected = torch.eye(n, device=gram.device, dtype=gram.dtype)
            label = "X.t() @ X - I"
        max_diff = (gram - expected).abs().max().item()
        passed = torch.allclose(gram, expected, atol=atol, rtol=rtol)
    elif check == "reference":
        reference = newton_schulz_reference(A.float(), coefficients).to(dtype)
        max_diff = (X - reference).abs().max().item()
        label = "distributed - reference"
        passed = torch.allclose(X, reference, atol=atol, rtol=rtol)
    else:
        raise ValueError(f"Unsupported check: {check}")

    if rank == 0:
        print(f"Max |{label}|: {max_diff:.6e}", flush=True)

    if not passed:
        raise AssertionError(
            "Newton-Schulz case failed: "
            f"check={check}, dtype={dtype_name}, matrix_shape={matrix_shape}, "
            f"num_iterations={num_iterations}, coeff_type={coeff_type}, api={api}, "
            f"partition_dim={partition_dim}, tp_mode={tp_mode}, max_diff={max_diff:.6e}"
        )


def run_all_tests(ctx: CusolverMpCtx) -> None:
    """Run all distributed Newton-Schulz checks in one torchrun invocation."""
    rank = ctx.rank
    world_size = ctx.nranks

    for config in itertools.product(
        DTYPES,
        _orthogonality_shapes(world_size),
        COEFFICIENT_CONFIGS,
    ):
        dtype_name, matrix_shape, (num_iterations, coeff_type) = config
        if rank == 0:
            print(f"Running orthogonality check with {config=}", flush=True)
        _run_case(
            ctx=ctx,
            check="orthogonality",
            dtype_name=dtype_name,
            matrix_shape=matrix_shape,
            num_iterations=num_iterations,
            coeff_type=coeff_type,
        )

    for config in itertools.product(
        DTYPES,
        _reference_shapes(world_size),
        COEFFICIENT_CONFIGS,
    ):
        dtype_name, matrix_shape, (num_iterations, coeff_type) = config
        if rank == 0:
            print(f"Running reference check with {config=}", flush=True)
        _run_case(
            ctx=ctx,
            check="reference",
            dtype_name=dtype_name,
            matrix_shape=matrix_shape,
            num_iterations=num_iterations,
            coeff_type=coeff_type,
        )

    for partition_dim, tp_mode in itertools.product((0, 1), ("duplicated", "distributed")):
        config = (partition_dim, tp_mode)
        if rank == 0:
            print(f"Running TP API reference check with {config=}", flush=True)
        _run_case(
            ctx=ctx,
            check="reference",
            dtype_name="float32",
            matrix_shape=_reference_shapes(world_size)[0],
            num_iterations=5,
            coeff_type="quintic",
            api="tp",
            partition_dim=partition_dim,
            tp_mode=tp_mode,
        )

    if rank == 0:
        print("NUMERICAL CHECK PASSED", flush=True)


@record
def main():
    parser = argparse.ArgumentParser(description="Newton-Schulz distributed test")
    parser.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # Initialize the NCCL communicator before passing its raw pointer to cuSolverMp.
    warmup = torch.ones((), device="cuda")
    dist.all_reduce(warmup, group=dist.group.WORLD)

    ctx = CusolverMpCtx(dist.group.WORLD)
    try:
        run_all_tests(ctx)
    finally:
        ctx.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
