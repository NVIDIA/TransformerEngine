# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Distributed Newton-Schulz test worker.

Launched via torchrun from test_newton_schulz.py.
"""

import argparse
import sys

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

from transformer_engine.pytorch.newton_schulz import (
    CusolverMpCtx,
    get_coefficients,
    newton_schulz,
)


def newton_schulz_reference(
    in_x: torch.Tensor, coefficients: list[tuple[float, float, float]]
) -> torch.Tensor:
    """Local Newton-Schulz reference mirroring the provided Octave update."""
    x = in_x.clone()
    for a, b, c in coefficients:
        xxt = x @ x.mT
        x = a * x + b * xxt @ x + c * xxt @ xxt @ x
    return x


@record
def main():
    parser = argparse.ArgumentParser(description="Newton-Schulz distributed test")
    parser.add_argument(
        "--check", type=str, default="orthogonality", choices=["orthogonality", "reference"]
    )
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--matrix-rows", type=int, default=256)
    parser.add_argument("--matrix-cols", type=int, default=None)
    parser.add_argument("--num-iterations", type=int, default=5)
    parser.add_argument("--coeff-type", type=str, default="quintic")
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    m = args.matrix_rows
    n = args.matrix_cols if args.matrix_cols is not None else args.matrix_rows
    coefficients = get_coefficients(args.num_iterations, args.coeff_type)

    # Ensure the distributed column dimension is divisible by world_size.
    assert n % world_size == 0, f"Matrix columns {n} must be divisible by world_size {world_size}"

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
        A = U @ torch.diag(singular_values) @ V.T
        A = A.to(dtype)
    else:
        A = torch.empty(m, n, device="cuda", dtype=dtype)

    # Broadcast the full matrix to all ranks
    dist.broadcast(A, src=0)

    # Scatter columns to each rank
    local_cols = n // world_size
    x_local = A[:, rank * local_cols : (rank + 1) * local_cols].contiguous()

    ctx = CusolverMpCtx(dist.group.WORLD)
    try:
        newton_schulz(x_local, ctx, args.num_iterations, coefficients=coefficients)
    finally:
        ctx.destroy()

    # Gather results
    gathered = [torch.empty_like(x_local) for _ in range(world_size)]
    dist.all_gather(gathered, x_local)
    X = torch.cat(gathered, dim=1)

    # Check: the resulting matrix should be orthogonal, or match a local reference.
    if rank == 0:
        if args.check == "orthogonality":
            if m <= n:
                gram = X @ X.t()
                expected = torch.eye(m, device=gram.device, dtype=gram.dtype)
                max_diff = (gram - expected).abs().max().item()
                print(f"Max |X @ X.t() - I|: {max_diff:.6e}", flush=True)
            else:
                gram = X.t() @ X
                expected = torch.eye(n, device=gram.device, dtype=gram.dtype)
                max_diff = (gram - expected).abs().max().item()
                print(f"Max |X.t() @ X - I|: {max_diff:.6e}", flush=True)
            passed = torch.allclose(gram, expected, atol=args.atol, rtol=args.rtol)
        else:
            reference = newton_schulz_reference(A.float(), coefficients).to(dtype)
            max_diff = (X - reference).abs().max().item()
            print(f"Max |distributed - reference|: {max_diff:.6e}", flush=True)
            passed = torch.allclose(X, reference, atol=args.atol, rtol=args.rtol)

        if passed:
            print("NUMERICAL CHECK PASSED", flush=True)
        else:
            print("NUMERICAL CHECK FAILED", flush=True, file=sys.stderr)
            sys.exit(1)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
