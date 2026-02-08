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


def newton_schulz_reference(X, num_iterations, coefficients):
    """Pure PyTorch reference Newton-Schulz inverse square root.

    Uses the polynomial iteration: X_{k+1} = sum_j coeff[j] * X_k^(2j+1)
    for a quintic polynomial with 5 coefficients.
    """
    for _ in range(num_iterations):
        X2 = X @ X
        # Quintic polynomial: c0*X + c1*X^3 + c2*X^5 + c3*X^7 + c4*X^9
        # = X * (c0 + X2 * (c1 + X2 * (c2 + X2 * (c3 + X2 * c4))))
        result = coefficients[4]
        result = coefficients[3] + X2 * result
        result = coefficients[2] + X2 * result
        result = coefficients[1] + X2 * result
        result = coefficients[0] + X2 * result
        X = X @ result
    return X


@record
def main():
    parser = argparse.ArgumentParser(description="Newton-Schulz distributed test")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--matrix-size", type=int, default=256)
    parser.add_argument("--num-iterations", type=int, default=5)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    N = args.matrix_size

    # Ensure N is divisible by world_size
    assert N % world_size == 0, f"Matrix size {N} must be divisible by world_size {world_size}"

    # Default quintic polynomial coefficients
    coefficients = [
        3069.0 / 1024.0,
        -7175.0 / 1024.0,
        9009.0 / 1024.0,
        -6435.0 / 1024.0,
        2835.0 / 2048.0,
    ]

    # Create a random symmetric positive definite matrix on rank 0
    # A = Q @ diag(eigenvalues) @ Q^T with eigenvalues in (0, 1)
    # This ensures Newton-Schulz converges
    if rank == 0:
        torch.manual_seed(42)
        Q, _ = torch.linalg.qr(torch.randn(N, N, device="cuda", dtype=torch.float32))
        eigenvalues = torch.rand(N, device="cuda", dtype=torch.float32) * 0.8 + 0.1
        A = Q @ torch.diag(eigenvalues) @ Q.T
        A = A.to(dtype)
    else:
        A = torch.empty(N, N, device="cuda", dtype=dtype)

    # Broadcast the full matrix to all ranks
    dist.broadcast(A, src=0)

    # Compute reference on the full matrix
    A_ref = A.clone()
    result_ref = newton_schulz_reference(A_ref, args.num_iterations, coefficients)

    # Scatter rows to each rank
    local_rows = N // world_size
    x_local = A[rank * local_rows : (rank + 1) * local_rows, :].contiguous()

    # Run the distributed Newton-Schulz
    from transformer_engine.pytorch.newton_schulz import newton_schulz

    group = dist.group.WORLD
    newton_schulz(x_local, group, args.num_iterations, coefficients)

    # Gather results
    gathered = [torch.empty_like(x_local) for _ in range(world_size)]
    dist.all_gather(gathered, x_local)
    result_distributed = torch.cat(gathered, dim=0)

    # Check numerical accuracy on rank 0
    if rank == 0:
        max_diff = (result_distributed - result_ref).abs().max().item()
        rel_diff = max_diff / (result_ref.abs().max().item() + 1e-12)
        print(f"Max absolute diff: {max_diff:.6e}", flush=True)
        print(f"Max relative diff: {rel_diff:.6e}", flush=True)

        if torch.allclose(result_distributed, result_ref, atol=args.atol, rtol=args.rtol):
            print("NUMERICAL CHECK PASSED", flush=True)
        else:
            print("NUMERICAL CHECK FAILED", flush=True, file=sys.stderr)
            sys.exit(1)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
