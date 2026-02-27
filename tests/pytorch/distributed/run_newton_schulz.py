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

    # Scatter rows to each rank
    local_rows = N // world_size
    x_local = A[rank * local_rows : (rank + 1) * local_rows, :].contiguous()

    # Run the distributed Newton-Schulz
    from transformer_engine.pytorch.newton_schulz import newton_schulz

    group = dist.group.WORLD
    newton_schulz(x_local, group, args.num_iterations)

    # Gather results
    gathered = [torch.empty_like(x_local) for _ in range(world_size)]
    dist.all_gather(gathered, x_local)
    X = torch.cat(gathered, dim=0)

    # Check: the resulting matrix should be orthogonal
    if rank == 0:
        XXT = X @ X.t()
        I = torch.eye(N, device=XXT.device, dtype=XXT.dtype)
        max_diff = (XXT - I).abs().max().item()
        print(f"Max |X @ X.t() - I|: {max_diff:.6e}", flush=True)

        if torch.allclose(XXT, I, atol=args.atol, rtol=args.rtol):
            print("NUMERICAL CHECK PASSED", flush=True)
        else:
            print("NUMERICAL CHECK FAILED", flush=True, file=sys.stderr)
            sys.exit(1)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
