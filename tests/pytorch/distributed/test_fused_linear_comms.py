# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
import torch.distributed._symmetric_memory as symm_mem
import time
import argparse
import os
import uuid
import math


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description=(
            "Run a linear layer with Transformer Engine, CUDA Graphs, and Tensor Parallelism"
        )
    )
    parser.add_argument("--hidden_size", type=int, default=8192, help="Input feature size")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--fc_factor", type=int, default=4, help="MLP hidden layer factor")
    parser.add_argument(
        "--cuda_graph", action="store_true", help="Use CUDA Graphs (pass this flag to enable)"
    )
    parser.add_argument("--validate", action="store_true", help="Validate allreduce ubnext")
    parser.add_argument(
        "--comm_type",
        type=str,
        default="sym",
        help="Comm type: none,nccl,sym,ub,ubnext,ubnext_add_rms",
    )
    parser.add_argument(
        "--sym_type",
        type=str,
        default="multimem_all_reduce",
        help="pytorch sym type: one_shot, two_shot, multimem_all_reduce",
    )
    parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations")
    parser.add_argument(
        "--tp_size",
        type=int,
        default=None,
        help="Tensor parallelism size (defaults to number of GPUs)",
    )
    parser.add_argument("--eps", type=float, default=1e-5, help="Epsilon")
    args = parser.parse_args()

    # Check CUDA availability and get device count
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Test requires NVIDIA GPUs.")

    num_devices = torch.cuda.device_count()
    if num_devices == 0:
        raise RuntimeError("No CUDA devices found.")

    # Set tensor parallelism size
    tp_size = (
        args.tp_size if args.tp_size is not None else int(os.environ.get("WORLD_SIZE", num_devices))
    )

    # Initialize distributed environment for each GPU
    myrank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(torch.cuda.device_count())))
    num_nodes = world_size // local_size
    if num_nodes > 1:
        assert (
            "MASTER_ADDR" in os.environ
        ), "Multi-node run requires MASTER_ADDR to be set in the environment."
    # Set device
    device = torch.device(f"cuda:{local_rank}")
    # Initialize torch.distributed for tensor parallelism
    # Only set defaults if not already set by torchrun
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    torch.cuda.set_device(device)

    torch.distributed.init_process_group(
        backend="nccl", world_size=tp_size, rank=myrank % tp_size, device_id=device
    )
    torch.distributed.barrier(group=torch.distributed.group.WORLD)
    # Transformer Engine handles tensor parallelism internally when distributed is initialized
    # Set environment variable for tensor parallelism size
    os.environ["NVTE_TP_SIZE"] = str(tp_size)

    ub_cfgs = {
        "proj_fprop": {
            "method": "pipeline",
            "num_splits": 4,
            "is_reduce_scatter": True,
            "num_sm": 32,
            "atomic_gemm": False,
            "aggregate": False,
            "cga_size": 4,
            "set_sm_margin": True,
            "fp8_buf": False,
            "use_ce": False,
        },
        "fc1_fprop": {
            "method": "ring_exchange",
            "num_splits": 1,
            "is_reduce_scatter": False,
            "num_sm": 1,
            "atomic_gemm": False,
            "aggregate": False,
            "cga_size": 1,
            "set_sm_margin": False,
            "fp8_buf": False,
            "use_ce": True,
        },
    }

    # Initialize model with BF16 precision
    if os.environ.get("NVTE_USE_UB_FOR_UBNEXT") or args.comm_type == "ub":
        te.module.base.initialize_ub(
            [args.batch_size, args.hidden_size],
            tp_size,
            use_fp8=False,
            dtype=torch.bfloat16,
            bootstrap_backend="nccl",
            ub_cfgs=ub_cfgs,
        )

    proj = te.Linear(
        in_features=args.hidden_size // tp_size if args.comm_type == "none" else args.hidden_size,
        out_features=args.hidden_size,
        bias=False,
        device=device,
        params_dtype=torch.bfloat16,
        tp_size=tp_size if args.comm_type != "none" else 1,
        parallel_mode="row" if args.comm_type != "none" else None,
        tp_group=torch.distributed.group.WORLD if args.comm_type != "none" else None,
        symmetric_ar_type=args.sym_type if args.comm_type == "sym" else args.comm_type,
        sequence_parallel=args.comm_type == "ub",
        ub_overlap_rs=args.comm_type == "ub",
        ub_name="proj" if args.comm_type == "ub" else None,
    )

    fc1 = te.LayerNormLinear(
        in_features=args.hidden_size,
        out_features=(
            args.hidden_size * args.fc_factor // tp_size
            if args.comm_type == "none"
            else args.hidden_size * args.fc_factor
        ),
        bias=False,
        device=device,
        params_dtype=torch.bfloat16,
        eps=args.eps,
        zero_centered_gamma=False,
        normalization="RMSNorm",
        tp_size=tp_size if args.comm_type != "none" else 1,
        parallel_mode="column" if args.comm_type != "none" else None,
        symmetric_ar_type="ubnext_add_rms" if args.comm_type == "ubnext_add_rms" else None,
        tp_group=torch.distributed.group.WORLD if args.comm_type != "none" else None,
        sequence_parallel=args.comm_type == "ub",
        ub_overlap_ag=args.comm_type == "ub",
        ub_name="fc1" if args.comm_type == "ub" else None,
    )
    # Create CUDA stream
    stream = torch.cuda.Stream()
    # Check for environment variable to override pool size
    torch.distributed.barrier(group=torch.distributed.group.WORLD)
    torch.cuda.synchronize()

    for logbatch in range(int(math.log2(args.batch_size)) + 1):
        batch = 2**logbatch
        if args.comm_type == "ub":  # and batch < tp_size:
            batch = args.batch_size  # tp_size
        # Create input tensor
        torch.cuda.synchronize()
        torch.distributed.barrier(group=torch.distributed.group.WORLD)
        residual = torch.randn(
            batch // tp_size if args.comm_type == "ub" else batch,
            args.hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        inp = torch.randn(batch, args.hidden_size // tp_size, device=device, dtype=torch.bfloat16)

        # Warm-up run
        if not args.comm_type.startswith("ubnext_add"):
            out_proj = proj(inp)
            out_proj.add_(residual)
            out = fc1(out_proj)
        else:
            out = proj(inp)
            out._allocator.residual_global = residual
            out = fc1(out)
            # this also allocates distributed internal residual

        torch.cuda.synchronize()
        if args.cuda_graph:
            with torch.cuda.stream(stream):
                # Create CUDA Graph
                graph = torch.cuda.CUDAGraph()

                with torch.cuda.graph(graph):
                    if not args.comm_type.startswith("ubnext_add"):
                        out_proj = proj(inp)
                        out_proj.add_(residual)
                        out = fc1(out_proj)
                    else:
                        out = fc1(proj(inp))

            # Warm-up the graph
            for _ in range(5):
                graph.replay()

            torch.cuda.synchronize()

        torch.distributed.barrier(group=torch.distributed.group.WORLD)
        torch.distributed.barrier(group=torch.distributed.group.WORLD)
        torch.cuda.synchronize()

        # Measure time for forward passes
        start_time = time.time()
        with torch.cuda.stream(stream):
            for _ in range(args.iterations):
                if args.cuda_graph:
                    graph.replay()
                else:
                    if not args.comm_type.startswith("ubnext_add"):
                        out_proj = proj(inp)
                        out_proj.add_(residual)
                        out = fc1(out_proj)
                    else:
                        out = fc1(proj(inp))

            torch.cuda.synchronize()
        end_time = time.time()
        elapsed = end_time - start_time

        # Calculate and print elapsed time (only on rank 0)
        if myrank == 0:
            print(f"Batch{batch},{(elapsed/ args.iterations) * 1e6:.4f}")
        if args.cuda_graph:
            # needed or NCCL would hang
            del graph

    # Cleanup
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    # Generate a unique run ID for distributed initialization
    os.environ["RUN_ID"] = str(uuid.uuid4())
    main()
