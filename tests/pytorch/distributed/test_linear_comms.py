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
    parser.add_argument("--in_features", type=int, default=8192, help="Input feature size")
    parser.add_argument("--out_features", type=int, default=8192, help="Output feature size")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument(
        "--cuda_graph", action="store_true", help="Use CUDA Graphs (pass this flag to enable)"
    )
    parser.add_argument("--validate", action="store_true", help="Validate allreduce ubnext")
    parser.add_argument("--comm_type", type=str, default="sym", help="Comm type: nccl,sym,ub")
    parser.add_argument(
        "--sym_type",
        type=str,
        default="multimem_all_reduce",
        help="sym type: one_shot, two_shot, multimem_all_reduce, ubnext",
    )
    parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations")
    parser.add_argument(
        "--tp_size",
        type=int,
        default=None,
        help="Tensor parallelism size (defaults to number of GPUs)",
    )
    parser.add_argument("--eps", type=float, default=1e-5, help="Epsilon")
    parser.add_argument("--rmsnorm", action="store_true", help="Use RMSNorm")
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
            "num_splits": 1,
            "is_reduce_scatter": True,
            "num_sm": 32,
            "atomic_gemm": False,
            "aggregate": False,
            "cga_size": 4,
            "set_sm_margin": False,
            "fp8_buf": False,
            "use_ce": False,
        }
    }

    # Initialize model with BF16 precision

    modelseq = te.Linear(
        in_features=int(args.in_features / tp_size),
        out_features=args.out_features,
        bias=False,
        device=device,
        params_dtype=torch.bfloat16,
    )

    modelnorm = te.RMSNorm(
        normalized_shape=int(args.out_features),
        eps=args.eps,
        device=device,
        dtype=torch.bfloat16,
        zero_centered_gamma=False,
    )
    residual = (
        torch.randn(args.batch_size, args.out_features, device=device, dtype=torch.bfloat16)
        if args.rmsnorm
        else None
    )

    ln_weight = modelnorm.weight.data if args.rmsnorm else None
    if (
        args.comm_type == "sym" and os.environ.get("NVTE_USE_UB_FOR_UBNEXT")
    ) or args.comm_type == "ub":
        te.module.base.initialize_ub(
            [args.batch_size, args.out_features],
            tp_size,
            use_fp8=False,
            dtype=torch.bfloat16,
            bootstrap_backend="nccl",
            ub_cfgs=ub_cfgs,
        )

    modelpar = None

    if args.comm_type == "sym" or args.comm_type == "nccl":
        modelpar = te.Linear(
            in_features=args.in_features,
            out_features=args.out_features,
            bias=False,
            device=device,
            params_dtype=torch.bfloat16,
            tp_size=tp_size,
            parallel_mode="row",
            tp_group=torch.distributed.group.WORLD,
            symmetric_ar_type=args.sym_type if args.comm_type == "sym" else None,
        )

    if args.comm_type == "ub":
        modelpar = te.Linear(
            in_features=args.in_features,
            out_features=args.out_features,
            bias=False,
            device=device,
            params_dtype=torch.bfloat16,
            tp_size=tp_size,
            parallel_mode="row",
            tp_group=torch.distributed.group.WORLD,
            sequence_parallel=True,
            ub_overlap_rs=True,
            ub_name="proj",
        )

    # Create CUDA stream
    stream = torch.cuda.Stream()
    # Check for environment variable to override pool size

    allocator = None
    if args.comm_type == "sym" and args.validate:
        pool_size = int(os.environ.get("NVTE_UB_SYMM_POOL_SIZE", 64)) * 1024 * 1024
        allocator = te.cpp_extensions.symm_allocator.SymmAllocator(
            pool_size, torch.device(device), torch.distributed.group.WORLD
        )

    # Run tensor comparison tests only for symmetric communication
    if args.comm_type == "sym" and args.validate:

        if args.rmsnorm:
            torch.manual_seed(57)
            torch.cuda.manual_seed(57)
            residual = torch.randn(1, args.out_features, dtype=torch.bfloat16, device=device)
            t = allocator.create_tensor(
                (
                    1,
                    args.out_features,
                ),
                dtype=torch.bfloat16,
            )
            # te.cpp_extensions.symm_allocator.ubsymm_free_residual(t)
            t.fill_(myrank)
            t_in = t.clone()
            torch.distributed.all_reduce(t_in)
            t_in.add_(residual)
            out1 = modelnorm(t_in)
            out2 = allocator.allreduce_simple(
                t,
                hidden_size=args.out_features,
                residual_in=residual,
                residual_out=residual,
                fuse_layernorm=True,
                eps=args.eps,
                gamma=modelnorm.weight.data,
            )
            abs_diff = torch.abs(out1 - out2)
            max_delta = torch.max(abs_diff).item()
            num_different = torch.sum(out1 != out2).item()
            print(f"FUSED RMSNorm Max delta: {max_delta}, num different: {num_different}")
            if myrank == 0:
                print(f"gamma: {modelnorm.weight.data}")
                print(f"FUSED RMSNorm output: {out1}")
                print(f"FUSED RMSNorm output: {out2}")

        # Test different tensor sizes from 64 to 1024*1024 elements
        all_max_deltas = []
        all_num_different = []
        all_total_elements = []
        all_sizes = []

        size = 64
        while size <= 1024 * 1024:
            # Allocate tensors
            t = allocator.create_tensor((size,), dtype=torch.bfloat16)
            t.fill_(0)
            t += torch.randn_like(t)  # Add random noise to each element
            tmain = t.clone()  # Create a copy since allreduce operates in-place
            torch.distributed.all_reduce(tmain)
            tlamport = allocator.allreduce_lamport(t)

            # Compare the two tensors
            abs_diff = torch.abs(tlamport - tmain)
            max_delta = torch.max(abs_diff).item()
            num_different = torch.sum(tlamport != tmain).item()

            # Store statistics
            all_max_deltas.append(max_delta)
            all_num_different.append(num_different)
            all_total_elements.append(tlamport.numel())
            all_sizes.append(size)

            # Free tensor (memory returned to pool)
            del t, tlamport, tmain, abs_diff

            # Double the size for next iteration
            size *= 2

        # Print summary statistics
        if myrank == 0:
            print("\n=== Tensor Comparison Summary ===")
            total_elements_tested = sum(all_total_elements)
            total_different_elements = sum(all_num_different)
            overall_max_delta = max(all_max_deltas)

            print(
                f"Tested sizes: {len(all_sizes)} different tensor sizes from {all_sizes[0]} to"
                f" {all_sizes[-1]} elements"
            )
            print(f"Total elements tested: {total_elements_tested}")
            print(f"Total different elements: {total_different_elements}")
            print(
                "Overall error rate:"
                f" {(total_different_elements / total_elements_tested) * 100:.6f}%"
            )
            print(f"Maximum delta across all tests: {overall_max_delta}")

            if total_different_elements > 0 or overall_max_delta > 0:
                print("\nPer-size breakdown:")
                for i, size in enumerate(all_sizes):
                    error_rate = (all_num_different[i] / all_total_elements[i]) * 100
                    print(
                        f"  Size {size:7d}:"
                        f" {all_num_different[i]:6d}/{all_total_elements[i]:7d} different"
                        f" ({error_rate:6.3f}%), max_delta: {all_max_deltas[i]:.6f}"
                    )
            print("================================\n")

    torch.distributed.barrier(group=torch.distributed.group.WORLD)
    torch.cuda.synchronize()

    for logbatch in range(int(math.log2(args.batch_size)) + 1):
        batch = 2**logbatch
        if args.comm_type == "ub" and batch < tp_size:
            batch = tp_size
        # Create input tensor
        inp = torch.randn(
            batch, int(args.in_features / tp_size), device=device, dtype=torch.bfloat16
        )
        # Warm-up run
        out = modelseq(inp)
        modelnorm(out)
        modelpar(inp)
        torch.cuda.synchronize()
        if args.cuda_graph:
            with torch.cuda.stream(stream):
                # Create CUDA Graph
                gseq = torch.cuda.CUDAGraph()
                gpar = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gseq):
                    output = modelseq(inp)
                    if args.rmsnorm:
                        output.add_(residual[:batch, : args.out_features])
                        output = modelnorm(output)
                with torch.cuda.graph(gpar):
                    output = modelpar(inp)
            # Warm-up the graph
            for _ in range(5):
                gseq.replay()
                gpar.replay()
            torch.cuda.synchronize()

        torch.distributed.barrier(group=torch.distributed.group.WORLD)
        torch.distributed.barrier(group=torch.distributed.group.WORLD)
        torch.cuda.synchronize()

        # Measure time for forward passes
        start_time = time.time()
        with torch.cuda.stream(stream):
            for _ in range(args.iterations):
                if args.cuda_graph:
                    gseq.replay()
                else:
                    modelseq(inp)

            torch.cuda.synchronize()
        end_time = time.time()
        seq_elapsed = end_time - start_time

        torch.distributed.barrier(group=torch.distributed.group.WORLD)
        torch.distributed.barrier(group=torch.distributed.group.WORLD)
        torch.cuda.synchronize()

        # Measure time for forward passes
        start_time = time.time()
        with torch.cuda.stream(stream):
            for _ in range(args.iterations):
                if args.cuda_graph:
                    gpar.replay()
                else:
                    modelpar(inp)

        torch.cuda.synchronize()
        end_time = time.time()
        par_elapsed = end_time - start_time
        nccl_elapsed = par_elapsed - seq_elapsed
        # Calculate and print elapsed time (only on rank 0)
        if myrank == 0:
            print(
                f"Batch{batch},{(seq_elapsed/ args.iterations) * 1e6:.4f}us,{(par_elapsed/ args.iterations) * 1e6:.4f} us,{(nccl_elapsed/ args.iterations) * 1e6:.4f}"
            )
        if args.cuda_graph:
            # needed or NCCL would hang
            del gseq, gpar

    # Cleanup
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    # Generate a unique run ID for distributed initialization
    os.environ["RUN_ID"] = str(uuid.uuid4())
    main()
