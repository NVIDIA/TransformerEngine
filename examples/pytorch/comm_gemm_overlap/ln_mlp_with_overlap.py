#!/usr/bin/python3

# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import socket
import argparse
import warnings

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def _parse_args(argv=None, namespace=None):
    parser = argparse.ArgumentParser(
        description="Test a te.LayerNormMLP module with GEMM+comm overlap via Userbuffers."
    )
    parser.add_argument(
        "-i", "--num-iters", type=int, default=5, help="Number of dummy 'training' iterations."
    )
    parser.add_argument("-b", "--batch-size", type=int, default=2, help="Input batch size.")
    parser.add_argument("-s", "--seq-length", type=int, default=2048, help="Input sequence length.")
    parser.add_argument(
        "-n", "--num-heads", type=int, default=64, help="Number of attention heads."
    )
    parser.add_argument(
        "-d", "--head-dim", type=int, default=128, help="Dimension of each attention head."
    )
    parser.add_argument(
        "--mlp-expansion-factor",
        type=int,
        default=4,
        help="MLP block intermediate size as a factor of hidden dimension.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed.")
    parser.add_argument(
        "--fp8", action="store_true", default=False, help="Enables the te.fp8_autocast() context."
    )
    parser.add_argument(
        "--no-comm-overlap",
        action="store_true",
        default=False,
        help="Disable the comm+GEMM overlap.",
    )
    parser.add_argument(
        "--num-replicas", type=int, default=1, help="Number of data-parallel model replicas."
    )
    parser.add_argument(
        "--tcp-init",
        action="store_true",
        default=False,
        help="Initialize torch.distributed with TcpStore.",
    )
    parser.add_argument(
        "--bind-to-device",
        action="store_true",
        default=False,
        help="Initialize torch.distributed with `device_id` to bind each rank to a single device.",
    )
    parser.add_argument(
        "--bootstrap-backend",
        type=str.lower,
        default="nccl",
        choices=["gloo", "mpi", "nccl"],
        help="Communications backend for host tensor collectives during Userbuffers bootstrapping.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print out from every rank instead of just the root rank of relevant process groups.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Print out additional debug information.",
    )
    args = parser.parse_args(argv, namespace)
    return args


def _train(opts):
    if "OMPI_COMM_WORLD_SIZE" in os.environ:
        # Execution with `mpirun -np N`
        WORLD_RANK = int(os.getenv("OMPI_COMM_WORLD_RANK", "0"))
        WORLD_SIZE = int(os.getenv("OMPI_COMM_WORLD_SIZE", "1"))
        LOCAL_RANK = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
        LOCAL_SIZE = int(os.getenv("OMPI_COMM_WORLD_LOCAL_SIZE", "1"))
        opts.tcp_init = True
        opts.bind_to_device = True
        opts.bootstrap_backend = "mpi"
    elif "TORCHELASTIC_RUN_ID" in os.environ:
        WORLD_RANK = int(os.getenv("RANK", "0"))
        WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
        LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
        LOCAL_SIZE = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
    else:
        raise RuntimeError(f"{__file__} must be launched with either `mpirun` or `torchrun`!")
    NUM_NODES = WORLD_SIZE // LOCAL_SIZE

    def dist_print(msg, group=None, end="\n", debug=False):
        if debug and not opts.debug:
            return
        group = dist.new_group() if group is None else group
        group_rank = dist.get_rank(group)
        group_size = dist.get_world_size(group)
        all_ranks = dist.get_process_group_ranks(group)
        ranks_skip = all_ranks[1] - all_ranks[0] > 1
        group_id = WORLD_RANK % group_size if ranks_skip else WORLD_RANK // group_size
        if group_rank == 0 or opts.verbose:
            print(f"[rank{WORLD_RANK}:node{group_id}] {msg}{end}", end="", flush=True)
        dist.barrier(group)

    # Initialize torch.distributed global process group and get DP/TP groups
    torch.cuda.set_device(LOCAL_RANK)
    dist_init_kwargs = {
        "backend": "nccl",
        "rank": WORLD_RANK,
        "world_size": WORLD_SIZE,
    }
    if opts.tcp_init or NUM_NODES > 1:
        if NUM_NODES > 1:
            assert (
                "MASTER_ADDR" in os.environ
            ), "Multi-node run requires MASTER_ADDR to be set in the environment."
        MASTER_ADDR = os.getenv("MASTER_ADDR", socket.gethostbyname(socket.gethostname()))
        MASTER_PORT = os.getenv("MASTER_PORT", "1234")
        dist_init_kwargs["init_method"] = f"tcp://{MASTER_ADDR}:{MASTER_PORT}"
    if opts.bind_to_device or opts.bootstrap_backend == "nccl":
        dist_init_kwargs["device_id"] = torch.device(f"cuda:{LOCAL_RANK}")
    assert dist.is_nccl_available()
    dist.init_process_group(**dist_init_kwargs)
    nccl_world = dist.new_group(backend="nccl")
    dist_print(f"Initialized default NCCL process group with {WORLD_RANK} GPUs", nccl_world)

    # Figure out process groups for tensor- and data-parallelism (if any)
    if NUM_NODES > 1:
        # Create a list of world ranks on this node
        hostnames = [None for _ in range(WORLD_SIZE)]
        hostname = socket.gethostname()
        dist.all_gather_object(hostnames, hostname)
        node_ranks = []
        for i, host in enumerate(hostnames):
            if host == hostname:
                node_ranks.append(i)

        if opts.num_replicas > 1:
            # Split node ranks into multiple replicas
            assert len(node_ranks) % opts.num_replicas == 0
            tp_size = len(node_ranks) // opts.num_replicas
            found_replica = False
            for replica in range(opts.num_replicas):
                start = replica * tp_size
                end = start + tp_size
                tp_ranks = node_ranks[start:end]
                if WORLD_RANK in tp_ranks:
                    found_replica = True
                    break
            assert found_replica
        else:
            # The entire node is the tensor-parallel group
            tp_ranks = node_ranks

        tp_group = dist.new_group(backend="nccl", ranks=tp_ranks)
        tp_size = dist.get_world_size(tp_group)
        tp_rank = dist.get_rank(tp_group)

        # Data-parallelism across TP groups
        dp_start = tp_rank
        dp_end = dp_start + WORLD_SIZE
        dp_ranks = list(range(dp_start, dp_end, tp_size))
        dp_group = dist.new_group(backend="nccl", ranks=dp_ranks)

    else:
        if opts.num_replicas > 1:
            # Mixed data- and tensor-parallelism on a single node
            # NOTE: Avoid dist.init_device_mesh() to support older PyTorch versions
            all_ranks = torch.tensor(list(range(LOCAL_SIZE)), dtype=torch.uint8, device="cpu")
            mesh2d = all_ranks.reshape((opts.num_replicas, LOCAL_SIZE // opts.num_replicas))
            node_idx = (mesh2d == LOCAL_RANK).nonzero().squeeze().tolist()

            tp_ranks = mesh2d[node_idx[0], :].tolist()
            tp_group = dist.new_group(backend="nccl", ranks=tp_ranks)

            dp_ranks = mesh2d[:, node_idx[1]].tolist()
            dp_group = dist.new_group(backend="nccl", ranks=dp_ranks)
        else:
            dp_group = None
            tp_group = nccl_world

        tp_rank = dist.get_rank(tp_group)
        tp_size = dist.get_world_size(tp_group)

    dist_print(
        f"Created tensor-parallel group: {dist.get_process_group_ranks(tp_group)}",
        group=tp_group,
    )
    if dp_group is not None:
        dist_print(
            f"Created data-parallel group: {dist.get_process_group_ranks(dp_group)}",
            group=dp_group,
        )

    # Intialize userbuffers
    hidden_size = opts.num_heads * opts.head_dim
    batched_size = opts.seq_length * opts.batch_size
    if not opts.no_comm_overlap:
        te.module.base.initialize_ub(
            [batched_size, hidden_size],
            tp_size,
            use_fp8=opts.fp8,
            dtype=torch.bfloat16,
            bootstrap_backend=opts.bootstrap_backend,
        )

    # Initialize the fused LayerNorm + Multi-layer Perceptron module
    torch.manual_seed(opts.seed + tp_rank)
    torch.cuda.manual_seed(opts.seed + tp_rank)
    model = te.LayerNormMLP(
        hidden_size,
        opts.mlp_expansion_factor * hidden_size,
        params_dtype=torch.bfloat16,
        device="cuda",
        tp_group=tp_group,
        tp_size=tp_size,
        set_parallel_mode=True,
        sequence_parallel=True,  # this is required for comm+GEMM overlap
        seq_length=opts.seq_length,
        ub_overlap_rs=not opts.no_comm_overlap,
        ub_overlap_ag=not opts.no_comm_overlap,
        ub_overlap_rs_dgrad=not opts.no_comm_overlap,
        ub_bulk_dgrad=False,
        ub_bulk_wgrad=not opts.no_comm_overlap,
    )
    if dp_group is not None:
        model = DistributedDataParallel(model, process_group=dp_group)

    # Initialize optimizer with model parameters
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Fp8 recipe setup
    fp8_format = Format.HYBRID
    fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=32, amax_compute_algo="max")

    # Start dummy "training" iterations
    dist_print("Starting training iterations...", nccl_world)
    for i in range(opts.num_iters):
        dist_print(f"    Iter {i+1}", tp_group, debug=True)

        dist_print("    |-- Generate random input batch", tp_group, debug=True)
        x = torch.rand(
            (opts.seq_length // tp_size, opts.batch_size, hidden_size),
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )

        dist_print("    |-- Forward pass", tp_group, debug=True)
        with te.fp8_autocast(enabled=opts.fp8, fp8_recipe=fp8_recipe, fp8_group=nccl_world):
            y = model(x)
            dist_print("    |-- Compute loss", tp_group, debug=True)
            loss = y.flatten().sum()

        dist_print("    |-- Backward pass", tp_group, debug=True)
        loss.backward()

        dist_print("    |-- Optimizer step", tp_group, debug=True)
        optim.step()

    torch.cuda.synchronize()
    dist_print("Finished training!")
    te.module.base.destroy_ub()

    dist_print("Destroying all process groups...", debug=True)
    dist.destroy_process_group()
    if opts.debug and WORLD_RANK == 0:
        print("Exiting...\n", end="", flush=True)

    return 0


if __name__ == "__main__":
    os._exit(_train(_parse_args()))
