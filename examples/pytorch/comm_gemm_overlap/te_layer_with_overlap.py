#!/usr/bin/python3

# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import sys
import socket
import fcntl
import struct
import argparse
import warnings

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import transformer_engine.pytorch as te
import transformer_engine.pytorch.cpp_extensions as tex
from transformer_engine.common.recipe import Format, DelayedScaling

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
if not tex.device_supports_multicast():
    os.environ["UB_SKIPMC"] = "1"


def _te_layer_argtype(name):
    te_layers = [
        te.Linear,
        te.LayerNormLinear,
        te.LayerNormMLP,
        te.MultiheadAttention,
        te.TransformerLayer,
    ]
    layer_map = dict(zip([layer.__name__.lower() for layer in te_layers], te_layers))
    if name.lower() not in layer_map.keys():
        raise argparse.ArgumentTypeError(
            f"Invalid TE layer name! Please choose from: {layer_map.keys()}"
        )
    return layer_map[name.lower()]


def _parse_args(argv=None, namespace=None):
    parser = argparse.ArgumentParser(
        description="Train a Transformer Engine module with GEMM+comm overlap via Userbuffers."
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
        "--layer-type",
        type=_te_layer_argtype,
        default=te.TransformerLayer,
        help="Transformer Engine layer to train with comm+GEMM overlap.",
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
        "--num-replicas",
        type=int,
        default=1,
        help="Number of data-parallel model replicas per node.",
    )
    parser.add_argument(
        "--use-global-replica-count",
        action="store_true",
        default=False,
        help="Treat '--num-replicas' as the total number of replicas.",
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
    if args.bootstrap_backend == "nccl":
        args.bind_to_device = True
    return args


def _get_layer_args(config, tp_group, tp_size, reference=False):
    hidden_size = config.num_heads * config.head_dim
    input_shape = [config.seq_length, config.batch_size, hidden_size]
    args = [hidden_size]
    kwargs = {
        "params_dtype": torch.float32,
        "device": "cuda",
        "tp_group": tp_group,
        "tp_size": tp_size,
        "sequence_parallel": True,
    }
    kwargs["ub_overlap_ag"] = not config.no_comm_overlap

    if config.layer_type is te.Linear:
        input_shape[2] = hidden_size // tp_size
        args.append(hidden_size)
        kwargs["parallel_mode"] = "row"
        kwargs["ub_overlap_rs"] = not config.no_comm_overlap
        kwargs["ub_name"] = "proj"
    else:
        input_shape[0] = config.seq_length // tp_size
        kwargs["ub_bulk_wgrad"] = not config.no_comm_overlap
        kwargs["ub_bulk_dgrad"] = not config.no_comm_overlap
        if config.layer_type is te.LayerNormLinear:
            args.append(3 * hidden_size)
            kwargs["parallel_mode"] = "column"
            kwargs["ub_name"] = "qkv"
        else:
            kwargs["set_parallel_mode"] = True
            kwargs["ub_overlap_rs"] = not config.no_comm_overlap
            if config.layer_type in [te.LayerNormMLP, te.TransformerLayer]:
                args.append(4 * hidden_size)
                kwargs["seq_length"] = config.seq_length
            if config.layer_type in [te.MultiheadAttention, te.TransformerLayer]:
                args.append(config.num_heads)
                kwargs["attention_dropout"] = 0.0
                kwargs["fuse_qkv_params"] = True
                if config.layer_type is te.MultiheadAttention:
                    kwargs["input_layernorm"] = True
                else:
                    kwargs["ub_tp_comm_overlap"] = not config.no_comm_overlap
                    kwargs["hidden_dropout"] = 0.0

    return args, kwargs, input_shape


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
    else:  # TORCHELASTIC, SLURM, etc...
        WORLD_RANK = int(os.getenv("RANK", "0"))
        WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
        LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
        LOCAL_SIZE = int(os.getenv("LOCAL_WORLD_SIZE", str(torch.cuda.device_count())))

    NUM_NODES = WORLD_SIZE // LOCAL_SIZE

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

    def dist_print(msg, end="\n", group=nccl_world, src=0, debug=False, error=False):
        if debug and not opts.debug:
            return
        group_rank = dist.get_rank(group)
        stream = sys.stderr if error else sys.stdout
        if group_rank == src:
            stream.write(f"[rank{WORLD_RANK}] {msg}{end}")
        dist.barrier(group)

    dist_print(f"Initialized default NCCL process group with {WORLD_SIZE} GPUs")

    total_replicas = (
        opts.num_replicas if opts.use_global_replica_count else opts.num_replicas * NUM_NODES
    )
    tp_size = WORLD_SIZE // total_replicas

    if total_replicas > 1:
        ranks_per_replica_list = [
            [i * tp_size + t for t in range(tp_size)] for i in range(total_replicas)
        ]

        tp_group, _ = dist.new_subgroups_by_enumeration(ranks_per_replica_list, backend="nccl")
        ranks_per_replica_tensor = torch.tensor(ranks_per_replica_list, dtype=torch.int32)
        dp_group, _ = dist.new_subgroups_by_enumeration(
            ranks_per_replica_tensor.transpose(0, 1).tolist(), backend="nccl"
        )
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
        dp_rank = dist.get_rank(dp_group)
        dist_print(
            f"Created data-parallel group: {dist.get_process_group_ranks(dp_group)}",
            group=dp_group,
        )
    else:
        dp_rank = 0

    # Intialize userbuffers
    hidden_size = opts.num_heads * opts.head_dim
    batched_size = opts.seq_length * opts.batch_size
    if not opts.no_comm_overlap:
        te.module.base.initialize_ub(
            [batched_size, hidden_size],
            tp_size,
            quantization_modes=[
                (
                    te.module.base.UserBufferQuantizationMode.FP8
                    if opts.fp8
                    else te.module.base.UserBufferQuantizationMode.NONE
                )
            ],
            dtype=torch.bfloat16,
            bootstrap_backend=opts.bootstrap_backend,
        )

    # Initialize the fused LayerNorm + Multi-layer Perceptron module
    torch.manual_seed(opts.seed + dp_rank)
    torch.cuda.manual_seed(opts.seed + tp_rank)
    layer_args, layer_kwargs, input_size = _get_layer_args(opts, tp_group, tp_size)
    model = opts.layer_type(*layer_args, **layer_kwargs)
    if dp_group is not None:
        model = DistributedDataParallel(model, dim=1, process_group=dp_group)

    # Initialize optimizer with model parameters
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Fp8 recipe setup
    fp8_format = Format.HYBRID
    fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=32, amax_compute_algo="max")

    # Start dummy "training" iterations
    dist_print("Starting training iterations...")
    for i in range(opts.num_iters):
        dist_print(f"    Iter {i+1}", group=tp_group, debug=True)

        dist_print("    |-- Generate random input batch", group=tp_group, debug=True)
        x = torch.randn(input_size, dtype=torch.float32, device="cuda", requires_grad=True)

        dist_print("    |-- Forward pass", group=tp_group, debug=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            with te.fp8_autocast(enabled=opts.fp8, fp8_recipe=fp8_recipe, fp8_group=nccl_world):
                y = model(x)
                if isinstance(y, tuple):
                    out, *_ = y
                else:
                    out = y
                dist_print("    |-- Compute loss", group=tp_group, debug=True)
                loss = out.sum()

        dist_print("    |-- Backward pass", group=tp_group, debug=True)
        loss.backward()

        dist_print("    |-- Optimizer step", group=tp_group, debug=True)
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
    sys.exit(_train(_parse_args()))
