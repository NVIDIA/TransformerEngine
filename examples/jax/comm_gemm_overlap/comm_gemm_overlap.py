# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Comm+GEMM Overlap with TE/JAX"""

import argparse
import numpy as np

from mpi4py import MPI

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.experimental import mesh_utils

import transformer_engine.jax as te
from transformer_engine import transformer_engine_jax as tex
from transformer_engine.jax.cpp_extensions import gemm_impl, copy_into_overlap_buffer
from transformer_engine.jax.gemm import (
    initialize_comm_gemm_overlaps,
    destroy_comm_gemm_overlaps,
    get_comm_overlap_config,
)
from transformer_engine.jax.sharding import get_padded_spec

jax.clear_caches()

# This script needs to be launched via `mpirun` with 1 process per GPU
myrank = MPI.COMM_WORLD.Get_rank()
numranks = MPI.COMM_WORLD.Get_size()
jax.distributed.initialize(cluster_detection_method="mpi4py")

parser = argparse.ArgumentParser()
parser.add_argument("-dp", "--dp-size", type=int, default=1)
parser.add_argument("-zp", "--fsdp-size", type=int, default=2)
parser.add_argument("-tp", "--tp-size", type=int, default=4)
parser.add_argument("-np", "--num-gpus", type=int, default=8)
parser.add_argument("--base-size", type=int, default=16)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--no-batch", action="store_true")
parser.add_argument("--no-fsdp", action="store_true")
parser.add_argument("--comm-type", type=str.upper, default="AG", choices=["AG", "RS"])
args = parser.parse_args()

# GEMM problem sizing
dtype = jnp.bfloat16
seq_length = args.base_size * 8
hidden_size = args.base_size * 6
ffn_hidden_size = args.base_size * 16

# Operand shapes
lhs_shape = [seq_length, hidden_size] if args.comm_type == "AG" else [seq_length, ffn_hidden_size]
rhs_shape = (
    [hidden_size, ffn_hidden_size] if args.comm_type == "AG" else [ffn_hidden_size, hidden_size]
)

# Operand partitioning
batched = not args.no_batch
fsdp = not args.no_fsdp
if batched:
    lhs_shape = [args.batch_size] + lhs_shape
    if fsdp:
        mesh_shape = {"dp": args.dp_size, "zp": args.fsdp_size, "tp": args.tp_size}
        mesh_resource = te.MeshResource(
            dp_resource="dp", tp_resource="tp", cp_resource="tp", fsdp_resource="zp"
        )
        if args.comm_type == "AG":
            input_specs = [("dp", "zp"), "tp", None]
            weight_specs = ["zp", "tp"]
            weight_no_fsdp = [None, "tp"]
        elif args.comm_type == "RS":
            input_specs = [("dp", "zp"), None, "tp"]
            weight_specs = ["tp", "zp"]
            weight_no_fsdp = ["tp", None]
    else:
        mesh_shape = {"dp": args.dp_size, "tp": args.tp_size}
        mesh_resource = te.MeshResource(
            dp_resource="dp",
            tp_resource="tp",
            cp_resource="tp",
        )
        if args.comm_type == "AG":
            input_specs = ["dp", "tp", None]
            weight_specs = [None, "tp"]
        elif args.comm_type == "RS":
            input_specs = ["dp", None, "tp"]
            weight_specs = ["tp", None]
        weight_no_fsdp = weight_specs
else:
    mesh_shape = {"tp": args.tp_size}
    mesh_resource = te.MeshResource(tp_resource="tp", cp_resource="cp")
    if args.comm_type == "AG":
        input_specs = ["tp", None]
        weight_specs = [None, "tp"]
    elif args.comm_type == "RS":
        input_specs = [None, "tp"]
        weight_specs = ["tp", None]
    weight_no_fsdp = weight_specs

# Mesh setup and sharding definitions
devices = mesh_utils.create_device_mesh((args.num_gpus,), devices=jax.devices()[: args.num_gpus])
mesh = Mesh(np.array(devices).reshape(tuple(mesh_shape.values())), tuple(mesh_shape.keys()))
input_sharding = NamedSharding(mesh, PartitionSpec(*input_specs))
weight_sharding = NamedSharding(mesh, PartitionSpec(*weight_specs))
weight_no_fsdp_sharding = NamedSharding(mesh, PartitionSpec(*weight_no_fsdp))

# Operand initialization
key = jax.random.PRNGKey(0)
key1, key2 = jax.random.split(key, 2)
lhs = jax.device_put(jax.random.normal(key1, lhs_shape, dtype=dtype), input_sharding)
rhs = jax.device_put(jax.random.normal(key2, rhs_shape, dtype=dtype), weight_sharding)

# Name of comm+GEMM overlap layer
overlap_name = "ag_gemm" if args.comm_type == "AG" else "gemm_rs"

# Bootstrap Userbuffers communicators and communication buffers
initialize_comm_gemm_overlaps(
    lhs_shape,
    mesh,
    myrank,
    numranks,
    tp_resource="tp",
    overlap_configs={
        overlap_name: {
            "method": "ring_exchange",   # "pipeline" for collective kernels instead of send/recv
            "comm_type": (
                tex.CommOverlapType.AG
                if args.comm_type == "AG"
                else tex.CommOverlapType.RS
            ),
            "num_splits": args.tp_size,   # independent of TP size for "pipeline"
            "cga_size": 1,   # default is 2 for "pipeline"
            "num_sm": 1,   # ignored for "ring_exchange", must be tuned for "pipeline"
            "set_sm_margin": False,   # set to True for "pipeline"
            "atomic_gemm": False,   # more performant when not using CUDA Graphs
            "use_ce": True,   # ignored (always False) for "pipeline" method
        }
    },
)

if myrank == 0:
    print(
        f"{myrank}: INPUTS {lhs.shape} x {rhs.shape}\n"
        + f"{myrank}:    LHS sharding: {lhs.sharding.spec}\n"
        + f"{myrank}:    RHS sharding: {rhs.sharding.spec}\n",
        flush=True,
    )


@jax.jit
def te_gemm(A, B):
    # For AG overlap, LHS needs to be copied into the comm. buffer before GEMM. This can usually
    # be circumvented by extracting the comm. buffer as a JAX array via
    # `buffer = jax.dlpack.from_dlpack(tex.get_overlap_buffer(overlap_name: str, sharded: bool))`
    # and directly writing the result of a preceding operation into it (e.g.. LayerNorm output
    # written directly into the communication buffer before AG+GEMM in a QKV projection)
    if args.comm_type == "AG":
        copy_into_overlap_buffer(A, overlap_name, True)
        return_idx = 0
    else:
        # For RS overlap, the scattered output is in the `extra_out` array.
        return_idx = -1

    return gemm_impl(
        A,
        jax.lax.with_sharding_constraint(B, weight_no_fsdp_sharding),   # all-gather FSDP weights
        batched_output=True,   # internal option, will be hidden by the FWD/BWD wrapper
        comm_overlap_config=get_comm_overlap_config(overlap_name),
    )[return_idx]


with te.sharding.global_shard_guard(mesh_resource):
    output = te_gemm(lhs, rhs)

if myrank == 0:
    print(
        f"{myrank}: {'AG -> GEMM' if args.comm_type == 'AG' else 'GEMM -> RS'} OUTPUT "
        + f"{output.shape}\n"
        + f"{myrank}:    Sharding: {get_padded_spec(output.sharding.spec, output.ndim)}\n",
        flush=True,
    )

destroy_comm_gemm_overlaps()
