# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Comm+GEMM Overlap with TE/JAX"""

import argparse
import numpy as np
from functools import partial
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
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--seq-length", type=int, default=8192)
parser.add_argument("--num-heads", type=int, default=128)
parser.add_argument("--head-dim", type=int, default=128)
parser.add_argument("--activation-size", type=int, default=53248)
parser.add_argument("--no-batch", action="store_true")
parser.add_argument("--no-fsdp", action="store_true")
parser.add_argument("--comm-type", type=str.upper, default="AG", choices=["AG", "RS"])
parser.add_argument("--check-result", action="store_true")
args = parser.parse_args()

# Operand shapes
dtype = jnp.bfloat16
hidden_size = args.num_heads * args.head_dim
lhs_shape = (
    [args.seq_length, hidden_size]
    if args.comm_type == "AG"
    else [args.seq_length, args.activation_size]
)
rhs_shape = (
    [hidden_size, args.activation_size]
    if args.comm_type == "AG"
    else [args.activation_size, hidden_size]
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
    if fsdp:
        mesh_shape = {"zp": args.fsdp_size, "tp": args.tp_size}
        mesh_resource = te.MeshResource(fsdp_resource="zp", tp_resource="tp", cp_resource="cp")
        if args.comm_type == "AG":
            input_specs = ["tp", None]
            weight_specs = ["zp", "tp"]
        elif args.comm_type == "RS":
            input_specs = [None, "tp"]
            weight_specs = ["tp", "zp"]
        weight_no_fsdp = ["tp", None]
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
no_sharding = NamedSharding(mesh, PartitionSpec(None))
input_sharding = NamedSharding(mesh, PartitionSpec(*input_specs))
weight_sharding = NamedSharding(mesh, PartitionSpec(*weight_specs))
weight_no_fsdp_sharding = NamedSharding(mesh, PartitionSpec(*weight_no_fsdp))

# Operand initialization
key = jax.random.PRNGKey(0)
key1, key2 = jax.random.split(key, 2)
lhs_data = jax.random.normal(key1, lhs_shape, dtype=dtype)
rhs_data = jax.random.normal(key2, rhs_shape, dtype=dtype)
lhs = jax.device_put(lhs_data, input_sharding)
rhs = jax.device_put(rhs_data, weight_sharding)

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
            "method": "ring_exchange",  # "pipeline" for collective kernels instead of send/recv
            "comm_type": (
                tex.CommOverlapType.AG if args.comm_type == "AG" else tex.CommOverlapType.RS
            ),
            "num_splits": args.tp_size,  # independent of TP size for "pipeline"
            "cga_size": 1,  # default is 2 for "pipeline"
            "num_sm": 1,  # ignored for "ring_exchange", must be tuned for "pipeline"
            "set_sm_margin": False,  # set to True for "pipeline"
            "atomic_gemm": False,  # more performant when not using CUDA Graphs
            "use_ce": True,  # ignored (always False) for "pipeline" method
        },
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
        jax.lax.with_sharding_constraint(B, weight_no_fsdp_sharding),  # all-gather FSDP weights
        batched_output=not args.no_batch,  # internal option, will be hidden by the FWD/BWD wrapper
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

if args.check_result:
    ref_global = jnp.matmul(
        jax.device_put(lhs_data, no_sharding), jax.device_put(rhs_data, no_sharding)
    )
    if myrank == 0:
        print(f"{myrank}: Global reference: {ref_global}\n", flush=True)

    output_global = jax.lax.with_sharding_constraint(output, no_sharding)
    if myrank == 0:
        print(f"{myrank}: Global output: {output_global}\n", flush=True)

    diff = jnp.abs(ref_global - output_global).flatten()
    if myrank == 0:
        print(f"{myrank}: Global difference: {diff}\n", flush=True)

    m = jnp.argmax(diff).item()
    abs_err = diff[m].item()
    rel_err = abs_err / max(abs(ref_global.flatten()[m]), 1e-5)

    rtol = 0.02
    atol = 0.001
    numerics_failed = False
    if rel_err > rtol and abs_err > atol:
        numerics_failed = True
        numerics_info = (
            "NUMERICAL CHECK FAILED: "
            + f"Outputs not close enough at index {m} "
            + f"with {output.flatten()[m].item()} vs {ref_global.flatten()[m].item()} | "
            + f"rel. error = {rel_err} (tol = {rtol}) | "
            + f"abs. error = {abs_err} (tol = {atol})"
        )
    else:
        numerics_info = "NUMERICAL CHECK PASSED: "
        if rel_err <= rtol:
            numerics_info += f"rel. error = {rel_err} (tol = {rtol})" + (
                " | " if abs_err < atol else ""
            )
        if abs_err <= atol:
            numerics_info += f"abs. error = {abs_err} (tol = {atol})"

    if myrank == 0:
        print(numerics_info + "\n", end="", flush=True)

destroy_comm_gemm_overlaps()
