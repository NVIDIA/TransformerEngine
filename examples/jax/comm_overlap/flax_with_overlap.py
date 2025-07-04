# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Comm+GEMM Overlap with TE/JAX"""
import os
import argparse
from functools import partial

from mpi4py import MPI

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.experimental import mesh_utils
from flax.linen import partitioning as nn_partitioning

import transformer_engine.jax as te
import transformer_engine_jax as tex
from transformer_engine.jax.sharding import (
    get_padded_spec,
    MeshResource,
    HIDDEN_AXES,
    HIDDEN_TP_AXES,
    BATCH_AXES,
    SEQLEN_TP_AXES,
    SEQLEN_AXES,
    W_NO_SHARD_AXES,
    W_FSDP_AXES,
    W_TP_AXES,
    W_JOINED_AXES,
)
from transformer_engine.jax.flax import DenseGeneral, LayerNormDenseGeneral, LayerNormMLP
from transformer_engine.common import recipe

# This script needs to be launched via `mpirun` with 1 process per GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
myrank = MPI.COMM_WORLD.Get_rank()
numranks = MPI.COMM_WORLD.Get_size()
jax.clear_caches()
jax.distributed.initialize(cluster_detection_method="mpi4py")
assert (
    jax.local_device_count() == 1
), f"[{myrank}|{numranks}] Expected 1 GPU per process, found {jax.local_device_count()}"

# Parse script arguments
_supported_layers = (DenseGeneral, LayerNormDenseGeneral, LayerNormMLP)
_layer_map = dict((layer.__name__.lower(), layer) for layer in _supported_layers)


def _te_flax_layer(layer_name):
    assert isinstance(layer_name, str) and layer_name.lower() in _layer_map
    return _layer_map[layer_name.lower()]


parser = argparse.ArgumentParser()
parser.add_argument("-dp", "--dp-size", type=int, default=2)
parser.add_argument("-tp", "--tp-size", type=int, default=numranks // 2)
parser.add_argument("-np", "--num-gpus", type=int, default=numranks)
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--seq-length", type=int, default=8192)
parser.add_argument("--hidden-size", type=int, default=16384)
parser.add_argument("--activation-size", type=int, default=53248)
parser.add_argument("--no-batch", action="store_true")
parser.add_argument("--no-fsdp", action="store_true")
parser.add_argument(
    "--layer-type", type=_te_flax_layer, default=DenseGeneral, choices=_supported_layers
)
parser.add_argument(
    "--fp8-recipe", type=str.lower, default="none", choices=["none", "current", "delayed", "mxfp8"]
)
parser.add_argument("--check-result", action="store_true")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# FP8 recipe
fp8_recipe = None
match args.fp8_recipe:
    case "current":
        fp8_recipe = recipe.Float8CurrentScaling()
    case "delayed":
        fp8_recipe = recipe.DelayedScaling()
    case "mxfp8":
        fp8_recipe = recipe.MXFP8BlockScaling()
    case _:
        fp8_recipe = None

# Single GPU evaluation
layer_kwargs = {"use_bias": True}
match args.layer_type:
    case DenseGeneral:
        layer_kwargs.update({"features": args.hidden_size, "name": "proj"})
    case LayerNormDenseGeneral:
        layer_kwargs.update(
            {"features": 3 * args.hidden_size, "return_layernorm_output": False, "name": "qkv"}
        )
    case LayerNormMLP:
        layer_kwargs.update(
            {
                "intermediate_dim": args.activation_size,
                "return_layernorm_output": False,
                "name": "mlp",
            }
        )

rng = jax.random.PRNGKey(args.seed)
rng, params_rng = jax.random_split(rng)
init_rngs = {"params": params_rng}

dtype = jnp.bfloat16
input_shape = (args.seq_length, args.hidden_size)
if not args.no_batch:
    input_shape = (args.batch_size,) + input_shape
x = jnp.random.normal(rng, input_shape, dtype=jnp.bfloat16)

with te.fp8_autocast(enabled=fp8_recipe is not None, fp8_recipe=fp8_recipe):
    model_single = partial(args.layer_type, **layer_kwargs)
    params_single = model_single.init(init_rngs, x, deterministic=True)
    output_single = model_single.apply(params_single, x, deterministic=True)

# Resources and partition specs
DEVICE_DP_AXIS = "dp"
DEVICE_TP_AXIS = "tp"
mesh_shape = (args.dp_size, args.tp_size)
mesh_axes = (DEVICE_DP_AXIS, DEVICE_TP_AXIS)
mesh_resource = MeshResource(
    dp_resource=DEVICE_DP_AXIS if args.no_fsdp else None,
    fsdp_resource=None if args.no_fsdp else DEVICE_DP_AXIS,
    tp_resource=DEVICE_TP_AXIS,
)

INPUT_AXES = (
    SEQLEN_TP_AXES if args.layer_type != DenseGeneral else SEQLEN_AXES,
    HIDDEN_AXES if args.layer_type != DenseGeneral else HIDDEN_TP_AXES,
)
INTERMEDIATE_AXES = (SEQLEN_AXES, HIDDEN_TP_AXES)
if not args.no_batch:
    INPUT_AXES = (BATCH_AXES,) + INPUT_AXES
    INTERMEDIATE_AXES = (BATCH_AXES,) + INTERMEDIATE_AXES

LN_SCALE_AXES = LN_BIAS_AXES = (W_NO_SHARD_AXES,)

KERNEL_AXES_ROW_PARALLEL = (W_TP_AXES, W_FSDP_AXES)
BIAS_AXES_ROW_PARALLEL = (W_NO_SHARD_AXES,)
KERNEL_AXES_COL_PARALLEL = (W_FSDP_AXES, W_TP_AXES)
BIAS_AXES_COL_PARALLEL = (W_TP_AXES,)
if args.layer_type == LayerNormMLP:
    KERNEL_AXES_COL_PARALLEL = (W_FSDP_AXES, W_JOINED_AXES, W_TP_AXES)
    BIAS_AXES_COL_PARALLEL = (W_JOINED_AXES, W_NO_SHARD_AXES)

# Multi GPU evaluation
layer_kwargs.update({"enable_comm_overlap": True})
if args.layer_type in (DenseGeneral, LayerNormDenseGeneral):
    layer_kwargs.update(
        {
            "kernel_axes": KERNEL_AXES_COL_PARALLEL,
            "bias_axes": BIAS_AXES_COL_PARALLEL,
            "comm_overlap_config": {"method": tex.CommOverlapMethod.RING_EXCHANGE},
        }
    )
    if args.layer_type == LayerNormDenseGeneral:
        layer_kwargs.update(
            {
                "layernorm_input_axes": INPUT_AXES,
                "scale_axes": LN_SCALE_AXES,
                "ln_bias_axes": LN_BIAS_AXES,
                "dot_input_axes": INPUT_AXES,
            }
        )
else:
    layer_kwargs.update(
        {
            "layernorm_input_axes": INPUT_AXES,
            "scale_axes": LN_SCALE_AXES,
            "ln_bias_axes": LN_BIAS_AXES,
            "dot_1_input_axes": INPUT_AXES,
            "kernel_1_axes": KERNEL_AXES_COL_PARALLEL,
            "bias_axes_1": BIAS_AXES_COL_PARALLEL,
            "dot_2_input_axes": INTERMEDIATE_AXES,
            "kernel_2_axes": KERNEL_AXES_ROW_PARALLEL,
            "bias_axes_2": BIAS_AXES_ROW_PARALLEL,
            "dot_1_comm_overlap_config": {"method": tex.CommOverlapMethod.RING_EXCHANGE},
            "dot_2_comm_overlap_config": {"method": tex.CommOverlapMethod.RING_EXCHANGE},
        }
    )

device_mesh = mesh_utils.create_device_mesh((args.dp_size, args.tp_size))
mesh = Mesh(devices=device_mesh, axis_names=(DEVICE_DP_AXIS, DEVICE_TP_AXIS))
axis_rules = nn_partitioning.axis_rules(
    (
        (BATCH_AXES, DEVICE_DP_AXIS),
        (SEQLEN_AXES, None),
        (SEQLEN_TP_AXES, DEVICE_TP_AXIS),
        (HIDDEN_AXES, None),
        (HIDDEN_TP_AXES, DEVICE_TP_AXIS),
        (W_NO_SHARD_AXES, None),
        (W_JOINED_AXES, None),
        (W_FSDP_AXES, None if args.no_fsdp else DEVICE_DP_AXIS),
        (W_TP_AXES, DEVICE_TP_AXIS),
    )
)
with (
    mesh,
    axis_rules,
    te.fp8_autocast(
        enabled=fp8_recipe is not None,
        fp8_recipe=fp8_recipe,
        mesh_resource=mesh_resource,
    ),
):
    model_sharded = partial(args.layer_type, **layer_kwargs)
    params_sharded = model_sharded.init(init_rngs, x, deterministic=True)
    output_sharded = model_sharded.apply(params_sharded, x, deterministic=True)

if myrank == 0:
    print(
        f"{myrank}: {args.layer_type.__name__} OUTPUT {output_sharded.shape}\n"
        + f"    Sharding: {get_padded_spec(output_sharded.sharding.spec, output_sharded.ndim)}\n",
        flush=True,
    )

if args.check_result:
    output_gathered = jax.lax.with_sharding_constraint(
        output_sharded, NamedSharding(mesh, PartitionSpec(None))
    )
    jax.block_until_ready(output_gathered)

    diff = jnp.abs(output_single - output_gathered).flatten()
    if myrank == 0:
        print(f"{myrank}: Global output difference: {diff}\n", flush=True)

    m = jnp.argmax(diff).item()
    abs_err = diff[m].item()
    rel_err = abs_err / max(abs(output_single.flatten()[m]), 1e-5)

    rtol = 0.02
    atol = 0.001
    numerics_failed = False
    if rel_err > rtol and abs_err > atol:
        numerics_failed = True
        numerics_info = (
            "NUMERICAL CHECK FAILED: "
            + f"Outputs not close enough at index {m} "
            + f"with {output_gathered.flatten()[m].item()} vs {output_single.flatten()[m].item()} "
            + f"| rel. error = {rel_err} (tol = {rtol}) "
            + f"| abs. error = {abs_err} (tol = {atol})"
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

tex.destroy_all_comm_overlap_buffers()
