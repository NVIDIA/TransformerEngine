# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Collective GEMM test on multi-GPU with tensor parallelism"""
import argparse
import unittest
import os

from mpi4py import MPI

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec, NamedSharding

from common import assert_allclose, assert_allclose_print_index

import transformer_engine.jax.cpp_extensions as tex

# from transformer_engine.jax.quantize import is_fp8_available, ScalingMode, Quantizer, QuantizeConfig, fp8_autocast
from transformer_engine.jax.quantize import fp8_autocast
from transformer_engine.jax.cpp_extensions.gemm import (
    CollectiveGemmConfig,
    CollectiveOp,
    noop_cgemm_config,
)
from transformer_engine.jax.sharding import MeshResource

DEVICE_DP_AXIS = "data"
DEVICE_TPSP_AXIS = "tensor_sequence"
PARAMS_KEY = "params"

jax.clear_caches()
jax.config.update(
    "jax_use_shardy_partitioner", False
)  # CollectiveGEMM does not work with Shardy yet

# FOR NOW: This script needs to be launched via `mpirun` with 1 process per GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
myrank = MPI.COMM_WORLD.Get_rank()
numranks = MPI.COMM_WORLD.Get_size()
jax.distributed.initialize(cluster_detection_method="mpi4py")
assert (
    jax.local_device_count() == 1
), f"[{myrank}|{numranks}] Expected 1 GPU per process, found {jax.local_device_count()}"


def _get_operand_sharding(mesh, collective_op, is_with_dp):

    dp_axis = DEVICE_DP_AXIS if is_with_dp else None
    if collective_op == CollectiveOp.ALL_GATHER:
        x_sharding = NamedSharding(mesh, PartitionSpec(dp_axis, DEVICE_TPSP_AXIS, None))
        weight_sharding = NamedSharding(mesh, PartitionSpec(None, DEVICE_TPSP_AXIS))
        bias_sharding = NamedSharding(mesh, PartitionSpec(DEVICE_TPSP_AXIS))
    else:  # RS
        x_sharding = NamedSharding(mesh, PartitionSpec(dp_axis, None, DEVICE_TPSP_AXIS))
        weight_sharding = NamedSharding(mesh, PartitionSpec(DEVICE_TPSP_AXIS, None))
        bias_sharding = NamedSharding(mesh, PartitionSpec(None))

    return x_sharding, weight_sharding, bias_sharding


def _create_mesh(args):
    """Create mesh configuration with proper validation."""
    num_gpu = jax.device_count()
    assert num_gpu == numranks, f"Requires {num_gpu} processes for {num_gpu} GPUs, got {numranks}!"
    num_gpu_dp = 2 if args.enable_data_parallel else 1
    assert (
        num_gpu > 1 and num_gpu % num_gpu_dp == 0
    ), "Number of GPUs must be greater than 1 and divisible by number of data parallel GPUs"

    num_gpu_tp = num_gpu // num_gpu_dp
    assert num_gpu_tp > 1, f"Number of GPUs for tensor parallelism ({num_gpu_tp}) must be > 1"
    print(f"Using {num_gpu_dp}x{num_gpu_tp} mesh ({num_gpu_dp * num_gpu_tp} total GPUs)")

    device_mesh = mesh_utils.create_device_mesh((num_gpu_dp, num_gpu_tp))
    mesh = jax.sharding.Mesh(devices=device_mesh, axis_names=(DEVICE_DP_AXIS, DEVICE_TPSP_AXIS))
    jax.sharding.set_mesh(mesh)

    return mesh


def _jitted_cgemm(x, weight, bias, contracting_dims, cgemm_config):
    return jax.jit(tex.gemm, static_argnames=("contracting_dims", "cgemm_config"))(
        x,
        weight,
        bias=bias,
        contracting_dims=contracting_dims,
        cgemm_config=cgemm_config,
    )


def run_gemm_tests(args, mesh=None):
    """Execute GEMM tests."""
    print(args)
    # Collective GEMM requires Shardy partitioner to be disabled
    jax.config.update("jax_use_shardy_partitioner", False)
    mesh = mesh or _create_mesh(args)

    # Create test data
    rng = jax.random.PRNGKey(0)
    rng, x_rng, weight_rng, bias_rng = jax.random.split(rng, 4)
    x = jax.random.normal(
        x_rng, (args.batch_size, args.seq_len, args.hidden_in), dtype=jnp.bfloat16
    )
    weight = jax.random.normal(weight_rng, (args.hidden_in, args.hidden_out), dtype=jnp.bfloat16)
    bias = jax.random.normal(bias_rng, (args.hidden_out,), dtype=jnp.bfloat16)

    with mesh, fp8_autocast(
        enabled=False,
        fp8_recipe=None,
        mesh_resource=MeshResource(dp_resource=DEVICE_DP_AXIS, tpsp_resource=DEVICE_TPSP_AXIS),
    ):
        print(f"Device mesh: {mesh}")

        # Collective GEMM configs need to be created under the mesh_resource context
        collective_op = (
            CollectiveOp.ALL_GATHER
            if args.collective_type == "all_gather"
            else CollectiveOp.REDUCE_SCATTER
        )
        cgemm_config = CollectiveGemmConfig.create(collective_op=collective_op)

        x_sharding, weight_sharding, bias_sharding = _get_operand_sharding(
            mesh, collective_op, args.enable_data_parallel
        )
        x_sharded = jax.device_put(x, x_sharding)
        weight_sharded = jax.device_put(weight, weight_sharding)
        bias_sharded = jax.device_put(bias, bias_sharding)

        ref_output = _jitted_cgemm(
            x_sharded,
            weight_sharded,
            bias_sharded,
            contracting_dims=((2,), (0,)),
            cgemm_config=noop_cgemm_config,
        )
        output = _jitted_cgemm(
            x_sharded,
            weight_sharded,
            bias_sharded,
            contracting_dims=((2,), (0,)),
            cgemm_config=cgemm_config,
        )
        gathered_ref_output = jax.lax.with_sharding_constraint(
            ref_output, NamedSharding(mesh, PartitionSpec(None))
        )
        gathered_output = jax.lax.with_sharding_constraint(
            output, NamedSharding(mesh, PartitionSpec(None))
        )
        jax.block_until_ready(gathered_ref_output)
        jax.block_until_ready(gathered_output)

    if args.enable_result_check and myrank == 0:
        assert_allclose(gathered_ref_output, gathered_output)
        # assert_allclose(gathered_ref_output, gathered_output, atol=1e-3, rtol=2e-2)
        # assert_allclose_print_index(gathered_ref_output, gathered_output)
        # assert_allclose_print_index(gathered_ref_output, gathered_output, rtol=1e-2, atol=1e-5)


def gemm_parser(args):
    """Test settings."""
    parser = argparse.ArgumentParser(description="JAX Collective GEMM Test")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        metavar="N",
        help="input batch size (default: 4)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        metavar="N",
        help="sequence length (default: 2048)",
    )
    parser.add_argument(
        "--hidden-in",
        type=int,
        default=1024,
        metavar="N",
        help="input hidden dimension (default: 1024)",
    )
    parser.add_argument(
        "--hidden-out",
        type=int,
        default=2048,
        metavar="N",
        help="output hidden dimension (default: 2048)",
    )
    parser.add_argument(
        "--collective-type",
        type=str,
        default="all_gather",
        choices=["all_gather", "reduce_scatter"],
        help="Collective operation type (default: all_gather)",
    )
    parser.add_argument(
        "--fp8-recipe",
        type=str,
        default="DelayedScaling",
        help="FP8 recipe (default: DelayedScaling)",
    )
    parser.add_argument(
        "--enable-data-parallel",
        action="store_true",
        default=False,
        help="Enable data parallel (default: False)",
    )
    parser.add_argument(
        "--enable-result-check",
        action="store_true",
        default=False,
        help="Enable result check (default: False)",
    )
    return parser.parse_args(args)


class TestCollectiveGemm(unittest.TestCase):
    """Collective GEMM unittests"""

    # is_fp8_supported, fp8_reason = is_fp8_available(ScalingMode.DELAYED_TENSOR_SCALING)
    # is_mxfp8_supported, mxfp8_reason = is_fp8_available(ScalingMode.MXFP8_1D_SCALING)

    def setUp(self):
        """Set up test environment for pytest execution."""
        # Init the arg parser
        self.args = gemm_parser(["--batch-size", "1"])
        # Create mesh once for all tests
        self.mesh = _create_mesh(self.args)
        jax.sharding.set_mesh(self.mesh)
        self.args.enable_result_check = True
        os.environ["NVTE_JAX_ALL_REDUCE_IN_FP32"] = "1"

    def tearDown(self):
        """Clean up after each test."""
        # Clear the mesh to prevent interference between tests
        jax.sharding.set_mesh(None)
        os.environ.pop("NVTE_JAX_ALL_REDUCE_IN_FP32", None)

    def test_te_bf16_all_gather(self):
        """Test Collective GEMM with AllGather"""
        self.args.collective_type = "all_gather"
        run_gemm_tests(self.args, self.mesh)

    def test_te_bf16_reduce_scatter(self):
        """Test Collective GEMM with ReduceScatter"""
        self.args.collective_type = "reduce_scatter"
        run_gemm_tests(self.args, self.mesh)


class TestCollectiveGemmWithDP(unittest.TestCase):
    """Collective GEMM with DP unittests"""

    def setUp(self):
        """Set up test environment for pytest execution."""
        # Init the arg parser
        self.args = gemm_parser(["--batch-size", "8"])
        # Create mesh once for all tests
        self.args.enable_data_parallel = True
        self.mesh = _create_mesh(self.args)
        jax.sharding.set_mesh(self.mesh)
        self.args.enable_result_check = True
        os.environ["NVTE_JAX_ALL_REDUCE_IN_FP32"] = "1"

    def tearDown(self):
        """Clean up after each test."""
        # Clear the mesh to prevent interference between tests
        jax.sharding.set_mesh(None)
        os.environ.pop("NVTE_JAX_ALL_REDUCE_IN_FP32", None)

    def test_te_bf16_all_gather_with_dp(self):
        """Test Collective GEMM with AllGather"""
        self.args.collective_type = "all_gather"
        run_gemm_tests(self.args, self.mesh)

    def test_te_bf16_reduce_scatter_with_dp(self):
        """Test Collective GEMM with ReduceScatter"""
        self.args.collective_type = "reduce_scatter"
        run_gemm_tests(self.args, self.mesh)


if __name__ == "__main__":
    run_gemm_tests(gemm_parser(None), mesh=None)
