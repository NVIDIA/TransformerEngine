# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Collective GEMM test on multi-GPU with tensor parallelism

This script uses custom distributed initialization with the following arguments:
- --coordinator-address: Coordinator address for distributed initialization
- --num-processes: Number of processes for distributed initialization
- --process-id: Process ID for distributed initialization
- --local-device-ids: Local device IDs for distributed initialization

Example:
    python test_gemm.py --coordinator-address localhost:1234 --num-processes 2 --process-id 0 --local-device-ids 0,1,2,3
"""
import unittest
import os
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec, NamedSharding

from common import (
    assert_allclose,
    _initialize_distributed,
    _get_dp_and_tp_sizes,
    _create_mesh,
    DP_AXIS,
    TPSP_AXIS,
    PARAMS_KEY,
    cgemm_parser,
)

import transformer_engine.jax.cpp_extensions as tex
from transformer_engine.jax.quantize import autocast
from transformer_engine.jax.cpp_extensions.gemm import CollectiveOp
from transformer_engine.jax.sharding import MeshResource


def _get_operand_sharding(mesh, collective_op, is_with_dp):

    dp_axis = DP_AXIS if is_with_dp else None
    if collective_op == CollectiveOp.ALL_GATHER:
        x_sharding = NamedSharding(mesh, PartitionSpec(dp_axis, TPSP_AXIS, None))
        weight_sharding = NamedSharding(mesh, PartitionSpec(None, TPSP_AXIS))
        bias_sharding = NamedSharding(mesh, PartitionSpec(TPSP_AXIS))
        output_sharding = NamedSharding(mesh, PartitionSpec(dp_axis, None, TPSP_AXIS))
    else:  # RS
        x_sharding = NamedSharding(mesh, PartitionSpec(dp_axis, None, TPSP_AXIS))
        weight_sharding = NamedSharding(mesh, PartitionSpec(TPSP_AXIS, None))
        bias_sharding = NamedSharding(mesh, PartitionSpec(None))
        output_sharding = NamedSharding(mesh, PartitionSpec(dp_axis, TPSP_AXIS, None))

    return x_sharding, weight_sharding, bias_sharding, output_sharding


def _get_dp_and_tp_sizes(args):
    num_gpu = args.num_processes * args.num_devices_per_process
    if args.tensor_parallel_size is None:
        num_gpu_dp = 2 if args.enable_data_parallel else 1
        assert (
            num_gpu > 1 and num_gpu % num_gpu_dp == 0
        ), "Number of GPUs must be greater than 1 and divisible by number of data parallel GPUs"
        num_gpu_tp = num_gpu // num_gpu_dp
    else:
        num_gpu_tp = args.tensor_parallel_size
        assert (
            num_gpu > 1 and num_gpu % num_gpu_tp == 0
        ), "Number of GPUs must be greater than 1 and divisible by number of data parallel GPUs"
        num_gpu_dp = num_gpu // num_gpu_tp
    return num_gpu_dp, num_gpu_tp


@partial(jax.jit, static_argnames=("contracting_dims", "collective_op", "output_sharding"))
def _jitted_cgemm(x, weight, bias, contracting_dims, collective_op, output_sharding):
    output = tex.gemm(
        x,
        weight,
        bias=bias,
        contracting_dims=contracting_dims,
        collective_op=collective_op,
    )
    if output_sharding is not None:
        output = jax.lax.with_sharding_constraint(output, output_sharding)
    return output


def run_gemm_tests(args, mesh=None):
    """Execute GEMM tests."""
    print(args)
    # Collective GEMM requires Shardy partitioner to be disabled
    jax.config.update("jax_use_shardy_partitioner", False)

    # Initialize distributed with provided arguments
    _initialize_distributed(args)
    mesh = mesh or _create_mesh(args)

    # Create test data
    rng = jax.random.PRNGKey(0)
    rng, x_rng, weight_rng, bias_rng = jax.random.split(rng, 4)
    x = jax.random.normal(
        x_rng, (args.batch_size, args.seq_len, args.hidden_in), dtype=jnp.bfloat16
    )
    weight = jax.random.normal(weight_rng, (args.hidden_in, args.hidden_out), dtype=jnp.bfloat16)
    bias = jax.random.normal(bias_rng, (args.hidden_out,), dtype=jnp.bfloat16)
    collective_op = (
        CollectiveOp.ALL_GATHER
        if args.collective_type == "all_gather"
        else CollectiveOp.REDUCE_SCATTER
    )

    with mesh, autocast(
        enabled=False,
        recipe=None,
        mesh_resource=MeshResource(dp_resource=DP_AXIS, tpsp_resource=TPSP_AXIS),
    ):
        print(f"Device mesh: {mesh}")

        x_sharding, weight_sharding, bias_sharding, output_sharding = _get_operand_sharding(
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
            collective_op=CollectiveOp.NONE,
            output_sharding=output_sharding,
        )
        output = _jitted_cgemm(
            x_sharded,
            weight_sharded,
            bias_sharded,
            contracting_dims=((2,), (0,)),
            collective_op=collective_op,
            # CollectiveGEMM output should have a correct sharding without applying sharding constraint
            output_sharding=None,
        )
        assert (
            ref_output.sharding == output.sharding
        ), f"ref_output.sharding={ref_output.sharding}, output.sharding={output.sharding}"
        gathered_ref_output = jax.lax.with_sharding_constraint(
            ref_output, NamedSharding(mesh, PartitionSpec(None))
        )
        gathered_output = jax.lax.with_sharding_constraint(
            output, NamedSharding(mesh, PartitionSpec(None))
        )
        jax.block_until_ready(gathered_ref_output)
        jax.block_until_ready(gathered_output)

    if args.enable_result_check and args.process_id == 0:
        assert_allclose(gathered_ref_output, gathered_output)


class TestCollectiveGemmWithDP(unittest.TestCase):
    """Collective GEMM with DP unittests"""

    def setUp(self):
        self.args = cgemm_parser(
            "Collective GEMM test on multi-GPU with tensor parallelism"
        ).parse_args([])
        self.args.coordinator_address = self.coordinator_address
        self.args.num_processes = self.num_processes
        self.args.process_id = self.process_id
        self.args.local_device_ids = self.local_device_ids
        self.args.num_devices_per_process = self.num_devices_per_process
        self.args.enable_data_parallel = True
        self.args.tensor_parallel_size = _get_dp_and_tp_sizes(self.args)[1]
        _initialize_distributed(self.args)
        self.mesh = _create_mesh(self.args)
        jax.sharding.set_mesh(self.mesh)
        self.args.enable_result_check = True
        os.environ["NVTE_JAX_ALL_REDUCE_IN_FP32"] = "1"

    def tearDown(self):
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
    import sys

    if len(sys.argv) < 5:  # Need at least the 3 required distributed args
        print("Error: This script requires distributed initialization arguments.")
        print(
            "Usage: python test_gemm.py --coordinator-address <address> --num-processes <num>"
            " --process-id <id> [--local-device-ids <ids>] [other args]"
        )
        sys.exit(1)

    args = cgemm_parser("Collective GEMM test on multi-GPU with tensor parallelism").parse_args()
    _initialize_distributed(args)
    run_gemm_tests(args, mesh=None)
