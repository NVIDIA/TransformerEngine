# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Collective Dense Gradient test on multi-GPU with tensor parallelism"""
import argparse
import unittest
import os

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec, NamedSharding
import flax

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

from transformer_engine.jax.dense import dense

from transformer_engine.jax.quantize import autocast
from transformer_engine.jax.cpp_extensions.gemm import (
    CollectiveOp,
    CollectiveOpSet,
    noop_collective_op_set,
)
from transformer_engine.jax.sharding import MeshResource
import transformer_engine.jax.flax as te_flax


def _get_logical_axes(collective_op):
    if collective_op.is_all_gather:
        input_axes = (DP_AXIS, TPSP_AXIS, None)
        weight_axes = (None, TPSP_AXIS)
        bias_axes = (TPSP_AXIS,)
        output_axes = (DP_AXIS, None, TPSP_AXIS)
    else:  # RS
        input_axes = (DP_AXIS, None, TPSP_AXIS)
        weight_axes = (TPSP_AXIS, None)
        bias_axes = (None,)
        output_axes = (DP_AXIS, TPSP_AXIS, None)
    return input_axes, weight_axes, bias_axes, output_axes


def _get_operand_sharding(mesh, collective_op):
    input_axes, weight_axes, bias_axes, _ = _get_logical_axes(collective_op)
    x_sharding = NamedSharding(mesh, PartitionSpec(*input_axes))
    weight_sharding = NamedSharding(mesh, PartitionSpec(*weight_axes))
    bias_sharding = NamedSharding(mesh, PartitionSpec(*bias_axes))
    return x_sharding, weight_sharding, bias_sharding


def _mean_dense(x, weight, bias, input_axes, weight_axes, output_axes, collective_op_set):
    output = dense(
        x,
        weight,
        bias,
        contracting_dims=((2,), (0,)),
        input_axes=input_axes,
        kernel_axes=weight_axes,
        output_axes=output_axes,
        collective_op_set=collective_op_set,
    )
    return jnp.mean(output.astype(jnp.float32))


def _value_and_grad_dense(x, weight, bias, input_axes, weight_axes, output_axes, collective_op_set):
    return jax.jit(jax.value_and_grad(_mean_dense, (0, 1, 2)), static_argnums=(3, 4, 5, 6))(
        x, weight, bias, input_axes, weight_axes, output_axes, collective_op_set
    )


def run_dense_grad_tests(args, mesh=None):
    """Execute Dense Gradient tests."""
    print(args)
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
    collective_op_set = CollectiveOpSet.create(forward_collective_op=collective_op)

    with mesh, autocast(
        enabled=False,
        recipe=None,
        mesh_resource=MeshResource(dp_resource=DP_AXIS, tpsp_resource=TPSP_AXIS),
    ):
        # Get the base axis rules and extend them with TE's rules. This must be done inside autocast
        axis_rules = flax.linen.get_logical_axis_rules()
        axis_rules += ((TPSP_AXIS, TPSP_AXIS), (DP_AXIS, DP_AXIS))
        te_extended_axis_rules = te_flax.extend_logical_axis_rules(axis_rules)
        with flax.linen.logical_axis_rules(te_extended_axis_rules):

            x_sharding, weight_sharding, bias_sharding = _get_operand_sharding(mesh, collective_op)
            x_sharded = jax.device_put(x, x_sharding)
            weight_sharded = jax.device_put(weight, weight_sharding)
            bias_sharded = jax.device_put(bias, bias_sharding)

            input_axes, weight_axes, _, output_axes = _get_logical_axes(collective_op)
            ref_output, ref_grads = _value_and_grad_dense(
                x_sharded,
                weight_sharded,
                bias_sharded,
                input_axes,
                weight_axes,
                output_axes,
                noop_collective_op_set,
            )
            output, sharded_grads = _value_and_grad_dense(
                x_sharded,
                weight_sharded,
                bias_sharded,
                input_axes,
                weight_axes,
                output_axes,
                collective_op_set,
            )
        jax.block_until_ready(ref_output)
        jax.block_until_ready(output)
        gathered_grads = []
        gathered_ref_grads = []
        for ref_grad, grad in zip(ref_grads, sharded_grads):
            gathered_grads.append(
                jax.lax.with_sharding_constraint(grad, NamedSharding(mesh, PartitionSpec(None)))
            )
            gathered_ref_grads.append(
                jax.lax.with_sharding_constraint(ref_grad, NamedSharding(mesh, PartitionSpec(None)))
            )
        jax.block_until_ready(gathered_grads)
        jax.block_until_ready(gathered_ref_grads)

    if args.enable_result_check and args.process_id == 0:
        assert_allclose(ref_output, output, dtype=jnp.bfloat16)
        for ref_grad, gathered_grad in zip(gathered_ref_grads, gathered_grads):
            assert_allclose(ref_grad, gathered_grad, dtype=jnp.bfloat16)


class TestCollectiveDenseGradient(unittest.TestCase):
    """Collective Dense Gradient unittests"""

    def setUp(self):
        self.args = cgemm_parser(
            "Collective Dense Gradient test on multi-GPU with tensor parallelism"
        ).parse_args([])
        self.args.coordinator_address = self.coordinator_address
        self.args.num_processes = self.num_processes
        self.args.process_id = self.process_id
        self.args.local_device_ids = self.local_device_ids
        self.args.num_devices_per_process = self.num_devices_per_process
        self.args.enable_data_parallel = True
        self.args.tensor_parallel_size = _get_dp_and_tp_sizes(self.args)[1]
        _initialize_distributed(self.args)
        # Create mesh once for all tests
        self.mesh = _create_mesh(self.args)
        jax.sharding.set_mesh(self.mesh)
        self.args.enable_result_check = True
        os.environ["NVTE_JAX_ALL_REDUCE_IN_FP32"] = "1"

    def tearDown(self):
        os.environ.pop("NVTE_JAX_ALL_REDUCE_IN_FP32", None)

    def test_te_bf16_all_gather(self):
        """Test Collective Dense Gradient with AllGather"""
        self.args.collective_type = "all_gather"
        run_dense_grad_tests(self.args, self.mesh)

    def test_te_bf16_reduce_scatter(self):
        """Test Collective Dense Gradient with ReduceScatter"""
        self.args.collective_type = "reduce_scatter"
        run_dense_grad_tests(self.args, self.mesh)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 7:  # Need at least the 3 required distributed args
        print("Error: This script requires distributed initialization arguments.")
        print(
            "Usage: python test_dense_grad.py --coordinator-address <address> --num-processes <num>"
            " --process-id <id> [--local-device-ids <ids>] [other args]"
        )
        print(
            "Example: python test_dense_grad.py --coordinator-address localhost:1234"
            " --num-processes 4 --process-id 0"
        )
        print(
            "Example: python test_dense_grad.py --coordinator-address localhost:1234"
            " --num-processes 2 --process-id 0 --local-device-ids 0,1,2,3"
        )
        sys.exit(1)

    args = cgemm_parser(
        "Collective Dense Gradient test on multi-GPU with tensor parallelism"
    ).parse_args([])
    _initialize_distributed(args)
    run_dense_grad_tests(args, mesh=None)
