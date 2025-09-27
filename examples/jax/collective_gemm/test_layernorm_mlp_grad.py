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

from transformer_engine.jax.layernorm_mlp import layernorm_mlp

from transformer_engine.jax.quantize import fp8_autocast
from transformer_engine.jax.cpp_extensions.gemm import (
    CollectiveOpSet,
    CollectiveOp,
    noop_collective_op_set,
)
from transformer_engine.jax.sharding import MeshResource
import transformer_engine.jax.flax as te_flax


def _get_logical_axes():
    input_1_axes = (DP_AXIS, TPSP_AXIS, None)
    weight_1_axes = (None, None, TPSP_AXIS)
    bias_axes_1 = (None, TPSP_AXIS)
    input_2_axes = (DP_AXIS, None, TPSP_AXIS)
    weight_2_axes = (TPSP_AXIS, None)
    bias_axes_2 = (None,)
    return input_1_axes, weight_1_axes, bias_axes_1, input_2_axes, weight_2_axes, bias_axes_2


def _get_operand_sharding(mesh):
    input_1_axes, weight_1_axes, bias_axes_1, input_2_axes, weight_2_axes, bias_axes_2 = (
        _get_logical_axes()
    )
    x_sharding = NamedSharding(mesh, PartitionSpec(*input_1_axes))
    weight_1_sharding = NamedSharding(mesh, PartitionSpec(*weight_1_axes))
    bias_1_sharding = NamedSharding(mesh, PartitionSpec(*bias_axes_1))
    weight_2_sharding = NamedSharding(mesh, PartitionSpec(*weight_2_axes))
    bias_2_sharding = NamedSharding(mesh, PartitionSpec(*bias_axes_2))
    return x_sharding, weight_1_sharding, bias_1_sharding, weight_2_sharding, bias_2_sharding


def _mean_layernorm_mlp(
    x,
    weight_1,
    bias_1,
    weight_2,
    bias_2,
    gamma,
    input_1_axes,
    input_2_axes,
    weight_1_axes,
    weight_2_axes,
    collective_op_sets,
):
    output = layernorm_mlp(
        x,
        gamma,
        beta=None,
        kernels=[weight_1, weight_2],
        biases=[bias_1, bias_2],
        norm_type="rmsnorm",
        dot_1_input_axes=input_1_axes,
        dot_2_input_axes=input_2_axes,
        kernel_1_axes=weight_1_axes,
        kernel_2_axes=weight_2_axes,
        activation_type=("gelu",),
        collective_op_sets=collective_op_sets,
    )
    return jnp.mean(output)


def _value_and_grad_layernorm_mlp(
    x,
    weight_1,
    bias_1,
    weight_2,
    bias_2,
    gamma,
    input_1_axes,
    input_2_axes,
    weight_1_axes,
    weight_2_axes,
    collective_op_sets,
):
    return jax.jit(
        jax.value_and_grad(_mean_layernorm_mlp, (0, 1, 2, 3, 4, 5)), static_argnums=(6, 7, 8, 9, 10)
    )(
        x,
        weight_1,
        bias_1,
        weight_2,
        bias_2,
        gamma,
        input_1_axes,
        input_2_axes,
        weight_1_axes,
        weight_2_axes,
        collective_op_sets,
    )


def run_layernorm_mlp_grad_tests(args, mesh=None):
    """Execute Dense Gradient tests."""
    print(args)
    # Collective GEMM requires Shardy partitioner to be disabled
    jax.config.update("jax_use_shardy_partitioner", False)

    # Initialize distributed with provided arguments
    _initialize_distributed(args)

    mesh = mesh or _create_mesh(args)

    # Create test data
    rng = jax.random.PRNGKey(0)
    rng, x_rng, weight_1_rng, bias_1_rng, weight_2_rng, bias_2_rng, gamma_rng = jax.random.split(
        rng, 7
    )
    x = jax.random.normal(
        x_rng, (args.batch_size, args.seq_len, args.hidden_in), dtype=jnp.bfloat16
    )
    weight_1 = jax.random.normal(
        weight_1_rng, (args.hidden_in, 1, args.hidden_out), dtype=jnp.bfloat16
    ) / jnp.sqrt(args.hidden_in)
    bias_1 = jax.random.normal(bias_1_rng, (1, args.hidden_out), dtype=jnp.bfloat16)
    weight_2 = jax.random.normal(
        weight_2_rng, (args.hidden_out, args.hidden_in), dtype=jnp.bfloat16
    ) / jnp.sqrt(args.hidden_out)
    bias_2 = jax.random.normal(bias_2_rng, (args.hidden_in,), dtype=jnp.bfloat16)
    gamma = jax.random.normal(gamma_rng, (args.hidden_in,), dtype=jnp.bfloat16) / jnp.sqrt(
        args.hidden_in
    )
    collective_op_set_1 = CollectiveOpSet.create(forward_collective_op=CollectiveOp.ALL_GATHER)
    collective_op_set_2 = CollectiveOpSet.create(forward_collective_op=CollectiveOp.REDUCE_SCATTER)
    collective_op_sets = (collective_op_set_1, collective_op_set_2)
    noop_collective_op_sets = (noop_collective_op_set, noop_collective_op_set)

    with mesh, fp8_autocast(
        enabled=False,
        fp8_recipe=None,
        mesh_resource=MeshResource(dp_resource=DP_AXIS, tpsp_resource=TPSP_AXIS),
    ):
        # Get the base axis rules and extend them with TE's rules. This must be done inside fp8_autocast
        axis_rules = flax.linen.get_logical_axis_rules()
        axis_rules += ((TPSP_AXIS, TPSP_AXIS), (DP_AXIS, DP_AXIS))
        te_extended_axis_rules = te_flax.extend_logical_axis_rules(axis_rules)
        with flax.linen.logical_axis_rules(te_extended_axis_rules):
            x_sharding, weight_1_sharding, bias_1_sharding, weight_2_sharding, bias_2_sharding = (
                _get_operand_sharding(mesh)
            )
            x_sharded = jax.device_put(x, x_sharding)
            weight_1_sharded = jax.device_put(weight_1, weight_1_sharding)
            bias_1_sharded = jax.device_put(bias_1, bias_1_sharding)
            weight_2_sharded = jax.device_put(weight_2, weight_2_sharding)
            bias_2_sharded = jax.device_put(bias_2, bias_2_sharding)

            input_1_axes, weight_1_axes, _, input_2_axes, weight_2_axes, _ = _get_logical_axes()
            ref_output, ref_grads = _value_and_grad_layernorm_mlp(
                x_sharded,
                weight_1_sharded,
                bias_1_sharded,
                weight_2_sharded,
                bias_2_sharded,
                gamma,
                input_1_axes,
                input_2_axes,
                weight_1_axes,
                weight_2_axes,
                noop_collective_op_sets,
            )
            output, sharded_grads = _value_and_grad_layernorm_mlp(
                x_sharded,
                weight_1_sharded,
                bias_1_sharded,
                weight_2_sharded,
                bias_2_sharded,
                gamma,
                input_1_axes,
                input_2_axes,
                weight_1_axes,
                weight_2_axes,
                collective_op_sets,
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


class TestCollectiveLayerNormMLPGradient(unittest.TestCase):
    """Collective Dense Gradient unittests"""

    def setUp(self):
        self.args = cgemm_parser(
            "Collective LayerNorm MLP Gradient test on multi-GPU with tensor parallelism"
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

    def test_te_bf16_layernorm_mlp_grad(self):
        """Test Collective Dense Gradient with AllGather"""
        run_layernorm_mlp_grad_tests(self.args, self.mesh)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 7:  # Need at least the 3 required distributed args
        print("Error: This script requires distributed initialization arguments.")
        print(
            "Usage: python test_layernorm_mlp_grad.py --coordinator-address <address>"
            " --num-processes <num> --process-id <id> [--local-device-ids <ids>] [other args]"
        )
        print(
            "Example: python test_layernorm_mlp_grad.py --coordinator-address localhost:1234"
            " --num-processes 4 --process-id 0"
        )
        print(
            "Example: python test_layernorm_mlp_grad.py --coordinator-address localhost:1234"
            " --num-processes 2 --process-id 0 --local-device-ids 0,1,2,3"
        )
        sys.exit(1)

    args = cgemm_parser(
        "Collective LayerNorm MLP Gradient test on multi-GPU with tensor parallelism"
    ).parse_args([])
    _initialize_distributed(args)
    run_layernorm_mlp_grad_tests(args, mesh=None)
