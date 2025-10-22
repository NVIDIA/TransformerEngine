# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from functools import partial

from distributed_test_base import generate_configs
from utils import assert_allclose, pytest_parametrize_wrapper

import transformer_engine.jax.cpp_extensions as tex
from transformer_engine.jax import autocast
from transformer_engine.jax.dense import dense


DTYPES = [jnp.bfloat16]

GEMM_INPUT_SHAPES = [[256, 128, 256]]  # [batch, seq_len, hidden_in]

WEIGHT_SHAPES = [[256, 256]]  # [hidden_in, hidden_out]


def _generate_inputs(input_shape, weight_shape, dtype):
    """Generate test inputs for GEMM operations"""
    _, _, hidden_in = input_shape
    hidden_in_w, hidden_out = weight_shape
    assert hidden_in == hidden_in_w, f"Dimension mismatch: {hidden_in} != {hidden_in_w}"

    bias_shape = (hidden_out,)

    # Generate random inputs
    x = random.normal(random.PRNGKey(1124), input_shape, dtype=dtype)
    weight = random.normal(random.PRNGKey(2248), weight_shape, dtype=dtype) / jnp.sqrt(hidden_in_w)
    bias = random.normal(random.PRNGKey(3372), bias_shape, dtype=dtype) / jnp.sqrt(hidden_out)

    return x, weight, bias


def _get_sharding_for_gemm(mesh, mesh_resource, partition_layout="rowwise"):
    """Get sharding patterns for GEMM inputs and outputs"""

    dp_axis = mesh_resource.dp_resource
    tp_axis = mesh_resource.tpsp_resource

    if partition_layout == "colwise":
        x_spec = PartitionSpec(dp_axis, None, None)
        weight_spec = PartitionSpec(None, tp_axis)
        bias_spec = PartitionSpec(tp_axis)
        output_spec = PartitionSpec(dp_axis, None, tp_axis)
    elif partition_layout == "rowwise":
        x_spec = PartitionSpec(dp_axis, None, tp_axis)
        weight_spec = PartitionSpec(tp_axis, None)
        bias_spec = PartitionSpec(None)
        output_spec = PartitionSpec(dp_axis, None, None)
    else:
        raise ValueError(f"Invalid partition: {partition_layout}")

    x_sharding = NamedSharding(mesh, x_spec)
    weight_sharding = NamedSharding(mesh, weight_spec)
    bias_sharding = NamedSharding(mesh, bias_spec)
    output_sharding = NamedSharding(mesh, output_spec)

    return x_sharding, weight_sharding, bias_sharding, output_sharding


@partial(jax.jit, static_argnames=("contracting_dims", "output_sharding"))
def _jitted_gemm(x, weight, bias, contracting_dims, output_sharding):
    output = tex.gemm(
        x,
        weight,
        bias=bias,
        contracting_dims=contracting_dims,
        fuse_bias=True,
    )
    if output_sharding is not None:
        output = jax.lax.with_sharding_constraint(output, output_sharding)
    return output


# TODO(Phuong):
# 1. Add supported recipes after FP4 is added
# 2. Add communication type/byte checks
class TestDistributedDense:
    """Test distributed GEMM without collective operations vs JAX dot"""

    @pytest_parametrize_wrapper(
        "device_count,mesh_shape,mesh_axes,mesh_resource",
        generate_configs(),
    )
    @pytest_parametrize_wrapper("dtype", DTYPES)
    @pytest_parametrize_wrapper("input_shape", GEMM_INPUT_SHAPES)
    @pytest_parametrize_wrapper("weight_shape", WEIGHT_SHAPES)
    @pytest_parametrize_wrapper("partition", ["rowwise", "colwise"])
    def test_distributed_gemm(
        self,
        device_count,
        mesh_shape,
        mesh_axes,
        mesh_resource,
        dtype,
        input_shape,
        weight_shape,
        partition,
    ):
        """Test TE GEMM against JAX dot with bf16 dtype"""
        devices = np.asarray(jax.devices()[:device_count]).reshape(*mesh_shape)
        mesh = Mesh(devices, mesh_axes)

        # Generate inputs
        x, weight, bias = _generate_inputs(input_shape, weight_shape, dtype)

        # Get sharding patterns
        x_sharding, weight_sharding, bias_sharding, output_sharding = _get_sharding_for_gemm(
            mesh, mesh_resource, partition_layout=partition
        )

        # Shard inputs
        x_sharded = jax.device_put(x, x_sharding)
        weight_sharded = jax.device_put(weight, weight_sharding)
        bias_sharded = jax.device_put(bias, bias_sharding)

        contracting_dims = ((2,), (0,))  # Contract on hidden_in dimension

        with mesh, autocast(enabled=False, mesh_resource=mesh_resource):
            # TE GEMM result
            te_result = _jitted_gemm(
                x_sharded,
                weight_sharded,
                bias_sharded,
                contracting_dims=contracting_dims,
                output_sharding=output_sharding,
            )

            # JAX dot reference result
            jax_result = (
                jax.lax.dot_general(
                    x_sharded, weight_sharded, dimension_numbers=(contracting_dims, ((), ()))
                )
                + bias_sharded
            )

            assert te_result.sharding == jax_result.sharding
            # Ensure computation is complete
            jax.block_until_ready(te_result)
            jax.block_until_ready(jax_result)

            # Gather results for comparison
            gathered_te = jax.lax.with_sharding_constraint(
                te_result, NamedSharding(mesh, PartitionSpec(None))
            )
            gathered_jax = jax.lax.with_sharding_constraint(
                jax_result, NamedSharding(mesh, PartitionSpec(None))
            )

            # Compare results
            assert_allclose(gathered_te, gathered_jax, dtype=dtype)

    def _te_sum_dense(self, x, weight, bias, contracting_dims):
        """TE GEMM function for gradient testing"""
        return jnp.sum(dense(x, weight, bias=bias, contracting_dims=contracting_dims))

    def _jax_sum_dense(self, x, weight, bias, contracting_dims):
        """JAX dot function for gradient testing"""
        result = (
            jax.lax.dot_general(x, weight, dimension_numbers=(contracting_dims, ((), ()))) + bias
        )
        return jnp.sum(result)

    @pytest_parametrize_wrapper(
        "device_count,mesh_shape,mesh_axes,mesh_resource",
        generate_configs(),
    )
    @pytest_parametrize_wrapper("dtype", DTYPES)
    @pytest_parametrize_wrapper("input_shape", GEMM_INPUT_SHAPES)
    @pytest_parametrize_wrapper("weight_shape", WEIGHT_SHAPES)
    @pytest_parametrize_wrapper("partition", ["rowwise", "colwise"])
    def test_te_distributed_dense_grad(
        self,
        device_count,
        mesh_shape,
        mesh_axes,
        mesh_resource,
        dtype,
        input_shape,
        weight_shape,
        partition,
    ):
        """Test TE GEMM gradients against JAX dot gradients"""
        devices = np.asarray(jax.devices()[:device_count]).reshape(*mesh_shape)
        mesh = Mesh(devices, mesh_axes)

        # Generate inputs
        x, weight, bias = _generate_inputs(input_shape, weight_shape, dtype)

        # Get sharding patterns
        x_sharding, weight_sharding, bias_sharding, output_sharding = _get_sharding_for_gemm(
            mesh, mesh_resource, partition_layout=partition
        )

        x_sharded = jax.device_put(x, x_sharding)
        weight_sharded = jax.device_put(weight, weight_sharding)
        bias_sharded = jax.device_put(bias, bias_sharding)

        contracting_dims = ((2,), (0,))

        with mesh, autocast(enabled=False, mesh_resource=mesh_resource):
            # Test gradients w.r.t. all inputs
            te_grad_func = jax.jit(
                jax.value_and_grad(self._te_sum_dense, argnums=(0, 1, 2)),
                static_argnames=("contracting_dims",),
            )
            jax_grad_func = jax.jit(
                jax.value_and_grad(self._jax_sum_dense, argnums=(0, 1, 2)),
                static_argnames=("contracting_dims",),
            )

            te_val, te_grads = te_grad_func(
                x_sharded, weight_sharded, bias_sharded, contracting_dims
            )
            jax_val, jax_grads = jax_grad_func(
                x_sharded, weight_sharded, bias_sharded, contracting_dims
            )

            # Compare forward pass
            assert_allclose(te_val, jax_val, dtype=dtype)

            # Compare gradients
            for i, (te_grad, jax_grad) in enumerate(zip(te_grads, jax_grads)):
                te_grad_spec = tuple(i for i in te_grad.sharding.spec if i is not None)
                jax_grad_spec = tuple(i for i in jax_grad.sharding.spec if i is not None)
                assert te_grad_spec == jax_grad_spec, f"Gradient sharding mismatch at te_grads[{i}]"
                gathered_te_grad = jax.lax.with_sharding_constraint(
                    te_grad, NamedSharding(mesh, PartitionSpec(None))
                )
                gathered_jax_grad = jax.lax.with_sharding_constraint(
                    jax_grad, NamedSharding(mesh, PartitionSpec(None))
                )
                assert_allclose(
                    gathered_te_grad,
                    gathered_jax_grad,
                    dtype=dtype,
                    err_msg=f"Gradient mismatch for argument {i}",
                )


if __name__ == "__main__":
    unittest.main()
