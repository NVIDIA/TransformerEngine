# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import warnings
import pytest

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from distributed_test_base import generate_configs, generate_collectives_count
from distributed_test_base import compare_ops
from utils import pytest_parametrize_wrapper

from transformer_engine.jax import fp8_autocast
from transformer_engine.common import recipe
from transformer_engine.jax.layernorm import layernorm
from transformer_engine.jax.quantize import QuantizerFactory, ScalingMode, is_fp8_available


DTYPES = [jnp.bfloat16, jnp.float32]

NORM_INPUT_SHAPES = {
    "L0": [[64, 64]],
    "L1": [[64, 64]],
    "L2": [[64, 64]],
}

is_fp8_supported, reason = is_fp8_available()
is_mxfp8_supported, reason = is_fp8_available(ScalingMode.MXFP8_1D_SCALING)

SUPPORTED_RECIPES = []
if is_fp8_supported:
    SUPPORTED_RECIPES.append(pytest.param(recipe.DelayedScaling(), id="DelayedScaling"))
    SUPPORTED_RECIPES.append(pytest.param(recipe.Float8CurrentScaling(), id="CurrentScaling"))
if is_mxfp8_supported:
    SUPPORTED_RECIPES.append(pytest.param(recipe.MXFP8BlockScaling(), id="MXFP8BlockScaling"))


class TestDistributedLayernorm:

    def generate_inputs(self, shape, mesh_resource, dtype, shard_weights):
        weight_shape = (shape[-1],)

        x = random.normal(random.PRNGKey(1124), shape, dtype=dtype)
        gamma = jnp.ones(weight_shape, dtype=dtype)
        beta = jnp.ones(weight_shape, dtype=dtype)

        if len(shape) == 2:
            x_pspec = PartitionSpec(mesh_resource.dp_resource, None)
        elif len(shape) == 3:
            x_pspec = PartitionSpec(mesh_resource.dp_resource, None, None)
        else:
            raise NotImplementedError

        g_pspec = b_pspec = (
            PartitionSpec(mesh_resource.dp_resource) if shard_weights else PartitionSpec(None)
        )

        return (x, gamma, beta), (x_pspec, g_pspec, b_pspec)

    def generate_collectives_count_ref(
        self, mesh_resource, ln_type, shape, dtype, mesh_axes, fp8_recipe
    ):
        jax_dtype = jax.dtypes.canonicalize_dtype(dtype)
        is_dp_enabled = mesh_resource.dp_resource is not None
        assert ln_type in ["layernorm", "rmsnorm"]
        all_reduce_loss_bytes = 4  # 1 * FP32
        # for loss, dgamma and dbeta
        # TODO(Jeremy): debug this check because layernorm should always have 2x weights regardless of dp
        weight_count = 2 if (ln_type == "layernorm" and "dp" in mesh_axes) else 1
        allreduce_total_bytes = (
            all_reduce_loss_bytes + weight_count * shape[-1] * jax_dtype.itemsize
        )
        other_bytes = 0
        if fp8_recipe == recipe.Float8CurrentScaling():
            allreduce_total_bytes += jax_dtype.itemsize  # 1 * dtype for the amax reduction
        return generate_collectives_count(
            allreduce=allreduce_total_bytes * int(is_dp_enabled), allgather=0, other=other_bytes
        )

    @pytest.mark.parametrize("device_count,mesh_shape,mesh_axes,mesh_resource", generate_configs())
    @pytest_parametrize_wrapper("data_shape", NORM_INPUT_SHAPES)
    @pytest_parametrize_wrapper("dtype", DTYPES)
    @pytest_parametrize_wrapper("zero_centered_gamma", [False, True])
    @pytest_parametrize_wrapper("shard_weights", [False, True])
    @pytest_parametrize_wrapper("fp8_recipe", SUPPORTED_RECIPES)
    @pytest_parametrize_wrapper("use_shardy", [False, True])
    def test_layernorm(
        self,
        device_count,
        mesh_shape,
        mesh_axes,
        mesh_resource,
        data_shape,
        dtype,
        zero_centered_gamma,
        shard_weights,
        fp8_recipe,
        use_shardy,
    ):
        jax.config.update("jax_use_shardy_partitioner", use_shardy)
        epsilon = 1e-6
        ln_type = "layernorm"
        q_dtype = jnp.float8_e4m3fn

        def target_func(x, gamma, beta):
            quantizer = QuantizerFactory.create_set().x
            return jnp.mean(
                layernorm(
                    x, gamma, beta, ln_type, zero_centered_gamma, epsilon, quantizer=quantizer
                )
            )

        def ref_func(x, gamma, beta):
            x_ = jnp.asarray(x, jnp.float32)
            mean = jnp.mean(x_, axis=-1, keepdims=True)
            var = jnp.mean(jnp.square(x_ - mean), axis=-1, keepdims=True)
            normed_input = (x_ - mean) * jax.lax.rsqrt(var + epsilon)
            if zero_centered_gamma:
                output = jnp.asarray(normed_input * (gamma + 1) + beta).astype(x.dtype)
            else:
                output = jnp.asarray(normed_input * gamma + beta).astype(x.dtype)
            return jnp.mean(output)

        (x, gamma, beta), (x_pspec, g_pspec, b_pspec) = self.generate_inputs(
            data_shape, mesh_resource, dtype, shard_weights
        )
        collective_count_ref = self.generate_collectives_count_ref(
            mesh_resource, ln_type, data_shape, dtype, mesh_axes, fp8_recipe
        )
        devices = np.asarray(jax.devices()[:device_count]).reshape(*mesh_shape)
        mesh = Mesh(devices, mesh_axes)
        with mesh, fp8_autocast(enabled=True, fp8_recipe=fp8_recipe, mesh_resource=mesh_resource):
            x_ = jax.device_put(x, NamedSharding(mesh, x_pspec))
            gamma_ = jax.device_put(gamma, NamedSharding(mesh, g_pspec))
            beta_ = jax.device_put(beta, NamedSharding(mesh, b_pspec))

            with warnings.catch_warnings(record=True) as warns:
                try:
                    compare_ops(
                        target_func,
                        ref_func,
                        [x_, gamma_, beta_],
                        collective_count_ref,
                        grad_args=(0, 1, 2),
                        metric_fwd_dtype=q_dtype,
                        metric_bwd_dtype=q_dtype,
                        in_shardings=(x_pspec, g_pspec, b_pspec),
                        out_shardings=(None, (x_pspec, g_pspec, b_pspec)),
                    )
                except AssertionError as err:
                    # Layernorm should still produce the correct numerical result with
                    # gamma/beta sharded. However, the collective count may not be the same
                    # when XLA is forced to unshard gamma and/or beta. We can catch
                    # and ignore that specific error here.
                    if (
                        g_pspec[-1] is None and b_pspec[-1] is None
                    ) or "Expected collective count" not in str(err):
                        raise err
                finally:
                    for w in warns:
                        assert "Enforcing no sharding of parameters hidden dim!" in str(w), (
                            "Layernorm primitive did not raise the correct warning for "
                            "unsupported sharding of gamma and/or beta"
                        )

    @pytest.mark.parametrize("device_count,mesh_shape,mesh_axes,mesh_resource", generate_configs())
    @pytest_parametrize_wrapper("data_shape", NORM_INPUT_SHAPES)
    @pytest_parametrize_wrapper("dtype", DTYPES)
    @pytest_parametrize_wrapper("shard_weights", [False, True])
    @pytest_parametrize_wrapper("fp8_recipe", SUPPORTED_RECIPES)
    @pytest_parametrize_wrapper("use_shardy", [False, True])
    def test_rmsnorm(
        self,
        device_count,
        mesh_shape,
        mesh_axes,
        mesh_resource,
        data_shape,
        dtype,
        shard_weights,
        fp8_recipe,
        use_shardy,
    ):
        jax.config.update("jax_use_shardy_partitioner", use_shardy)
        epsilon = 1e-6
        ln_type = "rmsnorm"
        q_dtype = jnp.float8_e4m3fn

        def target_func(x, gamma):
            quantizer = QuantizerFactory.create_set().x
            return jnp.mean(layernorm(x, gamma, None, ln_type, False, epsilon, quantizer=quantizer))

        def ref_func(x, gamma):
            x = jnp.asarray(x, jnp.float32)
            mean2 = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
            y = jnp.asarray(x * jax.lax.rsqrt(mean2 + epsilon), dtype)
            output = y * gamma
            return jnp.mean(output)

        (x, gamma, _), (x_pspec, g_pspec, _) = self.generate_inputs(
            data_shape, mesh_resource, dtype, shard_weights
        )
        collective_count_ref = self.generate_collectives_count_ref(
            mesh_resource, ln_type, data_shape, dtype, mesh_axes, fp8_recipe
        )
        devices = np.asarray(jax.devices()[:device_count]).reshape(*mesh_shape)
        mesh = Mesh(devices, mesh_axes)
        with mesh, fp8_autocast(enabled=True, fp8_recipe=fp8_recipe, mesh_resource=mesh_resource):
            x_ = jax.device_put(x, NamedSharding(mesh, x_pspec))
            gamma_ = jax.device_put(gamma, NamedSharding(mesh, g_pspec))

            with warnings.catch_warnings(record=True) as warns:
                try:
                    compare_ops(
                        target_func,
                        ref_func,
                        [x_, gamma_],
                        collective_count_ref,
                        grad_args=(0, 1),
                        metric_fwd_dtype=q_dtype,
                        metric_bwd_dtype=q_dtype,
                        in_shardings=(x_pspec, g_pspec),
                        out_shardings=(None, (x_pspec, g_pspec)),
                    )
                except AssertionError as err:
                    # RmsNorm should still produce the correct numerical result with
                    # gamma/beta sharded. However, the collective count may not be the same
                    # when XLA is forced to unshard gamma. We can catch
                    # and ignore that specific error here.
                    if g_pspec[-1] is None or "Expected collective count" not in str(err):
                        raise err
                finally:
                    for w in warns:
                        assert "Enforcing no sharding of parameters hidden dim!" in str(w), (
                            "RmsNorm primitive did not raise the correct warning for "
                            "unsupported sharding of gamma and/or beta"
                        )
