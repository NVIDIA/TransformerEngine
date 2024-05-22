# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest

import jax
import jax.numpy as jnp
import numpy as np

from typing import Callable, Sequence, Union

from jax.sharding import Mesh, NamedSharding, PartitionSpec
""" from distributed_test_base import compare_ops """
from transformer_engine.jax.fp8 import FP8MetaPackage, FP8Helper
from transformer_engine.jax.fp8 import is_fp8_available
from transformer_engine.jax import fp8_autocast
from transformer_engine.jax.mlp import fused_layernorm_fp8_mlp
from transformer_engine.jax.sharding import HIDDEN_AXES, HIDDEN_TP_AXES, \
    BATCH_AXES, SEQLEN_TP_AXES, SEQLEN_AXES
from transformer_engine.jax.sharding import with_sharding_constraint_by_logical_axes
from distributed_test_base import generate_configs
from utils import assert_allclose


is_fp8_supported, reason = is_fp8_available()
DTYPES = [jnp.bfloat16, jnp.float16]

LAYERNORM_INPUT_AXES = (BATCH_AXES, SEQLEN_TP_AXES, HIDDEN_AXES)
DOT_1_INPUT_AXES = (BATCH_AXES, SEQLEN_AXES, HIDDEN_AXES)
DOT_2_INPUT_AXES = (BATCH_AXES, SEQLEN_AXES, HIDDEN_TP_AXES)


class TestDistributedLayernormMLP:

    def generate_inputs(self, input_shape, activation_type, use_bias, dtype):
        # m, n, k = shape
        batch, seqlen, hidden_in = input_shape
        hidden_out = 32
        intermediate = 32

        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 6)

        x = jax.random.normal(subkeys[0], (batch, seqlen, hidden_in), dtype)
        gamma = jax.random.normal(subkeys[5], (hidden_in,), dtype=dtype)
        k1 = jax.random.normal(subkeys[1], (hidden_in, len(activation_type),
                                            intermediate), dtype) / jnp.sqrt(hidden_in)
        k2 = jax.random.normal(subkeys[2], (intermediate, hidden_out),
                               dtype) / jnp.sqrt(intermediate)
        if use_bias:
            b1 = jax.random.normal(subkeys[3], (len(activation_type), intermediate), dtype)
            b2 = jax.random.normal(subkeys[4], (hidden_out,), dtype)
        else:
            b1 = None
            b2 = None

        return (x, gamma, k1, k2, b1, b2)


    def layernorm_fp8_mlp_prim_func(self, x: jnp.ndarray, ln_scale: jnp.ndarray,
                          kernel_1: jnp.ndarray, kernel_2: jnp.ndarray,
                          bias_1: jnp.ndarray, bias_2: jnp.ndarray,
                          fp8_max: jnp.ndarray, fp8_metas_amax: jnp.ndarray,
                          fp8_metas_scale: jnp.ndarray, fp8_metas_scale_inv: jnp.ndarray,
                          layernorm_type: str = "rmsnorm",
                          activation_type: Sequence[Union[str, Callable]] = ('gelu',),
                          use_bias: bool = True,
                          multi_gpus: bool = False,
                          ) -> jnp.ndarray:

        fp8_meta_pkg = FP8MetaPackage(2, fp8_max, fp8_metas_amax, fp8_metas_scale,
                                      fp8_metas_scale_inv)

        if (multi_gpus):
            layernorm_input_axes = LAYERNORM_INPUT_AXES
            dot_1_input_axes = DOT_1_INPUT_AXES
            dot_2_input_axes = DOT_2_INPUT_AXES
        else:
            layernorm_input_axes = None
            dot_1_input_axes = None
            dot_2_input_axes = None

        # out = ((x * kernel_1) + bias_1) * kernel_2 + bias_2
        return jnp.mean(
            fused_layernorm_fp8_mlp(x, ln_scale, None,
                                    [kernel_1, kernel_2], [bias_1, bias_2],
                                    fp8_meta_pkg,
                                    layernorm_type,
                                    layernorm_input_axes=layernorm_input_axes,
                                    dot_1_input_axes=dot_1_input_axes,
                                    dot_2_input_axes=dot_2_input_axes,
                                    activation_type=activation_type,
                                    use_bias=use_bias))

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize('input_shape', [[16, 32, 16]])  # [seqlen, batch, hidden_in]
    @pytest.mark.parametrize('activation_type', [("gelu",)])
                                                 #('gelu', 'linear')])
    @pytest.mark.parametrize('dtype', DTYPES)
    @pytest.mark.parametrize('use_bias', [True, False])
    def test_layernorm_fp8_mlp_primitive(self,
                                     activation_type, use_bias,
                                     input_shape, dtype):
        # Only test with tp = 2 as dp is not used
        device_count, mesh_shape, mesh_axes, mesh_resource = generate_configs()[1]
        layernorm_type = 'rmsnorm'

        fp8_max = FP8Helper.generate_fp8_max_array(FP8Helper.NUM_META_PER_GEMM * 2)
        fp8_metas_amax = jnp.zeros((FP8Helper.NUM_META_PER_GEMM * 2,
                                    FP8Helper.AMAX_HISTORY_LEN), jnp.float32)
        fp8_metas_scale = jnp.ones((FP8Helper.NUM_META_PER_GEMM * 2, 1), jnp.float32)
        fp8_metas_scale_inv = jnp.ones((FP8Helper.NUM_META_PER_GEMM * 2, 1), jnp.float32)

        inputs = [x, gamma, k1, k2, b1, b2] = \
            self.generate_inputs(input_shape, activation_type, use_bias, dtype)
        inputs = [*inputs, fp8_max, fp8_metas_amax, fp8_metas_scale, fp8_metas_scale_inv]
        static_inputs = [layernorm_type,
                         activation_type,
                         use_bias]
        value_and_grad_func = jax.value_and_grad(self.layernorm_fp8_mlp_prim_func,
                                                 argnums=range(len(inputs)))

        # Single GPU
        single_jitter = jax.jit(value_and_grad_func,
                                static_argnums=range(len(inputs),
                                                     len(static_inputs)+len(inputs)))
        # Multi GPUs
        devices = np.asarray(jax.devices()[:device_count]).reshape(*mesh_shape)
        with Mesh(devices=devices, axis_names=mesh_axes) as shard_mesh:
            k1_sharding = NamedSharding(shard_mesh,
                                        PartitionSpec(None, None, mesh_resource.tp_resource))
            k2_sharding = NamedSharding(shard_mesh,
                                        PartitionSpec(mesh_resource.tp_resource, None))
            k1_ = jax.device_put(k1, k1_sharding)
            k2_ = jax.device_put(k2, k2_sharding)
            if use_bias:
                b1_sharding = NamedSharding(shard_mesh,
                                        PartitionSpec(None, mesh_resource.tp_resource))
                b1_ = jax.device_put(b1, b1_sharding)
            else:
                b1_sharding = b1_ = None
            multi_inputs = [*inputs[:2], k1_, k2_, b1_, *inputs[5:]]

            # Position ref for sharding pspec lists
            #   x, gamma, k1, k2, b1,
            #   b2, fp8_max, fp8_metas_amax, fp8_metas_scale, fp8_metas_scale_inv
            in_shardings = (None, None, k1_sharding, k2_sharding, b1_sharding,
                            None, None, None, None, None)
            out_shardings = (None, (None, None, k1_sharding, k2_sharding, b1_sharding,
                                    None, None, None, None, None))

            multi_jitter = jax.jit(
                value_and_grad_func,
                in_shardings=in_shardings,
                out_shardings=out_shardings,
                static_argnums=range(len(multi_inputs),
                                     len(static_inputs)+len(multi_inputs)+1))   # +1 for multi_gpus

            with fp8_autocast(enabled=True, mesh_resource=mesh_resource):
                single_fwd, single_grads = single_jitter(*inputs, *static_inputs)
                multi_fwd, multi_grads = multi_jitter(*multi_inputs, *static_inputs, True)

                assert_allclose(multi_fwd, single_fwd, dtype=dtype)

                for i in range(len(inputs)):
                    if multi_grads[i] is not None:
                        assert_allclose(multi_grads[i], single_grads[i], dtype=dtype)
