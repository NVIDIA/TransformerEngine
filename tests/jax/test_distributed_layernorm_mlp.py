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
from transformer_engine.jax.flax import LayerNormMLP
from transformer_engine.jax.mlp import fused_layernorm_fp8_mlp
from transformer_engine.jax.sharding import HIDDEN_AXES, HIDDEN_TP_AXES, \
    BATCH_AXES, SEQLEN_TP_AXES, SEQLEN_AXES, \
    W_NO_SHARD_AXES, W_FSDP_AXES, W_TP_AXES, W_JOINED_AXES
from transformer_engine.jax.sharding import MeshResource
from utils import assert_allclose, is_devices_enough


is_fp8_supported, reason = is_fp8_available()
DTYPES = [jnp.bfloat16, jnp.float16]
INPUT_SHAPE = [[64, 64, 32]]    # [seqlen, batch, hidden_in]

LAYERNORM_INPUT_AXES = (BATCH_AXES, SEQLEN_TP_AXES, HIDDEN_AXES)
DOT_1_INPUT_AXES = (BATCH_AXES, SEQLEN_AXES, HIDDEN_AXES)
DOT_2_INPUT_AXES = (BATCH_AXES, SEQLEN_AXES, HIDDEN_TP_AXES)
INTERMEDIATE = 16


def generate_fsdp_and_tp_configs():
    configs = []
    if is_devices_enough(2):
        configs.append(
            [2, (1, 2), ('fsdp', 'tp'),
             MeshResource(fsdp_resource='fsdp', tp_resource='tp')])

    if is_devices_enough(4):
        configs.append(
            [4, (2, 2), ('fsdp', 'tp'),
             MeshResource(fsdp_resource='fsdp', tp_resource='tp')])
    return configs


class TestDistributedLayernormMLP:

    def generate_inputs(self, input_shape, activation_type, use_bias, dtype):
        # m, n, k = shape
        batch, seqlen, hidden_in = input_shape
        hidden_out = hidden_in

        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 6)

        x = jax.random.normal(subkeys[0], (batch, seqlen, hidden_in), dtype)
        gamma = jax.random.normal(subkeys[5], (hidden_in,), dtype=dtype)
        k1 = jax.random.normal(subkeys[1], (hidden_in, len(activation_type),
                                            INTERMEDIATE), dtype) / jnp.sqrt(hidden_in)
        k2 = jax.random.normal(subkeys[2], (INTERMEDIATE, hidden_out),
                               dtype) / jnp.sqrt(INTERMEDIATE)
        if use_bias:
            b1 = jax.random.normal(subkeys[3], (len(activation_type), INTERMEDIATE), dtype)
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
    @pytest.mark.parametrize('mesh_config', generate_fsdp_and_tp_configs())
    @pytest.mark.parametrize('input_shape', INPUT_SHAPE)
    @pytest.mark.parametrize('activation_type', [("gelu",),
                                                 ('gelu', 'linear')])
    @pytest.mark.parametrize('dtype', DTYPES)
    @pytest.mark.parametrize('use_bias', [True, False])
    def test_layernorm_fp8_mlp_primitive(self, mesh_config,
                                     activation_type, use_bias,
                                     input_shape, dtype):
        # Only test with tp = 2 as dp is not used
        device_count, mesh_shape, mesh_axes, mesh_resource = mesh_config
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
        with fp8_autocast(enabled=True):
            single_fwd, single_grads = single_jitter(*inputs, *static_inputs)

        # Multi GPUs
        devices = np.asarray(jax.devices()[:device_count]).reshape(*mesh_shape)
        mesh = Mesh(devices, mesh_axes)
        with mesh, fp8_autocast(enabled=True, mesh_resource=mesh_resource):
            k1_sharding = NamedSharding(mesh, PartitionSpec('fsdp', None, 'tp'))
            k2_sharding = NamedSharding(mesh, PartitionSpec('tp', 'fsdp'))
            k1_ = jax.device_put(k1, k1_sharding)
            k2_ = jax.device_put(k2, k2_sharding)
            if use_bias:
                b1_sharding = NamedSharding(mesh, PartitionSpec(None, 'tp'))
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

            multi_fwd, multi_grads = multi_jitter(*multi_inputs, *static_inputs, True)

        assert_allclose(multi_fwd, single_fwd, dtype=dtype)
        for i in range(len(inputs)):
            if multi_grads[i] is not None:
                assert_allclose(multi_grads[i], single_grads[i], dtype=dtype)

    def _test_layernorm_mlp(self, mesh_config, activation_type, use_bias,
                            input_shape, dtype, use_fp8):
        batch, seqlen, hidden_in = input_shape
        layernorm_type = 'rmsnorm'

        rng = jax.random.PRNGKey(0)
        subkeys = jax.random.split(rng, 2)

        x = jax.random.normal(subkeys[0], (batch, seqlen, hidden_in), dtype)
        init_rngs = {'params': subkeys[1]}

        # Single GPUs
        with fp8_autocast(enabled=use_fp8):
            ln_mlp_single = LayerNormMLP(
                layernorm_type=layernorm_type,
                transpose_batch_sequence=False, # input: [batch, seqlen, hidden]
                intermediate_dim=INTERMEDIATE,
                activations=activation_type,
                dtype=dtype,
                use_bias=use_bias,
            )
            params_single = ln_mlp_single.init(init_rngs, x)
            mlp_out_single, ln_out_single = ln_mlp_single.apply(params_single, x, deterministic=True)

        # Multi GPUs
        device_count, mesh_shape, mesh_axes, mesh_resource = mesh_config
        devices = np.asarray(jax.devices()[:device_count]).reshape(*mesh_shape)
        mesh = Mesh(devices, mesh_axes)
        with mesh, fp8_autocast(enabled=use_fp8, mesh_resource=mesh_resource):
            ln_mlp_sharded = LayerNormMLP(
                layernorm_type=layernorm_type,
                transpose_batch_sequence=False,
                intermediate_dim=INTERMEDIATE,
                activations=activation_type,
                dtype=dtype,
                scale_axes=(W_NO_SHARD_AXES,),
                ln_bias_axes=(W_NO_SHARD_AXES,),
                kernel_axes_1=(W_FSDP_AXES, W_JOINED_AXES, W_TP_AXES),
                kernel_axes_2=(W_TP_AXES, W_FSDP_AXES),
                use_bias=use_bias,
                bias_axes_1=(W_JOINED_AXES, W_TP_AXES),
                bias_axes_2=(W_NO_SHARD_AXES,),
                layernorm_input_axes=LAYERNORM_INPUT_AXES,
                dot_1_input_axes=DOT_1_INPUT_AXES,
                dot_2_input_axes=DOT_2_INPUT_AXES,
                name='mlp'
            )
            params_sharded = ln_mlp_sharded.init(init_rngs, x)
            mlp_out_sharded, ln_out_sharded = ln_mlp_sharded.apply(params_sharded, x,
                                                                   deterministic=True)

        # Make sure params values are the same
        jax.tree_util.tree_map(lambda x, y: assert_allclose(x, y, dtype=dtype), \
                               params_sharded['params'], params_single['params'])
        assert_allclose(ln_out_sharded, ln_out_sharded, dtype=dtype)
        assert_allclose(mlp_out_sharded, mlp_out_single, dtype=dtype)

    @pytest.mark.parametrize('input_shape', INPUT_SHAPE)
    @pytest.mark.parametrize('mesh_config', generate_fsdp_and_tp_configs())
    @pytest.mark.parametrize('activation_type', [("gelu",),
                                                 ('silu', 'linear'),
                                                 ('gelu', 'gelu')])
    @pytest.mark.parametrize('dtype', DTYPES)
    @pytest.mark.parametrize('use_bias', [True, False])
    def test_layernorm_mlp_layer(self, mesh_config, activation_type, use_bias,
                                 input_shape, dtype):
        self._test_layernorm_mlp(mesh_config, activation_type, use_bias, input_shape, dtype,
                                 use_fp8=False)

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize('mesh_config', generate_fsdp_and_tp_configs())
    @pytest.mark.parametrize('activation_type', [("gelu",),
                                                 ('gelu', 'linear'),
                                                 ('gelu', 'gelu')])
    @pytest.mark.parametrize('use_bias', [True, False])
    @pytest.mark.parametrize('input_shape', INPUT_SHAPE)
    @pytest.mark.parametrize('dtype', DTYPES)
    def test_layernorm_fp8_mlp_layer(self, mesh_config, activation_type, use_bias,
                                     input_shape, dtype):
        self._test_layernorm_mlp(mesh_config, activation_type, use_bias, input_shape, dtype,
                                 use_fp8=True)
