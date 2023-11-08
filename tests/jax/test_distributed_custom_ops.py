# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import pytest
import numpy as np
from functools import partial

import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import NamedSharding

from utils import is_devices_enough
from distributed_configs_helper import *
from distributed_ops_helper import *
from transformer_engine.jax.sharding import global_shard_guard
from transformer_engine.jax.fused_attn import AttnBiasType, AttnMaskType

configs = DistributedConfigsHelper()   # default device count is len(jax.devices())
ops = DistributedOpsHelper()                     # default data type is jnp.float16

@pytest.mark.skipif(not is_devices_enough(configs.device_count),
                    reason='Insufficient number of GPUs, need at least 2.')
@pytest.mark.skipif(not ops.use_custom_partitioning(),
                    reason='TE/JAX version does not support sharding with ' + \
                           'jax.experimental.custom_partitioning.')
class TestCustomPartitioningOpsGenerator:

    @pytest.mark.parametrize('mesh_shape, mesh_names, sharding_type, collective_ref',
                             configs.layernorm_refs)
    @pytest.mark.parametrize('zero_centered_gamma', [False, True])
    def test_layernorm(self, mesh_shape, mesh_names, sharding_type, collective_ref,
                       zero_centered_gamma):
        epsilon = 1e-6

        custom_func = partial(ops.custom_layernorm,
                              zero_centered_gamma=zero_centered_gamma,
                              epsilon=epsilon,
                              sharding_type=sharding_type)
        
        reference_func = partial(ops.reference_layernorm,
                                 zero_centered_gamma=zero_centered_gamma,
                                 epsilon=epsilon)

        batch_size, _, num_heads, head_dim = ops.qkv_shape
        hidden_size = num_heads*head_dim
        input_shape = (batch_size, hidden_size)
        other_shape = (hidden_size, )
        x_ = random.normal(random.PRNGKey(1124), input_shape, dtype=ops.dtype)
        gamma_ = jnp.ones(other_shape, dtype=ops.dtype)
        beta_ = jnp.ones(other_shape, dtype=ops.dtype)

        x_spec, gamma_spec, beta_spec = ops.get_sharding_spec(mesh_names, sharding_type)
        devices = np.asarray(jax.devices()[:configs.device_count]).reshape(*mesh_shape)
        mesh = jax.sharding.Mesh(devices, mesh_names)
        with mesh, global_shard_guard(ops.get_sharding_resource(mesh_names, sharding_type)):
            x_ = jax.device_put(x_, NamedSharding(mesh, x_spec))
            gamma_ = jax.device_put(gamma_, NamedSharding(mesh, gamma_spec))
            beta_ = jax.device_put(beta_, NamedSharding(mesh, beta_spec))
            ops.compare_ops(
                custom_func, reference_func, collective_ref,
                x_, gamma_, beta_, grad_args=(0, 1, 2), dtype=ops.dtype,
                in_shardings=[x_spec, gamma_spec, beta_spec],
                out_shardings=(None, (x_spec, gamma_spec, beta_spec))
            )

    @pytest.mark.parametrize('mesh_shape, mesh_names, sharding_type, collective_ref',
                             configs.layernorm_refs)
    def test_rmsnorm(self, mesh_shape, mesh_names, sharding_type, collective_ref):
        epsilon = 1e-6
        custom_func = partial(ops.custom_rmsnorm, epsilon=epsilon,sharding_type=sharding_type)
        reference_func = partial(ops.reference_rmsnorm, epsilon=epsilon)

        batch_size, _, num_heads, head_dim = ops.qkv_shape
        hidden_size = num_heads*head_dim
        input_shape = (batch_size, hidden_size)
        other_shape = (hidden_size, )
        x_ = random.normal(random.PRNGKey(1124), input_shape, dtype=ops.dtype)
        gamma_ = jnp.ones(other_shape, dtype=ops.dtype)

        x_spec, gamma_spec = ops.get_sharding_spec(mesh_names, sharding_type)
        devices = np.asarray(jax.devices()[:configs.device_count]).reshape(*mesh_shape)
        mesh = jax.sharding.Mesh(devices, mesh_names)
        with mesh, global_shard_guard(ops.get_sharding_resource(mesh_names, sharding_type)):
            x_ = jax.device_put(x_, NamedSharding(mesh, x_spec))
            gamma_ = jax.device_put(gamma_, NamedSharding(mesh, gamma_spec))
            ops.compare_ops(
                custom_func, reference_func, collective_ref,
                x_, gamma_, grad_args=(0, 1), dtype=ops.dtype,
                in_shardings=[x_spec, gamma_spec],
                out_shardings=(None, (x_spec, gamma_spec))
            )

    @pytest.mark.parametrize('mesh_shape, mesh_names, sharding_type, collective_ref',
                             configs.softmax_refs)
    @pytest.mark.parametrize('softmax_type', configs.softmax_types)
    def test_softmax(self, mesh_shape, mesh_names, sharding_type, collective_ref,
                     softmax_type):
        batch_size, seq_len, num_heads, head_dim = ops.qkv_shape
        scale_factor = 1./jnp.sqrt(head_dim)
        
        custom_func = partial(ops.custom_softmax,
                              scale_factor=scale_factor,
                              softmax_type=softmax_type,
                              sharding_type=sharding_type)
        reference_func = partial(ops.reference_softmax,
                                 scale_factor=scale_factor,
                                 softmax_type=softmax_type)

        input_size = (batch_size, num_heads, seq_len, seq_len)
        x_ = random.normal(random.PRNGKey(1124), input_size, dtype=ops.dtype)

        pad_len = int(seq_len * ops.pad_ratio)
        valid_len = seq_len - pad_len
        tokens = jnp.concatenate((jnp.ones((batch_size, valid_len), dtype=ops.dtype),
                                  jnp.zeros((batch_size, pad_len), dtype=ops.dtype)),
                                 axis=-1)
        mask_ = ops.make_mask(tokens, tokens, AttnMaskType.PADDING_MASK)

        x_spec, mask_spec = ops.get_sharding_spec(mesh_names, sharding_type)
        devices = np.asarray(jax.devices()[:configs.device_count]).reshape(*mesh_shape)
        mesh = jax.sharding.Mesh(devices, mesh_names)
        with mesh, global_shard_guard(ops.get_sharding_resource(mesh_names, sharding_type)):
            x_ = jax.device_put(x_, NamedSharding(mesh, x_spec))
            mask_ = jax.device_put(mask_, NamedSharding(mesh, mask_spec))
            ops.compare_ops(
                custom_func, reference_func, collective_ref,
                (0), x_, mask_, grad_args=(0), dtype=ops.dtype,
                in_shardings=[x_spec, mask_spec],
                out_shardings=(None, (x_spec))
            )

    @pytest.mark.parametrize(
        'mesh_shape, mesh_names, sharding_type, attn_bias_type, collective_ref',
        configs.self_attn_refs)
    @pytest.mark.parametrize('attn_mask_type', configs.self_attn_mask_types)
    def test_self_fused_attn(self, mesh_shape, mesh_names, sharding_type, collective_ref,
                             attn_bias_type, attn_mask_type, backend):
        batch_size, seq_len, num_heads, head_dim = ops.qkv_shape
        ops.check_fused_attn_inputs(seq_len, seq_len, head_dim,
                                       ops.pad_ratio, ops.dropout_prob,
                                       attn_bias_type, attn_mask_type, backend)

        dropout_rng = random.PRNGKey(91023051)
        split_rng = random.split(dropout_rng, configs.device_count)
        scale_factor = 1./jnp.sqrt(head_dim)

        custom_func = partial(ops.custom_self_fused_attn,
                              rng_key=split_rng,
                              dropout_prob=ops.dropout_prob,
                              attn_bias_type=attn_bias_type,
                              attn_mask_type=attn_mask_type,
                              scaling_factor=scale_factor,
                              sharding_type=sharding_type)
        reference_func = partial(ops.reference_self_fused_attn,
                                 rng_key=dropout_rng,
                                 dropout_prob=ops.dropout_prob,
                                 attn_bias_type=attn_bias_type,
                                 attn_mask_type=attn_mask_type,
                                 scaling_factor=scale_factor)

        key = random.PRNGKey(1124)
        subkeys = random.split(key, 2)

        qkv_shape = (batch_size, seq_len, 3, num_heads, head_dim)
        qkv_ = random.normal(subkeys[0], qkv_shape, dtype=ops.dtype)
        bias_shape = (1, num_heads, seq_len, seq_len)
        bias_ = random.normal(subkeys[1], bias_shape, dtype=ops.dtype)
        
        pad_len = int(seq_len * ops.pad_ratio)
        valid_len = seq_len - pad_len
        tokens = jnp.concatenate((jnp.ones((batch_size, valid_len), dtype=ops.dtype),
                                  jnp.zeros((batch_size, pad_len), dtype=ops.dtype)),
                                 axis=-1)
        mask_ = ops.make_mask(tokens, tokens, attn_mask_type)

        qkv_spec, bias_spec, mask_spec = ops.get_sharding_spec(mesh_names, sharding_type)
        devices = np.asarray(jax.devices()[:configs.device_count]).reshape(*mesh_shape)
        mesh = jax.sharding.Mesh(devices, mesh_names)
        with mesh, global_shard_guard(ops.get_sharding_resource(mesh_names, sharding_type)):
            qkv_ = jax.device_put(qkv_, NamedSharding(mesh, qkv_spec))
            bias_ = jax.device_put(bias_, NamedSharding(mesh, bias_spec))
            mask_ = jax.device_put(mask_, NamedSharding(mesh, mask_spec))
            ops.compare_ops(
                custom_func, reference_func, collective_ref,
                qkv_, bias_, mask_, grad_args=(0, 1), dtype=ops.dtype,
                in_shardings=[qkv_spec, bias_spec, mask_spec],
                out_shardings=(None, (qkv_spec, bias_spec))
            )

    @pytest.mark.parametrize('mesh_shape, mesh_names, sharding_type, collective_ref',
                             configs.cross_attn_refs)
    @pytest.mark.parametrize('attn_mask_type', configs.cross_attn_mask_types)
    def test_cross_fused_attn(self, mesh_shape, mesh_names, sharding_type, collective_ref,
                              attn_mask_type, backend):
        batch_size, seq_len, num_heads, head_dim = ops.qkv_shape
        ops.check_fused_attn_inputs(seq_len, seq_len, head_dim,
                                       ops.pad_ratio, ops.dropout_prob,
                                       AttnBiasType.NO_BIAS, attn_mask_type, backend)
        
        dropout_rng = random.PRNGKey(91023051)
        split_rng = random.split(dropout_rng, configs.device_count)
        scale_factor = 1./jnp.sqrt(head_dim)

        custom_func = partial(ops.custom_cross_fused_attn,
                              rng_key=split_rng,
                              dropout_prob=ops.dropout_prob,
                              attn_mask_type=attn_mask_type,
                              scaling_factor=scale_factor,
                              sharding_type=sharding_type)
        reference_func = partial(ops.reference_cross_fused_attn,
                                 rng_key=split_rng,
                                 dropout_prob=ops.dropout_prob,
                                 attn_mask_type=attn_mask_type,
                                 scaling_factor=scale_factor)

        key = random.PRNGKey(1124)
        subkeys = random.split(key, 2)

        q_shape = (batch_size, seq_len, num_heads, head_dim)
        q_ = random.normal(subkeys[0], q_shape, dtype=ops.dtype)
        kv_shape = (batch_size, seq_len, 2, num_heads, head_dim)
        kv_ = random.normal(subkeys[1], kv_shape, dtype=ops.dtype)
        
        pad_len = int(seq_len * ops.pad_ratio)
        valid_len = seq_len - pad_len
        tokens = jnp.concatenate((jnp.ones((batch_size, valid_len), dtype=ops.dtype),
                                  jnp.zeros((batch_size, pad_len), dtype=ops.dtype)),
                                 axis=-1)
        mask_ = ops.make_mask(tokens, tokens, attn_mask_type)

        q_spec, kv_spec, mask_spec = ops.get_sharding_spec(mesh_names, sharding_type)
        devices = np.asarray(jax.devices()[:configs.device_count]).reshape(*mesh_shape)
        mesh = jax.sharding.Mesh(devices, mesh_names)
        with mesh, global_shard_guard(ops.get_sharding_resource(mesh_names, sharding_type)):
            q_ = jax.device_put(q_, NamedSharding(mesh, q_spec))
            kv_= jax.device_put(kv_, NamedSharding(mesh, kv_spec))
            mask_ = jax.device_put(mask_, NamedSharding(mesh, mask_spec))
            ops.compare_ops(
                custom_func, reference_func, collective_ref,
                q_, kv_, mask_, grad_args=(0, 1), dtype=ops.dtype,
                in_shardings=[q_spec, kv_spec, mask_spec],
                out_shardings=(None, (q_spec, kv_spec))
            )
