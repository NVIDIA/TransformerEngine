# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
#
# Regression tests for TE-JAX custom ops with xmap-based sharding
# https://jax.readthedocs.io/en/latest/notebooks/xmap_tutorial.html
#
import os
import pytest
import numpy as np
from functools import partial

import jax
import jax.numpy as jnp
from jax import random

from utils import is_devices_enough
from sharding_configs import *
from custom_ops_helper import *
from transformer_engine.jax.sharding import global_shard_guard
from transformer_engine.jax.fused_attn import AttnBiasType, AttnMaskType

configs = ShardingConfigs(num_gpus=8)
helper = CustomOpsTestHelper()

@pytest.fixture(name="backend", params=[FusedAttnBackend.Max512, FusedAttnBackend.Arbitrary])
def fixture_backend(request):
    backend = request.param
    os.environ["NVTE_FUSED_ATTN_BACKEND"] = backend.value
    yield backend
    os.environ["NVTE_FUSED_ATTN_BACKEND"] = ""


@pytest.mark.skipif(not is_devices_enough(configs.device_count), reason='Num of GPU is not enough')
class TestXmapOpsGenerator:

    @pytest.mark.parametrize('mesh_shape, mesh_names, sharding_type, collective_ref',
                             configs.layernorm_refs)
    @pytest.mark.parametrize('zero_centered_gamma', [False, True])
    def test_layernorm(self, mesh_shape, mesh_names, sharding_type, collective_ref,
                       zero_centered_gamma):
        epsilon = 1e-6
        custom_func = partial(helper.custom_layernorm,
                              zero_centered_gamma=zero_centered_gamma,
                              epsilon=epsilon,
                              sharding_type=sharding_type)
        reference_func = partial(helper.reference_layernorm,
                                 zero_centered_gamma=zero_centered_gamma,
                                 epsilon=epsilon)

        batch_size, _, num_heads, head_dim = helper.qkv_shape
        hidden_size = num_heads*head_dim
        input_shape = (batch_size, hidden_size)
        other_shape = (hidden_size, )

        devices = np.asarray(jax.devices()[:configs.device_count]).reshape(*mesh_shape)
        mesh = jax.sharding.Mesh(devices, mesh_names)
        with mesh, global_shard_guard(helper.get_sharding_resource(mesh_names, sharding_type)[0]):
            x_ = random.normal(random.PRNGKey(1124), input_shape, dtype=helper.dtype)
            gamma_ = jnp.ones(other_shape, dtype=helper.dtype)
            beta_ = jnp.ones(other_shape, dtype=helper.dtype)
            helper.compare_ops(custom_func, reference_func, collective_ref,
                               x_, gamma_, beta_, grad_args=(0, 1, 2))

    @pytest.mark.parametrize('mesh_shape, mesh_names, sharding_type, collective_ref',
                             configs.layernorm_refs)
    def test_rmsnorm(self, mesh_shape, mesh_names, sharding_type, collective_ref):
        epsilon = 1e-6
        custom_func = partial(helper.custom_rmsnorm, epsilon=epsilon, sharding_type=sharding_type)
        reference_func = partial(helper.reference_rmsnorm, epsilon=epsilon)

        batch_size, _, num_heads, head_dim = helper.qkv_shape
        hidden_size = num_heads*head_dim
        input_shape = (batch_size, hidden_size)
        other_shape = (hidden_size, )

        devices = np.asarray(jax.devices()[:configs.device_count]).reshape(*mesh_shape)
        mesh = jax.sharding.Mesh(devices, mesh_names)
        with mesh, global_shard_guard(helper.get_sharding_resource(mesh_names, sharding_type)[0]):
            x_ = random.normal(random.PRNGKey(1124), input_shape, dtype=helper.dtype)
            gamma_ = jnp.ones(other_shape, dtype=helper.dtype)
            helper.compare_ops(custom_func, reference_func, collective_ref,
                               x_, gamma_, grad_args=(0, 1))

    @pytest.mark.parametrize('mesh_shape, mesh_names, sharding_type, collective_ref',
                             configs.softmax_refs)
    @pytest.mark.parametrize('softmax_type', configs.softmax_types)
    def test_softmax(self, mesh_shape, mesh_names, sharding_type, collective_ref,
                     softmax_type):
        batch_size, seq_len, num_heads, head_dim = helper.qkv_shape
        scale_factor = 1./jnp.sqrt(head_dim)

        custom_func = partial(helper.custom_softmax,
                              scale_factor=scale_factor,
                              softmax_type=softmax_type,
                              sharding_type=sharding_type)
        reference_func = partial(helper.reference_softmax,
                                 scale_factor=scale_factor,
                                 softmax_type=softmax_type)

        input_size = (batch_size, num_heads, seq_len, seq_len)
        pad_len = int(seq_len * helper.pad_ratio)
        valid_len = seq_len - pad_len

        devices = np.asarray(jax.devices()[:configs.device_count]).reshape(*mesh_shape)
        mesh = jax.sharding.Mesh(devices, mesh_names)
        with mesh, global_shard_guard(helper.get_sharding_resource(mesh_names, sharding_type)[0]):
            x_ = random.normal(random.PRNGKey(1124), input_size, dtype=helper.dtype)
            tokens = jnp.concatenate((jnp.ones((batch_size, valid_len), dtype=helper.dtype),
                                      jnp.zeros((batch_size, pad_len), dtype=helper.dtype)),
                                     axis=-1)
            mask_ = helper.make_mask(tokens, tokens, AttnMaskType.PADDING_MASK)
            helper.compare_ops(custom_func, reference_func, collective_ref,
                               x_, mask_, grad_args=(0))

    @pytest.mark.parametrize(
        'mesh_shape, mesh_names, sharding_type, attn_bias_type, collective_ref',
        configs.self_attn_refs)
    @pytest.mark.parametrize('attn_mask_type', configs.self_attn_mask_types)
    def test_self_fused_attn(self, mesh_shape, mesh_names, sharding_type, collective_ref,
                             attn_bias_type, attn_mask_type, backend):
        batch_size, seq_len, num_heads, head_dim = helper.qkv_shape
        helper.check_fused_attn_inputs(seq_len, seq_len, head_dim, 
                                       helper.pad_ratio, helper.dropout_prob,
                                       attn_bias_type, attn_mask_type, backend)
        
        dropout_rng = random.PRNGKey(91023051)
        split_rng = random.split(dropout_rng, configs.device_count)
        scale_factor = 1./jnp.sqrt(head_dim)

        custom_func = partial(helper.custom_self_fused_attn,
                              rng_key=split_rng,
                              dropout_prob=helper.dropout_prob,
                              attn_bias_type=attn_bias_type,
                              attn_mask_type=attn_mask_type,
                              scaling_factor=scale_factor,
                              sharding_type=sharding_type)
        reference_func = partial(helper.reference_self_fused_attn,
                                 rng_key=dropout_rng,
                                 dropout_prob=helper.dropout_prob,
                                 attn_bias_type=attn_bias_type,
                                 attn_mask_type=attn_mask_type,
                                 scaling_factor=scale_factor)

        key = random.PRNGKey(1124)
        subkeys = random.split(key, 2)

        qkv_shape = (batch_size, seq_len, 3, num_heads, head_dim)
        bias_shape = (1, num_heads, seq_len, seq_len)
        pad_len = int(seq_len * helper.pad_ratio)
        valid_len = seq_len - pad_len

        devices = np.asarray(jax.devices()[:configs.device_count]).reshape(*mesh_shape)
        mesh = jax.sharding.Mesh(devices, mesh_names)
        with mesh, global_shard_guard(helper.get_sharding_resource(mesh_names, sharding_type)[0]):
            qkv_ = random.normal(subkeys[0], qkv_shape, dtype=helper.dtype)
            bias_ = random.normal(subkeys[1], bias_shape, dtype=helper.dtype)
            tokens = jnp.concatenate((jnp.ones((batch_size, valid_len), dtype=helper.dtype),
                                      jnp.zeros((batch_size, pad_len), dtype=helper.dtype)),
                                     axis=-1)
            mask_ = helper.make_mask(tokens, tokens, attn_mask_type)
            helper.compare_ops(custom_func, reference_func, collective_ref,
                               qkv_, bias_, mask_, grad_args=(0, 1))

    @pytest.mark.parametrize('mesh_shape, mesh_names, sharding_type, collective_ref',
                             configs.cross_attn_refs)
    @pytest.mark.parametrize('attn_mask_type', configs.cross_attn_mask_types)
    def test_cross_fused_attn(self, mesh_shape, mesh_names, sharding_type, collective_ref,
                              attn_mask_type, backend):
        batch_size, seq_len, num_heads, head_dim = helper.qkv_shape
        helper.check_fused_attn_inputs(seq_len, seq_len, head_dim,
                                       helper.pad_ratio, helper.dropout_prob,
                                       AttnBiasType.NO_BIAS, attn_mask_type, backend)
        
        dropout_rng = random.PRNGKey(91023051)
        split_rng = random.split(dropout_rng, configs.device_count)
        scale_factor = 1./jnp.sqrt(head_dim)

        custom_func = partial(helper.custom_cross_fused_attn,
                              rng_key=split_rng,
                              dropout_prob=helper.dropout_prob,
                              attn_mask_type=attn_mask_type,
                              scaling_factor=scale_factor,
                              sharding_type=sharding_type)
        reference_func = partial(helper.reference_cross_fused_attn,
                                 rng_key=dropout_rng,
                                 dropout_prob=helper.dropout_prob,
                                 attn_mask_type=attn_mask_type,
                                 scaling_factor=scale_factor)

        key = random.PRNGKey(1124)
        subkeys = random.split(key, 2)

        q_shape = (batch_size, seq_len, num_heads, head_dim)
        kv_shape = (batch_size, seq_len, 2, num_heads, head_dim)
        pad_len = int(seq_len * helper.pad_ratio)
        valid_len = seq_len - pad_len

        devices = np.asarray(jax.devices()[:configs.device_count]).reshape(*mesh_shape)
        mesh = jax.sharding.Mesh(devices, mesh_names)
        with mesh, global_shard_guard(helper.get_sharding_resource(mesh_names, sharding_type)[0]):
            q_ = random.normal(subkeys[0], q_shape, dtype=helper.dtype)
            kv_ = random.normal(subkeys[1], kv_shape, dtype=helper.dtype)
            tokens = jnp.concatenate((jnp.ones((batch_size, valid_len), dtype=helper.dtype),
                                      jnp.zeros((batch_size, pad_len), dtype=helper.dtype)),
                                     axis=-1)
            mask_ = helper.make_mask(tokens, tokens, attn_mask_type)
            helper.compare_ops(custom_func, reference_func, collective_ref,
                               q_, kv_, mask_, grad_args=(0, 1))
