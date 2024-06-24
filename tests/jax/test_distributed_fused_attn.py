# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest

import jax
import jax.numpy as jnp
import numpy as np
from flax.linen import dot_product_attention
from jax import random
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from distributed_test_base import generate_configs, generate_collectives_count, compare_ops
from utils import make_causal_mask, make_self_mask
from transformer_engine.jax import fp8_autocast
from transformer_engine.jax.attention import (
    is_fused_attn_kernel_available,
    fused_attn_qkvpacked,
    fused_attn_kvpacked,
    AttnBiasType,
    AttnMaskType,
    QKVLayout,
)


DTYPES = [jnp.float16, jnp.bfloat16]


class TestDistributedSelfAttn:

    def generate_collectives_count_ref(
        self, mesh_shape, mesh_axes, mesh_resource, with_bias, shape, dtype
    ):
        jax_dtype = jax.dtypes.canonicalize_dtype(dtype)
        _, seqlen, _, heads, _ = shape
        is_dp_enabled = mesh_resource.dp_resource is not None
        tp_size = 1
        if mesh_resource.tp_resource is not None:
            idx = mesh_axes.index(mesh_resource.tp_resource)
            tp_size = mesh_shape[idx]

        all_reduce_loss_bytes = 4  # 1 * FP32
        bias_bytes = int(with_bias) * (heads // tp_size) * seqlen * seqlen * jax_dtype.itemsize
        allreduce_total_bytes = all_reduce_loss_bytes + (bias_bytes * is_dp_enabled)
        # for loss and dbias
        return generate_collectives_count(allreduce=allreduce_total_bytes, allgather=0, other=0)

    def generate_inputs(self, shape, mesh_resource, with_bias, attn_mask_type, dtype):
        batch, seqlen, _, heads, _ = shape

        qkv = random.normal(random.PRNGKey(1124), shape, dtype=dtype)

        bias = (
            random.normal(random.PRNGKey(1125), (1, heads, seqlen, seqlen), dtype)
            if with_bias
            else None
        )

        mask = None
        if attn_mask_type == AttnMaskType.PADDING_MASK:
            mask = make_causal_mask(batch, seqlen)
        elif attn_mask_type == AttnMaskType.CAUSAL_MASK:
            mask = make_self_mask(batch, seqlen)

        qkv_pspec = PartitionSpec(
            mesh_resource.dp_resource, None, None, mesh_resource.tp_resource, None
        )
        bias_pspec = (
            PartitionSpec(None, mesh_resource.tp_resource, None, None) if with_bias else None
        )
        mask_pspec = (
            PartitionSpec(mesh_resource.dp_resource, None, None, None)
            if attn_mask_type != AttnMaskType.NO_MASK
            else None
        )

        return (qkv, bias, mask), (qkv_pspec, bias_pspec, mask_pspec)

    @pytest.mark.parametrize("device_count,mesh_shape,mesh_axes,mesh_resource", generate_configs())
    @pytest.mark.parametrize("data_shape", [[32, 512, 3, 12, 64], [32, 1024, 3, 16, 128]])
    @pytest.mark.parametrize(
        "attn_bias_type",
        [AttnBiasType.NO_BIAS, AttnBiasType.PRE_SCALE_BIAS, AttnBiasType.POST_SCALE_BIAS],
    )
    @pytest.mark.parametrize(
        "attn_mask_type", [AttnMaskType.PADDING_MASK, AttnMaskType.CAUSAL_MASK]
    )
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_self_attn(
        self,
        device_count,
        mesh_shape,
        mesh_axes,
        mesh_resource,
        data_shape,
        attn_bias_type,
        attn_mask_type,
        dtype,
    ):
        dropout_prob = 0.0
        is_training = True
        scaling_factor = 1.0

        _, seqlen, _, num_head, hidden = data_shape

        if not is_fused_attn_kernel_available(
            dtype,
            dtype,
            QKVLayout.BS3HD,
            attn_bias_type,
            attn_mask_type,
            dropout_prob,
            num_head,
            num_head,
            seqlen,
            seqlen,
            hidden,
        ):
            pytest.skip(f"No FusedAttn backwend found")

        def target_func(qkv, bias, mask):
            return jnp.mean(
                fused_attn_qkvpacked(
                    qkv,
                    bias,
                    mask,
                    None,
                    attn_bias_type=attn_bias_type,
                    attn_mask_type=attn_mask_type,
                    scaling_factor=scaling_factor,
                    dropout_probability=dropout_prob,
                    is_training=is_training,
                )
            )

        def ref_func(qkv, bias, mask):
            query, key, value = jnp.split(qkv, [1, 2], axis=-3)
            query = jnp.squeeze(query)
            key = jnp.squeeze(key)
            value = jnp.squeeze(value)

            output = dot_product_attention(
                query,
                key,
                value,
                bias=bias,
                mask=mask,
                deterministic=is_training,
                dropout_rate=dropout_prob,
                dropout_rng=None,
                dtype=jnp.float32,
            )

            return jnp.mean(output).astype(dtype)

        with_bias = attn_bias_type != AttnBiasType.NO_BIAS
        (qkv, bias, mask), (qkv_pspec, bias_pspec, mask_pspec) = self.generate_inputs(
            data_shape, mesh_resource, with_bias, attn_mask_type, dtype
        )
        collective_count_ref = self.generate_collectives_count_ref(
            mesh_shape, mesh_axes, mesh_resource, with_bias, data_shape, dtype
        )
        devices = np.asarray(jax.devices()[:device_count]).reshape(*mesh_shape)
        mesh = Mesh(devices, mesh_axes)
        with mesh, fp8_autocast(mesh_resource=mesh_resource):
            qkv_ = jax.device_put(qkv, NamedSharding(mesh, qkv_pspec))
            bias_ = (
                jax.device_put(bias, NamedSharding(mesh, bias_pspec)) if bias is not None else bias
            )
            mask_ = (
                jax.device_put(mask, NamedSharding(mesh, mask_pspec)) if mask is not None else mask
            )

            grad_args = (0, 1) if with_bias else (0,)
            out_grad_shardings = (qkv_pspec, bias_pspec) if with_bias else (qkv_pspec,)

            compare_ops(
                target_func,
                ref_func,
                [qkv_, bias_, mask_],
                collective_count_ref,
                grad_args=grad_args,
                metric_fwd_dtype=dtype,
                metric_bwd_dtype=dtype,
                in_shardings=(qkv_pspec, bias_pspec, mask_pspec),
                out_shardings=(None, out_grad_shardings),
            )


class TestDistributedCrossAttn:

    def generate_collectives_count_ref(self):
        # for loss
        all_reduce_loss_bytes = 4  # 1 * FP32
        return generate_collectives_count(allreduce=all_reduce_loss_bytes, allgather=0, other=0)

    def generate_inputs(self, shape, mesh_resource, attn_mask_type, dtype):
        batch, seqlen, heads, hidden = shape

        q = random.normal(random.PRNGKey(1124), shape, dtype=dtype)
        kv = random.normal(random.PRNGKey(1125), (batch, seqlen, 2, heads, hidden), dtype=dtype)

        mask = None
        if attn_mask_type == AttnMaskType.PADDING_MASK:
            mask = make_causal_mask(batch, seqlen)
        elif attn_mask_type == AttnMaskType.CAUSAL_MASK:
            mask = make_self_mask(batch, seqlen)

        q_pspec = PartitionSpec(mesh_resource.dp_resource, None, mesh_resource.tp_resource, None)

        kv_pspec = PartitionSpec(
            mesh_resource.dp_resource, None, None, mesh_resource.tp_resource, None
        )
        mask_pspec = (
            PartitionSpec(mesh_resource.dp_resource, None, None, None)
            if attn_mask_type != AttnMaskType.NO_MASK
            else None
        )

        return (q, kv, mask), (q_pspec, kv_pspec, mask_pspec)

    @pytest.mark.parametrize("device_count,mesh_shape,mesh_axes,mesh_resource", generate_configs())
    @pytest.mark.parametrize("data_shape", [[32, 128, 12, 64], [32, 512, 16, 64]])
    @pytest.mark.parametrize(
        "attn_mask_type", [AttnMaskType.PADDING_MASK, AttnMaskType.CAUSAL_MASK]
    )
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_cross_attn(
        self, device_count, mesh_shape, mesh_axes, mesh_resource, data_shape, attn_mask_type, dtype
    ):
        attn_bias_type = AttnBiasType.NO_BIAS
        dropout_prob = 0.0
        is_training = True
        scaling_factor = 1.0

        _, seqlen, num_head, hidden = data_shape

        if not is_fused_attn_kernel_available(
            dtype,
            dtype,
            QKVLayout.BSHD_BS2HD,
            attn_bias_type,
            attn_mask_type,
            dropout_prob,
            num_head,
            num_head,
            seqlen,
            seqlen,
            hidden,
        ):
            pytest.skip(f"No FusedAttn backwend found")

        def target_func(q, kv, mask):
            return jnp.mean(
                fused_attn_kvpacked(
                    q,
                    kv,
                    None,
                    mask,
                    None,
                    attn_bias_type=attn_bias_type,
                    attn_mask_type=attn_mask_type,
                    scaling_factor=scaling_factor,
                    dropout_probability=dropout_prob,
                    is_training=is_training,
                )
            )

        def ref_func(query, kv, mask):
            key, value = jnp.split(kv, [1], axis=-3)
            query = jnp.squeeze(query)
            key = jnp.squeeze(key)
            value = jnp.squeeze(value)

            output = dot_product_attention(
                query,
                key,
                value,
                bias=None,
                mask=mask,
                deterministic=is_training,
                dropout_rate=dropout_prob,
                dropout_rng=None,
                dtype=jnp.float32,
            )

            return jnp.mean(output).astype(dtype)

        (q, kv, mask), (q_pspec, kv_pspec, mask_pspec) = self.generate_inputs(
            data_shape, mesh_resource, attn_mask_type, dtype
        )
        collective_count_ref = self.generate_collectives_count_ref()
        devices = np.asarray(jax.devices()[:device_count]).reshape(*mesh_shape)
        mesh = Mesh(devices, mesh_axes)
        with mesh, fp8_autocast(mesh_resource=mesh_resource):
            q_ = jax.device_put(q, NamedSharding(mesh, q_pspec))
            kv_ = jax.device_put(kv, NamedSharding(mesh, kv_pspec))
            mask_ = (
                jax.device_put(mask, NamedSharding(mesh, mask_pspec)) if mask is not None else mask
            )

            compare_ops(
                target_func,
                ref_func,
                [q_, kv_, mask_],
                collective_count_ref,
                grad_args=(0, 1),
                metric_fwd_dtype=dtype,
                metric_bwd_dtype=dtype,
                in_shardings=(q_pspec, kv_pspec, mask_pspec),
                out_shardings=(None, (q_pspec, kv_pspec)),
            )
