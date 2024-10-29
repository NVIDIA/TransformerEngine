# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax.linen import dot_product_attention
from jax import random
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from distributed_test_base import (
    generate_configs,
    generate_context_parallel_configs,
    generate_collectives_count,
    compare_ops,
)
from utils import (
    make_causal_mask,
    make_self_mask,
    assert_tree_like_allclose,
    assert_allclose,
    print_debug_tensor_stats,
)
from transformer_engine.jax import fp8_autocast
from transformer_engine.jax.attention import (
    is_fused_attn_kernel_available,
    fused_attn,
    AttnBiasType,
    AttnMaskType,
    QKVLayout,
    QKVFormat,
    get_qkv_format,
    reorder_causal_load_balancing,
    inverse_reorder_causal_load_balancing,
)

# We will use the golden reference model from our non distributed attention test fixture.
from test_fused_attn import general_dot_product_attention, make_mask

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
            None,  # no window
            False,  # not context parallel
        ):
            pytest.skip(f"No FusedAttn backend found")

        def target_func(qkv, bias, mask):
            return jnp.mean(
                fused_attn(
                    (qkv,),
                    bias,
                    mask,
                    None,
                    attn_bias_type=attn_bias_type,
                    attn_mask_type=attn_mask_type,
                    qkv_layout=QKVLayout.BS3HD,
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
            None,  # no window
            False,  # not context parallel
        ):
            pytest.skip(f"No FusedAttn backend found")

        def target_func(q, kv, mask):
            return jnp.mean(
                fused_attn(
                    (q, kv),
                    None,
                    mask,
                    None,
                    attn_bias_type=attn_bias_type,
                    attn_mask_type=attn_mask_type,
                    qkv_layout=QKVLayout.BSHD_BS2HD,
                    scaling_factor=scaling_factor,
                    dropout_probability=dropout_prob,
                    is_training=is_training,
                ),
                dtype=jnp.float32,
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

            return jnp.mean(output, dtype=jnp.float32)

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


class TestDistributedContextParallelSelfAttn:

    def generate_inputs(self, shape, kv_groups: int, attn_mask_type: AttnMaskType, dtype):
        batch, seqlen, heads, hidden = shape
        kv_shape = (batch, seqlen, heads // kv_groups, hidden)
        qkey, kkey, vkey = random.split(random.PRNGKey(1124), 3)
        q = random.normal(qkey, shape, dtype=dtype)
        k = random.normal(kkey, (batch, seqlen, heads // kv_groups, hidden), dtype=dtype)
        v = random.normal(vkey, (batch, seqlen, heads // kv_groups, hidden), dtype=dtype)

        def gen_valid(bs, max_seqlen, pad_ratio):
            pad_len = int(max_seqlen * pad_ratio)
            valid_len = max_seqlen - pad_len
            tokens = jnp.concatenate([jnp.ones((bs, valid_len)), jnp.zeros((bs, pad_len))], axis=-1)
            return tokens, jnp.logical_not(tokens)

        from test_fused_attn import make_mask

        q_idx, _ = gen_valid(batch, seqlen, 0.0)
        kv_idx, _ = gen_valid(batch, seqlen, 0.0)
        mask = make_mask(q_idx, kv_idx, None, None, attn_mask_type)

        return q, k, v, mask

    def qkv_to_layout(self, q, k, v, qkv_layout):
        qkv_args = ()
        match qkv_layout:
            case QKVLayout.BSHD_BS2HD:
                k, v = map(partial(jnp.expand_dims, axis=-3), [k, v])
                kv = jnp.concatenate((k, v), axis=-3)
                qkv_args = (q, kv)
            case QKVLayout.BSHD_BSHD_BSHD:
                qkv_args = (q, k, v)
            case _:
                raise ValueError(f"Unsupported {qkv_layout=}")
        return qkv_args

    @pytest.mark.parametrize(
        "device_count,mesh_shape,mesh_axes,mesh_resource", generate_context_parallel_configs()
    )
    @pytest.mark.parametrize(
        "data_shape",
        [
            pytest.param([2, 512, 12, 128], id="2-512-12-128"),
            pytest.param([4, 1024, 16, 64], id="4-1024-16-64"),
        ],
    )
    @pytest.mark.parametrize("kv_groups", [1, 4, 8, 12, 16])
    @pytest.mark.parametrize(
        "attn_mask_type",
        [
            pytest.param(AttnMaskType.CAUSAL_MASK, id="CAUSAL_MASK"),
            pytest.param(AttnMaskType.NO_MASK, id="NO_MASK"),
        ],
    )
    @pytest.mark.parametrize("dtype", [jnp.bfloat16])
    @pytest.mark.parametrize(
        "qkv_layout",
        [
            pytest.param(QKVLayout.BSHD_BS2HD, id="COMBINED_KV"),
            pytest.param(QKVLayout.BSHD_BSHD_BSHD, id="SEPARATE"),
        ],
    )
    @pytest.mark.parametrize(
        "load_balanced",
        [pytest.param(False, id="UNBALANCED"), pytest.param(True, id="BALANCED")],
    )
    def test_contex_parallel_self_attn(
        self,
        device_count,
        mesh_shape,
        mesh_axes,
        mesh_resource,
        data_shape,
        kv_groups,
        attn_mask_type,
        dtype,
        qkv_layout,
        load_balanced,
    ):
        attn_bias_type = AttnBiasType.NO_BIAS
        dropout_prob = 0.0
        is_training = True
        dp_size, cp_size, tp_size = mesh_shape
        qkv_format = get_qkv_format(qkv_layout)

        batch, seqlen, num_head, hidden = data_shape
        num_kv_heads = num_head // kv_groups
        scaling_factor = 1.0 / np.sqrt(num_head)

        if not is_fused_attn_kernel_available(
            dtype,
            dtype,
            qkv_layout,
            attn_bias_type,
            attn_mask_type,
            dropout_prob,
            num_head,
            num_kv_heads,
            seqlen,
            seqlen,
            hidden,
            None,  # no window
            cp_size > 1,
        ):
            pytest.skip(f"No FusedAttn backend found")

        if dp_size > 1 and batch % dp_size != 0:
            pytest.skip(f"Skipping {batch=} not a multiple of {dp_size=}")

        # make sure the mesh even divides cp and tp axis
        if num_head % kv_groups != 0 or (num_head // kv_groups) % tp_size != 0:
            pytest.skip(f"Skipping {kv_groups=} not multiple of {data_shape=} or {tp_size=}")

        def target_func(q, k, v, mask):
            return fused_attn(
                self.qkv_to_layout(q, k, v, qkv_layout),
                None,  # bias
                mask,
                None,  # seed
                attn_bias_type=attn_bias_type,
                attn_mask_type=attn_mask_type,
                qkv_layout=qkv_layout,
                scaling_factor=scaling_factor,
                dropout_probability=dropout_prob,
                is_training=is_training,
                context_parallel_causal_load_balanced=load_balanced,
                context_parallel_axis="cp",
            ).astype(dtype)

        def ref_func(q, k, v, mask):
            output = general_dot_product_attention(
                q,
                k,
                v,
                bias=None,
                mask=mask,
                deterministic=not is_training,
                scale_factor=scaling_factor,
                dropout_rate=dropout_prob,
                dropout_rng=None,
                dtype=jnp.float32,
            )
            return output.astype(dtype)

        def grad_func(func, *args, **kwargs):
            # Gradient is small, use a gradient multiplier to amplify the gradient
            _, max_seq_len, num_heads, _ = data_shape
            gradient_multiplier = max_seq_len * num_heads
            if attn_mask_type in [AttnMaskType.CAUSAL_MASK, AttnMaskType.CAUSAL_BOTTOM_RIGHT_MASK]:
                gradient_multiplier /= 10
            ret_valid = func(*args, **kwargs)
            return (jnp.mean(ret_valid, dtype=jnp.float32) * gradient_multiplier).astype(dtype)

        q, k, v, mask = self.generate_inputs(data_shape, kv_groups, attn_mask_type, dtype)

        diff_argnums = (0, 1, 2)

        # Single GPU (reference)
        ref_func_jit = jax.jit(
            jax.value_and_grad(
                lambda q, k, v, mask: grad_func(ref_func, q, k, v, mask), argnums=diff_argnums
            )
        )
        ref_fwd, ref_grads = ref_func_jit(q, k, v, mask)

        # Multi GPU (function under test)
        devices = np.asarray(jax.devices()[:device_count]).reshape(*mesh_shape)
        mesh = Mesh(devices, mesh_axes)
        with mesh, fp8_autocast(mesh_resource=mesh_resource, enabled=False):
            qkv_ps = PartitionSpec(
                mesh_resource.dp_resource,
                mesh_resource.cp_resource,
                mesh_resource.tp_resource,
                None,
            )
            qkv_sharding = NamedSharding(mesh, qkv_ps)

            mask_ps = PartitionSpec(
                mesh_resource.dp_resource, None, mesh_resource.cp_resource, None
            )
            mask_sharding = NamedSharding(mesh, mask_ps)

            reorder = partial(
                reorder_causal_load_balancing, cp_size=cp_size, tensor_format=qkv_format
            )
            inverse_reorder = partial(
                inverse_reorder_causal_load_balancing, cp_size=cp_size, tensor_format=qkv_format
            )

            if load_balanced:
                q, k, v = jax.tree.map(reorder, (q, k, v))

            q_, k_, v_ = map(partial(jax.device_put, device=qkv_sharding), [q, k, v])
            mask_ = jax.device_put(mask, device=mask_sharding)

            target_func_jit = jax.jit(
                jax.value_and_grad(
                    lambda q, k, v, mask: grad_func(target_func, q, k, v, mask),
                    argnums=diff_argnums,
                ),
                in_shardings=[qkv_sharding, qkv_sharding, qkv_sharding, mask_sharding],
                out_shardings=(None, (qkv_sharding, qkv_sharding, qkv_sharding)),
            )

            target_fwd, target_grads = target_func_jit(q_, k_, v_, mask_)

            if load_balanced:
                target_dq, target_dk, target_dv = jax.tree.map(inverse_reorder, target_grads[0:3])
                target_grads = (target_dq, target_dk, target_dv, *target_grads[3:])

            has_diffs = False

            print_debug_tensor_stats("target", target_fwd)
            print_debug_tensor_stats("ref", ref_fwd)
            print_debug_tensor_stats("diff", jnp.abs(target_fwd - ref_fwd))
            assert_allclose(target_fwd, ref_fwd, dtype=dtype)

            for i in range(len(target_grads)):
                if ref_grads[i] is None or target_grads[i] is None:
                    # expect both none if one is
                    assert target_grads[i] is None and ref_grads[i] is None
                else:
                    print_debug_tensor_stats(f"target_grad[{i}]", target_grads[i])
                    print_debug_tensor_stats(f"ref_grad[{i}]", ref_grads[i])
                    print_debug_tensor_stats(
                        f"diff_grad[{i}]", jnp.abs(target_grads[i] - ref_grads[i])
                    )

                assert_allclose(target_grads[i], ref_grads[i], dtype=dtype)


class TestReorderCausalLoadBalancing:
    @pytest.mark.parametrize("cp_size", [2, 4, 8])
    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param([1, 16, 1, 1], id="1-16-1-1"),
            pytest.param([4, 32, 12, 32], id="4-32-12-32"),
            pytest.param([3, 32, 8, 64], id="3-32-8-64"),
        ],
    )
    @pytest.mark.parametrize("qkv_format", [QKVFormat.BSHD, QKVFormat.SBHD])
    def test(self, cp_size, shape, qkv_format):
        tensor = random.normal(random.PRNGKey(1124), shape, dtype=jnp.bfloat16)
        if qkv_format == QKVFormat.SBHD:
            tensor = tensor.swapaxes(0, 1)

        ref = tensor.copy()

        reorder = jax.jit(reorder_causal_load_balancing, static_argnums=[1, 2])
        inverse = jax.jit(inverse_reorder_causal_load_balancing, static_argnums=[1, 2])

        reordered = reorder(tensor, cp_size, qkv_format)
        inversed = inverse(reordered, cp_size, qkv_format)

        assert jnp.array_equal(inversed, ref)
