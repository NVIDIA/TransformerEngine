# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from distributed_test_base import (
    generate_configs,
    generate_context_parallel_configs,
    generate_collectives_count,
)
from test_fused_attn import FusedAttnRunner, BiasShape, SeqDescFormat
from transformer_engine.jax.attention import (
    is_fused_attn_kernel_available,
    AttnBiasType,
    AttnMaskType,
    QKVLayout,
    QKVFormat,
    reorder_causal_load_balancing,
    inverse_reorder_causal_load_balancing,
    CPStrategy,
)


DTYPES = [jnp.bfloat16]


class TestDistributedSelfAttn:

    def generate_collectives_count_ref(
        self, mesh_shape, mesh_axes, mesh_resource, with_bias, shape, dtype
    ):
        jax_dtype = jax.dtypes.canonicalize_dtype(dtype)
        _, seqlen, heads, _ = shape
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

    @pytest.mark.parametrize("device_count,mesh_shape,mesh_axes,mesh_resource", generate_configs())
    @pytest.mark.parametrize(
        "data_shape",
        [
            pytest.param((32, 512, 12, 64), id="32-512-12-64"),
            pytest.param((32, 1024, 16, 128), id="32-1024-16-128"),
        ],
    )
    @pytest.mark.parametrize(
        "attn_bias_type, bias_shape",
        [
            pytest.param(AttnBiasType.NO_BIAS, None, id="NO_BIAS"),
            pytest.param(AttnBiasType.PRE_SCALE_BIAS, BiasShape._1HSS, id="PRE_SCALE_BIAS-1HSS"),
            pytest.param(AttnBiasType.POST_SCALE_BIAS, BiasShape._1HSS, id="POST_SCALE_BIAS-1HSS"),
        ],
    )
    @pytest.mark.parametrize(
        "attn_mask_type",
        [
            pytest.param(AttnMaskType.PADDING_MASK, id="PADDING_MASK"),
            pytest.param(AttnMaskType.CAUSAL_MASK, id="CAUSAL_MASK"),
        ],
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
        bias_shape,
        attn_mask_type,
        dtype,
    ):
        dropout_prob = 0.0
        is_training = True

        batch, seqlen, num_head, hidden = data_shape

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
        ):
            pytest.skip(f"No FusedAttn backend found")

        col_ref = self.generate_collectives_count_ref(
            mesh_shape,
            mesh_axes,
            mesh_resource,
            attn_bias_type != AttnBiasType.NO_BIAS,
            data_shape,
            dtype,
        )
        runner = FusedAttnRunner(
            batch,
            seqlen,
            seqlen,
            num_head,
            num_head,
            hidden,
            attn_bias_type,
            attn_mask_type,
            dropout_prob,
            dtype,
            is_training,
            QKVLayout.BS3HD,
            bias_shape,
            None,
            SeqDescFormat.Seqlens,
            number_of_devices=device_count,
            mesh_shape=mesh_shape,
            mesh_axes=mesh_axes,
            mesh_resource=mesh_resource,
            coll_count_ref=col_ref,
        )
        runner.test_backward()


class TestDistributedCrossAttn:

    def generate_collectives_count_ref(self):
        # for loss
        all_reduce_loss_bytes = 4  # 1 * FP32
        return generate_collectives_count(allreduce=all_reduce_loss_bytes, allgather=0, other=0)

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
        bias_shape = None
        dropout_prob = 0.0
        is_training = True

        batch, seqlen, num_head, hidden = data_shape

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
        ):
            pytest.skip(f"No FusedAttn backend found")

        col_ref = self.generate_collectives_count_ref()
        runner = FusedAttnRunner(
            batch,
            seqlen,
            seqlen,
            num_head,
            num_head,
            hidden,
            attn_bias_type,
            attn_mask_type,
            dropout_prob,
            dtype,
            is_training,
            QKVLayout.BSHD_BS2HD,
            bias_shape,
            None,
            SeqDescFormat.Seqlens,
            number_of_devices=device_count,
            mesh_shape=mesh_shape,
            mesh_axes=mesh_axes,
            mesh_resource=mesh_resource,
            coll_count_ref=col_ref,
        )
        runner.test_backward()


@pytest.mark.parametrize(
    "device_count,mesh_shape,mesh_axes,mesh_resource", generate_context_parallel_configs()
)
@pytest.mark.parametrize(
    "data_shape",
    [
        # Sequence lengths will be scaled by CP so that we don't run with tiny sizes.
        pytest.param([2, 128, 12, 128], id="2-128xCP-12-128"),
        pytest.param([4, 256, 16, 64], id="4-256xCP-16-64"),
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
class TestDistributedContextParallelSelfAttn:

    def impl_test_context_parallel_attn(
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
        cp_strategy,
    ):
        attn_bias_type = AttnBiasType.NO_BIAS
        bias_shape = None
        dropout_prob = 0.0
        is_training = True
        dp_size, cp_size, tp_size = mesh_shape
        qkv_format = qkv_layout.get_qkv_format()

        batch, seqlen, num_head, hidden = data_shape

        # Scale the sequence length by 2*CP so its never too small as we scale up test.
        # 2*CP is used since we split into two CP groups for load balancing.
        seqlen = seqlen * cp_size * 2
        data_shape = batch, seqlen, num_head, hidden

        num_kv_heads = num_head // kv_groups
        scaling_factor = 1.0 / np.sqrt(num_head)

        runner = FusedAttnRunner(
            batch,
            seqlen,
            seqlen,
            num_head,
            num_kv_heads,
            hidden,
            attn_bias_type,
            attn_mask_type,
            dropout_prob,
            dtype,
            is_training,
            qkv_layout,
            bias_shape,
            None,
            SeqDescFormat.Seqlens,
            number_of_devices=device_count,
            mesh_shape=mesh_shape,
            mesh_axes=mesh_axes,
            mesh_resource=mesh_resource,
            cp_strategy=cp_strategy,
            cp_load_balanced=load_balanced,
        )

        def check_has_backend_for_mask(mask_type):
            return is_fused_attn_kernel_available(
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
                None,
            )  # no SWA for CP

        # For causal masking we depend on having bottom right support also.
        # The API does not check this and instead we rely on lower level checks to raise
        # and exception if the step backend is not supported. This was a deliberate API
        # decision to keep the CP size or flag out of the function.
        has_backend = check_has_backend_for_mask(attn_mask_type)
        if cp_size > 1 and attn_mask_type == AttnMaskType.CAUSAL_MASK:
            has_backend &= check_has_backend_for_mask(AttnMaskType.CAUSAL_BOTTOM_RIGHT_MASK)

        if not has_backend:
            pytest.skip(f"No FusedAttn backend found {cp_size=} {attn_mask_type=}.")

        if dp_size > 1 and batch % dp_size != 0:
            pytest.skip(f"Skipping {batch=} not a multiple of {dp_size=}")

        # make sure the mesh even divides cp and tp axis
        if num_head % kv_groups != 0 or (num_head // kv_groups) % tp_size != 0:
            pytest.skip(f"Skipping {kv_groups=} not multiple of {data_shape=} or {tp_size=}")

        runner.test_backward()

    def test_context_parallel_allgather_attn(
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
        return self.impl_test_context_parallel_attn(
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
            CPStrategy.ALL_GATHER,
        )

    @pytest.mark.parametrize(
        "use_scan",
        [pytest.param(False, id="NO_SCAN"), pytest.param(True, id="USE_SCAN")],
    )
    def test_context_parallel_ring_attn(
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
        use_scan,
    ):
        if use_scan:
            os.environ["NVTE_FUSED_RING_ATTENTION"] = "1"

        self.impl_test_context_parallel_attn(
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
            CPStrategy.RING,
        )
        if use_scan:
            del os.environ["NVTE_FUSED_RING_ATTENTION"]


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
