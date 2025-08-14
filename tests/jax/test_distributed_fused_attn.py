# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import pytest
import jax
import jax.numpy as jnp
from jax import random
from distributed_test_base import (
    generate_configs,
    generate_context_parallel_configs_for_attn,
    generate_collectives_count,
)
from test_fused_attn import FusedAttnRunner, BiasShape, SeqDescFormat
from utils import pytest_parametrize_wrapper
from transformer_engine.jax.attention import (
    is_fused_attn_kernel_available,
    AttnBiasType,
    AttnMaskType,
    QKVLayout,
    QKVFormat,
    reorder_causal_load_balancing,
    inverse_reorder_causal_load_balancing,
    CPStrategy,
    ReorderStrategy,
)


DTYPES = [jnp.bfloat16]

DISTRIBUTED_SELF_ATTN_DATA_SHAPES = {
    "L0": [()],
    "L1": [(32, 1024, 16, 128)],
    "L2": [(32, 512, 12, 64)],
}


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

    def impl_test_self_attn(
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
        use_shardy,
    ):
        jax.config.update("jax_use_shardy_partitioner", use_shardy)
        dropout_prob = 0.0
        is_training = True
        batch, seqlen, num_head, hidden = data_shape

        if not is_fused_attn_kernel_available(
            is_training,
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
            hidden,
            None,  # no window
        ):
            pytest.skip("No FusedAttn backend found")

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

    @pytest.mark.parametrize("device_count,mesh_shape,mesh_axes,mesh_resource", generate_configs())
    @pytest_parametrize_wrapper("data_shape", DISTRIBUTED_SELF_ATTN_DATA_SHAPES)
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
        self.impl_test_self_attn(
            device_count,
            mesh_shape,
            mesh_axes,
            mesh_resource,
            data_shape,
            attn_bias_type,
            bias_shape,
            attn_mask_type,
            dtype,
            use_shardy=False,
        )

    @pytest.mark.parametrize("device_count,mesh_shape,mesh_axes,mesh_resource", generate_configs())
    @pytest.mark.parametrize(
        "attn_bias_type, bias_shape",
        [
            pytest.param(AttnBiasType.NO_BIAS, None, id="NO_BIAS"),
            pytest.param(AttnBiasType.PRE_SCALE_BIAS, BiasShape._1HSS, id="PRE_SCALE_BIAS-1HSS"),
        ],
    )
    def test_self_attn_shardy(
        self, device_count, mesh_shape, mesh_axes, mesh_resource, attn_bias_type, bias_shape
    ):
        data_shape = (32, 512, 12, 64)
        self.impl_test_self_attn(
            device_count,
            mesh_shape,
            mesh_axes,
            mesh_resource,
            data_shape,
            attn_bias_type,
            bias_shape,
            AttnMaskType.PADDING_MASK,
            jnp.bfloat16,
            use_shardy=True,
        )


DISTRIBUTED_CROSS_ATTN_DATA_SHAPES = {
    "L0": [()],
    "L1": [[32, 512, 16, 64]],
    "L2": [[32, 128, 12, 64]],
}


class TestDistributedCrossAttn:

    def generate_collectives_count_ref(self):
        # for loss
        all_reduce_loss_bytes = 4  # 1 * FP32
        return generate_collectives_count(allreduce=all_reduce_loss_bytes, allgather=0, other=0)

    @pytest.mark.parametrize("device_count,mesh_shape,mesh_axes,mesh_resource", generate_configs())
    @pytest_parametrize_wrapper("data_shape", DISTRIBUTED_CROSS_ATTN_DATA_SHAPES)
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
            is_training,
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
            hidden,
            None,  # no window
        ):
            pytest.skip("No FusedAttn backend found")

        col_ref = self.generate_collectives_count_ref()
        runner = FusedAttnRunner(
            batch,
            seqlen,
            seqlen,
            num_head,
            num_head,
            hidden,
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


DISTRIBUTED_CONTEXT_SELF_ATTN_LAYOUTS_MASKS = [
    pytest.param(QKVLayout.BSHD_BS2HD, AttnMaskType.CAUSAL_MASK, id="BSHD_KVPACKED-CAUSAL"),
    pytest.param(QKVLayout.BSHD_BSHD_BSHD, AttnMaskType.CAUSAL_MASK, id="BSHD_SEPARATE-CAUSAL"),
    pytest.param(QKVLayout.BSHD_BS2HD, AttnMaskType.NO_MASK, id="HD_KVPACKED-NO_MASK"),
    pytest.param(QKVLayout.BSHD_BSHD_BSHD, AttnMaskType.NO_MASK, id="BSHD_SEPARATE-NO_MASK"),
    pytest.param(
        QKVLayout.THD_THD_THD, AttnMaskType.PADDING_CAUSAL_MASK, id="THD_SEPARATE-PADDING_CAUSAL"
    ),
]

DISTRIBUTED_CONTEXT_SELF_ATTN_DATA_SHAPES = [
    # Sequence lengths will be scaled by CP so that we don't run with tiny sizes.
    pytest.param([2, 128, 8, 128], id="2-128xCP-8-128"),
    pytest.param([4, 256, 16, 64], id="4-256xCP-16-64"),
]


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
        use_shardy,
        use_scan_ring=False,
        window_size=None,
    ):
        if qkv_layout.is_thd():
            if cp_strategy == CPStrategy.ALL_GATHER:
                pytest.skip("THD doesn't support all gather context parallelism.")
            if not load_balanced and cp_strategy == CPStrategy.RING:
                pytest.skip("THD + ring doesn't support unbalanced context parallelism.")

        assert not use_scan_ring or cp_strategy == CPStrategy.RING

        if use_scan_ring:
            os.environ["NVTE_FUSED_RING_ATTENTION_USE_SCAN"] = "1"
        else:
            os.environ["NVTE_FUSED_RING_ATTENTION_USE_SCAN"] = "0"

        jax.config.update("jax_use_shardy_partitioner", use_shardy)
        attn_bias_type = AttnBiasType.NO_BIAS
        bias_shape = None
        dropout_prob = 0.0
        is_training = True
        dp_size, cp_size, tp_size = mesh_shape

        batch, seqlen, num_head, hidden = data_shape

        # Scale the sequence length by 2*CP so its never too small as we scale up test.
        # 2*CP is used since we split into two CP groups for load balancing.
        seqlen = seqlen * cp_size * 2
        data_shape = batch, seqlen, num_head, hidden

        num_kv_heads = num_head // kv_groups

        runner = FusedAttnRunner(
            batch,
            seqlen,
            seqlen,
            num_head,
            num_kv_heads,
            hidden,
            hidden,
            attn_bias_type,
            attn_mask_type,
            dropout_prob,
            dtype,
            is_training,
            qkv_layout,
            bias_shape,
            window_size,
            SeqDescFormat.SegmentIDs,
            number_of_devices=device_count,
            mesh_shape=mesh_shape,
            mesh_axes=mesh_axes,
            mesh_resource=mesh_resource,
            cp_strategy=cp_strategy,
            cp_load_balanced=load_balanced,
        )

        def check_has_backend_for_mask(mask_type):
            return is_fused_attn_kernel_available(
                is_training,
                dtype,
                dtype,
                qkv_layout,
                attn_bias_type,
                mask_type,
                dropout_prob,
                num_head,
                num_kv_heads,
                seqlen,
                seqlen,
                hidden,
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
        del os.environ["NVTE_FUSED_RING_ATTENTION_USE_SCAN"]

    @pytest_parametrize_wrapper(
        "device_count,mesh_shape,mesh_axes,mesh_resource",
        generate_context_parallel_configs_for_attn(),
    )
    @pytest.mark.parametrize("data_shape", DISTRIBUTED_CONTEXT_SELF_ATTN_DATA_SHAPES[:1])
    @pytest.mark.parametrize("dtype", [pytest.param(jnp.bfloat16, id="BF16")])
    @pytest.mark.parametrize(
        "qkv_layout, attn_mask_type",
        DISTRIBUTED_CONTEXT_SELF_ATTN_LAYOUTS_MASKS,
    )
    def test_context_parallel_allgather_attn_shardy(
        self,
        device_count,
        mesh_shape,
        mesh_axes,
        mesh_resource,
        data_shape,
        attn_mask_type,
        dtype,
        qkv_layout,
    ):
        kv_groups = 8
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
            load_balanced=True,
            cp_strategy=CPStrategy.ALL_GATHER,
            use_shardy=True,
        )

    @pytest_parametrize_wrapper(
        "device_count,mesh_shape,mesh_axes,mesh_resource",
        generate_context_parallel_configs_for_attn(),
    )
    @pytest.mark.parametrize("data_shape", DISTRIBUTED_CONTEXT_SELF_ATTN_DATA_SHAPES)
    @pytest.mark.parametrize("kv_groups", [1, 8])
    @pytest.mark.parametrize("dtype", [pytest.param(jnp.bfloat16, id="BF16")])
    @pytest.mark.parametrize(
        "qkv_layout, attn_mask_type",
        DISTRIBUTED_CONTEXT_SELF_ATTN_LAYOUTS_MASKS,
    )
    @pytest.mark.parametrize(
        "load_balanced",
        [pytest.param(True, id="BALANCED"), pytest.param(False, id="UNBALANCED")],
    )
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
            CPStrategy.ALL_GATHER,
            use_shardy=False,
        )

    @pytest_parametrize_wrapper(
        "device_count,mesh_shape,mesh_axes,mesh_resource",
        generate_context_parallel_configs_for_attn(),
    )
    @pytest.mark.parametrize("data_shape", DISTRIBUTED_CONTEXT_SELF_ATTN_DATA_SHAPES)
    @pytest.mark.parametrize("kv_groups", [1, 8])
    @pytest.mark.parametrize("dtype", [pytest.param(jnp.bfloat16, id="BF16")])
    @pytest.mark.parametrize(
        "qkv_layout, attn_mask_type",
        DISTRIBUTED_CONTEXT_SELF_ATTN_LAYOUTS_MASKS,
    )
    @pytest.mark.parametrize(
        "load_balanced",
        [pytest.param(True, id="BALANCED"), pytest.param(False, id="UNBALANCED")],
    )
    @pytest.mark.parametrize(
        "use_scan",
        [pytest.param(False, id="NO_SCAN"), pytest.param(True, id="USE_SCAN")],
    )
    @pytest.mark.parametrize(
        "window_size",
        [
            pytest.param((-1, -1), id="window_size(-1, -1)"),
            pytest.param((20, 0), id="window_size(20, 0)"),
        ],
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
        window_size,
    ):
        if window_size != (-1, -1) and not qkv_layout.is_thd():
            pytest.skip("Sliding window attention is only supported for THD layout")
        if window_size != (-1, -1) and qkv_layout.is_thd() and use_scan:
            pytest.skip(
                "When context parallelism and sliding window attention are used, "
                "scanloop is not supported"
            )
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
            use_shardy=False,
            use_scan_ring=use_scan,
            window_size=window_size,
        )

    @pytest_parametrize_wrapper(
        "device_count,mesh_shape,mesh_axes,mesh_resource",
        generate_context_parallel_configs_for_attn(),
    )
    @pytest.mark.parametrize("data_shape", DISTRIBUTED_CONTEXT_SELF_ATTN_DATA_SHAPES[:1])
    @pytest.mark.parametrize("dtype", [pytest.param(jnp.bfloat16, id="BF16")])
    @pytest.mark.parametrize(
        "qkv_layout, attn_mask_type",
        DISTRIBUTED_CONTEXT_SELF_ATTN_LAYOUTS_MASKS,
    )
    def test_context_parallel_ring_attn_shardy(
        self,
        device_count,
        mesh_shape,
        mesh_axes,
        mesh_resource,
        data_shape,
        attn_mask_type,
        dtype,
        qkv_layout,
    ):
        kv_groups = 8
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
            load_balanced=True,
            cp_strategy=CPStrategy.RING,
            use_shardy=False,
            use_scan_ring=True,
        )


REORDER_CAUSAL_LOAD_BALANCING_DATA_SHAPES = {
    "L0": [[]],
    "L1": [[3, 32, 8, 64]],
    "L2": [[4, 32, 12, 32], [1, 16, 1, 1]],
}


class TestReorderCausalLoadBalancing:
    @pytest.mark.parametrize("cp_size", [2, 4, 8])
    @pytest_parametrize_wrapper("shape", REORDER_CAUSAL_LOAD_BALANCING_DATA_SHAPES)
    @pytest.mark.parametrize("qkv_format", [QKVFormat.BSHD, QKVFormat.SBHD])
    @pytest.mark.parametrize(
        "reorder_strategy",
        [
            pytest.param(ReorderStrategy.DualChunkSwap, id="DualChunkSwap"),
            pytest.param(ReorderStrategy.Striped, id="Striped"),
        ],
    )
    def test(self, cp_size, shape, qkv_format, reorder_strategy):
        tensor = random.normal(random.PRNGKey(1124), shape, dtype=jnp.bfloat16)
        seq_dim = 1
        if qkv_format == QKVFormat.SBHD:
            tensor = tensor.swapaxes(0, 1)
            seq_dim = 0

        ref = tensor.copy()

        reorder = jax.jit(reorder_causal_load_balancing, static_argnums=[1, 2, 3])
        inverse = jax.jit(inverse_reorder_causal_load_balancing, static_argnums=[1, 2, 3])

        reordered = reorder(tensor, reorder_strategy, cp_size, seq_dim)
        inversed = inverse(reordered, reorder_strategy, cp_size, seq_dim)

        assert jnp.array_equal(inversed, ref)
