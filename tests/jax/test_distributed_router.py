# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for distributed/sharded execution of fused MoE router primitives.

Testing Strategy:
=================
Router operations process each token independently (1 warp per token), so
sharded execution on the token dimension should produce identical results
to processing each shard independently with the reference implementation.

For fused_topk_with_score_function and fused_compute_score_for_moe_aux_loss:
- Input logits [num_tokens, num_experts] are sharded on num_tokens (DP axis)
- Expert dimension is replicated
- Each GPU processes its local tokens independently
- We verify sharded output matches per-shard reference, concatenated

For fused_moe_aux_loss:
- This is a global reduction to a scalar
- All inputs and outputs are replicated (partition function forces this)
- We verify the op works correctly under a mesh context

These tests exercise: partition, infer_sharding_from_operands, batcher,
and shardy_sharding_rule from the router primitives.
"""

import pytest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from distributed_test_base import generate_configs
from utils import assert_allclose, pytest_parametrize_wrapper

from transformer_engine.jax.router import (
    fused_topk_with_score_function,
    fused_compute_score_for_moe_aux_loss,
    fused_moe_aux_loss,
)

from test_fused_router import (
    reference_topk_softmax_sigmoid,
    reference_compute_scores_for_aux_loss,
    reference_aux_loss,
    make_logits,
)

# (num_tokens, num_experts, topk)
ALL_TOPK_CASES = [
    (128, 32, 4),
    (2048, 128, 8),
]
TOPK_CASES = {
    "L0": ALL_TOPK_CASES[0:1],
    "L2": ALL_TOPK_CASES,
}

ALL_AUX_LOSS_CASES = [
    (128, 32, 4),
    (2048, 128, 4),
]
AUX_LOSS_CASES = {
    "L0": ALL_AUX_LOSS_CASES[0:1],
    "L2": ALL_AUX_LOSS_CASES,
}


class TestDistributedFusedTopk:
    """Test distributed execution of fused_topk_with_score_function.

    Shards logits on the token dimension. Each GPU independently runs the
    fused kernel on its local tokens. We compare against the reference
    implementation run per-shard and concatenated.
    """

    def _impl_test(
        self,
        device_count,
        mesh_shape,
        mesh_axes,
        mesh_resource,
        num_tokens,
        num_experts,
        topk,
        score_function,
        use_shardy,
    ):
        jax.config.update("jax_use_shardy_partitioner", use_shardy)

        logits = make_logits(num_tokens, num_experts, score_function)

        devices = np.asarray(jax.devices()[:device_count]).reshape(*mesh_shape)
        mesh = Mesh(devices, mesh_axes)

        dp_axis = mesh_resource.dp_resource
        sharded_pspec = PartitionSpec(dp_axis, None)
        num_dp_devices = mesh.shape[dp_axis] if dp_axis else 1
        local_num_tokens = num_tokens // num_dp_devices

        with mesh:
            logits_sharding = NamedSharding(mesh, sharded_pspec)
            logits_sharded = jax.device_put(logits, logits_sharding)

            # === Forward ===
            @jax.jit
            def target_fwd(x):
                return fused_topk_with_score_function(
                    x, topk=topk, score_function=score_function,
                )

            target_probs, target_routing_map = target_fwd(logits_sharded)

            logits_shards = jnp.reshape(
                logits, (num_dp_devices, local_num_tokens, num_experts)
            )
            ref_fwd_fn = jax.jit(lambda x: reference_topk_softmax_sigmoid(
                x, topk=topk, score_function=score_function,
            ))
            ref_probs_list = []
            ref_routing_list = []
            for i in range(num_dp_devices):
                p, rm = ref_fwd_fn(logits_shards[i])
                ref_probs_list.append(p)
                ref_routing_list.append(rm)

            ref_probs = jnp.concatenate(ref_probs_list, axis=0)
            ref_routing = jnp.concatenate(ref_routing_list, axis=0)

            assert_allclose(
                jax.device_get(target_probs), ref_probs, dtype=jnp.float32,
            )
            assert jnp.array_equal(
                jax.device_get(target_routing_map), ref_routing,
            ), "Routing map mismatch in distributed fused_topk"

            # === Backward ===
            def target_loss(x):
                p, _ = fused_topk_with_score_function(
                    x, topk=topk, score_function=score_function,
                )
                return jnp.sum(p)

            def ref_chunk_loss(x_chunk):
                p, _ = reference_topk_softmax_sigmoid(
                    x_chunk, topk=topk, score_function=score_function,
                )
                return jnp.sum(p)

            target_grad = jax.jit(jax.grad(target_loss))(logits_sharded)

            ref_grads = []
            ref_chunk_grad_fn = jax.jit(jax.grad(ref_chunk_loss))
            for i in range(num_dp_devices):
                ref_grads.append(ref_chunk_grad_fn(logits_shards[i]))
            ref_grad = jnp.concatenate(ref_grads, axis=0)

            assert_allclose(
                jax.device_get(target_grad), ref_grad, dtype=jnp.float32,
            )

    @pytest.mark.parametrize("device_count,mesh_shape,mesh_axes,mesh_resource", generate_configs())
    @pytest_parametrize_wrapper(
        "num_tokens,num_experts,topk", TOPK_CASES,
    )
    @pytest.mark.parametrize("score_function", ["softmax", "sigmoid"])
    @pytest.mark.parametrize("use_shardy", [True])
    def test_distributed_topk(
        self,
        device_count,
        mesh_shape,
        mesh_axes,
        mesh_resource,
        num_tokens,
        num_experts,
        topk,
        score_function,
        use_shardy,
    ):
        self._impl_test(
            device_count,
            mesh_shape,
            mesh_axes,
            mesh_resource,
            num_tokens,
            num_experts,
            topk,
            score_function,
            use_shardy,
        )

    @pytest.mark.parametrize("device_count,mesh_shape,mesh_axes,mesh_resource", generate_configs())
    @pytest.mark.parametrize("score_function", ["softmax", "sigmoid"])
    def test_distributed_topk_gspmd(
        self,
        device_count,
        mesh_shape,
        mesh_axes,
        mesh_resource,
        score_function,
    ):
        """GSPMD test using value_and_grad with explicit shardings.

        GSPMD (non-shardy) requires explicit in/out shardings on jax.jit
        to correctly partition custom ops, matching the compare_ops pattern
        used by other TE distributed tests (softmax, permutation).
        """
        num_tokens, num_experts, topk = 128, 32, 4
        jax.config.update("jax_use_shardy_partitioner", False)

        logits = make_logits(num_tokens, num_experts, score_function)

        devices = np.asarray(jax.devices()[:device_count]).reshape(*mesh_shape)
        mesh = Mesh(devices, mesh_axes)
        dp_axis = mesh_resource.dp_resource
        sharded_pspec = PartitionSpec(dp_axis, None)

        with mesh:
            logits_sharding = NamedSharding(mesh, sharded_pspec)
            logits_sharded = jax.device_put(logits, logits_sharding)

            def target_loss(x):
                p, _ = fused_topk_with_score_function(
                    x, topk=topk, score_function=score_function,
                )
                return jnp.sum(p)

            def ref_loss(x):
                p, _ = reference_topk_softmax_sigmoid(
                    x, topk=topk, score_function=score_function,
                )
                return jnp.sum(p)

            target_vg = jax.jit(
                jax.value_and_grad(target_loss),
                in_shardings=(logits_sharding,),
                out_shardings=(None, logits_sharding),
            )
            ref_vg = jax.jit(
                jax.value_and_grad(ref_loss),
                in_shardings=(logits_sharding,),
                out_shardings=(None, logits_sharding),
            )
            target_fwd, target_grad = target_vg(logits_sharded)
            ref_fwd, ref_grad = ref_vg(logits_sharded)

            assert_allclose(target_fwd, ref_fwd, dtype=jnp.float32)
            assert_allclose(
                jax.device_get(target_grad),
                jax.device_get(ref_grad),
                dtype=jnp.float32,
            )


class TestDistributedScoreForAuxLoss:
    """Test distributed execution of fused_compute_score_for_moe_aux_loss.

    Same sharding strategy as fused_topk: shard on token dim, replicate experts.
    Each GPU independently computes scores and routing map for its local tokens.
    """

    def _impl_test(
        self,
        device_count,
        mesh_shape,
        mesh_axes,
        mesh_resource,
        num_tokens,
        num_experts,
        topk,
        score_function,
        use_shardy,
    ):
        jax.config.update("jax_use_shardy_partitioner", use_shardy)

        logits = make_logits(num_tokens, num_experts, score_function)

        devices = np.asarray(jax.devices()[:device_count]).reshape(*mesh_shape)
        mesh = Mesh(devices, mesh_axes)

        dp_axis = mesh_resource.dp_resource
        sharded_pspec = PartitionSpec(dp_axis, None)
        num_dp_devices = mesh.shape[dp_axis] if dp_axis else 1
        local_num_tokens = num_tokens // num_dp_devices

        with mesh:
            logits_sharding = NamedSharding(mesh, sharded_pspec)
            logits_sharded = jax.device_put(logits, logits_sharding)

            # === Forward ===
            @jax.jit
            def target_fwd(x):
                return fused_compute_score_for_moe_aux_loss(
                    x, topk=topk, score_function=score_function,
                )

            target_routing_map, target_scores = target_fwd(logits_sharded)

            logits_shards = jnp.reshape(
                logits, (num_dp_devices, local_num_tokens, num_experts)
            )
            ref_fwd_fn = jax.jit(lambda x: reference_compute_scores_for_aux_loss(
                x, topk=topk, score_function=score_function,
            ))
            ref_routing_list = []
            ref_scores_list = []
            for i in range(num_dp_devices):
                rm, s = ref_fwd_fn(logits_shards[i])
                ref_routing_list.append(rm)
                ref_scores_list.append(s)

            ref_routing = jnp.concatenate(ref_routing_list, axis=0)
            ref_scores = jnp.concatenate(ref_scores_list, axis=0)

            assert_allclose(
                jax.device_get(target_scores), ref_scores, dtype=jnp.float32,
            )
            assert jnp.array_equal(
                jax.device_get(target_routing_map), ref_routing,
            ), "Routing map mismatch in distributed score_for_aux_loss"

            # === Backward ===
            def target_loss(x):
                _, s = fused_compute_score_for_moe_aux_loss(
                    x, topk=topk, score_function=score_function,
                )
                return jnp.sum(s)

            def ref_chunk_loss(x_chunk):
                _, s = reference_compute_scores_for_aux_loss(
                    x_chunk, topk=topk, score_function=score_function,
                )
                return jnp.sum(s)

            target_grad = jax.jit(jax.grad(target_loss))(logits_sharded)

            ref_grads = []
            ref_chunk_grad_fn = jax.jit(jax.grad(ref_chunk_loss))
            for i in range(num_dp_devices):
                ref_grads.append(ref_chunk_grad_fn(logits_shards[i]))
            ref_grad = jnp.concatenate(ref_grads, axis=0)

            assert_allclose(
                jax.device_get(target_grad), ref_grad, dtype=jnp.float32,
            )

    @pytest.mark.parametrize("device_count,mesh_shape,mesh_axes,mesh_resource", generate_configs())
    @pytest_parametrize_wrapper(
        "num_tokens,num_experts,topk", TOPK_CASES,
    )
    @pytest.mark.parametrize("score_function", ["softmax", "sigmoid"])
    @pytest.mark.parametrize("use_shardy", [True])
    def test_distributed_score_for_aux_loss(
        self,
        device_count,
        mesh_shape,
        mesh_axes,
        mesh_resource,
        num_tokens,
        num_experts,
        topk,
        score_function,
        use_shardy,
    ):
        self._impl_test(
            device_count,
            mesh_shape,
            mesh_axes,
            mesh_resource,
            num_tokens,
            num_experts,
            topk,
            score_function,
            use_shardy,
        )

    @pytest.mark.parametrize("device_count,mesh_shape,mesh_axes,mesh_resource", generate_configs())
    @pytest.mark.parametrize("score_function", ["softmax", "sigmoid"])
    def test_distributed_score_for_aux_loss_gspmd(
        self,
        device_count,
        mesh_shape,
        mesh_axes,
        mesh_resource,
        score_function,
    ):
        """GSPMD test using value_and_grad with explicit shardings."""
        num_tokens, num_experts, topk = 128, 32, 4
        jax.config.update("jax_use_shardy_partitioner", False)

        logits = make_logits(num_tokens, num_experts, score_function)

        devices = np.asarray(jax.devices()[:device_count]).reshape(*mesh_shape)
        mesh = Mesh(devices, mesh_axes)
        dp_axis = mesh_resource.dp_resource
        sharded_pspec = PartitionSpec(dp_axis, None)

        with mesh:
            logits_sharding = NamedSharding(mesh, sharded_pspec)
            logits_sharded = jax.device_put(logits, logits_sharding)

            def target_loss(x):
                _, s = fused_compute_score_for_moe_aux_loss(
                    x, topk=topk, score_function=score_function,
                )
                return jnp.sum(s)

            def ref_loss(x):
                _, s = reference_compute_scores_for_aux_loss(
                    x, topk=topk, score_function=score_function,
                )
                return jnp.sum(s)

            target_vg = jax.jit(
                jax.value_and_grad(target_loss),
                in_shardings=(logits_sharding,),
                out_shardings=(None, logits_sharding),
            )
            ref_vg = jax.jit(
                jax.value_and_grad(ref_loss),
                in_shardings=(logits_sharding,),
                out_shardings=(None, logits_sharding),
            )
            target_fwd, target_grad = target_vg(logits_sharded)
            ref_fwd, ref_grad = ref_vg(logits_sharded)

            assert_allclose(target_fwd, ref_fwd, dtype=jnp.float32)
            assert_allclose(
                jax.device_get(target_grad),
                jax.device_get(ref_grad),
                dtype=jnp.float32,
            )


class TestDistributedMoEAuxLoss:
    """Test distributed execution of fused_moe_aux_loss.

    Aux loss is a global reduction to a scalar. The partition function forces
    all inputs to be replicated. We verify the op produces correct results
    under a mesh context with replicated sharding, testing both forward
    (scalar loss) and backward (gradient w.r.t. probs).
    """

    def _impl_test(
        self,
        device_count,
        mesh_shape,
        mesh_axes,
        mesh_resource,
        num_tokens,
        num_experts,
        topk,
        use_shardy,
    ):
        jax.config.update("jax_use_shardy_partitioner", use_shardy)

        key = jax.random.PRNGKey(42)
        key, subkey1, subkey2 = jax.random.split(key, 3)

        offset = jnp.arange(-num_tokens // 2, num_tokens // 2, dtype=jnp.float32) * 1e-4
        probs = jnp.arange(-num_experts // 2, num_experts // 2, dtype=jnp.float32) * 1e-2
        probs = probs[None, :].repeat(num_tokens, axis=0) + offset[:, None]

        tokens_per_expert = jax.random.randint(
            subkey1, (num_experts,), 1, 1000
        ).astype(jnp.int32)
        coeff = 0.01

        devices = np.asarray(jax.devices()[:device_count]).reshape(*mesh_shape)
        mesh = Mesh(devices, mesh_axes)

        replicated_2d_pspec = PartitionSpec(None, None)
        replicated_1d_pspec = PartitionSpec(None)

        with mesh:
            probs_sharding = NamedSharding(mesh, replicated_2d_pspec)
            tpe_sharding = NamedSharding(mesh, replicated_1d_pspec)

            probs_dev = jax.device_put(probs, probs_sharding)
            tpe_dev = jax.device_put(tokens_per_expert, tpe_sharding)

            # === Forward ===
            @jax.jit
            def target_fwd(p, tpe):
                return fused_moe_aux_loss(
                    p, tpe,
                    total_num_tokens=num_tokens,
                    num_experts=num_experts,
                    topk=topk,
                    coeff=coeff,
                )

            target_loss = target_fwd(probs_dev, tpe_dev)

            ref_fwd_fn = jax.jit(lambda p: reference_aux_loss(
                p, tokens_per_expert, num_tokens, topk, num_experts, coeff,
            ))
            ref_loss = ref_fwd_fn(probs)

            assert_allclose(
                jax.device_get(target_loss), ref_loss, dtype=jnp.float32,
            )

            # === Backward ===
            def target_loss_fn(p):
                return fused_moe_aux_loss(
                    p, tokens_per_expert,
                    total_num_tokens=num_tokens,
                    num_experts=num_experts,
                    topk=topk,
                    coeff=coeff,
                )

            def ref_loss_fn(p):
                return reference_aux_loss(
                    p, tokens_per_expert, num_tokens, topk, num_experts, coeff,
                )

            target_grad = jax.jit(jax.grad(target_loss_fn))(probs_dev)
            ref_grad = jax.jit(jax.grad(ref_loss_fn))(probs)

            assert_allclose(
                jax.device_get(target_grad), ref_grad, dtype=jnp.float32,
            )

    @pytest.mark.parametrize("device_count,mesh_shape,mesh_axes,mesh_resource", generate_configs())
    @pytest_parametrize_wrapper(
        "num_tokens,num_experts,topk", AUX_LOSS_CASES,
    )
    @pytest.mark.parametrize("use_shardy", [True, False])
    def test_distributed_aux_loss(
        self,
        device_count,
        mesh_shape,
        mesh_axes,
        mesh_resource,
        num_tokens,
        num_experts,
        topk,
        use_shardy,
    ):
        self._impl_test(
            device_count,
            mesh_shape,
            mesh_axes,
            mesh_resource,
            num_tokens,
            num_experts,
            topk,
            use_shardy,
        )
