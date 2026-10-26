# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Single-GPU coverage for the EP=1 local-permutation path."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from transformer_engine.jax.cpp_extensions import ep as ep_extensions
from transformer_engine.jax.cpp_extensions.misc import jax_dtype_to_te_dtype
from transformer_engine.jax.ep import (
    EpLayerConfig,
    ep_bootstrap,
    ep_combine,
    ep_dispatch,
    ep_finalize,
)
from transformer_engine.jax.sharding import MeshResource, global_shard_guard

pytestmark = pytest.mark.skipif(jax.device_count() != 1, reason="run with exactly one visible GPU")


@pytest.mark.parametrize("ep_size", [None, 1, 2])
def test_ep_finalize_is_idempotent(monkeypatch, ep_size):
    """Finalize tolerates absent state and releases only initialized NCCL EP."""
    config = None
    if ep_size is not None:
        config = ep_extensions.EpConfig(
            world_size=ep_size,
            rank=0,
            ep_size=ep_size,
            num_ep_groups=1,
            num_experts=4,
            num_local_experts=4 // ep_size,
            max_tokens_per_rank=8,
            recv_capacity_per_rank=16,
            hidden_dim=8,
        )
    monkeypatch.setattr(ep_extensions, "_ep_config", config)
    calls = []
    monkeypatch.setattr(jax, "clear_caches", lambda: calls.append("clear_caches"))
    monkeypatch.setattr(
        "transformer_engine_jax.release_ep_resources",
        lambda: calls.append("release_ep_resources"),
    )

    ep_finalize()
    ep_finalize()

    expected = [] if ep_size is None else ["clear_caches"]
    if ep_size is not None and ep_size > 1:
        expected.append("release_ep_resources")
    assert calls == expected
    # Other EP APIs must continue to reject use before bootstrap.
    with pytest.raises(RuntimeError, match="EpConfig has not been set"):
        ep_extensions.get_ep_config()


def test_ep1_bootstrap_dispatch_combine_and_gradients():
    """Bootstrap is a no-op and dispatch/combine remain differentiable."""
    num_experts = 4
    top_k = 2
    batch = 2
    sequence = 3
    hidden = 8
    num_tokens = batch * sequence
    recv_capacity = num_tokens * top_k
    mesh = Mesh(np.asarray(jax.devices()).reshape(1), ("ep",))
    resource = MeshResource(ep_resource="ep")

    cfg = EpLayerConfig(
        top_k=top_k,
        dispatch_output_per_expert_alignment=0,
        topk_idx_dtype=int(jax_dtype_to_te_dtype(jnp.int32)),
    )
    tokens = jnp.arange(batch * sequence * hidden, dtype=jnp.float32).reshape(
        batch, sequence, hidden
    )
    topk_idx = jnp.asarray(
        [
            [[0, 1], [2, 3], [1, 2]],
            [[3, 0], [2, 1], [0, 3]],
        ],
        dtype=jnp.int32,
    )
    topk_weights = jnp.asarray(
        [
            [[0.25, 0.75], [0.6, 0.4], [0.1, 0.9]],
            [[0.8, 0.2], [0.3, 0.7], [0.55, 0.45]],
        ],
        dtype=jnp.float32,
    )

    with mesh, global_shard_guard(resource):
        ep_finalize()
        ep_finalize()
        # world_size=1 used to fail before any EP configuration was installed.
        ep_bootstrap(
            world_size=1,
            rank=0,
            num_experts=num_experts,
            max_tokens_per_rank=num_tokens,
            recv_capacity_per_rank=recv_capacity,
            hidden_dim=hidden,
        )
        try:

            def run(x, weights):
                recv, recv_weights, handle, counts, total = ep_dispatch(
                    cfg, topk_idx, x, weights, recv_capacity
                )
                output = ep_combine(
                    cfg,
                    handle,
                    counts,
                    recv * recv_weights[..., None],
                    num_local_tokens=(batch, sequence),
                )
                return output, counts, total

            output, counts, total = jax.jit(run)(tokens, topk_weights)
            expected = tokens * jnp.sum(topk_weights, axis=-1, keepdims=True)
            np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-6)
            np.testing.assert_array_equal(jnp.sum(counts), recv_capacity)
            np.testing.assert_array_equal(total, jnp.asarray([[recv_capacity]], jnp.int32))

            def loss(x, weights):
                return jnp.sum(run(x, weights)[0])

            grad_tokens, grad_weights = jax.jit(jax.grad(loss, argnums=(0, 1)))(
                tokens, topk_weights
            )
            np.testing.assert_allclose(
                grad_tokens,
                jnp.broadcast_to(
                    jnp.sum(topk_weights, axis=-1, keepdims=True),
                    tokens.shape,
                ),
                rtol=1e-6,
                atol=1e-6,
            )
            expected_weight_grad = jnp.repeat(
                jnp.sum(tokens, axis=-1, keepdims=True), top_k, axis=-1
            )
            np.testing.assert_allclose(grad_weights, expected_weight_grad, rtol=1e-6, atol=1e-6)
        finally:
            ep_finalize()
            ep_finalize()
