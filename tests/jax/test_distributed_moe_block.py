# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Distributed tests for ``transformer_engine.jax.flax.MoEBlock``."""

import sys

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec

from utils import assert_allclose, is_devices_enough


@pytest.fixture(autouse=True, scope="function")
def _inject_moe(request):
    """Lazy-load ``MoEBlock`` only for tests marked ``triton``."""
    if not request.node.get_closest_marker("triton"):
        yield
        return

    from transformer_engine.jax import MeshResource, autocast
    from transformer_engine.jax.flax import MoEBlock

    mod = sys.modules[__name__]
    mod.MeshResource = MeshResource
    mod.autocast = autocast
    mod.MoEBlock = MoEBlock
    yield


DTYPE = jnp.bfloat16
BATCH_SIZE = 2
SEQUENCE_LENGTH = 16
HIDDEN_SIZE = 64
INTERMEDIATE_SIZE = 128
NUM_EXPERTS = 8
NUM_EXPERTS_PER_TOK = 2


def _make_inputs(key: jax.Array) -> jax.Array:
    return jax.random.normal(
        key, (BATCH_SIZE, SEQUENCE_LENGTH, HIDDEN_SIZE), dtype=DTYPE
    )


def _unwrap_partitioned(x):
    return x.value if hasattr(x, "value") else x


@pytest.mark.triton
class TestDistributedMoEBlock:
    @pytest.mark.parametrize("permutation_backend", ["pure_jax", "triton"])
    def test_ep2_fsdp2_matches_single_device(self, permutation_backend):
        if not is_devices_enough(4):
            pytest.skip("MoE distributed test requires 4 devices for EP=2 x FSDP=2.")

        key = jax.random.PRNGKey(11)
        init_key, data_key = jax.random.split(key)
        inputs = _make_inputs(data_key)

        base_kwargs = dict(
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=NUM_EXPERTS_PER_TOK,
            intermediate_size=INTERMEDIATE_SIZE,
            permutation_backend=permutation_backend,
            aux_loss_coeff=1e-2,
            dtype=DTYPE,
        )

        single_block = MoEBlock(**base_kwargs)

        def loss_fn(block, variables, x):
            output, aux_loss = block.apply(variables, x)
            loss = jnp.mean(output.astype(jnp.float32) ** 2)
            if aux_loss is not None:
                loss = loss + aux_loss.astype(jnp.float32)
            return loss, (output, aux_loss)

        with autocast(enabled=False, mesh_resource=MeshResource()):
            single_variables = single_block.init(init_key, inputs)
            (single_loss, (single_output, single_aux)), single_grads = jax.value_and_grad(
                loss_fn, argnums=1, has_aux=True
            )(single_block, single_variables, inputs)

        devices = np.asarray(jax.devices()[:4]).reshape(2, 2)
        mesh = Mesh(devices, ("ep", "fsdp"))
        # FSDP-style sharding: weights are sharded on a *non-contracting*
        # weight axis (gathered before the GEMM); activations stay sharded on
        # the *batch* axis throughout - the same fsdp mesh axis is reused for
        # both. The TE primitives' custom_partitioning rules expect activations
        # FSDP-sharded on batch, so we declare ("batch", "fsdp") AND pass
        # ``input_axes=("batch", None, None)`` to enforce it on the inputs to
        # the block. ("embed", "fsdp") shards the weight's hidden dim, which
        # is gathered inside grouped_dense's custom_partitioning before GEMM
        # (no reshard of activations needed because their layout is unchanged).
        logical_axis_rules = (
            ("exp", "ep"),
            ("batch", "fsdp"),
            ("embed", "fsdp"),
        )
        sharded_block = MoEBlock(
            expert_parallelism_axis="ep",
            mesh=mesh,
            input_axes=("batch", None, None),
            **base_kwargs,
        )

        with mesh, autocast(enabled=False, mesh_resource=MeshResource(fsdp_resource="fsdp")):
            with nn.logical_axis_rules(logical_axis_rules):
                sharded_variables = sharded_block.init(init_key, inputs)
                (sharded_loss, (sharded_output, sharded_aux)), sharded_grads = (
                    jax.value_and_grad(loss_fn, argnums=1, has_aux=True)(
                        sharded_block, sharded_variables, inputs
                    )
                )

        wi_0 = _unwrap_partitioned(sharded_variables["params"]["wi_0"])
        wi_1 = _unwrap_partitioned(sharded_variables["params"]["wi_1"])
        wo = _unwrap_partitioned(sharded_variables["params"]["wo"])
        assert wi_0.sharding.spec == PartitionSpec("ep", "fsdp", None)
        assert wi_1.sharding.spec == PartitionSpec("ep", "fsdp", None)
        assert wo.sharding.spec == PartitionSpec("ep", None, "fsdp")

        assert_allclose(sharded_output, single_output, dtype=DTYPE, atol=5e-2, rtol=5e-2)
        assert_allclose(sharded_loss, single_loss, dtype=jnp.float32, atol=5e-2, rtol=5e-2)
        assert_allclose(sharded_aux, single_aux, dtype=jnp.float32, atol=5e-2, rtol=5e-2)

        for name in ("gate_kernel", "wi_0", "wi_1", "wo"):
            grad_single = _unwrap_partitioned(single_grads["params"][name])
            grad_sharded = _unwrap_partitioned(sharded_grads["params"][name])
            assert_allclose(
                grad_sharded,
                grad_single,
                dtype=DTYPE,
                atol=1e-1,
                rtol=1e-1,
                err_msg=f"Distributed gradient mismatch for {name}",
            )
