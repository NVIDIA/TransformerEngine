# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Multiprocess MXFP8BlockScaling VJP coverage for the TE EP MoE path."""

import os
import sys
from contextlib import ExitStack

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("NVTE_JAX_ENFORCE_V2_GROUPED_GEMM", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.linen import partitioning as nn_partitioning
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def _read_mp_options():
    num_processes = 0
    process_id = 0
    for index, arg in enumerate(sys.argv):
        if arg.startswith("--num-process="):
            num_processes = int(arg.split("=", 1)[1])
        elif arg == "--num-process" and index + 1 < len(sys.argv):
            num_processes = int(sys.argv[index + 1])
        elif arg.startswith("--process-id="):
            process_id = int(arg.split("=", 1)[1])
        elif arg == "--process-id" and index + 1 < len(sys.argv):
            process_id = int(sys.argv[index + 1])
    return num_processes, process_id


_NUM_PROCESSES, _PROCESS_ID = _read_mp_options()
if _NUM_PROCESSES <= 1:
    pytest.skip("requires tests/jax/run_te_ep_moe.sh", allow_module_level=True)

jax.distributed.initialize(
    coordinator_address=os.environ.get("TE_EP_MOE_COORDINATOR_ADDRESS", "127.0.0.1:13457"),
    num_processes=_NUM_PROCESSES,
    process_id=_PROCESS_ID,
    local_device_ids=_PROCESS_ID,
)

from transformer_engine.common.recipe import MXFP8BlockScaling
from transformer_engine.jax.ep import ep_bootstrap
from transformer_engine.jax.moe import (
    get_moe_recv_capacity_per_rank,
    moe,
    record_ep_bootstrap_signature_for_moe,
)
from transformer_engine.jax.quantize import (
    QuantizeMeta,
    QuantizeMetaSet,
    QuantizerFactory,
    QuantizerSet,
    TensorSource,
    get_quantize_config_with_recipe,
)
from transformer_engine.jax.sharding import MeshResource, global_shard_guard
from transformer_engine_jax import get_device_compute_capability


if get_device_compute_capability(0) < 100:
    pytest.skip("MXFP8 grouped GEMM requires Blackwell", allow_module_level=True)

EP_AXIS = "ep"
FSDP_AXIS = "fsdp"
EP_SIZE = 2
FSDP_SIZE = jax.device_count() // EP_SIZE
NUM_EXPERTS = 8
TOPK = 2
BATCH = jax.device_count() * 2
SEQ = 32
HIDDEN = 128
INTERMEDIATE = 128
DTYPE = jnp.bfloat16

LOGICAL_AXIS_RULES = (
    ("batch", (FSDP_AXIS, EP_AXIS)),
    ("exp", EP_AXIS),
    ("embed", FSDP_AXIS),
    ("mlp", None),
)


def _make_mxfp8_quantizer_sets():
    """Construct MaxText-shaped quantizers from MXFP8BlockScaling."""
    recipe = MXFP8BlockScaling()
    config = get_quantize_config_with_recipe(recipe)
    meta_set = QuantizeMetaSet(x=QuantizeMeta(), kernel=QuantizeMeta(), grad=QuantizeMeta())

    def _set(n_token_groups, n_expert_groups):
        token_set = QuantizerFactory.create_set(
            fp8_recipe=recipe,
            n_groups=n_token_groups,
            quantize_meta_set=meta_set,
        )
        expert_set = QuantizerFactory.create_set(
            fp8_recipe=recipe,
            n_groups=n_expert_groups,
            quantize_meta_set=meta_set,
        )
        for source in TensorSource:
            assert config.get_scaling_mode(source).is_mxfp8_scaling
        return QuantizerSet(x=token_set.x, kernel=expert_set.kernel, dgrad=token_set.dgrad)

    # Match MaxText: global dispatch groups include FSDP replicas, while
    # expert weights have one group per global expert.
    return tuple(_set(FSDP_SIZE * NUM_EXPERTS, NUM_EXPERTS) for _ in range(2))


@pytest.fixture(scope="module")
def mesh():
    devices = mesh_utils.create_device_mesh((FSDP_SIZE, EP_SIZE))
    mesh_obj = Mesh(devices, axis_names=(FSDP_AXIS, EP_AXIS))
    max_tokens_per_rank = (BATCH // jax.process_count()) * SEQ
    recv_capacity = get_moe_recv_capacity_per_rank(
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOPK,
        max_tokens_per_rank=max_tokens_per_rank,
        ep_size=EP_SIZE,
    )
    with mesh_obj, global_shard_guard(
        MeshResource(ep_resource=EP_AXIS, fsdp_resource=FSDP_AXIS)
    ):
        ep_bootstrap(
            world_size=jax.process_count(),
            rank=jax.process_index(),
            num_experts=NUM_EXPERTS,
            max_tokens_per_rank=max_tokens_per_rank,
            recv_capacity_per_rank=recv_capacity,
            hidden_dim=HIDDEN,
            max_token_dtype=DTYPE,
        )
    record_ep_bootstrap_signature_for_moe(
        num_experts=NUM_EXPERTS,
        max_tokens_per_rank=max_tokens_per_rank,
        recv_capacity_per_rank=recv_capacity,
        hidden_dim=HIDDEN,
        ep_size=EP_SIZE,
    )
    return mesh_obj


def _context(mesh):
    stack = ExitStack()
    stack.enter_context(mesh)
    stack.enter_context(
        global_shard_guard(MeshResource(ep_resource=EP_AXIS, fsdp_resource=FSDP_AXIS))
    )
    stack.enter_context(nn_partitioning.axis_rules(LOGICAL_AXIS_RULES))
    return stack


def _global_array(value, mesh):
    with mesh:
        value = jax.jit(
            lambda x: jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P()))
        )(value)
        value.block_until_ready()
    return np.asarray(jax.device_get(value.addressable_data(0)))


def test_mxfp8_block_scaling_forward_and_vjp(mesh):
    keys = jax.random.split(jax.random.PRNGKey(123), 5)
    x = jax.random.normal(keys[0], (BATCH, SEQ, HIDDEN), DTYPE)
    gate = jax.random.normal(keys[1], (HIDDEN, NUM_EXPERTS), DTYPE) / jnp.sqrt(HIDDEN)
    wi = jax.random.normal(keys[2], (NUM_EXPERTS, HIDDEN, 2 * INTERMEDIATE), DTYPE) / jnp.sqrt(
        HIDDEN
    )
    wo = jax.random.normal(keys[3], (NUM_EXPERTS, INTERMEDIATE, HIDDEN), DTYPE) / jnp.sqrt(
        INTERMEDIATE
    )
    cotangent = jax.random.normal(keys[4], x.shape, DTYPE)
    quantizer_sets = _make_mxfp8_quantizer_sets()

    def forward(x_arg, gate_arg, wi_arg, wo_arg):
        output, _, total_recv_tokens = moe(
            x_arg,
            gate_arg,
            wi_arg,
            wo_arg,
            None,
            None,
            None,
            None,
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=TOPK,
            apply_topk_weights_early=True,
            quantizer_sets=quantizer_sets,
            ep_axis=EP_AXIS,
            data_parallelism_axes=(FSDP_AXIS,),
            input_axes=("batch", None, None),
            gate_kernel_axes=("embed", "exp"),
            wi_kernel_axes=("exp", "embed", "mlp"),
            wo_kernel_axes=("exp", "mlp", "embed"),
            dtype=DTYPE,
        )
        return output, total_recv_tokens

    def value_and_vjp(x_arg, gate_arg, wi_arg, wo_arg, cotangent_arg):
        output, pullback = jax.vjp(lambda a, b, c, d: forward(a, b, c, d)[0], x_arg, gate_arg, wi_arg, wo_arg)
        return output, pullback(cotangent_arg)

    with _context(mesh):
        x = jax.lax.with_sharding_constraint(
            x, NamedSharding(mesh, P((FSDP_AXIS, EP_AXIS), None, None))
        )
        output, grads = jax.jit(value_and_vjp)(x, gate, wi, wo, cotangent)
        jax.block_until_ready((output, grads))

    assert output.shape == x.shape
    assert output.dtype == DTYPE
    output_np = _global_array(output, mesh)
    assert np.all(np.isfinite(output_np))
    assert np.any(output_np != 0)

    expected_shapes = (x.shape, gate.shape, wi.shape, wo.shape)
    for name, grad, shape in zip(("x", "gate", "wi", "wo"), grads, expected_shapes):
        assert grad.shape == shape, f"{name} gradient shape mismatch"
        grad_np = _global_array(grad, mesh)
        assert np.all(np.isfinite(grad_np)), f"{name} gradient has NaN/Inf"
        assert np.any(grad_np != 0), f"{name} gradient is identically zero"
