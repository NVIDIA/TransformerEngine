# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""JAX: context-parallel THD attention with Transformer Engine.

Companion source for ``attention_context_parallel.rst``. Code blocks between
``# ATTENTION_CP_*_START`` / ``# ATTENTION_CP_*_END`` markers are pulled into
the RST via ``literalinclude``.

Run as a script to exercise the example end-to-end:

    python docs/examples/jax/attention_context_parallel.py
"""

# ATTENTION_CP_IMPORTS_START
import os
import time
from typing import Tuple

# Ring + SWA uses the non-scan Ring implementation. Set this before JAX compiles
# the first fused attention call so the example follows the distributed tests.
os.environ.setdefault("NVTE_FUSED_RING_ATTENTION_USE_SCAN", "0")

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import transformer_engine.jax as te
from transformer_engine.jax.attention import (
    AttnBiasType,
    AttnMaskType,
    AttnSoftmaxType,
    CPStrategy,
    QKVLayout,
    ReorderStrategy,
    SequenceDescriptor,
    fused_attn,
    inverse_reorder_causal_load_balancing,
    is_fused_attn_kernel_available,
    reorder_causal_load_balancing,
)
from transformer_engine.jax.sharding import MeshResource

# ATTENTION_CP_IMPORTS_END


# ATTENTION_CP_INPUTS_START
cp_size = 4
batch, seq, num_attention_heads, head_dim = 2, 65536, 128, 128
runtime_segments_per_seq = 16
max_segments_per_seq = runtime_segments_per_seq
window_size = (128, 0)
dtype = jnp.bfloat16
timing_iters = 5
warmup_iters = 2
ring_stripe_size = 1
ag_stripe_size = 4096


def create_qkv_inputs(seed: int = 2026):
    q_key, k_key, v_key, dout_key = jax.random.split(jax.random.PRNGKey(seed), 4)
    shape = (batch, seq, num_attention_heads, head_dim)
    q = jax.random.normal(q_key, shape).astype(dtype)
    k = jax.random.normal(k_key, shape).astype(dtype)
    v = jax.random.normal(v_key, shape).astype(dtype)
    dout = jax.random.normal(dout_key, shape).astype(dtype)
    return q, k, v, dout


def create_packed_segment_ids_and_pos():
    """Pack padded causal segments into each THD batch row."""

    segment_slot_len = seq // runtime_segments_per_seq
    valid_segment_len = 3 * segment_slot_len // 4
    segment_ids_per_row = []
    segment_pos_per_row = []

    for segment_id in range(1, runtime_segments_per_seq + 1):
        valid_ids = jnp.full((valid_segment_len,), segment_id, dtype=jnp.int32)
        padded_ids = jnp.zeros((segment_slot_len - valid_segment_len,), dtype=jnp.int32)
        segment_ids_per_row.append(jnp.concatenate([valid_ids, padded_ids]))
        segment_pos_per_row.append(jnp.arange(segment_slot_len, dtype=jnp.int32))

    segment_ids = jnp.concatenate(segment_ids_per_row)
    segment_pos = jnp.concatenate(segment_pos_per_row)
    segment_ids = jnp.tile(segment_ids[None, :], (batch, 1))
    segment_pos = jnp.tile(segment_pos[None, :], (batch, 1))
    return segment_ids, segment_pos


def create_sequence_descriptor(segment_ids_arg, segment_pos_arg):
    """Create the THD sequence descriptor from segment IDs and positions."""

    return SequenceDescriptor.from_segment_ids_and_pos(segment_ids_arg, segment_pos_arg)


q, k, v, dout = create_qkv_inputs()
segment_ids, segment_pos = create_packed_segment_ids_and_pos()
sequence_descriptor = create_sequence_descriptor(segment_ids, segment_pos)
# ATTENTION_CP_INPUTS_END


# ATTENTION_CP_MESH_START
def build_cp_mesh():
    """Use one JAX mesh axis for context parallelism over sequence."""

    devices = np.asarray(jax.devices()[:cp_size])
    mesh = Mesh(devices, axis_names=("cp",))
    mesh_resource = MeshResource(cp_resource="cp")
    return mesh, mesh_resource


# ATTENTION_CP_MESH_END


# ATTENTION_CP_FUSED_ATTENTION_START
def fused_thd_attention(
    qkv_tensors,
    seq_desc,
    *,
    context_parallel_axis: str = "",
    context_parallel_strategy: CPStrategy = CPStrategy.DEFAULT,
    context_parallel_causal_load_balanced: bool = False,
    stripe_size: int | None = None,
):
    """Call TE fused attention on separate THD Q, K, V tensors."""

    return fused_attn(
        qkv_tensors,
        None,
        seq_desc,
        None,
        attn_bias_type=AttnBiasType.NO_BIAS,
        attn_mask_type=AttnMaskType.PADDING_CAUSAL_MASK,
        qkv_layout=QKVLayout.THD_THD_THD,
        softmax_type=AttnSoftmaxType.VANILLA_SOFTMAX,
        scaling_factor=head_dim**-0.5,
        dropout_probability=0.0,
        is_training=True,
        max_segments_per_seq=max_segments_per_seq,
        window_size=window_size,
        context_parallel_strategy=context_parallel_strategy,
        context_parallel_causal_load_balanced=context_parallel_causal_load_balanced,
        context_parallel_axis=context_parallel_axis,
        stripe_size=stripe_size,
    )


def apply_context_parallel_attention(
    _variables,
    qkv_tensors,
    *,
    seq_desc,
    context_parallel_strategy: CPStrategy,
    stripe_size: int,
    rngs=None,
):
    del rngs
    return fused_thd_attention(
        qkv_tensors,
        seq_desc,
        context_parallel_axis="cp",
        context_parallel_strategy=context_parallel_strategy,
        context_parallel_causal_load_balanced=True,
        stripe_size=stripe_size,
    )


# ATTENTION_CP_FUSED_ATTENTION_END


# ATTENTION_CP_REORDER_START
def reorder_for_context_parallel(x, stripe_size: int):
    return reorder_causal_load_balancing(
        x,
        strategy=ReorderStrategy.Striped,
        cp_size=cp_size,
        seq_dim=1,
        stripe_size=stripe_size,
    )


def inverse_reorder_from_context_parallel(x, stripe_size: int):
    return inverse_reorder_causal_load_balancing(
        x,
        strategy=ReorderStrategy.Striped,
        cp_size=cp_size,
        seq_dim=1,
        stripe_size=stripe_size,
    )


def create_reordered_sequence_descriptor(stripe_size: int):
    reordered_ids = reorder_for_context_parallel(segment_ids, stripe_size)
    reordered_pos = reorder_for_context_parallel(segment_pos, stripe_size)
    return create_sequence_descriptor(reordered_ids, reordered_pos)


# ATTENTION_CP_REORDER_END


# ATTENTION_CP_SHARD_START
def shard_sequence_descriptor(mesh, seq_desc):
    def put_leaf(x):
        if x.ndim == 1:
            sharding = NamedSharding(mesh, P(None))
        else:
            sharding = NamedSharding(mesh, P(None, "cp"))
        return jax.device_put(x, sharding)

    return jax.tree.map(put_leaf, seq_desc)


def shard_for_context_parallel(mesh, stripe_size: int):
    qkv_sharding = NamedSharding(mesh, P(None, "cp", None, None))
    dout_sharding = NamedSharding(mesh, P(None, "cp", None, None))
    reordered_seq_desc = create_reordered_sequence_descriptor(stripe_size)

    return {
        "qkv": tuple(
            jax.device_put(reorder_for_context_parallel(x, stripe_size), qkv_sharding)
            for x in (q, k, v)
        ),
        "dout": jax.device_put(reorder_for_context_parallel(dout, stripe_size), dout_sharding),
        "sequence_descriptor": shard_sequence_descriptor(mesh, reordered_seq_desc),
    }


# ATTENTION_CP_SHARD_END


def _strategy_name(strategy: CPStrategy):
    return "Ring" if strategy == CPStrategy.RING else "AllGather"


def context_parallel_supported() -> Tuple[bool, str]:
    if len(jax.devices()) < cp_size:
        return False, f"needs {cp_size} GPUs"

    has_kernel = is_fused_attn_kernel_available(
        True,
        dtype,
        dtype,
        QKVLayout.THD_THD_THD,
        AttnBiasType.NO_BIAS,
        AttnMaskType.PADDING_CAUSAL_MASK,
        AttnSoftmaxType.VANILLA_SOFTMAX,
        0.0,
        num_attention_heads,
        num_attention_heads,
        seq,
        seq,
        head_dim,
        head_dim,
        window_size,
    )
    if not has_kernel:
        return False, "no fused attention kernel for the THD SWA shape"
    return True, ""


def run_reference_attention():
    out = fused_thd_attention((q, k, v), sequence_descriptor)
    return jax.block_until_ready(out)


# ATTENTION_CP_RUN_START
def _context_parallel_jit_fns(strategy: CPStrategy, stripe_size: int, sharded):
    qkv_shardings = tuple(x.sharding for x in sharded["qkv"])
    seq_desc_shardings = jax.tree.map(lambda x: x.sharding, sharded["sequence_descriptor"])
    dout_sharding = sharded["dout"].sharding

    def loss_fn(qkv_arg, seq_desc_arg, dout_arg):
        out = apply_context_parallel_attention(
            {},
            qkv_arg,
            seq_desc=seq_desc_arg,
            context_parallel_strategy=strategy,
            stripe_size=stripe_size,
        )
        return jnp.vdot(out.astype(jnp.float32), dout_arg.astype(jnp.float32))

    def forward_fn(qkv_arg, seq_desc_arg):
        out = apply_context_parallel_attention(
            {},
            qkv_arg,
            seq_desc=seq_desc_arg,
            context_parallel_strategy=strategy,
            stripe_size=stripe_size,
        )
        return inverse_reorder_from_context_parallel(out, stripe_size)

    grad_fn = jax.jit(
        jax.value_and_grad(loss_fn),
        in_shardings=(qkv_shardings, seq_desc_shardings, dout_sharding),
        out_shardings=(None, qkv_shardings),
    )
    forward_jit = jax.jit(
        forward_fn,
        in_shardings=(qkv_shardings, seq_desc_shardings),
    )
    return grad_fn, forward_jit


def run_context_parallel_case(strategy: CPStrategy, stripe_size: int):
    mesh, mesh_resource = build_cp_mesh()
    sharded = shard_for_context_parallel(mesh, stripe_size)
    grad_fn, forward_jit = _context_parallel_jit_fns(strategy, stripe_size, sharded)

    with jax.set_mesh(mesh), te.autocast(mesh_resource=mesh_resource):
        loss, grads = grad_fn(
            sharded["qkv"],
            sharded["sequence_descriptor"],
            sharded["dout"],
        )
        out = forward_jit(sharded["qkv"], sharded["sequence_descriptor"])

    jax.block_until_ready((loss, grads, out))
    return {"loss": loss, "grads": grads, "output": out}


def run_context_parallel_bench(strategy: CPStrategy, stripe_size: int):
    mesh, mesh_resource = build_cp_mesh()
    sharded = shard_for_context_parallel(mesh, stripe_size)
    grad_fn, _ = _context_parallel_jit_fns(strategy, stripe_size, sharded)

    print(f"THD CP {_strategy_name(strategy)} stripe_size={stripe_size}:")
    with jax.set_mesh(mesh), te.autocast(mesh_resource=mesh_resource):
        for _ in range(warmup_iters):
            result = grad_fn(
                sharded["qkv"],
                sharded["sequence_descriptor"],
                sharded["dout"],
            )
        jax.block_until_ready(result)

        start = time.time()
        for _ in range(timing_iters):
            result = grad_fn(
                sharded["qkv"],
                sharded["sequence_descriptor"],
                sharded["dout"],
            )
        jax.block_until_ready(result)
        end = time.time()

    print(f"Mean time: {(end - start) * 1000 / timing_iters} ms")


# ATTENTION_CP_RUN_END


if __name__ == "__main__":
    supported, reason = context_parallel_supported()
    if not supported:
        print(f"skipped context-parallel example: {reason}")
    else:
        print("# RING_OUTPUT_START")
        run_context_parallel_bench(CPStrategy.RING, ring_stripe_size)
        ring_result = run_context_parallel_case(CPStrategy.RING, ring_stripe_size)
        print(
            "Ring output shape="
            f"{tuple(ring_result['output'].shape)}, dtype={ring_result['output'].dtype}"
        )
        print(f"Ring grad shapes={[tuple(grad.shape) for grad in ring_result['grads']]}")
        print("# RING_OUTPUT_END")

        print("\n# AG_OUTPUT_START")
        run_context_parallel_bench(CPStrategy.ALL_GATHER, ag_stripe_size)
        ag_result = run_context_parallel_case(CPStrategy.ALL_GATHER, ag_stripe_size)
        print(
            "AllGather output shape="
            f"{tuple(ag_result['output'].shape)}, dtype={ag_result['output'].dtype}"
        )
        print(f"AllGather grad shapes={[tuple(grad.shape) for grad in ag_result['grads']]}")
        print("# AG_OUTPUT_END")
