# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX Expert Parallelism (EP) API."""

import atexit
import ctypes
from functools import partial

import jax
import jax.numpy as jnp
import jax.experimental.multihost_utils as jmu
import numpy as np

import transformer_engine_jax
import transformer_engine.jax.cpp_extensions as tex
from transformer_engine.jax.cpp_extensions.ep import EpHandle
from transformer_engine.jax.sharding import global_mesh_resource, get_mesh_axis_size

ep_prepare = tex.ep_prepare

__all__ = [
    "EpHandle",
    "ep_bootstrap",
    "ep_prepare",
    "ep_dispatch",
    "ep_combine",
]

_atexit_registered = False


# ── Bootstrap ────────────────────────────────────────────────────────────────


def ep_bootstrap(
    world_size,
    rank,
    ep_size,
    num_experts,
    max_tokens_per_rank,
    recv_capacity_per_rank,
    hidden_dim,
    max_num_sms=0,
):
    """Initialize the EP communicator. Call once per process before any EP op.

    max_num_sms caps the SMs allotted to EP kernels (0 = auto).
    """
    if world_size < 2:
        raise ValueError(
            f"ep_bootstrap requires world_size >= 2 (got {world_size}); NCCL EP needs"
            " at least 2 ranks to form a group."
        )
    if world_size % ep_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by ep_size ({ep_size}); otherwise"
            " some EP groups would have fewer than ep_size ranks and ncclCommInitRank would hang."
        )
    if num_experts % ep_size != 0:
        raise ValueError(f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size}).")
    if jax.local_device_count() != 1:
        raise ValueError(
            "ep_bootstrap requires one local device per process (got"
            f" jax.local_device_count() = {jax.local_device_count()}); NCCL EP does not"
            " support single-process multi-device setups."
        )
    UID_SIZE = 128
    dp_color = rank // ep_size
    rank_within_group = rank % ep_size
    is_color_root = rank_within_group == 0
    if is_color_root:
        try:
            from nccl import get_unique_id

            uid_bytes = bytes(get_unique_id())[:UID_SIZE]
        except ImportError:
            libnccl = ctypes.CDLL("libnccl.so.2", use_errno=True)
            uid_arr = (ctypes.c_uint8 * UID_SIZE)()
            ret = libnccl.ncclGetUniqueId(ctypes.cast(uid_arr, ctypes.c_void_p))
            assert ret == 0, f"ncclGetUniqueId failed with code {ret}"
            uid_bytes = bytes(uid_arr)
    else:
        uid_bytes = bytes(UID_SIZE)

    uid_arr = jnp.frombuffer(uid_bytes, dtype=jnp.uint8)
    all_uids = jmu.process_allgather(uid_arr).reshape(world_size, UID_SIZE)
    uid_bytes = bytes(np.asarray(all_uids[dp_color * ep_size]).tolist())

    ep_resource = global_mesh_resource().ep_resource
    if ep_resource is None:
        raise ValueError(
            "ep_bootstrap requires MeshResource.ep_resource to be set; enter a"
            " global_shard_guard(MeshResource(..., ep_resource=<axis name>)) before bootstrap."
        )
    mesh_ep_size = get_mesh_axis_size(ep_resource)
    if mesh_ep_size != ep_size:
        raise ValueError(
            f"ep_bootstrap: EpConfig.ep_size ({ep_size}) does not match mesh axis"
            f" '{ep_resource}' size ({mesh_ep_size})."
        )

    # Eager NCCL init while ranks are barrier-synced by the UID broadcast above.
    transformer_engine_jax.set_ep_bootstrap_params(
        uid_bytes,
        ep_size,
        rank_within_group,
        num_experts,
        max_tokens_per_rank,
        recv_capacity_per_rank,
        hidden_dim,
        max_num_sms=int(max_num_sms),
    )

    # Release the C++ anchor at interpreter shutdown so RAII can tear down NCCL.
    global _atexit_registered
    if not _atexit_registered:
        atexit.register(transformer_engine_jax.release_ep_resources)
        _atexit_registered = True

    tex.ep.set_ep_config(
        tex.ep.EpConfig(
            world_size=world_size,
            rank=rank,
            ep_size=ep_size,
            num_experts=num_experts,
            num_local_experts=num_experts // ep_size,
            max_tokens_per_rank=max_tokens_per_rank,
            recv_capacity_per_rank=recv_capacity_per_rank,
            hidden_dim=hidden_dim,
        )
    )


# ── ep_dispatch (custom_vjp) ─────────────────────────────────────────────────


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def ep_dispatch(
    topk_idx,
    tokens,
    topk_weights,
    recv_capacity_per_rank,
    dispatch_output_per_expert_alignment=0,
):
    """Scatter tokens and weights to expert ranks.

    Inputs are 2D ``[T, H]`` or 3D ``[B, S, H]``. Only the leading dim may
    be sharded — axis ∈ {ep, (dp, ep), dp, None}; trailing dims replicated.

    Args:
        topk_idx:               ``[..., top_k]`` int32/int64 routing indices.
        tokens:                 ``[..., H]`` activations (matching leading dims).
        topk_weights:           ``[..., top_k]`` float32 routing weights.
        recv_capacity_per_rank: STATIC int. Per-rank recv slot count.
        dispatch_output_per_expert_alignment: STATIC int. Per-expert slot
                                alignment; 0 disables.

    Returns:
        ``(recv_tokens, recv_topk_weights, handle, token_counts)`` where
        ``recv_tokens`` is 3D ``[num_procs, recv_capacity_per_rank, H]``
        sharded ``(("dp","ep"), None, None)`` (or ``("ep", None, None)`` if
        DP is unset), and ``recv_topk_weights`` is 2D
        ``[num_procs, recv_capacity_per_rank]`` similarly sharded. Pass
        ``handle`` to the matching ``ep_combine``.
    """
    return _dispatch_fwd(
        topk_idx,
        tokens,
        topk_weights,
        recv_capacity_per_rank,
        dispatch_output_per_expert_alignment,
    )[0]


def _dispatch_fwd(
    topk_idx,
    tokens,
    topk_weights,
    recv_capacity_per_rank,
    dispatch_output_per_expert_alignment,
):
    top_k = int(topk_weights.shape[-1])
    token_counts, handle = tex.ep_prepare(topk_idx, dispatch_output_per_expert_alignment)
    recv_tokens, recv_topk_weights, handle = tex.ep_dispatch_fwd(
        handle, topk_idx, tokens, topk_weights, recv_capacity_per_rank
    )
    out_leading = tuple(tokens.shape[:-1])
    primal = (recv_tokens, recv_topk_weights, handle, token_counts)
    return primal, (handle, out_leading, top_k)


def _dispatch_bwd(recv_capacity_per_rank, dispatch_output_per_expert_alignment, res, g_outputs):
    del recv_capacity_per_rank, dispatch_output_per_expert_alignment
    handle, out_leading, top_k = res
    # Re-pin cotangent sharding: XLA transpose can drop the EP axis on a
    # single-fwd-output cotangent, landing a global tensor in the FFI.
    gsr = global_mesh_resource()
    ep_axis = gsr.ep_resource
    outer = gsr.dp_resource or gsr.fsdp_resource
    leading = (outer, ep_axis) if outer is not None else ep_axis
    g_recv_tokens = jax.lax.with_sharding_constraint(
        g_outputs[0], jax.sharding.PartitionSpec(leading, None, None)
    )
    g_recv_topk_weights = jax.lax.with_sharding_constraint(
        g_outputs[1], jax.sharding.PartitionSpec(leading, None)
    )
    grad_tokens, grad_topk_weights = tex.ep_dispatch_bwd(
        handle, g_recv_tokens, g_recv_topk_weights, top_k, out_leading
    )
    return (None, grad_tokens, grad_topk_weights)


ep_dispatch.defvjp(_dispatch_fwd, _dispatch_bwd)


# ── ep_combine (custom_vjp) ──────────────────────────────────────────────────


@partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def ep_combine(
    handle, token_counts, expert_out, recv_topk_weights, num_local_tokens, out_sharding=None
):
    """Reduce weighted expert outputs back to source ranks.

    Args:
        handle:            ``EpHandle`` from a matching ``ep_dispatch`` call.
        token_counts:      ``[num_procs, num_local_experts]`` int32 (passed through).
        expert_out:        ``[num_procs, recv_capacity_per_rank, H]`` post-FFN activations.
        recv_topk_weights: ``[num_procs, recv_capacity_per_rank]`` float32 weights
                           returned by ``ep_dispatch``.
        num_local_tokens:  STATIC int or tuple. int → 2D output ``[T, H]``;
                           tuple → N-D output ``[*tuple, H]``.
        out_sharding:      STATIC optional ``PartitionSpec`` tuple for the
                           output. Defaults to ``(("dp","ep"), *None)`` when
                           DP is set, else ``("ep", *None)``. Pass a custom
                           spec to override; only the leading dim may be
                           sharded.

    Returns:
        ``[..., H]`` combined output shaped per ``num_local_tokens``.
    """
    return _combine_fwd(
        handle, token_counts, expert_out, recv_topk_weights, num_local_tokens, out_sharding
    )[0]


def _make_valid_mask(recv_topk_weights, dtype):
    # recv_topk_weights == 0 marks a padded slot.
    return (recv_topk_weights != 0).astype(dtype)[..., None]


def _combine_fwd(
    handle, token_counts, expert_out, recv_topk_weights, num_local_tokens, out_sharding
):
    del token_counts
    w = recv_topk_weights[..., None]
    mask = _make_valid_mask(recv_topk_weights, jnp.float32)
    weighted = (expert_out.astype(jnp.float32) * w * mask).astype(expert_out.dtype)
    result = tex.ep_combine_fwd(handle, weighted, num_local_tokens, out_partition_spec=out_sharding)
    return result, (handle, recv_topk_weights, expert_out)


def _combine_bwd(_num_local_tokens, _out_sharding, res, g_result):
    handle, recv_topk_weights, expert_out = res
    # expert_out is [..., recv_pr, H]; pull recv_pr from the second-to-last dim.
    recv_capacity_per_rank = expert_out.shape[-2]
    # Re-pin cotangent sharding: same XLA-transpose workaround as _dispatch_bwd.
    gsr = global_mesh_resource()
    if _out_sharding is not None:
        spec = jax.sharding.PartitionSpec(*_out_sharding)
    else:
        ep_axis = gsr.ep_resource
        outer = gsr.dp_resource or gsr.fsdp_resource
        leading = (outer, ep_axis) if outer is not None and ep_axis is not None else ep_axis
        spec = (
            jax.sharding.PartitionSpec(leading, *([None] * (g_result.ndim - 1)))
            if leading is not None
            else None
        )
    if spec is not None:
        g_result = jax.lax.with_sharding_constraint(g_result, spec)
    grad_weighted = tex.ep_combine_bwd(handle, g_result, recv_capacity_per_rank)
    w = recv_topk_weights[..., None]
    mask = _make_valid_mask(recv_topk_weights, jnp.float32)
    grad_weighted_f32 = grad_weighted.astype(jnp.float32)
    grad_expert_out = (grad_weighted_f32 * w * mask).astype(grad_weighted.dtype)
    grad_recv_topk_weights = (
        (grad_weighted_f32 * expert_out.astype(jnp.float32) * mask)
        .sum(axis=-1)
        .astype(recv_topk_weights.dtype)
    )
    return (None, None, grad_expert_out, grad_recv_topk_weights)


ep_combine.defvjp(_combine_fwd, _combine_bwd)
