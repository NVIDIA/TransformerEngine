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
from transformer_engine.jax.cpp_extensions.ep import _ep_outer_axis
from transformer_engine.jax.cpp_extensions.misc import jax_dtype_to_te_dtype
from transformer_engine.jax.sharding import (
    _get_mesh,
    get_num_devices_in_mesh,
    global_mesh_resource,
    get_mesh_axis_size,
    with_sharding_constraint,
)

ep_prepare = tex.ep_prepare
EpLayerConfig = tex.EpLayerConfig
ep_handle_mem_size = tex.ep_handle_mem_size

__all__ = [
    "EpLayerConfig",
    "ep_bootstrap",
    "ep_handle_mem_size",
    "ep_prepare",
    "ep_dispatch",
    "ep_combine",
]

_atexit_registered = False


def _allgather_uid(uid_arr, world_size, uid_size):
    """Allgather UID bytes across all processes.

    Tries ``jax.experimental.multihost_utils.process_allgather`` first;
    falls back to an XLA collective (process-local sharded global array
    replicated via ``jax.jit``) when the multihost helper returns a
    short buffer, which has been observed under some launchers.
    """
    try:
        gathered = jmu.process_allgather(uid_arr, tiled=True)
        if gathered.size == world_size * uid_size:
            return np.asarray(gathered).reshape(world_size, uid_size)
    except Exception:  # pylint: disable=broad-except
        pass
    devices = np.asarray(jax.devices())
    if devices.size != world_size:
        raise RuntimeError(
            f"_allgather_uid fallback expected {world_size} global devices, got {devices.size}."
        )
    mesh = jax.sharding.Mesh(devices, ("_uid_all",))
    sharded = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("_uid_all", None))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    local = np.asarray(uid_arr).reshape(1, uid_size)
    g_in = jax.make_array_from_process_local_data(sharded, local, (world_size, uid_size))
    g_out = jax.jit(lambda x: x, out_shardings=replicated)(g_in)
    return np.asarray(g_out).reshape(world_size, uid_size)


# ── Bootstrap ────────────────────────────────────────────────────────────────


def _ep_domain_for_rank(mesh, ep_resource, rank, device_to_rank=None):
    """Resolve the EP domain (NCCL comm) for ``rank`` from the mesh layout.

    One domain groups ranks sharing all non-ep coordinates, so any orthogonal
    axis (tp, pp, cp, ...) yields its own domains. Returns
    ``(root_rank, rank_within_group, num_domains)``; ``root_rank`` (ep
    coordinate 0) posts the domain's NCCL unique id.
    """
    if device_to_rank is None:

        def device_to_rank(d):
            return d.process_index

    ep_pos = mesh.axis_names.index(ep_resource)
    ep_size = mesh.shape[ep_resource]
    ranks = np.vectorize(device_to_rank, otypes=[np.int64])(mesh.devices)
    # Move ep last and flatten: each row is one domain (all non-ep coords fixed).
    grid = np.moveaxis(ranks, ep_pos, -1).reshape(-1, ep_size)
    loc = np.argwhere(grid == rank)
    if loc.shape[0] != 1:
        raise ValueError(
            f"ep_bootstrap: rank {rank} must appear exactly once in the mesh device"
            f" grid; found {loc.shape[0]} occurrences."
        )
    row, col = int(loc[0, 0]), int(loc[0, 1])
    return int(grid[row, 0]), col, int(grid.shape[0])


def ep_bootstrap(
    world_size,
    rank,
    num_experts,
    max_tokens_per_rank,
    recv_capacity_per_rank,
    hidden_dim,
    max_token_dtype=jnp.bfloat16,
    max_num_sms=0,
):
    """Initialize the EP communicator. Call once per process before any EP op.

    Must run inside the active JAX Mesh and a global_shard_guard; ep_size and
    num_ep_groups are read from the mesh axes named by MeshResource.ep_resource
    and MeshResource.dp_resource/fsdp_resource. Axes orthogonal to EP (tp, pp,
    cp, ...) are supported and replicated across EP tensors.

    Args:
        world_size: Total number of processes (product of all mesh axes).
        rank: Global rank of the calling process.
        num_experts: Total experts across the EP group.
        max_tokens_per_rank: Max tokens one rank dispatches per step (sizes send buffers).
        recv_capacity_per_rank: Max tokens one rank receives per step; set to
            at least ep_size * max_tokens_per_rank * top_k to avoid drops.
        hidden_dim: Feature dimension of token tensors passed to ep_dispatch.
        max_token_dtype: Widest dtype the group will dispatch (only bfloat16 supported).
        max_num_sms: SM budget for EP kernels; 0 = auto.
    """
    if jnp.dtype(max_token_dtype) != jnp.bfloat16:
        raise NotImplementedError(
            "ep_bootstrap: only max_token_dtype=jnp.bfloat16 is supported today, got"
            f" {jnp.dtype(max_token_dtype)}."
        )
    if world_size < 2:
        raise ValueError(
            f"ep_bootstrap requires world_size >= 2 (got {world_size}); NCCL EP needs"
            " at least 2 ranks to form a group."
        )
    if jax.local_device_count() != 1:
        raise ValueError(
            "ep_bootstrap requires one local device per process (got"
            f" jax.local_device_count() = {jax.local_device_count()}); NCCL EP does not"
            " support single-process multi-device setups."
        )

    gsr = global_mesh_resource()
    ep_resource = gsr.ep_resource
    if ep_resource is None:
        raise ValueError(
            "ep_bootstrap requires MeshResource.ep_resource to be set; enter a"
            " global_shard_guard(MeshResource(..., ep_resource=<axis name>)) before bootstrap."
        )
    mesh = _get_mesh()
    if mesh.empty:
        raise ValueError(
            "ep_bootstrap must run inside an active jax.sharding.Mesh; enter"
            " `with mesh:` (or jax.set_mesh(mesh)) before calling it."
        )
    if get_num_devices_in_mesh(mesh) != world_size:
        raise ValueError(
            f"ep_bootstrap: mesh device count ({get_num_devices_in_mesh(mesh)}) must equal"
            f" world_size ({world_size})."
        )
    ep_size = get_mesh_axis_size(ep_resource)
    # num_ep_groups counts only the distinct-token (dp/fsdp) axes; replicated
    # axes (tp, pp, ...) do not create distinct EP-output slabs.
    outer_axis = _ep_outer_axis()
    num_ep_groups = 1 if outer_axis is None else get_mesh_axis_size(outer_axis)
    if num_experts % ep_size != 0:
        raise ValueError(f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size}).")

    UID_SIZE = 128
    root_rank, rank_within_group, _num_domains = _ep_domain_for_rank(mesh, ep_resource, rank)
    is_color_root = rank_within_group == 0
    if is_color_root:
        libnccl = ctypes.CDLL("libnccl.so.2", use_errno=True)
        uid_arr = (ctypes.c_uint8 * UID_SIZE)()
        ret = libnccl.ncclGetUniqueId(ctypes.cast(uid_arr, ctypes.c_void_p))
        assert ret == 0, f"ncclGetUniqueId failed with code {ret}"
        uid_bytes = bytes(uid_arr)
    else:
        uid_bytes = bytes(UID_SIZE)

    uid_arr = jnp.frombuffer(uid_bytes, dtype=jnp.uint8)
    all_uids = _allgather_uid(uid_arr, world_size, UID_SIZE)
    uid_bytes = bytes(np.asarray(all_uids[root_rank]).tolist())

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
        max_token_dtype=int(jax_dtype_to_te_dtype(max_token_dtype)),
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
            num_ep_groups=num_ep_groups,
            num_experts=num_experts,
            num_local_experts=num_experts // ep_size,
            max_tokens_per_rank=max_tokens_per_rank,
            recv_capacity_per_rank=recv_capacity_per_rank,
            hidden_dim=hidden_dim,
        )
    )


def _default_out_partition_spec():
    """Leading-axis default: ``(("dp","ep"),)`` if DP/FSDP is set, else ``("ep",)``."""
    gsr = global_mesh_resource()
    if gsr.ep_resource is None:
        raise ValueError(
            "ep_resource is not set on the active MeshResource; pass out_sharding=... explicitly."
        )
    outer = _ep_outer_axis()
    leading = (outer, gsr.ep_resource) if outer is not None else gsr.ep_resource
    return (leading,)


# ── ep_dispatch (custom_vjp) ─────────────────────────────────────────────────


@partial(jax.custom_vjp, nondiff_argnums=(0, 4))
def ep_dispatch(cfg, topk_idx, tokens, topk_weights, recv_capacity_per_rank):
    """Scatter tokens and weights to expert ranks.

    ``cfg`` is a per-layer ``EpLayerConfig``; distinct layers may share a
    ``cfg`` (the pointer-keyed C++ cache keys on handle_mem, not on cfg).
    Inputs are ``[..., H]`` with only the leading dim sharded as ``ep`` or
    ``(dp, ep)``. Returns
    ``(recv_tokens, recv_topk_weights, handle_mem, token_counts)``; pass
    ``handle_mem`` and ``token_counts`` to the matching ``ep_combine``.
    """
    return _dispatch_fwd(cfg, topk_idx, tokens, topk_weights, recv_capacity_per_rank)[0]


def _dispatch_fwd(cfg, topk_idx, tokens, topk_weights, recv_capacity_per_rank):
    if not jnp.issubdtype(topk_weights.dtype, jnp.floating):
        raise TypeError(
            f"ep_dispatch: topk_weights must be a floating dtype; got {topk_weights.dtype}."
        )
    token_counts, handle_mem = tex.ep_prepare(cfg, topk_idx)
    recv_tokens, recv_topk_weights = tex.ep_dispatch_fwd(
        cfg, handle_mem, topk_idx, tokens, topk_weights, recv_capacity_per_rank
    )
    out_leading = tuple(tokens.shape[:-1])
    primal = (recv_tokens, recv_topk_weights, handle_mem, token_counts)
    return primal, (handle_mem, out_leading)


def _dispatch_bwd(cfg, recv_capacity_per_rank, res, g_outputs):
    del recv_capacity_per_rank
    handle_mem, out_leading = res
    # Re-pin cotangent: XLA transpose can drop the EP axis and feed the FFI a global tensor.
    out_spec = _default_out_partition_spec()
    spec = jax.sharding.PartitionSpec(*out_spec)
    g_recv_tokens = with_sharding_constraint(g_outputs[0], spec)
    g_recv_topk_weights = with_sharding_constraint(g_outputs[1], spec)
    grad_tokens, grad_topk_weights = tex.ep_dispatch_bwd(
        cfg,
        handle_mem,
        g_recv_tokens,
        g_recv_topk_weights,
        out_leading,
        out_partition_spec=out_spec,
    )
    return (None, grad_tokens, grad_topk_weights)


ep_dispatch.defvjp(_dispatch_fwd, _dispatch_bwd)


# ── ep_combine (custom_vjp) ──────────────────────────────────────────────────


@partial(jax.custom_vjp, nondiff_argnums=(0, 4, 5))
def ep_combine(
    cfg,
    handle_mem,
    token_counts,
    expert_out,
    num_local_tokens,
    out_sharding=None,
):
    """Scatter-sum expert outputs back to source ranks. **Unweighted.**

    Caller must pre-multiply ``expert_out`` by ``recv_topk_weights`` (and
    zero padded slots); gradients w.r.t. weights flow through that hadamard,
    not through this op. ``num_local_tokens`` is STATIC: int -> ``[T, H]``,
    tuple -> ``[*tuple, H]``. ``out_sharding`` defaults via
    ``_default_out_partition_spec``; only the leading dim may be sharded.
    """
    return _combine_fwd(
        cfg,
        handle_mem,
        token_counts,
        expert_out,
        num_local_tokens,
        out_sharding,
    )[0]


def _combine_fwd(
    cfg,
    handle_mem,
    token_counts,
    expert_out,
    num_local_tokens,
    out_sharding,
):
    del token_counts
    if out_sharding is None:
        out_sharding = _default_out_partition_spec()
    result = tex.ep_combine_fwd(
        cfg, handle_mem, expert_out, num_local_tokens, out_partition_spec=out_sharding
    )
    return result, (handle_mem, expert_out.shape[-2])


def _combine_bwd(cfg, _num_local_tokens, _out_sharding, res, g_result):
    handle_mem, recv_capacity_per_rank = res
    # Re-pin cotangent (same XLA-transpose workaround as _dispatch_bwd).
    if _out_sharding is None:
        _out_sharding = _default_out_partition_spec()
    spec = jax.sharding.PartitionSpec(*_out_sharding)
    g_result = with_sharding_constraint(g_result, spec)
    grad_expert_out = tex.ep_combine_bwd(cfg, handle_mem, g_result, recv_capacity_per_rank)
    return (None, None, grad_expert_out)


ep_combine.defvjp(_combine_fwd, _combine_bwd)
