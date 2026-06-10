# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX/TE custom ops for Expert Parallelism (EP).

Sharding model:
  - EpPrepare / EpDispatch outputs carry a single leading ``num_procs`` dim.
    Sharded compound ``(dp_resource, ep_resource)`` when DP is set, else
    ``ep_resource`` alone.
  - EpDispatch inputs are 2D ``[T, H]`` or 3D ``[B, S, H]``; only the first
    dim may be sharded, with axis in {ep, (dp, ep), dp, None}. Trailing dims
    must be replicated. ``dp`` alone gets ``ep`` folded in locally.
  - EpCombine output sharding comes from ``out_sharding`` or defaults to the
    compound ``(dp, ep)`` axis on the leading dim.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import dtypes, ffi
from jax.sharding import NamedSharding, PartitionSpec

import transformer_engine_jax
from .base import BasePrimitive, register_primitive
from ..sharding import global_mesh_resource

__all__ = [
    "EpConfig",
    "EpLayerConfig",
    "set_ep_config",
    "get_ep_config",
    "get_ep_num_local_experts",
    "ep_handle_mem_size",
    "ep_prepare",
    "ep_dispatch_fwd",
    "ep_combine_fwd",
    "ep_dispatch_bwd",
    "ep_combine_bwd",
]


# ── Module-level EP config ──────────────────────────────────────────────────


@dataclass(frozen=True)
class EpConfig:
    """Snapshot of the EP bootstrap config (see ep_bootstrap).

    num_ep_groups is the size of the outer dp/fsdp mesh axis (1 if neither
    is set), captured at bootstrap so abstract-eval never reads the mesh.
    """

    world_size: int
    rank: int
    ep_size: int
    num_ep_groups: int
    num_experts: int
    num_local_experts: int
    max_tokens_per_rank: int
    recv_capacity_per_rank: int
    hidden_dim: int


_ep_config: EpConfig = None


def set_ep_config(config: EpConfig) -> None:
    """Cache the EP config for abstract-eval / sharding helpers. Call once."""
    global _ep_config
    _ep_config = config


def get_ep_config() -> EpConfig:
    """Return the process-wide EpConfig set by ep_bootstrap."""
    if _ep_config is None:
        raise RuntimeError("EpConfig has not been set. Did you call ep_bootstrap()?")
    return _ep_config


def get_ep_num_local_experts() -> int:
    """Number of experts owned by this EP rank."""
    return get_ep_config().num_local_experts


@dataclass(frozen=True)
class EpLayerConfig:
    """Per-layer EP config; mirrors C ``NVTEEpLayerConfig``.

    Threaded through every per-step op so the pointer-keyed C++ cache can
    validate consistency across a handle_mem's prepare / dispatch / combine.
    Reserved for future per-call fields (fp8 scale, overflow policy, ...).
    """

    top_k: int
    dispatch_output_per_expert_alignment: int = 0


def ep_handle_mem_size(cfg: EpLayerConfig) -> int:
    """Return the handle_mem byte size for ``cfg``. Host-only; cheap."""
    return int(
        transformer_engine_jax.ep_handle_mem_size(
            int(cfg.top_k), int(cfg.dispatch_output_per_expert_alignment)
        )
    )


def _leading_axis_ok(spec):
    """Validate an EP input spec; return ``(ok, ep_axis, outer_axes)``.

    Leading dim is ``ep`` or a tuple ending in ``ep`` (outer dp/fsdp axes
    first); all other dims must be replicated.
    """
    gsr = global_mesh_resource()
    ep_axis = gsr.ep_resource
    outer_axes = tuple(a for a in (gsr.dp_resource, gsr.fsdp_resource) if a is not None)
    if len(spec) < 2 or ep_axis is None:
        return False, ep_axis, outer_axes
    if any(ax is not None for ax in spec[1:]):
        return False, ep_axis, outer_axes
    leading = spec[0]
    elts = leading if isinstance(leading, tuple) else (leading,)
    if ep_axis not in elts:
        return False, ep_axis, outer_axes
    allowed = set(outer_axes) | {ep_axis}
    return all(a in allowed for a in elts), ep_axis, outer_axes


def _ep_outer_axis():
    """The single dp/fsdp axis (if any) sitting outside ep on EP-output tensors.

    When set, EP-output globals carry an extra leading ``dp_size`` dim so SPMD
    sees each DP color's slab as distinct (rather than replicated across DP).
    """
    gsr = global_mesh_resource()
    return gsr.dp_resource or gsr.fsdp_resource


def _ep_leading_dims(is_outer):
    """Leading dim of an EP-output tensor: num_ep_groups*ep_size globally,
    1 per shard. Read from EpConfig so abstract-eval needs no active mesh."""
    cfg = get_ep_config()
    if not is_outer:
        return (1,)
    return (cfg.num_ep_groups * cfg.ep_size,)


def _ep_output_spec(*trailing):
    """PartitionSpec for an EP-output tensor: ``(("dp","ep"), *trailing)`` when
    DP is set (compound leading axis on a single dim), else ``("ep",*trailing)``."""
    gsr = global_mesh_resource()
    outer = _ep_outer_axis()
    if outer is None:
        return PartitionSpec(gsr.ep_resource, *trailing)
    return PartitionSpec((outer, gsr.ep_resource), *trailing)


def _ep_spec_ok(spec, trailing_count):
    """Accept ``(ep, *[None])`` (no DP) or ``((dp,ep), *[None])`` /
    ``(("dp",), *[None])`` / ``("dp", *[None])`` / ``(None, *[None])`` (with DP)
    on an EP-output tensor's single leading dim. JAX may collapse a size-1
    mesh axis to ``None`` (matters for dp_size=1 like 1x4)."""
    gsr = global_mesh_resource()
    ep_axis = gsr.ep_resource
    outer = _ep_outer_axis()
    expected_len = 1 + trailing_count
    if len(spec) != expected_len:
        return False
    if any(ax is not None for ax in spec[1:]):
        return False
    leading = spec[0]
    if outer is None:
        return leading == ep_axis
    allowed = {ep_axis, outer, None}
    elts = leading if isinstance(leading, tuple) else (leading,)
    return all(a in allowed for a in elts)


# ── ep_prepare ──────────────────────────────────────────────────────────────


class EpPreparePrimitive(BasePrimitive):
    """FFI primitive for nvte_ep_prepare: routing setup and per-expert token counts."""

    name = "te_ep_prepare_ffi"
    multiple_results = True
    impl_static_args = (1, 2, 3)  # top_k, dispatch_output_per_expert_alignment, is_outer
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(topk_idx_aval, *, top_k, dispatch_output_per_expert_alignment, is_outer):
        # is_outer=True: global leading dim = (dp*ep,) (or (ep,) with no DP);
        # False: per-shard = (1,).
        cfg = get_ep_config()
        num_local_experts = cfg.num_local_experts
        assert (
            len(topk_idx_aval.shape) >= 2
        ), f"topk_idx must be at least 2D [..., top_k], got shape {topk_idx_aval.shape}"
        handle_mem_size = int(
            transformer_engine_jax.ep_handle_mem_size(
                int(top_k), int(dispatch_output_per_expert_alignment)
            )
        )
        leading = _ep_leading_dims(is_outer)
        token_counts_aval = jax.core.ShapedArray(leading + (num_local_experts,), jnp.int32)
        handle_mem_aval = jax.core.ShapedArray(leading + (handle_mem_size,), jnp.uint8)
        # FFI scratch for the int32 -> int64 topk_idx upcast. int32 with last
        # dim doubled to keep the int64 byte count without JAX_ENABLE_X64.
        # TODO(phuong): drop once NCCL EP supports int32 topk_idx.
        workspace_shape = topk_idx_aval.shape[:-1] + (topk_idx_aval.shape[-1] * 2,)
        workspace_aval = jax.core.ShapedArray(workspace_shape, jnp.int32)
        return token_counts_aval, handle_mem_aval, workspace_aval

    @staticmethod
    def outer_abstract(topk_idx_aval, *, top_k, dispatch_output_per_expert_alignment, is_outer):
        del is_outer
        avals = EpPreparePrimitive.abstract(
            topk_idx_aval,
            top_k=top_k,
            dispatch_output_per_expert_alignment=dispatch_output_per_expert_alignment,
            is_outer=True,
        )
        return avals[:2]

    @staticmethod
    def lowering(ctx, topk_idx, *, top_k, dispatch_output_per_expert_alignment, is_outer):
        del is_outer
        return ffi.ffi_lowering(EpPreparePrimitive.name)(
            ctx,
            topk_idx,
            top_k=int(top_k),
            dispatch_output_per_expert_alignment=int(dispatch_output_per_expert_alignment),
        )

    @staticmethod
    def impl(topk_idx, top_k, dispatch_output_per_expert_alignment, is_outer):
        assert EpPreparePrimitive.inner_primitive is not None
        token_counts, handle_mem, _workspace = EpPreparePrimitive.inner_primitive.bind(
            topk_idx,
            top_k=top_k,
            dispatch_output_per_expert_alignment=dispatch_output_per_expert_alignment,
            is_outer=is_outer,
        )
        return token_counts, handle_mem

    @staticmethod
    def batcher(
        batched_args, batch_dims, *, top_k, dispatch_output_per_expert_alignment, is_outer
    ):
        raise NotImplementedError("EpPreparePrimitive does not support vmap")

    @staticmethod
    def partition(
        top_k, dispatch_output_per_expert_alignment, is_outer, mesh, arg_infos, result_infos
    ):
        del is_outer, result_infos
        idx_spec = arg_infos[0].sharding.spec
        ok, ep_axis, outer_axes = _leading_axis_ok(idx_spec)
        if not ok:
            raise NotImplementedError(
                "EpPrepare: topk_idx leading dim must include ep_resource"
                f" ('{ep_axis}'), optionally tupled with {outer_axes},"
                f" with the topk dim replicated; got spec={idx_spec}."
            )
        arg_shardings = tuple(a.sharding for a in arg_infos)
        # token_counts / handle_mem inherit the input's leading axis (trailing dims auto-pad to None).
        leading_spec = PartitionSpec(idx_spec[0])
        tc_sharding = NamedSharding(mesh, leading_spec)
        hm_sharding = NamedSharding(mesh, leading_spec)

        def sharded_impl(topk_idx):
            return EpPreparePrimitive.impl(
                topk_idx, top_k, dispatch_output_per_expert_alignment, False
            )

        return mesh, sharded_impl, (tc_sharding, hm_sharding), arg_shardings

    @staticmethod
    def shardy_sharding_rule(*args):
        # Signature: (*static_args, mesh, value_types, result_types). Static args
        # for this primitive are (top_k, dispatch_alignment, is_outer).
        value_types = args[-2]
        topk_idx_rank = len(value_types[0].shape)
        in_axes = " ".join(f"L{i}" for i in range(topk_idx_rank - 1)) + " topk"
        return f"{in_axes} -> EPL nle, EPL hm"


register_primitive(EpPreparePrimitive)


# ── ep_dispatch ─────────────────────────────────────────────────────────────


class EpDispatchPrimitive(BasePrimitive):
    """FFI primitive for nvte_ep_dispatch (forward)."""

    name = "te_ep_dispatch_ffi"
    multiple_results = True
    impl_static_args = (4, 5, 6, 7)  # top_k, dispatch_output_per_expert_alignment,
    #                                  recv_capacity_per_rank, is_outer
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        handle_mem_aval,
        topk_idx_aval,
        tokens_aval,
        topk_weights_aval,
        *,
        top_k,
        dispatch_output_per_expert_alignment,
        recv_capacity_per_rank,
        is_outer,
    ):
        # is_outer=True: global leading dim = (dp*ep,) (or (ep,) with no DP);
        # False: per-shard = (1,).
        del topk_weights_aval, top_k, dispatch_output_per_expert_alignment, handle_mem_aval
        assert (
            len(tokens_aval.shape) >= 2
        ), f"tokens must be at least 2D [..., H], got shape {tokens_aval.shape}"
        recv_pr = recv_capacity_per_rank
        tok_dtype = dtypes.canonicalize_dtype(tokens_aval.dtype)
        hidden_dim = tokens_aval.shape[-1]
        leading = _ep_leading_dims(is_outer)
        recv_tokens_aval = jax.core.ShapedArray(leading + (recv_pr, hidden_dim), tok_dtype)
        recv_topk_weights_aval = jax.core.ShapedArray(leading + (recv_pr,), jnp.float32)
        # int32 with last dim doubled to keep the int64 byte count without JAX_ENABLE_X64.
        workspace_shape = topk_idx_aval.shape[:-1] + (topk_idx_aval.shape[-1] * 2,)
        workspace_aval = jax.core.ShapedArray(workspace_shape, jnp.int32)
        return (recv_tokens_aval, recv_topk_weights_aval, workspace_aval)

    @staticmethod
    def outer_abstract(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["is_outer"] = True
        avals = EpDispatchPrimitive.abstract(*args, **kwargs)  # pylint: disable=missing-kwoa
        return avals[:2]

    @staticmethod
    def lowering(
        ctx,
        handle_mem,
        topk_idx,
        tokens,
        topk_weights,
        *,
        top_k,
        dispatch_output_per_expert_alignment,
        recv_capacity_per_rank,
        is_outer,
    ):
        del recv_capacity_per_rank, is_outer
        return ffi.ffi_lowering(EpDispatchPrimitive.name)(
            ctx,
            handle_mem,
            topk_idx,
            tokens,
            topk_weights,
            top_k=int(top_k),
            dispatch_output_per_expert_alignment=int(dispatch_output_per_expert_alignment),
        )

    @staticmethod
    def impl(
        handle_mem,
        topk_idx,
        tokens,
        topk_weights,
        top_k,
        dispatch_output_per_expert_alignment,
        recv_capacity_per_rank,
        is_outer,
    ):
        assert EpDispatchPrimitive.inner_primitive is not None
        recv_tokens, recv_topk_weights, _workspace = EpDispatchPrimitive.inner_primitive.bind(
            handle_mem,
            topk_idx,
            tokens,
            topk_weights,
            top_k=top_k,
            dispatch_output_per_expert_alignment=dispatch_output_per_expert_alignment,
            recv_capacity_per_rank=recv_capacity_per_rank,
            is_outer=is_outer,
        )
        return recv_tokens, recv_topk_weights

    @staticmethod
    def batcher(
        batched_args,
        batch_dims,
        *,
        top_k,
        dispatch_output_per_expert_alignment,
        recv_capacity_per_rank,
        is_outer,
    ):
        raise NotImplementedError("EpDispatchPrimitive does not support vmap")

    @staticmethod
    def partition(
        top_k,
        dispatch_output_per_expert_alignment,
        recv_capacity_per_rank,
        is_outer,
        mesh,
        arg_infos,
        result_infos,
    ):
        del is_outer, result_infos
        tokens_spec = arg_infos[2].sharding.spec
        ok, ep_axis, outer_axes = _leading_axis_ok(tokens_spec)
        if not ok:
            raise NotImplementedError(
                "EpDispatch: tokens leading dim must include ep_resource"
                f" ('{ep_axis}'), optionally tupled with {outer_axes},"
                f" hidden dim replicated; got spec={tokens_spec}."
            )
        idx_spec = arg_infos[1].sharding.spec
        tw_spec = arg_infos[3].sharding.spec
        if idx_spec[0] != tokens_spec[0] or tw_spec[0] != tokens_spec[0]:
            raise NotImplementedError(
                "EpDispatch: topk_idx, tokens, topk_weights must share the leading"
                f" axis; got topk_idx={idx_spec}, tokens={tokens_spec}, topk_weights={tw_spec}."
            )
        # Recv outputs share the tokens leading-only spec (trailing dims auto-pad to None).
        leading_spec = PartitionSpec(tokens_spec[0])
        arg_shardings = tuple(a.sharding for a in arg_infos)
        out_shardings = (
            NamedSharding(mesh, leading_spec),
            NamedSharding(mesh, leading_spec),
        )

        def sharded_impl(handle_mem, topk_idx, tokens, topk_weights):
            return EpDispatchPrimitive.impl(
                handle_mem,
                topk_idx,
                tokens,
                topk_weights,
                top_k,
                dispatch_output_per_expert_alignment,
                recv_capacity_per_rank,
                False,
            )

        return mesh, sharded_impl, out_shardings, arg_shardings

    @staticmethod
    def shardy_sharding_rule(*args):
        # Signature: (*static_args, mesh, value_types, result_types). Static args
        # for this primitive are (top_k, dispatch_alignment, recv_capacity_per_rank, is_outer).
        value_types = args[-2]
        # Inputs: handle_mem, topk_idx, tokens, topk_weights.
        idx_rank = len(value_types[1].shape)
        tok_rank = len(value_types[2].shape)
        tw_rank = len(value_types[3].shape)
        idx_axes = " ".join(f"I{i}" for i in range(idx_rank - 1)) + " topk_in"
        tok_axes = " ".join(f"T{i}" for i in range(tok_rank - 1)) + " H"
        tw_axes = " ".join(f"W{i}" for i in range(tw_rank - 1)) + " topk"
        return f"EPL hm, {idx_axes}, {tok_axes}, {tw_axes} -> EPL recv_pr H, EPL recv_pr"


register_primitive(EpDispatchPrimitive)


# ── ep_combine ──────────────────────────────────────────────────────────────
# `expert_out` here is the post-weight buffer; ep.ep_combine applies the
# hadamard before calling.


def _normalize_leading_shape(s):
    return s if isinstance(s, tuple) else (int(s),)


def _prod(seq):
    p = 1
    for x in seq:
        p *= int(x)
    return p


def _leading_per_shard(out_leading_shape, leading_axis, mesh):
    """Per-shard leading shape: divide ``out_leading_shape[0]`` by the mesh factor on ``leading_axis``."""
    axes = leading_axis if isinstance(leading_axis, tuple) else (leading_axis,)
    factor = 1
    for a in axes:
        factor *= mesh.shape[a]
    assert out_leading_shape[0] % factor == 0, (
        f"leading dim {out_leading_shape[0]} not divisible by shard factor"
        f" {factor} on axes {axes}"
    )
    return (out_leading_shape[0] // factor,) + tuple(out_leading_shape[1:])


class EpCombinePrimitive(BasePrimitive):
    """FFI primitive for nvte_ep_combine (forward)."""

    name = "te_ep_combine_ffi"
    multiple_results = False
    impl_static_args = (2, 3, 4, 5)  # top_k, dispatch_output_per_expert_alignment,
    #                                   out_leading_shape, out_partition_spec
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        handle_mem_aval,
        expert_out_aval,
        *,
        top_k,
        dispatch_output_per_expert_alignment,
        out_leading_shape,
        out_partition_spec,
    ):
        del top_k, dispatch_output_per_expert_alignment, out_partition_spec, handle_mem_aval
        assert (
            len(expert_out_aval.shape) == 3
        ), f"expert_out must be 3D [num_procs, recv_pr, H], got shape {expert_out_aval.shape}"
        eo_dtype = dtypes.canonicalize_dtype(expert_out_aval.dtype)
        hidden_dim = expert_out_aval.shape[-1]
        out_shape = tuple(out_leading_shape) + (hidden_dim,)
        return jax.core.ShapedArray(out_shape, eo_dtype)

    @staticmethod
    def lowering(
        ctx,
        handle_mem,
        expert_out,
        *,
        top_k,
        dispatch_output_per_expert_alignment,
        out_leading_shape,
        out_partition_spec,
    ):
        del out_partition_spec
        return ffi.ffi_lowering(EpCombinePrimitive.name)(
            ctx,
            handle_mem,
            expert_out,
            top_k=int(top_k),
            dispatch_output_per_expert_alignment=int(dispatch_output_per_expert_alignment),
            num_local_tokens=_prod(out_leading_shape),
        )

    @staticmethod
    def impl(
        handle_mem,
        expert_out,
        top_k,
        dispatch_output_per_expert_alignment,
        out_leading_shape,
        out_partition_spec,
    ):
        assert EpCombinePrimitive.inner_primitive is not None
        return EpCombinePrimitive.inner_primitive.bind(
            handle_mem,
            expert_out,
            top_k=top_k,
            dispatch_output_per_expert_alignment=dispatch_output_per_expert_alignment,
            out_leading_shape=out_leading_shape,
            out_partition_spec=out_partition_spec,
        )

    @staticmethod
    def batcher(
        batched_args,
        batch_dims,
        *,
        top_k,
        dispatch_output_per_expert_alignment,
        out_leading_shape,
        out_partition_spec,
    ):
        raise NotImplementedError("EpCombinePrimitive does not support vmap")

    @staticmethod
    def partition(
        top_k,
        dispatch_output_per_expert_alignment,
        out_leading_shape,
        out_partition_spec,
        mesh,
        arg_infos,
        result_infos,
    ):
        del result_infos
        eo_spec = arg_infos[1].sharding.spec
        if not _ep_spec_ok(eo_spec, trailing_count=2):
            raise NotImplementedError(
                "EpCombine: expert_out must be sharded as PartitionSpec(ep_resource,"
                " None, None) (or ((dp, ep), None, None) when dp/fsdp is set)"
                f" over [num_procs, recv_pr, H]; got spec={eo_spec}."
            )
        per_shard_leading = _leading_per_shard(out_leading_shape, out_partition_spec[0], mesh)
        arg_shardings = tuple(a.sharding for a in arg_infos)
        out_sharding = NamedSharding(mesh, PartitionSpec(*out_partition_spec))

        def sharded_impl(handle_mem, expert_out):
            return EpCombinePrimitive.impl(
                handle_mem,
                expert_out,
                top_k,
                dispatch_output_per_expert_alignment,
                per_shard_leading,
                out_partition_spec,
            )

        return mesh, sharded_impl, out_sharding, arg_shardings

    @staticmethod
    def shardy_sharding_rule(*args):
        # Signature: (*static_args, mesh, value_types, result_types). Static args:
        # (top_k, dispatch_alignment, out_leading_shape, out_partition_spec).
        result_types = args[-1]
        out_rank = len(result_types[0].shape)
        out_axes = " ".join(f"O{i}" for i in range(out_rank - 1)) + " H"
        return f"EPL hm, EPL recv_pr H -> {out_axes}"


register_primitive(EpCombinePrimitive)


# ── ep_dispatch_bwd ─────────────────────────────────────────────────────────


class EpDispatchBwdPrimitive(BasePrimitive):
    """FFI primitive for the backward of nvte_ep_dispatch."""

    name = "te_ep_dispatch_bwd_ffi"
    multiple_results = True
    impl_static_args = (3, 4, 5, 6)  # top_k, dispatch_output_per_expert_alignment,
    #                                   out_leading_shape, out_partition_spec
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        handle_mem_aval,
        grad_aval,
        g_recv_topk_weights_aval,
        *,
        top_k,
        dispatch_output_per_expert_alignment,
        out_leading_shape,
        out_partition_spec,
    ):
        del dispatch_output_per_expert_alignment
        del g_recv_topk_weights_aval, out_partition_spec, handle_mem_aval
        assert (
            len(grad_aval.shape) == 3
        ), f"grad must be 3D [num_procs, recv_pr, H], got shape {grad_aval.shape}"
        g_dtype = dtypes.canonicalize_dtype(grad_aval.dtype)
        hidden_dim = grad_aval.shape[-1]
        result_aval = jax.core.ShapedArray(tuple(out_leading_shape) + (hidden_dim,), g_dtype)
        grad_topk_weights_aval = jax.core.ShapedArray(
            tuple(out_leading_shape) + (top_k,), jnp.float32
        )
        return result_aval, grad_topk_weights_aval

    @staticmethod
    def lowering(
        ctx,
        handle_mem,
        grad,
        g_recv_topk_weights,
        *,
        top_k,
        dispatch_output_per_expert_alignment,
        out_leading_shape,
        out_partition_spec,
    ):
        del out_partition_spec
        return ffi.ffi_lowering(EpDispatchBwdPrimitive.name)(
            ctx,
            handle_mem,
            grad,
            g_recv_topk_weights,
            top_k=int(top_k),
            dispatch_output_per_expert_alignment=int(dispatch_output_per_expert_alignment),
            num_local_tokens=_prod(out_leading_shape),
        )

    @staticmethod
    def impl(
        handle_mem,
        grad,
        g_recv_topk_weights,
        top_k,
        dispatch_output_per_expert_alignment,
        out_leading_shape,
        out_partition_spec,
    ):
        assert EpDispatchBwdPrimitive.inner_primitive is not None
        return EpDispatchBwdPrimitive.inner_primitive.bind(
            handle_mem,
            grad,
            g_recv_topk_weights,
            top_k=top_k,
            dispatch_output_per_expert_alignment=dispatch_output_per_expert_alignment,
            out_leading_shape=out_leading_shape,
            out_partition_spec=out_partition_spec,
        )

    @staticmethod
    def batcher(
        batched_args,
        batch_dims,
        *,
        top_k,
        dispatch_output_per_expert_alignment,
        out_leading_shape,
        out_partition_spec,
    ):
        raise NotImplementedError("EpDispatchBwdPrimitive does not support vmap")

    @staticmethod
    def partition(
        top_k,
        dispatch_output_per_expert_alignment,
        out_leading_shape,
        out_partition_spec,
        mesh,
        arg_infos,
        result_infos,
    ):
        del result_infos
        g_spec = arg_infos[1].sharding.spec
        if not _ep_spec_ok(g_spec, trailing_count=2):
            raise NotImplementedError(
                "EpDispatchBwd: grad must be sharded as PartitionSpec(ep_resource,"
                " None, None) (or ((dp, ep), None, None) when dp/fsdp is set)"
                f" over [num_procs, recv_pr, H]; got spec={g_spec}."
            )
        gw_spec = arg_infos[2].sharding.spec
        if not _ep_spec_ok(gw_spec, trailing_count=1):
            raise NotImplementedError(
                "EpDispatchBwd: g_recv_topk_weights must be sharded as"
                " PartitionSpec(ep_resource, None) (or ((dp, ep), None) when dp/fsdp is set)"
                f" over [num_procs, recv_pr]; got spec={gw_spec}."
            )
        if gw_spec[0] != g_spec[0]:
            raise NotImplementedError(
                "EpDispatchBwd: grad and g_recv_topk_weights must share the leading"
                f" axis; got grad={g_spec}, g_recv_topk_weights={gw_spec}."
            )
        per_shard_leading = _leading_per_shard(out_leading_shape, out_partition_spec[0], mesh)
        arg_shardings = tuple(a.sharding for a in arg_infos)
        out_sharding = NamedSharding(mesh, PartitionSpec(*out_partition_spec))
        out_shardings = [out_sharding, out_sharding]

        def sharded_impl(handle_mem, grad, g_recv_topk_weights):
            return EpDispatchBwdPrimitive.impl(
                handle_mem,
                grad,
                g_recv_topk_weights,
                top_k,
                dispatch_output_per_expert_alignment,
                per_shard_leading,
                out_partition_spec,
            )

        return mesh, sharded_impl, out_shardings, arg_shardings

    @staticmethod
    def shardy_sharding_rule(*args):
        # Signature: (*static_args, mesh, value_types, result_types). Result rank
        # follows out_leading_shape (static arg #2): rank = len(out_leading) + 1.
        result_types = args[-1]
        out_rank = len(result_types[0].shape)
        out_axes = " ".join(f"O{i}" for i in range(out_rank - 1))
        return f"EPL hm, EPL recv_pr H, EPL recv_pr -> {out_axes} H, {out_axes} k"


register_primitive(EpDispatchBwdPrimitive)


# ── ep_combine_bwd ──────────────────────────────────────────────────────────


class EpCombineBwdPrimitive(BasePrimitive):
    """FFI primitive for the backward of nvte_ep_combine."""

    name = "te_ep_combine_bwd_ffi"
    multiple_results = False
    impl_static_args = (2, 3, 4, 5)  # top_k, dispatch_output_per_expert_alignment,
    #                                   recv_capacity_per_rank, is_outer
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        handle_mem_aval,
        grad_aval,
        *,
        top_k,
        dispatch_output_per_expert_alignment,
        recv_capacity_per_rank,
        is_outer,
    ):
        # is_outer=True: global leading dim = (dp*ep,) (or (ep,) with no DP);
        # False: per-shard = (1,).
        del top_k, dispatch_output_per_expert_alignment, handle_mem_aval
        assert (
            len(grad_aval.shape) >= 2
        ), f"grad must be at least 2D [..., H], got shape {grad_aval.shape}"
        g_dtype = dtypes.canonicalize_dtype(grad_aval.dtype)
        hidden_dim = grad_aval.shape[-1]
        leading = _ep_leading_dims(is_outer)
        return jax.core.ShapedArray(leading + (recv_capacity_per_rank, hidden_dim), g_dtype)

    @staticmethod
    def outer_abstract(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["is_outer"] = True
        return EpCombineBwdPrimitive.abstract(*args, **kwargs)  # pylint: disable=missing-kwoa

    @staticmethod
    def lowering(
        ctx,
        handle_mem,
        grad,
        *,
        top_k,
        dispatch_output_per_expert_alignment,
        recv_capacity_per_rank,
        is_outer,
    ):
        del recv_capacity_per_rank, is_outer
        return ffi.ffi_lowering(EpCombineBwdPrimitive.name)(
            ctx,
            handle_mem,
            grad,
            top_k=int(top_k),
            dispatch_output_per_expert_alignment=int(dispatch_output_per_expert_alignment),
        )

    @staticmethod
    def impl(
        handle_mem,
        grad,
        top_k,
        dispatch_output_per_expert_alignment,
        recv_capacity_per_rank,
        is_outer,
    ):
        assert EpCombineBwdPrimitive.inner_primitive is not None
        return EpCombineBwdPrimitive.inner_primitive.bind(
            handle_mem,
            grad,
            top_k=top_k,
            dispatch_output_per_expert_alignment=dispatch_output_per_expert_alignment,
            recv_capacity_per_rank=recv_capacity_per_rank,
            is_outer=is_outer,
        )

    @staticmethod
    def batcher(
        batched_args,
        batch_dims,
        *,
        top_k,
        dispatch_output_per_expert_alignment,
        recv_capacity_per_rank,
        is_outer,
    ):
        raise NotImplementedError("EpCombineBwdPrimitive does not support vmap")

    @staticmethod
    def partition(
        top_k,
        dispatch_output_per_expert_alignment,
        recv_capacity_per_rank,
        is_outer,
        mesh,
        arg_infos,
        result_infos,
    ):
        del is_outer, result_infos
        arg_shardings = tuple(a.sharding for a in arg_infos)
        # EP-output leading (trailing dims auto-pad to None).
        out_sharding = NamedSharding(mesh, _ep_output_spec())

        def sharded_impl(handle_mem, grad):
            return EpCombineBwdPrimitive.impl(
                handle_mem,
                grad,
                top_k,
                dispatch_output_per_expert_alignment,
                recv_capacity_per_rank,
                False,
            )

        return mesh, sharded_impl, out_sharding, arg_shardings

    @staticmethod
    def shardy_sharding_rule(*args):
        # T axes are dynamic-rank based on the actual cotangent shape.
        value_types = args[-2]
        g_rank = len(value_types[1].shape)
        g_axes = " ".join(f"T{i}" for i in range(g_rank - 1)) + " H"
        return f"EPL hm, {g_axes} -> EPL recv_pr H"


register_primitive(EpCombineBwdPrimitive)


# ── Public-ish helpers (used by jax/ep.py) ──────────────────────────────────


def ep_prepare(cfg: EpLayerConfig, topk_idx):
    """Exchange routing metadata for ``cfg``; return ``(token_counts, handle_mem)``."""
    return EpPreparePrimitive.outer_primitive.bind(
        topk_idx,
        top_k=int(cfg.top_k),
        dispatch_output_per_expert_alignment=int(cfg.dispatch_output_per_expert_alignment),
        is_outer=True,
    )


def ep_dispatch_fwd(cfg: EpLayerConfig, handle_mem, topk_idx, tokens, topk_weights,
                    recv_capacity_per_rank):
    """Scatter tokens and weights to expert ranks; returns (recv_tokens, recv_topk_weights)."""
    return EpDispatchPrimitive.outer_primitive.bind(
        handle_mem,
        topk_idx,
        tokens,
        topk_weights,
        top_k=int(cfg.top_k),
        dispatch_output_per_expert_alignment=int(cfg.dispatch_output_per_expert_alignment),
        recv_capacity_per_rank=recv_capacity_per_rank,
        is_outer=True,
    )


def ep_combine_fwd(cfg: EpLayerConfig, handle_mem, expert_out, num_local_tokens,
                   out_partition_spec=None):
    """Gather expert outputs back to home ranks. expert_out is pre-weighted."""
    out_leading = _normalize_leading_shape(num_local_tokens)
    return EpCombinePrimitive.outer_primitive.bind(
        handle_mem,
        expert_out,
        top_k=int(cfg.top_k),
        dispatch_output_per_expert_alignment=int(cfg.dispatch_output_per_expert_alignment),
        out_leading_shape=out_leading,
        out_partition_spec=out_partition_spec,
    )


def ep_dispatch_bwd(
    cfg: EpLayerConfig, handle_mem, grad, g_recv_topk_weights, num_local_tokens,
    out_partition_spec=None,
):
    """Backward of dispatch; returns (grad_tokens, grad_topk_weights)."""
    out_leading = _normalize_leading_shape(num_local_tokens)
    return EpDispatchBwdPrimitive.outer_primitive.bind(
        handle_mem,
        grad,
        g_recv_topk_weights,
        top_k=int(cfg.top_k),
        dispatch_output_per_expert_alignment=int(cfg.dispatch_output_per_expert_alignment),
        out_leading_shape=out_leading,
        out_partition_spec=out_partition_spec,
    )


def ep_combine_bwd(cfg: EpLayerConfig, handle_mem, grad, recv_capacity_per_rank):
    """Backward of combine; returns grad_expert_out [num_procs, recv_capacity_per_rank, H]."""
    return EpCombineBwdPrimitive.outer_primitive.bind(
        handle_mem,
        grad,
        top_k=int(cfg.top_k),
        dispatch_output_per_expert_alignment=int(cfg.dispatch_output_per_expert_alignment),
        recv_capacity_per_rank=recv_capacity_per_rank,
        is_outer=True,
    )
