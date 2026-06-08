# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Mixture-of-Experts (MoE) layer for TransformerEngine JAX.

This module exposes :func:`moe`, a single fused MoE forward pass + bwd
built on top of TE's NCCL-backed Expert Parallelism primitives
(``tex.ep_dispatch`` / ``tex.ep_combine``). The block runs::

    gate  ->  topk  ->  ep_dispatch  ->  per-expert FFN (grouped GEMMs)
          ->  ep_combine  ->  output

under a single ``jax.custom_vjp`` so the routing, dispatch, FFN and
combine steps fuse cleanly under XLA without leaking intermediate
residuals into the user-facing autograd graph.

Sharding model
--------------
* Inbound activations are 3D ``[B, S, H]`` sharded
  ``((*data_parallelism_axes, ep_axis), None, None)``. The public
  :func:`moe` soft-repins this on entry and warns when a reshard is
  inserted.
* The EP primitives operate at global view (their custom_partitioning
  rules handle per-shard execution). The FFN GEMMs run per-shard inside
  a small ``shard_map`` whose ``in_specs`` and ``out_specs`` mirror the
  same ``((dp, ep), ...)`` layout.

Out-of-scope (for now)
----------------------
FP8 / MXFP8 quantizer sets are not yet wired on this path; turning
them on requires recipe-aware residual specs and ``ScaledTensor``
leaves across the ``shard_map`` boundary. ``aux_loss_coeff`` and
``expert_bias`` are supported (the former forces a per-step
all-gather over the routing-side logits, which lives off the critical
path and overlaps with the dispatch collective).
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union
import warnings

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from . import cpp_extensions as tex
from .quantize import (
    TensorUsage,
    noop_quantizer_set,
    with_sharding_constraint_by_logical_axes,
)
from .flax.module import _convert_to_activation_function
from .router import ScoreFunction, _validate_score_function
from .sharding import _get_mesh

__all__ = ["moe"]


# =============================================================================
# Process-level NCCL EP bootstrap (must run eagerly, outside jax.jit)
# =============================================================================
#
# ``tex.ep_bootstrap`` does a NCCL UID allgather over the JAX runtime, which
# cannot run from inside a jit-traced function. The caller must bootstrap
# eagerly once per process before any jitted MoE call, then record the
# bootstrap signature via ``record_ep_bootstrap_signature_for_moe``. The
# per-call check below verifies the recorded signature is wide enough for
# the current MoE invocation (smaller per-call usage is fine since the C++
# backend reserves worst-case buffers at bootstrap time).

_te_ep_bootstrap_signature: Optional[Tuple[int, int, int, int, int]] = None


def record_ep_bootstrap_signature_for_moe(
    num_experts: int,
    max_tokens_per_rank: int,
    recv_capacity_per_rank: int,
    hidden_dim: int,
    ep_size: int,
) -> None:
    """Record the params passed to ``ep_bootstrap`` so the per-call check
    in ``_moe_fwd_rule`` can verify compatibility. Call this once per
    process immediately after ``ep_bootstrap``.
    """
    global _te_ep_bootstrap_signature
    _te_ep_bootstrap_signature = (
        num_experts,
        max_tokens_per_rank,
        recv_capacity_per_rank,
        hidden_dim,
        ep_size,
    )


# Per-(top_k, alignment) EpHandle cache. ``tex.ep_make_handle`` mints a
# fresh handle_id from a singleton pool capped at NVTE_EP_HANDLE_CACHE_SIZE
# (default 8192); caching here keeps the pool steady across many jit traces
# of the same MoE block configuration.
_te_ep_handle_cache: Dict[Tuple[int, int], Any] = {}


def _get_or_make_ep_handle(top_k: int, dispatch_output_per_expert_alignment: int):
    key = (int(top_k), int(dispatch_output_per_expert_alignment))
    h = _te_ep_handle_cache.get(key)
    if h is None:
        h = tex.ep_make_handle(
            top_k=key[0],
            dispatch_output_per_expert_alignment=key[1],
        )
        _te_ep_handle_cache[key] = h
    return h


def _te_ep_assert_compatible_bootstrap(
    num_experts: int,
    max_tokens_per_rank: int,
    recv_capacity_per_rank: int,
    hidden_dim: int,
    ep_size: int,
) -> None:
    """Verify a prior eager ``ep_bootstrap`` is wide enough for this call."""
    if _te_ep_bootstrap_signature is None:
        raise RuntimeError(
            "TE EP was not bootstrapped. Call"
            " transformer_engine.jax.ep.ep_bootstrap(...) eagerly (outside"
            " any jax.jit) once per process, then"
            " transformer_engine.jax.moe.record_ep_bootstrap_signature_for_moe(...)"
            " with the same params, before invoking moe()."
        )
    b_num_experts, b_max_tpr, b_recv_pr, b_hidden, b_ep_size = _te_ep_bootstrap_signature
    if (
        num_experts != b_num_experts
        or hidden_dim != b_hidden
        or ep_size != b_ep_size
        or max_tokens_per_rank > b_max_tpr
        or recv_capacity_per_rank > b_recv_pr
    ):
        raise ValueError(
            "TE EP was already bootstrapped with signature"
            f" (num_experts={b_num_experts}, max_tokens_per_rank={b_max_tpr},"
            f" recv_capacity_per_rank={b_recv_pr}, hidden_dim={b_hidden},"
            f" ep_size={b_ep_size}); this moe() call needs"
            f" (num_experts={num_experts}, max_tokens_per_rank={max_tokens_per_rank},"
            f" recv_capacity_per_rank={recv_capacity_per_rank}, hidden_dim={hidden_dim},"
            f" ep_size={ep_size}). Re-bootstrap with wider params (or matching exact"
            " sizes) is required."
        )


# =============================================================================
# Residual container threaded fwd -> bwd
# =============================================================================


@jax.tree_util.register_pytree_node_class
@dataclass
class _Ctx:
    """Residuals carried from the fwd rule into the bwd rule."""

    x: jnp.ndarray
    gate_kernel: jnp.ndarray
    expert_bias: jnp.ndarray
    logits_2d: jnp.ndarray
    saved_scores: jnp.ndarray
    routing_map: jnp.ndarray
    handle: Any
    handle_mem: Any
    token_counts: jnp.ndarray
    recv_topk_weights: jnp.ndarray
    casted_sorted_x_lhs_trans: Any
    casted_wi_rhs_trans: Any
    gate_proj_out: jnp.ndarray
    up_proj_out: jnp.ndarray
    casted_intermediate_lhs_trans: Any
    casted_wo_rhs_trans: Any
    expert_outputs: jnp.ndarray
    local_group_sizes: jnp.ndarray
    # Aux-loss residuals; None when aux_loss_coeff == 0.
    aux_const_buf: Any = None
    aux_tokens_per_expert: Any = None
    aux_saved_scores: Any = None

    def tree_flatten(self):
        children = (
            self.x,
            self.gate_kernel,
            self.expert_bias,
            self.logits_2d,
            self.saved_scores,
            self.routing_map,
            self.handle_mem,
            self.token_counts,
            self.recv_topk_weights,
            self.casted_sorted_x_lhs_trans,
            self.casted_wi_rhs_trans,
            self.gate_proj_out,
            self.up_proj_out,
            self.casted_intermediate_lhs_trans,
            self.casted_wo_rhs_trans,
            self.expert_outputs,
            self.local_group_sizes,
            self.aux_const_buf,
            self.aux_tokens_per_expert,
            self.aux_saved_scores,
        )
        return children, self.handle

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            x,
            gate_kernel,
            expert_bias,
            logits_2d,
            saved_scores,
            routing_map,
            handle_mem,
            token_counts,
            recv_topk_weights,
            casted_sorted_x_lhs_trans,
            casted_wi_rhs_trans,
            gate_proj_out,
            up_proj_out,
            casted_intermediate_lhs_trans,
            casted_wo_rhs_trans,
            expert_outputs,
            local_group_sizes,
            aux_const_buf,
            aux_tokens_per_expert,
            aux_saved_scores,
        ) = children
        return cls(
            x=x,
            gate_kernel=gate_kernel,
            expert_bias=expert_bias,
            logits_2d=logits_2d,
            saved_scores=saved_scores,
            routing_map=routing_map,
            handle=aux_data,
            handle_mem=handle_mem,
            token_counts=token_counts,
            recv_topk_weights=recv_topk_weights,
            casted_sorted_x_lhs_trans=casted_sorted_x_lhs_trans,
            casted_wi_rhs_trans=casted_wi_rhs_trans,
            gate_proj_out=gate_proj_out,
            up_proj_out=up_proj_out,
            casted_intermediate_lhs_trans=casted_intermediate_lhs_trans,
            casted_wo_rhs_trans=casted_wo_rhs_trans,
            expert_outputs=expert_outputs,
            local_group_sizes=local_group_sizes,
            aux_const_buf=aux_const_buf,
            aux_tokens_per_expert=aux_tokens_per_expert,
            aux_saved_scores=aux_saved_scores,
        )


# =============================================================================
# Per-shard FFN body (runs inside shard_map)
# =============================================================================


def _ffn_fwd_per_shard(
    recv_tokens_local: jnp.ndarray,
    recv_topk_weights_local: jnp.ndarray,
    wi_0: jnp.ndarray,
    wi_1: jnp.ndarray,
    wo: jnp.ndarray,
    wi_0_bias: Optional[jnp.ndarray],
    wi_1_bias: Optional[jnp.ndarray],
    wo_bias: Optional[jnp.ndarray],
    *,
    num_local_experts: int,
    slots_per_expert: int,
    activation_type: str,
    apply_topk_weights_early: bool,
):
    """Per-shard FFN forward.

    Operates on the shard-local ``[1, recv_pr, H]`` slice that
    ``tex.ep_dispatch`` produces. Returns the expert outputs (shaped
    ``[1, recv_pr, H_out]`` so the surrounding ``shard_map`` reassembles
    them as ``[num_procs, recv_pr, H_out]``) plus the residuals consumed
    by the bwd.
    """
    hidden = recv_tokens_local.shape[-1]
    sorted_x = recv_tokens_local.reshape(-1, hidden)
    recv_w_flat = recv_topk_weights_local.reshape(-1)
    local_group_sizes = jnp.full((num_local_experts,), slots_per_expert, dtype=jnp.int32)

    wi_0 = wi_0.astype(sorted_x.dtype)
    wi_1 = wi_1.astype(sorted_x.dtype)
    wo = wo.astype(sorted_x.dtype)

    wi_combined = jnp.concatenate([wi_0, wi_1], axis=-1)
    wi_combined_bias = (
        jnp.concatenate([wi_0_bias, wi_1_bias], axis=-1) if wi_0_bias is not None else None
    )

    q_set = noop_quantizer_set
    casted_sorted_x = tex.grouped_quantize(sorted_x, q_set.x, local_group_sizes, flatten_axis=-1)
    casted_wi = tex.grouped_quantize(wi_combined, q_set.kernel, flatten_axis=-1)
    combined_out = tex.grouped_gemm(
        casted_sorted_x.get_tensor(usage=TensorUsage.LHS),
        casted_wi.get_tensor(usage=TensorUsage.RHS),
        contracting_dims=((1,), (1,)),
        bias=wi_combined_bias,
    )
    gate_proj_out, up_proj_out = jnp.split(combined_out, 2, axis=-1)
    casted_sorted_x_lhs_trans = casted_sorted_x.get_tensor(usage=TensorUsage.LHS_TRANS)
    casted_wi_rhs_trans = casted_wi.get_tensor(usage=TensorUsage.RHS_TRANS)

    act_fn = _convert_to_activation_function(activation_type)
    intermediate = act_fn(gate_proj_out) * up_proj_out

    if apply_topk_weights_early:
        # Fold the per-token combine weights into the FFN intermediate;
        # the downstream wo GEMM is linear so this is equivalent to the
        # late-weighting path, modulo elementwise op fusion gains. w_b is
        # cast to intermediate.dtype so the multiply doesn't promote
        # expert_outputs to f32 (NCCL EP combine hard-asserts bf16).
        w_b = recv_w_flat[:, None].astype(intermediate.dtype)
        mask_b = (recv_w_flat != 0).astype(intermediate.dtype)[:, None]
        intermediate = intermediate * w_b * mask_b

    casted_intermediate = tex.grouped_quantize(
        intermediate, q_set.x, local_group_sizes, flatten_axis=-1
    )
    casted_wo = tex.grouped_quantize(wo, q_set.kernel, flatten_axis=-1)
    expert_outputs = tex.grouped_gemm(
        casted_intermediate.get_tensor(usage=TensorUsage.LHS),
        casted_wo.get_tensor(usage=TensorUsage.RHS),
        contracting_dims=((1,), (1,)),
        bias=wo_bias,
    )
    casted_intermediate_lhs_trans = casted_intermediate.get_tensor(usage=TensorUsage.LHS_TRANS)
    casted_wo_rhs_trans = casted_wo.get_tensor(usage=TensorUsage.RHS_TRANS)

    expert_outputs_3d = expert_outputs.reshape(1, expert_outputs.shape[0], expert_outputs.shape[1])
    residuals = (
        casted_sorted_x_lhs_trans,
        casted_wi_rhs_trans,
        gate_proj_out,
        up_proj_out,
        casted_intermediate_lhs_trans,
        casted_wo_rhs_trans,
        local_group_sizes,
    )
    return expert_outputs_3d, residuals


def _ffn_bwd_per_shard(
    d_expert_outputs_local: jnp.ndarray,
    casted_sorted_x_lhs_trans,
    casted_wi_rhs_trans,
    gate_proj_out: jnp.ndarray,
    up_proj_out: jnp.ndarray,
    casted_intermediate_lhs_trans,
    casted_wo_rhs_trans,
    local_group_sizes: jnp.ndarray,
    recv_topk_weights_local: jnp.ndarray,
    *,
    activation_type: str,
    apply_topk_weights_early: bool,
    has_bias: bool,
):
    """Per-shard FFN backward.

    Mirrors :func:`_ffn_fwd_per_shard`. Returns
    ``(d_sorted_x [1, recv_pr, H], d_recv_w [1, recv_pr], d_wi_0, d_wi_1, d_wo,
    d_wi_0_bias, d_wi_1_bias, d_wo_bias)``.
    """
    d_eo_2d = d_expert_outputs_local.reshape(-1, d_expert_outputs_local.shape[-1])
    recv_w_flat = recv_topk_weights_local.reshape(-1)
    q_set = noop_quantizer_set

    # wo bwd
    casted_d_eo = tex.grouped_quantize(d_eo_2d, q_set.dgrad, local_group_sizes, flatten_axis=-1)
    d_intermediate = tex.grouped_gemm(
        casted_d_eo.get_tensor(usage=TensorUsage.LHS),
        casted_wo_rhs_trans,
        contracting_dims=((1,), (2,)),
    )
    d_wo = tex.grouped_gemm(
        casted_intermediate_lhs_trans,
        casted_d_eo.get_tensor(usage=TensorUsage.RHS),
        contracting_dims=((0,), (0,)),
    )
    d_wo_bias = tex.grouped_dbias(d_eo_2d, local_group_sizes) if has_bias else None

    act_fn = _convert_to_activation_function(activation_type)
    if apply_topk_weights_early:
        # intermediate' = intermediate * w * mask. Split the cotangent
        # across both factors before the activation bwd consumes it.
        # Cast w_b so the multiply stays in d_intermediate.dtype and
        # d_sorted_x (downstream into ep_dispatch_bwd) stays bf16.
        w_b = recv_w_flat[:, None].astype(d_intermediate.dtype)
        mask_b = (recv_w_flat != 0).astype(d_intermediate.dtype)[:, None]
        intermediate_unweighted = act_fn(gate_proj_out) * up_proj_out
        d_recv_w_from_intermediate = jnp.sum(
            d_intermediate * intermediate_unweighted * mask_b, axis=-1
        ).astype(recv_w_flat.dtype)
        d_intermediate = d_intermediate * w_b * mask_b
    else:
        d_recv_w_from_intermediate = jnp.zeros_like(recv_w_flat)

    # Activation bwd
    act_gate_proj_out, dact_gate_proj_pullback = jax.vjp(act_fn, gate_proj_out)
    d_up_proj_out = d_intermediate * act_gate_proj_out
    (d_gate_proj_out,) = dact_gate_proj_pullback(d_intermediate * up_proj_out)

    # wi bwd (fused gate/up)
    d_combined = jnp.concatenate([d_gate_proj_out, d_up_proj_out], axis=-1)
    casted_d_combined = tex.grouped_quantize(
        d_combined, q_set.dgrad, local_group_sizes, flatten_axis=-1
    )
    d_sorted_x = tex.grouped_gemm(
        casted_d_combined.get_tensor(usage=TensorUsage.LHS),
        casted_wi_rhs_trans,
        contracting_dims=((1,), (2,)),
    )
    d_wi_combined = tex.grouped_gemm(
        casted_sorted_x_lhs_trans,
        casted_d_combined.get_tensor(usage=TensorUsage.RHS),
        contracting_dims=((0,), (0,)),
    )
    d_wi_0, d_wi_1 = jnp.split(d_wi_combined, 2, axis=-1)
    if has_bias:
        d_wi_combined_bias = tex.grouped_dbias(d_combined, local_group_sizes)
        d_wi_0_bias, d_wi_1_bias = jnp.split(d_wi_combined_bias, 2, axis=-1)
    else:
        d_wi_0_bias = None
        d_wi_1_bias = None

    d_sorted_x_3d = d_sorted_x.reshape(1, d_sorted_x.shape[0], d_sorted_x.shape[1])
    d_recv_w_3d = d_recv_w_from_intermediate.reshape(1, -1)
    return (
        d_sorted_x_3d,
        d_recv_w_3d,
        d_wi_0,
        d_wi_1,
        d_wo,
        d_wi_0_bias,
        d_wi_1_bias,
        d_wo_bias,
    )


# =============================================================================
# Full fwd / bwd rules (custom_vjp halves)
# =============================================================================


def _moe_fwd_rule(
    x,
    gate_kernel,
    wi_0,
    wi_1,
    wo,
    wi_0_bias,
    wi_1_bias,
    wo_bias,
    expert_bias,
    num_experts,
    num_experts_per_tok,
    activation_type,
    score_function,
    use_pre_softmax,
    num_groups,
    group_topk,
    scaling_factor,
    aux_loss_coeff,
    ep_axis,
    data_parallelism_axes,
    input_axes,
    gate_kernel_axes,
    wi_kernel_axes,
    wo_kernel_axes,
    dtype,
    apply_topk_weights_early,
    align_size,
):
    """Forward: gate -> topk -> ep_dispatch -> shard_map(FFN) -> ep_combine.

    Returns ``(output, aux_loss)``. ``aux_loss`` is a zero scalar when
    ``aux_loss_coeff == 0``.
    """
    del gate_kernel_axes, wi_kernel_axes, wo_kernel_axes  # used in bwd only
    from jax.experimental.shard_map import shard_map

    x = with_sharding_constraint_by_logical_axes(x, input_axes)

    mesh = _get_mesh()
    if mesh is None or mesh.empty:
        raise ValueError("moe(...) requires an active jax.sharding.Mesh.")
    if ep_axis is None:
        raise ValueError("moe(...) requires ep_axis to be set (TE EP backend).")
    num_ep = mesh.shape[ep_axis]
    if num_experts % num_ep != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by EP size={num_ep}")
    num_local_experts = num_experts // num_ep

    dp_size = 1
    for ax in data_parallelism_axes:
        dp_size *= mesh.shape[ax]
    num_procs = num_ep * dp_size

    B, S, H = x.shape
    K = num_experts_per_tok
    if B % num_procs != 0:
        raise ValueError(f"batch={B} not divisible by ep*dp={num_procs}")

    # Per-rank receive capacity (dropless): every rank may receive all of one
    # replica's K-expanded tokens. ``slots_per_expert`` is rounded up to a
    # multiple of ``align_size`` (FP8 recipes typically need 128 here); the
    # rounded value is what we feed to ``tex.ep_prepare`` as the
    # ``dispatch_output_per_expert_alignment`` so each local expert's slot
    # block starts on the alignment boundary that grouped_gemm expects.
    natural_recv_pr = (B // dp_size) * S * K
    natural_spe = (natural_recv_pr + num_local_experts - 1) // num_local_experts
    # NCCL EP requires each expert-major output block to be at least
    # 128-token aligned. Keep larger caller-requested alignments, but do
    # not emit the smaller natural block size for tiny tests.
    effective_align = max(int(align_size), 128)
    slots_per_expert = ((natural_spe + effective_align - 1) // effective_align) * effective_align
    recv_pr = num_local_experts * slots_per_expert
    # Per-rank input token count: B/num_procs rows x S tokens. The bootstrap
    # uses this to size the dispatch send buffer; recv_pr above sizes the
    # per-rank receive buffer.
    max_tokens_per_rank = (B // num_procs) * S

    _te_ep_assert_compatible_bootstrap(
        num_experts=num_experts,
        max_tokens_per_rank=max_tokens_per_rank,
        recv_capacity_per_rank=recv_pr,
        hidden_dim=H,
        ep_size=num_ep,
    )

    if not data_parallelism_axes:
        batch_pspec_axis: Any = ep_axis
    else:
        # ep must be innermost: ep_bootstrap forms NCCL EP comms from
        # consecutive global ranks (dp_color = rank // ep_size), so the
        # comm only stays within one model replica under (outer_dp, ep).
        batch_pspec_axis = (*data_parallelism_axes, ep_axis)
    ep3_spec = P(batch_pspec_axis, None, None)
    ep2_spec = P(batch_pspec_axis, None)
    x = jax.lax.with_sharding_constraint(x, NamedSharding(mesh, ep3_spec))

    # ---------------- Gate (global view) ----------------
    gate_kernel_cast = gate_kernel.astype(x.dtype)
    gate_logits = jnp.einsum("bsh,he->bse", x, gate_kernel_cast)
    logits_2d = gate_logits.reshape(-1, num_experts)

    # ---------------- Routing (global view) ----------------
    # expert_bias is an empty (shape-(0,)) sentinel when the caller did
    # not enable it; the primitive treats that as "no bias".
    eb_arg = expert_bias if expert_bias.shape != (0,) else jnp.zeros((0,), dtype=jnp.float32)
    sparse_probs, routing_map, saved_scores = tex.fused_topk_with_score_function_fwd(
        logits_2d,
        topk=K,
        use_pre_softmax=use_pre_softmax,
        num_groups=-1 if num_groups is None else num_groups,
        group_topk=-1 if group_topk is None else group_topk,
        scaling_factor=scaling_factor,
        score_function=score_function,
        expert_bias=eb_arg,
        compute_aux_scores=False,
    )
    sparse_probs = sparse_probs.astype(dtype)

    # ---------------- Aux loss (global view, replicated) ----------------
    # ``fused_moe_aux_loss_fwd`` sums probs and tokens_per_expert across
    # all tokens, which is wrong when T is sharded. Force-replicate the
    # gate logits and recompute the routing map at global view so the
    # kernel sees a complete [T_global, E] tensor. The replication is a
    # single all-gather over (*dp, ep) and lives off the dispatch
    # critical path.
    if aux_loss_coeff > 0.0:
        global_logits_2d = jax.lax.with_sharding_constraint(
            logits_2d, NamedSharding(mesh, P())
        )
        _, global_routing_map, _ = tex.fused_topk_with_score_function_fwd(
            global_logits_2d,
            topk=K,
            use_pre_softmax=use_pre_softmax,
            num_groups=-1 if num_groups is None else num_groups,
            group_topk=-1 if group_topk is None else group_topk,
            scaling_factor=scaling_factor,
            score_function=score_function,
            expert_bias=eb_arg,
            compute_aux_scores=False,
        )
        aux_tokens_per_expert = jnp.sum(global_routing_map.astype(jnp.int32), axis=0)
        # compute_aux_scores=True takes a separate kernel path: clean
        # per-expert softmax, no grouping / bias / scaling.
        aux_probs, _aux_rm, aux_saved_scores = tex.fused_topk_with_score_function_fwd(
            global_logits_2d.astype(jnp.float32),
            topk=K,
            use_pre_softmax=False,
            num_groups=-1,
            group_topk=-1,
            scaling_factor=1.0,
            score_function=score_function,
            expert_bias=jnp.zeros((0,), dtype=jnp.float32),
            compute_aux_scores=True,
        )
        aux_loss, aux_const_buf = tex.fused_moe_aux_loss_fwd(
            aux_probs.astype(jnp.float32),
            aux_tokens_per_expert.astype(jnp.int32),
            topk=K,
            coeff=aux_loss_coeff,
        )
        aux_loss = aux_loss.astype(dtype)
    else:
        aux_loss = jnp.zeros((), dtype=dtype)
        aux_const_buf = None
        aux_tokens_per_expert = None
        aux_saved_scores = None

    # ---------------- Routing -> (topk_idx, topk_w) at 3D ----------------
    # argsort on a bool tensor places True last (False=0 < True=1), so the
    # last K indices are the selected expert IDs.
    selected_experts = jnp.argsort(routing_map, axis=-1)[..., -K:]
    routing_weights = jnp.take_along_axis(sparse_probs, selected_experts, axis=-1)
    topk_idx_3d = selected_experts.reshape(B, S, K).astype(jnp.int32)
    topk_w_3d = routing_weights.reshape(B, S, K).astype(jnp.float32)
    # tex.ep_prepare/dispatch's partition only folds ep_axis into a replicated
    # leading dim, not the outer dp/fsdp axes, so a replicated topk_idx makes
    # each rank see B/ep rows (not B/num_procs) and overrun the bootstrap-sized
    # send buffer. Pin both routing tensors to the (outer, ep) leading sharding
    # so per-rank token counts match max_tokens_per_rank.
    topk_idx_3d = jax.lax.with_sharding_constraint(
        topk_idx_3d, NamedSharding(mesh, ep3_spec)
    )
    topk_w_3d = jax.lax.with_sharding_constraint(
        topk_w_3d, NamedSharding(mesh, ep3_spec)
    )

    # ---------------- TE EP dispatch (global view) ----------------
    handle = _get_or_make_ep_handle(
        top_k=K, dispatch_output_per_expert_alignment=slots_per_expert
    )
    token_counts, handle_mem = tex.ep_prepare(topk_idx_3d, handle)
    recv_tokens, recv_topk_weights = tex.ep_dispatch_fwd(
        handle, handle_mem, topk_idx_3d, x, topk_w_3d, recv_pr
    )
    recv_tokens = jax.lax.with_sharding_constraint(recv_tokens, NamedSharding(mesh, ep3_spec))
    recv_topk_weights = jax.lax.with_sharding_constraint(
        recv_topk_weights, NamedSharding(mesh, ep2_spec)
    )

    # ---------------- FFN (per-shard via shard_map) ----------------
    has_bias = wi_0_bias is not None
    kernel_spec = P(ep_axis, None, None)
    bias_spec = P(ep_axis, None) if has_bias else None
    ffn_in_specs = (ep3_spec, ep2_spec, kernel_spec, kernel_spec, kernel_spec)
    ffn_in_args = [recv_tokens, recv_topk_weights, wi_0, wi_1, wo]
    if has_bias:
        ffn_in_specs = ffn_in_specs + (bias_spec, bias_spec, bias_spec)
        ffn_in_args.extend([wi_0_bias, wi_1_bias, wo_bias])

    # FFN residuals live entirely on the local ep rank, so the leading
    # "experts" / "rows" dims map to P() (already shard-local).
    residuals_spec = (
        P(),                    # casted_sorted_x_lhs_trans
        P(ep_axis, None, None), # casted_wi_rhs_trans
        P(),                    # gate_proj_out
        P(),                    # up_proj_out
        P(),                    # casted_intermediate_lhs_trans
        P(ep_axis, None, None), # casted_wo_rhs_trans
        P(),                    # local_group_sizes
    )
    out_specs = (ep3_spec, residuals_spec)

    def _body(*args):
        if has_bias:
            (r_tok, r_w, w0, w1, w_o, w0b, w1b, wob) = args
        else:
            (r_tok, r_w, w0, w1, w_o) = args
            w0b = w1b = wob = None
        return _ffn_fwd_per_shard(
            r_tok,
            r_w,
            w0,
            w1,
            w_o,
            w0b,
            w1b,
            wob,
            num_local_experts=num_local_experts,
            slots_per_expert=slots_per_expert,
            activation_type=activation_type,
            apply_topk_weights_early=apply_topk_weights_early,
        )

    expert_outputs, ffn_residuals = shard_map(
        _body,
        mesh=mesh,
        in_specs=ffn_in_specs,
        out_specs=out_specs,
        check_rep=False,
    )(*ffn_in_args)
    expert_outputs = jax.lax.with_sharding_constraint(
        expert_outputs, NamedSharding(mesh, ep3_spec)
    )

    # ---------------- TE EP combine (global view) ----------------
    out_partition_spec = (batch_pspec_axis, None, None)
    if apply_topk_weights_early:
        # expert_outputs is already weighted upstream.
        output = tex.ep_combine_fwd(
            handle,
            handle_mem,
            expert_outputs,
            num_local_tokens=(B, S),
            out_partition_spec=out_partition_spec,
        )
    else:
        w = recv_topk_weights[..., None].astype(expert_outputs.dtype)
        mask = (recv_topk_weights != 0).astype(expert_outputs.dtype)[..., None]
        weighted = expert_outputs * w * mask
        output = tex.ep_combine_fwd(
            handle,
            handle_mem,
            weighted,
            num_local_tokens=(B, S),
            out_partition_spec=out_partition_spec,
        )

    (
        casted_sorted_x_lhs_trans,
        casted_wi_rhs_trans,
        gate_proj_out,
        up_proj_out,
        casted_intermediate_lhs_trans,
        casted_wo_rhs_trans,
        local_group_sizes,
    ) = ffn_residuals

    ctx = _Ctx(
        x=x,
        gate_kernel=gate_kernel,
        expert_bias=expert_bias,
        logits_2d=logits_2d,
        saved_scores=saved_scores,
        routing_map=routing_map,
        handle=handle,
        handle_mem=handle_mem,
        token_counts=token_counts,
        recv_topk_weights=recv_topk_weights,
        casted_sorted_x_lhs_trans=casted_sorted_x_lhs_trans,
        casted_wi_rhs_trans=casted_wi_rhs_trans,
        gate_proj_out=gate_proj_out,
        up_proj_out=up_proj_out,
        casted_intermediate_lhs_trans=casted_intermediate_lhs_trans,
        casted_wo_rhs_trans=casted_wo_rhs_trans,
        expert_outputs=expert_outputs,
        local_group_sizes=local_group_sizes,
        aux_const_buf=aux_const_buf,
        aux_tokens_per_expert=aux_tokens_per_expert,
        aux_saved_scores=aux_saved_scores,
    )
    static = {
        "has_bias": has_bias,
        "x_shape": x.shape,
        "recv_pr": recv_pr,
    }
    return (output, aux_loss), (ctx, static)


def _moe_bwd_rule(
    num_experts,
    num_experts_per_tok,
    activation_type,
    score_function,
    use_pre_softmax,
    num_groups,
    group_topk,
    scaling_factor,
    aux_loss_coeff,
    ep_axis,
    data_parallelism_axes,
    input_axes,
    gate_kernel_axes,
    wi_kernel_axes,
    wo_kernel_axes,
    dtype,
    apply_topk_weights_early,
    align_size,
    residuals,
    cotangents,
):
    """Backward mirror of :func:`_moe_fwd_rule`."""
    del num_groups, group_topk, dtype, align_size  # captured in residuals / unused in bwd
    from jax.experimental.shard_map import shard_map

    d_output, d_aux_loss = cotangents

    ctx, static = residuals
    has_bias = static["has_bias"]
    x_shape = static["x_shape"]
    recv_pr = static["recv_pr"]

    mesh = _get_mesh()
    if mesh is None or mesh.empty:
        raise ValueError("moe(...) requires an active jax.sharding.Mesh.")
    num_ep = mesh.shape[ep_axis]
    dp_size = 1
    for ax in data_parallelism_axes:
        dp_size *= mesh.shape[ax]

    B, S, _ = x_shape
    K = num_experts_per_tok
    if not data_parallelism_axes:
        batch_pspec_axis: Any = ep_axis
    else:
        batch_pspec_axis = (*data_parallelism_axes, ep_axis)
    ep3_spec = P(batch_pspec_axis, None, None)
    ep2_spec = P(batch_pspec_axis, None)
    out_partition_spec = (batch_pspec_axis, None, None)

    # ---------------- Combine bwd (global view) ----------------
    d_output = jax.lax.with_sharding_constraint(d_output, NamedSharding(mesh, ep3_spec))
    grad_pre_combine = tex.ep_combine_bwd(ctx.handle, ctx.handle_mem, d_output, recv_pr)
    grad_pre_combine = jax.lax.with_sharding_constraint(
        grad_pre_combine, NamedSharding(mesh, ep3_spec)
    )

    if apply_topk_weights_early:
        # combine_fwd consumed already-weighted expert_outputs; the recv_w
        # cotangent flows through the early-weighting step inside the FFN bwd.
        d_expert_outputs = grad_pre_combine
        d_recv_w_from_combine = jnp.zeros_like(ctx.recv_topk_weights)
    else:
        # combine_fwd consumed weighted = expert_out * w * mask;
        # split the cotangent across both factors. w is cast to
        # grad_pre_combine.dtype so the multiply stays bf16 and
        # d_sorted_x (downstream into ep_dispatch_bwd) stays bf16.
        w = ctx.recv_topk_weights[..., None].astype(grad_pre_combine.dtype)
        mask = (ctx.recv_topk_weights != 0).astype(grad_pre_combine.dtype)[..., None]
        d_expert_outputs = grad_pre_combine * w * mask
        d_recv_w_from_combine = (grad_pre_combine * ctx.expert_outputs * mask).sum(axis=-1)
        d_recv_w_from_combine = d_recv_w_from_combine.astype(ctx.recv_topk_weights.dtype)

    # ---------------- FFN bwd (per-shard via shard_map) ----------------
    kernel_spec = P(ep_axis, None, None)
    bias_spec = P(ep_axis, None) if has_bias else None

    bwd_in_specs = (
        ep3_spec,                # d_expert_outputs
        P(),                     # casted_sorted_x_lhs_trans
        P(ep_axis, None, None),  # casted_wi_rhs_trans
        P(),                     # gate_proj_out
        P(),                     # up_proj_out
        P(),                     # casted_intermediate_lhs_trans
        P(ep_axis, None, None),  # casted_wo_rhs_trans
        P(),                     # local_group_sizes
        ep2_spec,                # recv_topk_weights
    )
    bwd_in_args = [
        d_expert_outputs,
        ctx.casted_sorted_x_lhs_trans,
        ctx.casted_wi_rhs_trans,
        ctx.gate_proj_out,
        ctx.up_proj_out,
        ctx.casted_intermediate_lhs_trans,
        ctx.casted_wo_rhs_trans,
        ctx.local_group_sizes,
        ctx.recv_topk_weights,
    ]
    bwd_out_specs = (
        ep3_spec,                            # d_sorted_x
        ep2_spec,                            # d_recv_w_from_intermediate
        kernel_spec,                         # d_wi_0
        kernel_spec,                         # d_wi_1
        kernel_spec,                         # d_wo
        bias_spec if has_bias else None,     # d_wi_0_bias
        bias_spec if has_bias else None,     # d_wi_1_bias
        bias_spec if has_bias else None,     # d_wo_bias
    )

    def _bwd_body(*args):
        (
            d_sorted_x_3d,
            d_recv_w_3d,
            d_wi_0,
            d_wi_1,
            d_wo,
            d_wi_0_bias,
            d_wi_1_bias,
            d_wo_bias,
        ) = _ffn_bwd_per_shard(
            *args,
            activation_type=activation_type,
            apply_topk_weights_early=apply_topk_weights_early,
            has_bias=has_bias,
        )
        # Weight grads accumulate per-DP-shard inside the body; psum across
        # DP axes so each replica sees the full sum (matches out_specs
        # P(ep_axis, ...) which is DP-replicated).
        if data_parallelism_axes:
            dp = tuple(data_parallelism_axes)
            d_wi_0 = jax.lax.psum(d_wi_0, axis_name=dp)
            d_wi_1 = jax.lax.psum(d_wi_1, axis_name=dp)
            d_wo = jax.lax.psum(d_wo, axis_name=dp)
            if has_bias:
                d_wi_0_bias = jax.lax.psum(d_wi_0_bias, axis_name=dp)
                d_wi_1_bias = jax.lax.psum(d_wi_1_bias, axis_name=dp)
                d_wo_bias = jax.lax.psum(d_wo_bias, axis_name=dp)
        return (
            d_sorted_x_3d,
            d_recv_w_3d,
            d_wi_0,
            d_wi_1,
            d_wo,
            d_wi_0_bias,
            d_wi_1_bias,
            d_wo_bias,
        )

    (
        d_sorted_x,
        d_recv_w_from_intermediate,
        d_wi_0,
        d_wi_1,
        d_wo,
        d_wi_0_bias,
        d_wi_1_bias,
        d_wo_bias,
    ) = shard_map(
        _bwd_body,
        mesh=mesh,
        in_specs=bwd_in_specs,
        out_specs=bwd_out_specs,
        check_rep=False,
    )(*bwd_in_args)

    d_recv_w_total = d_recv_w_from_combine + d_recv_w_from_intermediate

    # ---------------- Dispatch bwd (global view) ----------------
    d_sorted_x = jax.lax.with_sharding_constraint(d_sorted_x, NamedSharding(mesh, ep3_spec))
    d_recv_w_total = jax.lax.with_sharding_constraint(
        d_recv_w_total, NamedSharding(mesh, ep2_spec)
    )
    d_x_from_dispatch, d_topk_w = tex.ep_dispatch_bwd(
        ctx.handle,
        ctx.handle_mem,
        d_sorted_x,
        d_recv_w_total,
        top_k=K,
        num_local_tokens=(B, S),
        out_partition_spec=out_partition_spec,
    )

    # ---------------- Routing bwd (global view) ----------------
    # The cotangent on routing_weights is a sparse scatter into sparse_probs
    # at the selected_experts indices.
    selected_experts = jnp.argsort(ctx.routing_map, axis=-1)[..., -K:]
    d_topk_w_flat = d_topk_w.reshape(-1, K)
    d_sparse_probs = jnp.zeros(ctx.routing_map.shape, dtype=d_topk_w_flat.dtype)
    d_sparse_probs = d_sparse_probs.at[
        jnp.arange(ctx.routing_map.shape[0])[:, None], selected_experts
    ].set(d_topk_w_flat)

    d_logits_2d = tex.fused_topk_with_score_function_bwd(
        ctx.routing_map,
        ctx.saved_scores,
        d_sparse_probs.astype(ctx.saved_scores.dtype),
        topk=K,
        use_pre_softmax=use_pre_softmax,
        scaling_factor=scaling_factor,
        score_function=score_function,
        compute_aux_scores=False,
    )

    # ---------------- Aux loss bwd (global view, replicated) ----------------
    # Reverse the fwd's all-gather/aux pipeline: aux_loss_bwd produces
    # d_aux_probs, then topk_bwd(compute_aux_scores=True) produces the
    # extra d_logits contribution. The replicated tensor adds into the
    # T-sharded routing-side d_logits via JAX's normal broadcast.
    if aux_loss_coeff > 0.0:
        T_global = ctx.logits_2d.shape[0]
        d_aux_loss_scalar = d_aux_loss.reshape(()).astype(jnp.float32)
        d_aux_probs = tex.fused_moe_aux_loss_bwd(
            ctx.aux_const_buf,
            ctx.aux_tokens_per_expert.astype(jnp.int32),
            d_aux_loss_scalar,
            num_tokens=int(T_global),
        )
        # routing_map is ignored by the kernel when compute_aux_scores=True,
        # so pass a zero placeholder of the right shape/dtype.
        zero_routing_map = jnp.zeros(
            ctx.aux_saved_scores.shape, dtype=ctx.routing_map.dtype
        )
        d_logits_aux = tex.fused_topk_with_score_function_bwd(
            zero_routing_map,
            ctx.aux_saved_scores,
            d_aux_probs.astype(ctx.aux_saved_scores.dtype),
            topk=K,
            use_pre_softmax=False,
            scaling_factor=1.0,
            score_function=score_function,
            compute_aux_scores=True,
        )
        d_logits_2d = d_logits_2d + d_logits_aux.astype(d_logits_2d.dtype)

    # ---------------- Gate bwd (global view) ----------------
    d_gate_logits = d_logits_2d.reshape(B, S, num_experts)
    gate_kernel_cast = ctx.gate_kernel.astype(ctx.x.dtype)
    d_x_from_gate = jnp.einsum("bse,he->bsh", d_gate_logits, gate_kernel_cast)
    d_gate_kernel = jnp.einsum("bsh,bse->he", ctx.x, d_gate_logits).astype(ctx.gate_kernel.dtype)
    d_x = d_x_from_gate + d_x_from_dispatch

    # Pin output grads to the declared logical axes so downstream
    # optimizers see consistent shardings.
    d_x = with_sharding_constraint_by_logical_axes(d_x, input_axes)
    d_gate_kernel = with_sharding_constraint_by_logical_axes(d_gate_kernel, gate_kernel_axes)
    d_wi_0 = with_sharding_constraint_by_logical_axes(d_wi_0, wi_kernel_axes)
    d_wi_1 = with_sharding_constraint_by_logical_axes(d_wi_1, wi_kernel_axes)
    d_wo = with_sharding_constraint_by_logical_axes(d_wo, wo_kernel_axes)

    # expert_bias has no learnable bwd path through fused_topk: the
    # primitive's bwd returns None for the bias slot. Match that with a
    # zero cotangent of the right shape so custom_vjp's arity check
    # passes.
    d_expert_bias = jnp.zeros_like(ctx.expert_bias)

    return (
        d_x,
        d_gate_kernel,
        d_wi_0,
        d_wi_1,
        d_wo,
        d_wi_0_bias if has_bias else None,
        d_wi_1_bias if has_bias else None,
        d_wo_bias if has_bias else None,
        d_expert_bias,
    )


# =============================================================================
# custom_vjp + public entry
# =============================================================================


@partial(jax.custom_vjp, nondiff_argnums=tuple(range(9, 27)))
def _moe(
    x,
    gate_kernel,
    wi_0,
    wi_1,
    wo,
    wi_0_bias,
    wi_1_bias,
    wo_bias,
    expert_bias,
    num_experts,
    num_experts_per_tok,
    activation_type,
    score_function,
    use_pre_softmax,
    num_groups,
    group_topk,
    scaling_factor,
    aux_loss_coeff,
    ep_axis,
    data_parallelism_axes,
    input_axes,
    gate_kernel_axes,
    wi_kernel_axes,
    wo_kernel_axes,
    dtype,
    apply_topk_weights_early,
    align_size,
):
    primal, _ = _moe_fwd_rule(
        x,
        gate_kernel,
        wi_0,
        wi_1,
        wo,
        wi_0_bias,
        wi_1_bias,
        wo_bias,
        expert_bias,
        num_experts,
        num_experts_per_tok,
        activation_type,
        score_function,
        use_pre_softmax,
        num_groups,
        group_topk,
        scaling_factor,
        aux_loss_coeff,
        ep_axis,
        data_parallelism_axes,
        input_axes,
        gate_kernel_axes,
        wi_kernel_axes,
        wo_kernel_axes,
        dtype,
        apply_topk_weights_early,
        align_size,
    )
    return primal


_moe.defvjp(_moe_fwd_rule, _moe_bwd_rule)


def moe(
    x: jnp.ndarray,
    gate_kernel: jnp.ndarray,
    wi_0: jnp.ndarray,
    wi_1: jnp.ndarray,
    wo: jnp.ndarray,
    wi_0_bias: Optional[jnp.ndarray] = None,
    wi_1_bias: Optional[jnp.ndarray] = None,
    wo_bias: Optional[jnp.ndarray] = None,
    expert_bias: Optional[jnp.ndarray] = None,
    *,
    num_experts: int,
    num_experts_per_tok: int,
    activation_type: str = "silu",
    score_function: Union[str, ScoreFunction] = "softmax",
    use_pre_softmax: bool = False,
    num_groups: Optional[int] = None,
    group_topk: Optional[int] = None,
    scaling_factor: float = 1.0,
    aux_loss_coeff: float = 0.0,
    apply_topk_weights_early: bool = False,
    align_size: int = 0,
    ep_axis: str,
    data_parallelism_axes: Tuple[str, ...] = (),
    input_axes: Tuple[Optional[str], ...] = (),
    gate_kernel_axes: Tuple[Optional[str], ...] = (),
    wi_kernel_axes: Tuple[Optional[str], ...] = ("exp", "embed", "mlp"),
    wo_kernel_axes: Tuple[Optional[str], ...] = ("exp", "mlp", "embed"),
    dtype: jnp.dtype = jnp.float32,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Run a full MoE block under a single fused custom_vjp on the TE EP path.

    Returns ``(output, aux_loss)``. ``aux_loss`` is ``None`` when
    ``aux_loss_coeff == 0`` and a 0-d scalar otherwise.

    Parameters
    ----------
    expert_bias : Optional[jnp.ndarray]
        ``[num_experts]`` learnable router bias added before the top-k
        when ``score_function='sigmoid'``. Pass ``None`` to disable.
        The bias has no gradient through the top-k primitive itself (it
        only steers expert selection); a zero cotangent is returned for
        it.
    aux_loss_coeff : float
        Per-step expert-load-balance loss coefficient. ``0.0`` (default)
        disables the aux loss entirely. When non-zero, an extra
        all-gather over the routing-side logits is inserted so the
        ``fused_moe_aux_loss`` kernel sees a global ``[T_global, E]``
        view; this lives off the dispatch critical path.
    align_size : int
        Minimum per-expert slot alignment passed to ``tex.ep_prepare``
        as ``dispatch_output_per_expert_alignment``. ``0`` (default)
        means use the natural slot count
        ``ceil((B/dp)*S*K / num_local_experts)``. Any positive value
        rounds that count up to the nearest multiple, growing the
        per-rank receive buffer accordingly. Set to ``128`` for FP8
        recipes that require 128-aligned grouped-GEMM tiles.

    See module docstring for the rest of the parameter semantics and the
    surrounding design rationale.
    """
    score_function = _validate_score_function(score_function)

    # Enforce ((outer_dp..., ep), None, None) on inbound activations. The
    # EP comm groups consecutive global ranks (dp_color = rank // ep_size),
    # so ep MUST be innermost in the partition spec. Soft re-pin: free if
    # upstream already matches, single reshard otherwise.
    mesh = _get_mesh()
    if mesh is None or mesh.empty:
        raise ValueError("moe(...) requires an active jax.sharding.Mesh.")
    expected_leading: Any = (
        (*data_parallelism_axes, ep_axis) if data_parallelism_axes else ep_axis
    )
    expected_spec = P(expected_leading, None, None)
    actual_spec = getattr(getattr(x, "sharding", None), "spec", None)
    if actual_spec is not None and tuple(actual_spec) != tuple(expected_spec):
        warnings.warn(
            f"moe(...): inbound x sharding {actual_spec} does not match expected "
            f"{expected_spec}; inserting a reshard. Apply "
            "jax.lax.with_sharding_constraint upstream to avoid this overhead.",
            UserWarning,
            stacklevel=2,
        )
    x = jax.lax.with_sharding_constraint(x, NamedSharding(mesh, expected_spec))

    # custom_vjp can't trace through None args; lower expert_bias to an
    # empty shape-(0,) tensor that fused_topk_with_score_function treats
    # as "no bias".
    if expert_bias is None:
        expert_bias_arg = jnp.zeros((0,), dtype=jnp.float32)
    else:
        expert_bias_arg = expert_bias

    output, aux_loss = _moe(
        x,
        gate_kernel,
        wi_0,
        wi_1,
        wo,
        wi_0_bias,
        wi_1_bias,
        wo_bias,
        expert_bias_arg,
        num_experts,
        num_experts_per_tok,
        activation_type,
        score_function,
        use_pre_softmax,
        num_groups,
        group_topk,
        scaling_factor,
        float(aux_loss_coeff),
        ep_axis,
        data_parallelism_axes,
        input_axes,
        gate_kernel_axes,
        wi_kernel_axes,
        wo_kernel_axes,
        dtype,
        apply_topk_weights_early,
        align_size,
    )
    if aux_loss_coeff <= 0.0:
        aux_loss = None
    return output, aux_loss
