# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Functional Mixture-of-Experts (MoE) entry point with a single fused VJP.

This module exposes :func:`moe`, the framework-agnostic flat function that
implements an entire MoE block (gate -> top-k routing -> token dispatch ->
per-expert FFN -> token combine, plus optional expert parallelism via a
shard_map / ragged_all_to_all collective) under a *single*
``jax.custom_vjp``. It is the moral analog of
:func:`transformer_engine.jax.layernorm_mlp.layernorm_mlp` for MoE: one
custom_vjp boundary covers the whole block so future fusions (FP8 over the
EP wire, fused ``ragged_all_to_all + grouped_gemm``, gate+route+dispatch
fusion) can land without re-architecting the call site.

Design rationale
----------------

The earlier MoE block (:class:`transformer_engine.jax.flax.moe._MoEBlock`)
composed many narrower custom_vjps -- one per :func:`grouped_dense`, one
per :func:`token_dispatch`, etc. Every nested custom_vjp is a place where
a quantized :class:`ScaledTensor` cannot survive (JAX requires custom_vjp
inputs / outputs to be plain ``jnp.ndarray`` ish pytrees). To enable
end-to-end FP8 flow -- in particular FP8 carried over the EP
ragged_all_to_all -- the dispatch's quantize, the a2a, the per-expert
FFN, the inverse a2a, and the combine all have to live inside the same
VJP. This file collapses them into one.

Implementation conventions
--------------------------

* No nested ``custom_vjp``. Every primitive's ``_fwd`` and ``_bwd`` is
  called directly (e.g. :func:`tex.fused_topk_with_score_function_fwd` /
  ``_bwd``, :func:`unpermute_with_mask_map`,
  :func:`unpermute_bwd_with_merging_probs`,
  :func:`sort_chunks_by_map(is_forward=False)`,
  forward + reverse :func:`jax.lax.ragged_all_to_all`) so the outer
  ``_moe_bwd_rule`` controls the bwd graph end-to-end without invoking
  ``jax.vjp`` for re-linearization.
* The fwd/bwd context (``ctx``) is a plain ``dict`` whose keys depend on
  the static configuration (permutation backend, EP active or not,
  presence of biases, aux loss enabled). The ``_moe_fwd_rule`` builds a
  matching ``ctx_specs`` dict in lockstep when opening the EP shard_map
  so ``out_specs`` structurally matches the body's return.
* :func:`_dispatch` is the helper that wraps
  ``permute -> a2a -> local_permute`` (forward); :func:`_combine` is its
  inverse. Their ``_bwd`` siblings drive the inverse collectives in the
  bwd rule. None of these helpers form a custom_vjp boundary.
"""

from enum import Enum
from functools import partial
from typing import Any, Callable, NewType, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from . import cpp_extensions as tex
from .permutation import (
    PureJaxPermState,
    compute_ragged_all_to_all_params,
    compute_reverse_ragged_all_to_all_params,
    pure_jax_token_combine,
    pure_jax_token_dispatch,
    routing_map_to_selected_experts,
)
from .quantize import (
    QuantizerSet,
    ScaledTensor,
    TensorUsage,
    noop_quantizer_set,
    with_sharding_constraint_by_logical_axes,
)
from .router import ScoreFunction, _validate_score_function
from .sharding import _get_mesh
from .triton_extensions.permutation import (
    make_chunk_sort_map,
    make_row_id_map,
    permute_with_mask_map,
    permute_with_mask_map_and_pad,
    sort_chunks_by_map,
    unpermute_bwd_with_merging_probs,
    unpermute_bwd_with_merging_probs_and_unpad,
    unpermute_with_mask_map,
    unpermute_with_mask_map_and_unpad,
)
from .flax.module import _convert_to_activation_function

PRNGKey = Any
Shape = Tuple[int, ...]
DType = NewType("DType", jnp.dtype)
Array = NewType("Array", jnp.ndarray)


__all__ = ["moe", "PermutationBackend"]


# =============================================================================
# Enums
# =============================================================================


class PermutationBackend(Enum):
    """Token-dispatch / combine backend used by :func:`moe`.

    * ``PURE_JAX``: ``jnp.argsort`` + gather paths compiled as plain XLA;
      typically faster than ``TRITON`` in current testing because XLA can
      fuse the ops with surrounding work.
    * ``TRITON``: TE's fused Triton kernels.
    """

    PURE_JAX = "pure_jax"
    TRITON = "triton"


# =============================================================================
# ctx / dispatch-state key conventions
# =============================================================================
#
# Both ``ctx`` (carried fwd_rule -> bwd_rule) and the dispatch state
# (carried _dispatch -> _combine / _dispatch_bwd / _combine_bwd) are plain
# python dicts. Using a dict (rather than a flax_struct.dataclass) lets us
# vary the populated keys with the static config without breaking
# ``shard_map``'s ``out_specs`` structural match: the spec dict and the
# value dict are built with the SAME keys via :func:`_build_ctx_specs`.
#
# Below is the key glossary so the rest of the file reads cleanly.
#
# DispatchState (dict): values are jnp.ndarray unless noted
#   Always present:
#     "group_sizes"             [n_groups]   per-expert token counts
#                                            (n_groups = E for no-EP,
#                                             E_local for EP)
#     "ep_active"               bool         (carried as a Python flag,
#                                             not in the dict; passed
#                                             alongside)
#   PURE_JAX backend:
#     "sorted_indices"          [num_real + padding] argsort indices
#     "routing_weights"         [num_tokens, topk]   per-token-per-expert weights
#   TRITON backend:
#     "row_id_map"              [num_tokens, 2*E + 1]
#     "pad_offsets"             [E] or None
#     "merging_probs"           [num_tokens, E]
#   EP-only:
#     "all_shards_tokens_per_expert" [num_ep, E]
#     "local_perm_row_id_map"   [recv_buffer_rows]
#     "local_perm_inv_row_id_map" [recv_buffer_rows]
#
# NOTE: per-shard compile-time-constant shapes (num_real_tokens,
# padding_size, pre/post_a2a_buffer_shape) are NOT stored in this
# dict; they are recomputed in _body_fwd/_body_bwd via
# _compute_static_shape_info and passed as Python ints / int tuples to
# the dispatch/combine helpers. Storing them in the dict would cause
# JAX's pytree-flatten across the shard_map boundary to coerce them
# into JitTracer 0-d arrays, which breaks Python-level control flow
# (e.g. ``if padding > 0``) and ``jnp.zeros(shape)`` in the bwd.
#
# MoECtx (dict): values are jnp.ndarray / ScaledTensor unless noted
#   Always present:
#     "x"                       [B, S, H]
#     "gate_kernel"             [H, E] (only meaningful when gate_inside_vjp=True)
#     "logits_2d"               [T, E]   T = local-batch * S
#     "saved_scores"            [T, E]   from fused_topk fwd primitive
#     "routing_map"             [T, E]
#     "dispatch"                DispatchState dict
#     "casted_sorted_x_lhs_trans"   ScaledTensor or ndarray
#     "casted_wi_0_rhs_trans"   ScaledTensor or ndarray
#     "casted_wi_1_rhs_trans"   ScaledTensor or ndarray
#     "layer_w0"                ndarray  (pre-activation)
#     "layer_w1"                ndarray
#     "casted_intermediate_lhs_trans" ScaledTensor or ndarray
#     "casted_wo_rhs_trans"     ScaledTensor or ndarray
#     "expert_outputs"          ndarray  (FFN output, needed for TRITON
#                                         combine_bwd's
#                                         unpermute_bwd_with_merging_probs)
#     "local_group_sizes"       [n_groups] -- mirrors dispatch.group_sizes
#                                              but kept here for FFN bwd
#                                              convenience
#   Optional:
#     "expert_bias"             [E]   only when expert_bias was provided
#     "wi_0_bias_shape"         tuple -- only when bias is used (carried
#                                        non-diff via static side; here
#                                        only if needed)
#     "aux_const_buf"           ndarray  -- only when aux_loss_coeff > 0
#     "aux_tokens_per_expert"   [E]      -- ditto
#     "aux_logits_for_score"    [global_T, E] -- ditto, may be the
#                                                gathered global logits
#                                                or the local logits


# =============================================================================
# Static shape helper
# =============================================================================
#
# A set of per-shard shape/size values that the dispatch and combine
# helpers (both fwd and bwd) need. They're all derivable from existing
# static args, so we recompute them in both ``_body_fwd`` and
# ``_body_bwd`` and pass them as Python ints / int-tuples through
# explicit kwargs. We MUST NOT stash them inside the dynamic
# ``state`` / ``ctx`` dict: when the dict crosses the EP shard_map's
# out_specs/in_specs boundary, JAX's pytree-flatten coerces any Python
# int leaves into traced 0-d arrays, which then breaks dependent Python
# code in the bwd (e.g. ``if padding > 0`` and ``jnp.zeros(shape)``).


def _compute_static_shape_info(
    *,
    batch_size: int,
    sequence_length: int,
    hidden: int,
    num_experts: int,
    num_experts_per_tok: int,
    align_size: int,
    ep_active: bool,
    num_ep: int = 1,
    fsdp_sizes: Tuple[int, ...] = (),
    recv_buffer_rows: int = 0,
    batch_is_per_shard: bool = True,
) -> dict:
    """Compute per-shard compile-time-constant shape info used by both
    dispatch/combine fwd and dispatch/combine bwd.

    Returned dict has Python ints / int tuples (NOT jnp arrays) so the
    caller can pass them as ordinary static keyword args. See the
    module-level comment above for why this matters.

    ``batch_is_per_shard`` controls whether ``batch_size`` is already
    sharded (True -- e.g. when this is called from inside a shard_map
    body, where ``x.shape[0]`` reports the per-shard batch size) or
    global (False -- e.g. when computing from x.shape outside the
    shard_map body).

    Keys
    ----
    num_real_tokens : int
        Per-shard count of real (non-padding) permuted tokens, i.e.
        ``per_shard_num_tokens * num_experts_per_tok``.
    padding_size : int
        Per-shard number of alignment-padding tokens appended to the
        sort buffer (``num_experts * (align_size - 1)`` when
        ``align_size > 0``, else ``0``). Matches the convention used
        by ``pure_jax_token_dispatch``.
    pre_a2a_buffer_shape : tuple[int, int]
        ``(num_real_tokens + padding_size, hidden)`` -- the per-shard
        shape of the sorted-inputs buffer that is sent over the EP
        ragged_all_to_all in the fwd direction.
    post_a2a_buffer_shape : Optional[tuple[int, int]]
        ``(recv_buffer_rows, hidden)`` when EP is active, ``None``
        otherwise.
    """
    import math

    if ep_active and not batch_is_per_shard:
        dp_size = math.prod(fsdp_sizes) if fsdp_sizes else 1
        per_shard_batch = batch_size // (num_ep * dp_size)
    else:
        per_shard_batch = batch_size
    per_shard_num_tokens = per_shard_batch * sequence_length
    num_real_tokens = per_shard_num_tokens * num_experts_per_tok
    padding_size = num_experts * (align_size - 1) if align_size > 0 else 0
    pre_a2a_buffer_shape = (num_real_tokens + padding_size, hidden)
    post_a2a_buffer_shape = (recv_buffer_rows, hidden) if ep_active else None
    return dict(
        num_real_tokens=num_real_tokens,
        padding_size=padding_size,
        pre_a2a_buffer_shape=pre_a2a_buffer_shape,
        post_a2a_buffer_shape=post_a2a_buffer_shape,
    )


# =============================================================================
# Dispatch / combine helpers (no VJP boundary -- pure Python)
# =============================================================================


def _dispatch(
    inputs_2d: jnp.ndarray,
    sparse_probs: jnp.ndarray,
    routing_map: jnp.ndarray,
    *,
    backend: PermutationBackend,
    num_experts: int,
    num_experts_per_tok: int,
    align_size: int,
    # EP-only:
    ep_active: bool,
    ep_axis: Optional[str],
    num_ep: int,
    recv_buffer_rows: int,
    shard_id: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, dict]:
    """``permute -> (a2a -> local_permute) iff ep_active``.

    Returns ``(sorted_x, state)`` where ``sorted_x`` has shape
    ``[buffer_rows, hidden]`` -- ``E`` groups (no-EP) or ``E_local`` groups
    (EP) -- and ``state`` is a dict carrying everything :func:`_combine`
    and the bwd helpers need to reverse the operation.

    Bypasses the ``custom_vjp``-wrapped public ``token_dispatch`` /
    ``pure_jax_token_dispatch`` wrappers (well, mostly: PURE_JAX still
    composes through ``pure_jax_token_dispatch`` because that helper has
    no ``custom_vjp`` itself -- only its inner ``_sort_activations`` does,
    which is fine since we never auto-diff through it from this layer).
    For TRITON we call the underlying ``permute_with_mask_map`` /
    ``permute_with_mask_map_and_pad`` primitives directly.
    """
    num_tokens, hidden = inputs_2d.shape
    topk = num_experts_per_tok
    state: dict = {}

    # ------------------------------------------------------------------
    # Step 1: global permute (every shard routes its own tokens over the
    # full expert axis). Backend-specific.
    # ------------------------------------------------------------------
    if backend is PermutationBackend.PURE_JAX:
        selected_experts, routing_weights = routing_map_to_selected_experts(
            sparse_probs, routing_map, topk
        )
        sorted_inputs, perm_state, group_sizes = pure_jax_token_dispatch(
            inputs_2d,
            selected_experts,
            num_experts=num_experts,
            num_experts_per_tok=topk,
            align_size=align_size,
        )
        # NOTE: ``perm_state.num_real_tokens`` and ``perm_state.padding_size``
        # are compile-time Python ints; intentionally NOT stored in
        # ``state`` (would be coerced to JitTracer 0-d arrays under
        # the EP shard_map's pytree flatten). Recompute via
        # ``_compute_static_shape_info`` in the bwd / EP-combine
        # call sites that need them.
        state["sorted_indices"] = perm_state.sorted_indices
        state["routing_weights"] = routing_weights
    else:
        # TRITON backend -- inline the underlying primitive sequence
        # (mirrors ``_token_dispatch_fwd_rule`` but exposes the residuals
        # to our ctx instead of saving them inside another custom_vjp).
        num_out_tokens = num_tokens * topk
        row_id_map = make_row_id_map(routing_map, num_tokens, num_experts)
        tokens_per_expert = jnp.sum(routing_map, axis=0).astype(jnp.int32)
        if align_size > 0:
            target_tokens_per_expert = (
                jnp.ceil(tokens_per_expert / align_size) * align_size
            ).astype(jnp.int32)
            pad_lengths = target_tokens_per_expert - tokens_per_expert
            cum_pad = jnp.cumsum(pad_lengths)
            pad_offsets = jnp.concatenate([jnp.array([0], dtype=cum_pad.dtype), cum_pad[:-1]])
            worst_case_out_tokens = (
                (num_out_tokens + num_experts * (align_size - 1)) // align_size
            ) * align_size
            sorted_inputs, _ = permute_with_mask_map_and_pad(
                inputs_2d,
                row_id_map,
                None,
                pad_offsets,
                num_tokens,
                num_experts,
                worst_case_out_tokens,
                hidden,
                align_size=align_size,
            )
            group_sizes = target_tokens_per_expert
        else:
            sorted_inputs, _ = permute_with_mask_map(
                inputs_2d,
                row_id_map,
                None,
                num_tokens,
                num_experts,
                num_out_tokens,
                hidden,
            )
            pad_offsets = None
            group_sizes = tokens_per_expert
        state["row_id_map"] = row_id_map
        state["pad_offsets"] = pad_offsets
        state["merging_probs"] = sparse_probs

    state["group_sizes"] = group_sizes

    if not ep_active:
        return sorted_inputs, state

    # ------------------------------------------------------------------
    # Step 2 (EP only): all_gather per-expert counts so every shard knows
    # the [num_ep, num_experts] token-count matrix.
    # ------------------------------------------------------------------
    all_shards_tokens_per_expert = jax.lax.all_gather(
        group_sizes[None, :],
        axis_name=ep_axis,
        axis=0,
        tiled=True,
    )

    # ------------------------------------------------------------------
    # Step 3 (EP only): forward ragged_all_to_all over the EP axis.
    # ------------------------------------------------------------------
    in_off, send_sz, out_off, recv_sz = compute_ragged_all_to_all_params(
        all_shards_tokens_per_expert, shard_id, num_ep
    )
    pre_a2a_buffer_shape = sorted_inputs.shape
    post_a2a_buffer_shape = (recv_buffer_rows, hidden)
    recv_buf = jnp.zeros(post_a2a_buffer_shape, dtype=sorted_inputs.dtype)
    x_recv = jax.lax.ragged_all_to_all(
        sorted_inputs, recv_buf, in_off, send_sz, out_off, recv_sz, axis_name=ep_axis
    )

    # ------------------------------------------------------------------
    # Step 4 (EP only): local permute -- (source_shard, expert) ->
    # (expert, shard). Inlined ``local_permute_after_a2a`` so we control
    # both the row_id_map and its inverse for the bwd.
    # ------------------------------------------------------------------
    num_experts_local = num_experts // num_ep
    local_expert_start = shard_id * num_experts_local
    local_expert_columns = jax.lax.dynamic_slice(
        all_shards_tokens_per_expert,
        start_indices=(0, local_expert_start),
        slice_sizes=(num_ep, num_experts_local),
    )
    split_sizes = local_expert_columns.reshape(-1)  # source-major
    indices_matrix = jnp.arange(num_ep * num_experts_local, dtype=jnp.int32).reshape(
        num_ep, num_experts_local
    )
    sorted_chunk_indices = indices_matrix.T.reshape(-1)  # source-major -> expert-major
    num_chunks = num_ep * num_experts_local
    # Build a SINGLE row_id_map. ``is_forward=True`` permutes
    # source-major -> expert-major; ``is_forward=False`` is the exact
    # inverse (this is exactly what ``_sort_chunks_by_index_bwd_rule``
    # uses on the saved residual). _MoEBlock builds two row_id_maps
    # only because it calls ``sort_chunks_by_index`` twice -- once in
    # ``local_permute_after_a2a`` and again in ``local_unpermute_before_a2a``;
    # each of those wrappers calls ``make_chunk_sort_map`` internally.
    # Here we share one map across (fwd permute, fwd inverse-permute,
    # bwd permute, bwd inverse-permute).
    local_perm_row_id_map = make_chunk_sort_map(
        split_sizes, sorted_chunk_indices, recv_buffer_rows, num_chunks
    )
    sorted_x, _ = sort_chunks_by_map(
        x_recv, local_perm_row_id_map, None, recv_buffer_rows, hidden, is_forward=True
    )
    local_group_sizes = jnp.sum(local_expert_columns, axis=0)

    state["all_shards_tokens_per_expert"] = all_shards_tokens_per_expert
    state["local_perm_row_id_map"] = local_perm_row_id_map
    # NOTE: pre_a2a_buffer_shape and post_a2a_buffer_shape are compile-
    # time int tuples; intentionally NOT stored in ``state`` (would be
    # coerced to JitTracer 0-d arrays under the EP shard_map's pytree
    # flatten). Recompute via ``_compute_static_shape_info`` in the
    # bwd call sites that need them.
    # For EP, we override ``group_sizes`` to be the per-local-expert
    # counts (the FFN runs over E_local groups, not E). The original
    # global ``group_sizes`` lives inside ``all_shards_tokens_per_expert``
    # if anyone needs it for diagnostics.
    state["group_sizes"] = local_group_sizes

    return sorted_x, state


def _combine(
    expert_outputs: jnp.ndarray,
    state: dict,
    *,
    backend: PermutationBackend,
    ep_active: bool,
    batch_size: int,
    sequence_length: int,
    dtype: jnp.dtype,
    num_experts_per_tok: int,
    # Per-shard compile-time-constant shape info (Python ints / int tuples).
    # Computed by _compute_static_shape_info in the caller, passed here
    # rather than stored in ``state`` to survive shard_map crossings.
    num_real_tokens: int,
    padding_size: int,
    pre_a2a_buffer_shape: Tuple[int, int],
    # EP-only:
    ep_axis: Optional[str],
    shard_id: Optional[jnp.ndarray] = None,
    num_ep: int = 1,
) -> jnp.ndarray:
    """Inverse of :func:`_dispatch`. Returns ``[B, S, H]``."""
    if ep_active:
        # Step 1 (EP): inverse local permute. Reuse the SAME row_id_map
        # built in _dispatch by setting is_forward=False (this is the
        # exact inverse, identical to what
        # ``_sort_chunks_by_index_bwd_rule`` does with the saved residual).
        recv_buffer_rows, hidden = expert_outputs.shape
        x_send_back, _ = sort_chunks_by_map(
            expert_outputs,
            state["local_perm_row_id_map"],
            None,
            recv_buffer_rows,
            hidden,
            is_forward=False,
        )
        # Step 2 (EP): reverse ragged_all_to_all.
        in_off_r, send_sz_r, out_off_r, recv_sz_r = compute_reverse_ragged_all_to_all_params(
            state["all_shards_tokens_per_expert"], shard_id, num_ep
        )
        send_back_buf = jnp.zeros(pre_a2a_buffer_shape, dtype=expert_outputs.dtype)
        expert_outputs = jax.lax.ragged_all_to_all(
            x_send_back,
            send_back_buf,
            in_off_r,
            send_sz_r,
            out_off_r,
            recv_sz_r,
            axis_name=ep_axis,
        )

    # Step 3: global combine.
    if backend is PermutationBackend.PURE_JAX:
        # Reuse the reference pure-jax implementation; it has no
        # custom_vjp on its outer surface so we can call it freely.
        perm_state = PureJaxPermState(
            sorted_indices=state["sorted_indices"],
            num_real_tokens=num_real_tokens,
            padding_size=padding_size,
        )
        return pure_jax_token_combine(
            expert_outputs,
            perm_state,
            state["routing_weights"],
            num_experts_per_tok=num_experts_per_tok,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
    # TRITON
    num_tokens = state["row_id_map"].shape[0]
    num_experts = (state["row_id_map"].shape[1] - 1) // 2
    hidden = expert_outputs.shape[-1]
    if state["pad_offsets"] is not None:
        out_2d, _ = unpermute_with_mask_map_and_unpad(
            expert_outputs,
            state["row_id_map"],
            state["merging_probs"],
            None,
            state["pad_offsets"],
            num_tokens,
            num_experts,
            hidden,
        )
    else:
        out_2d, _ = unpermute_with_mask_map(
            expert_outputs,
            state["row_id_map"],
            state["merging_probs"],
            None,
            num_tokens,
            num_experts,
            hidden,
        )
    return out_2d.reshape(batch_size, sequence_length, hidden).astype(dtype)


def _combine_bwd(
    d_output: jnp.ndarray,
    state: dict,
    expert_outputs: jnp.ndarray,
    *,
    backend: PermutationBackend,
    ep_active: bool,
    batch_size: int,
    sequence_length: int,
    dtype: jnp.dtype,
    num_experts: int,
    num_experts_per_tok: int,
    # Per-shard compile-time-constant shape info (Python ints / int tuples).
    # See ``_compute_static_shape_info`` and the note in ``_dispatch``
    # for why these are kwargs rather than state-dict entries.
    num_real_tokens: int,
    padding_size: int,
    post_a2a_buffer_shape: Optional[Tuple[int, int]],
    # EP-only:
    ep_axis: Optional[str],
    shard_id: Optional[jnp.ndarray] = None,
    num_ep: int = 1,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Inverse of :func:`_combine` on the cotangent.

    Returns ``(d_expert_outputs, d_routing_weights_or_merging_probs)``.

    ``expert_outputs`` is the *forward* output of the FFN (same value the
    fwd handed to :func:`_combine`). It's required by the TRITON
    combine_bwd kernel; for PURE_JAX we don't need it but accept it for
    a symmetric signature.
    """
    # Step 3 inverse: global combine bwd.
    d_output_2d = d_output.reshape(-1, d_output.shape[-1])
    if backend is PermutationBackend.PURE_JAX:
        # The pure-jax combine is:
        #   unsort = _sort_activations(expert_outputs, argsort(sorted_indices))
        #   if pad: unsort = unsort[:num_real]
        #   reshape -> einsum BKE,BK -> BE -> reshape to BSE
        # Hand-derive the bwd in plain JAX (no custom_vjp involved):
        unsort_indices = jnp.argsort(state["sorted_indices"])
        topk = num_experts_per_tok
        num_real = num_real_tokens
        padding = padding_size
        # Recover the unsorted intermediate that the fwd produced (we
        # need it for the d_routing_weights pullback). Apply the same
        # gather the fwd did.
        unsort_intermediate = expert_outputs[unsort_indices]
        if padding > 0:
            unsort_intermediate = unsort_intermediate[:num_real]
        # Bwd of einsum/reshape:
        # output[B, E] = sum_K intermediate[B, K, E] * weights[B, K]
        # d_intermediate[B, K, E] = d_output[B, E] * weights[B, K]
        # d_weights[B, K]         = sum_E d_output[B, E] * intermediate[B, K, E]
        rw = state["routing_weights"].reshape(-1, topk)
        intermediate_3d = unsort_intermediate.reshape(rw.shape[0], topk, -1)
        rw_cast = rw.astype(intermediate_3d.dtype)
        d_intermediate_3d = jnp.einsum("BE,BK -> BKE", d_output_2d, rw_cast)
        d_routing_weights = jnp.einsum("BE,BKE -> BK", d_output_2d, intermediate_3d).astype(
            state["routing_weights"].dtype
        )
        d_routing_weights = d_routing_weights.reshape(state["routing_weights"].shape)
        d_unsort_intermediate = d_intermediate_3d.reshape(num_real, -1)
        # Pad back with zeros if the fwd stripped padding.
        if padding > 0:
            d_unsort_intermediate = jnp.concatenate(
                [
                    d_unsort_intermediate,
                    jnp.zeros(
                        (padding, d_unsort_intermediate.shape[-1]),
                        dtype=d_unsort_intermediate.dtype,
                    ),
                ],
                axis=0,
            )
        # Bwd of the gather is gather-by-original-indices:
        #   sorted = unsort[argsort(sorted_indices)]
        #   d_sorted = scatter d_unsort via argsort(sorted_indices)
        #            = d_unsort[sorted_indices]  (gather by original sorted_indices,
        #              which is the inverse of argsort(sorted_indices)).
        d_expert_outputs_global = d_unsort_intermediate[state["sorted_indices"]]
    else:
        # TRITON combine bwd: requires fwd_input (expert_outputs).
        num_tokens = state["row_id_map"].shape[0]
        n_experts = (state["row_id_map"].shape[1] - 1) // 2
        hidden = d_output_2d.shape[-1]
        num_out_tokens = expert_outputs.shape[0]
        if state["pad_offsets"] is not None:
            d_expert_outputs_global, d_merging_probs = unpermute_bwd_with_merging_probs_and_unpad(
                d_output_2d,
                state["row_id_map"],
                expert_outputs,
                state["merging_probs"],
                state["pad_offsets"],
                num_tokens,
                n_experts,
                num_out_tokens,
                hidden,
            )
            # The kernel only writes positions tokens map to; padded
            # positions may contain NaN. Replace with zeros (matches
            # ``_token_combine_bwd_rule``).
            d_expert_outputs_global = jnp.where(
                jnp.isnan(d_expert_outputs_global), 0.0, d_expert_outputs_global
            )
        else:
            d_expert_outputs_global, d_merging_probs = unpermute_bwd_with_merging_probs(
                d_output_2d,
                state["row_id_map"],
                expert_outputs,
                state["merging_probs"],
                num_tokens,
                n_experts,
                num_out_tokens,
                hidden,
            )
        d_routing_weights = d_merging_probs

    if not ep_active:
        return d_expert_outputs_global, d_routing_weights

    # Step 2 (EP) inverse: bwd of reverse ragged_all_to_all is a forward
    # ragged_all_to_all using the SAME forward parameters (sender /
    # receiver roles swap from the reverse direction back to forward).
    in_off_f, send_sz_f, out_off_f, recv_sz_f = compute_ragged_all_to_all_params(
        state["all_shards_tokens_per_expert"], shard_id, num_ep
    )
    recv_buf_for_bwd = jnp.zeros(post_a2a_buffer_shape, dtype=d_expert_outputs_global.dtype)
    d_x_send_back = jax.lax.ragged_all_to_all(
        d_expert_outputs_global,
        recv_buf_for_bwd,
        in_off_f,
        send_sz_f,
        out_off_f,
        recv_sz_f,
        axis_name=ep_axis,
    )
    # Step 1 (EP) inverse: combine fwd applied is_forward=False; the
    # bwd is is_forward=True with the SAME row_id_map.
    recv_buffer_rows, hidden = d_x_send_back.shape
    d_expert_outputs, _ = sort_chunks_by_map(
        d_x_send_back,
        state["local_perm_row_id_map"],
        None,
        recv_buffer_rows,
        hidden,
        is_forward=True,
    )
    return d_expert_outputs, d_routing_weights


def _dispatch_bwd(
    d_sorted_x: jnp.ndarray,
    state: dict,
    inputs_2d_shape: Tuple[int, ...],
    *,
    backend: PermutationBackend,
    ep_active: bool,
    num_experts: int,
    num_experts_per_tok: int,
    # Per-shard compile-time-constant shape info (Python ints / int tuples).
    # See ``_compute_static_shape_info`` and the note in ``_dispatch``
    # for why these are kwargs rather than state-dict entries.
    num_real_tokens: int,
    padding_size: int,
    pre_a2a_buffer_shape: Tuple[int, int],
    # EP-only:
    ep_axis: Optional[str],
    shard_id: Optional[jnp.ndarray] = None,
    num_ep: int = 1,
) -> jnp.ndarray:
    """Inverse of :func:`_dispatch` on the cotangent. Returns ``d_inputs_2d``.

    The probs path through dispatch is always discarded (PURE_JAX never
    threads probs through dispatch; TRITON technically does but the
    caller drops ``permuted_probs``, so its cotangent is structurally
    zero). The probs gradient instead flows back through
    :func:`_combine_bwd`.
    """
    if ep_active:
        # Step 4 inverse: dispatch fwd applied is_forward=True; bwd is
        # is_forward=False with the SAME row_id_map.
        recv_buffer_rows, hidden = d_sorted_x.shape
        d_x_recv, _ = sort_chunks_by_map(
            d_sorted_x,
            state["local_perm_row_id_map"],
            None,
            recv_buffer_rows,
            hidden,
            is_forward=False,
        )
        # Step 3 inverse: bwd of forward ragged_a2a is the reverse-direction
        # ragged_a2a using the SAME params with sender/receiver swapped.
        in_off_r, send_sz_r, out_off_r, recv_sz_r = compute_reverse_ragged_all_to_all_params(
            state["all_shards_tokens_per_expert"], shard_id, num_ep
        )
        recv_buf_pre = jnp.zeros(pre_a2a_buffer_shape, dtype=d_x_recv.dtype)
        d_sorted_x = jax.lax.ragged_all_to_all(
            d_x_recv,
            recv_buf_pre,
            in_off_r,
            send_sz_r,
            out_off_r,
            recv_sz_r,
            axis_name=ep_axis,
        )

    # Step 1 inverse: global permute bwd.
    if backend is PermutationBackend.PURE_JAX:
        # Fwd was: replicated = repeat(inputs_2d, topk, axis=0)
        #          padded = pad(replicated, (0, padding_size))
        #          sorted = padded[sorted_indices]
        # Bwd:     d_padded = scatter via sorted_indices
        #                   = d_sorted[argsort(sorted_indices)]
        #          d_replicated = d_padded[:num_real]
        #          d_inputs_2d  = d_replicated.reshape(T, topk, H).sum(axis=1)
        sorted_indices = state["sorted_indices"]
        num_real = num_real_tokens
        padding = padding_size
        topk = num_experts_per_tok
        unsort_indices = jnp.argsort(sorted_indices)
        d_padded = d_sorted_x[unsort_indices]
        if padding > 0:
            d_replicated = d_padded[:num_real]
        else:
            d_replicated = d_padded
        num_tokens = inputs_2d_shape[0]
        hidden = inputs_2d_shape[-1]
        d_inputs_2d = d_replicated.reshape(num_tokens, topk, hidden).sum(axis=1)
        return d_inputs_2d

    # TRITON: bwd is unpermute_with_mask_map[_and_unpad].
    num_tokens = inputs_2d_shape[0]
    hidden = inputs_2d_shape[-1]
    if state["pad_offsets"] is not None:
        d_inputs_2d, _ = unpermute_with_mask_map_and_unpad(
            d_sorted_x,
            state["row_id_map"],
            None,
            None,
            state["pad_offsets"],
            num_tokens,
            num_experts,
            hidden,
        )
    else:
        d_inputs_2d, _ = unpermute_with_mask_map(
            d_sorted_x,
            state["row_id_map"],
            None,
            None,
            num_tokens,
            num_experts,
            hidden,
        )
    return d_inputs_2d


# =============================================================================
# Per-shard body
# =============================================================================


def _body_fwd(
    captured: dict,
    *,
    # Statics
    num_experts: int,
    num_experts_per_tok: int,
    activation_type: str,
    score_function: ScoreFunction,
    use_pre_softmax: bool,
    num_groups: Optional[int],
    group_topk: Optional[int],
    scaling_factor: float,
    aux_loss_coeff: float,
    permutation_backend: PermutationBackend,
    align_size: int,
    gate_inside_vjp: bool,
    quantizer_sets: Tuple[QuantizerSet, QuantizerSet, QuantizerSet],
    dtype: jnp.dtype,
    # EP-only statics
    ep_active: bool,
    ep_axis: Optional[str],
    data_parallelism_axes: Tuple[str, ...],
    fsdp_sizes: Tuple[int, ...],
    num_ep: int,
    num_experts_local: int,
    recv_buffer_rows: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:
    """Per-shard forward body. Returns ``(output, aux_loss, ctx_dict)``.

    ``aux_loss`` is always materialized (zeros scalar when disabled) so
    the ``shard_map``'s ``out_specs`` has a static structure.
    """
    if not gate_inside_vjp:
        raise NotImplementedError(
            "gate_inside_vjp=False is deferred to a follow-up PR; for now"
            " the gate GEMM lives inside the MoE VJP."
        )

    x = captured["inputs"]
    gate_kernel = captured["gate_kernel"]
    wi_0 = captured["wi_0"]
    wi_1 = captured["wi_1"]
    wo = captured["wo"]
    wi_0_bias = captured.get("wi_0_bias")
    wi_1_bias = captured.get("wi_1_bias")
    wo_bias = captured.get("wo_bias")
    expert_bias = captured.get("expert_bias")

    batch_size, sequence_length, hidden = x.shape

    # ---------------- Stage 1: gate ----------------
    gate_kernel_cast = gate_kernel.astype(x.dtype)
    gate_logits = jnp.einsum("bsh,he->bse", x, gate_kernel_cast)
    logits_2d = gate_logits.reshape(-1, num_experts)
    inputs_2d = x.reshape(-1, hidden)

    # ---------------- Stage 2: routing ----------------
    # Under EP, expert_bias is sharded P(ep_axis); the router needs the
    # full E-dim view, so all_gather it.
    if ep_active and expert_bias is not None:
        full_expert_bias = jax.lax.all_gather(expert_bias, axis_name=ep_axis, tiled=True)
    else:
        full_expert_bias = expert_bias
    # Pass an empty array sentinel when expert_bias is unused (the
    # underlying primitive expects a real ndarray, not None).
    eb_arg = (
        full_expert_bias if full_expert_bias is not None else jnp.zeros((0,), dtype=jnp.float32)
    )
    sparse_probs, routing_map, saved_scores = tex.fused_topk_with_score_function_fwd(
        logits_2d,
        topk=num_experts_per_tok,
        use_pre_softmax=use_pre_softmax,
        num_groups=-1 if num_groups is None else num_groups,
        group_topk=-1 if group_topk is None else group_topk,
        scaling_factor=scaling_factor,
        score_function=score_function,
        expert_bias=eb_arg,
        compute_aux_scores=False,
    )
    sparse_probs = sparse_probs.astype(dtype)

    # ---------------- Stage 2b: aux loss ----------------
    if aux_loss_coeff > 0.0:
        if ep_active:
            collective_axes: Any = (
                ep_axis if not data_parallelism_axes else (ep_axis, *data_parallelism_axes)
            )
            global_logits_2d = jax.lax.all_gather(
                logits_2d, axis_name=collective_axes, axis=0, tiled=True
            )
            _, global_routing_map, _ = tex.fused_topk_with_score_function_fwd(
                global_logits_2d,
                topk=num_experts_per_tok,
                use_pre_softmax=use_pre_softmax,
                num_groups=-1 if num_groups is None else num_groups,
                group_topk=-1 if group_topk is None else group_topk,
                scaling_factor=scaling_factor,
                score_function=score_function,
                expert_bias=eb_arg,
                compute_aux_scores=False,
            )
            aux_tokens_per_expert = jnp.sum(global_routing_map.astype(jnp.int32), axis=0)
            aux_logits_for_score = global_logits_2d
        else:
            aux_tokens_per_expert = jnp.sum(routing_map.astype(jnp.int32), axis=0)
            aux_logits_for_score = logits_2d
        # Aux-side scores: clean per-expert scores (no grouped routing,
        # no bias). compute_aux_scores=True takes a separate path that
        # ignores the grouping knobs.
        aux_probs, _aux_routing_map, aux_saved_scores = tex.fused_topk_with_score_function_fwd(
            aux_logits_for_score.astype(jnp.float32),
            topk=num_experts_per_tok,
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
            topk=num_experts_per_tok,
            coeff=aux_loss_coeff,
        )
    else:
        aux_loss = jnp.zeros((), dtype=dtype)
        aux_const_buf = None
        aux_tokens_per_expert = None
        aux_logits_for_score = None
        aux_saved_scores = None

    # ---------------- Stage 3: dispatch ----------------
    shard_id = jax.lax.axis_index(ep_axis) if ep_active else None
    sorted_x, dispatch_state = _dispatch(
        inputs_2d,
        sparse_probs,
        routing_map,
        backend=permutation_backend,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        align_size=align_size,
        ep_active=ep_active,
        ep_axis=ep_axis,
        num_ep=num_ep,
        recv_buffer_rows=recv_buffer_rows,
        shard_id=shard_id,
    )
    local_group_sizes = dispatch_state["group_sizes"]

    # ---------------- Stage 4: per-expert FFN (inlined) ----------------
    q_set_w0, q_set_w1, q_set_wo = quantizer_sets
    if q_set_w0 == noop_quantizer_set:
        wi_0 = wi_0.astype(sorted_x.dtype)
    if q_set_w1 == noop_quantizer_set:
        wi_1 = wi_1.astype(sorted_x.dtype)
    if q_set_wo == noop_quantizer_set:
        wo = wo.astype(sorted_x.dtype)

    # GEMM 1: layer_w0 = sorted_x @ wi_0
    casted_sorted_x_w0 = tex.grouped_quantize(
        sorted_x, q_set_w0.x, local_group_sizes, flatten_axis=-1
    )
    casted_wi_0 = tex.grouped_quantize(wi_0, q_set_w0.kernel, flatten_axis=-1)
    layer_w0 = tex.grouped_gemm(
        casted_sorted_x_w0.get_tensor(usage=TensorUsage.LHS),
        casted_wi_0.get_tensor(usage=TensorUsage.RHS),
        contracting_dims=((1,), (1,)),
        bias=wi_0_bias,
    )
    casted_sorted_x_lhs_trans = casted_sorted_x_w0.get_tensor(usage=TensorUsage.LHS_TRANS)
    casted_wi_0_rhs_trans = casted_wi_0.get_tensor(usage=TensorUsage.RHS_TRANS)
    if isinstance(casted_sorted_x_lhs_trans, ScaledTensor):
        casted_sorted_x_lhs_trans = casted_sorted_x_lhs_trans.checkpoint(q_set_w0.x)
    if isinstance(casted_wi_0_rhs_trans, ScaledTensor):
        casted_wi_0_rhs_trans = casted_wi_0_rhs_trans.checkpoint(q_set_w0.kernel)

    # GEMM 2: layer_w1 = sorted_x @ wi_1
    casted_sorted_x_w1 = tex.grouped_quantize(
        sorted_x, q_set_w1.x, local_group_sizes, flatten_axis=-1
    )
    casted_wi_1 = tex.grouped_quantize(wi_1, q_set_w1.kernel, flatten_axis=-1)
    layer_w1 = tex.grouped_gemm(
        casted_sorted_x_w1.get_tensor(usage=TensorUsage.LHS),
        casted_wi_1.get_tensor(usage=TensorUsage.RHS),
        contracting_dims=((1,), (1,)),
        bias=wi_1_bias,
    )
    casted_wi_1_rhs_trans = casted_wi_1.get_tensor(usage=TensorUsage.RHS_TRANS)
    if isinstance(casted_wi_1_rhs_trans, ScaledTensor):
        casted_wi_1_rhs_trans = casted_wi_1_rhs_trans.checkpoint(q_set_w1.kernel)

    # Activation: intermediate = act(layer_w0) * layer_w1
    act_fn = _convert_to_activation_function(activation_type)
    intermediate = act_fn(layer_w0) * layer_w1

    # GEMM 3: expert_outputs = intermediate @ wo
    casted_intermediate = tex.grouped_quantize(
        intermediate, q_set_wo.x, local_group_sizes, flatten_axis=-1
    )
    casted_wo = tex.grouped_quantize(wo, q_set_wo.kernel, flatten_axis=-1)
    expert_outputs = tex.grouped_gemm(
        casted_intermediate.get_tensor(usage=TensorUsage.LHS),
        casted_wo.get_tensor(usage=TensorUsage.RHS),
        contracting_dims=((1,), (1,)),
        bias=wo_bias,
    )
    casted_intermediate_lhs_trans = casted_intermediate.get_tensor(usage=TensorUsage.LHS_TRANS)
    casted_wo_rhs_trans = casted_wo.get_tensor(usage=TensorUsage.RHS_TRANS)
    if isinstance(casted_intermediate_lhs_trans, ScaledTensor):
        casted_intermediate_lhs_trans = casted_intermediate_lhs_trans.checkpoint(q_set_wo.x)
    if isinstance(casted_wo_rhs_trans, ScaledTensor):
        casted_wo_rhs_trans = casted_wo_rhs_trans.checkpoint(q_set_wo.kernel)

    # ---------------- Stage 5: combine ----------------
    # Compute per-shard static shape info once and pass through both
    # _combine and (later) the bwd helpers via kwargs -- never via the
    # state dict, which gets pytree-flattened across shard_map and would
    # coerce Python ints into JitTracer 0-d arrays.
    _static_shape = _compute_static_shape_info(
        batch_size=batch_size,
        sequence_length=sequence_length,
        hidden=hidden,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        align_size=align_size,
        ep_active=ep_active,
        num_ep=num_ep,
        fsdp_sizes=fsdp_sizes,
        recv_buffer_rows=recv_buffer_rows,
    )
    output = _combine(
        expert_outputs,
        dispatch_state,
        backend=permutation_backend,
        ep_active=ep_active,
        batch_size=batch_size,
        sequence_length=sequence_length,
        dtype=dtype,
        num_experts_per_tok=num_experts_per_tok,
        num_real_tokens=_static_shape["num_real_tokens"],
        padding_size=_static_shape["padding_size"],
        pre_a2a_buffer_shape=_static_shape["pre_a2a_buffer_shape"],
        ep_axis=ep_axis,
        shard_id=shard_id,
        num_ep=num_ep,
    )

    # ---------------- Build ctx dict ----------------
    ctx: dict = {
        "x": x,
        "gate_kernel": gate_kernel,
        "logits_2d": logits_2d,
        "saved_scores": saved_scores,
        "routing_map": routing_map,
        "dispatch": dispatch_state,
        "casted_sorted_x_lhs_trans": casted_sorted_x_lhs_trans,
        "casted_wi_0_rhs_trans": casted_wi_0_rhs_trans,
        "casted_wi_1_rhs_trans": casted_wi_1_rhs_trans,
        "layer_w0": layer_w0,
        "layer_w1": layer_w1,
        "casted_intermediate_lhs_trans": casted_intermediate_lhs_trans,
        "casted_wo_rhs_trans": casted_wo_rhs_trans,
        "expert_outputs": expert_outputs,
        "local_group_sizes": local_group_sizes,
    }
    if expert_bias is not None:
        ctx["expert_bias"] = expert_bias
    if wi_0_bias is not None:
        ctx["has_wi_bias"] = True  # NOTE: this is python bool; we DON'T store it
        # (we only store array leaves in ctx; structural flags travel via statics).
        del ctx["has_wi_bias"]
    if aux_loss_coeff > 0.0:
        ctx["aux_const_buf"] = aux_const_buf
        ctx["aux_tokens_per_expert"] = aux_tokens_per_expert
        ctx["aux_logits_for_score"] = aux_logits_for_score
        ctx["aux_saved_scores"] = aux_saved_scores

    return output, aux_loss, ctx


def _body_bwd(
    ctx: dict,
    dy_pair: Tuple[jnp.ndarray, jnp.ndarray],
    *,
    num_experts: int,
    num_experts_per_tok: int,
    activation_type: str,
    score_function: ScoreFunction,
    use_pre_softmax: bool,
    num_groups: Optional[int],
    group_topk: Optional[int],
    scaling_factor: float,
    aux_loss_coeff: float,
    permutation_backend: PermutationBackend,
    align_size: int,
    gate_inside_vjp: bool,
    quantizer_sets: Tuple[QuantizerSet, QuantizerSet, QuantizerSet],
    dtype: jnp.dtype,
    ep_active: bool,
    ep_axis: Optional[str],
    data_parallelism_axes: Tuple[str, ...],
    fsdp_sizes: Tuple[int, ...],
    num_ep: int,
    num_experts_local: int,
    recv_buffer_rows: int,
    # Static side info (kept here rather than inside ctx because they're
    # python flags / shapes, not array leaves):
    has_wi_bias: bool,
    has_wo_bias: bool,
    has_expert_bias: bool,
    x_shape: Tuple[int, ...],
) -> dict:
    """Per-shard backward body. Returns a dict of grads keyed identically
    to the ``captured`` dict consumed by :func:`_body_fwd`."""
    if not gate_inside_vjp:
        raise NotImplementedError("gate_inside_vjp=False is deferred to a follow-up PR.")

    d_output, d_aux_loss = dy_pair
    q_set_w0, q_set_w1, q_set_wo = quantizer_sets
    batch_size, sequence_length, hidden = x_shape
    shard_id = jax.lax.axis_index(ep_axis) if ep_active else None

    # Recompute per-shard static shape info from existing statics
    # (Python ints / int tuples). Plumbed via kwargs to _combine_bwd
    # and _dispatch_bwd -- NOT through the ctx dict, because the
    # dict gets pytree-flattened across the bwd shard_map's in_specs
    # and Python ints would be coerced into JitTracer 0-d arrays
    # (breaking ``if padding > 0`` and ``jnp.zeros(shape)`` callsites).
    # ``batch_size`` here is the GLOBAL batch size (captured in
    # ``x_shape`` by the outer fwd rule), hence ``batch_is_per_shard=False``.
    _static_shape = _compute_static_shape_info(
        batch_size=batch_size,
        sequence_length=sequence_length,
        hidden=hidden,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        align_size=align_size,
        ep_active=ep_active,
        num_ep=num_ep,
        fsdp_sizes=fsdp_sizes,
        recv_buffer_rows=recv_buffer_rows,
        batch_is_per_shard=False,
    )

    # Compute per-shard input shape: under the EP shard_map body, the
    # gradient tensors live at per-shard shape, so the dispatch_bwd
    # reshape target and ``d_x_from_dispatch.reshape(x_shape)`` below
    # must use the per-shard shape rather than the captured global
    # ``x_shape``.
    if ep_active:
        import math as _math  # local import keeps the no-EP path zero-overhead.

        dp_size = _math.prod(fsdp_sizes) if fsdp_sizes else 1
        per_shard_batch = batch_size // (num_ep * dp_size)
        per_shard_x_shape: Tuple[int, ...] = (per_shard_batch, sequence_length, hidden)
    else:
        per_shard_x_shape = x_shape

    # ---------------- Combine bwd ----------------
    d_expert_outputs, d_routing_weights = _combine_bwd(
        d_output,
        ctx["dispatch"],
        ctx["expert_outputs"],
        backend=permutation_backend,
        ep_active=ep_active,
        batch_size=batch_size,
        sequence_length=sequence_length,
        dtype=dtype,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        num_real_tokens=_static_shape["num_real_tokens"],
        padding_size=_static_shape["padding_size"],
        post_a2a_buffer_shape=_static_shape["post_a2a_buffer_shape"],
        ep_axis=ep_axis,
        shard_id=shard_id,
        num_ep=num_ep,
    )

    # ---------------- FFN bwd: GEMM 3 (wo) ----------------
    casted_d_eo = tex.grouped_quantize(
        d_expert_outputs, q_set_wo.dgrad, ctx["local_group_sizes"], flatten_axis=-1
    )
    d_intermediate = tex.grouped_gemm(
        casted_d_eo.get_tensor(usage=TensorUsage.LHS),
        ctx["casted_wo_rhs_trans"],
        contracting_dims=((1,), (2,)),
    )
    d_wo = tex.grouped_gemm(
        ctx["casted_intermediate_lhs_trans"],
        casted_d_eo.get_tensor(usage=TensorUsage.RHS),
        contracting_dims=((0,), (0,)),
    )
    d_wo_bias = (
        tex.grouped_dbias(d_expert_outputs, ctx["local_group_sizes"]) if has_wo_bias else None
    )

    # ---------------- Activation bwd ----------------
    # intermediate = act(layer_w0) * layer_w1
    # d(layer_w0) = vjp(act, layer_w0)(d_intermediate * layer_w1)
    # d(layer_w1) = d_intermediate * act(layer_w0)
    act_fn = _convert_to_activation_function(activation_type)
    act_w0, dact_w0_pullback = jax.vjp(act_fn, ctx["layer_w0"])
    d_layer_w1 = d_intermediate * act_w0
    (d_layer_w0,) = dact_w0_pullback(d_intermediate * ctx["layer_w1"])

    # ---------------- FFN bwd: GEMM 2 (wi_1) ----------------
    casted_d_layer_w1 = tex.grouped_quantize(
        d_layer_w1, q_set_w1.dgrad, ctx["local_group_sizes"], flatten_axis=-1
    )
    d_sorted_x_from_w1 = tex.grouped_gemm(
        casted_d_layer_w1.get_tensor(usage=TensorUsage.LHS),
        ctx["casted_wi_1_rhs_trans"],
        contracting_dims=((1,), (2,)),
    )
    d_wi_1 = tex.grouped_gemm(
        ctx["casted_sorted_x_lhs_trans"],
        casted_d_layer_w1.get_tensor(usage=TensorUsage.RHS),
        contracting_dims=((0,), (0,)),
    )
    d_wi_1_bias = tex.grouped_dbias(d_layer_w1, ctx["local_group_sizes"]) if has_wi_bias else None

    # ---------------- FFN bwd: GEMM 1 (wi_0) ----------------
    casted_d_layer_w0 = tex.grouped_quantize(
        d_layer_w0, q_set_w0.dgrad, ctx["local_group_sizes"], flatten_axis=-1
    )
    d_sorted_x_from_w0 = tex.grouped_gemm(
        casted_d_layer_w0.get_tensor(usage=TensorUsage.LHS),
        ctx["casted_wi_0_rhs_trans"],
        contracting_dims=((1,), (2,)),
    )
    d_wi_0 = tex.grouped_gemm(
        ctx["casted_sorted_x_lhs_trans"],
        casted_d_layer_w0.get_tensor(usage=TensorUsage.RHS),
        contracting_dims=((0,), (0,)),
    )
    d_wi_0_bias = tex.grouped_dbias(d_layer_w0, ctx["local_group_sizes"]) if has_wi_bias else None

    d_sorted_x = d_sorted_x_from_w0 + d_sorted_x_from_w1

    # ---------------- Dispatch bwd ----------------
    inputs_2d_shape = (per_shard_x_shape[0] * per_shard_x_shape[1], hidden)
    d_inputs_2d = _dispatch_bwd(
        d_sorted_x,
        ctx["dispatch"],
        inputs_2d_shape=inputs_2d_shape,
        backend=permutation_backend,
        ep_active=ep_active,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        num_real_tokens=_static_shape["num_real_tokens"],
        padding_size=_static_shape["padding_size"],
        pre_a2a_buffer_shape=_static_shape["pre_a2a_buffer_shape"],
        ep_axis=ep_axis,
        shard_id=shard_id,
        num_ep=num_ep,
    )
    d_x_from_dispatch = d_inputs_2d.reshape(per_shard_x_shape)

    # ---------------- Routing bwd ----------------
    # The probs cotangent comes from _combine_bwd. For PURE_JAX it's the
    # cotangent of routing_weights (post-routing_map_to_selected_experts);
    # we need to bridge back to sparse_probs. For TRITON it's already the
    # cotangent of merging_probs == sparse_probs.
    if d_routing_weights is not None:
        if permutation_backend is PermutationBackend.PURE_JAX:
            # routing_map_to_selected_experts:
            #   selected_experts = argsort(routing_map)[..., -topk:]
            #   weights = take_along_axis(sparse_probs, selected_experts, axis=-1)
            # routing_map is bool (non-diff); the gradient of weights
            # w.r.t. sparse_probs is a scatter-into-zero along the
            # selected_experts indices.
            selected_experts = jnp.argsort(ctx["routing_map"], axis=-1)[..., -num_experts_per_tok:]
            d_sparse_probs = jnp.zeros_like(ctx["saved_scores"]).astype(d_routing_weights.dtype)
            d_sparse_probs = jnp.take_along_axis(d_sparse_probs, selected_experts, axis=-1)
            # Actually scatter: build via jnp.zeros + .at[].set
            d_sparse_probs = jnp.zeros(ctx["routing_map"].shape, dtype=d_routing_weights.dtype)
            d_sparse_probs = d_sparse_probs.at[
                jnp.arange(ctx["routing_map"].shape[0])[:, None], selected_experts
            ].set(d_routing_weights)
        else:
            d_sparse_probs = d_routing_weights.astype(jnp.float32)
    else:
        d_sparse_probs = jnp.zeros(ctx["routing_map"].shape, dtype=jnp.float32)

    # Topk bwd primitive: returns d_logits (no d_expert_bias).
    d_logits_2d_main = tex.fused_topk_with_score_function_bwd(
        ctx["routing_map"],
        ctx["saved_scores"],
        d_sparse_probs.astype(ctx["saved_scores"].dtype),
        topk=num_experts_per_tok,
        use_pre_softmax=use_pre_softmax,
        scaling_factor=scaling_factor,
        score_function=score_function,
        compute_aux_scores=False,
    )

    # ---------------- Aux loss bwd ----------------
    if aux_loss_coeff > 0.0:
        # Step 1: aux_loss bwd -> d_aux_probs
        aux_num_tokens = ctx["aux_logits_for_score"].shape[0]
        d_aux_probs = tex.fused_moe_aux_loss_bwd(
            ctx["aux_const_buf"],
            ctx["aux_tokens_per_expert"].astype(jnp.int32),
            d_aux_loss.reshape(()),
            num_tokens=aux_num_tokens,
        )
        # Step 2: aux-side topk bwd (compute_aux_scores=True path).
        # The routing_map argument is ignored in this branch (the kernel
        # uses saved_scores); pass any shape-correct integer tensor.
        d_aux_logits = tex.fused_topk_with_score_function_bwd(
            jnp.zeros(ctx["aux_logits_for_score"].shape, dtype=jnp.bool_),
            ctx["aux_saved_scores"],
            d_aux_probs.astype(ctx["aux_saved_scores"].dtype),
            topk=num_experts_per_tok,
            use_pre_softmax=False,
            scaling_factor=1.0,
            score_function=score_function,
            compute_aux_scores=True,
        )
        # Step 3: under EP the aux logits were all_gathered along
        # ``(ep_axis, *data_parallelism_axes)`` (the latter being FSDP
        # axes that shard the batch). The bwd is the inverse of that
        # multi-axis tiled all_gather: ``dynamic_slice`` to pick out
        # this shard's local rows from the global cotangent.
        #
        # JAX's convention for tiled ``all_gather(axis_name=(a, b, ...))``
        # is row-major over the tuple: the shard at mesh position
        # ``(i_a, i_b, ...)`` writes to rows
        # ``[(i_a * size_b * ... + i_b * ... + ...) * local_T :
        #    + local_T)``. We invert that by computing the same flat
        # index here and slicing.
        if ep_active:
            local_T_aux = ctx["logits_2d"].shape[0]
            flat_shard = shard_id  # ep is the outermost axis in the gather tuple
            for ax, sz in zip(data_parallelism_axes, fsdp_sizes):
                flat_shard = flat_shard * sz + jax.lax.axis_index(ax)
            d_aux_logits_local = jax.lax.dynamic_slice(
                d_aux_logits.astype(ctx["logits_2d"].dtype),
                start_indices=(flat_shard * local_T_aux, 0),
                slice_sizes=(local_T_aux, num_experts),
            )
        else:
            d_aux_logits_local = d_aux_logits.astype(d_logits_2d_main.dtype)
        d_logits_2d = d_logits_2d_main + d_aux_logits_local.astype(d_logits_2d_main.dtype)
    else:
        d_logits_2d = d_logits_2d_main

    # ---------------- Gate bwd ----------------
    d_gate_logits = d_logits_2d.reshape(per_shard_x_shape[0], per_shard_x_shape[1], num_experts)
    gate_kernel_cast = ctx["gate_kernel"].astype(ctx["x"].dtype)
    d_x_from_gate = jnp.einsum("bse,he->bsh", d_gate_logits, gate_kernel_cast)
    d_gate_kernel = jnp.einsum("bsh,bse->he", ctx["x"], d_gate_logits).astype(
        ctx["gate_kernel"].dtype
    )
    d_x = d_x_from_gate + d_x_from_dispatch

    # Reduce per-rank partial contributions to match the out_specs
    # declared by _build_grads_specs:
    #   gate_kernel  : P()              -> psum across (ep, *fsdp)
    #   wi_0/wi_1/wo : P(ep_axis, ...)  -> psum across (*fsdp) only
    #   inputs       : P((ep, fsdp), ...) -> already shard-local, no reduction
    if ep_active:
        replicate_all = (ep_axis,) + tuple(data_parallelism_axes)
        d_gate_kernel = jax.lax.psum(d_gate_kernel, axis_name=replicate_all)
        if data_parallelism_axes:
            replicate_fsdp = tuple(data_parallelism_axes)
            d_wi_0 = jax.lax.psum(d_wi_0, axis_name=replicate_fsdp)
            d_wi_1 = jax.lax.psum(d_wi_1, axis_name=replicate_fsdp)
            d_wo = jax.lax.psum(d_wo, axis_name=replicate_fsdp)
            if has_wi_bias:
                d_wi_0_bias = jax.lax.psum(d_wi_0_bias, axis_name=replicate_fsdp)
                d_wi_1_bias = jax.lax.psum(d_wi_1_bias, axis_name=replicate_fsdp)
            if has_wo_bias:
                d_wo_bias = jax.lax.psum(d_wo_bias, axis_name=replicate_fsdp)

    grads: dict = {
        "inputs": d_x,
        "gate_kernel": d_gate_kernel,
        "wi_0": d_wi_0,
        "wi_1": d_wi_1,
        "wo": d_wo,
    }
    if has_wi_bias:
        grads["wi_0_bias"] = d_wi_0_bias
        grads["wi_1_bias"] = d_wi_1_bias
    if has_wo_bias:
        grads["wo_bias"] = d_wo_bias
    if has_expert_bias:
        # expert_bias has no gradient through topk (the topk bwd returns
        # None for it). Emit a structural zero so the outer rule has
        # something to package.
        grads["expert_bias"] = jnp.zeros_like(ctx["expert_bias"])
    return grads


# =============================================================================
# Spec builders for shard_map (lockstep with ctx_dict / captured_dict)
# =============================================================================


def _build_in_specs(
    ep_axis: str,
    batch_pspec_axis: Any,
    *,
    has_bias: bool,
    has_expert_bias: bool,
) -> dict:
    """Build the ``in_specs`` dict for the EP fwd shard_map."""
    specs: dict = {
        "inputs": P(batch_pspec_axis, None, None),
        "gate_kernel": P(),
        "wi_0": P(ep_axis, None, None),
        "wi_1": P(ep_axis, None, None),
        "wo": P(ep_axis, None, None),
    }
    if has_bias:
        for name in ("wi_0_bias", "wi_1_bias", "wo_bias"):
            specs[name] = P(ep_axis, None)
    if has_expert_bias:
        specs["expert_bias"] = P(ep_axis)
    return specs


def _build_dispatch_specs(
    ep_axis: str,
    *,
    backend: PermutationBackend,
    ep_active: bool,
) -> dict:
    """Build the spec dict for a DispatchState dict returned by
    :func:`_dispatch` from inside a shard_map. Keys must match what
    :func:`_dispatch` actually populates for the given (backend, ep_active)."""
    specs: dict = {"group_sizes": P()}
    if backend is PermutationBackend.PURE_JAX:
        specs["sorted_indices"] = P()
        specs["routing_weights"] = P()
    else:
        specs["row_id_map"] = P()
        specs["pad_offsets"] = P()
        specs["merging_probs"] = P()
    if ep_active:
        specs["all_shards_tokens_per_expert"] = P()
        specs["local_perm_row_id_map"] = P()
    # NOTE: per-shard compile-time-constant shape info
    # (num_real_tokens, padding_size, pre/post_a2a_buffer_shape)
    # is intentionally NOT in the state dict; see _compute_static_shape_info.
    return specs


def _build_ctx_specs(
    ep_axis: str,
    batch_pspec_axis: Any,
    *,
    backend: PermutationBackend,
    ep_active: bool,
    has_bias: bool,
    has_expert_bias: bool,
    aux_loss_enabled: bool,
) -> dict:
    """Build the spec dict for the ``ctx`` returned by :func:`_body_fwd`."""
    specs: dict = {
        # Per-shard local activations along the batch axis.
        "x": P(batch_pspec_axis, None, None),
        "gate_kernel": P(),
        "logits_2d": P(batch_pspec_axis, None),
        "saved_scores": P(batch_pspec_axis, None),
        "routing_map": P(batch_pspec_axis, None),
        "dispatch": _build_dispatch_specs(ep_axis, backend=backend, ep_active=ep_active),
        # FFN residuals: the LHS_TRANS / RHS_TRANS variants of
        # grouped_quantize have leading "rows"/"experts" dims that are
        # already shard-local (post-dispatch). Use P(ep_axis,...) on
        # leading dim; that works whether the leaf is a plain ndarray
        # or a ScaledTensor (shard_map applies the spec leaf-wise to
        # the registered ScaledTensor pytree).
        "casted_sorted_x_lhs_trans": P(),
        "casted_wi_0_rhs_trans": P(ep_axis, None, None),
        "casted_wi_1_rhs_trans": P(ep_axis, None, None),
        "layer_w0": P(),
        "layer_w1": P(),
        "casted_intermediate_lhs_trans": P(),
        "casted_wo_rhs_trans": P(ep_axis, None, None),
        "expert_outputs": P(),
        "local_group_sizes": P(),
    }
    if has_expert_bias:
        specs["expert_bias"] = P(ep_axis)
    if aux_loss_enabled:
        specs["aux_const_buf"] = P()
        specs["aux_tokens_per_expert"] = P()
        specs["aux_logits_for_score"] = P()
        specs["aux_saved_scores"] = P()
    return specs


def _build_grads_specs(
    ep_axis: str,
    batch_pspec_axis: Any,
    *,
    has_bias: bool,
    has_expert_bias: bool,
) -> dict:
    """Spec dict for the grads dict returned by :func:`_body_bwd`."""
    return _build_in_specs(
        ep_axis,
        batch_pspec_axis,
        has_bias=has_bias,
        has_expert_bias=has_expert_bias,
    )


# =============================================================================
# Top-level VJP rules
# =============================================================================


def _moe_fwd_rule(
    # IMPORTANT — calling convention for jax.custom_vjp fwd rule.
    #
    # JAX uses ``_argnums_partial`` (jax/_src/api_util.py) when wiring up
    # the fwd rule. That helper preserves the ORIGINAL positional order
    # of the decorated function: dyn (= diff) args sit at their original
    # positions and static (= nondiff) args fill the remaining slots in
    # nondiff_argnums order. So the fwd rule MUST take args in the
    # SAME positional order as ``_moe`` -- diff first (positions 0..8),
    # then nondiff (positions 9..28), all POSITIONAL (no ``*,`` -- they
    # arrive as positional, not as kwargs).
    #
    # NOTE: this is the OPPOSITE convention from ``_moe_bwd_rule``, which
    # uses ``prepend_static_args`` -- there the static args come FIRST,
    # followed by ``ctx`` and ``dy_pair``.
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
    permutation_backend,
    align_size,
    gate_inside_vjp,
    ep_axis,
    data_parallelism_axes,
    input_axes,
    gate_kernel_axes,
    wi_kernel_axes,
    wo_kernel_axes,
    quantizer_sets,
    dtype,
):
    x = with_sharding_constraint_by_logical_axes(x, input_axes)
    ep_active = ep_axis is not None
    body_kwargs = dict(
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        activation_type=activation_type,
        score_function=score_function,
        use_pre_softmax=use_pre_softmax,
        num_groups=num_groups,
        group_topk=group_topk,
        scaling_factor=scaling_factor,
        aux_loss_coeff=aux_loss_coeff,
        permutation_backend=permutation_backend,
        align_size=align_size,
        gate_inside_vjp=gate_inside_vjp,
        quantizer_sets=quantizer_sets,
        dtype=dtype,
        ep_axis=ep_axis,
        data_parallelism_axes=data_parallelism_axes,
    )
    captured: dict = {
        "inputs": x,
        "gate_kernel": gate_kernel,
        "wi_0": wi_0,
        "wi_1": wi_1,
        "wo": wo,
    }
    has_bias = wi_0_bias is not None
    has_expert_bias = expert_bias is not None
    if has_bias:
        captured["wi_0_bias"] = wi_0_bias
        captured["wi_1_bias"] = wi_1_bias
        captured["wo_bias"] = wo_bias
    if has_expert_bias:
        captured["expert_bias"] = expert_bias

    if not ep_active:
        output, aux_loss, ctx = _body_fwd(
            captured,
            **body_kwargs,
            ep_active=False,
            fsdp_sizes=(),
            num_ep=1,
            num_experts_local=num_experts,
            recv_buffer_rows=0,
        )
        # Carry static side info into ctx for the bwd rule (as Python
        # objects on the dict; not part of the tree pytree leaves).
        ctx["__static__"] = dict(
            has_wi_bias=has_bias,
            has_wo_bias=has_bias,
            has_expert_bias=has_expert_bias,
            x_shape=x.shape,
            num_experts_local=num_experts,
            recv_buffer_rows=0,
        )
        return (output, aux_loss), ctx

    # ---------------- EP path ----------------
    from jax.experimental.shard_map import shard_map

    mesh = _get_mesh()
    if mesh is None or mesh.empty:
        raise ValueError("moe(...) requires an active jax.sharding.Mesh when ep_axis is set.")
    num_ep = mesh.shape[ep_axis]
    if num_experts % num_ep != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by EP size={num_ep}")
    num_experts_local = num_experts // num_ep

    if not data_parallelism_axes:
        batch_pspec_axis: Any = ep_axis
    else:
        batch_pspec_axis = (ep_axis, *data_parallelism_axes)
    dp_size = 1
    for ax in data_parallelism_axes:
        dp_size *= mesh.shape[ax]

    global_batch_size, sequence_length, _hidden = x.shape
    topk = num_experts_per_tok
    if global_batch_size % (num_ep * dp_size) != 0:
        raise ValueError(f"batch={global_batch_size} not divisible by ep*dp={num_ep * dp_size}")
    recv_buffer_rows = (global_batch_size // dp_size) * sequence_length * topk
    if align_size > 0:
        recv_buffer_rows += num_experts * (align_size - 1)

    in_specs = _build_in_specs(
        ep_axis,
        batch_pspec_axis,
        has_bias=has_bias,
        has_expert_bias=has_expert_bias,
    )
    output_spec = P(batch_pspec_axis, None, None)
    aux_spec = P()
    ctx_spec = _build_ctx_specs(
        ep_axis,
        batch_pspec_axis,
        backend=permutation_backend,
        ep_active=True,
        has_bias=has_bias,
        has_expert_bias=has_expert_bias,
        aux_loss_enabled=(aux_loss_coeff > 0.0),
    )

    _fsdp_sizes: Tuple[int, ...] = tuple(mesh.shape[ax] for ax in data_parallelism_axes)

    def _shardmap_body(captured_local):
        return _body_fwd(
            captured_local,
            **body_kwargs,
            ep_active=True,
            fsdp_sizes=_fsdp_sizes,
            num_ep=num_ep,
            num_experts_local=num_experts_local,
            recv_buffer_rows=recv_buffer_rows,
        )

    output, aux_loss, ctx = shard_map(
        _shardmap_body,
        mesh=mesh,
        in_specs=(in_specs,),
        out_specs=(output_spec, aux_spec, ctx_spec),
        check_rep=False,
    )(captured)
    ctx["__static__"] = dict(
        has_wi_bias=has_bias,
        has_wo_bias=has_bias,
        has_expert_bias=has_expert_bias,
        x_shape=x.shape,
        num_experts_local=num_experts_local,
        recv_buffer_rows=recv_buffer_rows,
    )
    return (output, aux_loss), ctx


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
    permutation_backend,
    align_size,
    gate_inside_vjp,
    ep_axis,
    data_parallelism_axes,
    input_axes,
    gate_kernel_axes,
    wi_kernel_axes,
    wo_kernel_axes,
    quantizer_sets,
    dtype,
    ctx,
    dy_pair,
):
    static = ctx.pop("__static__")
    has_wi_bias = static["has_wi_bias"]
    has_wo_bias = static["has_wo_bias"]
    has_expert_bias = static["has_expert_bias"]
    x_shape = static["x_shape"]
    num_experts_local = static["num_experts_local"]
    recv_buffer_rows = static["recv_buffer_rows"]

    ep_active = ep_axis is not None
    mesh = _get_mesh() if ep_active else None
    fsdp_sizes: Tuple[int, ...] = (
        tuple(mesh.shape[ax] for ax in data_parallelism_axes) if ep_active else ()
    )
    body_kwargs = dict(
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        activation_type=activation_type,
        score_function=score_function,
        use_pre_softmax=use_pre_softmax,
        num_groups=num_groups,
        group_topk=group_topk,
        scaling_factor=scaling_factor,
        aux_loss_coeff=aux_loss_coeff,
        permutation_backend=permutation_backend,
        align_size=align_size,
        gate_inside_vjp=gate_inside_vjp,
        quantizer_sets=quantizer_sets,
        dtype=dtype,
        ep_axis=ep_axis,
        data_parallelism_axes=data_parallelism_axes,
        fsdp_sizes=fsdp_sizes,
        num_ep=1 if not ep_active else mesh.shape[ep_axis],
        num_experts_local=num_experts_local,
        recv_buffer_rows=recv_buffer_rows,
        has_wi_bias=has_wi_bias,
        has_wo_bias=has_wo_bias,
        has_expert_bias=has_expert_bias,
        x_shape=x_shape,
    )

    if not ep_active:
        grads = _body_bwd(ctx, dy_pair, ep_active=False, **body_kwargs)
        # Apply sharding constraints on grads.
        grads["gate_kernel"] = with_sharding_constraint_by_logical_axes(
            grads["gate_kernel"], gate_kernel_axes
        )
        grads["wi_0"] = with_sharding_constraint_by_logical_axes(grads["wi_0"], wi_kernel_axes)
        grads["wi_1"] = with_sharding_constraint_by_logical_axes(grads["wi_1"], wi_kernel_axes)
        grads["wo"] = with_sharding_constraint_by_logical_axes(grads["wo"], wo_kernel_axes)
        grads["inputs"] = with_sharding_constraint_by_logical_axes(grads["inputs"], input_axes)
        return _grads_dict_to_tuple(grads, has_wi_bias, has_wo_bias, has_expert_bias)

    from jax.experimental.shard_map import shard_map

    if not data_parallelism_axes:
        batch_pspec_axis: Any = ep_axis
    else:
        batch_pspec_axis = (ep_axis, *data_parallelism_axes)
    ctx_spec = _build_ctx_specs(
        ep_axis,
        batch_pspec_axis,
        backend=permutation_backend,
        ep_active=True,
        has_bias=has_wi_bias,
        has_expert_bias=has_expert_bias,
        aux_loss_enabled=(aux_loss_coeff > 0.0),
    )
    dy_specs = (P(batch_pspec_axis, None, None), P())
    grads_spec = _build_grads_specs(
        ep_axis, batch_pspec_axis, has_bias=has_wi_bias, has_expert_bias=has_expert_bias
    )

    def _bwd_body(ctx_local, dy_local):
        return _body_bwd(ctx_local, dy_local, ep_active=True, **body_kwargs)

    grads = shard_map(
        _bwd_body,
        mesh=mesh,
        in_specs=(ctx_spec, dy_specs),
        out_specs=grads_spec,
        check_rep=False,
    )(ctx, dy_pair)

    grads["gate_kernel"] = with_sharding_constraint_by_logical_axes(
        grads["gate_kernel"], gate_kernel_axes
    )
    grads["wi_0"] = with_sharding_constraint_by_logical_axes(grads["wi_0"], wi_kernel_axes)
    grads["wi_1"] = with_sharding_constraint_by_logical_axes(grads["wi_1"], wi_kernel_axes)
    grads["wo"] = with_sharding_constraint_by_logical_axes(grads["wo"], wo_kernel_axes)
    grads["inputs"] = with_sharding_constraint_by_logical_axes(grads["inputs"], input_axes)
    return _grads_dict_to_tuple(grads, has_wi_bias, has_wo_bias, has_expert_bias)


def _grads_dict_to_tuple(
    grads: dict, has_wi_bias: bool, has_wo_bias: bool, has_expert_bias: bool
) -> Tuple:
    """Pack the body_bwd's grads dict into the positional tuple JAX expects."""
    return (
        grads["inputs"],
        grads["gate_kernel"],
        grads["wi_0"],
        grads["wi_1"],
        grads["wo"],
        grads.get("wi_0_bias") if has_wi_bias else None,
        grads.get("wi_1_bias") if has_wi_bias else None,
        grads.get("wo_bias") if has_wo_bias else None,
        grads.get("expert_bias") if has_expert_bias else None,
    )


# =============================================================================
# custom_vjp + public entry
# =============================================================================


@partial(jax.custom_vjp, nondiff_argnums=tuple(range(9, 29)))
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
    permutation_backend,
    align_size,
    gate_inside_vjp,
    ep_axis,
    data_parallelism_axes,
    input_axes,
    gate_kernel_axes,
    wi_kernel_axes,
    wo_kernel_axes,
    quantizer_sets,
    dtype,
):
    # Call in `_moe`'s own signature order to match what JAX will pass
    # the fwd rule via ``_argnums_partial``. See the comment block at
    # the top of ``_moe_fwd_rule`` for why this differs from
    # ``_moe_bwd_rule``'s convention.
    output_pair, _ = _moe_fwd_rule(
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
        permutation_backend,
        align_size,
        gate_inside_vjp,
        ep_axis,
        data_parallelism_axes,
        input_axes,
        gate_kernel_axes,
        wi_kernel_axes,
        wo_kernel_axes,
        quantizer_sets,
        dtype,
    )
    return output_pair


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
    # Architecture
    num_experts: int,
    num_experts_per_tok: int,
    activation_type: str = "silu",
    # Routing
    score_function: Union[str, ScoreFunction] = "softmax",
    use_pre_softmax: bool = False,
    num_groups: Optional[int] = None,
    group_topk: Optional[int] = None,
    scaling_factor: float = 1.0,
    aux_loss_coeff: float = 0.0,
    # Permutation
    permutation_backend: PermutationBackend = PermutationBackend.PURE_JAX,
    align_size: int = 0,
    # Gate placement (Phuong: "perhaps as an option")
    gate_inside_vjp: bool = True,
    # Parallelism (resolved by caller from MeshResource)
    ep_axis: Optional[str] = None,
    data_parallelism_axes: Tuple[str, ...] = (),
    # Logical axes for sharding constraints
    input_axes: Tuple[Optional[str], ...] = (),
    gate_kernel_axes: Tuple[Optional[str], ...] = (),
    wi_kernel_axes: Tuple[Optional[str], ...] = ("exp", "embed", "mlp"),
    wo_kernel_axes: Tuple[Optional[str], ...] = ("exp", "mlp", "embed"),
    # Quantization
    quantizer_sets: Tuple[QuantizerSet, QuantizerSet, QuantizerSet] = (
        noop_quantizer_set,
        noop_quantizer_set,
        noop_quantizer_set,
    ),
    dtype: jnp.dtype = jnp.float32,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Run a full MoE block under a single fused custom_vjp.

    Parameters and return are documented at the call site of
    ``_MoEBlock.__call__``. See module docstring for design rationale.
    """
    if not isinstance(permutation_backend, PermutationBackend):
        raise TypeError(
            f"permutation_backend must be a PermutationBackend, got {permutation_backend!r}"
        )
    # Normalize string score_function ("softmax" / "sigmoid") to the
    # ScoreFunction enum once here. The underlying primitive
    # ``tex.fused_topk_with_score_function_fwd`` expects an int-coercible
    # value (the enum has integer .value), and the public router wrapper
    # we bypass also normalizes here.
    score_function = _validate_score_function(score_function)

    output, aux_loss = _moe(
        x,
        gate_kernel,
        wi_0,
        wi_1,
        wo,
        wi_0_bias,
        wi_1_bias,
        wo_bias,
        expert_bias,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        activation_type=activation_type,
        score_function=score_function,
        use_pre_softmax=use_pre_softmax,
        num_groups=num_groups,
        group_topk=group_topk,
        scaling_factor=scaling_factor,
        aux_loss_coeff=aux_loss_coeff,
        permutation_backend=permutation_backend,
        align_size=align_size,
        gate_inside_vjp=gate_inside_vjp,
        ep_axis=ep_axis,
        data_parallelism_axes=data_parallelism_axes,
        input_axes=input_axes,
        gate_kernel_axes=gate_kernel_axes,
        wi_kernel_axes=wi_kernel_axes,
        wo_kernel_axes=wo_kernel_axes,
        quantizer_sets=quantizer_sets,
        dtype=dtype,
    )
    if aux_loss_coeff <= 0.0:
        aux_loss = None
    return output, aux_loss
