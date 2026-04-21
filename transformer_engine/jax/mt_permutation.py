# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Pure-JAX MoE Permutation API.

This module provides a MaxText-style, pure-JAX implementation of MoE token
dispatch / combine as an alternative to the Triton-backed primitives in
``transformer_engine.jax.permutation``. Empirically this path has been faster
than the Triton kernels on several E2E workloads.

The core design mirrors Maxtext's ``_mt_permute`` / ``_mt_unpermute`` in
``maxtext/src/maxtext/layers/moe.py``, with alignment-padding support ported
from `nvjax-svc-0/maxtext PR #36 <https://github.com/nvjax-svc-0/maxtext/pull/36/changes>`_
so each expert's group size is a multiple of ``align_size`` (required for
quantized grouped GEMM whose recipe-specific alignment must divide
``align_size``).

When ``align_size = 0`` padding is disabled (faster for the unquantized path);
when ``align_size > 0`` a static-size padding buffer of shape
``[num_experts * (align_size - 1)]`` is appended before the sort so the overall
shape is JIT-compatible.

The public API is:

* :func:`mt_token_dispatch` -- pure-JAX counterpart of ``token_dispatch``.
* :func:`mt_token_combine` -- pure-JAX counterpart of ``token_combine``.
* :class:`MTPermState` -- opaque state returned by ``mt_token_dispatch`` and
  consumed by ``mt_token_combine``.
"""

from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

__all__ = [
    "MTPermState",
    "mt_token_dispatch",
    "mt_token_combine",
]


# =============================================================================
# Custom-VJP argsort-based gather (``_sort_activations_custom``)
# =============================================================================
#
# ``inputs[sort_indices]`` has a known inverse: ``output[argsort(sort_indices)]``.
# Using a custom VJP lets the backward pass exploit that inverse instead of
# relying on the compiler to discover it from the scatter-style default
# gradient of a gather, which is typically less efficient.


@jax.custom_vjp
def _sort_activations_custom(inputs: jax.Array, sort_indices: jax.Array) -> jax.Array:
    """Sort ``inputs`` along the leading dim by ``sort_indices``."""
    return inputs[sort_indices, ...]


def _sort_activations_custom_fwd(
    inputs: jax.Array, sort_indices: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    return _sort_activations_custom(inputs, sort_indices), sort_indices


def _sort_activations_custom_bwd(
    residuals: jax.Array, grads: jax.Array
) -> Tuple[jax.Array, None]:
    sort_indices = residuals
    # Inverse permutation: gather-by-argsort undoes the forward gather.
    return _sort_activations_custom(grads, jnp.argsort(sort_indices)), None


_sort_activations_custom.defvjp(_sort_activations_custom_fwd, _sort_activations_custom_bwd)


def _sort_activations(
    inputs: jax.Array,
    sort_indices: jax.Array,
    use_custom_vjp: bool,
) -> jax.Array:
    """Sort activations by ``sort_indices``, optionally with the custom VJP."""
    assert inputs.shape[0] == sort_indices.shape[0], (
        f"inputs.shape[0]={inputs.shape[0]} must match"
        f" sort_indices.shape[0]={sort_indices.shape[0]}"
    )
    with jax.named_scope("mt_sort_activations"):
        if use_custom_vjp:
            return _sort_activations_custom(inputs, sort_indices)
        return inputs[sort_indices, ...]


# =============================================================================
# Permutation state carried from dispatch to combine
# =============================================================================


class MTPermState(NamedTuple):
    """Opaque state produced by :func:`mt_token_dispatch`.

    Attributes
    ----------
    sorted_indices : jnp.ndarray
        The argsort indices used in the forward sort. Needed to reverse the
        permutation in :func:`mt_token_combine`. Shape
        ``[num_real_tokens + padding_size]``.
    num_real_tokens : int
        Number of real (non-padding) permuted tokens, i.e.
        ``batch_size * sequence_length * num_experts_per_tok``. Compile-time
        constant.
    padding_size : int
        Number of alignment-padding tokens appended to the sort buffer. Equals
        ``num_experts * (align_size - 1)`` when ``align_size > 0``, else ``0``.
        Compile-time constant.
    """

    sorted_indices: jax.Array
    num_real_tokens: int
    padding_size: int


# =============================================================================
# Dispatch (permute)
# =============================================================================


def mt_token_dispatch(
    inputs: jnp.ndarray,
    selected_experts: jnp.ndarray,
    num_experts: int,
    num_experts_per_tok: int,
    align_size: int = 0,
    roll_to_expert_id: Optional[int] = None,
    use_custom_sort_vjp: bool = True,
) -> Tuple[jnp.ndarray, MTPermState, jnp.ndarray]:
    """Pure-JAX MaxText-style token dispatch.

    Parameters
    ----------
    inputs : jnp.ndarray
        Input tensor of shape ``[num_tokens, hidden_size]`` (or
        ``[batch, seq, hidden]``; it will be flattened).
    selected_experts : jnp.ndarray
        Per-token expert IDs, shape ``[num_tokens, num_experts_per_tok]`` (or
        ``[batch, seq, num_experts_per_tok]``). Integer dtype.
    num_experts : int
        Total number of experts.
    num_experts_per_tok : int
        Top-k. Must equal ``selected_experts.shape[-1]``.
    align_size : int, default 0
        Alignment for each expert's group size. ``0`` disables padding; a value
        ``> 0`` appends a static-size padding buffer so each resulting group
        size is a multiple of ``align_size``.
    roll_to_expert_id : Optional[int]
        If provided, rotates expert IDs by ``-roll_to_expert_id`` modulo
        ``num_experts`` before the sort (ring-of-experts EP). The returned
        ``group_sizes`` is rolled to match.
    use_custom_sort_vjp : bool, default True
        Whether to use the custom-VJP argsort gather for the sort.

    Returns
    -------
    sorted_inputs : jnp.ndarray
        Permuted tokens grouped by expert, shape
        ``[num_real_tokens + padding_size, hidden_size]``.
    perm_state : MTPermState
        State needed by :func:`mt_token_combine`.
    group_sizes : jnp.ndarray
        Token count per expert, shape ``[num_experts]``. Each entry is a
        multiple of ``align_size`` when ``align_size > 0``.
    """
    assert num_experts_per_tok == selected_experts.shape[-1], (
        f"num_experts_per_tok={num_experts_per_tok} must match"
        f" selected_experts.shape[-1]={selected_experts.shape[-1]}"
    )
    assert align_size >= 0, f"align_size must be >= 0, got {align_size}"

    hidden_size = inputs.shape[-1]
    # Flatten token dims.
    inputs_2d = inputs.reshape(-1, hidden_size)
    num_tokens = inputs_2d.shape[0]
    num_real_tokens = num_tokens * num_experts_per_tok

    flatten_selected_experts = jnp.ravel(selected_experts)

    if align_size > 0:
        # Per-expert token count, and how many extra tokens each expert needs
        # to become aligned to ``align_size``. Using
        # ``(align - count % align) % align`` gives 0 (not ``align``) when
        # already aligned, so we never exceed the per-expert slot capacity of
        # ``align_size - 1``.
        token_count_per_expert = jnp.bincount(
            flatten_selected_experts, length=num_experts
        )
        padding_tokens_required_per_expert = (
            (align_size - (token_count_per_expert % align_size)) % align_size
        )

        # Build a static-size padding buffer of shape
        # ``[num_experts * (align_size - 1)]``. Each expert ``i`` owns a slot
        # of ``align_size - 1`` positions (worst-case padding, which occurs
        # when ``token_count[i] % align_size == 1``). Within slot ``i``,
        # positions ``[0, padding_needed)`` are assigned expert ``i`` and act
        # as real padding; the rest are assigned to ``num_experts - 1`` as
        # overflow placeholders that keep the buffer statically sized for JIT.
        max_padding_per_expert = align_size - 1
        max_total_padding_size = num_experts * max_padding_per_expert
        positions = jnp.arange(max_total_padding_size)
        expert_for_pos = positions // max_padding_per_expert
        offset_in_slot = positions % max_padding_per_expert
        padding_needed = padding_tokens_required_per_expert[expert_for_pos]
        flatten_padding_selected_experts = jnp.where(
            offset_in_slot < padding_needed,
            expert_for_pos,
            num_experts - 1,
        )

        flatten_selected_experts = jnp.concatenate(
            [flatten_selected_experts, flatten_padding_selected_experts], axis=0
        )

        if roll_to_expert_id is not None:
            flatten_selected_experts = (
                flatten_selected_experts - roll_to_expert_id
            ) % num_experts

        sorted_selected_experts = jnp.argsort(flatten_selected_experts)

        replicated_inputs_2d = jnp.repeat(inputs_2d, num_experts_per_tok, axis=0)
        # Pad inputs with zeros so the sort operand shape matches the expanded
        # selected-experts vector.
        replicated_inputs_2d = jnp.pad(
            replicated_inputs_2d,
            pad_width=((0, max_total_padding_size), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )

        sorted_inputs = _sort_activations(
            replicated_inputs_2d, sorted_selected_experts, use_custom_sort_vjp
        )

        # Compute ``group_sizes`` directly from counts rather than via
        # ``bincount(flatten_selected_experts)``: the overflow placeholder
        # tokens would inflate ``group_sizes[num_experts - 1]``, breaking the
        # alignment guarantee. Direct computation gives each expert exactly
        # ``ceil(count / align) * align`` tokens.
        group_sizes = token_count_per_expert + padding_tokens_required_per_expert

        if roll_to_expert_id is not None:
            group_sizes = jnp.roll(group_sizes, -roll_to_expert_id)

        padding_size = max_total_padding_size
    else:
        if roll_to_expert_id is not None:
            flatten_selected_experts = (
                flatten_selected_experts - roll_to_expert_id
            ) % num_experts

        sorted_selected_experts = jnp.argsort(flatten_selected_experts)

        replicated_inputs_2d = jnp.repeat(inputs_2d, num_experts_per_tok, axis=0)
        sorted_inputs = _sort_activations(
            replicated_inputs_2d, sorted_selected_experts, use_custom_sort_vjp
        )

        group_sizes = jnp.bincount(flatten_selected_experts, length=num_experts)
        if roll_to_expert_id is not None:
            group_sizes = jnp.roll(group_sizes, -roll_to_expert_id)

        padding_size = 0

    perm_state = MTPermState(
        sorted_indices=sorted_selected_experts,
        num_real_tokens=num_real_tokens,
        padding_size=padding_size,
    )
    return sorted_inputs, perm_state, group_sizes


# =============================================================================
# Combine (unpermute + weighted sum)
# =============================================================================


def mt_token_combine(
    expert_outputs: jnp.ndarray,
    perm_state: MTPermState,
    routing_weights: jnp.ndarray,
    num_experts_per_tok: int,
    batch_size: int,
    sequence_length: int,
    use_custom_sort_vjp: bool = True,
) -> jnp.ndarray:
    """Pure-JAX MaxText-style token combine.

    Reverses the permutation performed by :func:`mt_token_dispatch`, strips
    any alignment-padding rows appended during dispatch, and applies a
    per-token weighted sum across the top-k experts.

    Parameters
    ----------
    expert_outputs : jnp.ndarray
        Output of the expert FFN, shape
        ``[num_real_tokens + padding_size, hidden_size]``.
    perm_state : MTPermState
        State returned by :func:`mt_token_dispatch`.
    routing_weights : jnp.ndarray
        Top-k routing weights, shape ``[batch*seq, num_experts_per_tok]``
        (or broadcastable to it after a ``reshape``).
    num_experts_per_tok : int
        Top-k.
    batch_size : int
        Original batch size.
    sequence_length : int
        Original sequence length.
    use_custom_sort_vjp : bool, default True
        Whether to use the custom-VJP argsort gather for the unsort.

    Returns
    -------
    output : jnp.ndarray
        Combined output tensor of shape ``[batch_size, sequence_length, hidden_size]``.
    """
    # Reverse the permutation: ``output[argsort(sorted_indices)]`` undoes
    # ``input[sorted_indices]``.
    unsort_intermediate = _sort_activations(
        expert_outputs,
        jnp.argsort(perm_state.sorted_indices),
        use_custom_sort_vjp,
    )

    # Strip alignment padding tokens appended during dispatch. After unsorting,
    # the first ``num_real_tokens`` rows hold the real per-(token, top-k)
    # outputs; any trailing rows are padding placeholders (zeros) and must be
    # discarded before the reshape below.
    if perm_state.padding_size > 0:
        unsort_intermediate = unsort_intermediate[: perm_state.num_real_tokens]

    hidden_size = unsort_intermediate.shape[-1]
    reshaped_weights = jnp.reshape(routing_weights, (-1, num_experts_per_tok))
    reshaped_intermediate = jnp.reshape(
        unsort_intermediate, (reshaped_weights.shape[0], num_experts_per_tok, hidden_size)
    )

    # Cast weights to match intermediate dtype (weighted sum happens in
    # intermediate dtype; callers can upcast before calling if higher
    # precision weight-sum is desired).
    reshaped_weights = reshaped_weights.astype(reshaped_intermediate.dtype)
    with jax.named_scope("mt_weight_sum"):
        output = jnp.einsum(
            "BKE,BK -> BE",
            reshaped_intermediate,
            reshaped_weights,
        )
    return output.reshape(batch_size, sequence_length, hidden_size)
