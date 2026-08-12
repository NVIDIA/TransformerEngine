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
* The EP, grouped-quantize, and grouped-GEMM primitives operate at global
  view. Their custom partitioning rules handle per-shard execution,
  including EP placement and DP/FSDP gathers and reductions.

FC1 and FC2 use independent quantizer sets. The sets are differentiable
``custom_vjp`` arguments and are returned by the backward rule so
stateful recipes follow the same update semantics as the other TE MLPs.
``aux_loss_coeff`` and ``expert_bias`` are also supported.
"""

import math
import os
import sys
import warnings
from functools import partial
from typing import Any, Optional, Tuple, Union

import flax.struct
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from . import cpp_extensions as tex
from .quantize import (
    QuantizerSet,
    TensorUsage,
    noop_quantizer_set,
    with_sharding_constraint_by_logical_axes,
)
from .flax.module import _convert_to_activation_function
from .router import ScoreFunction, _validate_score_function
from .sharding import _get_mesh

__all__ = ["get_moe_recv_capacity_per_rank", "moe"]


# Per-expert dispatch-slot alignment fed to ``tex.ep_prepare`` as
# ``dispatch_output_per_expert_alignment``. NCCL EP HT requires the
# per-expert recv block to be at least 128-token aligned, and all current
# TE grouped-GEMM recipes (bf16/fp16/fp8/mxfp8) are satisfied by the
# same 128-token tile, so a single constant covers every supported path.
_ALIGN_SIZE = 128


def get_moe_recv_capacity_per_rank(
    *,
    num_experts: int,
    num_experts_per_tok: int,
    max_tokens_per_rank: int,
    ep_size: int,
    recv_capacity_factor: Optional[float] = None,
    alignment: int = _ALIGN_SIZE,
) -> int:
    """Return the aligned receive capacity for one EP rank.

    ``recv_capacity_factor=None`` reserves the dropless worst case. A finite
    factor >= 1 scales the capacity needed by perfectly balanced routing and
    is capped at the worst case. The balanced baseline includes the independent
    per-local-expert alignment required by NCCL EP.
    """
    if num_experts <= 0 or num_experts_per_tok <= 0 or max_tokens_per_rank <= 0:
        raise ValueError("num_experts, num_experts_per_tok, and max_tokens_per_rank must be positive")
    if ep_size <= 0 or num_experts % ep_size != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by ep_size={ep_size}")
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    if recv_capacity_factor is not None:
        recv_capacity_factor = float(recv_capacity_factor)
        if not math.isfinite(recv_capacity_factor) or recv_capacity_factor < 1.0:
            raise ValueError(
                "recv_capacity_factor must be finite and >= 1.0, or None for worst-case capacity; "
                f"got {recv_capacity_factor}"
            )

    num_local_experts = num_experts // ep_size
    tokens_per_ep_group = ep_size * max_tokens_per_rank
    max_local_assignments = tokens_per_ep_group * min(
        num_experts_per_tok, num_local_experts
    )
    max_nonempty_experts = min(num_local_experts, max_local_assignments)
    padded_total_bound = max_local_assignments + (alignment - 1) * max_nonempty_experts
    aligned_total_bound = (
        (padded_total_bound + alignment - 1) // alignment
    ) * alignment
    per_expert_bound = (
        num_local_experts
        * ((tokens_per_ep_group + alignment - 1) // alignment)
        * alignment
    )
    worst_case = min(per_expert_bound, aligned_total_bound)
    if recv_capacity_factor is None:
        return worst_case

    balanced_per_expert = (
        max_tokens_per_rank * num_experts_per_tok + num_local_experts - 1
    ) // num_local_experts
    balanced_aligned = (
        num_local_experts
        * ((balanced_per_expert + alignment - 1) // alignment)
        * alignment
    )
    requested = math.ceil(balanced_aligned * recv_capacity_factor)
    requested = ((requested + alignment - 1) // alignment) * alignment
    return min(requested, worst_case)


_debug_python_patch = os.getenv("NVTE_DEBUG_PYTHON_PATCH", "0") == "1"
_debug_moe_numerics = os.getenv("NVTE_DEBUG_MOE_NUMERICS", "0") == "1"
_debug_moe_input_grad = os.getenv("NVTE_DEBUG_MOE_INPUT_GRAD", "0") == "1"
_use_reference_fwd = os.getenv("NVTE_MOE_REFERENCE_FWD", "0") == "1"
_use_reference_dgrad = os.getenv("NVTE_MOE_REFERENCE_DGRAD", "0") == "1"
_zero_dispatch_weight_grad = os.getenv("NVTE_MOE_ZERO_DISPATCH_WEIGHT_GRAD", "0") == "1"
_refresh_ep_handle_in_bwd = os.getenv("NVTE_MOE_REFRESH_EP_HANDLE_IN_BWD", "0") == "1"
_refresh_ep_handle_before_combine = (
    os.getenv("NVTE_MOE_REFRESH_EP_HANDLE_BEFORE_COMBINE", "0") == "1"
)
_validate_ep_routing = os.getenv("NVTE_MOE_VALIDATE_EP_ROUTING", "0") == "1"
_debug_moe_fwd_recompute = os.getenv("NVTE_MOE_DEBUG_FWD_RECOMPUTE", "0") == "1"
_validate_ep_token_roundtrip = (
    os.getenv("NVTE_MOE_VALIDATE_EP_TOKEN_ROUNDTRIP", "0") == "1"
)
_zero_moe_input_grad = os.getenv("NVTE_MOE_ZERO_INPUT_GRAD", "0") == "1"
_skip_moe_backward = os.getenv("NVTE_MOE_SKIP_BACKWARD", "0") == "1"
_debug_handle_mem = os.getenv("NVTE_MOE_DEBUG_HANDLE_MEM", "0") == "1"
_validate_ep_forward_roundtrip = (
    os.getenv("NVTE_MOE_VALIDATE_EP_FORWARD_ROUNDTRIP", "0") == "1"
)
_zero_moe_output = os.getenv("NVTE_MOE_ZERO_OUTPUT", "0") == "1"
_skip_moe_forward = os.getenv("NVTE_MOE_SKIP_FORWARD", "0") == "1"
_EP_ROUTING_PROBE_WIDTH = 16
_debug_ffn_fwd_global_count = 0
_debug_reference_dgrad_count = 0
if _debug_python_patch:
    print(
        "[TE patch debug] imported transformer_engine.jax.moe "
        f"from {__file__} (rank={os.getenv('SLURM_PROCID', 'unknown')}, "
        f"reference_fwd={_use_reference_fwd}, reference_dgrad={_use_reference_dgrad})",
        file=sys.stderr,
        flush=True,
    )
if _use_reference_dgrad:
    print(
        "[TE reference dgrad] enabled: activation dgrad uses jax.lax.ragged_dot; "
        "forward and weight gradients remain on the TE grouped-GEMM path",
        file=sys.stderr,
        flush=True,
    )
if _use_reference_fwd:
    print(
        "[TE reference fwd] enabled: FFN forward uses shard-local "
        "jax.lax.ragged_dot; weight gradients remain on the TE grouped-GEMM path",
        file=sys.stderr,
        flush=True,
    )
if _zero_dispatch_weight_grad:
    print(
        "[TE dispatch diagnostic] routing-weight cotangent into ep_dispatch_bwd "
        "is forced to zero; token dgrad remains enabled",
        file=sys.stderr,
        flush=True,
    )
if _refresh_ep_handle_in_bwd:
    print(
        "[TE EP handle diagnostic] backward refreshes the NCCL EP routing "
        "handle from the saved routing map before combine/dispatch",
        file=sys.stderr,
        flush=True,
    )
if _refresh_ep_handle_before_combine:
    print(
        "[TE EP handle diagnostic] forward refreshes the NCCL EP routing "
        "handle immediately before combine",
        file=sys.stderr,
        flush=True,
    )
if _validate_ep_routing:
    print(
        "[TE EP routing probe] enabled: validates effective combine-fwd and "
        "dispatch-bwd handle mappings with deterministic expert codes",
        file=sys.stderr,
        flush=True,
    )
if _debug_moe_fwd_recompute:
    print(
        "[TE MoE fwd recompute debug] enabled: logs order-sensitive input/output "
        "statistics keyed by the routing signature",
        file=sys.stderr,
        flush=True,
    )
if _validate_ep_token_roundtrip:
    print(
        "[TE EP token roundtrip] enabled: validates intra-expert token ordering "
        "across dispatch-fwd and dispatch-bwd",
        file=sys.stderr,
        flush=True,
    )
if _zero_moe_input_grad:
    print(
        "[TE MoE input-grad diagnostic] the complete MoE input cotangent is "
        "forced to zero after routing and gate gradients are combined",
        file=sys.stderr,
        flush=True,
    )
if _skip_moe_backward:
    print(
        "[TE MoE backward diagnostic] bypassing the complete TE MoE backward "
        "body and returning zero activation/parameter cotangents",
        file=sys.stderr,
        flush=True,
    )
if _debug_handle_mem:
    print(
        "[TE EP handle-value diagnostic] logging each handle_mem's complete-byte "
        "signatures and exact head/tail byte samples after ep_prepare",
        file=sys.stderr,
        flush=True,
    )
if _validate_ep_forward_roundtrip:
    print(
        "[TE EP forward roundtrip] validating real dispatched token values and "
        "ordering through combine_fwd",
        file=sys.stderr,
        flush=True,
    )
if _zero_moe_output:
    print(
        "[TE MoE output diagnostic] forcing the routed-MoE forward output to "
        "zero after all TE forward operations and validation probes",
        file=sys.stderr,
        flush=True,
    )
if _skip_moe_forward:
    print(
        "[TE MoE forward diagnostic] bypassing the complete TE routed-MoE "
        "forward body and returning a zero output",
        file=sys.stderr,
        flush=True,
    )


def _debug_stable_stats(value, row_active=None):
    """Return bounded sampled stats without materializing a full float32 copy."""
    value = jnp.asarray(value)
    # A dispatch tensor is commonly [num_procs, recv_capacity, hidden] while
    # its activity mask is [num_procs, recv_capacity]. Treat every mask entry
    # as one logical row; using only value.shape[0] here would accidentally
    # apply the first few token-mask entries to whole process-sized slabs.
    if row_active is None:
        matrix = value.reshape(value.shape[0], -1)
    else:
        row_active = jnp.asarray(row_active, jnp.bool_).reshape(-1)
        if value.size % row_active.size != 0:
            raise ValueError(
                "Debug row mask must divide the sampled tensor size, but got "
                f"value.shape={value.shape} and row_active.shape={row_active.shape}."
            )
        matrix = value.reshape(row_active.size, -1)
    max_samples = 65536
    stride = max((matrix.size + max_samples - 1) // max_samples, 1)
    sample = matrix.reshape(-1)[::stride][:max_samples].astype(jnp.float32)
    # Avoid forming flattened indices that can exceed int32 for production
    # dispatch buffers (>180B logical elements).
    stride_rows, stride_remainder = divmod(stride, matrix.shape[1])
    sample_indices = jnp.arange(sample.size, dtype=jnp.int32)
    sampled_rows = (
        sample_indices * stride_rows
        + (sample_indices * stride_remainder) // matrix.shape[1]
    )
    if row_active is None:
        active = jnp.ones(sample.shape, dtype=jnp.bool_)
    else:
        active = row_active[sampled_rows]
    finite = jnp.isfinite(sample) & active
    finite_value = jnp.where(finite, sample, 0.0)
    absmax = jnp.max(jnp.abs(finite_value))
    safe_scale = jnp.where(absmax > 0, absmax, 1.0)
    scaled = finite_value / safe_scale
    element_count = jnp.maximum(jnp.sum(active, dtype=jnp.float32), 1.0)
    scaled_mean = jnp.sum(scaled) / element_count
    scaled_square_mean = jnp.sum(jnp.square(scaled)) / element_count
    mean = absmax * scaled_mean
    abs_mean = absmax * (jnp.sum(jnp.abs(scaled)) / element_count)
    stddev = absmax * jnp.sqrt(jnp.maximum(scaled_square_mean - jnp.square(scaled_mean), 0.0))
    finite_fraction = jnp.sum(finite, dtype=jnp.float32) / element_count
    return mean, abs_mean, stddev, absmax, finite_fraction


def _ep_routing_probe_code_table(num_experts, dtype):
    """Return deterministic ±1 codes that identify experts across 16 channels."""
    expert = jnp.arange(num_experts, dtype=jnp.uint32)[:, None]
    channel = jnp.arange(_EP_ROUTING_PROBE_WIDTH, dtype=jnp.uint32)[None, :]
    value = (expert + jnp.uint32(1)) * jnp.uint32(0x9E3779B1)
    value ^= (channel + jnp.uint32(1)) * jnp.uint32(0x85EBCA77)
    value ^= value >> jnp.uint32(16)
    value *= jnp.uint32(0xC2B2AE3D)
    value ^= value >> jnp.uint32(13)
    return jnp.where((value & jnp.uint32(1)) != 0, 1, -1).astype(dtype)


def _ep_routing_probe_packed_codes(
    token_counts,
    recv_capacity_per_rank,
    num_ep,
    num_local_experts,
    dtype,
):
    """Build expert-major probe rows matching the native EP receive layout."""
    leading_size = token_counts.shape[0]
    ep_rank = jnp.arange(leading_size, dtype=jnp.int32) % num_ep
    local_expert = jnp.arange(num_local_experts, dtype=jnp.int32)
    global_expert = ep_rank[:, None] * num_local_experts + local_expert[None, :]
    code_table = _ep_routing_probe_code_table(num_ep * num_local_experts, dtype)
    codes_by_group = code_table[global_expert]

    def _repeat_one_leading_group(group_codes, group_counts):
        return jnp.repeat(
            group_codes,
            group_counts.astype(jnp.int32),
            axis=0,
            total_repeat_length=recv_capacity_per_rank,
        )

    packed = jax.vmap(_repeat_one_leading_group)(codes_by_group, token_counts)
    active_rows = (
        jnp.arange(recv_capacity_per_rank, dtype=jnp.int32)[None, :]
        < jnp.sum(token_counts, axis=-1, dtype=jnp.int32)[:, None]
    )
    return jnp.where(active_rows[..., None], packed, jnp.zeros_like(packed))


def _ep_routing_probe_signature(topk_idx):
    """Return two order-sensitive uint32 signatures for the intended map."""
    flat = topk_idx.reshape(-1).astype(jnp.uint32)
    position = jnp.arange(flat.size, dtype=jnp.uint32)
    signature_0 = jnp.sum(
        (flat + jnp.uint32(1))
        * (position * jnp.uint32(0x9E3779B1) + jnp.uint32(0x85EBCA77)),
        dtype=jnp.uint32,
    )
    signature_1 = jnp.sum(
        (flat + jnp.uint32(17))
        * (position * jnp.uint32(0xC2B2AE3D) + jnp.uint32(0x27D4EB2F)),
        dtype=jnp.uint32,
    )
    return signature_0, signature_1


def _print_handle_mem_value(label, handle_mem, topk_idx):
    """Log exact handle bytes plus full-buffer fingerprints after ep_prepare.

    ``handle_mem`` is an opaque uint8 tensor with one row per global EP/DP
    rank. Printing every byte for every scanned layer would make the
    multi-host log impractically large, so the exact first/last 64 bytes are
    printed and two order-sensitive signatures cover every byte in each row.
    The routing-map signature in the same record makes repeated-forward
    comparisons unambiguous.
    """
    if not _debug_handle_mem:
        return
    rows = handle_mem.reshape(-1, handle_mem.shape[-1]).astype(jnp.uint32)
    position = jnp.arange(rows.shape[-1], dtype=jnp.uint32)
    signature_0 = jnp.sum(
        (rows + jnp.uint32(1))
        * (position * jnp.uint32(0x9E3779B1) + jnp.uint32(0x85EBCA77)),
        axis=-1,
        dtype=jnp.uint32,
    )
    signature_1 = jnp.sum(
        (rows + jnp.uint32(17))
        * (position * jnp.uint32(0xC2B2AE3D) + jnp.uint32(0x27D4EB2F)),
        axis=-1,
        dtype=jnp.uint32,
    )
    byte_sum = jnp.sum(rows, axis=-1, dtype=jnp.uint32)
    route_signature_0, route_signature_1 = _ep_routing_probe_signature(topk_idx)
    sample_width = min(64, handle_mem.shape[-1])
    jax.debug.print(
        f"[TE EP handle value] label={label} "
        f"shape={handle_mem.shape} "
        "route_sig=({route_sig0},{route_sig1}) "
        "byte_sum={byte_sum} handle_sig0={handle_sig0} "
        "handle_sig1={handle_sig1} head={head} tail={tail}",
        route_sig0=route_signature_0,
        route_sig1=route_signature_1,
        byte_sum=byte_sum,
        handle_sig0=signature_0,
        handle_sig1=signature_1,
        head=handle_mem[..., :sample_width],
        tail=handle_mem[..., -sample_width:],
        ordered=False,
    )


def _debug_ordered_tensor_stats(value):
    """Return stable scalar stats plus two order-sensitive sampled projections."""
    value = jnp.asarray(value)
    max_samples = 65536
    stride = max((value.size + max_samples - 1) // max_samples, 1)
    sample = value.reshape(-1)[::stride][:max_samples].astype(jnp.float32)
    finite = jnp.isfinite(sample)
    finite_value = jnp.where(finite, sample, 0.0)
    absmax = jnp.max(jnp.abs(finite_value))
    safe_scale = jnp.where(absmax > 0, absmax, 1.0)
    scaled = finite_value / safe_scale
    count = jnp.maximum(jnp.sum(finite, dtype=jnp.float32), 1.0)
    mean = absmax * jnp.sum(scaled) / count
    square_mean = jnp.sum(jnp.square(scaled)) / count
    stddev = absmax * jnp.sqrt(
        jnp.maximum(square_mean - jnp.square(jnp.sum(scaled) / count), 0.0)
    )
    position = jnp.arange(sample.size, dtype=jnp.uint32)
    sign_0 = jnp.where(
        ((position * jnp.uint32(0x9E3779B1) + jnp.uint32(0x85EBCA77)) >> 31) != 0,
        1.0,
        -1.0,
    )
    sign_1 = jnp.where(
        ((position * jnp.uint32(0xC2B2AE3D) + jnp.uint32(0x27D4EB2F)) >> 31) != 0,
        1.0,
        -1.0,
    )
    projection_0 = jnp.sum(scaled * sign_0) / count
    projection_1 = jnp.sum(scaled * sign_1) / count
    return mean, stddev, absmax, projection_0, projection_1, jnp.mean(finite)


def _print_ep_routing_probe_result(label, topk_idx, actual, expected, tolerance):
    """Print an elementwise comparison for an expert-code routing probe."""
    difference = actual.astype(jnp.float32) - expected.astype(jnp.float32)
    abs_difference = jnp.abs(difference)
    mismatch = abs_difference > tolerance
    signature_0, signature_1 = _ep_routing_probe_signature(topk_idx)
    jax.debug.print(
        "[TE EP routing probe] {label} route_sig=({signature_0},{signature_1}) "
        "match={match} "
        "mismatch_fraction={mismatch_fraction:.6e} "
        "absmean={absmean:.6e} absmax={absmax:.6e}",
        label=label,
        signature_0=signature_0,
        signature_1=signature_1,
        match=jnp.all(~mismatch),
        mismatch_fraction=jnp.mean(mismatch.astype(jnp.float32)),
        absmean=jnp.mean(abs_difference),
        absmax=jnp.max(abs_difference),
        ordered=False,
    )


def _with_sharding_constraint_cast_bwd(x: jnp.ndarray, sharding) -> jnp.ndarray:
    """Sharding constraint that keeps bwd cotangents in the primal dtype.

    Plain ``jax.lax.with_sharding_constraint`` is identity on the fwd
    but does not constrain the dtype of the cotangent that flows back
    through it. In this MoE bwd, ``d_x`` is built from two paths:

      * ``d_x_from_dispatch`` from ``ep_dispatch_bwd`` -- primal dtype
        (bf16 in mixed precision).
      * ``d_x_from_gate = d_logits_2d @ gate_kernel.T`` where
        ``d_logits_2d`` is produced by
        ``fused_topk_with_score_function_bwd``. That primitive runs at
        fp32 because the fwd promoted ``logits_2d`` to fp32 (the fused
        topk/softmax/sigmoid kernels are only validated at fp32).

    JAX's type promotion then makes ``d_x_from_gate + d_x_from_dispatch``
    fp32, so the user-visible ``d_x`` ends up wider than ``x``. That
    doubles activation-grad bandwidth and breaks any downstream kernel
    that pins a bf16 input layout. This wrapper inserts an explicit
    cast back to the primal dtype on the bwd side and re-asserts the
    same sharding there as well.
    """

    @jax.custom_vjp
    def _constraint(y):
        return jax.lax.with_sharding_constraint(y, sharding)

    def _constraint_fwd(y):
        return jax.lax.with_sharding_constraint(y, sharding), jnp.zeros((), dtype=y.dtype)

    def _constraint_bwd(dtype_ref, grad):
        return (jax.lax.with_sharding_constraint(grad.astype(dtype_ref.dtype), sharding),)

    _constraint.defvjp(_constraint_fwd, _constraint_bwd)
    return _constraint(x)


# =============================================================================
# Process-level NCCL EP bootstrap (must run eagerly, outside jax.jit)
# =============================================================================
#
# ``tex.ep_bootstrap`` does a NCCL UID allgather over the JAX runtime, which
# cannot run from inside a jit-traced function. The caller must bootstrap
# eagerly once per process before any jitted MoE call, then record the
# bootstrap signature via ``record_ep_bootstrap_signature_for_moe``. The
# per-call check below verifies the recorded signature matches the current
# MoE invocation. NCCL EP permits a smaller token count than the bootstrap
# maximum, but the dispatch receive capacity itself must match exactly.

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


def _te_ep_assert_compatible_bootstrap(
    num_experts: int,
    max_tokens_per_rank: int,
    recv_capacity_per_rank: int,
    hidden_dim: int,
    ep_size: int,
) -> None:
    """Verify a prior eager ``ep_bootstrap`` is compatible with this call."""
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
        or recv_capacity_per_rank != b_recv_pr
    ):
        raise ValueError(
            "TE EP was already bootstrapped with signature"
            f" (num_experts={b_num_experts}, max_tokens_per_rank={b_max_tpr},"
            f" recv_capacity_per_rank={b_recv_pr}, hidden_dim={b_hidden},"
            f" ep_size={b_ep_size}); this moe() call needs"
            f" (num_experts={num_experts}, max_tokens_per_rank={max_tokens_per_rank},"
            f" recv_capacity_per_rank={recv_capacity_per_rank}, hidden_dim={hidden_dim},"
            f" ep_size={ep_size}). Re-bootstrap with wider params (or matching exact"
            " sizes) is required. NCCL EP dispatch capacity must exactly match bootstrap."
        )


# =============================================================================
# Residual container threaded fwd -> bwd
# =============================================================================


@flax.struct.dataclass
class _Ctx:
    """Residuals carried from the fwd rule into the bwd rule.

    Flattened automatically by jax.custom_vjp; ``cfg`` is the only
    static field (the rest are jnp.ndarray, GroupedNoScaleTensor, or
    None when aux_loss_coeff == 0).
    """

    x: jnp.ndarray
    gate_kernel: jnp.ndarray
    expert_bias: jnp.ndarray
    logits_2d: jnp.ndarray
    saved_scores: jnp.ndarray
    routing_map: jnp.ndarray
    cfg: Any = flax.struct.field(pytree_node=False)
    handle_mem: jnp.ndarray
    token_counts: jnp.ndarray
    recv_topk_weights: jnp.ndarray
    recv_token_probe: Any
    casted_sorted_x_lhs_trans: Any
    casted_wi_rhs_trans: Any
    gate_proj_out: jnp.ndarray
    up_proj_out: jnp.ndarray
    casted_intermediate_lhs_trans: Any
    casted_wo_rhs_trans: Any
    wi: Any
    wo: Any
    wi_0_bias: Any
    wi_1_bias: Any
    wo_bias: Any
    expert_outputs: jnp.ndarray
    local_group_sizes: jnp.ndarray
    quantizer_sets: Any
    aux_const_buf: Any = None
    aux_tokens_per_expert: Any = None
    aux_saved_scores: Any = None


# =============================================================================
# Per-shard FFN body
# =============================================================================


def _ffn_fwd_per_shard(
    recv_tokens_local: jnp.ndarray,
    recv_topk_weights_local: jnp.ndarray,
    token_counts_local: jnp.ndarray,
    wi: jnp.ndarray,
    wo: jnp.ndarray,
    wi_0_bias: Optional[jnp.ndarray],
    wi_1_bias: Optional[jnp.ndarray],
    wo_bias: Optional[jnp.ndarray],
    quantizer_sets: Tuple[QuantizerSet, QuantizerSet],
    *,
    num_local_experts: int,
    activation_type: str,
    apply_topk_weights_early: bool,
):
    """Run the grouped FFN on one shard's EP receive buffer."""
    hidden = recv_tokens_local.shape[-1]
    sorted_x = recv_tokens_local.reshape(-1, hidden)
    recv_w_flat = recv_topk_weights_local.reshape(-1)
    group_sizes = token_counts_local.reshape(-1).astype(jnp.int32)

    wi = wi.astype(sorted_x.dtype)
    wo = wo.astype(sorted_x.dtype)

    if _use_reference_fwd:
        if wi_0_bias is not None:
            raise ValueError("NVTE_MOE_REFERENCE_FWD does not support expert biases.")

    # ``wi`` is stored in its gated-SwiGLU layout [expert, hidden, 2*mlp].
    # Keeping it contiguous lets grouped quantize/GEMM consume it directly.
    wi_combined_bias = (
        jnp.concatenate([wi_0_bias, wi_1_bias], axis=-1) if wi_0_bias is not None else None
    )

    fc1_quantizer_set, fc2_quantizer_set = quantizer_sets
    casted_sorted_x = tex.grouped_quantize(
        sorted_x,
        fc1_quantizer_set.x,
        group_sizes,
        flatten_axis=-1,
    )
    casted_wi = tex.grouped_quantize(wi, fc1_quantizer_set.kernel, flatten_axis=-1)
    if _use_reference_fwd:
        combined_out = jax.lax.ragged_dot(sorted_x, wi, group_sizes)
    else:
        combined_out = tex.grouped_gemm(
            casted_sorted_x.get_tensor(usage=TensorUsage.LHS),
            casted_wi.get_tensor(usage=TensorUsage.RHS),
            contracting_dims=((1,), (1,)),
            bias=wi_combined_bias,
        )
    gate_proj_out, up_proj_out = jnp.split(combined_out, 2, axis=-1)
    casted_sorted_x_lhs_trans = casted_sorted_x.get_tensor(usage=TensorUsage.LHS_TRANS).checkpoint(
        fc1_quantizer_set.x
    )
    casted_wi_rhs_trans = casted_wi.get_tensor(usage=TensorUsage.RHS_TRANS).checkpoint(
        fc1_quantizer_set.kernel
    )

    # Activation inputs (gate_proj_out, up_proj_out) stay in the wi GEMM
    # output dtype; the activation output (`intermediate`) stays in the
    # dtype the wo GEMM / wo's quantized input consumes. For bf16 compute
    # that's all bf16; for FP8/FP4 the downstream grouped_quantize is what
    # transitions to the target precision.
    act_fn = _convert_to_activation_function(activation_type)
    intermediate = act_fn(gate_proj_out) * up_proj_out
    if apply_topk_weights_early:
        # Fold the per-token combine weights into the FFN intermediate;
        # the downstream wo GEMM is linear so this is equivalent to the
        # late-weighting path. Grouped GEMM skips padding automatically.
        intermediate = intermediate * recv_w_flat[:, None].astype(intermediate.dtype)

    casted_intermediate = tex.grouped_quantize(
        intermediate,
        fc2_quantizer_set.x,
        group_sizes,
        flatten_axis=-1,
    )
    casted_wo = tex.grouped_quantize(wo, fc2_quantizer_set.kernel, flatten_axis=-1)
    if _use_reference_fwd:
        expert_outputs = jax.lax.ragged_dot(intermediate, wo, group_sizes)
    else:
        expert_outputs = tex.grouped_gemm(
            casted_intermediate.get_tensor(usage=TensorUsage.LHS),
            casted_wo.get_tensor(usage=TensorUsage.RHS),
            contracting_dims=((1,), (1,)),
            bias=wo_bias,
        )
    casted_intermediate_lhs_trans = casted_intermediate.get_tensor(
        usage=TensorUsage.LHS_TRANS
    ).checkpoint(fc2_quantizer_set.x)
    casted_wo_rhs_trans = casted_wo.get_tensor(usage=TensorUsage.RHS_TRANS).checkpoint(
        fc2_quantizer_set.kernel
    )

    expert_outputs_3d = expert_outputs.reshape(1, expert_outputs.shape[0], expert_outputs.shape[1])
    group_sizes_2d = group_sizes.reshape(1, num_local_experts)
    residuals = (
        casted_sorted_x_lhs_trans,
        casted_wi_rhs_trans,
        gate_proj_out,
        up_proj_out,
        casted_intermediate_lhs_trans,
        casted_wo_rhs_trans,
        group_sizes_2d,
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
    wi: jnp.ndarray,
    wo: jnp.ndarray,
    quantizer_sets: Tuple[QuantizerSet, QuantizerSet],
    *,
    activation_type: str,
    apply_topk_weights_early: bool,
    has_bias: bool,
):
    """Backward mirror of :func:`_ffn_fwd_per_shard`."""
    group_sizes = local_group_sizes.reshape(-1).astype(jnp.int32)
    d_eo_2d = d_expert_outputs_local.reshape(-1, d_expert_outputs_local.shape[-1])
    recv_w_flat = recv_topk_weights_local.reshape(-1)
    fc1_quantizer_set, fc2_quantizer_set = quantizer_sets
    wgrad_group_active = (group_sizes > 0)[:, None, None]

    # wo bwd
    casted_d_eo = tex.grouped_quantize(
        d_eo_2d,
        fc2_quantizer_set.dgrad,
        group_sizes,
        flatten_axis=-1,
    )
    _casted_d_eo_lhs = casted_d_eo.get_tensor(usage=TensorUsage.LHS)
    _casted_d_eo_rhs = casted_d_eo.get_tensor(usage=TensorUsage.RHS)
    if _use_reference_dgrad:
        d_intermediate = jax.lax.ragged_dot(d_eo_2d, jnp.swapaxes(wo, -1, -2), group_sizes)
    else:
        d_intermediate = tex.grouped_gemm(
            _casted_d_eo_lhs,
            casted_wo_rhs_trans,
            contracting_dims=((1,), (2,)),
        )
    d_wo = tex.grouped_gemm(
        casted_intermediate_lhs_trans,
        _casted_d_eo_rhs,
        contracting_dims=((0,), (0,)),
    )
    d_wo = jnp.where(wgrad_group_active, d_wo, jnp.zeros_like(d_wo))
    d_wo_bias = tex.grouped_dbias(d_eo_2d, group_sizes) if has_bias else None

    act_fn = _convert_to_activation_function(activation_type)
    if apply_topk_weights_early:
        # intermediate' = intermediate * w. The subsequent dgrad and EP
        # operations consume group_sizes, so they skip padded ragged rows.
        w_b = recv_w_flat[:, None].astype(d_intermediate.dtype)
        gate_proj_for_bwd = gate_proj_out
        up_proj_for_bwd = up_proj_out
        intermediate_unweighted = act_fn(gate_proj_out) * up_proj_out
        d_recv_w_from_intermediate = jnp.sum(
            d_intermediate * intermediate_unweighted,
            axis=-1,
        ).astype(recv_w_flat.dtype)
        d_intermediate = d_intermediate * w_b
    else:
        gate_proj_for_bwd = gate_proj_out
        up_proj_for_bwd = up_proj_out
        d_recv_w_from_intermediate = jnp.zeros_like(recv_w_flat)

    # Activation bwd, symmetric with the fwd: silu' and the two
    # elementwise products run in the GEMM dtype (no fp32 island), so
    # the chain rule composes through at the same precision the wi/wo
    # GEMMs consume.
    act_gp, dact_pullback = jax.vjp(act_fn, gate_proj_for_bwd)
    d_up_proj_out = d_intermediate * act_gp
    (d_gate_proj_out,) = dact_pullback(d_intermediate * up_proj_for_bwd)

    # wi bwd (fused gate/up via concat). Mirror the fused fwd: pack the
    # gate/up cotangents along the trailing axis, run a single
    # grouped_quantize + two grouped_gemm pair (one dgrad, one wgrad)
    # against the fused casted_wi_rhs_trans residual, then split the
    # wgrad result remains in the contiguous gated-SwiGLU ``wi`` layout.
    d_combined = jnp.concatenate([d_gate_proj_out, d_up_proj_out], axis=-1)
    casted_d_combined = tex.grouped_quantize(
        d_combined,
        fc1_quantizer_set.dgrad,
        group_sizes,
        flatten_axis=-1,
    )
    if _use_reference_dgrad:
        d_sorted_x = jax.lax.ragged_dot(d_combined, jnp.swapaxes(wi, -1, -2), group_sizes)
    else:
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
    d_wi_combined = jnp.where(
        wgrad_group_active, d_wi_combined, jnp.zeros_like(d_wi_combined)
    )
    if has_bias:
        d_wi_combined_bias = tex.grouped_dbias(d_combined, group_sizes)
        d_wi_0_bias, d_wi_1_bias = jnp.split(d_wi_combined_bias, 2, axis=-1)
    else:
        d_wi_0_bias = None
        d_wi_1_bias = None

    d_sorted_x_3d = d_sorted_x.reshape(1, d_sorted_x.shape[0], d_sorted_x.shape[1])
    d_recv_w_3d = d_recv_w_from_intermediate.reshape(1, -1)
    return (
        d_sorted_x_3d,
        d_recv_w_3d,
        d_wi_combined,
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
    wi,
    wo,
    wi_0_bias,
    wi_1_bias,
    wo_bias,
    expert_bias,
    quantizer_sets,
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
    recv_capacity_per_rank,
):
    """Forward: gate -> topk -> ep_dispatch -> FFN -> ep_combine.

    Returns ``(output, aux_loss)``. ``aux_loss`` is a zero scalar when
    ``aux_loss_coeff == 0``.
    """
    del gate_kernel_axes, wi_kernel_axes, wo_kernel_axes  # used in bwd only
    from jax.experimental.shard_map import shard_map

    x = with_sharding_constraint_by_logical_axes(x, input_axes)

    if _skip_moe_forward:
        if not _skip_moe_backward:
            raise RuntimeError(
                "NVTE_MOE_SKIP_FORWARD requires NVTE_MOE_SKIP_BACKWARD=1."
            )
        has_bias = wi_0_bias is not None
        ctx = _Ctx(
            x=x,
            gate_kernel=gate_kernel,
            expert_bias=expert_bias,
            logits_2d=None,
            saved_scores=None,
            routing_map=None,
            cfg=None,
            handle_mem=None,
            token_counts=None,
            recv_topk_weights=None,
            recv_token_probe=None,
            casted_sorted_x_lhs_trans=None,
            casted_wi_rhs_trans=None,
            gate_proj_out=None,
            up_proj_out=None,
            casted_intermediate_lhs_trans=None,
            casted_wo_rhs_trans=None,
            wi=wi,
            wo=wo,
            wi_0_bias=wi_0_bias if has_bias else None,
            wi_1_bias=wi_1_bias if has_bias else None,
            wo_bias=wo_bias if has_bias else None,
            expert_outputs=None,
            local_group_sizes=None,
            quantizer_sets=quantizer_sets,
        )
        static = {
            "has_bias": has_bias,
            "x_shape": x.shape,
            "recv_pr": 0,
        }
        return (
            jnp.zeros_like(x),
            jnp.zeros((), dtype=x.dtype),
            jnp.zeros((1,), dtype=jnp.int32),
        ), (ctx, static)

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

    # Per-rank send capacity: B/num_procs rows x S tokens per rank.
    max_tokens_per_rank = (B // num_procs) * S
    worst_case_recv_pr = get_moe_recv_capacity_per_rank(
        num_experts=num_experts,
        num_experts_per_tok=K,
        max_tokens_per_rank=max_tokens_per_rank,
        ep_size=num_ep,
    )
    if recv_capacity_per_rank is None:
        recv_pr = worst_case_recv_pr
    else:
        recv_pr = int(recv_capacity_per_rank)
        if recv_pr <= 0 or recv_pr % _ALIGN_SIZE != 0:
            raise ValueError(
                f"recv_capacity_per_rank must be a positive multiple of {_ALIGN_SIZE}, got {recv_pr}"
            )

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
    # tex.fused_topk_with_score_function is only validated against its
    # pytorch reference at fp32 (see tests/pytorch/test_fused_router.py:
    # parametrize gates dtype on torch.float32 only; the tolerance helper
    # raises NotImplementedError for any other dtype). Keeping logits in
    # the activation dtype (e.g. bf16) lets sigmoid / softmax / topk
    # accumulate at low precision and silently produce NaNs on tokens
    # whose normalised weights underflow. Cast to fp32 here to stay in
    # the validated regime.
    gate_kernel_cast = gate_kernel.astype(x.dtype)
    gate_logits = jnp.einsum("bsh,he->bse", x, gate_kernel_cast)
    logits_2d = gate_logits.reshape(-1, num_experts).astype(jnp.float32)

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
        global_logits_2d = jax.lax.with_sharding_constraint(logits_2d, NamedSharding(mesh, P()))
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
    topk_idx_3d = jax.lax.with_sharding_constraint(topk_idx_3d, NamedSharding(mesh, ep3_spec))
    topk_w_3d = jax.lax.with_sharding_constraint(topk_w_3d, NamedSharding(mesh, ep3_spec))

    # ---------------- TE EP dispatch (global view) ----------------
    cfg = tex.EpLayerConfig(
        top_k=K,
        dispatch_output_per_expert_alignment=_ALIGN_SIZE,
    )
    token_counts, total_recv_tokens, handle_mem = tex.ep_prepare(cfg, topk_idx_3d)
    _print_handle_mem_value("forward_prepare", handle_mem, topk_idx_3d)
    token_counts = jax.lax.with_sharding_constraint(token_counts, NamedSharding(mesh, ep2_spec))
    recv_tokens, recv_topk_weights = tex.ep_dispatch_fwd(
        cfg, handle_mem, topk_idx_3d, x, topk_w_3d, recv_pr
    )
    recv_tokens = jax.lax.with_sharding_constraint(recv_tokens, NamedSharding(mesh, ep3_spec))
    recv_topk_weights = jax.lax.with_sharding_constraint(
        recv_topk_weights, NamedSharding(mesh, ep2_spec)
    )

    # ---------------- FFN (per-shard via shard_map) ----------------
    has_bias = wi_0_bias is not None
    kernel_spec = P(ep_axis, None, None)
    bias_spec = P(ep_axis, None)
    ffn_in_specs = (ep3_spec, ep2_spec, ep2_spec, kernel_spec, kernel_spec)
    ffn_in_args = [recv_tokens, recv_topk_weights, token_counts, wi, wo]
    if has_bias:
        ffn_in_specs += (bias_spec, bias_spec, bias_spec)
        ffn_in_args.extend([wi_0_bias, wi_1_bias, wo_bias])

    residuals_spec = (
        P(),
        P(ep_axis, None, None),
        P(),
        P(),
        P(),
        P(ep_axis, None, None),
        ep2_spec,
    )

    def _ffn_fwd_body(*args):
        if has_bias:
            r_tok, r_w, tc, local_wi, local_wo, w0b, w1b, wob = args
        else:
            r_tok, r_w, tc, local_wi, local_wo = args
            w0b = w1b = wob = None
        return _ffn_fwd_per_shard(
            r_tok,
            r_w,
            tc,
            local_wi,
            local_wo,
            w0b,
            w1b,
            wob,
            quantizer_sets,
            num_local_experts=num_local_experts,
            activation_type=activation_type,
            apply_topk_weights_early=apply_topk_weights_early,
        )

    expert_outputs, ffn_residuals = shard_map(
        _ffn_fwd_body,
        mesh=mesh,
        in_specs=ffn_in_specs,
        out_specs=(ep3_spec, residuals_spec),
        check_rep=False,
    )(*ffn_in_args)
    expert_outputs = jax.lax.with_sharding_constraint(expert_outputs, NamedSharding(mesh, ep3_spec))

    # ---------------- TE EP combine (global view) ----------------
    out_partition_spec = (batch_pspec_axis, None, None)
    combine_handle_mem = handle_mem
    if _refresh_ep_handle_before_combine:
        # A scanned forward can reuse one physical handle_mem allocation across
        # loop iterations. Re-prepare the current routing map at the combine
        # consumption point so combine cannot rely on routing state left by a
        # different iteration. Keep the original handle in the VJP residual:
        # this diagnostic intentionally isolates forward combine.
        refreshed_token_counts, combine_handle_mem = tex.ep_prepare(cfg, topk_idx_3d)
        _print_handle_mem_value(
            "forward_pre_combine_refresh", combine_handle_mem, topk_idx_3d
        )
        refreshed_token_counts = jax.lax.with_sharding_constraint(
            refreshed_token_counts, NamedSharding(mesh, ep2_spec)
        )
        if _debug_moe_numerics:
            jax.debug.print(
                "[TE EP pre-combine refresh] token_counts_match={match}",
                match=jnp.all(refreshed_token_counts == token_counts),
                ordered=False,
            )
    if apply_topk_weights_early:
        # expert_outputs is already weighted upstream.
        output = tex.ep_combine_fwd(
            cfg,
            combine_handle_mem,
            expert_outputs,
            num_local_tokens=(B, S),
            out_partition_spec=out_partition_spec,
        )
    else:
        # HT combine is unweighted; apply routing weights before calling it.
        # Padded recv slots are ignored by combine via handle_mem metadata.
        w = recv_topk_weights[..., None].astype(expert_outputs.dtype)
        weighted = expert_outputs * w
        output = tex.ep_combine_fwd(
            cfg,
            combine_handle_mem,
            weighted,
            num_local_tokens=(B, S),
            out_partition_spec=out_partition_spec,
        )
    # output of MLP should be sharded the same way as the activation input
    output = with_sharding_constraint_by_logical_axes(output, input_axes)

    if _validate_ep_forward_roundtrip:
        # This validates more than the synthetic expert-code probe below:
        # dispatch the actual input token values, combine them immediately,
        # and compare against the weighted identity analytically. It detects
        # a mutually consistent but wrong dispatch/combine token permutation.
        probe_width = min(_EP_ROUTING_PROBE_WIDTH, H)
        roundtrip_weighted = (
            recv_tokens[..., :probe_width].astype(jnp.float32)
            * recv_topk_weights[..., None]
        ).astype(x.dtype)
        roundtrip_output = tex.ep_combine_fwd(
            cfg,
            combine_handle_mem,
            roundtrip_weighted,
            num_local_tokens=(B, S),
            out_partition_spec=out_partition_spec,
        )
        expected_roundtrip = (
            x[..., :probe_width].astype(jnp.float32)
            * jnp.sum(topk_w_3d, axis=-1, keepdims=True)
        ).astype(x.dtype)
        _print_ep_routing_probe_result(
            "forward_token_roundtrip",
            topk_idx_3d,
            roundtrip_output,
            expected_roundtrip,
            tolerance=6.25e-2,
        )

    if _validate_ep_routing:
        probe_code_table = _ep_routing_probe_code_table(num_experts, x.dtype)
        probe_packed = _ep_routing_probe_packed_codes(
            token_counts,
            recv_pr,
            num_ep,
            num_local_experts,
            x.dtype,
        )
        probe_packed = jax.lax.with_sharding_constraint(
            probe_packed, NamedSharding(mesh, ep3_spec)
        )
        probe_weighted = (
            probe_packed.astype(jnp.float32) * recv_topk_weights[..., None]
        ).astype(x.dtype)
        probe_combined = tex.ep_combine_fwd(
            cfg,
            combine_handle_mem,
            probe_weighted,
            num_local_tokens=(B, S),
            out_partition_spec=out_partition_spec,
        )
        expected_probe_terms = (
            probe_code_table[topk_idx_3d].astype(jnp.float32)
            * topk_w_3d[..., None]
        ).astype(x.dtype)
        expected_probe_combined = jnp.sum(
            expected_probe_terms.astype(jnp.float32), axis=-2
        ).astype(x.dtype)
        _print_ep_routing_probe_result(
            "combine_fwd",
            topk_idx_3d,
            probe_combined,
            expected_probe_combined,
            tolerance=6.25e-2,
        )

    if _zero_moe_output:
        output = jnp.zeros_like(output)

    if _debug_moe_fwd_recompute:
        signature_0, signature_1 = _ep_routing_probe_signature(topk_idx_3d)
        x_stats = _debug_ordered_tensor_stats(x)
        output_stats = _debug_ordered_tensor_stats(output)
        jax.debug.print(
            "[TE MoE fwd recompute] route_sig=({signature_0},{signature_1}) "
            "input(mean={x_mean:.6e},std={x_std:.6e},absmax={x_absmax:.6e},"
            "proj=({x_proj0:.9e},{x_proj1:.9e}),finite={x_finite:.6f}) "
            "output(mean={out_mean:.6e},std={out_std:.6e},absmax={out_absmax:.6e},"
            "proj=({out_proj0:.9e},{out_proj1:.9e}),finite={out_finite:.6f})",
            signature_0=signature_0,
            signature_1=signature_1,
            x_mean=x_stats[0],
            x_std=x_stats[1],
            x_absmax=x_stats[2],
            x_proj0=x_stats[3],
            x_proj1=x_stats[4],
            x_finite=x_stats[5],
            out_mean=output_stats[0],
            out_std=output_stats[1],
            out_absmax=output_stats[2],
            out_proj0=output_stats[3],
            out_proj1=output_stats[4],
            out_finite=output_stats[5],
            ordered=False,
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
        cfg=cfg,
        handle_mem=handle_mem,
        token_counts=token_counts,
        recv_topk_weights=recv_topk_weights,
        recv_token_probe=(
            jax.lax.with_sharding_constraint(
                recv_tokens[..., :_EP_ROUTING_PROBE_WIDTH],
                NamedSharding(mesh, ep3_spec),
            )
            if _validate_ep_token_roundtrip
            else None
        ),
        casted_sorted_x_lhs_trans=casted_sorted_x_lhs_trans,
        casted_wi_rhs_trans=casted_wi_rhs_trans,
        gate_proj_out=gate_proj_out,
        up_proj_out=up_proj_out,
        casted_intermediate_lhs_trans=casted_intermediate_lhs_trans,
        casted_wo_rhs_trans=casted_wo_rhs_trans,
        wi=wi if (_use_reference_dgrad or _skip_moe_backward) else None,
        wo=wo if (_use_reference_dgrad or _skip_moe_backward) else None,
        wi_0_bias=wi_0_bias if (has_bias and _skip_moe_backward) else None,
        wi_1_bias=wi_1_bias if (has_bias and _skip_moe_backward) else None,
        wo_bias=wo_bias if (has_bias and _skip_moe_backward) else None,
        expert_outputs=expert_outputs,
        local_group_sizes=local_group_sizes,
        quantizer_sets=quantizer_sets,
        aux_const_buf=aux_const_buf,
        aux_tokens_per_expert=aux_tokens_per_expert,
        aux_saved_scores=aux_saved_scores,
    )
    static = {
        "has_bias": has_bias,
        "x_shape": x.shape,
        "recv_pr": recv_pr,
    }
    # total_recv_tokens is a non-differentiable overflow signal (see moe()).
    return (output, aux_loss, total_recv_tokens), (ctx, static)


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
    recv_capacity_per_rank,
    residuals,
    cotangents,
):
    """Backward mirror of :func:`_moe_fwd_rule`."""
    del num_groups, group_topk, dtype, recv_capacity_per_rank  # captured / unused in bwd
    from jax.experimental.shard_map import shard_map

    # total_recv_tokens is a non-differentiable output; its cotangent is unused.
    d_output, d_aux_loss, _d_total_recv_tokens = cotangents

    ctx, static = residuals
    has_bias = static["has_bias"]
    x_shape = static["x_shape"]
    recv_pr = static["recv_pr"]

    if _skip_moe_backward:
        # Strong isolation diagnostic: do not execute combine_bwd, grouped
        # GEMM backward, dispatch_bwd, or router backward. This differs from
        # NVTE_MOE_ZERO_INPUT_GRAD, which discards d_x only after all of those
        # operations have already run and therefore cannot rule out an
        # asynchronous side effect from a backward custom call.
        if ctx.wi is None or ctx.wo is None:
            raise RuntimeError(
                "NVTE_MOE_SKIP_BACKWARD requires wi/wo in the VJP residual."
            )
        d_x = with_sharding_constraint_by_logical_axes(
            jnp.zeros_like(ctx.x), input_axes
        )
        d_gate_kernel = with_sharding_constraint_by_logical_axes(
            jnp.zeros_like(ctx.gate_kernel), gate_kernel_axes
        )
        d_wi = with_sharding_constraint_by_logical_axes(
            jnp.zeros_like(ctx.wi), wi_kernel_axes
        )
        d_wo = with_sharding_constraint_by_logical_axes(
            jnp.zeros_like(ctx.wo), wo_kernel_axes
        )
        if has_bias:
            wi_bias_axes = (wi_kernel_axes[0], *wi_kernel_axes[2:])
            wo_bias_axes = (wo_kernel_axes[0], *wo_kernel_axes[2:])
            d_wi_0_bias = with_sharding_constraint_by_logical_axes(
                jnp.zeros_like(ctx.wi_0_bias), wi_bias_axes
            )
            d_wi_1_bias = with_sharding_constraint_by_logical_axes(
                jnp.zeros_like(ctx.wi_1_bias), wi_bias_axes
            )
            d_wo_bias = with_sharding_constraint_by_logical_axes(
                jnp.zeros_like(ctx.wo_bias), wo_bias_axes
            )
        else:
            d_wi_0_bias = None
            d_wi_1_bias = None
            d_wo_bias = None
        return (
            d_x,
            d_gate_kernel,
            d_wi,
            d_wo,
            d_wi_0_bias,
            d_wi_1_bias,
            d_wo_bias,
            jnp.zeros_like(ctx.expert_bias),
            ctx.quantizer_sets,
        )

    mesh = _get_mesh()
    if mesh is None or mesh.empty:
        raise ValueError("moe(...) requires an active jax.sharding.Mesh.")
    num_ep = mesh.shape[ep_axis]
    num_local_experts = num_experts // num_ep
    B, S, _ = x_shape
    K = num_experts_per_tok
    if not data_parallelism_axes:
        batch_pspec_axis: Any = ep_axis
    else:
        batch_pspec_axis = (*data_parallelism_axes, ep_axis)
    ep3_spec = P(batch_pspec_axis, None, None)
    ep2_spec = P(batch_pspec_axis, None)
    out_partition_spec = (batch_pspec_axis, None, None)

    # A scanned layer can reuse the same physical handle_mem buffer for
    # different loop iterations. The native NCCL EP cache is keyed by that
    # buffer pointer and retains routing state established by ep_prepare.
    # Refreshing here updates the cached handle to the routing plan for the
    # current reverse-scan iteration before either backward EP operation.
    bwd_handle_mem = ctx.handle_mem
    if _refresh_ep_handle_in_bwd:
        bwd_selected_experts = jnp.argsort(ctx.routing_map, axis=-1)[..., -K:]
        bwd_topk_idx = bwd_selected_experts.reshape(B, S, K).astype(jnp.int32)
        bwd_topk_idx = jax.lax.with_sharding_constraint(
            bwd_topk_idx, NamedSharding(mesh, ep3_spec)
        )
        refreshed_token_counts, bwd_handle_mem = tex.ep_prepare(ctx.cfg, bwd_topk_idx)
        _print_handle_mem_value(
            "backward_refresh", bwd_handle_mem, bwd_topk_idx
        )
        refreshed_token_counts = jax.lax.with_sharding_constraint(
            refreshed_token_counts, NamedSharding(mesh, ep2_spec)
        )
        if _debug_moe_numerics:
            token_count_match = jnp.all(refreshed_token_counts == ctx.token_counts)
            jax.debug.print(
                "[TE EP handle refresh] token_counts_match={match}",
                match=token_count_match,
                ordered=False,
            )

    # ---------------- Combine bwd (global view) ----------------
    d_output = jax.lax.with_sharding_constraint(d_output, NamedSharding(mesh, ep3_spec))
    grad_pre_combine = tex.ep_combine_bwd(ctx.cfg, bwd_handle_mem, d_output, recv_pr)
    grad_pre_combine = jax.lax.with_sharding_constraint(
        grad_pre_combine, NamedSharding(mesh, ep3_spec)
    )
    # The EP kernel writes only the per-process packed expert prefix. Its
    # over-allocation tail is intentionally left uninitialized, which is safe
    # only while every downstream consumer is perfectly handle/group aware.
    # Materialize the contract here so padding cannot leak through elementwise
    # weighting, compiler fusion, or a later dispatch backward.
    active_recv_rows = (
        jnp.arange(ctx.recv_topk_weights.shape[-1])[None, :]
        < jnp.sum(ctx.token_counts, axis=-1, dtype=jnp.int32)[:, None]
    )
    grad_pre_combine = jnp.where(
        active_recv_rows[..., None],
        grad_pre_combine,
        jnp.zeros_like(grad_pre_combine),
    )

    if apply_topk_weights_early:
        # combine_fwd consumed already-weighted expert_outputs; the recv_w
        # cotangent flows through the early-weighting step inside the FFN bwd.
        d_expert_outputs = grad_pre_combine
        d_recv_w_from_combine = jnp.zeros_like(ctx.recv_topk_weights)
    else:
        # Reverse the late-weighting multiply. The subsequent dgrad and EP
        # operations consume group_sizes, so they skip padded ragged rows.
        w = ctx.recv_topk_weights[..., None].astype(grad_pre_combine.dtype)
        d_expert_outputs = grad_pre_combine * w
        d_recv_w_from_combine = (grad_pre_combine * ctx.expert_outputs).sum(axis=-1)
        d_recv_w_from_combine = d_recv_w_from_combine.astype(ctx.recv_topk_weights.dtype)

    if _debug_moe_numerics:
        d_output_stats = _debug_stable_stats(d_output)
        grad_pre_combine_stats = _debug_stable_stats(
            grad_pre_combine, active_recv_rows
        )
        d_expert_output_stats = _debug_stable_stats(
            d_expert_outputs, active_recv_rows
        )
        d_recv_w_stats = _debug_stable_stats(
            d_recv_w_from_combine, active_recv_rows
        )
        jax.debug.print(
            "[TE combine-bwd stats] "
            "upstream(absmean={up_absmean:.3e},std={up_std:.3e},"
            "absmax={up_absmax:.3e},finite={up_finite:.6f}) "
            "combine(absmean={combine_absmean:.3e},std={combine_std:.3e},"
            "absmax={combine_absmax:.3e},finite={combine_finite:.6f}) "
            "weighted(absmean={weighted_absmean:.3e},std={weighted_std:.3e},"
            "absmax={weighted_absmax:.3e},finite={weighted_finite:.6f}) "
            "dweight(absmean={dw_absmean:.3e},std={dw_std:.3e},"
            "absmax={dw_absmax:.3e},finite={dw_finite:.6f})",
            up_absmean=d_output_stats[1],
            up_std=d_output_stats[2],
            up_absmax=d_output_stats[3],
            up_finite=d_output_stats[4],
            combine_absmean=grad_pre_combine_stats[1],
            combine_std=grad_pre_combine_stats[2],
            combine_absmax=grad_pre_combine_stats[3],
            combine_finite=grad_pre_combine_stats[4],
            weighted_absmean=d_expert_output_stats[1],
            weighted_std=d_expert_output_stats[2],
            weighted_absmax=d_expert_output_stats[3],
            weighted_finite=d_expert_output_stats[4],
            dw_absmean=d_recv_w_stats[1],
            dw_std=d_recv_w_stats[2],
            dw_absmax=d_recv_w_stats[3],
            dw_finite=d_recv_w_stats[4],
            ordered=False,
        )

    # ---------------- FFN bwd (per-shard via shard_map) ----------------
    kernel_spec = P(ep_axis, None, None)
    bias_spec = P(ep_axis, None)
    residuals_specs = (
        P(),
        P(ep_axis, None, None),
        P(),
        P(),
        P(),
        P(ep_axis, None, None),
        ep2_spec,
    )
    bwd_in_specs = (ep3_spec, *residuals_specs, ep2_spec)
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
    if _use_reference_dgrad:
        bwd_in_specs += (kernel_spec, kernel_spec)
        bwd_in_args.extend([ctx.wi, ctx.wo])

    def _ffn_bwd_body(*args):
        if _use_reference_dgrad:
            *common_args, local_wi, local_wo = args
        else:
            common_args = args
            local_wi = local_wo = None
        grads = _ffn_bwd_per_shard(
            *common_args,
            local_wi,
            local_wo,
            ctx.quantizer_sets,
            activation_type=activation_type,
            apply_topk_weights_early=apply_topk_weights_early,
            has_bias=has_bias,
        )
        (
            d_sorted_x_local,
            d_recv_w_local,
            d_wi_local,
            d_wo_local,
            d_wi_0_bias_local,
            d_wi_1_bias_local,
            d_wo_bias_local,
        ) = grads
        if data_parallelism_axes:
            dp_axes = tuple(data_parallelism_axes)
            d_wi_local = jax.lax.psum(d_wi_local, axis_name=dp_axes)
            d_wo_local = jax.lax.psum(d_wo_local, axis_name=dp_axes)
            if has_bias:
                d_wi_0_bias_local = jax.lax.psum(d_wi_0_bias_local, axis_name=dp_axes)
                d_wi_1_bias_local = jax.lax.psum(d_wi_1_bias_local, axis_name=dp_axes)
                d_wo_bias_local = jax.lax.psum(d_wo_bias_local, axis_name=dp_axes)
        return (
            d_sorted_x_local,
            d_recv_w_local,
            d_wi_local,
            d_wo_local,
            d_wi_0_bias_local,
            d_wi_1_bias_local,
            d_wo_bias_local,
        )

    if has_bias:
        bwd_out_specs = (
            ep3_spec,
            ep2_spec,
            kernel_spec,
            kernel_spec,
            bias_spec,
            bias_spec,
            bias_spec,
        )
    else:
        bwd_out_specs = (ep3_spec, ep2_spec, kernel_spec, kernel_spec, None, None, None)

    (
        d_sorted_x,
        d_recv_w_from_intermediate,
        d_wi,
        d_wo,
        d_wi_0_bias,
        d_wi_1_bias,
        d_wo_bias,
    ) = shard_map(
        _ffn_bwd_body,
        mesh=mesh,
        in_specs=bwd_in_specs,
        out_specs=bwd_out_specs,
        check_rep=False,
    )(*bwd_in_args)

    d_recv_w_total = d_recv_w_from_combine + d_recv_w_from_intermediate

    # ---------------- Dispatch bwd (global view) ----------------
    d_sorted_x = jax.lax.with_sharding_constraint(d_sorted_x, NamedSharding(mesh, ep3_spec))
    d_recv_w_total = jax.lax.with_sharding_constraint(d_recv_w_total, NamedSharding(mesh, ep2_spec))
    dispatch_weight_cotangent = (
        jnp.zeros_like(d_recv_w_total) if _zero_dispatch_weight_grad else d_recv_w_total
    )
    d_x_from_dispatch, d_topk_w = tex.ep_dispatch_bwd(
        ctx.cfg,
        bwd_handle_mem,
        d_sorted_x,
        dispatch_weight_cotangent,
        num_local_tokens=(B, S),
        out_partition_spec=out_partition_spec,
    )

    if _validate_ep_token_roundtrip:
        roundtrip_topk_idx = jnp.argsort(ctx.routing_map, axis=-1)[..., -K:]
        roundtrip_topk_idx_3d = roundtrip_topk_idx.reshape(B, S, K).astype(jnp.int32)
        roundtrip_topk_idx_3d = jax.lax.with_sharding_constraint(
            roundtrip_topk_idx_3d, NamedSharding(mesh, ep3_spec)
        )
        roundtrip_tokens, _ = tex.ep_dispatch_bwd(
            ctx.cfg,
            bwd_handle_mem,
            ctx.recv_token_probe,
            jnp.zeros_like(ctx.recv_topk_weights),
            num_local_tokens=(B, S),
            out_partition_spec=out_partition_spec,
        )
        expected_roundtrip = (
            ctx.x[..., :_EP_ROUTING_PROBE_WIDTH].astype(jnp.float32) * float(K)
        ).astype(roundtrip_tokens.dtype)
        _print_ep_routing_probe_result(
            "dispatch_token_roundtrip",
            roundtrip_topk_idx_3d,
            roundtrip_tokens,
            expected_roundtrip,
            tolerance=6.25e-2,
        )

    if _validate_ep_routing:
        probe_topk_idx = jnp.argsort(ctx.routing_map, axis=-1)[..., -K:]
        probe_topk_idx_3d = probe_topk_idx.reshape(B, S, K).astype(jnp.int32)
        probe_topk_idx_3d = jax.lax.with_sharding_constraint(
            probe_topk_idx_3d, NamedSharding(mesh, ep3_spec)
        )
        probe_code_table = _ep_routing_probe_code_table(num_experts, d_sorted_x.dtype)
        probe_packed = _ep_routing_probe_packed_codes(
            ctx.token_counts,
            recv_pr,
            num_ep,
            num_local_experts,
            d_sorted_x.dtype,
        )
        probe_packed = jax.lax.with_sharding_constraint(
            probe_packed, NamedSharding(mesh, ep3_spec)
        )
        probe_dispatch, _ = tex.ep_dispatch_bwd(
            ctx.cfg,
            bwd_handle_mem,
            probe_packed,
            jnp.zeros_like(ctx.recv_topk_weights),
            num_local_tokens=(B, S),
            out_partition_spec=out_partition_spec,
        )
        expected_probe_dispatch = jnp.sum(
            probe_code_table[probe_topk_idx_3d].astype(jnp.float32),
            axis=-2,
        ).astype(d_sorted_x.dtype)
        _print_ep_routing_probe_result(
            "dispatch_bwd",
            probe_topk_idx_3d,
            probe_dispatch,
            expected_probe_dispatch,
            tolerance=1.0e-3,
        )

    if _debug_moe_numerics or _debug_moe_input_grad:
        d_sorted_x_stats = _debug_stable_stats(d_sorted_x, active_recv_rows)
        d_recv_combine_stats = _debug_stable_stats(
            d_recv_w_from_combine, active_recv_rows
        )
        d_recv_intermediate_stats = _debug_stable_stats(
            d_recv_w_from_intermediate, active_recv_rows
        )
        d_recv_input_stats = _debug_stable_stats(
            dispatch_weight_cotangent, active_recv_rows
        )
        d_dispatch_stats = _debug_stable_stats(d_x_from_dispatch)
        d_topk_stats = _debug_stable_stats(d_topk_w)
        jax.debug.print(
            "[TE dispatch-bwd stats] "
            "token_in(absmean={token_in_absmean:.3e},std={token_in_std:.3e},"
            "absmax={token_in_absmax:.3e},finite={token_in_finite:.6f}) "
            "weight_combine(absmean={wc_absmean:.3e},absmax={wc_absmax:.3e}) "
            "weight_ffn(absmean={wf_absmean:.3e},absmax={wf_absmax:.3e}) "
            "weight_input(absmean={wi_absmean:.3e},absmax={wi_absmax:.3e}) "
            "token_out(absmean={token_out_absmean:.3e},std={token_out_std:.3e},"
            "absmax={token_out_absmax:.3e},finite={token_out_finite:.6f}) "
            "weight_out(absmean={weight_out_absmean:.3e},std={weight_out_std:.3e},"
            "absmax={weight_out_absmax:.3e},finite={weight_out_finite:.6f})",
            token_in_absmean=d_sorted_x_stats[1],
            token_in_std=d_sorted_x_stats[2],
            token_in_absmax=d_sorted_x_stats[3],
            token_in_finite=d_sorted_x_stats[4],
            wc_absmean=d_recv_combine_stats[1],
            wc_absmax=d_recv_combine_stats[3],
            wf_absmean=d_recv_intermediate_stats[1],
            wf_absmax=d_recv_intermediate_stats[3],
            wi_absmean=d_recv_input_stats[1],
            wi_absmax=d_recv_input_stats[3],
            token_out_absmean=d_dispatch_stats[1],
            token_out_std=d_dispatch_stats[2],
            token_out_absmax=d_dispatch_stats[3],
            token_out_finite=d_dispatch_stats[4],
            weight_out_absmean=d_topk_stats[1],
            weight_out_std=d_topk_stats[2],
            weight_out_absmax=d_topk_stats[3],
            weight_out_finite=d_topk_stats[4],
            ordered=False,
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

    if _debug_moe_numerics or _debug_moe_input_grad:
        sparse_prob_stats = _debug_stable_stats(d_sparse_probs)
        logits_stats = _debug_stable_stats(d_logits_2d)
        jax.debug.print(
            "[TE router-bwd stats] "
            "sparse_in(absmean={sparse_absmean:.3e},std={sparse_std:.3e},"
            "absmax={sparse_absmax:.3e},finite={sparse_finite:.6f}) "
            "logits_out(absmean={logits_absmean:.3e},std={logits_std:.3e},"
            "absmax={logits_absmax:.3e},finite={logits_finite:.6f})",
            sparse_absmean=sparse_prob_stats[1],
            sparse_std=sparse_prob_stats[2],
            sparse_absmax=sparse_prob_stats[3],
            sparse_finite=sparse_prob_stats[4],
            logits_absmean=logits_stats[1],
            logits_std=logits_stats[2],
            logits_absmax=logits_stats[3],
            logits_finite=logits_stats[4],
            ordered=False,
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
        zero_routing_map = jnp.zeros(ctx.aux_saved_scores.shape, dtype=ctx.routing_map.dtype)
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
    if _zero_moe_input_grad:
        d_x = jnp.zeros_like(d_x)

    if _debug_moe_numerics or _debug_moe_input_grad:
        dispatch_stats = _debug_stable_stats(d_x_from_dispatch)
        gate_stats = _debug_stable_stats(d_x_from_gate)
        total_stats = _debug_stable_stats(d_x)
        jax.debug.print(
            "[TE input-grad stats] "
            "dispatch(absmean={dispatch_absmean:.3e},std={dispatch_std:.3e},"
            "absmax={dispatch_absmax:.3e}) "
            "gate(absmean={gate_absmean:.3e},std={gate_std:.3e},absmax={gate_absmax:.3e}) "
            "total(absmean={total_absmean:.3e},std={total_std:.3e},"
            "absmax={total_absmax:.3e},finite={total_finite:.6f})",
            dispatch_absmean=dispatch_stats[1],
            dispatch_std=dispatch_stats[2],
            dispatch_absmax=dispatch_stats[3],
            gate_absmean=gate_stats[1],
            gate_std=gate_stats[2],
            gate_absmax=gate_stats[3],
            total_absmean=total_stats[1],
            total_std=total_stats[2],
            total_absmax=total_stats[3],
            total_finite=total_stats[4],
            ordered=False,
        )

    # Pin output grads to the declared logical axes so downstream
    # optimizers see consistent shardings.
    d_x = with_sharding_constraint_by_logical_axes(d_x, input_axes)
    d_gate_kernel = with_sharding_constraint_by_logical_axes(d_gate_kernel, gate_kernel_axes)
    d_wi = with_sharding_constraint_by_logical_axes(d_wi, wi_kernel_axes)
    d_wo = with_sharding_constraint_by_logical_axes(d_wo, wo_kernel_axes)
    if has_bias:
        wi_bias_axes = (wi_kernel_axes[0], *wi_kernel_axes[2:])
        wo_bias_axes = (wo_kernel_axes[0], *wo_kernel_axes[2:])
        d_wi_0_bias = with_sharding_constraint_by_logical_axes(d_wi_0_bias, wi_bias_axes)
        d_wi_1_bias = with_sharding_constraint_by_logical_axes(d_wi_1_bias, wi_bias_axes)
        d_wo_bias = with_sharding_constraint_by_logical_axes(d_wo_bias, wo_bias_axes)

    # expert_bias has no learnable bwd path through fused_topk: the
    # primitive's bwd returns None for the bias slot. Match that with a
    # zero cotangent of the right shape so custom_vjp's arity check
    # passes.
    d_expert_bias = jnp.zeros_like(ctx.expert_bias)

    return (
        d_x,
        d_gate_kernel,
        d_wi,
        d_wo,
        d_wi_0_bias if has_bias else None,
        d_wi_1_bias if has_bias else None,
        d_wo_bias if has_bias else None,
        d_expert_bias,
        ctx.quantizer_sets,
    )


# =============================================================================
# custom_vjp + public entry
# =============================================================================


@partial(jax.custom_vjp, nondiff_argnums=tuple(range(9, 27)))
def _moe(
    x,
    gate_kernel,
    wi,
    wo,
    wi_0_bias,
    wi_1_bias,
    wo_bias,
    expert_bias,
    quantizer_sets,
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
    recv_capacity_per_rank,
):
    primal, _ = _moe_fwd_rule(
        x,
        gate_kernel,
        wi,
        wo,
        wi_0_bias,
        wi_1_bias,
        wo_bias,
        expert_bias,
        quantizer_sets,
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
        recv_capacity_per_rank,
    )
    return primal


_moe.defvjp(_moe_fwd_rule, _moe_bwd_rule)


def moe(
    x: jnp.ndarray,
    gate_kernel: jnp.ndarray,
    wi: jnp.ndarray,
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
    quantizer_sets: Tuple[QuantizerSet, QuantizerSet] = (
        noop_quantizer_set,
        noop_quantizer_set,
    ),
    ep_axis: str,
    data_parallelism_axes: Tuple[str, ...] = (),
    input_axes: Tuple[Optional[str], ...] = (),
    gate_kernel_axes: Tuple[Optional[str], ...] = (),
    wi_kernel_axes: Tuple[Optional[str], ...] = ("exp", "embed", "mlp"),
    wo_kernel_axes: Tuple[Optional[str], ...] = ("exp", "mlp", "embed"),
    dtype: jnp.dtype = jnp.float32,
    recv_capacity_per_rank: Optional[int] = None,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], jnp.ndarray]:
    """Run a full MoE block under a single fused custom_vjp on the TE EP path.

    Returns ``(output, aux_loss, total_recv_tokens)``. ``aux_loss`` is ``None``
    when ``aux_loss_coeff == 0``, else a 0-d scalar. ``total_recv_tokens`` is a
    non-differentiable pre-drop recv-slot total (grad ``None``); see
    ``ep_dispatch`` for using it to detect overflow.

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
    quantizer_sets : Tuple[QuantizerSet, QuantizerSet]
        Independent FC1 and FC2 quantizer sets. They are differentiable
        custom-VJP arguments so recipe state is threaded through backward.
    recv_capacity_per_rank : Optional[int]
        Exact aligned receive-buffer capacity for each EP rank. ``None``
        (default) reserves the dropless aligned worst case. The value must match
        the capacity used by ``ep_bootstrap``. Overflow is reported through
        ``total_recv_tokens`` when bootstrap used ``drop_on_overflow=True``.

    Note that the per-expert dispatch-slot alignment is fixed internally
    at 128 tokens (``_ALIGN_SIZE``); see that constant's docstring for
    rationale and how to extend if a future recipe needs >128.

    Axis-name parameters:

    * ``ep_axis`` and ``data_parallelism_axes`` are *physical mesh
      axis names* -- they index ``jax.sharding.Mesh.shape`` directly
      (to compute ``num_ep`` / ``dp_size`` and to construct
      ``P((dp..., ep), None, None)`` for the physical
      ``jax.lax.with_sharding_constraint`` calls that JAX requires
      to refer to real mesh axes).
    * ``input_axes``, ``gate_kernel_axes``, ``wi_kernel_axes``,
      ``wo_kernel_axes`` are *logical axis names* (e.g.
      ``"batch"``, ``"embed"``, ``"mlp"``, ``"exp"``) -- they get
      resolved via the active Flax logical-axis rules and consumed
      by ``with_sharding_constraint_by_logical_axes``. They are
      ``Optional[str]`` tuples so a rule of ``None`` means
      "replicated on this axis".

    Logical-axis support for ``ep_axis`` / ``data_parallelism_axes``
    is intentionally out of scope: the EP comm-group construction
    (``dp_color = rank // ep_size``) and the bootstrap signature
    check both require concrete integer sizes, so a logical name
    would have to be resolved to a physical one anyway before any
    EP primitive is called. If a downstream pipeline needs to plumb
    logical names all the way to ``moe()``, do the rule lookup at
    the call site.

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
    expected_leading: Any = (*data_parallelism_axes, ep_axis) if data_parallelism_axes else ep_axis
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
    x = _with_sharding_constraint_cast_bwd(x, NamedSharding(mesh, expected_spec))

    # custom_vjp can't trace through None args; lower expert_bias to an
    # empty shape-(0,) tensor that fused_topk_with_score_function treats
    # as "no bias".
    if expert_bias is None:
        expert_bias_arg = jnp.zeros((0,), dtype=jnp.float32)
    else:
        expert_bias_arg = expert_bias.astype(jnp.float32)

    output, aux_loss, total_recv_tokens = _moe(
        x,
        gate_kernel,
        wi,
        wo,
        wi_0_bias,
        wi_1_bias,
        wo_bias,
        expert_bias_arg,
        quantizer_sets,
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
        recv_capacity_per_rank,
    )
    if aux_loss_coeff <= 0.0:
        aux_loss = None
    assert output.dtype == x.dtype, f"moe() output dtype {output.dtype} != input dtype {x.dtype}"
    return output, aux_loss, total_recv_tokens
