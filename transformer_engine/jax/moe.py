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

Out-of-scope (for now)
----------------------
FP8 / MXFP8 quantizer sets are not yet wired on this path; turning
them on requires recipe-aware residual handling for ``ScaledTensor``
leaves. ``aux_loss_coeff`` and ``expert_bias`` are supported (the former
forces a per-step all-gather over the routing-side logits, which lives
off the critical path and overlaps with the dispatch collective).
"""

from functools import partial
from typing import Any, Optional, Tuple, Union
import warnings

import flax.struct
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


# Per-expert dispatch-slot alignment fed to ``tex.ep_prepare`` as
# ``dispatch_output_per_expert_alignment``. NCCL EP HT requires the
# per-expert recv block to be at least 128-token aligned, and all current
# TE grouped-GEMM recipes (bf16/fp16/fp8/mxfp8) are satisfied by the
# same 128-token tile, so a single constant covers every supported path.
_ALIGN_SIZE = 128


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
    casted_sorted_x_lhs_trans: Any
    casted_wi_rhs_trans: Any
    gate_proj_out: jnp.ndarray
    up_proj_out: jnp.ndarray
    casted_intermediate_lhs_trans: Any
    casted_wo_rhs_trans: Any
    expert_outputs: jnp.ndarray
    local_group_sizes: jnp.ndarray
    aux_const_buf: Any = None
    aux_tokens_per_expert: Any = None
    aux_saved_scores: Any = None


# =============================================================================
# Global-view FFN body
# =============================================================================


def _ffn_fwd_global(
    recv_tokens: jnp.ndarray,
    recv_topk_weights: jnp.ndarray,
    token_counts: jnp.ndarray,
    wi_0: jnp.ndarray,
    wi_1: jnp.ndarray,
    wo: jnp.ndarray,
    wi_0_bias: Optional[jnp.ndarray],
    wi_1_bias: Optional[jnp.ndarray],
    wo_bias: Optional[jnp.ndarray],
    *,
    dp_size: int,
    num_ep: int,
    num_local_experts: int,
    activation_type: str,
    apply_topk_weights_early: bool,
    flat_token_sharding: NamedSharding,
    flat_group_sharding: NamedSharding,
    grouped_weight_sharding: NamedSharding,
    grouped_bias_sharding: NamedSharding,
):
    """Run the FFN on global EP-dispatch buffers.

    Grouped-operation custom partitioning lowers the global operands to
    the per-device problem. ``token_counts`` from ``tex.ep_prepare`` is
    passed through as the dynamic grouped-GEMM group sizes, so cuBLAS
    skips both 0-token experts and dispatch-buffer over-allocation.
    """
    hidden = recv_tokens.shape[-1]
    sorted_x = recv_tokens.reshape(-1, hidden)
    recv_w_flat = recv_topk_weights.reshape(-1)
    group_sizes = token_counts.reshape(-1).astype(jnp.int32)
    sorted_x = jax.lax.with_sharding_constraint(sorted_x, flat_token_sharding)
    recv_w_flat = jax.lax.with_sharding_constraint(recv_w_flat, flat_group_sharding)
    group_sizes = jax.lax.with_sharding_constraint(group_sizes, flat_group_sharding)

    wi_0 = wi_0.astype(sorted_x.dtype)
    wi_1 = wi_1.astype(sorted_x.dtype)
    wo = wo.astype(sorted_x.dtype)

    # Dispatch groups flatten in (dp, ep, local_expert) order. Broadcast
    # each global expert parameter set over the outer DP dimension before
    # flattening so the grouped weights have exactly the same ordering.
    num_groups = dp_size * num_ep * num_local_experts

    def _broadcast_experts(value, sharding):
        value = jnp.broadcast_to(
            value.reshape(1, num_ep, num_local_experts, *value.shape[1:]),
            (dp_size, num_ep, num_local_experts, *value.shape[1:]),
        ).reshape(num_groups, *value.shape[1:])
        return jax.lax.with_sharding_constraint(value, sharding)

    wi_0 = _broadcast_experts(wi_0, grouped_weight_sharding)
    wi_1 = _broadcast_experts(wi_1, grouped_weight_sharding)
    wo = _broadcast_experts(wo, grouped_weight_sharding)
    if wi_0_bias is not None:
        wi_0_bias = _broadcast_experts(wi_0_bias, grouped_bias_sharding)
        wi_1_bias = _broadcast_experts(wi_1_bias, grouped_bias_sharding)
        wo_bias = _broadcast_experts(wo_bias, grouped_bias_sharding)

    # Concat wi_0/wi_1 along the trailing axis (NOT stack on a new
    # axis). grouped_gemm requires the 3D (G, K, N) weight layout with
    # contracting_dims=((1,), (1,)); a 4D stack variant walks off the
    # end of the RHS and returns NaN.
    wi_combined = jnp.concatenate([wi_0, wi_1], axis=-1)
    wi_combined_bias = (
        jnp.concatenate([wi_0_bias, wi_1_bias], axis=-1) if wi_0_bias is not None else None
    )

    q_set = noop_quantizer_set
    casted_sorted_x = tex.grouped_quantize(sorted_x, q_set.x, group_sizes, flatten_axis=-1)
    casted_wi = tex.grouped_quantize(wi_combined, q_set.kernel, flatten_axis=-1)
    combined_out = tex.grouped_gemm(
        casted_sorted_x.get_tensor(usage=TensorUsage.LHS),
        casted_wi.get_tensor(usage=TensorUsage.RHS),
        contracting_dims=((1,), (1,)),
        bias=wi_combined_bias,
    )
    combined_out = jax.lax.with_sharding_constraint(combined_out, flat_token_sharding)
    gate_proj_out, up_proj_out = jnp.split(combined_out, 2, axis=-1)
    casted_sorted_x_lhs_trans = casted_sorted_x.get_tensor(usage=TensorUsage.LHS_TRANS)
    casted_wi_rhs_trans = casted_wi.get_tensor(usage=TensorUsage.RHS_TRANS)

    # Activation inputs (gate_proj_out, up_proj_out) stay in the wi GEMM
    # output dtype; the activation output (`intermediate`) stays in the
    # dtype the wo GEMM / wo's quantized input consumes. For bf16 compute
    # that's all bf16; for FP8/FP4 the downstream grouped_quantize is what
    # transitions to the target precision.
    act_fn = _convert_to_activation_function(activation_type)
    intermediate = act_fn(gate_proj_out) * up_proj_out
    intermediate = jax.lax.with_sharding_constraint(intermediate, flat_token_sharding)

    if apply_topk_weights_early:
        # Fold the per-token combine weights into the FFN intermediate;
        # the downstream wo GEMM is linear so this is equivalent to the
        # late-weighting path. Padded recv slots can contain uninitialized
        # data, so overwrite inactive rows with literal zeros instead of
        # relying on multiplication by a zero mask (IEEE NaN * 0 = NaN).
        # ``w_b`` is cast to ``intermediate.dtype`` so the multiply doesn't
        # promote expert_outputs above the EP buffer's element width.
        w_b = recv_w_flat[:, None].astype(intermediate.dtype)
        active = (recv_w_flat != 0)[:, None]
        intermediate = jnp.where(active, intermediate * w_b, jnp.zeros_like(intermediate))

    casted_intermediate = tex.grouped_quantize(intermediate, q_set.x, group_sizes, flatten_axis=-1)
    casted_wo = tex.grouped_quantize(wo, q_set.kernel, flatten_axis=-1)
    expert_outputs = tex.grouped_gemm(
        casted_intermediate.get_tensor(usage=TensorUsage.LHS),
        casted_wo.get_tensor(usage=TensorUsage.RHS),
        contracting_dims=((1,), (1,)),
        bias=wo_bias,
    )
    expert_outputs = jax.lax.with_sharding_constraint(expert_outputs, flat_token_sharding)
    casted_intermediate_lhs_trans = casted_intermediate.get_tensor(usage=TensorUsage.LHS_TRANS)
    casted_wo_rhs_trans = casted_wo.get_tensor(usage=TensorUsage.RHS_TRANS)

    expert_outputs_3d = expert_outputs.reshape(*recv_tokens.shape[:-1], expert_outputs.shape[-1])
    group_sizes_nd = group_sizes.reshape(token_counts.shape)
    residuals = (
        casted_sorted_x_lhs_trans,
        casted_wi_rhs_trans,
        gate_proj_out,
        up_proj_out,
        casted_intermediate_lhs_trans,
        casted_wo_rhs_trans,
        group_sizes_nd,
    )
    return expert_outputs_3d, residuals


def _ffn_bwd_global(
    d_expert_outputs: jnp.ndarray,
    casted_sorted_x_lhs_trans,
    casted_wi_rhs_trans,
    gate_proj_out: jnp.ndarray,
    up_proj_out: jnp.ndarray,
    casted_intermediate_lhs_trans,
    casted_wo_rhs_trans,
    local_group_sizes: jnp.ndarray,
    recv_topk_weights: jnp.ndarray,
    *,
    activation_type: str,
    apply_topk_weights_early: bool,
    has_bias: bool,
    flat_token_sharding: NamedSharding,
    flat_group_sharding: NamedSharding,
    grouped_weight_sharding: NamedSharding,
    grouped_bias_sharding: NamedSharding,
):
    """Run the FFN backward on global residuals.

    Mirrors :func:`_ffn_fwd_global`. Returns
    ``(d_sorted_x [num_procs, recv_pr, H], d_recv_w [num_procs, recv_pr],
    d_wi_0, d_wi_1, d_wo, d_wi_0_bias, d_wi_1_bias, d_wo_bias)``.
    """
    group_sizes = local_group_sizes.reshape(-1).astype(jnp.int32)
    d_eo_2d = d_expert_outputs.reshape(-1, d_expert_outputs.shape[-1])
    recv_w_flat = recv_topk_weights.reshape(-1)
    d_eo_2d = jax.lax.with_sharding_constraint(d_eo_2d, flat_token_sharding)
    recv_w_flat = jax.lax.with_sharding_constraint(recv_w_flat, flat_group_sharding)
    group_sizes = jax.lax.with_sharding_constraint(group_sizes, flat_group_sharding)
    q_set = noop_quantizer_set
    # cuBLAS grouped_gemm skips size_g == 0 groups without zero-filling
    # the output slice; mask 0-token-expert wgrads to zero so the
    # optimizer never sees uninit memory.
    wgrad_group_active = (group_sizes > 0)[:, None, None]

    # wo bwd
    casted_d_eo = tex.grouped_quantize(d_eo_2d, q_set.dgrad, group_sizes, flatten_axis=-1)
    _casted_d_eo_lhs = casted_d_eo.get_tensor(usage=TensorUsage.LHS)
    _casted_d_eo_rhs = casted_d_eo.get_tensor(usage=TensorUsage.RHS)
    d_intermediate = tex.grouped_gemm(
        _casted_d_eo_lhs,
        casted_wo_rhs_trans,
        contracting_dims=((1,), (2,)),
    )
    d_intermediate = jax.lax.with_sharding_constraint(d_intermediate, flat_token_sharding)
    d_wo = tex.grouped_gemm(
        casted_intermediate_lhs_trans,
        _casted_d_eo_rhs,
        contracting_dims=((0,), (0,)),
    )
    d_wo = jnp.where(wgrad_group_active, d_wo, jnp.zeros_like(d_wo))
    d_wo = jax.lax.with_sharding_constraint(d_wo, grouped_weight_sharding)
    d_wo_bias = tex.grouped_dbias(d_eo_2d, group_sizes) if has_bias else None
    if has_bias:
        d_wo_bias = jax.lax.with_sharding_constraint(d_wo_bias, grouped_bias_sharding)

    act_fn = _convert_to_activation_function(activation_type)
    if apply_topk_weights_early:
        # intermediate' = intermediate * w * mask. Split the cotangent
        # across both factors before the activation bwd consumes it. Padded
        # recv slots may still be NaN in the saved activation residuals, so
        # use zero-filled residuals on inactive rows before the activation VJP.
        w_b = recv_w_flat[:, None].astype(d_intermediate.dtype)
        active = (recv_w_flat != 0)[:, None]
        gate_proj_for_bwd = jnp.where(active, gate_proj_out, jnp.zeros_like(gate_proj_out))
        up_proj_for_bwd = jnp.where(active, up_proj_out, jnp.zeros_like(up_proj_out))
        intermediate_unweighted = act_fn(gate_proj_for_bwd) * up_proj_for_bwd
        d_recv_w_from_intermediate = jnp.sum(
            d_intermediate * intermediate_unweighted,
            axis=-1,
        ).astype(recv_w_flat.dtype)
        d_intermediate = jnp.where(active, d_intermediate * w_b, jnp.zeros_like(d_intermediate))
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
    # wgrad result back into d_wi_0 / d_wi_1 halves with jnp.split.
    d_combined = jnp.concatenate([d_gate_proj_out, d_up_proj_out], axis=-1)
    d_combined = jax.lax.with_sharding_constraint(d_combined, flat_token_sharding)
    casted_d_combined = tex.grouped_quantize(d_combined, q_set.dgrad, group_sizes, flatten_axis=-1)
    d_sorted_x = tex.grouped_gemm(
        casted_d_combined.get_tensor(usage=TensorUsage.LHS),
        casted_wi_rhs_trans,
        contracting_dims=((1,), (2,)),
    )
    d_sorted_x = jax.lax.with_sharding_constraint(d_sorted_x, flat_token_sharding)
    d_wi_combined = tex.grouped_gemm(
        casted_sorted_x_lhs_trans,
        casted_d_combined.get_tensor(usage=TensorUsage.RHS),
        contracting_dims=((0,), (0,)),
    )
    d_wi_combined = jnp.where(wgrad_group_active, d_wi_combined, jnp.zeros_like(d_wi_combined))
    d_wi_combined = jax.lax.with_sharding_constraint(d_wi_combined, grouped_weight_sharding)
    d_wi_0, d_wi_1 = jnp.split(d_wi_combined, 2, axis=-1)
    if has_bias:
        d_wi_combined_bias = tex.grouped_dbias(d_combined, group_sizes)
        d_wi_combined_bias = jax.lax.with_sharding_constraint(
            d_wi_combined_bias, grouped_bias_sharding
        )
        d_wi_0_bias, d_wi_1_bias = jnp.split(d_wi_combined_bias, 2, axis=-1)
    else:
        d_wi_0_bias = None
        d_wi_1_bias = None

    d_sorted_x_3d = d_sorted_x.reshape(*d_expert_outputs.shape[:-1], d_sorted_x.shape[-1])
    d_recv_w_3d = d_recv_w_from_intermediate.reshape(recv_topk_weights.shape)
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
):
    """Forward: gate -> topk -> ep_dispatch -> FFN -> ep_combine.

    Returns ``(output, aux_loss)``. ``aux_loss`` is a zero scalar when
    ``aux_loss_coeff == 0``.
    """
    del gate_kernel_axes, wi_kernel_axes, wo_kernel_axes  # used in bwd only

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

    # Per-rank send capacity: B/num_procs rows x S tokens per rank.
    max_tokens_per_rank = (B // num_procs) * S
    # Per-rank receive capacity. NCCL EP HT expert-major lays out variable
    # per-expert zones in one flat recv buffer, with each non-empty zone padded
    # to ``dispatch_output_per_expert_alignment``.
    tokens_per_ep_group = num_ep * max_tokens_per_rank
    max_local_assignments = tokens_per_ep_group * min(K, num_local_experts)
    max_nonempty_experts = min(num_local_experts, max_local_assignments)
    padded_total_bound = max_local_assignments + (_ALIGN_SIZE - 1) * max_nonempty_experts
    aligned_total_bound = ((padded_total_bound + _ALIGN_SIZE - 1) // _ALIGN_SIZE) * _ALIGN_SIZE
    per_expert_bound = (
        num_local_experts * ((tokens_per_ep_group + _ALIGN_SIZE - 1) // _ALIGN_SIZE) * _ALIGN_SIZE
    )
    recv_pr = min(per_expert_bound, aligned_total_bound)

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
    flat_token_sharding = NamedSharding(mesh, P(batch_pspec_axis, None))
    flat_group_sharding = NamedSharding(mesh, P(batch_pspec_axis))
    grouped_weight_sharding = NamedSharding(mesh, P(batch_pspec_axis, None, None))
    grouped_bias_sharding = NamedSharding(mesh, P(batch_pspec_axis, None))
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
    token_counts, handle_mem = tex.ep_prepare(cfg, topk_idx_3d)
    token_counts = jax.lax.with_sharding_constraint(token_counts, NamedSharding(mesh, ep2_spec))
    recv_tokens, recv_topk_weights = tex.ep_dispatch_fwd(
        cfg, handle_mem, topk_idx_3d, x, topk_w_3d, recv_pr
    )
    recv_tokens = jax.lax.with_sharding_constraint(recv_tokens, NamedSharding(mesh, ep3_spec))
    recv_topk_weights = jax.lax.with_sharding_constraint(
        recv_topk_weights, NamedSharding(mesh, ep2_spec)
    )

    # ---------------- FFN (global view, custom-partitioned primitives) ----------------
    has_bias = wi_0_bias is not None
    # The NCCL EP receive buffer may contain uninitialized padded slots.
    # Dynamic group sizes keep grouped GEMMs from reading those rows; the
    # backward masks skipped wgrad groups, while EP combine/dispatch bwd
    # consume only positions described by handle_mem.
    expert_outputs, ffn_residuals = _ffn_fwd_global(
        recv_tokens,
        recv_topk_weights,
        token_counts,
        wi_0,
        wi_1,
        wo,
        wi_0_bias if has_bias else None,
        wi_1_bias if has_bias else None,
        wo_bias if has_bias else None,
        dp_size=dp_size,
        num_ep=num_ep,
        num_local_experts=num_local_experts,
        activation_type=activation_type,
        apply_topk_weights_early=apply_topk_weights_early,
        flat_token_sharding=flat_token_sharding,
        flat_group_sharding=flat_group_sharding,
        grouped_weight_sharding=grouped_weight_sharding,
        grouped_bias_sharding=grouped_bias_sharding,
    )
    expert_outputs = jax.lax.with_sharding_constraint(expert_outputs, NamedSharding(mesh, ep3_spec))

    # ---------------- TE EP combine (global view) ----------------
    out_partition_spec = (batch_pspec_axis, None, None)
    if apply_topk_weights_early:
        # expert_outputs is already weighted upstream.
        output = tex.ep_combine_fwd(
            cfg,
            handle_mem,
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
        cfg=cfg,
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
    residuals,
    cotangents,
):
    """Backward mirror of :func:`_moe_fwd_rule`."""
    del num_groups, group_topk, dtype  # captured in residuals / unused in bwd

    d_output, d_aux_loss = cotangents

    ctx, static = residuals
    has_bias = static["has_bias"]
    x_shape = static["x_shape"]
    recv_pr = static["recv_pr"]

    mesh = _get_mesh()
    if mesh is None or mesh.empty:
        raise ValueError("moe(...) requires an active jax.sharding.Mesh.")
    num_ep = mesh.shape[ep_axis]
    num_local_experts = num_experts // num_ep
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
    flat_token_sharding = NamedSharding(mesh, P(batch_pspec_axis, None))
    flat_group_sharding = NamedSharding(mesh, P(batch_pspec_axis))
    grouped_weight_sharding = NamedSharding(mesh, P(batch_pspec_axis, None, None))
    grouped_bias_sharding = NamedSharding(mesh, P(batch_pspec_axis, None))
    out_partition_spec = (batch_pspec_axis, None, None)

    # ---------------- Combine bwd (global view) ----------------
    d_output = jax.lax.with_sharding_constraint(d_output, NamedSharding(mesh, ep3_spec))
    grad_pre_combine = tex.ep_combine_bwd(ctx.cfg, ctx.handle_mem, d_output, recv_pr)
    grad_pre_combine = jax.lax.with_sharding_constraint(
        grad_pre_combine, NamedSharding(mesh, ep3_spec)
    )

    if apply_topk_weights_early:
        # combine_fwd consumed already-weighted expert_outputs; the recv_w
        # cotangent flows through the early-weighting step inside the FFN bwd.
        d_expert_outputs = grad_pre_combine
        d_recv_w_from_combine = jnp.zeros_like(ctx.recv_topk_weights)
    else:
        # Reverse the late-weighting multiply. Padded expert-major rows are
        # part of the physical grouped-GEMM ranges, so write literal zero
        # cotangents for inactive rows instead of relying on NaN * 0.
        w = ctx.recv_topk_weights[..., None].astype(grad_pre_combine.dtype)
        mask_bool = (ctx.recv_topk_weights != 0)[..., None]
        d_expert_outputs = jnp.where(
            mask_bool, grad_pre_combine * w, jnp.zeros_like(grad_pre_combine)
        )
        d_recv_w_from_combine = (grad_pre_combine * ctx.expert_outputs).sum(axis=-1)
        d_recv_w_from_combine = d_recv_w_from_combine.astype(ctx.recv_topk_weights.dtype)

    # ---------------- FFN bwd (global view, custom-partitioned primitives) ----------------
    (
        d_sorted_x,
        d_recv_w_from_intermediate,
        d_wi_0,
        d_wi_1,
        d_wo,
        d_wi_0_bias,
        d_wi_1_bias,
        d_wo_bias,
    ) = _ffn_bwd_global(
        d_expert_outputs,
        ctx.casted_sorted_x_lhs_trans,
        ctx.casted_wi_rhs_trans,
        ctx.gate_proj_out,
        ctx.up_proj_out,
        ctx.casted_intermediate_lhs_trans,
        ctx.casted_wo_rhs_trans,
        ctx.local_group_sizes,
        ctx.recv_topk_weights,
        activation_type=activation_type,
        apply_topk_weights_early=apply_topk_weights_early,
        has_bias=has_bias,
        flat_token_sharding=flat_token_sharding,
        flat_group_sharding=flat_group_sharding,
        grouped_weight_sharding=grouped_weight_sharding,
        grouped_bias_sharding=grouped_bias_sharding,
    )

    # The forward broadcast introduced one expert-gradient group per DP
    # replica, ordered (dp, ep, local_expert). Sum that outer dimension
    # to recover the public parameter shapes [num_experts, ...].
    def _fold_dp_groups(grad):
        return (
            grad.reshape(dp_size, num_ep, num_local_experts, *grad.shape[1:])
            .sum(axis=0)
            .reshape(num_experts, *grad.shape[1:])
        )

    d_wi_0 = _fold_dp_groups(d_wi_0)
    d_wi_1 = _fold_dp_groups(d_wi_1)
    d_wo = _fold_dp_groups(d_wo)
    if has_bias:
        d_wi_0_bias = _fold_dp_groups(d_wi_0_bias)
        d_wi_1_bias = _fold_dp_groups(d_wi_1_bias)
        d_wo_bias = _fold_dp_groups(d_wo_bias)

    d_recv_w_total = d_recv_w_from_combine + d_recv_w_from_intermediate

    # ---------------- Dispatch bwd (global view) ----------------
    d_sorted_x = jax.lax.with_sharding_constraint(d_sorted_x, NamedSharding(mesh, ep3_spec))
    d_recv_w_total = jax.lax.with_sharding_constraint(d_recv_w_total, NamedSharding(mesh, ep2_spec))
    d_x_from_dispatch, d_topk_w = tex.ep_dispatch_bwd(
        ctx.cfg,
        ctx.handle_mem,
        d_sorted_x,
        d_recv_w_total,
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

    # Pin output grads to the declared logical axes so downstream
    # optimizers see consistent shardings.
    d_x = with_sharding_constraint_by_logical_axes(d_x, input_axes)
    d_gate_kernel = with_sharding_constraint_by_logical_axes(d_gate_kernel, gate_kernel_axes)
    d_wi_0 = with_sharding_constraint_by_logical_axes(d_wi_0, wi_kernel_axes)
    d_wi_1 = with_sharding_constraint_by_logical_axes(d_wi_1, wi_kernel_axes)
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


@partial(jax.custom_vjp, nondiff_argnums=tuple(range(9, 26)))
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
    )
    if aux_loss_coeff <= 0.0:
        aux_loss = None
    assert output.dtype == x.dtype, f"moe() output dtype {output.dtype} != input dtype {x.dtype}"
    return output, aux_loss
