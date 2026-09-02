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
import warnings
from functools import partial
from typing import Any, Optional, Tuple, Union

import flax.struct
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from . import cpp_extensions as tex
from .quantize import (
    GroupedQuantizer,
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
        raise ValueError(
            "num_experts, num_experts_per_tok, and max_tokens_per_rank must be positive"
        )
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
    max_local_assignments = tokens_per_ep_group * min(num_experts_per_tok, num_local_experts)
    max_nonempty_experts = min(num_local_experts, max_local_assignments)
    padded_total_bound = max_local_assignments + (alignment - 1) * max_nonempty_experts
    aligned_total_bound = ((padded_total_bound + alignment - 1) // alignment) * alignment
    per_expert_bound = (
        num_local_experts * ((tokens_per_ep_group + alignment - 1) // alignment) * alignment
    )
    worst_case = min(per_expert_bound, aligned_total_bound)
    if recv_capacity_factor is None:
        return worst_case

    balanced_per_expert = (
        max_tokens_per_rank * num_experts_per_tok + num_local_experts - 1
    ) // num_local_experts
    balanced_aligned = (
        num_local_experts * ((balanced_per_expert + alignment - 1) // alignment) * alignment
    )
    requested = math.ceil(balanced_aligned * recv_capacity_factor)
    requested = ((requested + alignment - 1) // alignment) * alignment
    return min(requested, worst_case)


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
    recv_topk_weights: jnp.ndarray
    casted_sorted_x_lhs_trans: Any
    casted_wi_rhs_trans: Any
    gate_proj_out: jnp.ndarray
    up_proj_out: jnp.ndarray
    casted_intermediate_lhs_trans: Any
    casted_wo_rhs_trans: Any
    expert_outputs: jnp.ndarray
    local_group_sizes: jnp.ndarray
    quantizer_sets: Any
    aux_const_buf: Any = None
    aux_tokens_per_expert: Any = None
    aux_saved_scores: Any = None


# =============================================================================
# Per-shard FFN body
# =============================================================================


def _validate_moe_quantizer_sets(
    quantizer_sets: Tuple[QuantizerSet, QuantizerSet],
    *,
    num_token_groups: int,
    num_expert_groups: int,
) -> None:
    """Validate the current global-view MoE quantizer contract.

    Quantizers passed to the public MoE API always describe the global logical
    operation. The shard-mapped FFN consumes only its local group count, but it
    must not rewrite that public metadata into a shard-local representation.

    Stateful grouped recipes will eventually require sharded leading group
    dimensions on their internal state. Until that representation exists, MoE
    supports only no-op quantizers and stateless MXFP8 grouped quantizers.
    """
    if not isinstance(quantizer_sets, tuple) or len(quantizer_sets) != 2:
        raise TypeError("MoE quantizer_sets must be a tuple of FC1 and FC2 QuantizerSet objects.")

    expected_groups = {
        "x": num_token_groups,
        "kernel": num_expert_groups,
        "dgrad": num_token_groups,
    }
    for set_name, quantizer_set in zip(("FC1", "FC2"), quantizer_sets):
        if not isinstance(quantizer_set, QuantizerSet):
            raise TypeError(f"MoE {set_name} quantizer must be a QuantizerSet.")
        quantizers = {
            "x": quantizer_set.x,
            "kernel": quantizer_set.kernel,
            "dgrad": quantizer_set.dgrad,
        }
        if all(quantizer is None for quantizer in quantizers.values()):
            continue
        if any(quantizer is None for quantizer in quantizers.values()):
            raise TypeError(
                f"MoE {set_name} must use either all no-op quantizers or all grouped MXFP8 "
                "quantizers."
            )

        for source, quantizer in quantizers.items():
            if not isinstance(quantizer, GroupedQuantizer):
                raise TypeError(
                    f"MoE {set_name} {source} quantizer must be a GroupedQuantizer; "
                    f"got {type(quantizer).__name__}."
                )
            if not quantizer.scaling_mode.is_mxfp8_scaling:
                raise NotImplementedError(
                    "TE MoE currently supports only BF16/no-op and stateless MXFP8 grouped "
                    f"quantizers; {set_name} {source} uses {quantizer.scaling_mode}."
                )
            if jax.tree_util.tree_leaves(quantizer):
                raise NotImplementedError(
                    "TE MoE does not yet support stateful grouped quantizers. Quantizer state "
                    "must first be represented with a sharded global group dimension."
                )
            expected = expected_groups[source]
            if quantizer.n_groups != expected or len(quantizer.quantizers) != expected:
                raise ValueError(
                    f"MoE {set_name} {source} quantizer must describe the global logical "
                    f"group count {expected}; got n_groups={quantizer.n_groups} and "
                    f"{len(quantizer.quantizers)} child quantizers."
                )


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
    combined_out = tex.grouped_gemm(
        casted_sorted_x.get_tensor(usage=TensorUsage.LHS),
        casted_wi.get_tensor(usage=TensorUsage.RHS),
        contracting_dims=((1,), (1,)),
        bias=wi_combined_bias,
    )
    gate_proj_out, up_proj_out = jnp.split(combined_out, 2, axis=-1)

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
        # late-weighting path. Grouped GEMM skips overallocation tail padding automatically.
        # Padding between groups is padded with zeros by NCCL EP.
        intermediate = intermediate * recv_w_flat[:, None].astype(intermediate.dtype)

    casted_intermediate = tex.grouped_quantize(
        intermediate,
        fc2_quantizer_set.x,
        group_sizes,
        flatten_axis=-1,
    )
    casted_wo = tex.grouped_quantize(wo, fc2_quantizer_set.kernel, flatten_axis=-1)
    expert_outputs = tex.grouped_gemm(
        casted_intermediate.get_tensor(usage=TensorUsage.LHS),
        casted_wo.get_tensor(usage=TensorUsage.RHS),
        contracting_dims=((1,), (1,)),
        bias=wo_bias,
    )
    expert_outputs_3d = expert_outputs.reshape(1, expert_outputs.shape[0], expert_outputs.shape[1])
    group_sizes_2d = group_sizes.reshape(1, num_local_experts)
    residuals = (
        casted_sorted_x.get_tensor(usage=TensorUsage.LHS_TRANS).checkpoint(fc1_quantizer_set.x),
        casted_wi.get_tensor(usage=TensorUsage.RHS_TRANS).checkpoint(fc1_quantizer_set.kernel),
        gate_proj_out,
        up_proj_out,
        casted_intermediate.get_tensor(usage=TensorUsage.LHS_TRANS).checkpoint(fc2_quantizer_set.x),
        casted_wo.get_tensor(usage=TensorUsage.RHS_TRANS).checkpoint(fc2_quantizer_set.kernel),
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

    # wo bwd
    casted_d_eo = tex.grouped_quantize(
        d_eo_2d,
        fc2_quantizer_set.dgrad,
        group_sizes,
        flatten_axis=-1,
    )
    _casted_d_eo_lhs = casted_d_eo.get_tensor(usage=TensorUsage.LHS)
    _casted_d_eo_rhs = casted_d_eo.get_tensor(usage=TensorUsage.RHS)
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
    d_wo_bias = tex.grouped_dbias(d_eo_2d, group_sizes) if has_bias else None

    act_fn = _convert_to_activation_function(activation_type)
    if apply_topk_weights_early:
        # intermediate' = intermediate * w.
        # Masking is not required as:
        # 1. Padding between groups is zero padded due to NCCL EP.
        # 2. Overallocated padding past all groups is uninitialized, but subsequent GEMMs and EP are all group-size aware and will not read past the final group.
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
    _validate_moe_quantizer_sets(
        quantizer_sets,
        num_token_groups=dp_size * num_experts,
        num_expert_groups=num_experts,
    )

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
                f"recv_capacity_per_rank must be a positive multiple of {_ALIGN_SIZE}, got"
                f" {recv_pr}"
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

    # Quantized grouped tensors store their data, scales, and group metadata
    # as physical buffers rather than in the source tensor's logical shape.
    # A PartitionSpec used as a pytree prefix applies the same ownership to
    # every array leaf of the grouped tensor: dispatched-token buffers belong
    # to the compound batch shard, while expert-weight buffers belong to EP.
    token_buffer_spec = P(batch_pspec_axis)
    token_matrix_spec = P(batch_pspec_axis, None)
    expert_buffer_spec = P(ep_axis)
    residuals_spec = (
        token_buffer_spec,
        expert_buffer_spec,
        token_matrix_spec,
        token_matrix_spec,
        token_buffer_spec,
        expert_buffer_spec,
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
    # output of MLP should be sharded the same way as the activation input
    output = with_sharding_constraint_by_logical_axes(output, input_axes)

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
        recv_topk_weights=recv_topk_weights,
        casted_sorted_x_lhs_trans=casted_sorted_x_lhs_trans,
        casted_wi_rhs_trans=casted_wi_rhs_trans,
        gate_proj_out=gate_proj_out,
        up_proj_out=up_proj_out,
        casted_intermediate_lhs_trans=casted_intermediate_lhs_trans,
        casted_wo_rhs_trans=casted_wo_rhs_trans,
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

    mesh = _get_mesh()
    if mesh is None or mesh.empty:
        raise ValueError("moe(...) requires an active jax.sharding.Mesh.")
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
        w = ctx.recv_topk_weights[..., None].astype(grad_pre_combine.dtype)
        d_expert_outputs = grad_pre_combine * w
        d_recv_w_from_combine = (grad_pre_combine * ctx.expert_outputs).sum(axis=-1)
        d_recv_w_from_combine = d_recv_w_from_combine.astype(ctx.recv_topk_weights.dtype)

    # ---------------- FFN bwd (per-shard via shard_map) ----------------
    kernel_spec = P(ep_axis, None, None)
    bias_spec = P(ep_axis, None)
    token_buffer_spec = P(batch_pspec_axis)
    token_matrix_spec = P(batch_pspec_axis, None)
    expert_buffer_spec = P(ep_axis)
    residuals_specs = (
        token_buffer_spec,
        expert_buffer_spec,
        token_matrix_spec,
        token_matrix_spec,
        token_buffer_spec,
        expert_buffer_spec,
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

    def _ffn_bwd_body(*args):
        grads = _ffn_bwd_per_shard(
            *args,
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
    )(
        *bwd_in_args
    )

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
        Independent FC1 and FC2 quantizer sets describing the global logical
        operation. Token quantizers have ``dp_size * num_experts`` groups and
        kernel quantizers have ``num_experts`` groups; shard-local FFN calls use
        this global descriptor unchanged. Currently only no-op (BF16) and
        stateless grouped MXFP8 quantizers are supported. They are differentiable
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
