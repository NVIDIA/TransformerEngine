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
import warnings
from dataclasses import dataclass, fields, replace
from functools import partial
from typing import Any, Literal, Optional, Tuple, Union

import flax.struct
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import NamedSharding, PartitionSpec as P

from . import cpp_extensions as tex
from .quantize import (
    GroupedScaledTensor1x,
    GroupedQuantizer,
    QuantizerSet,
    ScaledTensorFactory,
    ScalingMode,
    ScaledTensor2x,
    TensorUsage,
    noop_quantizer_set,
    with_sharding_constraint_by_logical_axes,
)
from .quantize.dequantizer import _unswizzle_mxfp8_grouped_scale
from .cpp_extensions.gemm import swizzled_scale
from .flax.module import _convert_to_activation_function
from .router import ScoreFunction, _validate_score_function
from .sharding import MeshResource, _get_mesh, global_mesh_resource, global_shard_guard

__all__ = ["WeightGather", "get_moe_recv_capacity_per_rank", "moe"]


@dataclass(frozen=True)
class WeightGather:
    """Deprecated compatibility policy; use ``quant_before_fsdp_ag`` instead.

    How MoE expert weights are gathered across a sharding mesh axis.

    The quantization recipe determines the wire format for ``quantized``;
    this policy only chooses whether quantization precedes the gather.
    """

    mode: Literal["full_precision", "quantized"] = "full_precision"
    axis: Optional[str] = None

    def __post_init__(self):
        if self.mode not in ("full_precision", "quantized"):
            raise ValueError(f"Unsupported weight gather mode: {self.mode!r}")
        if self.mode == "quantized" and not self.axis:
            raise ValueError("Quantized weight gather requires a mesh axis.")
        if self.mode == "full_precision" and self.axis is not None:
            raise ValueError("A weight gather axis is only used in quantized mode.")

    @classmethod
    def full_precision(cls) -> "WeightGather":
        """Gather weights before quantization (the default behavior)."""
        return cls()

    @classmethod
    def quantized(cls, *, axis: Optional[str] = None) -> "WeightGather":
        """Quantize local shards before gathering data and scales.

        With no explicit axis, use the active ``MeshResource.fsdp_resource``.
        Passing an axis bypasses the global resource lookup entirely.
        """
        if axis is None:
            try:
                axis = global_mesh_resource().fsdp_resource
            except AssertionError as exc:
                raise ValueError(
                    "WeightGather.quantized() requires an active MeshResource "
                    "with fsdp_resource, or an explicit axis."
                ) from exc
            if not axis:
                raise ValueError(
                    "WeightGather.quantized() requires MeshResource.fsdp_resource "
                    "or an explicit axis."
                )
        return cls(mode="quantized", axis=axis)


@dataclass
class _LegacyMoEMeshResource(MeshResource):
    """Preserve arbitrary ordered outer axes accepted by the deprecated API."""

    _legacy_data_parallelism_axes: Tuple[str, ...] = ()


def _moe_mesh_axes(resource: MeshResource):
    """Resolve physical MoE axes, keeping EP innermost in the batch shard."""
    if not isinstance(resource.ep_resource, str) or not resource.ep_resource:
        raise ValueError("TE MoE requires MeshResource.ep_resource to name a physical mesh axis.")
    if isinstance(resource, _LegacyMoEMeshResource):
        outer_axes = resource._legacy_data_parallelism_axes
    else:
        outer_axes = tuple(
            dict.fromkeys(
                axis for axis in (resource.dp_resource, resource.fsdp_resource) if axis is not None
            )
        )
    if any(not isinstance(axis, str) or not axis for axis in outer_axes):
        raise ValueError("TE MoE DP and FSDP resources must name physical mesh axes.")
    if resource.ep_resource in outer_axes or len(set(outer_axes)) != len(outer_axes):
        raise ValueError("TE MoE EP and outer data-parallel axes must be distinct.")
    return resource.ep_resource, outer_axes


def _resolve_moe_mesh_resource(
    mesh_resource=None,
    quant_before_fsdp_ag=False,
    ep_axis=None,
    data_parallelism_axes=None,
    weight_gather=None,
):
    """Resolve the canonical API and adapt deprecated axes/gather arguments."""
    if not isinstance(quant_before_fsdp_ag, bool):
        raise TypeError("quant_before_fsdp_ag must be a bool.")
    if mesh_resource is not None and not isinstance(mesh_resource, MeshResource):
        raise TypeError("mesh_resource must be a MeshResource or None.")
    explicit_resource = mesh_resource is not None
    if mesh_resource is None:
        try:
            mesh_resource = global_mesh_resource()
        except AssertionError:
            mesh_resource = None

    legacy = ep_axis is not None or data_parallelism_axes is not None or weight_gather is not None
    if legacy:
        warnings.warn(
            "ep_axis, data_parallelism_axes, and weight_gather are deprecated for TE MoE; "
            "pass mesh_resource=MeshResource(...) and quant_before_fsdp_ag instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        if weight_gather is not None and not isinstance(weight_gather, WeightGather):
            raise TypeError("weight_gather must be a WeightGather policy.")
        if weight_gather is not None:
            if quant_before_fsdp_ag and weight_gather.mode != "quantized":
                raise ValueError("quant_before_fsdp_ag conflicts with weight_gather.")
            quant_before_fsdp_ag = weight_gather.mode == "quantized"
        if explicit_resource:
            if ep_axis is not None and ep_axis != mesh_resource.ep_resource:
                raise ValueError("ep_axis conflicts with mesh_resource.ep_resource.")
            if (
                data_parallelism_axes is not None
                and tuple(data_parallelism_axes) != _moe_mesh_axes(mesh_resource)[1]
            ):
                raise ValueError("data_parallelism_axes conflicts with mesh_resource.")
            if (
                weight_gather is not None
                and weight_gather.axis is not None
                and weight_gather.axis != mesh_resource.fsdp_resource
            ):
                raise ValueError("weight_gather axis conflicts with mesh_resource.fsdp_resource.")
        elif ep_axis is not None or data_parallelism_axes is not None:
            # The old functional API defaulted to no outer axes even in a global context.
            axes = tuple(data_parallelism_axes or ())
            fsdp_axis = (
                weight_gather.axis
                if weight_gather is not None and weight_gather.axis is not None
                else getattr(mesh_resource, "fsdp_resource", None)
            )
            if (
                weight_gather is not None
                and weight_gather.axis is not None
                and weight_gather.axis not in axes
            ):
                raise ValueError(
                    "Quantized weight all-gather requires its FSDP axis among the outer batch axes."
                )
            if fsdp_axis not in axes:
                fsdp_axis = axes[-1] if axes else None
            dp_axes = tuple(axis for axis in axes if axis != fsdp_axis)
            resources = {
                field.name: getattr(mesh_resource, field.name, None)
                for field in fields(MeshResource)
            }
            resources.update(
                ep_resource=(
                    ep_axis if ep_axis is not None else getattr(mesh_resource, "ep_resource", None)
                ),
                dp_resource=dp_axes[0] if dp_axes else None,
                fsdp_resource=fsdp_axis,
            )
            mesh_resource = _LegacyMoEMeshResource(
                **resources,
                _legacy_data_parallelism_axes=axes,
            )
        elif weight_gather.axis is not None and mesh_resource is not None:
            mesh_resource = replace(mesh_resource, fsdp_resource=weight_gather.axis)

    if mesh_resource is None:
        raise ValueError(
            "TE MoE requires mesh_resource=MeshResource(...) or an active global_shard_guard"
            " context."
        )
    # Snapshot the mutable resource so the custom VJP retains its forward-time axes.
    mesh_resource = replace(mesh_resource)
    _moe_mesh_axes(mesh_resource)
    if quant_before_fsdp_ag and not mesh_resource.fsdp_resource:
        raise ValueError("quant_before_fsdp_ag=True requires MeshResource.fsdp_resource.")
    return mesh_resource, quant_before_fsdp_ag


# Per-expert dispatch-slot alignment fed to ``tex.ep_prepare`` as
# ``dispatch_output_per_expert_alignment``. NCCL EP HT requires the
# per-expert recv block to be at least 128-token aligned, and all current
# TE grouped-GEMM recipes (bf16/fp16/fp8/mxfp8) are satisfied by the
# same 128-token tile, so a single constant covers every supported path.
_ALIGN_SIZE = 128
_CUDNN_JAX_ALIGN_SIZE = 256
_CUDNN_JAX_ENV = "NVTE_JAX_TEMP_FLAG_FOR_ABHINAV_CUDNN_GROUPED_GEMM_FUSION"


def _use_cudnn_cutedsl_fusion_from_env() -> bool:
    value = os.getenv(_CUDNN_JAX_ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{_CUDNN_JAX_ENV} must be '0' or '1', got {value!r}")
    return value == "1"


def _select_cudnn_jax_fusion(rejection_reasons: list[str]) -> str | bool:
    """Select Rubin GLU, generic Blackwell+ SwiGLU, or unfused TE, in order."""
    from transformer_engine_jax import get_device_compute_capability

    reasons = list(rejection_reasons)
    path = False
    try:
        capability = get_device_compute_capability(0)
    except RuntimeError as exc:
        reasons.append(f"could not query GPU compute capability: {exc}")
    else:
        if not reasons:
            for candidate, supported in (
                ("rubin", capability == 107),
                ("blackwell", capability >= 100),
            ):
                if not supported:
                    requirement = "SM107" if candidate == "rubin" else "SM100+"
                    reasons.append(
                        f"{candidate} fused kernel requires {requirement}, got SM{capability}"
                    )
                    continue
                available, error = tex.grouped_gemm_swiglu_dependencies_available(
                    rubin=candidate == "rubin"
                )
                if available:
                    path = candidate
                    break
                reasons.append(f"{candidate} fused API is incompatible: {error}")
    if reasons:
        destination = "generic Blackwell+ fused kernel" if path else "unfused TE grouped-GEMM path"
        warnings.warn(
            f"{_CUDNN_JAX_ENV}=1: falling back to the {destination}: "
            + "; ".join(reasons)
            + ". Install cuDNN Frontend with compatible JAX APIs (TE's signatures match "
            "cuDNN Frontend 1.31.0) and CuTeDSL JAX support; use supported GPU hardware.",
            UserWarning,
            stacklevel=2,
        )
    return path

def _cudnn_jax_fusion_rejection_reasons(
    x,
    wi,
    wi_0_bias,
    wi_1_bias,
    quantizer_sets,
    *,
    num_experts,
    activation_type,
    ep_axis,
) -> list[str]:
    """Return reasons this call cannot use cuDNN's grouped SwiGLU JAX API."""
    errors = []
    if str(activation_type).lower() != "silu":
        errors.append("requires activation_type='silu'")
    if wi_0_bias is not None or wi_1_bias is not None:
        errors.append("does not support FC1 gate/up bias")
    hidden = x.shape[-1]
    standard_layout = wi.ndim == 3 and wi.shape[-2] == hidden and wi.shape[-1] % 64 == 0
    native_layout = wi.ndim == 3 and wi.shape[-1] == hidden and wi.shape[-2] % 64 == 0
    if not standard_layout and not native_layout:
        errors.append(
            "requires rank-3 wi in standard [E,K,2N] or cuDNN-native [E,2N,K] "
            f"layout with a 64-aligned gated dimension and K={hidden}, got {wi.shape}"
        )
    if x.dtype not in (jnp.bfloat16, jnp.float16):
        errors.append(f"requires BF16 or FP16 activations, got {x.dtype}")

    fc1_quantizer_set, fc2_quantizer_set = quantizer_sets
    required_quantizers = {
        "fc1.x": fc1_quantizer_set.x,
        "fc1.kernel": fc1_quantizer_set.kernel,
        "fc1.dgrad": fc1_quantizer_set.dgrad,
        "fc2.x": fc2_quantizer_set.x,
        "fc2.kernel": fc2_quantizer_set.kernel,
        "fc2.dgrad": fc2_quantizer_set.dgrad,
    }
    for name, quantizer in required_quantizers.items():
        if not isinstance(quantizer, GroupedQuantizer):
            errors.append(f"requires a grouped MXFP8 quantizer for {name}")
        elif quantizer.scaling_mode != ScalingMode.MXFP8_1D_SCALING:
            errors.append(f"requires MXFP8_1D_SCALING for {name}")
        elif not quantizer.q_layout.is_rowwise_colwise:
            errors.append(f"requires rowwise+colwise quantization for {name}")

    if all(isinstance(q, GroupedQuantizer) for q in required_quantizers.values()):
        if fc1_quantizer_set.x.q_dtype != fc1_quantizer_set.kernel.q_dtype:
            errors.append("requires identical FC1 activation and weight MXFP8 payload dtypes")
        supported = (jnp.float8_e4m3fn, jnp.float8_e5m2)
        for name, quantizer in required_quantizers.items():
            if quantizer.q_dtype not in supported:
                errors.append(f"unsupported MXFP8 payload dtype {quantizer.q_dtype} for {name}")

    mesh = _get_mesh()
    if mesh is not None and not mesh.empty and ep_axis in mesh.shape:
        num_local_experts = num_experts // mesh.shape[ep_axis]
        if num_local_experts > 1024:
            errors.append(f"requires at most 1024 local experts, got {num_local_experts}")

    return errors


def get_moe_recv_capacity_per_rank(
    *,
    num_experts: int,
    num_experts_per_tok: int,
    max_tokens_per_rank: int,
    ep_size: int,
    recv_capacity_factor: Optional[float] = None,
    alignment: Optional[int] = None,
) -> int:
    """Return the aligned receive capacity for one EP rank.

    ``recv_capacity_factor=None`` reserves the dropless worst case. A finite
    factor >= 1 scales the capacity needed by perfectly balanced routing and
    is capped at the worst case. The balanced baseline includes the independent
    per-local-expert alignment required by NCCL EP. When ``alignment`` is not
    supplied, it follows the active MoE implementation: 256 for the cuDNN
    grouped-SwiGLU fusion and 128 for the regular TE grouped GEMM. This keeps
    eager bootstrap callers in sync with the later compiled ``moe()`` call.
    """
    if alignment is None:
        alignment = _CUDNN_JAX_ALIGN_SIZE if _use_cudnn_cutedsl_fusion_from_env() else _ALIGN_SIZE
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
        topk/softmax/sigmoid/sqrtsoftplus kernels are only validated at fp32).

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


def _gather_quantized_weight(tensor, fsdp_axis: str, fsdp_size: int, sharded_axis: int):
    """Gather an MXFP8 grouped weight without gathering its BF16 source.

    Grouped tensor data and scales are flat, with scales independently padded
    and swizzled for each expert. Reassemble each expert in logical order,
    then pad and swizzle its gathered scales for grouped GEMM. Whole-expert
    shards can instead concatenate their already-swizzled scale blocks.
    """
    if isinstance(tensor, ScaledTensor2x):
        return ScaledTensor2x(
            _gather_quantized_weight(tensor.rowwise_tensor, fsdp_axis, fsdp_size, sharded_axis),
            _gather_quantized_weight(tensor.colwise_tensor, fsdp_axis, fsdp_size, sharded_axis),
        )
    if not isinstance(tensor, GroupedScaledTensor1x):
        raise TypeError("Quantized FSDP gather requires grouped MXFP8 weight tensors.")

    local_shape = tensor.original_shape
    num_experts = local_shape[0]
    # The T layout swaps the two matrix dimensions within each expert.
    data_axis = (
        3 - sharded_axis if tensor.data_layout == "T" and sharded_axis != 0 else sharded_axis
    )
    global_shape = list(local_shape)
    global_shape[data_axis] *= fsdp_size
    global_shape = tuple(global_shape)
    data = jax.lax.all_gather(
        tensor.data.reshape(local_shape), fsdp_axis, axis=data_axis, tiled=True
    ).reshape(-1)

    local_matrix = local_shape[1:]
    global_matrix = global_shape[1:]
    # Scales use a block-wise 2D view. The local shape can include padding
    # that differs from the padding required after the gather.
    local_scale_shape = tensor.scaling_mode.get_scale_shape(
        local_matrix,
        data_layout=tensor.data_layout,
        is_colwise=tensor.is_colwise,
        is_padded=True,
        flatten_axis=tensor.flatten_axis - 1,
    )
    if sharded_axis == 0:
        # The grouped allocation includes a worst-case padding tail. Gather
        # only actual expert scale blocks, then allocate the gathered tail.
        local_scale_size = math.prod(local_scale_shape)
        scale_inv = jax.lax.all_gather(
            tensor.scale_inv[: num_experts * local_scale_size],
            fsdp_axis,
            axis=0,
            tiled=True,
        )
    else:
        scale_inv = _gather_quantized_matrix_scales(
            tensor, fsdp_axis, data_axis, local_scale_shape, global_matrix
        )
    expected_scale_size = tensor.scaling_mode.get_grouped_scale_shape(
        global_shape,
        global_shape[0],
        tensor.is_colwise,
        is_padded=True,
        flatten_axis=tensor.flatten_axis,
    )[0]
    scale_inv = jnp.pad(scale_inv, (0, expected_scale_size - scale_inv.size))
    return GroupedScaledTensor1x(
        data=data,
        scale_inv=scale_inv,
        amax=tensor.amax,
        first_dims=None,
        last_dims=None,
        scaling_mode=tensor.scaling_mode,
        dq_dtype=tensor.dq_dtype,
        _dq_func=tensor._dq_func,
        is_colwise=tensor.is_colwise,
        data_layout=tensor.data_layout,
        flatten_axis=tensor.flatten_axis,
        original_shape=global_shape,
        pre_swizzled=True,
    )


def _gather_quantized_matrix_scales(tensor, fsdp_axis, data_axis, local_scale_shape, global_matrix):
    """Reassemble scales when FSDP splits each expert's matrix dimension."""
    local_matrix = tensor.original_shape[1:]
    local_unpadded_shape = tensor.scaling_mode.get_scale_shape(
        local_matrix,
        data_layout=tensor.data_layout,
        is_colwise=tensor.is_colwise,
        is_padded=False,
        flatten_axis=tensor.flatten_axis - 1,
    )
    global_unpadded_shape = tensor.scaling_mode.get_scale_shape(
        global_matrix,
        data_layout=tensor.data_layout,
        is_colwise=tensor.is_colwise,
        is_padded=False,
        flatten_axis=tensor.flatten_axis - 1,
    )
    global_padded_shape = tensor.scaling_mode.get_scale_shape(
        global_matrix,
        data_layout=tensor.data_layout,
        is_colwise=tensor.is_colwise,
        is_padded=True,
        flatten_axis=tensor.flatten_axis - 1,
    )
    scale_axis = data_axis - 1
    local_scale_size = math.prod(local_scale_shape)
    gathered_scales = []
    for expert in range(tensor.original_shape[0]):
        local_swizzled = jax.lax.dynamic_slice_in_dim(
            tensor.scale_inv, expert * local_scale_size, local_scale_size
        )
        local_plain = _unswizzle_mxfp8_grouped_scale(
            local_swizzled, local_scale_shape, tensor.is_colwise
        )
        local_plain = local_plain[: local_unpadded_shape[0], : local_unpadded_shape[1]]
        full_plain = jax.lax.all_gather(local_plain, fsdp_axis, axis=scale_axis, tiled=True)
        assert full_plain.shape == global_unpadded_shape
        full_padded = jnp.pad(
            full_plain,
            (
                (0, global_padded_shape[0] - global_unpadded_shape[0]),
                (0, global_padded_shape[1] - global_unpadded_shape[1]),
            ),
        )
        gathered_scales.append(swizzled_scale(full_padded, 1, tensor.is_colwise).reshape(-1))
    return jnp.concatenate(gathered_scales)


def _weight_fsdp_axis(spec, fsdp_axis):
    """Find the tensor dimension partitioned by the physical FSDP resource."""
    return next(
        (
            i
            for i, axes in enumerate(spec)
            if fsdp_axis in (axes if isinstance(axes, tuple) else (axes,))
        ),
        None,
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
    use_cudnn_jax_fusion: str | bool,
    wi_0_checkpoint_name: Optional[str],
    wi_1_checkpoint_name: Optional[str],
    wo_checkpoint_name: Optional[str],
    cudnn_native_weight_layout: bool,
    quant_before_fsdp_ag: bool,
    fsdp_axis: Optional[str],
    fsdp_size: int,
    wi_fsdp_axis: Optional[int],
    wo_fsdp_axis: Optional[int],
):
    """Run the grouped FFN on one shard's EP receive buffer."""
    hidden = recv_tokens_local.shape[-1]
    sorted_x = recv_tokens_local.reshape(-1, hidden)
    recv_w_flat = recv_topk_weights_local.reshape(-1)
    group_sizes = token_counts_local.reshape(-1).astype(jnp.int32)

    wi = wi.astype(sorted_x.dtype)
    wo = wo.astype(sorted_x.dtype)

    # The cuDNN-native parameter is persistent [E,2N,K] storage with alternating
    # 32-column gate/up blocks. Standard TE storage is [E,K,2N] with contiguous
    # gate/up halves. Keep conversion only as a compatibility fallback.
    if use_cudnn_jax_fusion:
        if cudnn_native_weight_layout:
            wi_for_gemm = wi
        else:
            wi_gate, wi_up = jnp.split(wi, 2, axis=-1)
            wi_for_gemm = tex.pack_swiglu_pair(wi_gate, wi_up)
    else:
        if cudnn_native_weight_layout:
            wi_interleaved = wi.transpose(0, 2, 1)
            wi_gate, wi_up = tex.unpack_swiglu_pair(wi_interleaved)
            wi_for_gemm = jnp.concatenate((wi_gate, wi_up), axis=-1)
        else:
            wi_for_gemm = wi
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
    casted_wi = tex.grouped_quantize(wi_for_gemm, fc1_quantizer_set.kernel, flatten_axis=-1)
    if quant_before_fsdp_ag and wi_fsdp_axis is not None:
        casted_wi = _gather_quantized_weight(
            casted_wi,
            fsdp_axis,
            fsdp_size,
            wi_fsdp_axis,
        )
    casted_intermediate = None
    if use_cudnn_jax_fusion:
        casted_sorted_x_lhs = casted_sorted_x.get_tensor(usage=TensorUsage.LHS)
        casted_wi_rhs = casted_wi.get_tensor(
            usage=TensorUsage.LHS if cudnn_native_weight_layout else TensorUsage.RHS
        )
        combined = wi_for_gemm.shape[-2] if cudnn_native_weight_layout else wi_for_gemm.shape[-1]
        padded_offsets = jnp.cumsum(group_sizes, dtype=jnp.int32)
        prob = (
            recv_w_flat[:, None, None]
            if apply_topk_weights_early
            else jnp.ones((sorted_x.shape[0], 1, 1), dtype=jnp.float32)
        )
        (
            combined_out_3d,
            intermediate_row,
            intermediate_col,
            intermediate_scale_row,
            intermediate_scale_col,
        ) = (tex.grouped_gemm_glu if use_cudnn_jax_fusion == "rubin" else tex.grouped_gemm_swiglu)(
            casted_sorted_x_lhs.data.reshape(sorted_x.shape[0], hidden, 1),
            (
                casted_wi_rhs.data.reshape(num_local_experts, combined, hidden)
                if cudnn_native_weight_layout
                else casted_wi_rhs.data.reshape(num_local_experts, hidden, combined).transpose(
                    0, 2, 1
                )
            ),
            casted_sorted_x_lhs.scale_inv,
            casted_wi_rhs.scale_inv,
            padded_offsets,
            prob,
            compute_dtype=sorted_x.dtype,
            output_dtype=fc2_quantizer_set.x.q_dtype,
        )
        combined_out = combined_out_3d.reshape(sorted_x.shape[0], combined)
        if wi_0_checkpoint_name is not None:
            combined_out = checkpoint_name(combined_out, wi_0_checkpoint_name)
        if wi_1_checkpoint_name is not None:
            combined_out = checkpoint_name(combined_out, wi_1_checkpoint_name)
        # The fused backward consumes interleaved C directly. Keep both residual
        # slots as aliases instead of materializing an otherwise-unused unpack.
        gate_proj_out = up_proj_out = combined_out

        intermediate_shape = (sorted_x.shape[0], combined // 2)
        scaling_mode = fc2_quantizer_set.x.scaling_mode
        row_scale_size = scaling_mode.get_grouped_scale_shape(
            intermediate_shape,
            num_local_experts,
            False,
            is_padded=True,
            flatten_axis=1,
        )[0]
        col_scale_size = scaling_mode.get_grouped_scale_shape(
            intermediate_shape,
            num_local_experts,
            True,
            is_padded=True,
            flatten_axis=1,
        )[0]
        intermediate_scale_row = jnp.pad(
            intermediate_scale_row,
            (0, row_scale_size - intermediate_scale_row.size),
        )
        intermediate_scale_col = jnp.pad(
            intermediate_scale_col,
            (0, col_scale_size - intermediate_scale_col.size),
        )
        for checkpoint_label in (wi_0_checkpoint_name, wi_1_checkpoint_name):
            if checkpoint_label is not None:
                intermediate_row = checkpoint_name(intermediate_row, checkpoint_label)
                intermediate_col = checkpoint_name(intermediate_col, checkpoint_label)
                intermediate_scale_row = checkpoint_name(intermediate_scale_row, checkpoint_label)
                intermediate_scale_col = checkpoint_name(intermediate_scale_col, checkpoint_label)
        casted_intermediate = ScaledTensorFactory.create(
            data=intermediate_row.reshape(-1),
            scale_inv=intermediate_scale_row,
            colwise_data=intermediate_col.reshape(-1),
            colwise_scale_inv=intermediate_scale_col,
            scaling_mode=scaling_mode,
            dq_dtype=sorted_x.dtype,
            data_layout=fc2_quantizer_set.x.data_layout,
            q_layout=fc2_quantizer_set.x.q_layout,
            flatten_axis=1,
            first_dims=group_sizes,
            original_shape=intermediate_shape,
            pre_swizzled=True,
        )
    else:
        combined_out = tex.grouped_gemm(
            casted_sorted_x.get_tensor(usage=TensorUsage.LHS),
            casted_wi.get_tensor(usage=TensorUsage.RHS),
            contracting_dims=((1,), (1,)),
            bias=wi_combined_bias,
        )
        gate_proj_out, up_proj_out = jnp.split(combined_out, 2, axis=-1)
    if not use_cudnn_jax_fusion:
        if wi_0_checkpoint_name is not None:
            gate_proj_out = checkpoint_name(gate_proj_out, wi_0_checkpoint_name)
        if wi_1_checkpoint_name is not None:
            up_proj_out = checkpoint_name(up_proj_out, wi_1_checkpoint_name)

    # Activation inputs (gate_proj_out, up_proj_out) stay in the wi GEMM
    # output dtype; the activation output (`intermediate`) stays in the
    # dtype the wo GEMM / wo's quantized input consumes. For bf16 compute
    # that's all bf16; for FP8/FP4 the downstream grouped_quantize is what
    # transitions to the target precision.
    act_fn = _convert_to_activation_function(activation_type)
    if not use_cudnn_jax_fusion:
        intermediate = act_fn(gate_proj_out) * up_proj_out
    if apply_topk_weights_early and not use_cudnn_jax_fusion:
        # Fold the per-token combine weights into the FFN intermediate;
        # the downstream wo GEMM is linear so this is equivalent to the
        # late-weighting path. Grouped GEMM skips overallocation tail padding automatically.
        # Padding between groups is padded with zeros by NCCL EP.
        intermediate = intermediate * recv_w_flat[:, None].astype(intermediate.dtype)

    if not use_cudnn_jax_fusion:
        casted_intermediate = tex.grouped_quantize(
            intermediate,
            fc2_quantizer_set.x,
            group_sizes,
            flatten_axis=-1,
        )
    casted_wo = tex.grouped_quantize(wo, fc2_quantizer_set.kernel, flatten_axis=-1)
    if quant_before_fsdp_ag and wo_fsdp_axis is not None:
        casted_wo = _gather_quantized_weight(casted_wo, fsdp_axis, fsdp_size, wo_fsdp_axis)
    expert_outputs = tex.grouped_gemm(
        casted_intermediate.get_tensor(usage=TensorUsage.LHS),
        casted_wo.get_tensor(usage=TensorUsage.RHS),
        contracting_dims=((1,), (1,)),
        bias=wo_bias,
    )
    if wo_checkpoint_name is not None:
        expert_outputs = checkpoint_name(expert_outputs, wo_checkpoint_name)
    expert_outputs_3d = expert_outputs.reshape(1, expert_outputs.shape[0], expert_outputs.shape[1])
    group_sizes_2d = group_sizes.reshape(1, num_local_experts)
    ffn_activation_residual = combined_out if use_cudnn_jax_fusion else gate_proj_out
    residuals = (
        casted_sorted_x.get_tensor(usage=TensorUsage.LHS_TRANS).checkpoint(fc1_quantizer_set.x),
        casted_wi.get_tensor(
            usage=(
                TensorUsage.LHS
                if use_cudnn_jax_fusion and cudnn_native_weight_layout
                else TensorUsage.RHS_TRANS
            )
        ).checkpoint(fc1_quantizer_set.kernel),
        ffn_activation_residual,
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
    use_cudnn_jax_fusion: str | bool,
    cudnn_native_weight_layout: bool,
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
    d_wo = tex.grouped_gemm(
        casted_intermediate_lhs_trans,
        _casted_d_eo_rhs,
        contracting_dims=((0,), (0,)),
    )
    d_wo_bias = tex.grouped_dbias(d_eo_2d, group_sizes) if has_bias else None

    if use_cudnn_jax_fusion:
        # The forward residual uses this slot for cuDNN's interleaved pre-activation C.
        combined_out = gate_proj_out
        rows, combined = combined_out.shape
        intermediate = combined // 2
        hidden = d_eo_2d.shape[-1]
        num_local_experts = group_sizes.size
        padded_offsets = jnp.cumsum(group_sizes, dtype=jnp.int32)
        prob = recv_w_flat if apply_topk_weights_early else jnp.ones((rows,), dtype=jnp.float32)
        (
            d_combined_row,
            d_combined_col,
            dprob,
            d_combined_scale_row,
            d_combined_scale_col,
        ) = tex.grouped_gemm_dswiglu(
            _casted_d_eo_lhs.data.reshape(rows, hidden),
            casted_wo_rhs_trans.data.reshape(num_local_experts, intermediate, hidden),
            combined_out,
            _casted_d_eo_lhs.scale_inv,
            casted_wo_rhs_trans.scale_inv,
            padded_offsets,
            prob,
            output_dtype=fc1_quantizer_set.dgrad.q_dtype,
        )
        d_recv_w_from_intermediate = (
            dprob.astype(recv_w_flat.dtype)
            if apply_topk_weights_early
            else jnp.zeros_like(recv_w_flat)
        )

        combined_shape = (rows, combined)
        scaling_mode = fc1_quantizer_set.dgrad.scaling_mode
        row_scale_size = scaling_mode.get_grouped_scale_shape(
            combined_shape,
            num_local_experts,
            False,
            is_padded=True,
            flatten_axis=1,
        )[0]
        col_scale_size = scaling_mode.get_grouped_scale_shape(
            combined_shape,
            num_local_experts,
            True,
            is_padded=True,
            flatten_axis=1,
        )[0]
        d_combined_scale_row = jnp.pad(
            d_combined_scale_row,
            (0, row_scale_size - d_combined_scale_row.size),
        )
        d_combined_scale_col = jnp.pad(
            d_combined_scale_col,
            (0, col_scale_size - d_combined_scale_col.size),
        )
        casted_d_combined = ScaledTensorFactory.create(
            data=d_combined_row.reshape(-1),
            scale_inv=d_combined_scale_row,
            colwise_data=d_combined_col.reshape(-1),
            colwise_scale_inv=d_combined_scale_col,
            scaling_mode=scaling_mode,
            dq_dtype=d_eo_2d.dtype,
            data_layout=fc1_quantizer_set.dgrad.data_layout,
            q_layout=fc1_quantizer_set.dgrad.q_layout,
            flatten_axis=1,
            first_dims=group_sizes,
            original_shape=combined_shape,
            pre_swizzled=True,
        )
        d_combined_for_bias = None
    else:
        d_intermediate = tex.grouped_gemm(
            _casted_d_eo_lhs,
            casted_wo_rhs_trans,
            contracting_dims=((1,), (2,)),
        )
        act_fn = _convert_to_activation_function(activation_type)
        if apply_topk_weights_early:
            # intermediate' = intermediate * w.
            # Masking is not required as grouped GEMMs consume only group-size rows.
            w_b = recv_w_flat[:, None].astype(d_intermediate.dtype)
            intermediate_unweighted = act_fn(gate_proj_out) * up_proj_out
            d_recv_w_from_intermediate = jnp.sum(
                d_intermediate * intermediate_unweighted,
                axis=-1,
            ).astype(recv_w_flat.dtype)
            d_intermediate = d_intermediate * w_b
        else:
            d_recv_w_from_intermediate = jnp.zeros_like(recv_w_flat)

        # Activation bwd stays in the GEMM dtype, matching the forward path.
        act_gp, dact_pullback = jax.vjp(act_fn, gate_proj_out)
        d_up_proj_out = d_intermediate * act_gp
        (d_gate_proj_out,) = dact_pullback(d_intermediate * up_proj_out)
        d_combined_for_bias = jnp.concatenate([d_gate_proj_out, d_up_proj_out], axis=-1)
        casted_d_combined = tex.grouped_quantize(
            d_combined_for_bias,
            fc1_quantizer_set.dgrad,
            group_sizes,
            flatten_axis=-1,
        )
    d_sorted_x = tex.grouped_gemm(
        casted_d_combined.get_tensor(usage=TensorUsage.LHS),
        casted_wi_rhs_trans,
        contracting_dims=((1,), (1 if use_cudnn_jax_fusion and cudnn_native_weight_layout else 2,)),
    )
    if use_cudnn_jax_fusion and cudnn_native_weight_layout:
        # dY^T @ X directly produces [E,2N,K], matching the persistent native
        # parameter, without a post-GEMM transpose or de-interleave/repack.
        d_wi_combined = tex.grouped_gemm(
            casted_d_combined.get_tensor(usage=TensorUsage.LHS_TRANS),
            casted_sorted_x_lhs_trans,
            contracting_dims=((0,), (0,)),
        )
    else:
        d_wi_combined = tex.grouped_gemm(
            casted_sorted_x_lhs_trans,
            casted_d_combined.get_tensor(usage=TensorUsage.RHS),
            contracting_dims=((0,), (0,)),
        )
    if use_cudnn_jax_fusion and not cudnn_native_weight_layout:
        d_wi_gate, d_wi_up = tex.unpack_swiglu_pair(d_wi_combined)
        d_wi_combined = jnp.concatenate([d_wi_gate, d_wi_up], axis=-1)
    elif not use_cudnn_jax_fusion and cudnn_native_weight_layout:
        d_wi_gate, d_wi_up = jnp.split(d_wi_combined, 2, axis=-1)
        d_wi_combined = tex.pack_swiglu_pair(d_wi_gate, d_wi_up).transpose(0, 2, 1)
    if has_bias:
        d_wi_combined_bias = tex.grouped_dbias(d_combined_for_bias, group_sizes)
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
    mesh_resource,
    input_axes,
    gate_kernel_axes,
    wi_kernel_axes,
    wo_kernel_axes,
    dtype,
    apply_topk_weights_early,
    recv_capacity_per_rank,
    use_cudnn_jax_fusion,
    wi_0_checkpoint_name,
    wi_1_checkpoint_name,
    wo_checkpoint_name,
    dispatch_checkpoint_name,
    quant_before_fsdp_ag,
):
    """Forward: gate -> topk -> ep_dispatch -> FFN -> ep_combine.

    Returns ``(output, aux_loss)``. ``aux_loss`` is a zero scalar when
    ``aux_loss_coeff == 0``.
    """
    with global_shard_guard(mesh_resource):
        ep_axis, data_parallelism_axes = _moe_mesh_axes(mesh_resource)
        del gate_kernel_axes  # used in bwd only
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
        cudnn_native_weight_layout = wi.ndim == 3 and wi.shape[-1] == H
        kernel_spec = P(ep_axis, None, None)
        wi_input_spec = wo_input_spec = kernel_spec
        wi_fsdp_axis = wo_fsdp_axis = None
        if quant_before_fsdp_ag:
            # Logical axes describe parameter storage even under Auto mode,
            # where tracer types do not expose the physical input sharding.
            if nn.get_logical_axis_rules():
                wi_input_spec = nn.logical_to_mesh_axes(wi_kernel_axes)
                wo_input_spec = nn.logical_to_mesh_axes(wo_kernel_axes)
            else:
                # Preserve the existing no-Flax-rules convention.
                wi_input_spec = (
                    P(ep_axis, None, mesh_resource.fsdp_resource)
                    if cudnn_native_weight_layout
                    else P(ep_axis, mesh_resource.fsdp_resource, None)
                )
                wo_input_spec = P(ep_axis, None, mesh_resource.fsdp_resource)
            wi_fsdp_axis = _weight_fsdp_axis(wi_input_spec, mesh_resource.fsdp_resource)
            wo_fsdp_axis = _weight_fsdp_axis(wo_input_spec, mesh_resource.fsdp_resource)
            if mesh_resource.fsdp_resource not in data_parallelism_axes:
                raise ValueError(
                    "Quantized weight all-gather requires its FSDP axis among the outer batch axes."
                )
            if any(quantizer_set.kernel is None for quantizer_set in quantizer_sets):
                raise ValueError("Quantized weight all-gather requires MXFP8 kernel quantizers.")
            if any(
                axis is not None
                and axis != 0
                and weight.shape[axis] % (mesh.shape[mesh_resource.fsdp_resource] * 32)
                for weight, axis in ((wi, wi_fsdp_axis), (wo, wo_fsdp_axis))
            ):
                raise ValueError("FSDP weight shards must be divisible by the MXFP8 block size 32.")

        if B % num_procs != 0:
            raise ValueError(f"batch={B} not divisible by ep*dp={num_procs}")

        # Per-rank send capacity: B/num_procs rows x S tokens per rank.
        max_tokens_per_rank = (B // num_procs) * S
        # Keep capacity and alignment consistent with an EP bootstrap sized
        # for requested fusion, even when API/hardware checks choose unfused.
        dispatch_alignment = (
            _CUDNN_JAX_ALIGN_SIZE
            if use_cudnn_jax_fusion or _use_cudnn_cutedsl_fusion_from_env()
            else _ALIGN_SIZE
        )
        worst_case_recv_pr = get_moe_recv_capacity_per_rank(
            num_experts=num_experts,
            num_experts_per_tok=K,
            max_tokens_per_rank=max_tokens_per_rank,
            ep_size=num_ep,
            alignment=dispatch_alignment,
        )
        if recv_capacity_per_rank is None:
            recv_pr = worst_case_recv_pr
        else:
            recv_pr = int(recv_capacity_per_rank)
            if recv_pr <= 0 or recv_pr % dispatch_alignment != 0:
                raise ValueError(
                    "recv_capacity_per_rank must be a positive multiple of "
                    f"{dispatch_alignment}, got"
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
            dispatch_output_per_expert_alignment=dispatch_alignment,
        )
        token_counts, total_recv_tokens, handle_mem = tex.ep_prepare(cfg, topk_idx_3d)
        token_counts = jax.lax.with_sharding_constraint(token_counts, NamedSharding(mesh, ep2_spec))
        recv_tokens, recv_topk_weights = tex.ep_dispatch_fwd(
            cfg, handle_mem, topk_idx_3d, x, topk_w_3d, recv_pr
        )
        if dispatch_checkpoint_name is not None:
            recv_tokens = checkpoint_name(recv_tokens, dispatch_checkpoint_name)
            recv_topk_weights = checkpoint_name(recv_topk_weights, dispatch_checkpoint_name)
        recv_tokens = jax.lax.with_sharding_constraint(recv_tokens, NamedSharding(mesh, ep3_spec))
        recv_topk_weights = jax.lax.with_sharding_constraint(
            recv_topk_weights, NamedSharding(mesh, ep2_spec)
        )

        # ---------------- FFN (per-shard via shard_map) ----------------
        has_bias = wi_0_bias is not None
        bias_spec = P(ep_axis, None)
        ffn_in_specs = (ep3_spec, ep2_spec, ep2_spec, wi_input_spec, wo_input_spec)
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
                use_cudnn_jax_fusion=use_cudnn_jax_fusion,
                wi_0_checkpoint_name=wi_0_checkpoint_name,
                wi_1_checkpoint_name=wi_1_checkpoint_name,
                wo_checkpoint_name=wo_checkpoint_name,
                cudnn_native_weight_layout=cudnn_native_weight_layout,
                quant_before_fsdp_ag=quant_before_fsdp_ag,
                fsdp_axis=mesh_resource.fsdp_resource,
                fsdp_size=(
                    mesh.shape[mesh_resource.fsdp_resource]
                    if mesh_resource.fsdp_resource is not None
                    else 1
                ),
                wi_fsdp_axis=wi_fsdp_axis,
                wo_fsdp_axis=wo_fsdp_axis,
            )

        expert_outputs, ffn_residuals = shard_map(
            _ffn_fwd_body,
            mesh=mesh,
            in_specs=ffn_in_specs,
            out_specs=(ep3_spec, residuals_spec),
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
            "cudnn_native_weight_layout": cudnn_native_weight_layout,
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
    mesh_resource,
    input_axes,
    gate_kernel_axes,
    wi_kernel_axes,
    wo_kernel_axes,
    dtype,
    apply_topk_weights_early,
    recv_capacity_per_rank,
    use_cudnn_jax_fusion,
    wi_0_checkpoint_name,
    wi_1_checkpoint_name,
    wo_checkpoint_name,
    dispatch_checkpoint_name,
    quant_before_fsdp_ag,
    residuals,
    cotangents,
):
    """Backward mirror of :func:`_moe_fwd_rule`."""
    with global_shard_guard(mesh_resource):
        ep_axis, data_parallelism_axes = _moe_mesh_axes(mesh_resource)
        del (
            num_groups,
            group_topk,
            dtype,
            recv_capacity_per_rank,
            wi_0_checkpoint_name,
            wi_1_checkpoint_name,
            wo_checkpoint_name,
            dispatch_checkpoint_name,
            quant_before_fsdp_ag,
        )  # captured / unused in bwd
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
                use_cudnn_jax_fusion=use_cudnn_jax_fusion,
                cudnn_native_weight_layout=static["cudnn_native_weight_layout"],
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
        d_recv_w_total = jax.lax.with_sharding_constraint(
            d_recv_w_total, NamedSharding(mesh, ep2_spec)
        )
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
        d_gate_kernel = jnp.einsum("bsh,bse->he", ctx.x, d_gate_logits).astype(
            ctx.gate_kernel.dtype
        )
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


@partial(jax.custom_vjp, nondiff_argnums=tuple(range(9, 32)))
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
    mesh_resource,
    input_axes,
    gate_kernel_axes,
    wi_kernel_axes,
    wo_kernel_axes,
    dtype,
    apply_topk_weights_early,
    recv_capacity_per_rank,
    use_cudnn_jax_fusion,
    wi_0_checkpoint_name,
    wi_1_checkpoint_name,
    wo_checkpoint_name,
    dispatch_checkpoint_name,
    quant_before_fsdp_ag,
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
        mesh_resource,
        input_axes,
        gate_kernel_axes,
        wi_kernel_axes,
        wo_kernel_axes,
        dtype,
        apply_topk_weights_early,
        recv_capacity_per_rank,
        use_cudnn_jax_fusion,
        wi_0_checkpoint_name,
        wi_1_checkpoint_name,
        wo_checkpoint_name,
        dispatch_checkpoint_name,
        quant_before_fsdp_ag,
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
    mesh_resource: Optional[MeshResource] = None,
    quant_before_fsdp_ag: bool = False,
    ep_axis: Optional[str] = None,
    data_parallelism_axes: Optional[Tuple[str, ...]] = None,
    input_axes: Tuple[Optional[str], ...] = (),
    gate_kernel_axes: Tuple[Optional[str], ...] = (),
    wi_kernel_axes: Tuple[Optional[str], ...] = ("exp", "embed", "mlp"),
    wo_kernel_axes: Tuple[Optional[str], ...] = ("exp", "mlp", "embed"),
    dtype: jnp.dtype = jnp.float32,
    recv_capacity_per_rank: Optional[int] = None,
    wi_0_checkpoint_name: Optional[str] = None,
    wi_1_checkpoint_name: Optional[str] = None,
    wo_checkpoint_name: Optional[str] = None,
    dispatch_checkpoint_name: Optional[str] = None,
    weight_gather: Optional[WeightGather] = None,
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
        when ``score_function='sigmoid'`` or ``'sqrtsoftplus'``. Pass ``None`` to disable.
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
    wi_0_checkpoint_name : Optional[str]
        JAX rematerialization checkpoint name for the gate projection output.
        With cuDNN fusion, also names the shared fused outputs, including its quantized output.
        ``None`` leaves the value unnamed.
    wi_1_checkpoint_name : Optional[str]
        JAX rematerialization checkpoint name for the up projection output.
        With cuDNN fusion, also names the same fused outputs, including its quantized output.
        ``None`` leaves the value unnamed.
    wo_checkpoint_name : Optional[str]
        JAX rematerialization checkpoint name for the per-expert down projection output.
        ``None`` leaves the value unnamed.
    dispatch_checkpoint_name : Optional[str]
        JAX rematerialization checkpoint name for the EP dispatch outputs.
        ``None`` leaves these values unnamed; the opaque prepare handle is never named.
    mesh_resource : Optional[MeshResource]
        Physical parallelism resources. ``None`` uses the active
        ``global_shard_guard`` context; an explicit resource takes precedence.
        One of these is required. Batch sharding combines ``dp_resource`` then
        ``fsdp_resource`` (omitting unset/duplicate axes), with ``ep_resource``
        innermost. EP must be set and distinct from the outer axes.
    quant_before_fsdp_ag : bool
        Quantize expert-weight shards before gathering their MXFP8 data and
        scales on ``mesh_resource.fsdp_resource``. Default ``False`` gathers
        full-precision weights first. ``True`` requires an FSDP resource and
        MXFP8 kernel quantizers. With active Flax logical-axis rules, the weight
        input specs follow ``wi_kernel_axes`` and ``wo_kernel_axes``; FSDP may
        shard whole experts or a matrix dimension within each expert.
    ep_axis, data_parallelism_axes, weight_gather : deprecated
        Compatibility arguments converted into a MeshResource and boolean,
        with a DeprecationWarning. Conflicting old and new arguments raise.
    Per-expert dispatch-slot alignment defaults to 128 tokens (``_ALIGN_SIZE``).
    Requesting cuDNN fusion reserves 256 tokens, also when falling back, to
    preserve compatibility with EP bootstrap buffer sizing.

    Set ``NVTE_JAX_TEMP_FLAG_FOR_ABHINAV_CUDNN_GROUPED_GEMM_FUSION=1`` to use cuDNN's
    JAX grouped MXFP8 APIs: Rubin GLU first, then generic SM100+ SwiGLU.
    Ineligible calls warn
    and fall back to TE's regular grouped-GEMM implementation. API signatures
    and GPU capability determine support; fallbacks emit an actionable warning.

    MeshResource fields name physical mesh axes, not Flax logical axes.
    ``input_axes``, ``gate_kernel_axes``, ``wi_kernel_axes`` and
    ``wo_kernel_axes`` remain logical-axis tuples resolved through the active
    Flax rules. The selected MeshResource is also used by internal TE sharding
    in both forward and backward, independently of an enclosing global context.

    See module docstring for the rest of the parameter semantics and the
    surrounding design rationale.
    """
    if ep_axis is not None or data_parallelism_axes is not None or weight_gather is not None:
        call_args = locals().copy()
        resource, quantize = _resolve_moe_mesh_resource(
            mesh_resource, quant_before_fsdp_ag, ep_axis, data_parallelism_axes, weight_gather
        )
        for name in ("ep_axis", "data_parallelism_axes", "weight_gather"):
            call_args.pop(name)
        call_args.update(mesh_resource=resource, quant_before_fsdp_ag=quantize)
        return moe(**call_args)

    mesh_resource, quant_before_fsdp_ag = _resolve_moe_mesh_resource(
        mesh_resource, quant_before_fsdp_ag
    )
    ep_axis, data_parallelism_axes = _moe_mesh_axes(mesh_resource)
    score_function = _validate_score_function(score_function)

    with global_shard_guard(mesh_resource):
        # Enforce ((outer_dp..., ep), None, None) on inbound activations. The
        # EP comm groups consecutive global ranks (dp_color = rank // ep_size),
        # so ep MUST be innermost in the partition spec. Soft re-pin: free if
        # upstream already matches, single reshard otherwise.
        mesh = _get_mesh()
        if mesh is None or mesh.empty:
            raise ValueError("moe(...) requires an active jax.sharding.Mesh.")
        for axis in (ep_axis, *data_parallelism_axes):
            if axis not in mesh.shape:
                raise ValueError(f"TE MoE resource axis {axis!r} is not in the active mesh.")
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
        x = _with_sharding_constraint_cast_bwd(x, NamedSharding(mesh, expected_spec))

        # custom_vjp can't trace through None args; lower expert_bias to an
        # empty shape-(0,) tensor that fused_topk_with_score_function treats
        # as "no bias".
        if expert_bias is None:
            expert_bias_arg = jnp.zeros((0,), dtype=jnp.float32)
        else:
            expert_bias_arg = expert_bias.astype(jnp.float32)

        use_cudnn_jax_fusion = False
        cudnn_native_weight_layout = wi.ndim == 3 and wi.shape[-1] == x.shape[-1]
        if _use_cudnn_cutedsl_fusion_from_env():
            rejection_reasons = _cudnn_jax_fusion_rejection_reasons(
                x,
                wi,
                wi_0_bias,
                wi_1_bias,
                quantizer_sets,
                num_experts=num_experts,
                activation_type=activation_type,
                ep_axis=ep_axis,
            )
            use_cudnn_jax_fusion = _select_cudnn_jax_fusion(rejection_reasons)
        elif cudnn_native_weight_layout:
            raise ValueError(
                "cuDNN-native MoE weight layout requires the fused cuDNN grouped-GEMM path; "
                f"set {_CUDNN_JAX_ENV}=1."
            )

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
            mesh_resource,
            input_axes,
            gate_kernel_axes,
            wi_kernel_axes,
            wo_kernel_axes,
            dtype,
            apply_topk_weights_early,
            recv_capacity_per_rank,
            use_cudnn_jax_fusion,
            wi_0_checkpoint_name,
            wi_1_checkpoint_name,
            wo_checkpoint_name,
            dispatch_checkpoint_name,
            quant_before_fsdp_ag,
        )
        if aux_loss_coeff <= 0.0:
            aux_loss = None
        assert (
            output.dtype == x.dtype
        ), f"moe() output dtype {output.dtype} != input dtype {x.dtype}"
        return output, aux_loss, total_recv_tokens
