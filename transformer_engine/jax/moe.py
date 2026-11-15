# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Mixture-of-Experts (MoE) layer for TransformerEngine JAX.

This module exposes :func:`moe`, a fused MoE forward + backward path built on
top of TE's NCCL-backed Expert Parallelism primitives
(``tex.ep_dispatch`` / ``tex.ep_combine``). The ``router_info`` argument
selects how tokens are routed:

* :class:`RouterComputationInfo` -- TE computes the router::

      gate  ->  topk  ->  ep_dispatch  ->  per-expert FFN (grouped GEMMs)
            ->  ep_combine  ->  output

* :class:`RoutingMapInfo` -- the caller supplies expert indices and routing
  weights (e.g. hash routing), and the ``gate -> topk`` stage is skipped::

      external routing  ->  ep_dispatch  ->  per-expert FFN (grouped GEMMs)
                        ->  ep_combine  ->  output

Both paths run under a single ``jax.custom_vjp`` so the routing, dispatch, FFN and
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
from dataclasses import replace
from functools import partial
from typing import Any, Optional, Tuple, Union

import flax.struct
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import NamedSharding, PartitionSpec as P

from . import cpp_extensions as tex
from .cpp_extensions.misc import jax_dtype_to_te_dtype
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

__all__ = [
    "RouterComputationInfo",
    "RoutingMapInfo",
    "get_moe_recv_capacity_per_rank",
    "moe",
]


@flax.struct.dataclass
class RouterComputationInfo:
    """Router parameters and settings for :func:`moe` to compute routing itself.

    Array fields are pytree leaves; the remaining fields are static.

    Parameters
    ----------
    gate_kernel : jnp.ndarray
        ``[hidden, num_experts]`` router projection.
    num_experts_per_tok : int
        Top-k value for routing.
    expert_bias : Optional[jnp.ndarray]
        ``[num_experts]`` learnable router bias added before the top-k
        when ``score_function='sigmoid'`` or ``'sqrtsoftplus'``. ``None``
        disables it. The bias has no gradient through the top-k primitive
        itself (it only steers expert selection); a zero cotangent is
        returned for it.
    score_function : Union[str, ScoreFunction]
        ``"softmax"`` (default), ``"sigmoid"``, or ``"sqrtsoftplus"``.
    use_pre_softmax : bool
        Apply softmax before top-k (vs. after).
    num_groups, group_topk : Optional[int]
        Grouped top-k knobs (DeepSeek-style). ``None`` disables grouping.
    scaling_factor : float
        Multiplier on the routing weights.
    aux_loss_coeff : float
        Per-step expert-load-balance loss coefficient. ``0.0`` (default)
        disables the aux loss entirely. When non-zero, an extra
        all-gather over the routing-side logits is inserted so the
        ``fused_moe_aux_loss`` kernel sees a global ``[T_global, E]``
        view; this lives off the dispatch critical path.
    gate_kernel_axes : Tuple[Optional[str], ...]
        Logical sharding axes of ``gate_kernel``.
    """

    gate_kernel: jnp.ndarray
    num_experts_per_tok: int = flax.struct.field(pytree_node=False)
    expert_bias: Optional[jnp.ndarray] = None
    score_function: Union[str, ScoreFunction] = flax.struct.field(
        pytree_node=False, default="softmax"
    )
    use_pre_softmax: bool = flax.struct.field(pytree_node=False, default=False)
    num_groups: Optional[int] = flax.struct.field(pytree_node=False, default=None)
    group_topk: Optional[int] = flax.struct.field(pytree_node=False, default=None)
    scaling_factor: float = flax.struct.field(pytree_node=False, default=1.0)
    aux_loss_coeff: float = flax.struct.field(pytree_node=False, default=0.0)
    gate_kernel_axes: Tuple[Optional[str], ...] = flax.struct.field(pytree_node=False, default=())


@flax.struct.dataclass
class RoutingMapInfo:
    """Caller-computed routing for :func:`moe`, bypassing TE's router.

    Note: the shape of ``routing_indices`` provides the number of experts per token.

    Parameters
    ----------
    routing_indices : jnp.ndarray
        Integer expert indices with shape ``[batch, sequence, top_k]``.
        Non-differentiable. Each token's ``top_k`` indices must be distinct,
        otherwise the default receive-capacity bound is invalid.
    routing_weights : jnp.ndarray
        Routing weights with the same shape as ``routing_indices``.
        Gradients are returned through the custom VJP, so caller-side
        routers remain fully differentiable.
    """

    routing_indices: jnp.ndarray
    routing_weights: jnp.ndarray


def _num_experts_per_tok(router_info: Union[RouterComputationInfo, RoutingMapInfo]) -> int:
    if isinstance(router_info, RouterComputationInfo):
        return router_info.num_experts_per_tok
    return router_info.routing_indices.shape[-1]


def _moe_mesh_axes(resource: MeshResource):
    """Resolve physical MoE axes, keeping EP innermost in the batch shard."""
    if not isinstance(resource.ep_resource, str) or not resource.ep_resource:
        raise ValueError("TE MoE requires MeshResource.ep_resource to name a physical mesh axis.")
    outer_axes = tuple(
        dict.fromkeys(
            axis for axis in (resource.dp_resource, resource.fsdp_resource) if axis is not None
        )
    )
    if any(not isinstance(axis, str) or not axis for axis in outer_axes):
        raise ValueError("TE MoE DP and FSDP resources must name physical mesh axes.")
    if resource.ep_resource in outer_axes:
        raise ValueError("TE MoE EP and outer data-parallel axes must be distinct.")
    return resource.ep_resource, outer_axes


def _resolve_moe_mesh_resource(
    mesh_resource=None,
    quant_before_fsdp_ag=False,
):
    """Resolve and snapshot the explicit or active mesh resource."""
    if not isinstance(quant_before_fsdp_ag, bool):
        raise TypeError("quant_before_fsdp_ag must be a bool.")
    if mesh_resource is not None and not isinstance(mesh_resource, MeshResource):
        raise TypeError("mesh_resource must be a MeshResource or None.")
    if mesh_resource is None:
        try:
            mesh_resource = global_mesh_resource()
        except AssertionError:
            mesh_resource = None

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


def _select_cudnn_jax_fusion(rejection_reasons: list[str]) -> bool:
    """Select shared cuDNN GLU/dGLU fusion or the unfused TE path."""
    from transformer_engine_jax import get_device_compute_capability

    reasons = list(rejection_reasons)
    try:
        capability = get_device_compute_capability(0)
    except RuntimeError as exc:
        reasons.append(f"could not query GPU compute capability: {exc}")
    else:
        if capability < 100:
            reasons.append(f"fused kernels require SM100+, got SM{capability}")
        if not reasons:
            available, error = tex.grouped_gemm_glu_dependencies_available()
            if available:
                return True
            reasons.append(f"shared fused API is incompatible: {error}")
    warnings.warn(
        "use_cudnn_fusion=True: falling back to the unfused TE grouped-GEMM path: "
        + "; ".join(reasons)
        + ". Install cuDNN Frontend with shared grouped_gemm_glu/grouped_gemm_dglu "
        "JAX APIs for dense MXFP8 and CuTeDSL JAX support; use supported GPU hardware.",
        UserWarning,
        stacklevel=2,
    )
    return False


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
    """Return reasons this call cannot use cuDNN's shared grouped GLU/dGLU JAX APIs."""
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
        if fc1_quantizer_set.dgrad.q_dtype != jnp.float8_e4m3fn:
            errors.append("requires FC1 dgrad MXFP8 payload dtype float8_e4m3fn")
        if fc2_quantizer_set.dgrad.q_dtype != fc2_quantizer_set.kernel.q_dtype:
            errors.append("requires identical FC2 dgrad and weight MXFP8 payload dtypes")
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
    use_cudnn_fusion: bool = True,
) -> int:
    """Return the aligned receive capacity for one EP rank.

    ``recv_capacity_factor=None`` reserves the dropless worst case. A finite
    factor >= 1 scales the capacity needed by perfectly balanced routing and
    is capped at the worst case. The balanced baseline includes the independent
    per-local-expert alignment required by NCCL EP. When ``alignment`` is not
    supplied, ``use_cudnn_fusion=True`` (default) reserves 256-token alignment,
    including fallbacks; ``False`` reserves 128 for regular TE grouped GEMM. This keeps
    eager bootstrap callers in sync with the later compiled ``moe()`` call.
    The bound assumes each token's top-k expert indices are distinct.
    """
    if not isinstance(use_cudnn_fusion, bool):
        raise TypeError("use_cudnn_fusion must be a bool")
    if alignment is None:
        alignment = _CUDNN_JAX_ALIGN_SIZE if use_cudnn_fusion else _ALIGN_SIZE
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


def _prepare_moe_input(
    x: jnp.ndarray,
    ep_axis: str,
    data_parallelism_axes: Tuple[str, ...],
    *,
    caller: str,
) -> jnp.ndarray:
    """Validate and constrain an MoE input to the EP communication layout.

    NCCL EP forms communication groups from consecutive global ranks, with
    ``ep_axis`` as the innermost component of the compound leading mesh axis.
    The expected activation layout is therefore
    ``P((*data_parallelism_axes, ep_axis), None, None)``. If ``x`` has an
    incompatible known sharding, this function warns before inserting the
    reshard. The custom constraint also keeps the backward cotangent in the
    input activation dtype.

    Parameters
    ----------
    x : jnp.ndarray
        Rank-3 MoE input activation.
    ep_axis : str
        Physical mesh axis used for expert parallelism.
    data_parallelism_axes : Tuple[str, ...]
        Physical outer data-parallel mesh axes.
    caller : str
        Public API name used in diagnostics.

    Returns
    -------
    jnp.ndarray
        ``x`` constrained to the EP-compatible input sharding.
    """
    mesh = _get_mesh()
    if mesh is None or mesh.empty:
        raise ValueError(f"{caller}(...) requires an active jax.sharding.Mesh.")
    for axis in (ep_axis, *data_parallelism_axes):
        if axis not in mesh.shape:
            raise ValueError(f"TE MoE resource axis {axis!r} is not in the active mesh.")

    expected_leading: Any = (*data_parallelism_axes, ep_axis) if data_parallelism_axes else ep_axis
    expected_spec = P(expected_leading, None, None)
    actual_spec = getattr(getattr(x, "sharding", None), "spec", None)
    if actual_spec is not None and tuple(actual_spec) != tuple(expected_spec):
        warnings.warn(
            f"{caller}(...): inbound x sharding {actual_spec} does not match expected "
            f"{expected_spec}; inserting a reshard. Apply "
            "jax.lax.with_sharding_constraint upstream to avoid this overhead.",
            UserWarning,
            stacklevel=3,
        )
    return _with_sharding_constraint_cast_bwd(x, NamedSharding(mesh, expected_spec))


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

    Flattened automatically by jax.custom_vjp; ``cfg`` and
    ``routing_weights_dtype`` are static fields (the rest are jnp.ndarray,
    GroupedNoScaleTensor, or None when aux_loss_coeff == 0). ``router`` and the router-side
    residuals (``logits_2d``, ``saved_scores``, ``routing_map``) are None
    when routing is caller-supplied via :class:`RoutingMapInfo`.
    """

    x: jnp.ndarray
    router: Optional[RouterComputationInfo]
    logits_2d: Optional[jnp.ndarray]
    saved_scores: Optional[jnp.ndarray]
    routing_map: Optional[jnp.ndarray]
    selected_experts: jnp.ndarray
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
    # Caller-supplied routing only: dtype of the routing_weights cotangent.
    routing_weights_dtype: Any = flax.struct.field(pytree_node=False, default=None)


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


def _with_grouped_weight_buffers(tensor, data, scale_inv, original_shape):
    """Rebuild a dense grouped weight while preserving its quantization metadata."""
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
        original_shape=original_shape,
        pre_swizzled=True,
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
    return _with_grouped_weight_buffers(tensor, data, scale_inv, global_shape)


def _weight_scale_shapes(tensor, matrix_shape):
    """Return padded and active per-expert MXFP8 scale shapes."""
    kwargs = {
        "data_layout": tensor.data_layout,
        "is_colwise": tensor.is_colwise,
        "flatten_axis": tensor.flatten_axis - 1,
    }
    return (
        tensor.scaling_mode.get_scale_shape(matrix_shape, is_padded=True, **kwargs),
        tensor.scaling_mode.get_scale_shape(matrix_shape, is_padded=False, **kwargs),
    )


def _read_weight_scale(tensor, expert, padded_shape, plain_shape):
    """Read one expert's active scales in matrix order."""
    scale_size = math.prod(padded_shape)
    swizzled = jax.lax.dynamic_slice_in_dim(tensor.scale_inv, expert * scale_size, scale_size)
    plain = _unswizzle_mxfp8_grouped_scale(swizzled, padded_shape, tensor.is_colwise)
    return plain[: plain_shape[0], : plain_shape[1]]


def _swizzle_weight_scale(plain, padded_shape, is_colwise):
    """Pad and swizzle one expert's scale matrix for grouped GEMM."""
    padded = jnp.pad(
        plain, ((0, padded_shape[0] - plain.shape[0]), (0, padded_shape[1] - plain.shape[1]))
    )
    return swizzled_scale(padded, 1, is_colwise).reshape(-1)


def _gather_quantized_matrix_scales(tensor, fsdp_axis, data_axis, local_scale_shape, global_matrix):
    """Reassemble scales when FSDP splits each expert's matrix dimension."""
    _, local_unpadded_shape = _weight_scale_shapes(tensor, tensor.original_shape[1:])
    global_padded_shape, global_unpadded_shape = _weight_scale_shapes(tensor, global_matrix)
    scale_axis = data_axis - 1
    gathered_scales = []
    for expert in range(tensor.original_shape[0]):
        local_plain = _read_weight_scale(tensor, expert, local_scale_shape, local_unpadded_shape)
        full_plain = jax.lax.all_gather(local_plain, fsdp_axis, axis=scale_axis, tiled=True)
        assert full_plain.shape == global_unpadded_shape
        gathered_scales.append(
            _swizzle_weight_scale(full_plain, global_padded_shape, tensor.is_colwise)
        )
    return jnp.concatenate(gathered_scales)


def _resolve_cudnn_native_weight_layout(wi, hidden, layout=None):
    """Resolve FC1 parameter storage using its global shape or an explicit flag."""
    if layout is not None and not isinstance(layout, bool):
        raise TypeError("cudnn_native_weight_layout must be a bool or None")
    standard = wi.ndim == 3 and wi.shape[1] == hidden and wi.shape[2] % 2 == 0
    native = wi.ndim == 3 and wi.shape[2] == hidden and wi.shape[1] % 64 == 0
    if layout is None:
        if standard and native:
            raise ValueError(
                "Ambiguous square FC1 weight layout; pass cudnn_native_weight_layout="
                "False for standard [E,K,2N] or True for native [E,2N,K] storage."
            )
        if standard or native:
            return native
    elif native if layout else standard:
        return layout
    raise ValueError(f"Invalid FC1 weight shape {wi.shape} for K={hidden} and layout={layout}")


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


def _swiglu_column_indices(width, interleave):
    """Map contiguous gate/up halves to 32-column pairs, or undo that mapping."""
    if width % 64:
        raise ValueError("The global gated dimension must be divisible by 64")
    blocks = jnp.arange(width // 32)
    blocks = (
        blocks.reshape(2, -1).T.reshape(-1) if interleave else blocks.reshape(-1, 2).T.reshape(-1)
    )
    columns = (blocks[:, None] * 32 + jnp.arange(32)[None, :]).reshape(-1)
    return blocks, columns


def _reorder_quantized_swiglu_weight(tensor, *, interleave):
    """Permute gathered MXFP8 columns and their scales without requantization.

    Input logical storage is [E,K,2N]. Both rowwise and colwise copies may use
    physical N or T layout. Gate/up boundaries and permutations are 32-aligned,
    so each quantization block keeps its original data and inverse scale.
    """
    if isinstance(tensor, ScaledTensor2x):
        return ScaledTensor2x(
            _reorder_quantized_swiglu_weight(tensor.rowwise_tensor, interleave=interleave),
            _reorder_quantized_swiglu_weight(tensor.colwise_tensor, interleave=interleave),
        )
    if not isinstance(tensor, GroupedScaledTensor1x):
        raise TypeError("SwiGLU weight reordering requires grouped MXFP8 tensors")
    if tensor.scaling_mode != ScalingMode.MXFP8_1D_SCALING or not tensor.pre_swizzled:
        raise ValueError("SwiGLU weight reordering requires pre-swizzled MXFP8 scales")

    shape = tensor.original_shape
    data_axis = 1 if tensor.data_layout == "T" else 2
    block_indices, column_indices = _swiglu_column_indices(shape[data_axis], interleave)
    data = jnp.take(tensor.data.reshape(shape), column_indices, axis=data_axis).reshape(-1)
    padded_shape, plain_shape = _weight_scale_shapes(tensor, shape[1:])
    scale_axis = data_axis - 1
    # Rowwise scales cover 32 columns; colwise scales have one entry per column.
    scale_indices = (
        block_indices if plain_shape[scale_axis] == shape[data_axis] // 32 else column_indices
    )
    reordered_scales = []
    for expert in range(shape[0]):
        plain = _read_weight_scale(tensor, expert, padded_shape, plain_shape)
        reordered = jnp.take(plain, scale_indices, axis=scale_axis)
        reordered_scales.append(_swizzle_weight_scale(reordered, padded_shape, tensor.is_colwise))
    scale_inv = jnp.concatenate(reordered_scales)
    scale_inv = jnp.pad(scale_inv, (0, tensor.scale_inv.size - scale_inv.size))
    return _with_grouped_weight_buffers(tensor, data, scale_inv, shape)


def _prepare_fc1_weight(
    wi, quantizer, *, fused, native_layout, quant_before_fsdp_ag, fsdp_axis, fsdp_size, sharded_axis
):
    """Convert FC1 storage and gather quantized shards in the correct logical order."""
    gated_axis = 1 if native_layout else 2
    convert_layout = bool(fused) != native_layout
    convert_after_gather = convert_layout and quant_before_fsdp_ag and sharded_axis == gated_axis
    if native_layout and not fused:
        wi = wi.transpose(0, 2, 1)
        if sharded_axis in (1, 2):
            sharded_axis = 3 - sharded_axis
        if not convert_after_gather:
            gate, up = tex.unpack_gated_activation_pair(wi)
            wi = jnp.concatenate((gate, up), axis=-1)
    elif fused and not native_layout and not convert_after_gather:
        gate, up = jnp.split(wi, 2, axis=-1)
        wi = tex.pack_gated_activation_pair(gate, up)

    casted = tex.grouped_quantize(wi, quantizer, flatten_axis=-1)
    if quant_before_fsdp_ag and sharded_axis is not None:
        casted = _gather_quantized_weight(casted, fsdp_axis, fsdp_size, sharded_axis)
    if convert_after_gather:
        casted = _reorder_quantized_swiglu_weight(casted, interleave=bool(fused))
    return casted


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
    use_cudnn_jax_fusion: bool,
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
    casted_wi = _prepare_fc1_weight(
        wi,
        fc1_quantizer_set.kernel,
        fused=use_cudnn_jax_fusion,
        native_layout=cudnn_native_weight_layout,
        quant_before_fsdp_ag=quant_before_fsdp_ag,
        fsdp_axis=fsdp_axis,
        fsdp_size=fsdp_size,
        sharded_axis=wi_fsdp_axis,
    )
    casted_intermediate = None
    if use_cudnn_jax_fusion:
        casted_sorted_x_lhs = casted_sorted_x.get_tensor(usage=TensorUsage.LHS)
        casted_wi_rhs = casted_wi.get_tensor(
            usage=TensorUsage.LHS if cudnn_native_weight_layout else TensorUsage.RHS
        )
        # Quantized gathering may have expanded the gated dimension since packing.
        combined = casted_wi_rhs.data.size // (num_local_experts * hidden)
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
        ) = tex.grouped_gemm_glu(
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
                TensorUsage.RHS
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
    use_cudnn_jax_fusion: bool,
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
        ) = tex.grouped_gemm_dglu(
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
        contracting_dims=(
            (1,),
            (1 if use_cudnn_jax_fusion and cudnn_native_weight_layout else 2,),
        ),
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
        d_wi_gate, d_wi_up = tex.unpack_gated_activation_pair(d_wi_combined)
        d_wi_combined = jnp.concatenate([d_wi_gate, d_wi_up], axis=-1)
    elif not use_cudnn_jax_fusion and cudnn_native_weight_layout:
        d_wi_gate, d_wi_up = jnp.split(d_wi_combined, 2, axis=-1)
        d_wi_combined = tex.pack_gated_activation_pair(d_wi_gate, d_wi_up).transpose(0, 2, 1)
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
    router_info,
    wi,
    wo,
    wi_0_bias,
    wi_1_bias,
    wo_bias,
    quantizer_sets,
    num_experts,
    activation_type,
    mesh_resource,
    input_axes,
    wi_kernel_axes,
    wo_kernel_axes,
    dtype,
    apply_topk_weights_early,
    recv_capacity_per_rank,
    collect_expert_counts,
    use_cudnn_jax_fusion,
    wi_0_checkpoint_name,
    wi_1_checkpoint_name,
    wo_checkpoint_name,
    dispatch_checkpoint_name,
    quant_before_fsdp_ag,
    use_cudnn_fusion: bool = True,
    cudnn_native_weight_layout: Optional[bool] = None,
):
    """Forward: [gate -> route ->] ep_dispatch -> FFN -> ep_combine.

    Returns ``(output, aux_loss, total_recv_tokens, expert_counts)``.
    ``aux_loss`` is a zero scalar unless the router computes it and ``aux_loss_coeff`` > 0.
    """
    with global_shard_guard(mesh_resource):
        ep_axis, data_parallelism_axes = _moe_mesh_axes(mesh_resource)
        from jax.experimental.shard_map import shard_map

        router = router_info if isinstance(router_info, RouterComputationInfo) else None
        compute_aux_loss = router is not None and router.aux_loss_coeff > 0.0

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
        K = _num_experts_per_tok(router_info)
        cudnn_native_weight_layout = _resolve_cudnn_native_weight_layout(
            wi, H, cudnn_native_weight_layout
        )
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
        dispatch_alignment = _CUDNN_JAX_ALIGN_SIZE if use_cudnn_fusion else _ALIGN_SIZE
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

        # ------------------------ Routing ------------------------

        if router is None:
            # ---------------- Caller-supplied routing ----------------
            routing_indices = jax.lax.with_sharding_constraint(
                router_info.routing_indices, NamedSharding(mesh, ep3_spec)
            )
            external_weights = jax.lax.with_sharding_constraint(
                router_info.routing_weights, NamedSharding(mesh, ep3_spec)
            )
            selected_experts = routing_indices.reshape(-1, K).astype(jnp.int32)
            routing_weights = external_weights.reshape(-1, K).astype(dtype)
            logits_2d = routing_map = saved_scores = None
        else:
            # ---------------- Gate (global view) ----------------
            # tex.fused_topk_with_score_function is only validated against its
            # pytorch reference at fp32 (see tests/pytorch/test_fused_router.py:
            # parametrize gates dtype on torch.float32 only; the tolerance helper
            # raises NotImplementedError for any other dtype). Keeping logits in
            # the activation dtype (e.g. bf16) lets sigmoid / softmax / topk
            # accumulate at low precision and silently produce NaNs on tokens
            # whose normalised weights underflow. Cast to fp32 here to stay in
            # the validated regime.
            gate_kernel_cast = router.gate_kernel.astype(x.dtype)
            gate_logits = jnp.einsum("bsh,he->bse", x, gate_kernel_cast)
            logits_2d = gate_logits.reshape(-1, num_experts).astype(jnp.float32)

            # ---------------- Routing (global view) ----------------
            # The fused top-k primitive takes an empty shape-(0,) tensor as "no bias".
            if router.expert_bias is None:
                eb_arg = jnp.zeros((0,), dtype=jnp.float32)
            else:
                eb_arg = router.expert_bias.astype(jnp.float32)
            topk_kwargs = {
                "topk": K,
                "use_pre_softmax": router.use_pre_softmax,
                "num_groups": -1 if router.num_groups is None else router.num_groups,
                "group_topk": -1 if router.group_topk is None else router.group_topk,
                "scaling_factor": router.scaling_factor,
                "score_function": router.score_function,
                "expert_bias": eb_arg,
                "compute_aux_scores": False,
            }
            sparse_probs, routing_map, saved_scores = tex.fused_topk_with_score_function_fwd(
                logits_2d, **topk_kwargs
            )
            sparse_probs = sparse_probs.astype(dtype)
            # argsort on a bool tensor places True last (False=0 < True=1), so the
            # last K indices are the selected expert IDs.
            selected_experts = jnp.argsort(routing_map, axis=-1)[..., -K:]
            routing_weights = jnp.take_along_axis(sparse_probs, selected_experts, axis=-1)

        if collect_expert_counts:
            if router is None:
                expert_counts = jnp.sum(
                    jax.nn.one_hot(selected_experts, num_experts, dtype=jnp.int32),
                    axis=(0, 1),
                )
            else:
                expert_counts = jnp.sum(routing_map.astype(jnp.int32), axis=0)
            # The token dimension is sharded over the compound data/EP axis.
            # Replicating the reduced result emits the required all-reduce.
            expert_counts = jax.lax.with_sharding_constraint(
                expert_counts, NamedSharding(mesh, P())
            )
        else:
            # Keep the custom-VJP output structure stable without paying for an
            # otherwise-unused per-layer all-reduce.
            expert_counts = jnp.zeros((num_experts,), dtype=jnp.int32)

        # ---------------- Aux loss (global view, replicated) ----------------
        # ``fused_moe_aux_loss_fwd`` sums probs and tokens_per_expert across
        # all tokens, which is wrong when T is sharded. Force-replicate the
        # gate logits and recompute the routing map at global view so the
        # kernel sees a complete [T_global, E] tensor. The replication is a
        # single all-gather over (*dp, ep) and lives off the dispatch
        # critical path.
        if compute_aux_loss:
            global_logits_2d = jax.lax.with_sharding_constraint(logits_2d, NamedSharding(mesh, P()))
            _, global_routing_map, _ = tex.fused_topk_with_score_function_fwd(
                global_logits_2d, **topk_kwargs
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
                score_function=router.score_function,
                expert_bias=jnp.zeros((0,), dtype=jnp.float32),
                compute_aux_scores=True,
            )
            aux_loss, aux_const_buf = tex.fused_moe_aux_loss_fwd(
                aux_probs.astype(jnp.float32),
                aux_tokens_per_expert.astype(jnp.int32),
                topk=K,
                coeff=router.aux_loss_coeff,
            )
            aux_loss = aux_loss.astype(dtype)
        else:
            aux_loss = jnp.zeros((), dtype=dtype)
            aux_const_buf = None
            aux_tokens_per_expert = None
            aux_saved_scores = None

        # ---------------- Routing -> (topk_idx, topk_w) at 3D ----------------
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
            topk_idx_dtype=int(jax_dtype_to_te_dtype(topk_idx_3d.dtype)),
        )
        recv_tokens, recv_topk_weights, token_counts, total_recv_tokens, handle_mem = (
            tex.ep_prepare_and_dispatch_fwd(cfg, topk_idx_3d, x, topk_w_3d, recv_pr)
        )
        token_counts = jax.lax.with_sharding_constraint(token_counts, NamedSharding(mesh, ep2_spec))
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
            router=router,
            logits_2d=logits_2d,
            saved_scores=saved_scores,
            routing_map=routing_map,
            selected_experts=selected_experts,
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
            routing_weights_dtype=(
                None if router is not None else router_info.routing_weights.dtype
            ),
        )
        static = {
            "has_bias": has_bias,
            "x_shape": x.shape,
            "recv_pr": recv_pr,
            "cudnn_native_weight_layout": cudnn_native_weight_layout,
        }
        # The integer metadata outputs are non-differentiable (see moe()).
        return (output, aux_loss, total_recv_tokens, expert_counts), (ctx, static)


def _moe_bwd_rule(
    num_experts,
    activation_type,
    mesh_resource,
    input_axes,
    wi_kernel_axes,
    wo_kernel_axes,
    dtype,
    apply_topk_weights_early,
    recv_capacity_per_rank,
    collect_expert_counts,
    use_cudnn_jax_fusion,
    wi_0_checkpoint_name,
    wi_1_checkpoint_name,
    wo_checkpoint_name,
    dispatch_checkpoint_name,
    quant_before_fsdp_ag,
    use_cudnn_fusion,
    cudnn_native_weight_layout,
    residuals,
    cotangents,
):
    """Backward mirror of :func:`_moe_fwd_rule`."""
    with global_shard_guard(mesh_resource):
        ep_axis, data_parallelism_axes = _moe_mesh_axes(mesh_resource)
        del (
            dtype,
            recv_capacity_per_rank,
            collect_expert_counts,
            wi_0_checkpoint_name,
            wi_1_checkpoint_name,
            wo_checkpoint_name,
            dispatch_checkpoint_name,
            quant_before_fsdp_ag,
            use_cudnn_fusion,
            cudnn_native_weight_layout,
        )  # captured / unused in bwd
        from jax.experimental.shard_map import shard_map

        # The integer metadata outputs are non-differentiable; their cotangents are unused.
        d_output, d_aux_loss, _d_total_recv_tokens, _d_expert_counts = cotangents

        ctx, static = residuals
        router = ctx.router
        has_bias = static["has_bias"]
        x_shape = static["x_shape"]
        recv_pr = static["recv_pr"]

        mesh = _get_mesh()
        if mesh is None or mesh.empty:
            raise ValueError("moe(...) requires an active jax.sharding.Mesh.")
        B, S, _ = x_shape
        K = ctx.selected_experts.shape[-1]
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
            bwd_out_specs = (
                ep3_spec,
                ep2_spec,
                kernel_spec,
                kernel_spec,
                None,
                None,
                None,
            )

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

        if router is None:
            # Caller-supplied routing: d_topk_w is the routing-weights cotangent, and
            # the integer routing indices get a zero (None) cotangent.
            d_router_info = RoutingMapInfo(
                routing_indices=None,
                routing_weights=d_topk_w.astype(ctx.routing_weights_dtype),
            )
            d_x = d_x_from_dispatch
        else:
            # ---------------- Routing bwd (global view) ----------------
            # The cotangent on routing_weights is a sparse scatter into sparse_probs
            # at the selected_experts indices.
            d_topk_w_flat = d_topk_w.reshape(-1, K)
            d_sparse_probs = jnp.zeros(ctx.routing_map.shape, dtype=d_topk_w_flat.dtype)
            d_sparse_probs = d_sparse_probs.at[
                jnp.arange(ctx.routing_map.shape[0])[:, None], ctx.selected_experts
            ].set(d_topk_w_flat)

            d_logits_2d = tex.fused_topk_with_score_function_bwd(
                ctx.routing_map,
                ctx.saved_scores,
                d_sparse_probs.astype(ctx.saved_scores.dtype),
                topk=K,
                use_pre_softmax=router.use_pre_softmax,
                scaling_factor=router.scaling_factor,
                score_function=router.score_function,
                compute_aux_scores=False,
            )
            # ---------------- Aux loss bwd (global view, replicated) ----------------
            if router.aux_loss_coeff > 0.0:
                d_logits_2d = d_logits_2d + _moe_aux_loss_bwd(ctx, d_aux_loss, K).astype(
                    d_logits_2d.dtype
                )

            # ---------------- Gate bwd (global view) ----------------
            d_gate_logits = d_logits_2d.reshape(B, S, num_experts)
            gate_kernel_cast = router.gate_kernel.astype(ctx.x.dtype)
            d_x_from_gate = jnp.einsum("bse,he->bsh", d_gate_logits, gate_kernel_cast)
            d_gate_kernel = jnp.einsum("bsh,bse->he", ctx.x, d_gate_logits).astype(
                router.gate_kernel.dtype
            )
            d_gate_kernel = with_sharding_constraint_by_logical_axes(
                d_gate_kernel, router.gate_kernel_axes
            )
            d_x = d_x_from_gate + d_x_from_dispatch
            # expert_bias has no learnable bwd path through fused_topk: the
            # primitive's bwd returns None for the bias slot, so its cotangent is zero.
            d_router_info = router.replace(
                gate_kernel=d_gate_kernel,
                expert_bias=(
                    None if router.expert_bias is None else jnp.zeros_like(router.expert_bias)
                ),
            )

        # Pin output grads to the declared logical axes so downstream
        # optimizers see consistent shardings.
        d_x = with_sharding_constraint_by_logical_axes(d_x, input_axes)
        d_wi = with_sharding_constraint_by_logical_axes(d_wi, wi_kernel_axes)
        d_wo = with_sharding_constraint_by_logical_axes(d_wo, wo_kernel_axes)
        if has_bias:
            wi_bias_axes = (wi_kernel_axes[0], *wi_kernel_axes[2:])
            wo_bias_axes = (wo_kernel_axes[0], *wo_kernel_axes[2:])
            d_wi_0_bias = with_sharding_constraint_by_logical_axes(d_wi_0_bias, wi_bias_axes)
            d_wi_1_bias = with_sharding_constraint_by_logical_axes(d_wi_1_bias, wi_bias_axes)
            d_wo_bias = with_sharding_constraint_by_logical_axes(d_wo_bias, wo_bias_axes)

        return (
            d_x,
            d_router_info,
            d_wi,
            d_wo,
            d_wi_0_bias if has_bias else None,
            d_wi_1_bias if has_bias else None,
            d_wo_bias if has_bias else None,
            ctx.quantizer_sets,
        )


def _moe_aux_loss_bwd(ctx, d_aux_loss, K):
    """Aux-loss contribution to the router logits cotangent."""
    # Reverse the fwd's all-gather/aux pipeline: aux_loss_bwd produces
    # d_aux_probs, then topk_bwd(compute_aux_scores=True) produces the
    # extra d_logits contribution. The replicated tensor adds into the
    # T-sharded routing-side d_logits via JAX's normal broadcast.
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
    return tex.fused_topk_with_score_function_bwd(
        zero_routing_map,
        ctx.aux_saved_scores,
        d_aux_probs.astype(ctx.aux_saved_scores.dtype),
        topk=K,
        use_pre_softmax=False,
        scaling_factor=1.0,
        score_function=ctx.router.score_function,
        compute_aux_scores=True,
    )


# =============================================================================
# custom_vjp + public entry
# =============================================================================


@partial(jax.custom_vjp, nondiff_argnums=tuple(range(8, 26)))
def _moe(
    x,
    router_info,
    wi,
    wo,
    wi_0_bias,
    wi_1_bias,
    wo_bias,
    quantizer_sets,
    num_experts,
    activation_type,
    mesh_resource,
    input_axes,
    wi_kernel_axes,
    wo_kernel_axes,
    dtype,
    apply_topk_weights_early,
    recv_capacity_per_rank,
    collect_expert_counts,
    use_cudnn_jax_fusion,
    wi_0_checkpoint_name,
    wi_1_checkpoint_name,
    wo_checkpoint_name,
    dispatch_checkpoint_name,
    quant_before_fsdp_ag,
    use_cudnn_fusion: bool = True,
    cudnn_native_weight_layout: Optional[bool] = None,
):
    primal, _ = _moe_fwd_rule(
        x,
        router_info,
        wi,
        wo,
        wi_0_bias,
        wi_1_bias,
        wo_bias,
        quantizer_sets,
        num_experts,
        activation_type,
        mesh_resource,
        input_axes,
        wi_kernel_axes,
        wo_kernel_axes,
        dtype,
        apply_topk_weights_early,
        recv_capacity_per_rank,
        collect_expert_counts,
        use_cudnn_jax_fusion,
        wi_0_checkpoint_name,
        wi_1_checkpoint_name,
        wo_checkpoint_name,
        dispatch_checkpoint_name,
        quant_before_fsdp_ag,
        use_cudnn_fusion,
        cudnn_native_weight_layout,
    )
    return primal


_moe.defvjp(_moe_fwd_rule, _moe_bwd_rule)


def moe(
    x: jnp.ndarray,
    router_info: Union[RouterComputationInfo, RoutingMapInfo],
    wi: jnp.ndarray,
    wo: jnp.ndarray,
    wi_0_bias: Optional[jnp.ndarray] = None,
    wi_1_bias: Optional[jnp.ndarray] = None,
    wo_bias: Optional[jnp.ndarray] = None,
    *,
    num_experts: int,
    activation_type: str = "silu",
    apply_topk_weights_early: bool = False,
    quantizer_sets: Tuple[QuantizerSet, QuantizerSet] = (
        noop_quantizer_set,
        noop_quantizer_set,
    ),
    mesh_resource: Optional[MeshResource] = None,
    quant_before_fsdp_ag: bool = False,
    input_axes: Tuple[Optional[str], ...] = (),
    wi_kernel_axes: Tuple[Optional[str], ...] = ("exp", "embed", "mlp"),
    wo_kernel_axes: Tuple[Optional[str], ...] = ("exp", "mlp", "embed"),
    dtype: jnp.dtype = jnp.float32,
    recv_capacity_per_rank: Optional[int] = None,
    collect_expert_counts: bool = True,
    wi_0_checkpoint_name: Optional[str] = None,
    wi_1_checkpoint_name: Optional[str] = None,
    wo_checkpoint_name: Optional[str] = None,
    dispatch_checkpoint_name: Optional[str] = None,
    use_cudnn_fusion: bool = True,
    cudnn_native_weight_layout: Optional[bool] = None,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], jnp.ndarray, jnp.ndarray]:
    """Run a full MoE block under a single fused custom_vjp on the TE EP path.

    Returns ``(output, aux_loss, total_recv_tokens, expert_counts)``. ``aux_loss``
    is ``None`` when routing is caller-supplied or ``aux_loss_coeff == 0``, else
    a 0-d scalar. ``total_recv_tokens`` is a non-differentiable pre-drop
    recv-slot total; see ``ep_dispatch`` for using it to detect overflow.
    ``expert_counts`` is a non-differentiable int32 array of shape
    ``[num_experts]`` containing global top-k assignment counts (or zeros when
    ``collect_expert_counts=False``).

    Parameters
    ----------
    router_info : Union[RouterComputationInfo, RoutingMapInfo]
        :class:`RouterComputationInfo` makes TE compute the gate and top-k
        from the given router parameters. :class:`RoutingMapInfo` supplies
        precomputed expert indices and routing weights (e.g. hash routing)
        and skips TE's router.
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
    collect_expert_counts : bool
        Whether to compute global per-expert assignment counts. Disable this
        when the returned counts are unused to avoid an extra all-reduce.
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
        shard whole experts or a matrix dimension within each expert. When
        FSDP shards the gated dimension, layout conversion permutes gathered
        FP8 data and inverse scales, preserving global gate/up pairing without
        gathering or requantizing full-precision weights.
    use_cudnn_fusion : bool
        Defaults to ``True``: try cuDNN's shared JAX grouped MXFP8 GLU/dGLU APIs
        on SM100+ GPUs; cuDNN selects the architecture-specific kernels. ``False`` uses unfused TE grouped GEMM.
        Requires cuDNN Frontend >= 1.32.0 with compatible JAX APIs.
        Frontends whose JAX GLU/dGLU entry points only support BF16 discrete
        weights fall back to unfused TE grouped GEMM.
        Ineligible calls warn and fall back to TE's regular grouped-GEMM
        implementation. API signatures and GPU capability determine support.
    cudnn_native_weight_layout : Optional[bool]
        ``False`` selects standard contiguous gate/up [E,K,2N] storage;
        ``True`` selects native [E,2N,K] storage with alternating 32-column
        gate/up blocks. Defaults to ``None``, inferring storage from the global
        shape. Ambiguous square shapes require an explicit flag. This option
        describes parameter storage independently of the selected execution path.

    Notes
    -----
    Per-expert dispatch-slot alignment is 128 tokens (``_ALIGN_SIZE``) with
    ``use_cudnn_fusion=False``. Requesting cuDNN fusion (the default) reserves
    256 tokens, also when falling back, to preserve compatibility with EP
    bootstrap buffer sizing.

    MeshResource fields name physical mesh axes, not Flax logical axes.
    ``input_axes``, ``wi_kernel_axes``, ``wo_kernel_axes`` and
    ``RouterComputationInfo.gate_kernel_axes`` remain logical-axis tuples
    resolved through the active Flax rules. The selected MeshResource is also used by internal TE sharding
    in both forward and backward, independently of an enclosing global context.

    See module docstring for the rest of the parameter semantics and the
    surrounding design rationale.
    """
    if not isinstance(use_cudnn_fusion, bool):
        raise TypeError("use_cudnn_fusion must be a bool")
    mesh_resource, quant_before_fsdp_ag = _resolve_moe_mesh_resource(
        mesh_resource, quant_before_fsdp_ag
    )
    ep_axis, data_parallelism_axes = _moe_mesh_axes(mesh_resource)

    if isinstance(router_info, RouterComputationInfo):
        router_info = router_info.replace(
            score_function=_validate_score_function(router_info.score_function),
            aux_loss_coeff=float(router_info.aux_loss_coeff),
        )
        has_aux_loss = router_info.aux_loss_coeff > 0.0
    elif isinstance(router_info, RoutingMapInfo):
        routing_indices = router_info.routing_indices
        routing_weights = router_info.routing_weights
        if routing_indices.ndim != 3 or routing_indices.shape[:2] != x.shape[:2]:
            raise ValueError(
                "routing_indices must have shape [batch, sequence, top_k] matching x, "
                f"got x={x.shape}, routing_indices={routing_indices.shape}."
            )
        if routing_weights.shape != routing_indices.shape:
            raise ValueError(
                "routing_weights must have the same shape as routing_indices, got "
                f"{routing_weights.shape} and {routing_indices.shape}."
            )
        if routing_indices.shape[-1] <= 0:
            raise ValueError("routing_indices must select at least one expert per token.")
        router_info = router_info.replace(routing_indices=routing_indices.astype(jnp.int32))
        has_aux_loss = False
    else:
        raise TypeError(
            "router_info must be a RouterComputationInfo or RoutingMapInfo, got "
            f"{type(router_info).__name__}."
        )

    with global_shard_guard(mesh_resource):
        # Enforce ((outer_dp..., ep), None, None) on inbound activations. The
        # EP comm groups consecutive global ranks (dp_color = rank // ep_size),
        # so ep MUST be innermost in the partition spec. Soft re-pin: free if
        # upstream already matches, single reshard otherwise.
        x = _prepare_moe_input(x, ep_axis, data_parallelism_axes, caller="moe")

        cudnn_native_weight_layout = _resolve_cudnn_native_weight_layout(
            wi, x.shape[-1], cudnn_native_weight_layout
        )

        use_cudnn_jax_fusion = False
        if use_cudnn_fusion:
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

        output, aux_loss, total_recv_tokens, expert_counts = _moe(
            x,
            router_info,
            wi,
            wo,
            wi_0_bias,
            wi_1_bias,
            wo_bias,
            quantizer_sets,
            num_experts,
            activation_type,
            mesh_resource,
            input_axes,
            wi_kernel_axes,
            wo_kernel_axes,
            dtype,
            apply_topk_weights_early,
            recv_capacity_per_rank,
            collect_expert_counts,
            use_cudnn_jax_fusion,
            wi_0_checkpoint_name,
            wi_1_checkpoint_name,
            wo_checkpoint_name,
            dispatch_checkpoint_name,
            quant_before_fsdp_ag,
            use_cudnn_fusion,
            cudnn_native_weight_layout,
        )
        if not has_aux_loss:
            aux_loss = None
        assert (
            output.dtype == x.dtype
        ), f"moe() output dtype {output.dtype} != input dtype {x.dtype}"
        return output, aux_loss, total_recv_tokens, expert_counts
