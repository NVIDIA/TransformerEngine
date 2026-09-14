# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""
Fused functions used in the MoE router

Precision Notes:
- FP64 is currently not supported.
- Inputs are casted into FP32 when loading from global memory.
- All the math/calculations/accumulations are in FP32 in the kernels.
- "scores" is always in FP32 (match the MCore implementation).
- "intermediate_output" is always in FP32 for better backward precision.
- Only cast to low-precision when necessary and the casting only happens in writing to
  global memory. For example, the gradient is required to have the same dtype as the input.
"""

import math
from typing import Optional, Union

import torch
import transformer_engine_torch as tex

# Re-export the C++ enum NVTERoutingMapFormat under a friendlier Python name.
# Members:
#   RoutingMapFormat.BYTEMAP   — bool[num_tokens, num_experts]
#   RoutingMapFormat.BITMAP_U8 — uint8[num_tokens, ceil(num_experts/8)],
#                                LSB-first / little-endian packing along the
#                                expert axis.
RoutingMapFormat = tex.NVTERoutingMapFormat
QBHistogramMode = tex.NVTEQBHistogramMode


_ROUTING_MAP_FORMAT_FROM_STRING = {
    "bytemap": int(RoutingMapFormat.BYTEMAP),
    "bitmap_u8": int(RoutingMapFormat.BITMAP_U8),
}
_VALID_ROUTING_MAP_FORMAT_INTS = frozenset(_ROUTING_MAP_FORMAT_FROM_STRING.values())
_QB_HISTOGRAM_MODE_FROM_STRING = {
    "two_kernel": int(QBHistogramMode.TWO_KERNEL),
    "fused_atomic": int(QBHistogramMode.FUSED_ATOMIC),
}
_QB_BOUNDS_VALIDATED_VERSION_ATTR = "_nvte_qb_bounds_validated_version"


def _validate_qb_bin_bounds(bin_bounds: torch.Tensor) -> bool:
    """Validate CUDA-resident QB bounds once per PyTorch tensor version."""
    if not (
        isinstance(bin_bounds, torch.Tensor)
        and bin_bounds.is_cuda
        and bin_bounds.is_contiguous()
        and bin_bounds.dtype == torch.float32
        and bin_bounds.shape == (2,)
    ):
        # The C++ binding owns metadata validation and its detailed error messages.
        return False

    version = bin_bounds._version
    if getattr(bin_bounds, _QB_BOUNDS_VALIDATED_VERSION_ATTR, None) == version:
        return True
    with torch.cuda.device(bin_bounds.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "QB bin_bounds must be validated by an eager router call before CUDA graph capture"
            )
    lower, upper = bin_bounds.detach().cpu().tolist()
    if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
        raise ValueError(
            f"QB bin_bounds values must be finite with lower < upper, got [{lower}, {upper}]"
        )
    setattr(bin_bounds, _QB_BOUNDS_VALIDATED_VERSION_ATTR, version)
    return True


def _validate_routing_map_format(
    routing_map_format: Union[str, RoutingMapFormat, int],
) -> int:
    """Coerce user-supplied routing_map_format into a plain int (0 or 1).

    Accepts the enum, an int matching one of the enum's values, or the
    canonical lowercase strings ``"bytemap"`` / ``"bitmap_u8"``.
    """
    # NVTERoutingMapFormat is a standalone pybind11 enum, not a subclass of int.
    if isinstance(routing_map_format, RoutingMapFormat):
        return int(routing_map_format)
    if isinstance(routing_map_format, int):
        if routing_map_format in _VALID_ROUTING_MAP_FORMAT_INTS:
            return routing_map_format
        raise ValueError(
            f"routing_map_format int must be one of {sorted(_VALID_ROUTING_MAP_FORMAT_INTS)}; "
            f"got {routing_map_format!r}"
        )
    if isinstance(routing_map_format, str):
        val = _ROUTING_MAP_FORMAT_FROM_STRING.get(routing_map_format)
        if val is not None:
            return val
        raise ValueError(
            "routing_map_format string must be 'bytemap' or 'bitmap_u8' (lowercase); "
            f"got {routing_map_format!r}"
        )
    raise TypeError(
        "routing_map_format must be an int, a RoutingMapFormat enum, or 'bytemap' / "
        f"'bitmap_u8'; got {routing_map_format!r}"
    )


class FusedTopkScoreFunction(torch.autograd.Function):
    """
    Fused Topk with Score Function router.
    Currently, support "softmax", "sigmoid" and "sqrtsoftplus".
    """

    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        topk: int,
        use_pre_softmax: bool,
        num_groups: Optional[int],
        group_topk: Optional[int],
        scaling_factor: Optional[float],
        score_function: str,
        expert_bias: Optional[torch.Tensor],
        routing_map_format: int,
        topk_indices: Optional[torch.Tensor],
    ):
        # pylint: disable=missing-function-docstring
        probs, routing_output, intermediate_output = tex.fused_topk_with_score_function_fwd(
            logits,
            topk,
            use_pre_softmax,
            num_groups,
            group_topk,
            scaling_factor,
            score_function,
            expert_bias,
            routing_map_format,
            topk_indices,
        )
        if topk_indices is not None:
            routing_output = topk_indices
        if topk_indices is not None:
            ctx.mark_dirty(topk_indices)
        ctx.mark_non_differentiable(routing_output)
        ctx.save_for_backward(routing_output, intermediate_output)
        ctx.use_pre_softmax = use_pre_softmax
        ctx.topk = topk
        ctx.scaling_factor = scaling_factor
        ctx.score_function = score_function
        ctx.routing_map_format = routing_map_format
        ctx.use_dense_indices = topk_indices is not None
        return probs, routing_output

    @staticmethod
    def backward(ctx, grad_probs, _):
        # pylint: disable=missing-function-docstring
        routing_map, intermediate_output = ctx.saved_tensors
        if not grad_probs.is_contiguous():
            grad_probs = grad_probs.contiguous()
        grad_logits = torch.empty_like(grad_probs)
        tex.fused_topk_with_score_function_bwd(
            routing_map,
            intermediate_output,
            grad_probs,
            grad_logits,
            ctx.topk,
            ctx.use_pre_softmax,
            ctx.scaling_factor,
            ctx.score_function,
            ctx.use_dense_indices,
            ctx.routing_map_format,
        )
        return grad_logits, None, None, None, None, None, None, None, None, None


class FusedTopkScoreFunctionQB(torch.autograd.Function):
    """Kimi K3 QB router with histogram accumulation."""

    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        topk: int,
        scaling_factor: Optional[float],
        expert_bias: torch.Tensor,
        routing_map_format: int,
        topk_indices: Optional[torch.Tensor],
        histogram: torch.Tensor,
        bin_bounds: torch.Tensor,
        histogram_mode: int,
    ):
        # pylint: disable=missing-function-docstring
        bin_bounds_validated = _validate_qb_bin_bounds(bin_bounds)
        (
            probs,
            routing_output,
            intermediate_output,
            _cutoff,
            histogram_output,
        ) = tex.fused_topk_with_score_function_qb_fwd(
            logits,
            topk,
            scaling_factor,
            expert_bias,
            routing_map_format,
            topk_indices,
            histogram,
            bin_bounds,
            histogram_mode,
            bin_bounds_validated,
        )
        if topk_indices is not None:
            routing_output = topk_indices
            ctx.mark_dirty(topk_indices)
        ctx.mark_dirty(histogram)
        ctx.mark_non_differentiable(routing_output, histogram_output)
        ctx.save_for_backward(routing_output, intermediate_output)
        ctx.topk = topk
        ctx.scaling_factor = scaling_factor
        ctx.routing_map_format = routing_map_format
        ctx.use_dense_indices = topk_indices is not None
        return probs, routing_output, histogram_output

    @staticmethod
    def backward(ctx, grad_probs, _, _histogram_grad):
        # pylint: disable=missing-function-docstring
        routing_output, intermediate_output = ctx.saved_tensors
        if not grad_probs.is_contiguous():
            grad_probs = grad_probs.contiguous()
        grad_logits = torch.empty_like(grad_probs)
        tex.fused_topk_with_score_function_bwd(
            routing_output,
            intermediate_output,
            grad_probs,
            grad_logits,
            ctx.topk,
            False,
            ctx.scaling_factor,
            "sigmoid",
            ctx.use_dense_indices,
            ctx.routing_map_format,
        )
        return grad_logits, None, None, None, None, None, None, None, None


def fused_topk_with_score_function(
    logits: torch.Tensor,
    topk: int,
    use_pre_softmax: bool,
    num_groups: Optional[int],
    group_topk: Optional[int],
    scaling_factor: Optional[float],
    score_function: str,
    expert_bias: Optional[torch.Tensor],
    routing_map_format: Union[str, RoutingMapFormat, int] = RoutingMapFormat.BYTEMAP,
    topk_indices: Optional[torch.Tensor] = None,
    qb_histogram: Optional[torch.Tensor] = None,
    qb_bin_bounds: Optional[torch.Tensor] = None,
    qb_histogram_mode: Optional[str] = None,
):
    """
    Fused topk with score function router.
    Parameters
    ----------
    logits : torch.Tensor in fp32/bf16/fp16
    topk : int
    use_pre_softmax : bool
        if enabled, the computation order: softmax -> topk.
    num_groups : int, optional
        used in the group topk
    group_topk : int, optional
        used in the group topk
    scaling_factor : float, optional
    score_function : str
        currently support "softmax", "sigmoid" and "sqrtsoftplus".
    expert_bias : torch.Tensor, optional
        could be used with the sigmoid/sqrtsoftplus score functions.
    routing_map_format : Union[str, RoutingMapFormat, int], optional
        Output layout for routing_map. ``"bytemap"`` / ``RoutingMapFormat.BYTEMAP``
        (default) returns a bool[T, E] tensor; ``"bitmap_u8"`` /
        ``RoutingMapFormat.BITMAP_U8`` returns a uint8[T, ceil(E/8)] tensor with
        bit ``(e % 8)`` of byte ``(e / 8)`` set when token ``t`` routes to expert
        ``e`` (LSB-first / little-endian packing along the expert axis).
    topk_indices : torch.Tensor, optional
        Optional output buffer with shape [num_tokens, topk]. When provided, its dtype
        controls the dense index output dtype and the routing map is not materialized.
    qb_histogram : torch.Tensor, optional
        Caller-owned int32 ``[num_experts, num_bins]`` histogram accumulated in place.
    qb_bin_bounds : torch.Tensor, optional
        FP32 CUDA tensor ``[lower, upper]`` defining uniform QB histogram bins. Values must be
        finite with ``lower < upper``. Bounds are revalidated after PyTorch-tracked in-place
        updates; validate once with an eager call before CUDA graph capture.
    qb_histogram_mode : str, optional
        ``"two_kernel"`` or ``"fused_atomic"``. Must be provided with the two QB tensors.

    Returns
    -------
    probs : torch.Tensor in the same dtype as the "logits".
        Same shape as ``logits``.
    routing_map : torch.Tensor
        Same leading dims as ``logits``; trailing dim and dtype depend on
        routing_map_format, or dense top-k indices when topk_indices is provided:
        - BYTEMAP:   bool[*logits.shape[:-1], num_experts]
        - BITMAP_U8: uint8[*logits.shape[:-1], ceil(num_experts/8)]
          LSB-first bit-packed.
    """
    if logits.dtype == torch.float64:
        raise ValueError("Current TE does not support float64 router type.")
    routing_map_format = _validate_routing_map_format(routing_map_format)
    qb_arguments = (qb_histogram, qb_bin_bounds, qb_histogram_mode)
    if any(value is not None for value in qb_arguments):
        if any(value is None for value in qb_arguments):
            raise ValueError(
                "qb_histogram, qb_bin_bounds, and qb_histogram_mode must be provided together"
            )
        if score_function != "sigmoid":
            raise ValueError("Quantile Balancing only supports score_function='sigmoid'")
        if expert_bias is None:
            raise ValueError("Quantile Balancing requires expert_bias")
        if num_groups is not None or group_topk is not None:
            raise ValueError("Quantile Balancing does not support grouped Top-k")
        mode = _QB_HISTOGRAM_MODE_FROM_STRING.get(qb_histogram_mode)
        if mode is None:
            raise ValueError(
                "qb_histogram_mode must be 'two_kernel' or 'fused_atomic', "
                f"got {qb_histogram_mode!r}"
            )
        probs, routing_output, _ = FusedTopkScoreFunctionQB.apply(
            logits,
            topk,
            scaling_factor,
            expert_bias,
            routing_map_format,
            topk_indices,
            qb_histogram,
            qb_bin_bounds,
            mode,
        )
        return probs, routing_output
    return FusedTopkScoreFunction.apply(
        logits,
        topk,
        use_pre_softmax,
        num_groups,
        group_topk,
        scaling_factor,
        score_function,
        expert_bias,
        routing_map_format,
        topk_indices,
    )


class FusedComputeScoresForMoEAuxLoss(torch.autograd.Function):
    """
    Fused compute scores for MoE aux loss.
    """

    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        topk: int,
        score_function: str,
        routing_map_format: int,
    ):
        # pylint: disable=missing-function-docstring
        scores, routing_map, intermediate_output = tex.fused_score_for_moe_aux_loss_fwd(
            logits=logits,
            topk=topk,
            score_function=score_function,
            routing_map_format=routing_map_format,
        )
        ctx.save_for_backward(intermediate_output)
        ctx.topk = topk
        ctx.score_function = score_function
        # scores is FP32 but logits/grad_logits may be bf16/fp16 — remember the
        # input dtype for the backward allocation.
        ctx.logits_dtype = logits.dtype
        return routing_map, scores

    @staticmethod
    def backward(ctx, _, grad_scores):
        # pylint: disable=missing-function-docstring
        intermediate_output = ctx.saved_tensors[0]
        if not grad_scores.is_contiguous():
            grad_scores = grad_scores.contiguous()
        grad_logits = torch.empty(
            grad_scores.shape, dtype=ctx.logits_dtype, device=grad_scores.device
        )
        tex.fused_score_for_moe_aux_loss_bwd(
            intermediate_output=intermediate_output,
            grad_scores=grad_scores,
            grad_logits=grad_logits,
            topk=ctx.topk,
            score_function=ctx.score_function,
        )
        return grad_logits, None, None, None


def fused_compute_score_for_moe_aux_loss(
    logits: torch.Tensor,
    topk: int,
    score_function: str,
    routing_map_format: Union[str, RoutingMapFormat, int] = RoutingMapFormat.BYTEMAP,
):
    """
    Fused compute scores for MoE aux loss, subset of the fused_topk_with_score_function.
    Parameters
    ----------
    logits : torch.Tensor in fp32/bf16/fp16
    topk : int
    score_function : str
        currently support "softmax", "sigmoid" and "sqrtsoftplus".
    routing_map_format : Union[str, RoutingMapFormat, int], optional
        Output layout for routing_map; see :func:`fused_topk_with_score_function`.

    Returns
    -------
    routing_map : torch.Tensor
        Same leading dims as ``logits``; trailing dim and dtype depend on
        routing_map_format (bool[..., num_experts] for BYTEMAP,
        uint8[..., ceil(num_experts/8)] for BITMAP_U8).
    scores : torch.Tensor in fp32
        Same shape as ``logits``.
    """
    routing_map_format = _validate_routing_map_format(routing_map_format)
    return FusedComputeScoresForMoEAuxLoss.apply(logits, topk, score_function, routing_map_format)


class FusedAuxLoss(torch.autograd.Function):
    """
    Fused MoE aux loss. ``total_num_tokens`` may be either a Python int
    (host-folded coefficient, original fast path) or a 0-dim int64 CUDA
    tensor (device-folded coefficient, CUDA-graph-safe path).
    """

    @staticmethod
    def forward(
        ctx,
        probs: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        total_num_tokens: Union[int, torch.Tensor],
        num_experts: int,
        topk: int,
        coeff: float,
    ):
        # pylint: disable=missing-function-docstring
        num_rows = probs.size(0)
        num_cols = probs.size(1)
        if isinstance(total_num_tokens, torch.Tensor):
            aux_loss, Const_buf = tex.fused_moe_aux_loss_fwd_graph_safe(
                probs=probs,
                tokens_per_expert=tokens_per_expert,
                total_num_tokens=total_num_tokens,
                num_experts=num_experts,
                num_rows=num_rows,
                num_cols=num_cols,
                topk=topk,
                coeff=coeff,
            )
        else:
            aux_loss, Const_buf = tex.fused_moe_aux_loss_fwd(
                probs=probs,
                tokens_per_expert=tokens_per_expert,
                total_num_tokens=int(total_num_tokens),
                num_experts=num_experts,
                num_rows=num_rows,
                num_cols=num_cols,
                topk=topk,
                coeff=coeff,
            )
        ctx.save_for_backward(Const_buf, tokens_per_expert)
        ctx.num_rows = num_rows
        ctx.num_cols = num_cols
        return aux_loss

    @staticmethod
    def backward(ctx, grad_aux_loss):
        # pylint: disable=missing-function-docstring
        Const_buf, tokens_per_expert = ctx.saved_tensors
        grad_probs = tex.fused_moe_aux_loss_bwd(
            Const_buf=Const_buf,
            tokens_per_expert=tokens_per_expert,
            num_rows=ctx.num_rows,
            num_cols=ctx.num_cols,
            grad_aux_loss=grad_aux_loss,
        )
        return grad_probs, None, None, None, None, None


def fused_moe_aux_loss(
    probs: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    total_num_tokens: Union[int, torch.Tensor],
    num_experts: int,
    topk: int,
    coeff: float,
) -> torch.Tensor:
    """
    Fused MoE aux loss.
    Parameters
    ----------
    probs : torch.Tensor in fp32/bf16/fp16
    tokens_per_expert : torch.Tensor in int32/int64/fp32/bf16
        the number of tokens per expert.
    total_num_tokens : int or 0-dim int64 CUDA torch.Tensor
        the total number of tokens used in the aux loss calculation. Pass a
        Python int for the fastest path (coefficient folded on the host).
        Pass a 0-dim int64 CUDA tensor when the call is captured into a
        CUDA Graph and the value must stay dynamic across replays; the
        coefficient is computed on device by the main reduction kernel.
    num_experts : int
    topk : int
    coeff : float
        the coefficient of the aux loss.

    Returns
    -------
    aux_loss : torch.Tensor.
        A scalar tensor in the same dtype as the "probs".
    """
    return FusedAuxLoss.apply(probs, tokens_per_expert, total_num_tokens, num_experts, topk, coeff)
