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
from typing import Optional, Tuple, Union

import torch
import transformer_engine_torch as tex

from .jit import no_torch_dynamo

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
_QB_BOUNDS_CAPTURE_VALIDATED_VERSION_ATTR = "_nvte_qb_bounds_capture_validated_version"


def mark_qb_bin_bounds_validated(bin_bounds: torch.Tensor) -> None:
    """Mark the current QB bounds version as valid after a trusted device-side update.

    This function validates metadata only; it does not inspect or schedule validation of tensor
    values. The caller must guarantee finite FP32 bounds with ``lower < upper``. When called during
    CUDA graph capture, call it immediately after dispatching the trusted in-place update on the
    same stream. The Python marker itself is not replayed, so every value produced on replay must
    satisfy the bounds contract for every possible input and control-flow path.
    """
    if not (
        isinstance(bin_bounds, torch.Tensor)
        and bin_bounds.is_cuda
        and bin_bounds.is_contiguous()
        and bin_bounds.dtype == torch.float32
        and bin_bounds.shape == (2,)
    ):
        raise ValueError("QB bin_bounds must be a contiguous FP32 CUDA tensor with shape [2]")
    version = bin_bounds._version
    setattr(bin_bounds, _QB_BOUNDS_VALIDATED_VERSION_ATTR, version)
    with torch.cuda.device(bin_bounds.device):
        capture_validated_version = version if torch.cuda.is_current_stream_capturing() else None
    setattr(
        bin_bounds,
        _QB_BOUNDS_CAPTURE_VALIDATED_VERSION_ATTR,
        capture_validated_version,
    )


def _validate_qb_bin_bounds(bin_bounds: torch.Tensor) -> bool:
    """Validate QB bounds once per PyTorch tensor version."""
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
    validated_version = getattr(bin_bounds, _QB_BOUNDS_VALIDATED_VERSION_ATTR, None)
    capture_validated_version = getattr(bin_bounds, _QB_BOUNDS_CAPTURE_VALIDATED_VERSION_ATTR, None)
    if validated_version == version and capture_validated_version != version:
        return True
    with torch.cuda.device(bin_bounds.device):
        is_capturing = torch.cuda.is_current_stream_capturing()
        if validated_version == version and is_capturing:
            return True
        if is_capturing:
            raise RuntimeError(
                "QB bin_bounds current version must be validated before CUDA graph capture"
            )
    lower, upper = bin_bounds.detach().cpu().tolist()
    if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
        raise ValueError(
            f"QB bin_bounds values must be finite with lower < upper, got [{lower}, {upper}]"
        )
    mark_qb_bin_bounds_validated(bin_bounds)
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
        assume_valid_bin_bounds: bool,
    ):
        # pylint: disable=missing-function-docstring
        bin_bounds_validated = assume_valid_bin_bounds or _validate_qb_bin_bounds(bin_bounds)
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
        return grad_logits, None, None, None, None, None, None, None, None, None


def _needs_eager_router(call):
    if call.get("topk_indices") is not None and call.get("overwrite_topk_indices", True):
        return "router output buffers"
    if any(
        call.get(name) is not None
        for name in ("qb_histogram", "qb_bin_bounds", "qb_histogram_mode")
    ):
        if not call.get("return_qb_histogram_delta", False):
            return "in-place QB histogram accumulation"
        if not call.get("assume_valid_qb_bin_bounds", False):
            return "QB bin-bound value validation"
    return None


@no_torch_dynamo(when=_needs_eager_router)
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
    overwrite_topk_indices: bool = True,
    return_qb_histogram_delta: bool = False,
    assume_valid_qb_bin_bounds: bool = False,
):
    """
    Fused topk with score function router.

    ``torch.compile`` supports routing maps and dense indices with
    ``overwrite_topk_indices=False``. Quantile Balancing also requires
    ``return_qb_histogram_delta=True`` and ``assume_valid_qb_bin_bounds=True``.
    Buffer mutation and QB value validation fall back to eager execution and do not
    support ``fullgraph=True``.

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
        Optional output buffer with shape [*logits.shape[:-1], topk]. When provided, its dtype
        controls the dense index output dtype and the routing map is not materialized.
        With ``overwrite_topk_indices=False``, only its metadata is used.
    qb_histogram : torch.Tensor, optional
        Caller-owned int32 ``[num_experts, num_bins]`` histogram accumulated in place.
        With ``return_qb_histogram_delta=True``, only its metadata is used; the supplied
        histogram is neither read nor modified.
    qb_bin_bounds : torch.Tensor, optional
        FP32 CUDA tensor ``[lower, upper]`` defining uniform QB histogram bins. Values must be
        finite with ``lower < upper``. With default value validation, eager calls revalidate
        PyTorch-tracked in-place updates. Before eager CUDA graph capture, validate the
        current version eagerly or use
        :func:`mark_qb_bin_bounds_validated` after a trusted device-side update. CUDA graph replay
        does not execute Python validation or advance PyTorch tensor-version counters. Any bounds
        update visible to replay, whether issued outside the graph or captured inside it, is
        therefore trusted and must produce finite values with ``lower < upper`` on every replay.
        Invalid replay-time bounds may silently produce an incorrect histogram.
    qb_histogram_mode : str, optional
        ``"two_kernel"`` or ``"fused_atomic"``. Must be provided with the two QB tensors.
    overwrite_topk_indices : bool, default = True
        Write indices into ``topk_indices`` and return that buffer. If False, return a new
        tensor without modifying the supplied buffer. Ignored when ``topk_indices`` is None.
    return_qb_histogram_delta : bool, default = False
        Return the current call's histogram as a third output instead of accumulating into
        ``qb_histogram``. Requires all three QB arguments. The caller owns accumulation,
        for example ``histogram = histogram + histogram_delta`` across microbatches.
    assume_valid_qb_bin_bounds : bool, default = False
        Skip QB bounds value validation and its CPU synchronization. The caller must
        guarantee finite bounds with ``lower < upper`` on every call and CUDA graph replay,
        including after device-side updates. Tensor metadata is still checked. Ignored
        without QB. Setting this does not mark the tensor as validated for later checked calls.

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
        - Dense indices: [*logits.shape[:-1], topk], with the dtype of ``topk_indices``.
    histogram_delta : torch.Tensor, optional
        Returned only when ``return_qb_histogram_delta=True``. A new non-differentiable
        int32 ``[num_experts, num_bins]`` tensor containing this call's counts, excluding
        all previous counts in ``qb_histogram``.
    """
    if logits.dtype == torch.float64:
        raise ValueError("Current TE does not support float64 router type.")
    routing_map_format = _validate_routing_map_format(routing_map_format)
    if (
        not torch.compiler.is_compiling()
        and topk_indices is not None
        and not overwrite_topk_indices
    ):
        topk_indices = torch.empty_like(topk_indices)
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
        if torch.compiler.is_compiling():
            probs, routing_output, _, histogram_delta = _qb_forward(
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
            return probs, routing_output, histogram_delta
        if return_qb_histogram_delta:
            qb_histogram = torch.zeros_like(qb_histogram)
        probs, routing_output, histogram_output = FusedTopkScoreFunctionQB.apply(
            logits,
            topk,
            scaling_factor,
            expert_bias,
            routing_map_format,
            topk_indices,
            qb_histogram,
            qb_bin_bounds,
            mode,
            assume_valid_qb_bin_bounds,
        )
        if return_qb_histogram_delta:
            return probs, routing_output, histogram_output
        return probs, routing_output
    if return_qb_histogram_delta:
        raise ValueError("return_qb_histogram_delta requires all three QB arguments")
    if torch.compiler.is_compiling():
        probs, routing_output, _ = _topk_forward(
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
    Supports ``torch.compile(fullgraph=True)`` for forward and backward.
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
    if torch.compiler.is_compiling():
        scores, routing_map, _ = _score_forward(logits, topk, score_function, routing_map_format)
        return routing_map, scores
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
    Supports ``torch.compile(fullgraph=True)`` with either form of ``total_num_tokens``.
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
    if torch.compiler.is_compiling():
        token_count = total_num_tokens if isinstance(total_num_tokens, torch.Tensor) else None
        count = 0 if token_count is not None else total_num_tokens
        aux_loss, _ = _aux_loss_forward(
            probs, tokens_per_expert, count, token_count, num_experts, topk, coeff
        )
        return aux_loss
    return FusedAuxLoss.apply(probs, tokens_per_expert, total_num_tokens, num_experts, topk, coeff)


def _routing_map_fake(logits, routing_map_format):
    if routing_map_format == _ROUTING_MAP_FORMAT_FROM_STRING["bitmap_u8"]:
        return logits.new_empty(
            (*logits.shape[:-1], (logits.shape[-1] + 7) // 8), dtype=torch.uint8
        )
    return logits.new_empty(logits.shape, dtype=torch.bool)


@torch.library.custom_op("te_moe::router_topk_fwd", mutates_args=())
def _topk_forward(
    logits: torch.Tensor,
    topk: int,
    use_pre_softmax: bool,
    num_groups: Optional[int],
    group_topk: Optional[int],
    scaling_factor: Optional[float],
    score_function: str,
    expert_bias: Optional[torch.Tensor],
    routing_map_format: int,
    topk_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.cuda.device(logits.device):
        indices = torch.empty_like(topk_indices) if topk_indices is not None else None
        return tex.fused_topk_with_score_function_fwd(
            logits,
            topk,
            use_pre_softmax,
            num_groups,
            group_topk,
            scaling_factor,
            score_function,
            expert_bias,
            routing_map_format,
            indices,
        )


@_topk_forward.register_fake
def _topk_forward_fake(
    logits,
    topk,
    use_pre_softmax,
    num_groups,
    group_topk,
    scaling_factor,
    score_function,
    expert_bias,
    routing_map_format,
    topk_indices=None,
):
    del (
        topk,
        use_pre_softmax,
        num_groups,
        group_topk,
        scaling_factor,
        score_function,
        expert_bias,
    )
    return (
        logits.new_empty(logits.shape),
        (
            torch.empty_like(topk_indices)
            if topk_indices is not None
            else _routing_map_fake(logits, routing_map_format)
        ),
        logits.new_empty(logits.shape, dtype=torch.float32),
    )


@torch.library.custom_op("te_moe::router_topk_bwd", mutates_args=())
def _topk_backward(
    routing_map: torch.Tensor,
    intermediate_output: torch.Tensor,
    grad_probs: torch.Tensor,
    topk: int,
    use_pre_softmax: bool,
    scaling_factor: Optional[float],
    score_function: str,
    routing_map_format: int,
    use_dense_indices: bool = False,
) -> torch.Tensor:
    grad_probs = grad_probs.contiguous()
    grad_logits = torch.empty_like(grad_probs)
    with torch.cuda.device(grad_probs.device):
        tex.fused_topk_with_score_function_bwd(
            routing_map,
            intermediate_output,
            grad_probs,
            grad_logits,
            topk,
            use_pre_softmax,
            scaling_factor,
            score_function,
            use_dense_indices,
            routing_map_format,
        )
    return grad_logits


@_topk_backward.register_fake
def _topk_backward_fake(
    routing_map,
    intermediate_output,
    grad_probs,
    topk,
    use_pre_softmax,
    scaling_factor,
    score_function,
    routing_map_format,
    use_dense_indices=False,
):
    del routing_map, intermediate_output, topk, use_pre_softmax, scaling_factor
    del score_function, routing_map_format, use_dense_indices
    return grad_probs.new_empty(grad_probs.shape)


def _topk_setup_context(ctx, inputs, output):
    (
        _,
        ctx.topk,
        ctx.use_pre_softmax,
        _,
        _,
        ctx.scaling_factor,
        ctx.score_function,
        _,
        ctx.routing_map_format,
        topk_indices,
    ) = inputs
    ctx.use_dense_indices = topk_indices is not None
    _, routing_map, intermediate_output = output
    ctx.mark_non_differentiable(routing_map, intermediate_output)
    ctx.save_for_backward(routing_map, intermediate_output)


def _topk_autograd(ctx, grad_probs, _grad_map, _grad_intermediate):
    routing_map, intermediate_output = ctx.saved_tensors
    grad_logits = _topk_backward(
        routing_map,
        intermediate_output,
        grad_probs,
        ctx.topk,
        ctx.use_pre_softmax,
        ctx.scaling_factor,
        ctx.score_function,
        ctx.routing_map_format,
        ctx.use_dense_indices,
    )
    return grad_logits, None, None, None, None, None, None, None, None, None


_topk_forward.register_autograd(_topk_autograd, setup_context=_topk_setup_context)


@torch.library.custom_op("te_moe::router_qb_fwd", mutates_args=())
def _qb_forward(
    logits: torch.Tensor,
    topk: int,
    scaling_factor: Optional[float],
    expert_bias: torch.Tensor,
    routing_map_format: int,
    topk_indices: Optional[torch.Tensor],
    histogram: torch.Tensor,
    bin_bounds: torch.Tensor,
    histogram_mode: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.cuda.device(logits.device):
        indices = torch.empty_like(topk_indices) if topk_indices is not None else None
        histogram_delta = torch.zeros_like(histogram)
        probs, routing_output, intermediate_output, _, histogram_delta = (
            tex.fused_topk_with_score_function_qb_fwd(
                logits,
                topk,
                scaling_factor,
                expert_bias,
                routing_map_format,
                indices,
                histogram_delta,
                bin_bounds,
                histogram_mode,
                True,
            )
        )
    return probs, routing_output, intermediate_output, histogram_delta


@_qb_forward.register_fake
def _qb_forward_fake(
    logits,
    topk,
    scaling_factor,
    expert_bias,
    routing_map_format,
    topk_indices,
    histogram,
    bin_bounds,
    histogram_mode,
):
    del topk, scaling_factor, expert_bias, bin_bounds, histogram_mode
    return (
        logits.new_empty(logits.shape),
        (
            torch.empty_like(topk_indices)
            if topk_indices is not None
            else _routing_map_fake(logits, routing_map_format)
        ),
        logits.new_empty(logits.shape, dtype=torch.float32),
        torch.empty_like(histogram),
    )


def _qb_setup_context(ctx, inputs, output):
    _, ctx.topk, ctx.scaling_factor, _, ctx.routing_map_format, indices, _, _, _ = inputs
    ctx.use_dense_indices = indices is not None
    _, routing_output, intermediate_output, histogram_delta = output
    ctx.mark_non_differentiable(routing_output, intermediate_output, histogram_delta)
    ctx.save_for_backward(routing_output, intermediate_output)


def _qb_autograd(ctx, grad_probs, _grad_routing, _grad_intermediate, _grad_histogram):
    routing_output, intermediate_output = ctx.saved_tensors
    grad_logits = _topk_backward(
        routing_output,
        intermediate_output,
        grad_probs,
        ctx.topk,
        False,
        ctx.scaling_factor,
        "sigmoid",
        ctx.routing_map_format,
        ctx.use_dense_indices,
    )
    return grad_logits, None, None, None, None, None, None, None, None


_qb_forward.register_autograd(_qb_autograd, setup_context=_qb_setup_context)


@torch.library.custom_op("te_moe::router_score_fwd", mutates_args=())
def _score_forward(
    logits: torch.Tensor,
    topk: int,
    score_function: str,
    routing_map_format: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.cuda.device(logits.device):
        return tex.fused_score_for_moe_aux_loss_fwd(
            logits, topk, score_function, routing_map_format
        )


@_score_forward.register_fake
def _score_forward_fake(logits, topk, score_function, routing_map_format):
    del topk, score_function
    return (
        logits.new_empty(logits.shape, dtype=torch.float32),
        _routing_map_fake(logits, routing_map_format),
        logits.new_empty(logits.shape, dtype=torch.float32),
    )


@torch.library.custom_op("te_moe::router_score_bwd", mutates_args=())
def _score_backward(
    intermediate_output: torch.Tensor,
    grad_scores: torch.Tensor,
    topk: int,
    score_function: str,
    logits_dtype: torch.dtype,
) -> torch.Tensor:
    grad_scores = grad_scores.contiguous()
    grad_logits = grad_scores.new_empty(grad_scores.shape, dtype=logits_dtype)
    with torch.cuda.device(grad_scores.device):
        tex.fused_score_for_moe_aux_loss_bwd(
            intermediate_output, grad_scores, grad_logits, topk, score_function
        )
    return grad_logits


@_score_backward.register_fake
def _score_backward_fake(intermediate_output, grad_scores, topk, score_function, logits_dtype):
    del intermediate_output, topk, score_function
    return grad_scores.new_empty(grad_scores.shape, dtype=logits_dtype)


def _score_setup_context(ctx, inputs, output):
    logits, ctx.topk, ctx.score_function, _ = inputs
    ctx.logits_dtype = logits.dtype
    _, routing_map, intermediate_output = output
    ctx.mark_non_differentiable(routing_map, intermediate_output)
    ctx.save_for_backward(intermediate_output)


def _score_autograd(ctx, grad_scores, _grad_map, _grad_intermediate):
    (intermediate_output,) = ctx.saved_tensors
    grad_logits = _score_backward(
        intermediate_output, grad_scores, ctx.topk, ctx.score_function, ctx.logits_dtype
    )
    return grad_logits, None, None, None


_score_forward.register_autograd(_score_autograd, setup_context=_score_setup_context)


@torch.library.custom_op("te_moe::router_aux_loss_fwd", mutates_args=())
def _aux_loss_forward(
    probs: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    total_num_tokens: int,
    token_count: Optional[torch.Tensor],
    num_experts: int,
    topk: int,
    coeff: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.cuda.device(probs.device):
        if token_count is not None:
            return tex.fused_moe_aux_loss_fwd_graph_safe(
                probs,
                tokens_per_expert,
                token_count,
                num_experts,
                probs.shape[0],
                probs.shape[1],
                topk,
                coeff,
            )
        return tex.fused_moe_aux_loss_fwd(
            probs,
            tokens_per_expert,
            total_num_tokens,
            num_experts,
            probs.shape[0],
            probs.shape[1],
            topk,
            coeff,
        )


@_aux_loss_forward.register_fake
def _aux_loss_forward_fake(
    probs,
    tokens_per_expert,
    total_num_tokens,
    token_count,
    num_experts,
    topk,
    coeff,
):
    del tokens_per_expert, total_num_tokens, token_count, num_experts, topk, coeff
    return probs.new_empty(()), probs.new_empty((2,), dtype=torch.float32)


@torch.library.custom_op("te_moe::router_aux_loss_bwd", mutates_args=())
def _aux_loss_backward(
    const_buf: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    num_rows: int,
    num_cols: int,
    grad_aux_loss: torch.Tensor,
) -> torch.Tensor:
    with torch.cuda.device(grad_aux_loss.device):
        return tex.fused_moe_aux_loss_bwd(
            const_buf, tokens_per_expert, num_rows, num_cols, grad_aux_loss
        )


@_aux_loss_backward.register_fake
def _aux_loss_backward_fake(const_buf, tokens_per_expert, num_rows, num_cols, grad_aux_loss):
    del const_buf, tokens_per_expert
    return grad_aux_loss.new_empty((num_rows, num_cols))


def _aux_loss_setup_context(ctx, inputs, output):
    probs, tokens_per_expert, _, _, _, _, _ = inputs
    _, const_buf = output
    ctx.num_rows, ctx.num_cols = probs.shape
    ctx.mark_non_differentiable(const_buf)
    ctx.save_for_backward(const_buf, tokens_per_expert)


def _aux_loss_autograd(ctx, grad_aux_loss, _grad_const):
    const_buf, tokens_per_expert = ctx.saved_tensors
    grad_probs = _aux_loss_backward(
        const_buf, tokens_per_expert, ctx.num_rows, ctx.num_cols, grad_aux_loss
    )
    return grad_probs, None, None, None, None, None, None


_aux_loss_forward.register_autograd(_aux_loss_autograd, setup_context=_aux_loss_setup_context)
