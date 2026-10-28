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

import warnings
from typing import Optional, Tuple, Union

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


def mark_qb_bin_bounds_validated(
    bin_bounds: torch.Tensor,  # pylint: disable=unused-argument
) -> None:
    """Deprecated no-op.

    QB ``bin_bounds`` values are no longer validated, so there is nothing to mark. This function
    will be removed in a future release.
    """
    warnings.warn(
        "mark_qb_bin_bounds_validated is deprecated and has no effect because QB bin_bounds "
        "values are no longer validated. It will be removed in a future release.",
        category=DeprecationWarning,
        stacklevel=2,
    )


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
        ctx.compiled = False
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
    def backward(ctx, grad_probs, _, _grad_intermediate=None):
        # pylint: disable=missing-function-docstring
        routing_map, intermediate_output = ctx.saved_tensors
        if ctx.compiled:
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
        )
        ctx.compiled = False
        if topk_indices is not None:
            routing_output = topk_indices
            ctx.mark_dirty(topk_indices, histogram)
        else:
            ctx.mark_dirty(histogram)
        ctx.mark_non_differentiable(routing_output, histogram_output)
        ctx.save_for_backward(routing_output, intermediate_output)
        ctx.topk = topk
        ctx.scaling_factor = scaling_factor
        ctx.routing_map_format = routing_map_format
        ctx.use_dense_indices = topk_indices is not None
        return probs, routing_output, histogram_output

    @staticmethod
    def backward(ctx, grad_probs, _, _histogram_grad, _grad_intermediate=None):
        # pylint: disable=missing-function-docstring
        routing_output, intermediate_output = ctx.saved_tensors
        if ctx.compiled:
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


class _CompiledFusedTopkScoreFunction(FusedTopkScoreFunction):
    """
    Fused Topk with Score Function router.
    Currently, support "softmax", "sigmoid" and "sqrtsoftplus".
    """

    @staticmethod
    def forward(
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
        probs, routing_output, intermediate_output = _topk_forward(
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
        return probs, routing_output, intermediate_output

    @staticmethod
    def setup_context(ctx, inputs, output):
        # pylint: disable=missing-function-docstring
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
        _, routing_output, intermediate_output = output
        ctx.compiled = True
        if topk_indices is not None:
            ctx.mark_dirty(topk_indices)
        ctx.mark_non_differentiable(routing_output, intermediate_output)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(routing_output, intermediate_output)
        ctx.use_dense_indices = topk_indices is not None


class _CompiledFusedTopkScoreFunctionQB(FusedTopkScoreFunctionQB):
    """Kimi K3 QB router with histogram accumulation."""

    @staticmethod
    def forward(
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
        probs, routing_output, intermediate_output = _qb_forward(
            logits,
            topk,
            scaling_factor,
            expert_bias,
            routing_map_format,
            topk_indices,
            histogram,
            bin_bounds,
            histogram_mode,
        )
        histogram_output = histogram
        if topk_indices is not None:
            routing_output = topk_indices
        return probs, routing_output, histogram_output, intermediate_output

    @staticmethod
    def setup_context(ctx, inputs, output):
        # pylint: disable=missing-function-docstring
        (
            _,
            ctx.topk,
            ctx.scaling_factor,
            _,
            ctx.routing_map_format,
            topk_indices,
            histogram,
            _,
            _,
        ) = inputs
        _, routing_output, histogram_output, intermediate_output = output
        ctx.compiled = True
        if topk_indices is not None:
            ctx.mark_dirty(topk_indices, histogram)
        else:
            ctx.mark_dirty(histogram)
        ctx.mark_non_differentiable(routing_output, histogram_output, intermediate_output)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(routing_output, intermediate_output)
        ctx.use_dense_indices = topk_indices is not None


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
    Fused topk with score function router. Supports ``torch.compile(fullgraph=True)``.

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
        The buffer is overwritten and returned in both eager and compiled execution.
        Its contents must remain unchanged until the corresponding backward completes.
        For automatic CUDA graph replay, use a module buffer or
        ``torch._dynamo.mark_static_address``; otherwise capture may be skipped.
    qb_histogram : torch.Tensor, optional
        Caller-owned int32 ``[num_experts, num_bins]`` histogram accumulated in place in eager
        and compiled execution. For automatic CUDA graph replay, use a module buffer or
        ``torch._dynamo.mark_static_address``; otherwise capture may be skipped.
    qb_bin_bounds : torch.Tensor, optional
        FP32 CUDA tensor ``[lower, upper]`` defining uniform QB histogram bins. Values must be
        finite with ``lower < upper``. The values are only read on the device and are not
        validated, so the tensor may be updated in place at any time, including inside a CUDA
        graph. Invalid bounds silently produce an incorrect histogram.
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
        - Dense indices: [*logits.shape[:-1], topk], with the dtype of ``topk_indices``.
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
        function = (
            _CompiledFusedTopkScoreFunctionQB
            if torch.compiler.is_compiling()
            else FusedTopkScoreFunctionQB
        )
        outputs = function.apply(
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
        return outputs[:2]
    function = (
        _CompiledFusedTopkScoreFunction if torch.compiler.is_compiling() else FusedTopkScoreFunction
    )
    outputs = function.apply(
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
    return outputs[:2]


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


def _topk_forward(
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
):
    if topk_indices is None:
        return _topk_map_forward(
            logits,
            topk,
            use_pre_softmax,
            num_groups,
            group_topk,
            scaling_factor,
            score_function,
            expert_bias,
            routing_map_format,
        )
    probs, intermediate_output = _topk_indices_forward(
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
    return probs, topk_indices, intermediate_output


def _qb_forward(
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
    if topk_indices is None:
        return _qb_map_forward(
            logits,
            topk,
            scaling_factor,
            expert_bias,
            routing_map_format,
            histogram,
            bin_bounds,
            histogram_mode,
        )
    probs, intermediate_output = _qb_indices_forward(
        logits,
        topk,
        scaling_factor,
        expert_bias,
        routing_map_format,
        topk_indices,
        histogram,
        bin_bounds,
        histogram_mode,
    )
    return probs, topk_indices, intermediate_output


@torch.library.custom_op("te_moe::router_topk_map_fwd", mutates_args=())
def _topk_map_forward(
    logits: torch.Tensor,
    topk: int,
    use_pre_softmax: bool,
    num_groups: Optional[int],
    group_topk: Optional[int],
    scaling_factor: Optional[float],
    score_function: str,
    expert_bias: Optional[torch.Tensor],
    routing_map_format: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.cuda.device(logits.device):
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
            None,
        )


@_topk_map_forward.register_fake
def _topk_map_forward_fake(
    logits,
    topk,
    use_pre_softmax,
    num_groups,
    group_topk,
    scaling_factor,
    score_function,
    expert_bias,
    routing_map_format,
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
        _routing_map_fake(logits, routing_map_format),
        logits.new_empty(logits.shape, dtype=torch.float32),
    )


@torch.library.custom_op("te_moe::router_topk_indices_fwd", mutates_args=("topk_indices",))
def _topk_indices_forward(
    logits: torch.Tensor,
    topk: int,
    use_pre_softmax: bool,
    num_groups: Optional[int],
    group_topk: Optional[int],
    scaling_factor: Optional[float],
    score_function: str,
    expert_bias: Optional[torch.Tensor],
    routing_map_format: int,
    topk_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.cuda.device(logits.device):
        probs, _, intermediate_output = tex.fused_topk_with_score_function_fwd(
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
    return probs, intermediate_output


@_topk_indices_forward.register_fake
def _topk_indices_forward_fake(
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
):
    del topk, use_pre_softmax, num_groups, group_topk, scaling_factor
    del score_function, expert_bias, routing_map_format, topk_indices
    return torch.empty_like(logits), torch.empty_like(logits, dtype=torch.float32)


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


@torch.library.custom_op("te_moe::router_qb_map_fwd", mutates_args=("histogram",))
def _qb_map_forward(
    logits: torch.Tensor,
    topk: int,
    scaling_factor: Optional[float],
    expert_bias: torch.Tensor,
    routing_map_format: int,
    histogram: torch.Tensor,
    bin_bounds: torch.Tensor,
    histogram_mode: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.cuda.device(logits.device):
        probs, routing_output, intermediate_output, _, _ = (
            tex.fused_topk_with_score_function_qb_fwd(
                logits,
                topk,
                scaling_factor,
                expert_bias,
                routing_map_format,
                None,
                histogram,
                bin_bounds,
                histogram_mode,
            )
        )
    return probs, routing_output, intermediate_output


@_qb_map_forward.register_fake
def _qb_map_forward_fake(
    logits,
    topk,
    scaling_factor,
    expert_bias,
    routing_map_format,
    histogram,
    bin_bounds,
    histogram_mode,
):
    del topk, scaling_factor, expert_bias, histogram, bin_bounds, histogram_mode
    return (
        logits.new_empty(logits.shape),
        _routing_map_fake(logits, routing_map_format),
        logits.new_empty(logits.shape, dtype=torch.float32),
    )


@torch.library.custom_op(
    "te_moe::router_qb_indices_fwd", mutates_args=("topk_indices", "histogram")
)
def _qb_indices_forward(
    logits: torch.Tensor,
    topk: int,
    scaling_factor: Optional[float],
    expert_bias: torch.Tensor,
    routing_map_format: int,
    topk_indices: torch.Tensor,
    histogram: torch.Tensor,
    bin_bounds: torch.Tensor,
    histogram_mode: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.cuda.device(logits.device):
        probs, _, intermediate_output, _, _ = tex.fused_topk_with_score_function_qb_fwd(
            logits,
            topk,
            scaling_factor,
            expert_bias,
            routing_map_format,
            topk_indices,
            histogram,
            bin_bounds,
            histogram_mode,
        )
    return probs, intermediate_output


@_qb_indices_forward.register_fake
def _qb_indices_forward_fake(
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
    del topk, scaling_factor, expert_bias, routing_map_format, topk_indices
    del histogram, bin_bounds, histogram_mode
    return torch.empty_like(logits), torch.empty_like(logits, dtype=torch.float32)


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
