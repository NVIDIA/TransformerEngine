# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""
Fused functions used in the MoE router
"""
import torch
import transformer_engine_torch as tex


class FusedTopkScoreFunction(torch.autograd.Function):
    """
    Fused Topk with Score Function router.
    Currently, only support softmax and sigmoid.
    """

    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        topk: int,
        use_pre_softmax: bool,
        num_groups: int,
        group_topk: int,
        scaling_factor: float,
        score_function: str,
        expert_bias: torch.Tensor,
    ):
        # pylint: disable=missing-function-docstring
        probs, routing_map, intermediate_output = tex.fused_topk_with_score_function_fwd(
            logits,
            topk,
            use_pre_softmax,
            num_groups,
            group_topk,
            scaling_factor,
            score_function,
            expert_bias,
        )
        ctx.save_for_backward(routing_map, intermediate_output)
        ctx.num_tokens = logits.size(0)
        ctx.num_experts = logits.size(1)
        ctx.use_pre_softmax = use_pre_softmax
        ctx.topk = topk
        ctx.scaling_factor = scaling_factor
        ctx.score_function = score_function
        return probs, routing_map

    @staticmethod
    def backward(ctx, grad_probs, _):
        # pylint: disable=missing-function-docstring
        routing_map, intermediate_output = ctx.saved_tensors
        grad_logits = tex.fused_topk_with_score_function_bwd(
            ctx.num_tokens,
            ctx.num_experts,
            routing_map,
            intermediate_output,
            grad_probs.contiguous(),
            ctx.topk,
            ctx.use_pre_softmax,
            ctx.scaling_factor,
            ctx.score_function,
        )
        return grad_logits, None, None, None, None, None, None, None


def fused_topk_with_score_function(
    logits: torch.Tensor,
    topk: int,
    use_pre_softmax: bool,
    num_groups: int,
    group_topk: int,
    scaling_factor: float,
    score_function: str,
    expert_bias: torch.Tensor,
):
    """
    Fused topk with score function router.
    Parameters
    ----------
    logits: torch.Tensor
    topk: int
    use_pre_softmax: bool
        if enabled, the computation order: softmax -> topk
    num_groups: int
        used in the group topk
    group_topk: int
        used in the group topk
    scaling_factor: float
    score_function: str
        currently only support softmax and sigmoid
    expert_bias: torch.Tensor
        could be used in the sigmoid

    Returns
    -------
    probs: torch.Tensor
    routing_map: torch.Tensor
    """
    if logits.dtype == torch.float64:
        raise ValueError("Current TE does not support float64 router type")
    return FusedTopkScoreFunction.apply(
        logits,
        topk,
        use_pre_softmax,
        num_groups,
        group_topk,
        scaling_factor,
        score_function,
        expert_bias,
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
    ):
        # pylint: disable=missing-function-docstring
        scores, routing_map, intermediate_output = tex.fused_score_for_moe_aux_loss_fwd(
            logits=logits,
            topk=topk,
            score_function=score_function,
        )
        ctx.save_for_backward(intermediate_output)
        ctx.topk = topk
        ctx.score_function = score_function
        ctx.num_tokens = logits.size(0)
        ctx.num_experts = logits.size(1)
        return routing_map, scores

    @staticmethod
    def backward(ctx, _, grad_scores):
        # pylint: disable=missing-function-docstring
        intermediate_output = ctx.saved_tensors[0]
        grad_logits = tex.fused_score_for_moe_aux_loss_bwd(
            num_tokens=ctx.num_tokens,
            num_experts=ctx.num_experts,
            intermediate_output=intermediate_output,
            grad_scores=grad_scores.contiguous(),
            topk=ctx.topk,
            score_function=ctx.score_function,
        )
        return grad_logits, None, None


def fused_compute_score_for_moe_aux_loss(
    logits: torch.Tensor,
    topk: int,
    score_function: str,
):
    """
    Fused compute scores for MoE aux loss, subset of the fused_topk_with_score_function.
    Parameters
    ----------
    logits: torch.Tensor
    topk: int
    score_function: str
        currently only support softmax and sigmoid

    Returns
    -------
    routing_map: torch.Tensor
    scores: torch.Tensor
    """
    return FusedComputeScoresForMoEAuxLoss.apply(logits, topk, score_function)


class FusedAuxLoss(torch.autograd.Function):
    """
    Fused MoE aux loss.
    """

    @staticmethod
    def forward(
        ctx,
        probs: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        total_num_tokens: int,
        num_experts: int,
        topk: int,
        coeff: float,
    ):
        # pylint: disable=missing-function-docstring
        num_tokens = probs.size(0)
        aux_loss, Const_buf = tex.fused_moe_aux_loss_fwd(
            probs=probs,
            tokens_per_expert=tokens_per_expert,
            total_num_tokens=total_num_tokens,
            num_tokens=num_tokens,
            num_experts=num_experts,
            topk=topk,
            coeff=coeff,
        )
        ctx.save_for_backward(Const_buf, tokens_per_expert)
        ctx.num_tokens = num_tokens
        ctx.num_experts = num_experts
        return aux_loss

    @staticmethod
    def backward(ctx, grad_aux_loss):
        # pylint: disable=missing-function-docstring
        Const_buf, tokens_per_expert = ctx.saved_tensors
        grad_probs = tex.fused_moe_aux_loss_bwd(
            Const_buf=Const_buf,
            tokens_per_expert=tokens_per_expert,
            num_tokens=ctx.num_tokens,
            num_experts=ctx.num_experts,
            grad_aux_loss=grad_aux_loss,
        )
        return grad_probs, None, None, None, None, None


def fused_moe_aux_loss(
    probs: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    total_num_tokens: int,
    num_experts: int,
    topk: int,
    coeff: float,
):
    """
    Fused MoE aux loss.
    Parameters
    ----------
    probs: torch.Tensor
    tokens_per_expert: torch.Tensor
        the number of tokens per expert
    total_num_tokens: int
        the total number of tokens, involved in the aux loss calculation
    num_experts: int
    topk: int
    coeff: float
        the coefficient of the aux loss

    Returns
    -------
    aux_loss: torch.scalar
    """
    return FusedAuxLoss.apply(probs, tokens_per_expert, total_num_tokens, num_experts, topk, coeff)
