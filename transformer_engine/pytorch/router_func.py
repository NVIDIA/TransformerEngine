import torch
import transformer_engine_torch as tex


class FusedTopkSoftmaxSigmoid(torch.autograd.Function):
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
        probs, routing_map, intermediate_output = tex.fused_topk_softmax_sigmod_fwd(
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
    def backward(ctx, grad_probs, grad_routing_map):
        routing_map, intermediate_output = ctx.saved_tensors
        grad_logits = tex.fused_topk_softmax_sigmod_bwd(
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


def fused_topk_softmax_sigmoid(
    logits: torch.Tensor,
    topk: int,
    use_pre_softmax: bool,
    num_groups: int,
    group_topk: int,
    scaling_factor: float,
    score_function: str,
    expert_bias: torch.Tensor,
):
    if logits.dtype != torch.float32:
        raise ValueError(
            "Current TE only support float32 router type, but got {}".format(logits.dtype)
        )
    return FusedTopkSoftmaxSigmoid.apply(
        logits,
        topk,
        use_pre_softmax,
        num_groups,
        group_topk,
        scaling_factor,
        score_function,
        expert_bias,
    )


class FusedComputeScoresForAuxLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        topk: int,
        score_function: str,
    ):
        scores, routing_map, intermediate_output = tex.fused_scores_for_aux_loss_fwd(
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
    def backward(ctx, grad_routing_map, grad_scores):
        intermediate_output = ctx.saved_tensors[0]
        grad_logits = tex.fused_scores_for_aux_loss_bwd(
            num_tokens=ctx.num_tokens,
            num_experts=ctx.num_experts,
            intermediate_output=intermediate_output,
            grad_scores=grad_scores.contiguous(),
            topk=ctx.topk,
            score_function=ctx.score_function,
        )
        return grad_logits, None, None


def fused_compute_scores_for_aux_loss(
    logits: torch.Tensor,
    topk: int,
    score_function: str,
):
    return FusedComputeScoresForAuxLoss.apply(logits, topk, score_function)


class FusedAuxLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        probs: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        num_tokens: int,
        num_experts: int,
        topk: int,
        coeff: float,
    ):
        aux_loss, Const_buf = tex.fused_aux_loss_fwd(
            probs=probs,
            tokens_per_expert=tokens_per_expert,
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
        Const_buf, tokens_per_expert = ctx.saved_tensors
        grad_probs = tex.fused_aux_loss_bwd(
            Const_buf=Const_buf,
            tokens_per_expert=tokens_per_expert,
            num_tokens=ctx.num_tokens,
            num_experts=ctx.num_experts,
            grad_aux_loss=grad_aux_loss,
        )
        return grad_probs, None, None, None, None, None


def fused_aux_loss(
    probs: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    num_tokens: int,
    num_experts: int,
    topk: int,
    coeff: float,
):
    return FusedAuxLoss.apply(probs, tokens_per_expert, num_tokens, num_experts, topk, coeff)
