# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import torch
from typing import Optional
from transformer_engine.pytorch.router import (
    fused_topk_with_score_function,
    fused_compute_score_for_moe_aux_loss,
    fused_moe_aux_loss,
)
import pytest
from copy import deepcopy

seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


# Pytorch-based group topk
def group_limited_topk(
    scores: torch.Tensor,
    topk: int,
    num_tokens: int,
    num_experts: int,
    num_groups: int,
    group_topk: int,
):
    group_scores = (
        scores.view(num_tokens, num_groups, -1).topk(topk // group_topk, dim=-1)[0].sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=group_topk, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)

    # Mask the experts based on selection groups
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, num_groups, num_experts // num_groups)
        .reshape(num_tokens, -1)
    )

    masked_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
    probs, top_indices = torch.topk(masked_scores, k=topk, dim=-1)

    return probs, top_indices


# Pytorch-based topk softmax/sigmoid
def topk_score_function_pytorch(
    logits: torch.Tensor,
    topk: int,
    use_pre_softmax: bool = False,
    num_groups: Optional[int] = None,
    group_topk: Optional[int] = None,
    scaling_factor: Optional[float] = None,
    score_function: str = "softmax",
    expert_bias: Optional[torch.Tensor] = None,
):
    num_tokens, num_experts = logits.shape

    def compute_topk(scores, topk, num_groups=None, group_topk=None):
        if group_topk:
            return group_limited_topk(
                scores=scores,
                topk=topk,
                num_tokens=num_tokens,
                num_experts=num_experts,
                num_groups=num_groups,
                group_topk=group_topk,
            )
        else:
            return torch.topk(scores, k=topk, dim=1)

    if score_function == "softmax":
        if use_pre_softmax:
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            probs, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        else:
            scores, top_indices = compute_topk(logits, topk, num_groups, group_topk)
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).type_as(logits)
    elif score_function in ("sigmoid", "sqrtsoftplus"):
        if score_function == "sigmoid":
            scores = torch.sigmoid(logits.float()).type_as(logits)
        else:
            scores = torch.nn.functional.softplus(logits.float()).sqrt().type_as(logits)
        if expert_bias is not None:
            scores_for_routing = scores + expert_bias
            _, top_indices = compute_topk(scores_for_routing, topk, num_groups, group_topk)
            scores = torch.gather(scores, dim=1, index=top_indices).type_as(logits)
        else:
            scores, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else scores
    else:
        raise ValueError(f"Invalid score_function: {score_function}")

    if scaling_factor:
        probs = probs * scaling_factor

    topk_masked_gates = torch.zeros_like(logits).scatter(1, top_indices, probs)
    topk_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()

    return topk_masked_gates, topk_map


# Pytorch-based compute routing scores for aux loss
def compute_scores_for_aux_loss_pytorch(
    logits: torch.Tensor, topk: int, score_function: str
) -> torch.Tensor:
    if score_function == "softmax":
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits)
        scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else scores
    elif score_function == "sqrtsoftplus":
        scores = torch.nn.functional.softplus(logits.float()).sqrt().type_as(logits)
        scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else scores
    else:
        raise ValueError(f"Invalid score_function: {score_function}")

    _, top_indices = torch.topk(scores, k=topk, dim=1)
    routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()
    return routing_map, scores


# Pytorch-based aux loss
def aux_loss_pytorch(
    probs: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    total_num_tokens: int,
    topk: int,
    num_experts: int,
    moe_aux_loss_coeff: float,
):
    aggregated_probs_per_expert = probs.sum(dim=0)
    aux_loss = torch.sum(aggregated_probs_per_expert * tokens_per_expert) * (
        num_experts * moe_aux_loss_coeff / (topk * total_num_tokens * total_num_tokens)
    )
    return aux_loss


def run_comparison(
    dtype,
    num_tokens,
    num_experts,
    topk,
    use_pre_softmax,
    num_groups,
    group_topk,
    scaling_factor,
    score_function,
    enable_bias,
):
    # Set some parameters
    if score_function == "sigmoid":
        # Construct the special logits to avoid inf in the sigmoid function
        offset = torch.arange(-num_tokens // 2, num_tokens // 2, dtype=dtype, device="cuda") * 1e-4
        logits = (
            torch.arange(-num_experts // 2, num_experts // 2, device="cuda", dtype=dtype) * 1e-2
        )
        logits = logits.unsqueeze(0).repeat(num_tokens, 1) + offset.unsqueeze(1)
    else:
        logits = (
            torch.arange(
                -num_tokens * num_experts // 2,
                num_tokens * num_experts // 2,
                device="cuda",
                dtype=dtype,
            )
            * 1e-4
        )
        logits = logits.view(num_tokens, num_experts)
    logits.requires_grad = True
    if enable_bias and score_function in ("sigmoid", "sqrtsoftplus"):
        expert_bias = torch.arange(num_experts, device="cuda", dtype=dtype) * 0.1
        expert_bias = torch.flip(expert_bias, dims=[0])
        expert_bias.requires_grad = True
    else:
        expert_bias = None

    # Clone the input tensor
    logits_clone = deepcopy(logits)
    logits_clone.requires_grad = True
    if expert_bias is not None:
        expert_bias_clone = deepcopy(expert_bias)
        expert_bias_clone.requires_grad = True
    else:
        expert_bias_clone = None

    # Run the original implementation
    # We do not support the capacity factor case
    probs, routing_map = topk_score_function_pytorch(
        logits=logits,
        topk=topk,
        use_pre_softmax=use_pre_softmax,
        num_groups=num_groups,
        group_topk=group_topk,
        scaling_factor=scaling_factor,
        score_function=score_function,
        expert_bias=expert_bias,
    )

    # Run the fused implementation
    probs_fused, routing_map_fused = fused_topk_with_score_function(
        logits=logits_clone,
        topk=topk,
        use_pre_softmax=use_pre_softmax,
        num_groups=num_groups,
        group_topk=group_topk,
        scaling_factor=scaling_factor,
        score_function=score_function,
        expert_bias=expert_bias_clone,
    )

    torch.testing.assert_close(probs, probs_fused)
    torch.testing.assert_close(routing_map, routing_map_fused)

    # Fake the loss
    loss = torch.sum(probs)
    loss_fused = torch.sum(probs_fused)

    # Backward the loss
    loss.backward()
    loss_fused.backward()

    # Check the gradient
    torch.testing.assert_close(logits.grad, logits_clone.grad)


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("num_tokens", [2048, 7168, 8992])
@pytest.mark.parametrize("num_experts", [128, 32])
@pytest.mark.parametrize("topk", [4, 8])
@pytest.mark.parametrize("group_topk", [None, 4])
@pytest.mark.parametrize("scaling_factor", [None, 1.2])
@pytest.mark.parametrize("enable_bias", [True, False])
def test_topk_sigmoid(
    dtype,
    num_tokens,
    num_experts,
    topk,
    group_topk,
    scaling_factor,
    enable_bias,
):
    num_groups = 8 if group_topk else None
    run_comparison(
        dtype=dtype,
        num_tokens=num_tokens,
        num_experts=num_experts,
        topk=topk,
        use_pre_softmax=False,
        num_groups=num_groups,
        group_topk=group_topk,
        scaling_factor=scaling_factor,
        score_function="sigmoid",
        enable_bias=enable_bias,
    )


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("num_tokens", [2048, 7168, 8992])
@pytest.mark.parametrize("num_experts", [128, 32])
@pytest.mark.parametrize("topk", [4, 8])
@pytest.mark.parametrize("group_topk", [None, 4])
@pytest.mark.parametrize("scaling_factor", [None, 1.2])
@pytest.mark.parametrize("enable_bias", [True, False])
def test_topk_sqrtsoftplus(
    dtype,
    num_tokens,
    num_experts,
    topk,
    group_topk,
    scaling_factor,
    enable_bias,
):
    num_groups = 8 if group_topk else None
    run_comparison(
        dtype=dtype,
        num_tokens=num_tokens,
        num_experts=num_experts,
        topk=topk,
        use_pre_softmax=False,
        num_groups=num_groups,
        group_topk=group_topk,
        scaling_factor=scaling_factor,
        score_function="sqrtsoftplus",
        enable_bias=enable_bias,
    )


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("num_tokens", [2048, 7168, 14234])
@pytest.mark.parametrize("num_experts", [128, 32])
@pytest.mark.parametrize("topk", [4, 8])
@pytest.mark.parametrize("use_pre_softmax", [True, False])
@pytest.mark.parametrize("group_topk", [None, 4])
@pytest.mark.parametrize("scaling_factor", [None, 1.2])
def test_topk_softmax(
    dtype,
    num_tokens,
    num_experts,
    topk,
    use_pre_softmax,
    group_topk,
    scaling_factor,
):
    num_groups = 8 if group_topk else None
    run_comparison(
        dtype=dtype,
        num_tokens=num_tokens,
        num_experts=num_experts,
        topk=topk,
        use_pre_softmax=use_pre_softmax,
        num_groups=num_groups,
        group_topk=group_topk,
        scaling_factor=scaling_factor,
        score_function="softmax",
        enable_bias=False,
    )


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("num_tokens", [2048, 7168, 14234])
@pytest.mark.parametrize("num_experts", [256, 128, 32])
@pytest.mark.parametrize("topk", [4, 8])
@pytest.mark.parametrize("score_function", ["softmax", "sigmoid", "sqrtsoftplus"])
def test_fused_scores_for_aux_loss(dtype, num_tokens, num_experts, topk, score_function):
    if score_function == "sigmoid":
        # Construct the special logits to avoid inf in the sigmoid function
        offset = torch.arange(-num_tokens // 2, num_tokens // 2, dtype=dtype, device="cuda") * 1e-4
        logits = (
            torch.arange(-num_experts // 2, num_experts // 2, device="cuda", dtype=dtype) * 1e-2
        )
        logits = logits.unsqueeze(0).repeat(num_tokens, 1) + offset.unsqueeze(1)
    else:
        logits = (
            torch.arange(
                -num_tokens * num_experts // 2,
                num_tokens * num_experts // 2,
                device="cuda",
                dtype=dtype,
            )
            * 1e-4
        )
        logits = logits.view(num_tokens, num_experts)
    logits.requires_grad = True

    logits_clone = deepcopy(logits)
    logits_clone.requires_grad = True

    routing_map, scores = compute_scores_for_aux_loss_pytorch(
        logits=logits,
        topk=topk,
        score_function=score_function,
    )

    routing_map_fused, scores_fused = fused_compute_score_for_moe_aux_loss(
        logits=logits_clone,
        topk=topk,
        score_function=score_function,
    )

    torch.testing.assert_close(scores, scores_fused)
    torch.testing.assert_close(routing_map, routing_map_fused)

    loss = torch.sum(scores)
    loss.backward()
    loss_fused = torch.sum(scores_fused)
    loss_fused.backward()

    torch.testing.assert_close(logits.grad, logits_clone.grad)


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("num_tokens", [2048, 7168, 14234])
@pytest.mark.parametrize("num_experts", [256, 128, 32])
@pytest.mark.parametrize("topk", [4])
def test_fused_moe_aux_loss(dtype, num_tokens, num_experts, topk):
    # Construct the special probs to avoid inf in the sigmoid function
    offset = torch.arange(-num_tokens // 2, num_tokens // 2, dtype=dtype, device="cuda") * 1e-4
    probs = torch.arange(-num_experts // 2, num_experts // 2, device="cuda", dtype=dtype) * 1e-2
    probs = probs.unsqueeze(0).repeat(num_tokens, 1) + offset.unsqueeze(1)
    probs = probs.view(num_tokens, num_experts)
    probs.requires_grad = True

    tokens_per_expert = torch.randint(1, 1000, (num_experts,), device="cuda", dtype=torch.int32)
    coeff = 0.01

    probs_clone = deepcopy(probs)
    probs_clone.requires_grad = True

    aux_loss = aux_loss_pytorch(
        probs=probs,
        tokens_per_expert=tokens_per_expert,
        total_num_tokens=num_tokens,
        topk=topk,
        num_experts=num_experts,
        moe_aux_loss_coeff=coeff,
    )

    aux_loss_fused = fused_moe_aux_loss(
        probs=probs_clone,
        tokens_per_expert=tokens_per_expert,
        total_num_tokens=num_tokens,
        num_experts=num_experts,
        topk=topk,
        coeff=coeff,
    )

    torch.testing.assert_close(aux_loss, aux_loss_fused)

    # Backward
    aux_loss.backward()
    aux_loss_fused.backward()

    torch.testing.assert_close(probs.grad, probs_clone.grad)


def profile_topk_softmax(
    dtype,
    num_tokens,
    num_experts,
    topk,
    enable_bias,
    use_pre_softmax,
):
    group_topk = 4
    scaling_factor = 1.2
    test_topk_sigmoid(
        torch.float32, num_tokens, num_experts, topk, group_topk, scaling_factor, enable_bias
    )
    test_topk_softmax(
        torch.float32, num_tokens, num_experts, topk, use_pre_softmax, group_topk, scaling_factor
    )
    test_topk_sqrtsoftplus(
        torch.float32, num_tokens, num_experts, topk, group_topk, scaling_factor, enable_bias
    )
