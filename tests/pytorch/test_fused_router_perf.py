# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
from typing import Callable, Optional, Tuple

import pytest
import torch

from transformer_engine.pytorch.router import (
    fused_compute_score_for_moe_aux_loss,
    fused_moe_aux_loss,
    fused_topk_with_score_function,
)


seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


def _require_perf_env() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    if os.getenv("TE_RUN_PERF_TESTS", "0") != "1":
        pytest.skip("Set TE_RUN_PERF_TESTS=1 to enable router performance tests.")


def _benchmark_cuda_kernel(fn: Callable[[], object], warmup: int = 20, iters: int = 100) -> float:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_event.record()
    for _ in range(iters):
        fn()
    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event) / iters


def group_limited_topk(
    scores: torch.Tensor,
    topk: int,
    num_tokens: int,
    num_experts: int,
    num_groups: int,
    group_topk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    group_scores = (
        scores.view(num_tokens, num_groups, -1).topk(topk // group_topk, dim=-1)[0].sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=group_topk, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)

    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, num_groups, num_experts // num_groups)
        .reshape(num_tokens, -1)
    )
    masked_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
    probs, top_indices = torch.topk(masked_scores, k=topk, dim=-1)
    return probs, top_indices


def topk_softmax_sigmoid_pytorch(
    logits: torch.Tensor,
    topk: int,
    use_pre_softmax: bool = False,
    num_groups: Optional[int] = None,
    group_topk: Optional[int] = None,
    scaling_factor: Optional[float] = None,
    score_function: str = "softmax",
    expert_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens, num_experts = logits.shape

    def compute_topk(scores, topk_value, num_groups_value=None, group_topk_value=None):
        if group_topk_value:
            assert num_groups_value is not None
            return group_limited_topk(
                scores=scores,
                topk=topk_value,
                num_tokens=num_tokens,
                num_experts=num_experts,
                num_groups=num_groups_value,
                group_topk=group_topk_value,
            )
        return torch.topk(scores, k=topk_value, dim=1)

    if score_function == "softmax":
        if use_pre_softmax:
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            probs, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        else:
            scores, top_indices = compute_topk(logits, topk, num_groups, group_topk)
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).type_as(logits)
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits.float()).type_as(logits)
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


def compute_scores_for_aux_loss_pytorch(
    logits: torch.Tensor, topk: int, score_function: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    if score_function == "softmax":
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits)
        scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else scores
    else:
        raise ValueError(f"Invalid score_function: {score_function}")

    _, top_indices = torch.topk(scores, k=topk, dim=1)
    routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()
    return routing_map, scores


def aux_loss_pytorch(
    probs: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    total_num_tokens: int,
    topk: int,
    num_experts: int,
    moe_aux_loss_coeff: float,
) -> torch.Tensor:
    aggregated_probs_per_expert = probs.sum(dim=0)
    return torch.sum(aggregated_probs_per_expert * tokens_per_expert) * (
        num_experts * moe_aux_loss_coeff / (topk * total_num_tokens * total_num_tokens)
    )


def _make_router_logits(
    dtype: torch.dtype, num_tokens: int, num_experts: int, score_function: str
) -> torch.Tensor:
    if score_function == "sigmoid":
        offset = torch.arange(-num_tokens // 2, num_tokens // 2, dtype=dtype, device="cuda") * 1e-4
        logits = (
            torch.arange(-num_experts // 2, num_experts // 2, device="cuda", dtype=dtype) * 1e-2
        )
        return logits.unsqueeze(0).repeat(num_tokens, 1) + offset.unsqueeze(1)

    logits = (
        torch.arange(
            -num_tokens * num_experts // 2,
            num_tokens * num_experts // 2,
            device="cuda",
            dtype=dtype,
        )
        * 1e-4
    )
    return logits.view(num_tokens, num_experts)


def _make_router_bias(num_experts: int) -> torch.Tensor:
    bias = torch.arange(num_experts, device="cuda", dtype=torch.float32) * 0.1
    return torch.flip(bias, dims=[0])


def _print_perf_result(case_name: str, torch_ms: float, fused_ms: float) -> None:
    speedup = torch_ms / fused_ms
    print(f"{case_name}: torch={torch_ms:.6f} ms, fused={fused_ms:.6f} ms, speedup={speedup:.4f}x")


def _perf_assert_message(case_name: str, torch_ms: float, fused_ms: float) -> str:
    return (
        f"{case_name} perf result: torch={torch_ms:.6f} ms, "
        f"fused={fused_ms:.6f} ms, speedup={torch_ms / fused_ms:.4f}x"
    )


@pytest.mark.parametrize(
    "score_function,use_pre_softmax,enable_bias",
    [("softmax", False, False), ("sigmoid", False, True)],
    ids=["softmax", "sigmoid_with_bias"],
)
def test_fused_topk_router_perf_against_torch(
    score_function, use_pre_softmax, enable_bias, record_property
):
    _require_perf_env()

    dtype = torch.float32
    num_tokens = 4096
    num_experts = 192
    topk = 8
    num_groups = 8
    group_topk = 4
    scaling_factor = 1.2

    logits = _make_router_logits(dtype, num_tokens, num_experts, score_function)
    expert_bias = _make_router_bias(num_experts) if enable_bias else None

    torch_probs, torch_map = topk_softmax_sigmoid_pytorch(
        logits=logits,
        topk=topk,
        use_pre_softmax=use_pre_softmax,
        num_groups=num_groups,
        group_topk=group_topk,
        scaling_factor=scaling_factor,
        score_function=score_function,
        expert_bias=expert_bias,
    )
    fused_probs, fused_map = fused_topk_with_score_function(
        logits=logits,
        topk=topk,
        use_pre_softmax=use_pre_softmax,
        num_groups=num_groups,
        group_topk=group_topk,
        scaling_factor=scaling_factor,
        score_function=score_function,
        expert_bias=expert_bias,
    )

    torch_ms = _benchmark_cuda_kernel(
        lambda: topk_softmax_sigmoid_pytorch(
            logits=logits,
            topk=topk,
            use_pre_softmax=use_pre_softmax,
            num_groups=num_groups,
            group_topk=group_topk,
            scaling_factor=scaling_factor,
            score_function=score_function,
            expert_bias=expert_bias,
        )
    )
    fused_ms = _benchmark_cuda_kernel(
        lambda: fused_topk_with_score_function(
            logits=logits,
            topk=topk,
            use_pre_softmax=use_pre_softmax,
            num_groups=num_groups,
            group_topk=group_topk,
            scaling_factor=scaling_factor,
            score_function=score_function,
            expert_bias=expert_bias,
        )
    )

    record_property("torch_ms", round(torch_ms, 6))
    record_property("fused_ms", round(fused_ms, 6))
    record_property("speedup", round(torch_ms / fused_ms, 6))
    _print_perf_result(f"topk_router[{score_function}]", torch_ms, fused_ms)

    assert torch_ms > 0, _perf_assert_message(f"topk_router[{score_function}]", torch_ms, fused_ms)
    assert fused_ms > 0, _perf_assert_message(f"topk_router[{score_function}]", torch_ms, fused_ms)
    torch.testing.assert_close(torch_probs, fused_probs)
    torch.testing.assert_close(torch_map, fused_map)


@pytest.mark.parametrize("score_function", ["softmax", "sigmoid"])
def test_fused_scores_for_aux_loss_perf_against_torch(score_function, record_property):
    _require_perf_env()

    dtype = torch.float32
    num_tokens = 8192
    num_experts = 128
    topk = 8
    logits = _make_router_logits(dtype, num_tokens, num_experts, score_function)

    torch_map, torch_scores = compute_scores_for_aux_loss_pytorch(
        logits=logits,
        topk=topk,
        score_function=score_function,
    )
    fused_map, fused_scores = fused_compute_score_for_moe_aux_loss(
        logits=logits,
        topk=topk,
        score_function=score_function,
    )

    torch_ms = _benchmark_cuda_kernel(
        lambda: compute_scores_for_aux_loss_pytorch(
            logits=logits,
            topk=topk,
            score_function=score_function,
        )
    )
    fused_ms = _benchmark_cuda_kernel(
        lambda: fused_compute_score_for_moe_aux_loss(
            logits=logits,
            topk=topk,
            score_function=score_function,
        )
    )

    record_property("torch_ms", round(torch_ms, 6))
    record_property("fused_ms", round(fused_ms, 6))
    record_property("speedup", round(torch_ms / fused_ms, 6))
    _print_perf_result(f"scores_for_aux_loss[{score_function}]", torch_ms, fused_ms)

    assert torch_ms > 0, _perf_assert_message(
        f"scores_for_aux_loss[{score_function}]", torch_ms, fused_ms
    )
    assert fused_ms > 0, _perf_assert_message(
        f"scores_for_aux_loss[{score_function}]", torch_ms, fused_ms
    )
    torch.testing.assert_close(torch_scores, fused_scores)
    torch.testing.assert_close(torch_map, fused_map)


def test_fused_moe_aux_loss_perf_against_torch(record_property):
    _require_perf_env()

    dtype = torch.float32
    num_tokens = 8192
    num_experts = 128
    topk = 4
    coeff = 0.01

    offset = torch.arange(-num_tokens // 2, num_tokens // 2, dtype=dtype, device="cuda") * 1e-4
    probs = torch.arange(-num_experts // 2, num_experts // 2, device="cuda", dtype=dtype) * 1e-2
    probs = probs.unsqueeze(0).repeat(num_tokens, 1) + offset.unsqueeze(1)
    probs = probs.view(num_tokens, num_experts)
    tokens_per_expert = torch.randint(1, 1000, (num_experts,), device="cuda", dtype=torch.int32)

    torch_loss = aux_loss_pytorch(
        probs=probs,
        tokens_per_expert=tokens_per_expert,
        total_num_tokens=num_tokens,
        topk=topk,
        num_experts=num_experts,
        moe_aux_loss_coeff=coeff,
    )
    fused_loss = fused_moe_aux_loss(
        probs=probs,
        tokens_per_expert=tokens_per_expert,
        total_num_tokens=num_tokens,
        num_experts=num_experts,
        topk=topk,
        coeff=coeff,
    )

    torch_ms = _benchmark_cuda_kernel(
        lambda: aux_loss_pytorch(
            probs=probs,
            tokens_per_expert=tokens_per_expert,
            total_num_tokens=num_tokens,
            topk=topk,
            num_experts=num_experts,
            moe_aux_loss_coeff=coeff,
        )
    )
    fused_ms = _benchmark_cuda_kernel(
        lambda: fused_moe_aux_loss(
            probs=probs,
            tokens_per_expert=tokens_per_expert,
            total_num_tokens=num_tokens,
            num_experts=num_experts,
            topk=topk,
            coeff=coeff,
        )
    )

    record_property("torch_ms", round(torch_ms, 6))
    record_property("fused_ms", round(fused_ms, 6))
    record_property("speedup", round(torch_ms / fused_ms, 6))
    _print_perf_result("moe_aux_loss", torch_ms, fused_ms)

    assert torch_ms > 0, _perf_assert_message("moe_aux_loss", torch_ms, fused_ms)
    assert fused_ms > 0, _perf_assert_message("moe_aux_loss", torch_ms, fused_ms)
    torch.testing.assert_close(torch_loss, fused_loss)
