# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch routing backend for a single expert-parallel rank.

``MoeDispatch`` and ``MoeCombine`` use this backend when they are constructed
without a communication buffer, i.e. when the expert-parallel group holds a
single rank and every expert is local. Dispatch sorts local routes into
expert-major order and combine scatters expert outputs back to their source
tokens, so neither operation moves data between ranks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...ep import EpConfig


@dataclass(slots=True)
class RoutingPlan:
    """Expert-major route order with the routing metadata that dispatch outputs."""

    token_index: torch.Tensor  # int64 [R]: source token of each recv row
    route_index: torch.Tensor  # int64 [R]: flat (token, slot) route of each recv row
    weight: torch.Tensor  # float32 [R]: routing weight of each recv row
    tokens_per_expert: torch.Tensor  # int64 [num_local_experts]
    num_tokens: int
    topk_shape: torch.Size


def validate_local_config(config: EpConfig) -> None:
    """Reject communication settings that the single-rank backend cannot honor."""
    if config.ep_group is not None:
        raise ValueError("PyTorch EP=1 requires ep_group=None.")
    if config.dispatch_fwd_quant_recipe is not None or config.combine_bwd_quant_recipe is not None:
        raise NotImplementedError("PyTorch EP=1 does not support quantized communication.")
    if config.recv_capacity_per_rank is not None or config.drop_on_overflow:
        raise NotImplementedError(
            "PyTorch EP=1 does not support receive capacity or overflow policy."
        )
    if config.alignment or config.payload_dtype is not torch.bfloat16:
        raise NotImplementedError("PyTorch EP=1 requires unaligned BF16 payloads.")


def _expert_major_routes(topk_idx: torch.Tensor, config: EpConfig) -> torch.Tensor:
    """Validate routing indices and return the flat route of every recv row.

    Routes are ordered by local expert id and then by source token. An expert id
    of -1 drops the route, as in the expert-parallel reference implementation.
    """
    if topk_idx.dtype is not torch.int64:
        raise TypeError(f"topk_idx must be int64, got {topk_idx.dtype}.")
    if topk_idx.ndim != 2 or topk_idx.shape[-1] != config.top_k:
        raise ValueError(
            f"topk_idx must have shape (T, {config.top_k}), got {tuple(topk_idx.shape)}."
        )
    flat_expert = topk_idx.reshape(-1)
    if bool((flat_expert < -1).any()):
        raise ValueError("topk_idx must not contain expert ids below -1.")
    local = flat_expert >= 0
    remote = flat_expert >= config.num_local_experts
    if bool(remote.any()):
        raise ValueError(
            f"topk_idx routes to expert {int(flat_expert[remote][0])}, which is not one of "
            f"the {config.num_local_experts} local experts: this backend holds every expert "
            "of a single-rank expert-parallel group. Pass an EpBuffer to use NCCL EP."
        )
    order = torch.argsort(flat_expert[local], stable=True)
    return torch.nonzero(local, as_tuple=False).flatten().index_select(0, order)


def make_routing_plan(
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    config: EpConfig,
) -> RoutingPlan:
    """Validate dispatch routing metadata and build the expert-major route order."""
    route_index = _expert_major_routes(topk_idx, config)
    if topk_weights.device != topk_idx.device:
        raise ValueError(f"topk_weights must be on {topk_idx.device}, got {topk_weights.device}.")
    if topk_weights.shape != topk_idx.shape:
        raise ValueError(
            f"topk_weights must have shape {tuple(topk_idx.shape)}, "
            f"got {tuple(topk_weights.shape)}."
        )
    if topk_weights.dtype is not torch.float32:
        raise TypeError(f"topk_weights must be float32, got {topk_weights.dtype}.")
    return RoutingPlan(
        token_index=torch.div(route_index, config.top_k, rounding_mode="floor"),
        route_index=route_index,
        weight=topk_weights.reshape(-1).index_select(0, route_index),
        tokens_per_expert=torch.bincount(
            topk_idx.reshape(-1).index_select(0, route_index),
            minlength=config.num_local_experts,
        ),
        num_tokens=topk_idx.shape[0],
        topk_shape=topk_idx.shape,
    )


def make_token_order(topk_idx: torch.Tensor, config: EpConfig) -> torch.Tensor:
    """Validate combine routing indices and return the source token of every recv row."""
    route_index = _expert_major_routes(topk_idx, config)
    return torch.div(route_index, config.top_k, rounding_mode="floor")


def dispatch_forward(
    tokens: torch.Tensor,
    plan: RoutingPlan,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather tokens into expert-major order with their routing weights."""
    return tokens.index_select(0, plan.token_index), plan.weight


def dispatch_backward(
    plan: RoutingPlan,
    grad_recv_tokens: torch.Tensor,
    grad_recv_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter recv-row gradients back to their source tokens and routes."""
    grad_tokens = torch.zeros(
        plan.num_tokens,
        grad_recv_tokens.shape[-1],
        dtype=grad_recv_tokens.dtype,
        device=grad_recv_tokens.device,
    )
    grad_tokens.index_add_(0, plan.token_index, grad_recv_tokens)
    grad_topk_weights = torch.zeros(
        plan.topk_shape,
        dtype=torch.float32,
        device=grad_recv_weights.device,
    )
    grad_topk_weights.view(-1).index_add_(0, plan.route_index, grad_recv_weights)
    return grad_tokens, grad_topk_weights


def combine_forward(
    expert_out: torch.Tensor,
    token_index: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Sum expert outputs into their source tokens in floating-point."""
    if expert_out.shape[0] != token_index.numel():
        raise ValueError(
            f"MoeCombine input has {expert_out.shape[0]} rows, but the routing "
            f"metadata describes {token_index.numel()}."
        )
    output = torch.zeros(
        num_tokens,
        expert_out.shape[-1],
        dtype=torch.float32,
        device=expert_out.device,
    )
    output.index_add_(0, token_index, expert_out.float())
    return output.to(expert_out.dtype)


def combine_backward(token_index: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    """Gather token gradients into the expert-major order of the combine input."""
    return grad_output.index_select(0, token_index)
