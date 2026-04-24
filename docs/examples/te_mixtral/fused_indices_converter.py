# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Triton kernel for converting sparse MoE indices to multihot mask format."""

import math
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import torch


def _identity_decorator(fn):
    """Return the decorated callable unchanged (no-op decorator fallback)."""
    return fn


null_decorator = _identity_decorator

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    triton.autotune = null_decorator
    triton.heuristics = null_decorator
    tl = MagicMock()


if TYPE_CHECKING:
    import triton
    import triton.language as tl


@triton.jit
def _indices_to_multihot_kernel(
    indices_ptr,
    probs_in_indices_ptr,
    multihot_indices_ptr,
    probs_in_multihot_ptr,
    position_map_ptr,
    num_of_local_experts: tl.constexpr,
    num_of_local_experts_next_power_of_2: tl.constexpr,
    topk: tl.constexpr,
    topk_next_power_of_2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # noqa: N803
):
    """Convert sparse [N, topk] indices to dense [N, num_local_experts] multihot."""
    topk_row = tl.arange(0, topk_next_power_of_2)
    topk_row = tl.where(topk_row < topk, topk_row, -1)
    topk_row_mask = topk_row != -1
    num_exp_row = tl.arange(0, num_of_local_experts_next_power_of_2)
    num_exp_row = tl.where(num_exp_row < num_of_local_experts, num_exp_row, -1)
    num_exp_row_mask = num_exp_row != -1

    row_idx = tl.program_id(0)
    indices_row = tl.load(indices_ptr + row_idx * topk + topk_row, mask=topk_row_mask)
    indices_row = tl.where(topk_row_mask, indices_row, -1)
    probs_row = tl.load(probs_in_indices_ptr + row_idx * topk + topk_row, mask=topk_row_mask)

    position_row = tl.where(indices_row != -1, topk_row, -1)
    mask = (indices_row != -1) & (indices_row < num_of_local_experts)

    row_idx_offset = row_idx * num_of_local_experts
    tl.store(multihot_indices_ptr + row_idx_offset + num_exp_row, 0, mask=num_exp_row_mask)
    tl.store(probs_in_multihot_ptr + row_idx_offset + num_exp_row, 0, mask=num_exp_row_mask)
    tl.store(position_map_ptr + row_idx_offset + num_exp_row, -1, mask=num_exp_row_mask)
    tl.debug_barrier()
    tl.store(multihot_indices_ptr + row_idx_offset + indices_row, 1, mask)
    tl.store(probs_in_multihot_ptr + row_idx_offset + indices_row, probs_row, mask)
    tl.store(position_map_ptr + row_idx_offset + indices_row, position_row, mask)


@triton.jit
def _multihot_to_indices_kernel(
    probs_in_multihot_ptr,
    position_map_ptr,
    probs_indices_ptr,
    num_of_local_experts: tl.constexpr,
    num_of_local_experts_next_power_of_2: tl.constexpr,
    topk: tl.constexpr,
    topk_next_power_of_2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # noqa: N803
):
    """Convert dense [N, num_local_experts] multihot back to sparse [N, topk] probs."""
    topk_row = tl.arange(0, topk_next_power_of_2)
    topk_row = tl.where(topk_row < topk, topk_row, -1)
    topk_row_mask = topk_row != -1
    num_exp_row = tl.arange(0, num_of_local_experts_next_power_of_2)
    num_exp_row = tl.where(num_exp_row < num_of_local_experts, num_exp_row, -1)
    num_exp_row_mask = num_exp_row != -1

    row_idx = tl.program_id(0)
    ptr_offset = row_idx * num_of_local_experts + num_exp_row
    probs_in_multihot_row = tl.load(probs_in_multihot_ptr + ptr_offset, mask=num_exp_row_mask)

    position_map_row = tl.load(position_map_ptr + ptr_offset, mask=num_exp_row_mask)
    position_map_row = tl.where(num_exp_row_mask, position_map_row, -1)
    mask = position_map_row != -1

    tl.store(probs_indices_ptr + row_idx * topk + topk_row, 0, mask=topk_row_mask)
    tl.debug_barrier()
    tl.store(
        probs_indices_ptr + row_idx * topk + position_map_row,
        probs_in_multihot_row,
        mask,
    )


class IndicesToMultihot(torch.autograd.Function):
    """Convert MoE topk indices to multihot representation (differentiable)."""

    @staticmethod
    def forward(ctx, indices, probs_indices, num_of_local_experts):
        """Forward: sparse [N, topk] -> dense [N, num_local_experts] multihot + probs."""
        assert HAVE_TRITON, "Triton is not installed"
        num_of_tokens = indices.shape[0]
        assert indices.shape == probs_indices.shape
        topk = indices.shape[1]
        device = indices.device
        multihot_indices = torch.empty(
            (num_of_tokens, num_of_local_experts), dtype=torch.bool, device=device
        )
        probs_in_multihot = torch.empty(
            (num_of_tokens, num_of_local_experts),
            dtype=probs_indices.dtype,
            device=device,
        )
        position_map = torch.empty(
            (num_of_tokens, num_of_local_experts), dtype=torch.int32, device=device
        )
        topk_next_power_of_2 = 2 ** math.ceil(math.log2(topk))
        num_of_local_experts_next_power_of_2 = 2 ** math.ceil(math.log2(num_of_local_experts))
        grid = (num_of_tokens,)
        _indices_to_multihot_kernel[grid](
            indices,
            probs_indices,
            multihot_indices,
            probs_in_multihot,
            position_map,
            num_of_local_experts,
            num_of_local_experts_next_power_of_2,
            topk,
            topk_next_power_of_2,
            BLOCK_SIZE=32,
            num_warps=1,
        )

        ctx.save_for_backward(position_map)
        ctx.num_of_tokens = num_of_tokens
        ctx.num_of_local_experts = num_of_local_experts
        ctx.topk = topk
        return multihot_indices, probs_in_multihot

    @staticmethod
    def backward(ctx, grad_multihot_indices, grad_probs_in_multihot):
        """Backward: dense [N, num_local_experts] -> sparse [N, topk] grad probs."""
        position_map = ctx.saved_tensors[0]
        num_of_tokens = ctx.num_of_tokens
        num_of_local_experts = ctx.num_of_local_experts
        topk = ctx.topk

        grad_probs_indices = torch.empty(
            (num_of_tokens, topk),
            dtype=grad_probs_in_multihot.dtype,
            device=grad_probs_in_multihot.device,
        )
        topk_next_power_of_2 = 2 ** math.ceil(math.log2(topk))
        num_of_local_experts_next_power_of_2 = 2 ** math.ceil(math.log2(num_of_local_experts))

        grid = (num_of_tokens,)
        _multihot_to_indices_kernel[grid](
            grad_probs_in_multihot.contiguous(),
            position_map,
            grad_probs_indices,
            num_of_local_experts,
            num_of_local_experts_next_power_of_2,
            topk,
            topk_next_power_of_2,
            BLOCK_SIZE=32,
            num_warps=1,
        )
        return None, grad_probs_indices, None


def fused_indices_to_multihot(indices, probs_indices, num_of_local_experts):
    """Convert MoE topk indices to multihot representation."""
    return IndicesToMultihot.apply(indices, probs_indices, num_of_local_experts)
