# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from typing import Optional
import torch
import torch.nn.functional as F

__all__ = [
    "scaled_softmax_forward_torch",
    "scaled_softmax_backward_torch",
    "scaled_masked_softmax_forward_torch",
    "scaled_masked_softmax_backward_torch",
    "scaled_upper_triang_masked_softmax_forward_torch",
    "scaled_upper_triang_masked_softmax_backward_torch",
    "scaled_aligned_causal_masked_softmax_forward_torch",
    "scaled_aligned_causal_masked_softmax_backward_torch",
]


def scaled_softmax_forward_torch(input: torch.Tensor, scale: float) -> torch.Tensor:
    return F.softmax(input * scale, dim=-1)


def scaled_softmax_backward_torch(
    output_grad: torch.Tensor,
    softmax_output: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    # Compute in float32 for numerical stability (matching CUDA behavior)
    orig_dtype = output_grad.dtype
    output_grad_f32 = output_grad.float()
    softmax_output_f32 = softmax_output.float()

    grad_softmax = softmax_output_f32 * (
        output_grad_f32 - (softmax_output_f32 * output_grad_f32).sum(dim=-1, keepdim=True)
    )

    return (grad_softmax * scale).to(orig_dtype)


def scaled_masked_softmax_forward_torch(
    input: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    # Handle uint8 mask (CUDA format: 1=masked, 0=unmasked)
    # Convert to additive mask (-10000 for masked positions, 0 for unmasked)
    if mask.dtype == torch.uint8:
        additive_mask = torch.zeros_like(input, dtype=input.dtype)
        # Expand mask if needed (mask shape: batch, 1, seq_q, seq_k)
        if mask.dim() == 4 and mask.size(1) == 1 and input.dim() == 4:
            mask = mask.expand_as(input)
        additive_mask = additive_mask.masked_fill(mask.bool(), -10000.0)
    else:
        additive_mask = mask

    scaled_input = input * scale + additive_mask

    return F.softmax(scaled_input, dim=-1)


def scaled_masked_softmax_backward_torch(
    output_grad: torch.Tensor,
    softmax_output: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    # Compute in float32 for numerical stability (matching CUDA behavior)
    orig_dtype = output_grad.dtype
    output_grad_f32 = output_grad.float()
    softmax_output_f32 = softmax_output.float()

    grad_softmax = softmax_output_f32 * (
        output_grad_f32 - (softmax_output_f32 * output_grad_f32).sum(dim=-1, keepdim=True)
    )

    return (grad_softmax * scale).to(orig_dtype)


def scaled_upper_triang_masked_softmax_forward_torch(
    input: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    seq_len = input.size(-1)

    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float('-inf'), device=input.device, dtype=input.dtype),
        diagonal=1
    )

    scaled_input = input * scale + causal_mask

    return F.softmax(scaled_input, dim=-1)


def scaled_upper_triang_masked_softmax_backward_torch(
    output_grad: torch.Tensor,
    softmax_output: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    # Compute in float32 for numerical stability (matching CUDA behavior)
    orig_dtype = output_grad.dtype
    output_grad_f32 = output_grad.float()
    softmax_output_f32 = softmax_output.float()

    grad_softmax = softmax_output_f32 * (
        output_grad_f32 - (softmax_output_f32 * output_grad_f32).sum(dim=-1, keepdim=True)
    )

    return (grad_softmax * scale).to(orig_dtype)


def scaled_aligned_causal_masked_softmax_forward_torch(
    input: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return scaled_upper_triang_masked_softmax_forward_torch(input, scale)


def scaled_aligned_causal_masked_softmax_backward_torch(
    output_grad: torch.Tensor,
    softmax_output: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    # Compute in float32 for numerical stability (matching CUDA behavior)
    orig_dtype = output_grad.dtype
    output_grad_f32 = output_grad.float()
    softmax_output_f32 = softmax_output.float()

    grad_softmax = softmax_output_f32 * (
        output_grad_f32 - (softmax_output_f32 * output_grad_f32).sum(dim=-1, keepdim=True)
    )

    return (grad_softmax * scale).to(orig_dtype)
