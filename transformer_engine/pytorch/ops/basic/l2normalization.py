# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusable operation for L2 Normalization."""

from __future__ import annotations
from typing import Optional
import os

import torch

from ...utils import clear_tensor_data
from ... import torch_version
from .._common import maybe_dequantize
from ..op import BasicOperation, OperationContext
from ...jit import (
    l2normalization_fused,
    l2normalization_fwd_fused,
    l2normalization_backward_fused,
    set_jit_fusion_options,
    warmup_jit_l2normalization_all_dtypes,
)
from ...tensor import Quantizer


class L2Normalization(BasicOperation):
    r"""L2 Normalization

    Applies L2 normalization over the last dimension of input tensors.
    This is a parameter-free normalization that scales each vector to unit L2 norm.

    .. math::
        y = \frac{x}{\sqrt{\sum_{i} x_i^2 + \varepsilon}}

    This operation is used e.g. for query-key normalization in attention mechanisms.

    Parameters
    ----------
    eps : float, default = 1e-6
        A value added to the denominator for numerical stability
    seq_length: int, default = None
        sequence length of input samples. Needed for JIT Warmup, a technique where jit fused
        functions are warmed up before training to ensure same kernels are used for forward
        propagation and activation recompute phase.
    micro_batch_size: int, default = None
        batch size per training step. Needed for JIT Warmup, a technique where jit
        fused functions are warmed up before training to ensure same kernels are
        used for forward propagation and activation recompute phase.

    """

    def __init__(
        self,
        *,
        eps: float = 1e-6,
        seq_length: Optional[int] = None,
        micro_batch_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.eps: float = eps

        # JIT warmup for L2Normalization fused operations
        if seq_length and micro_batch_size:
            if (
                torch.cuda.is_available()
                and torch_version() >= (2, 0, 0)
                and bool(int(os.getenv("NVTE_TORCH_COMPILE", "1")))
            ):
                set_jit_fusion_options()
                # For L2Normalization, we don't know the hidden size until forward pass,
                # but we can warm up with common sizes. For QK normalization, this will be
                # the attention head dimension (hidden_size_per_attention_head), not the full
                # model hidden dimension. Common head dimensions are 32, 64, 80, 96, 128, 256.
                common_hidden_sizes = [32, 64, 80, 96, 128, 256]
                for hidden_size in common_hidden_sizes:
                    warmup_jit_l2normalization_all_dtypes(hidden_size, seq_length, micro_batch_size)

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
    ) -> torch.Tensor:
        # Use input directly - torch.compile can handle multi-dimensional tensors
        x = maybe_dequantize(input_)

        # Check if backward pass is needed
        requires_grad = ctx.requires_grad

        # Compute L2 normalization using fused implementation
        # L2 norm: x / sqrt(sum(x^2) + eps) = x * rsqrt(sum(x^2) + eps)
        if requires_grad:
            # Training: use version that returns output and intermediate values for backward pass
            y, rsqrt_norm = l2normalization_fwd_fused(x, self.eps)
        else:
            # Inference: use lightweight version that only returns output
            y = l2normalization_fused(x, self.eps)
            rsqrt_norm = None  # Not needed for inference

        # Save state for backward pass
        if requires_grad:
            ctx.save_for_backward(x, rsqrt_norm)

        return y

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:

        # Saved tensors from forward pass
        x, rsqrt_norm = ctx.saved_tensors

        dy = maybe_dequantize(grad_output)

        # Compute L2 norm backward pass using fused implementation - recalculates l2_norm_squared_eps
        dx = l2normalization_backward_fused(dy, x, rsqrt_norm, self.eps)

        # Clear saved tensors if possible
        clear_tensor_data(x)
        clear_tensor_data(rsqrt_norm)

        # No parameters, so empty tuple for param grads
        return dx, ()
