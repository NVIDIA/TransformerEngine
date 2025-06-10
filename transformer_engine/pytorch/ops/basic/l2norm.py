# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusable operation for L2 Normalization."""

from __future__ import annotations
from typing import Optional

import torch

from ...fp8 import FP8GlobalStateManager
from ...tensor import QuantizedTensor
from ...utils import (
    canonicalize_device,
    canonicalize_dtype,
    clear_tensor_data,
)
from ..op import BasicOperation, OperationContext
from .._common import maybe_autocast_dtype, reshape
from ...jit import l2norm_fused, l2norm_fwd_fused, l2norm_backward_fused


class L2Norm(BasicOperation):
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
    device: torch.device, default = default CUDA device
        Tensor device
    dtype: torch.dtype, default = default dtype
        Tensor datatype

    """

    def __init__(
        self,
        *,
        eps: float = 1e-6,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.eps: float = eps
        self.device = canonicalize_device(device)
        self.dtype = canonicalize_dtype(dtype)

    def reset_parameters(self) -> None:
        """L2Norm has no parameters to reset"""
        pass

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        prev_op: Optional[BasicOperation] = None,
        next_op: Optional[BasicOperation] = None,
    ) -> torch.Tensor:

        # Determine compute device and dtype
        device = self.device
        if device.type != "cuda":
            device = canonicalize_device(None)
        dtype = maybe_autocast_dtype(default_dtype=self.dtype or input_.dtype)

        # Reshape input for computation
        input_dims = tuple(input_.size())
        inner_dim = input_dims[-1]
        x = reshape(input_, (-1, inner_dim), device=device, dtype=dtype)

        if isinstance(x, QuantizedTensor):
            x = x.dequantize()

        # Check if backward pass is needed
        requires_grad = ctx.requires_grad

        # Check if output is quantized
        output_quantizer = None
        if (
            FP8GlobalStateManager.is_fp8_enabled()
            and next_op is not None
            and next_op.num_quantizers("forward") > 0
        ):
            output_quantizer = next_op.get_quantizer("forward", 0)

        # Compute L2 normalization using fused implementation
        # L2 norm: x / sqrt(sum(x^2) + eps) = x * rsqrt(sum(x^2) + eps)
        if requires_grad:
            # Training: use version that returns both output and intermediate values
            y, rsqrt_norm = l2norm_fwd_fused(x, self.eps)
        else:
            # Inference: use lightweight version that only returns output
            y = l2norm_fused(x, self.eps)
            rsqrt_norm = None  # Not needed for inference

        # Apply quantization if needed
        if output_quantizer is not None:
            y = output_quantizer(y)

        # Save state for backward pass
        if requires_grad:
            ctx.save_for_backward(x, rsqrt_norm)
            ctx.device = device
            ctx.dtype = dtype
            ctx.has_prev_op = prev_op is not None

        # Reshape output tensor
        out = reshape(y, input_dims)
        return out

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:

        # Saved tensors from forward pass
        x, rsqrt_norm = ctx.saved_tensors

        # Check input tensors
        device = ctx.device
        dtype = ctx.dtype
        dy = reshape(grad_output, x.size(), device=device, dtype=dtype)

        if isinstance(dy, QuantizedTensor):
            dy = dy.dequantize()

        # Compute L2 norm backward pass using fused implementation
        dx = l2norm_backward_fused(dy, x, rsqrt_norm, self.eps)

        # Clear saved tensors if possible
        if ctx.has_prev_op:
            clear_tensor_data(x)
        clear_tensor_data(rsqrt_norm)

        # Reshape results (no parameters, so empty tuple for param grads)
        grad_input = reshape(dx, grad_output.size())
        return grad_input, ()
