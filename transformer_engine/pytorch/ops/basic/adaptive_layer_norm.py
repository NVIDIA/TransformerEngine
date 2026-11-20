# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for timestep-conditioned layer normalization."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import math
from typing import Any, Optional

import torch

from ...cpu_offload import is_cpu_offload_enabled, mark_activation_offload
from ...tensor import Quantizer
from ...triton.adaptive_layer_norm import adaptive_layernorm_bwd, adaptive_layernorm_fwd
from .._common import maybe_dequantize
from ..op import BasicOperation, OperationContext


class AdaptiveLayerNorm(BasicOperation):
    r"""Layer normalization with a per-sample scale and shift.

    Normalizes the last dimension, then applies a timestep-conditioned affine
    transform, as used in diffusion transformers:

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \varepsilon}}
            (1 + \mathrm{scale}) + \mathrm{shift}

    This operation has no parameters. Scale and shift are extra tensor inputs,
    and gradients propagate to both. Call it as ``op(x, scale, shift)``, or
    pass the same extra inputs to a containing ``ops.Sequential``.

    The input has at least two dimensions. Conditions have shape
    ``(batch_size, hidden_size)``, or the same rank as the input with all
    dimensions except the batch and hidden dimensions equal to one. For
    example, batch-first input ``[B, S, H]`` accepts ``[B, H]`` or
    ``[B, 1, H]``; sequence-first input ``[S, B, H]`` with
    ``batch_dim=1`` accepts ``[B, H]`` or ``[1, B, H]``.

    CUDA float32, float16, and bfloat16 tensors are supported. Conditions can
    have a different dtype from the input. Normalization, the addition
    ``1 + scale``, and modulation are computed in float32 before casting
    the result to the input dtype. In particular, small bfloat16 scales are
    not rounded away by adding one in bfloat16. Gradients have the dtype of
    their respective inputs.

    Parameters
    ----------
    hidden_size : int
        Size of the last input dimension, between 1 and 16384.
    eps : float, default = 1e-5
        Non-negative value added to the variance for numerical stability.
    batch_dim : int, default = 0
        Non-negative index of the batch dimension. All remaining dimensions
        except the last dimension share the per-sample conditions.

    Notes
    -----
    This operation produces an unquantized output. A following operation can
    quantize it according to its quantization recipe. It does not fuse
    normalization and GEMM into a single kernel.
    """

    num_extra_inputs: int = 2

    def __init__(
        self,
        hidden_size: int,
        *,
        eps: float = 1e-5,
        batch_dim: int = 0,
    ) -> None:
        super().__init__()
        if not isinstance(hidden_size, int) or not 1 <= hidden_size <= 16384:
            raise ValueError("hidden_size must be an integer between 1 and 16384.")
        if not math.isfinite(eps) or eps < 0:
            raise ValueError("eps must be finite and non-negative.")
        if not isinstance(batch_dim, int) or batch_dim < 0:
            raise ValueError("batch_dim must be a non-negative integer.")
        self.hidden_size = hidden_size
        self.eps = eps
        self.batch_dim = batch_dim

    def op_forward(self, *args, **kwargs) -> None:
        raise RuntimeError("AdaptiveLayerNorm uses fuser_forward for its two extra inputs.")

    def op_backward(self, *args, **kwargs) -> None:
        raise RuntimeError("AdaptiveLayerNorm uses fuser_backward for its two extra inputs.")

    def fuser_forward(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        *,
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
        basic_op_kwargs: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, Sequence[Sequence[torch.Tensor]]]:
        ctx = basic_op_ctxs[0]
        scale, shift = basic_op_extra_inputs[0]
        if ctx.requires_grad:
            ctx.compute_dscale = scale.requires_grad
            ctx.compute_dshift = shift.requires_grad
        if input_.ndim < 2 or input_.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Input must have at least two dimensions and last dimension {self.hidden_size}, "
                f"got {tuple(input_.shape)}."
            )
        x = maybe_dequantize(input_).contiguous()
        scale = maybe_dequantize(scale).contiguous()
        shift = maybe_dequantize(shift).contiguous()
        output, mean, rstd = adaptive_layernorm_fwd(
            x, scale, shift, self.eps, batch_dim=self.batch_dim
        )
        if ctx.requires_grad:
            if is_cpu_offload_enabled():
                mark_activation_offload(x, scale, mean, rstd)
            ctx.save_for_backward(x, scale, mean, rstd)
            ctx.shift_shape = tuple(shift.shape)
            ctx.shift_dtype = shift.dtype
        return output, [()]

    def fuser_backward(
        self,
        basic_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
        *,
        basic_op_grad_extra_outputs: list[tuple[torch.Tensor, ...]],
    ) -> tuple[
        torch.Tensor,
        Iterable[Iterable[Optional[torch.Tensor]]],
        Iterable[Iterable[Optional[torch.Tensor]]],
    ]:
        ctx = basic_op_ctxs[0]
        x, scale, mean, rstd = ctx.saved_tensors
        # Intermediate tensors inside the fuser need not have requires_grad set,
        # even when their producers require gradients.
        dx, dscale, dshift = adaptive_layernorm_bwd(
            maybe_dequantize(grad_output).contiguous(),
            x,
            scale,
            mean,
            rstd,
            shift_shape=ctx.shift_shape,
            shift_dtype=ctx.shift_dtype,
            compute_dscale=ctx.compute_dscale,
            compute_dshift=ctx.compute_dshift,
            batch_dim=self.batch_dim,
        )
        return dx, [()], [(dscale, dshift)]
