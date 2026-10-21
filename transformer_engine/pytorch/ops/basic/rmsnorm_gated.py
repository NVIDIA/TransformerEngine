# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusable operation for RMSNorm with SiLU gating."""

from __future__ import annotations
from collections.abc import Iterable, Sequence
import math
import os
from typing import Any, Optional

import torch

from transformer_engine_torch import rmsnorm_bwd, rmsnorm_fwd
from ...constants import TE_DType
from ...cpu_offload import is_cpu_offload_enabled, mark_activation_offload
from ...tensor import Quantizer
from ...utils import (
    canonicalize_device,
    canonicalize_dtype,
    clear_tensor_data,
    devices_match,
)
from ..op import BasicOperation, OperationContext
from .._common import maybe_autocast_dtype, maybe_dequantize


class RMSNormGated(BasicOperation):
    r"""Root Mean Square Layer Normalization with SiLU gate

    Applies Root Mean Square Layer Normalization over a mini-batch of
    inputs and multiplies the result by the SiLU activation of a gate
    tensor:

    .. math::
        y = \frac{x}{\sqrt{\mathrm{Var}[x] + \varepsilon}} * \gamma * \mathrm{SiLU}(g)

    :math:`\gamma` is a learnable affine transform parameter that
    matches the inner-most dimensions of the input tensor. The gate
    tensor :math:`g` is an extra operation input and must have the
    same shape as the input tensor.

    Output quantization is not supported yet. The output is always
    returned in FP32, FP16, or BF16.

    Parameters
    ----------
    normalized_shape : int or iterable of int
        Inner dimensions of input tensor
    eps : float, default = 1e-5
        A value added to the denominator for numerical stability
    device : torch.device, default = default CUDA device
        Tensor device
    dtype : torch.dtype, default = default dtype
        Tensor datatype
    zero_centered_gamma : bool, default = False
        If ``True``, the :math:`\gamma` parameter is initialized to zero
        and the calculation changes to

            .. math::
                y = \frac{x}{\sqrt{\mathrm{Var}[x] + \varepsilon}} * (1 + \gamma) * \mathrm{SiLU}(g)

    sm_margin : int, default = 0
        Number of SMs to exclude when launching CUDA kernels. This
        helps overlap with other kernels, e.g. communication kernels.
        For more fine-grained control, provide a dict with the SM
        margin at each compute stage ("forward", "backward",
        "inference").

    """

    num_extra_inputs: int = 1

    def __init__(
        self,
        normalized_shape: Iterable[int] | int,
        *,
        eps: float = 1e-5,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
        zero_centered_gamma: bool = False,
        sm_margin: int = 0,
    ) -> None:
        super().__init__()
        self.eps: float = eps
        self.zero_centered_gamma: bool = zero_centered_gamma

        # Parameter shape
        if not isinstance(normalized_shape, Iterable):
            normalized_shape = (normalized_shape,)
        else:
            normalized_shape = tuple(normalized_shape)

        # Parameter device
        defer_param_init = False
        device = canonicalize_device(device)
        if device.type == "meta":
            defer_param_init = True

        # Initialize parameters if needed
        weight = torch.empty(
            normalized_shape,
            device=device,
            dtype=canonicalize_dtype(dtype),
        )
        weight = torch.nn.Parameter(weight)
        self.weight: torch.nn.Parameter
        self.register_parameter("weight", weight)
        if not defer_param_init:
            self.reset_parameters()

        # Number of SMs to exclude when launching CUDA kernels
        self._sm_margins: dict[str, int]
        if isinstance(sm_margin, dict):

            def getenv(name: str) -> int:
                return int(os.getenv(name, "0"))

            self._sm_margins = {
                "forward": sm_margin.get("forward", getenv("NVTE_FWD_LAYERNORM_SM_MARGIN")),
                "backward": sm_margin.get("backward", getenv("NVTE_BWD_LAYERNORM_SM_MARGIN")),
                "inference": sm_margin.get("inference", getenv("NVTE_INF_LAYERNORM_SM_MARGIN")),
            }
        else:

            def getenv(name: str) -> int:
                return int(os.getenv(name, str(sm_margin)))

            self._sm_margins = {
                "forward": getenv("NVTE_FWD_LAYERNORM_SM_MARGIN"),
                "backward": getenv("NVTE_BWD_LAYERNORM_SM_MARGIN"),
                "inference": getenv("NVTE_INF_LAYERNORM_SM_MARGIN"),
            }

    def reset_parameters(self) -> None:
        """Initialize parameter buffers and values"""

        # Parameter device
        weight = self.weight
        device = weight.device
        if device.type == "meta":
            device = canonicalize_device(None)

        # Initialize param buffers
        if not devices_match(weight.device, device):
            weight = torch.empty_like(weight, device=device)

        # Initialize values
        if self.zero_centered_gamma:
            torch.nn.init.zeros_(weight)
        else:
            torch.nn.init.ones_(weight)

        # Save updated parameter
        if not isinstance(weight, torch.nn.Parameter):
            weight = torch.nn.Parameter(weight)
        self.weight = weight

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
        del prev_op_grad_output_quantizer, basic_op_kwargs

        # Quantized output is not implemented in this fused operation
        if next_op_input_quantizer is not None:
            raise NotImplementedError(
                "RMSNormGated does not support quantized output "
                "(only FP32, FP16, or BF16). "
                f"Got {type(next_op_input_quantizer).__name__} for the next operation's input."
            )

        weight = self.weight
        weight_dims = tuple(weight.size())
        input_dims = tuple(input_.size())
        if len(input_dims) < len(weight_dims) or input_dims[-len(weight_dims) :] != weight_dims:
            raise ValueError(
                f"Input tensor (shape={input_dims}) "
                f"and weight tensor (shape={weight_dims}) are not compatible"
            )
        gate = basic_op_extra_inputs[0][0]
        gate_dims = tuple(gate.size())
        if gate_dims != input_dims:
            raise ValueError(
                f"Input tensor (shape={input_dims}) "
                f"and gate tensor (shape={gate_dims}) are not compatible"
            )

        inner_dim = math.prod(weight_dims)
        dtype = maybe_autocast_dtype(default_dtype=weight.dtype)
        x = maybe_dequantize(input_.contiguous(), dtype).view((-1, inner_dim))
        w = maybe_dequantize(self.weight, dtype).view((inner_dim,))
        g = maybe_dequantize(gate.contiguous()).view((-1, inner_dim))

        # Compute RMSNorm without quantized output
        n, _, rstdevs = rmsnorm_fwd(
            x,
            w,
            self.eps,
            None,
            None,
            TE_DType[dtype],
            self._sm_margins["forward" if basic_op_ctxs[0].requires_grad else "inference"],
            self.zero_centered_gamma,
        )

        # The SiLU gate is computed in FP32, like Megatron's fused implementation
        y = (n.float() * torch.nn.functional.silu(g.float())).to(dtype)

        # Save state for backward pass
        ctx = basic_op_ctxs[0]
        if ctx.requires_grad:
            if is_cpu_offload_enabled():
                mark_activation_offload(x, n, g, rstdevs)
            ctx.save_for_backward(x, n, g, rstdevs)
            ctx.dtype = dtype

        out = y.view(input_dims)
        return out, [()]

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
        del basic_op_grad_extra_outputs

        # Saved tensors from forward pass
        ctx = basic_op_ctxs[0]
        x, n, g, rstdevs = ctx.saved_tensors

        weight_dims = self.weight.size()
        inner_dim = math.prod(weight_dims)

        dtype = ctx.dtype
        dy = maybe_dequantize(grad_output.contiguous(), dtype).view(x.size())
        w = maybe_dequantize(self.weight, dtype).view((inner_dim,))

        # Gradients of the SiLU gate, computed in FP32. Round dn to the input
        # dtype so that the RMSNorm backward pass matches Megatron's implementation.
        g_fp32 = g.float()
        sigmoid = torch.sigmoid(g_fp32)
        dy_fp32 = dy.float()
        dn = (dy_fp32 * torch.nn.functional.silu(g_fp32)).to(dtype)
        silu_grad = sigmoid * (1 + g_fp32 * (1 - sigmoid))
        dg = (dy_fp32 * n.float() * silu_grad).to(g.dtype)

        dx, dw = rmsnorm_bwd(
            dn,
            x,
            rstdevs,
            w,
            self._sm_margins["backward"],
            self.zero_centered_gamma,
        )

        # Clear saved tensors if possible
        clear_tensor_data(x)
        clear_tensor_data(n)
        clear_tensor_data(g)
        clear_tensor_data(rstdevs)

        grad_input = dx.view(grad_output.size())
        grad_gate = dg.view(grad_output.size())
        grad_weight = dw.view(weight_dims)
        return grad_input, [(grad_weight,)], [(grad_gate,)]
