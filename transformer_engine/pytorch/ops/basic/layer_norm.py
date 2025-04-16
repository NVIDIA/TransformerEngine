# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusable operation for Layer Normalization."""

from __future__ import annotations
from collections.abc import Iterable
import math
import os
from typing import Optional

import torch

from transformer_engine_torch import layernorm_bwd, layernorm_fwd
from ...fp8 import FP8GlobalStateManager
from ...constants import TE_DType
from ...tensor import QuantizedTensor
from ...utils import (
    canonicalize_device,
    canonicalize_dtype,
    clear_tensor_data,
    devices_match,
)
from ..op import BasicOperation, OperationContext
from .._common import maybe_autocast_dtype, reshape


class LayerNorm(BasicOperation):
    r"""Layer Normalization

    Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \varepsilon}} * \gamma + \beta

    :math:`\gamma` and :math:`\beta` are learnable affine transform
    parameters that match the inner-most dimensions of the input
    tensor.

    Parameters
    ----------
    normalized_shape: int or iterable of int
        Inner dimensions of input tensor
    eps : float, default = 1e-5
        A value added to the denominator of layer normalization for
        numerical stability
    device: torch.device, default = default CUDA device
        Tensor device
    dtype: torch.dtype, default = default dtype
        Tensor datatype
    zero_centered_gamma : bool, default = 'False'
        If `True`, the :math:`\gamma` parameter is initialized to zero
        and the calculation changes to

            .. math::
                y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \varepsilon}} * (1 + \gamma) + \beta

    sm_margin: int or dict, default = 0
        Number of SMs to exclude when launching CUDA kernels. This
        helps overlap with other kernels, e.g. communication kernels.
        For more fine-grained control, provide a dict with the SM
        margin at each compute stage ("forward", "backward",
        "inference").

    """

    def __init__(
        self,
        normalized_shape: Iterable[int] | int,
        *,
        eps: float = 1e-5,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
        zero_centered_gamma: bool = False,
        sm_margin: int | dict[str, int] = 0,
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
        dtype = canonicalize_dtype(dtype)
        weight = torch.empty(
            normalized_shape,
            device=device,
            dtype=dtype,
        )
        bias = torch.empty(
            normalized_shape,
            device=device,
            dtype=dtype,
        )
        weight = torch.nn.Parameter(weight)
        bias = torch.nn.Parameter(bias)
        self.weight: torch.nn.Parameter
        self.bias: torch.nn.Parameter
        self.register_parameter("weight", weight)
        self.register_parameter("bias", bias)
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
        bias = self.bias
        device = weight.device
        if device.type == "meta":
            device = canonicalize_device(None)

        # Initialize param buffers
        if not devices_match(weight.device, device):
            weight = torch.empty_like(weight, device=device)
        if not devices_match(bias.device, device):
            bias = torch.empty_like(bias, device=device)

        # Initialize values
        if self.zero_centered_gamma:
            torch.nn.init.zeros_(weight)
        else:
            torch.nn.init.ones_(weight)
        torch.nn.init.zeros_(bias)

        # Save updated parameter
        if not isinstance(weight, torch.nn.Parameter):
            weight = torch.nn.Parameter(weight)
        if not isinstance(bias, torch.nn.Parameter):
            bias = torch.nn.Parameter(bias)
        self.weight = weight
        self.bias = bias

    def pre_forward(self, *args, **kwargs) -> None:
        super().pre_forward(*args, **kwargs)
        if self.weight.device.type == "meta" or self.bias.device.type == "meta":
            self.reset_parameters()

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        prev_op: Optional[BasicOperation] = None,
        next_op: Optional[BasicOperation] = None,
    ) -> torch.Tensor:

        # Check tensor dims
        weight = self.weight
        weight_dims = tuple(weight.size())
        input_dims = tuple(input_.size())
        if len(input_dims) < len(weight_dims) or input_dims[-len(weight_dims) :] != weight_dims:
            raise ValueError(
                f"Input tensor (shape={input_dims}) "
                f"and weight tensor (shape={weight_dims}) are not compatible"
            )

        # Check input tensors
        inner_dim = math.prod(weight_dims)
        device = weight.device
        if device.type != "cuda":
            device = canonicalize_device(None)
        dtype = maybe_autocast_dtype(default_dtype=weight.dtype)
        x = reshape(input_, (-1, inner_dim), device=device, dtype=dtype)
        w = reshape(self.weight, (inner_dim,), device=device, dtype=dtype)
        b = reshape(self.bias, (inner_dim,), device=device, dtype=dtype)
        if isinstance(x, QuantizedTensor):
            x = x.dequantize()
        if isinstance(w, QuantizedTensor):
            w = w.dequantize()
        if isinstance(b, QuantizedTensor):
            b = b.dequantize()

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

        # Compute layer norm
        sm_margin = self._sm_margins["forward" if requires_grad else "inference"]
        y, means, rstdevs = layernorm_fwd(
            x,
            w,
            b,
            self.eps,
            None,
            output_quantizer,
            TE_DType[dtype],
            sm_margin,
            self.zero_centered_gamma,
        )

        # Save state for backward pass
        if requires_grad:
            ctx.save_for_backward(x, means, rstdevs)
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
        x, means, rstdevs = ctx.saved_tensors

        # Tensor dims
        weight_dims = self.weight.size()
        inner_dim = math.prod(weight_dims)

        # Check input tensors
        device = ctx.device
        dtype = ctx.dtype
        dy = reshape(grad_output, x.size(), device=device, dtype=dtype)
        w = reshape(self.weight, (inner_dim,), device=device, dtype=dtype)
        if isinstance(w, QuantizedTensor):
            w = w.dequantize()
        if isinstance(dy, QuantizedTensor):
            dy = dy.dequantize()

        # Compute layer norm backward pass
        dx, dw, db = layernorm_bwd(
            dy,
            x,
            means,
            rstdevs,
            w,
            self._sm_margins["backward"],
            self.zero_centered_gamma,
        )

        # Clear saved tensors if possible
        if ctx.has_prev_op:
            clear_tensor_data(x)
        clear_tensor_data(means)
        clear_tensor_data(rstdevs)

        # Reshape results
        grad_input = reshape(dx, grad_output.size())
        grad_weight = reshape(dw, weight_dims)
        grad_bias = reshape(db, weight_dims)
        return grad_input, (grad_weight, grad_bias)
