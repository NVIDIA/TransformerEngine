# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusable operation for RMSNorm."""

from __future__ import annotations
from collections.abc import Iterable
import math
import os
from typing import Optional

import torch

from transformer_engine.pytorch.cpp_extensions import (
    rmsnorm_bwd,
    rmsnorm_fwd,
    rmsnorm_fwd_fp8,
    rmsnorm_fwd_fp8_inf,
    rmsnorm_fwd_inf,
)
from transformer_engine.pytorch.float8_tensor import Float8Tensor
from transformer_engine.pytorch.fp8 import (
    FP8GlobalStateManager,
    get_fp8_te_dtype,
)
from transformer_engine.pytorch.ops.op import (
    BasicOperation,
    OperationContext,
)
from .._common import (
    canonicalize_device,
    canonicalize_dtype,
    is_float8_tensor,
    reshape,
)
from ...utils import clear_tensor_data


class RMSNorm(BasicOperation):
    r"""Root Mean Square Layer Normalization

    Applies Root Mean Square Layer Normalization over a mini-batch of
    inputs as described in the paper
    `Root Mean Square Layer Normalization <https://arxiv.org/abs/1910.07467>`__

    .. math::
        y = \frac{x}{\sqrt{\mathrm{Var}[x] + \varepsilon}} * \gamma

    :math:`\gamma` is a learnable affine transform parameter that
    matches the inner-most dimensions of the input tensor.

    Parameters
    ----------
    normalized_shape: int or iterable of int
        Inner dimensions of input tensor
    eps : float, default = 1e-5
        A value added to the denominator for numerical stability
    device: torch.device, default = default CUDA device
        Tensor device
    dtype: torch.dtype, default = default dtype
        Tensor datatype
    zero_centered_gamma : bool, default = 'False'
        If `True`, the :math:`\gamma` parameter is initialized to zero
        and the calculation changes to

            .. math::
                y = \frac{x}{\sqrt{\mathrm{Var}[x] + \varepsilon}} * (1 + \gamma)

    sm_margin: int, default = 0
        Number of SMs to exclude when launching CUDA kernels. This
        helps overlap with other kernels, e.g. communication kernels.

    """

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
        self._eps: float = eps
        self._zero_centered_gamma: bool = zero_centered_gamma

        # Parameter shape
        if not isinstance(normalized_shape, Iterable):
            normalized_shape = (normalized_shape,)
        else:
            normalized_shape = tuple(normalized_shape)
        self._shape: tuple[int, ...] = normalized_shape

        # Parameter device
        defer_param_init = False
        device = canonicalize_device(device)
        if device.type == "meta":
            defer_param_init = True
            device = canonicalize_device(None)
        if device.type != "cuda":
            raise ValueError(f"Only CUDA devices are supported (got {device})")
        self.device: torch.device = device

        # Parameter datatype
        self.dtype: torch.dtype = canonicalize_dtype(dtype)

        # Initialize parameters if needed
        weight = torch.empty(
            self._shape,
            device="meta",
            dtype=dtype,
        )
        weight = torch.nn.Parameter(weight)
        self.register_parameter("weight", weight)
        if not defer_param_init:
            self.reset_parameters()

        # Number of SMs to exclude when launching CUDA kernels
        self._sm_margins: dict[str, int] = dict(
            fwd=int(os.getenv("NVTE_FWD_LAYERNORM_SM_MARGIN", str(sm_margin))),
            bwd=int(os.getenv("NVTE_BWD_LAYERNORM_SM_MARGIN", str(sm_margin))),
            inf=int(os.getenv("NVTE_INF_LAYERNORM_SM_MARGIN", str(sm_margin))),
        )

    def reset_parameters(self) -> None:
        """Initialize parameter buffers and values"""

        # Make sure parameter is initialized
        weight = self.weight
        if weight.device.type != "cuda":
            weight = torch.empty_like(weight, device=self.device)
        weight = weight.to(device=self.device, dtype=self.dtype)

        # Initialize values
        if self._zero_centered_gamma:
            weight.zero_()
        else:
            weight.fill_(1)

        # Save updated parameter
        if not isinstance(weight, torch.nn.Parameter):
            weight = torch.nn.Parameter(weight)
        self.weight = weight

    def pre_forward(self) -> None:
        super().pre_forward()
        if self.weight.device.type == "meta":
            self.reset_parameters()

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        prev_op: Optional[BasicOperation] = None,
        next_op: Optional[BasicOperation] = None,
    ) -> torch.Tensor:

        # Check tensor dims
        input_dims = tuple(input_.size())
        if len(input_dims) < len(self._shape) or input_dims[-len(self._shape) :] != self._shape:
            raise ValueError(
                f"Input tensor (shape={input_dims}) "
                f"and weight tensor (shape={self._shape}) are not compatible"
            )

        # Check input tensors
        inner_dim = math.prod(self._shape)
        device = self.device
        dtype = self.dtype
        x = reshape(input_, (-1, inner_dim), device=device, dtype=dtype)
        w = reshape(self.weight, (inner_dim,), device=device, dtype=dtype)
        if is_float8_tensor(x):
            x = x.from_float8()
        if is_float8_tensor(w):
            w = w.from_float8()

        # Check if backward pass is needed
        requires_grad = input_.requires_grad or self.weight.requires_grad

        # Check if FP8 is enabled
        with_fp8_output = (
            FP8GlobalStateManager.is_fp8_enabled()
            and next_op is not None
            and next_op.num_fp8_scales("input") > 0
        )
        output_fp8_meta = None
        if with_fp8_output:
            output_fp8_meta = next_op.get_fp8_meta("input")

        # Compute RMSNorm
        y = None
        rstdevs = None
        sm_margin = self._sm_margins["fwd" if requires_grad else "inf"]
        if with_fp8_output:
            fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(forward=True)
            fp8_dtype = get_fp8_te_dtype(output_fp8_meta["recipe"], fprop_tensor=True)
            args = (
                x,
                w,
                self._eps,
                output_fp8_meta[fp8_meta_key],
                0,  # fp8_meta_index
                fp8_dtype,
                sm_margin,
                self._zero_centered_gamma,
            )
            if requires_grad:
                data, rstdevs = rmsnorm_fwd_fp8(*args)
            else:
                data = rmsnorm_fwd_fp8_inf(*args)
            y = Float8Tensor(
                data=data,
                fp8_meta=output_fp8_meta,
                fp8_meta_forward=True,
                fp8_meta_index=0,
                fp8_dtype=fp8_dtype,
                dtype=dtype,
            )
        else:
            args = (
                x,
                w,
                self._eps,
                sm_margin,
                self._zero_centered_gamma,
            )
            if requires_grad:
                y, rstdevs = rmsnorm_fwd(*args)
            else:
                y = rmsnorm_fwd_inf(*args)

        # Save state for backward pass
        if not requires_grad:
            x = None
        ctx.save_for_backward(x, rstdevs)
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
        x, rstdevs = ctx.saved_tensors

        # Check input tensors
        inner_dim = x.size(-1)
        device = self.device
        dtype = self.dtype
        dy = reshape(grad_output, x.size(), device=device, dtype=dtype)
        w = reshape(self.weight, (inner_dim,), device=device, dtype=dtype)
        if is_float8_tensor(w):
            w = w.from_float8()
        if is_float8_tensor(dy):
            dy = dy.from_float8()

        # Compute RMSNorm backward pass
        dx, dw = rmsnorm_bwd(
            dy,
            x,
            rstdevs,
            w,
            self._sm_margins["bwd"],
            self._zero_centered_gamma,
        )

        # Clear saved tensors if possible
        if ctx.has_prev_op:
            clear_tensor_data(x)
        clear_tensor_data(rstdevs)

        # Reshape results
        grad_input = reshape(dx, grad_output.size())
        grad_weight = reshape(dw, self._shape)
        return grad_input, (grad_weight,)
