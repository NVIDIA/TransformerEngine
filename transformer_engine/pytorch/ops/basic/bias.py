# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for bias."""

from __future__ import annotations
from typing import Optional

import torch

from transformer_engine.pytorch.ops.op import (
    BasicOperation,
    OperationContext,
)
from .._common import (
    canonicalize_device,
    canonicalize_dtype,
)


class Bias(BasicOperation):
    """Apply additive bias

    This is equivalent to the additive bias in `torch.nn.Linear`.

    Parameters
    ----------
    size: int
        Inner dimension of input tensor
    device: torch.device, default = default CUDA device
        Tensor device
    dtype: torch.dtype, default = default dtype
        Tensor datatype
    tensor_parallel: bool, default = `False`
        Whether to distribute input tensor and bias tensors along
        inner dimension
    tensor_parallel_group: torch.distributed.ProcessGroup, default = world group
        Process group for tensor parallelism

    """

    def __init__(
        self,
        size: int,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
        tensor_parallel: bool = False,
        tensor_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        super().__init__()

        # Bias size
        self._size = size

        # Bias tensor device
        defer_param_init = False
        device = canonicalize_device(device)
        if device.type == "meta":
            defer_param_init = True
            device = canonicalize_device(None)
        self.device: torch.device = device

        # Tensor parallel configuration
        tensor_parallel_size = 1
        local_size = size
        if tensor_parallel:
            tensor_parallel_size = torch.distributed.get_world_size(tensor_parallel_group)
            tensor_parallel = tensor_parallel_size > 1
            if size % tensor_parallel_size != 0:
                raise ValueError(
                    "Invalid configuration for tensor parallelism "
                    f"({size=}, {tensor_parallel_size=})"
                )
            local_size //= tensor_parallel_size
        else:
            tensor_parallel_group = None
        self.tensor_parallel: bool = tensor_parallel
        self.tensor_parallel_group: Optional[torch.distributed.ProcessGroup] = tensor_parallel_group
        self.tensor_parallel_size: int = tensor_parallel_size
        self.local_size: int = local_size

        # Initialize parameters if needed
        bias = torch.empty(
            local_size,
            device="meta",
            dtype=canonicalize_dtype(dtype),
        )
        bias = torch.nn.Parameter(bias)
        self.bias: torch.nn.Parameter
        self.register_parameter("bias", bias)
        if not defer_param_init:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameter buffers and values"""

        # Make sure parameter is initialized
        bias = self.bias
        if bias.device.type != "cuda":
            bias = torch.empty_like(bias, device=self.device)
        else:
            bias = bias.to(device=self.device)

        # Initialize values
        bias.zero_()

        # Save updated parameter
        if not isinstance(bias, torch.nn.Parameter):
            bias = torch.nn.Parameter(bias)
        self.bias = bias

    def pre_forward(self, *args, **kwargs) -> None:
        super().pre_forward(*args, **kwargs)
        if self.bias.device.type == "meta":
            self.reset_parameters()

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        prev_op: Optional[BasicOperation] = None,
        next_op: Optional[BasicOperation] = None,
    ) -> torch.Tensor:
        x = input_
        b = self.bias.reshape([1] * (x.dim() - 1) + [self.local_size])
        return x + b

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:
        dy = grad_output
        if dy.dim() > 1:
            db = dy.sum(tuple(range(dy.dim() - 1)))
        else:
            db = dy
        return dy, (db,)
