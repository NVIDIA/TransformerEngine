# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

from collections.abc import Iterable

import torch

from transformer_engine.pytorch.fuser.ops.op import UnfusedOperation
from .._common import convert_tensor


class Reshape(UnfusedOperation):
    """Reshape tensor

    See `torch.reshape`.

    Parameters
    ----------
    shape: iterable of int
        Output tensor dimensions. If one dimension is -1, it is
        inferred based on input tensor dimensions.

    """

    def __init__(self, shape: Iterable[int]) -> None:
        super().__init__()
        self._shape = tuple(shape)

    def op_forward(
        self,
        ctx: OperationContext,
        input: torch.Tensor,
    ) -> torch.Tensor:
        ctx.input_shape = input.size()
        x = convert_tensor(input, memory_format=torch.contiguous_format)
        return x.view(self._shape)

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:
        dy = convert_tensor(grad_output, memory_format=torch.contiguous_format)
        return dy.view(ctx.input_shape), ()
