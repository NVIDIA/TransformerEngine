# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

from collections.abc import Iterable

import torch

from ._common import convert_tensor
from .op import FusableOperation


class Reshape(FusableOperation):

    def __init__(self, shape: Iterable[int]) -> None:
        super().__init__()
        self._shape = tuple(shape)

    def _unfused_op_forward(
        self,
        ctx: OperationContext,
        input: torch.Tensor,
    ) -> torch.Tensor:
        ctx.input_shape = input.size()
        x = convert_tensor(input, memory_format=torch.contiguous_format)
        return x.view(self._shape)

    def _unfused_op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:
        dy = convert_tensor(grad_output, memory_format=torch.contiguous_format)
        return dy.view(ctx.input_shape), ()
