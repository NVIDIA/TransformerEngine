# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for all-gather."""

from __future__ import annotations
from typing import Optional

import torch

from transformer_engine.pytorch.float8_tensor import Float8Tensor
from transformer_engine.pytorch.ops.op import (
    BasicOperation,
    OperationContext,
)
from .._common import convert_tensor, is_float8_tensor


class AllGather(BasicOperation):
    """All-gather tensor along outer dimension

    Equivalent to gathering tensors from all processes and
    concatenating along the first dimension.

    Parameters
    ----------
    process_group: torch.distributed.ProcessGroup, default = world group
        Process group for communication

    """

    def __init__(
        self,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        super().__init__()
        self.process_group: Optional[torch.distributed.ProcessGroup] = process_group
        self.process_group_size: int = torch.distributed.get_world_size(process_group)

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        prev_op: Optional[BasicOperation] = None,
        next_op: Optional[BasicOperation] = None,
    ) -> torch.Tensor:

        # Trivial case
        if self.process_group_size == 1:
            return input_

        # Tensor dimensions
        input_dims = input_.size()
        if not input_dims:
            raise RuntimeError(
                "Attempted to all-gather a tensor "
                f"with shape={list(input_dims)} "
                f"over {self.process_group_size} processes"
            )
        output_dims = list(input_dims)
        output_dims[0] *= self.process_group_size

        # Perform all-gather
        x = convert_tensor(input_, memory_format=torch.contiguous_format)
        y = None
        if is_float8_tensor(x):
            y = Float8Tensor.make_like(
                x,
                data=torch.empty(
                    output_dims,
                    dtype=torch.uint8,
                    device=x.device,
                ),
            )
            torch.distributed.all_gather_into_tensor(
                y._data,
                x._data,
                group=self.process_group,
            )
        else:
            y = torch.empty(output_dims, dtype=x.dtype, device=x.device)
            torch.distributed.all_gather_into_tensor(
                y,
                x,
                group=self.process_group,
            )
        return y

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:

        # Trivial case
        if self.process_group_size == 1:
            return grad_output, ()

        # Tensor dimensions
        output_dims = grad_output.size()
        if not output_dims or output_dims[0] % self.process_group_size != 0:
            raise RuntimeError(
                "Attempted to reduce-scatter a tensor "
                f"with shape={list(output_dims)} "
                f"over {self.process_group_size} processes"
            )
        input_dims = list(output_dims)
        input_dims[0] //= self.process_group_size

        # Check output gradient tensor
        dy = grad_output
        if is_float8_tensor(dy):
            dy = dy.from_float8()
        dy = dy.contiguous()

        # Perform reduce-scatter
        dx = torch.empty(input_dims, dtype=dy.dtype, device=dy.device)
        torch.distributed.reduce_scatter_tensor(
            dx,
            dy,
            group=self.process_group,
        )
        return dx, ()
