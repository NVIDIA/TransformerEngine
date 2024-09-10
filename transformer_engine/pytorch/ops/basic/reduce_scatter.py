# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for reduce-scatter."""

from __future__ import annotations
from typing import Optional

import torch

from ...tensor import Float8Tensor, QuantizedTensor
from ..op import BasicOperation, OperationContext
from .._common import convert_tensor


class ReduceScatter(BasicOperation):
    """Reduce-scatter tensor along outer dimension

    Equivalent to summing tensors from all processes and splitting
    along the first dimension.

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
        if not input_dims or input_dims[0] % self.process_group_size != 0:
            raise RuntimeError(
                "Attempted to reduce-scatter a tensor "
                f"with shape={list(input_dims)} "
                f"over {self.process_group_size} processes"
            )
        output_dims = list(input_dims)
        output_dims[0] //= self.process_group_size

        # Check input tensor
        x = input_
        if isinstance(x, QuantizedTensor):
            x = x.dequantize()
        x = x.contiguous()

        # Perform reduce-scatter
        y = torch.empty(output_dims, dtype=x.dtype, device=x.device)
        torch.distributed.reduce_scatter_tensor(y, x, group=self.process_group)
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
        if not output_dims:
            raise RuntimeError(
                "Attempted to all-gather a tensor "
                f"with shape={list(output_dims)} "
                f"over {self.process_group_size} processes"
            )
        input_dims = list(output_dims)
        input_dims[0] *= self.process_group_size

        # Perform all-gather
        dy = convert_tensor(grad_output, memory_format=torch.contiguous_format)
        dx = None
        if isinstance(dy, Float8Tensor):
            dx = Float8Tensor.make_like(
                dy,
                data=torch.empty(
                    input_dims,
                    dtype=torch.uint8,
                    device=dy.device,
                ),
            )
            torch.distributed.all_gather_into_tensor(
                dx._data,
                dy._data,
                group=self.process_group,
            )
        else:
            if isinstance(dy, QuantizedTensor):
                dy = dy.dequantize()
            dx = torch.empty(input_dims, dtype=dy.dtype, device=dy.device)
            torch.distributed.all_gather_into_tensor(
                dx,
                dy,
                group=self.process_group,
            )

        return dx, ()
